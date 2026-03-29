from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from main import vector_db
import httpx
import asyncio
import os

router = APIRouter()
client = OpenAI()

RAWG_KEY = os.getenv("RAWG_API_KEY")


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    category: str | None = None
    messages: list[ChatMessage] = []


CATEGORY_PROMPTS = {
    "gameplay": "You are a gameplay expert. ONLY answer questions strictly about game mechanics, strategies, bosses, combat, weapons, missions, quests, and in-game progression. If the question is not about gameplay, respond with: 'This question is not related to gameplay. Please switch to the appropriate category.'",
    "technical": "You are a technical support specialist. ONLY answer questions strictly about crashes, bugs, errors, performance issues, drivers, installations, and system requirements. If the question is not a technical issue, respond with: 'This question is not a technical issue. Please switch to the appropriate category.'",
    "account": "You are an account support specialist. ONLY answer questions strictly about account creation, login, password reset, 2FA, account linking, saves, and account security. Use the game's platform info (Steam, GOG, PlayStation, etc.) to give accurate platform-specific account guidance. If the question is not account-related, respond with: 'This question is not account-related. Please switch to the appropriate category.'",
    "billing": "You are a billing support specialist. ONLY answer questions strictly about pricing, purchases, refunds, DLC, subscriptions, and payment issues. Use the game's platform info to give accurate platform-specific billing guidance. If the question is not billing-related, respond with: 'This question is not billing-related. Please switch to the appropriate category.'",
    "community": "You are a community manager. ONLY answer questions strictly about multiplayer, co-op, PvP, clans, community events, reporting players, and social features. If the question is not community-related, respond with: 'This question is not community-related. Please switch to the appropriate category.'",
    "updates": "You are a patch notes expert. Answer questions about game updates, patches, changelogs, new seasons, balance changes, and upcoming features. Use the official Steam news if provided. If no Steam news is available, use your own training knowledge to answer about recent updates for the game. Only reject if the question is clearly unrelated to updates at all.",
    "general": "You are a helpful gaming assistant. Provide general information about the game including its overview, genre, platforms, release date, and any general facts. Do not go deep into any specific category.",
}

REDDIT_CATEGORIES = {"gameplay", "technical", "community", "general"}
OFFICIAL_ONLY_CATEGORIES = {"account", "billing"}


async def extract_game_name(question: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=20,
        messages=[
            {"role": "system", "content": "Extract the full official game title from the question. Expand abbreviations and numbered shorthand to the real title (e.g. 'warhammer 3' -> 'Total War: Warhammer III', 'fifa 24' -> 'EA Sports FC 24', 'elden ring' -> 'Elden Ring'). Return just the name, nothing else."},
            {"role": "user", "content": question},
        ],
    )
    return res.choices[0].message.content.strip()


async def is_franchise(name: str) -> bool:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=5,
        messages=[
            {"role": "system", "content": "Answer only 'yes' or 'no'. Is this a broad franchise/series name with NO specific installment indicated (e.g. 'Call of Duty', 'FIFA', 'Mario', 'Warhammer')? Answer 'no' if the name refers to a specific installment, even if it is part of a series (e.g. 'Total War: Warhammer III', 'FIFA 24', 'Call of Duty: Black Ops 6', 'Elden Ring')."},
            {"role": "user", "content": name},
        ],
    )
    return res.choices[0].message.content.strip().lower().startswith("yes")


async def get_rawg_game_info(game_name: str) -> dict | list:
    franchise = await is_franchise(game_name)
    page_size = 5 if franchise else 1

    async with httpx.AsyncClient() as http:
        r = await http.get(
            "https://api.rawg.io/api/games",
            params={"key": RAWG_KEY, "search": game_name, "page_size": page_size},
            headers={"User-Agent": "NexusCore/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return {} if not franchise else []

        def base_parse(g):
            return {
                "name": g.get("name"),
                "slug": g.get("slug"),
                "released": g.get("released"),
                "rating": g.get("rating"),
                "metacritic": g.get("metacritic"),
                "genres": [x["name"] for x in g.get("genres", [])],
                "platforms": [x["platform"]["name"] for x in g.get("platforms", [])],
                "steam_id": None,
            }

        if franchise:
            return [base_parse(g) for g in results]

        # For single game, fetch detail to get stores (which includes Steam app ID)
        game = base_parse(results[0])
        slug = results[0].get("slug")
        if slug:
            try:
                detail = await http.get(
                    f"https://api.rawg.io/api/games/{slug}",
                    params={"key": RAWG_KEY},
                    headers={"User-Agent": "NexusCore/1.0"},
                    timeout=10,
                )
                detail.raise_for_status()
                for store_entry in detail.json().get("stores", []):
                    url = store_entry.get("url", "")
                    if "store.steampowered.com/app/" in url:
                        try:
                            game["steam_id"] = int(url.split("/app/")[1].split("/")[0])
                        except (IndexError, ValueError):
                            pass
                        break
            except Exception:
                pass
        return game


async def resolve_steam_app_id(game_name: str, rawg_slug: str | None = None) -> int | None:
    import re
    async with httpx.AsyncClient() as http:
        # Method 1: RAWG detail endpoint (exact Steam store URL)
        if rawg_slug:
            try:
                r = await http.get(
                    f"https://api.rawg.io/api/games/{rawg_slug}",
                    params={"key": RAWG_KEY},
                    headers={"User-Agent": "NexusCore/1.0"},
                    timeout=10,
                )
                r.raise_for_status()
                for store_entry in r.json().get("stores", []):
                    url = store_entry.get("url", "")
                    if "store.steampowered.com/app/" in url:
                        return int(url.split("/app/")[1].split("/")[0])
            except Exception:
                pass

        # Method 2: Steam suggest endpoint (returns HTML — parse data-ds-appid)
        if game_name:
            try:
                r = await http.get(
                    "https://store.steampowered.com/search/suggest",
                    params={"term": game_name, "f": "games", "cc": "US", "realm": "1", "l": "english"},
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                    timeout=10,
                )
                r.raise_for_status()
                match = re.search(r'data-ds-appid="(\d+)"', r.text)
                if match:
                    return int(match.group(1))
            except Exception:
                pass

    return None


# Keep for backward compatibility with franchise path
async def get_steam_id_from_rawg_slug(slug: str | None) -> int | None:
    return await resolve_steam_app_id(game_name="", rawg_slug=slug)


async def get_steam_app_id(game_name: str) -> int | None:
    return await resolve_steam_app_id(game_name=game_name)


def strip_html(text: str) -> str:
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{[^}]+\}', '', text)  # strip BBCode-style tags
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


async def get_steam_news(app_id: int, detailed: bool = False) -> list[dict]:
    count = 3 if detailed else 5
    async with httpx.AsyncClient() as http:
        r = await http.get(
            "https://api.steampowered.com/ISteamNews/GetNewsForApp/v2/",
            params={"appid": app_id, "count": count, "maxlength": 0, "format": "json"},
            headers={"User-Agent": "NexusCore/1.0"},
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("appnews", {}).get("newsitems", [])
        return [
            {
                "title": item.get("title"),
                "contents": strip_html(item.get("contents", "")),
                "url": item.get("url"),
            }
            for item in items
        ]


async def extract_search_keywords(question: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=15,
        messages=[
            {"role": "system", "content": "Extract 1-3 Reddit search keywords from the question. Do NOT include the game name. Always append 'game' to disambiguate from movies, TV shows, or other media. Examples: 'review' -> 'game review', 'best build' -> 'best build game', 'tips tricks' -> 'game tips tricks', 'gameplay guide' -> 'game gameplay guide', 'lore explained' -> 'game lore'. Return only the keywords, nothing else."},
            {"role": "user", "content": question},
        ],
    )
    return res.choices[0].message.content.strip()


async def get_game_subreddit(game_name: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=20,
        messages=[
            {"role": "system", "content": "Return only the most specific game-dedicated subreddit name (without r/) for the given video game. Always prefer a game-specific subreddit over a general one. Examples: 'The Witcher 3' -> 'witcher3', 'Elden Ring' -> 'Eldenring', 'Total War: Warhammer III' -> 'totalwar', 'Minecraft' -> 'Minecraft', 'GTA 5' -> 'gtaonline', 'Red Dead Redemption 2' -> 'reddeadredemption2'. Return only the subreddit name, nothing else."},
            {"role": "user", "content": game_name},
        ],
    )
    return res.choices[0].message.content.strip().lstrip("r/")


def _parse_reddit_children(children: list) -> list[dict]:
    posts = [
        {
            "id": p["data"]["id"],
            "title": p["data"]["title"],
            "subreddit": p["data"]["subreddit"],
            "score": p["data"]["score"],
            "url": f"https://reddit.com{p['data']['permalink']}",
            "selftext": p["data"].get("selftext", "")[:2000],
            "num_comments": p["data"]["num_comments"],
        }
        for p in children
        if p["data"].get("selftext", "").strip() not in ("", "[deleted]", "[removed]")
    ]
    if not posts:
        posts = [
            {
                "id": p["data"]["id"],
                "title": p["data"]["title"],
                "subreddit": p["data"]["subreddit"],
                "score": p["data"]["score"],
                "url": f"https://reddit.com{p['data']['permalink']}",
                "selftext": p["data"].get("selftext", "")[:2000],
                "num_comments": p["data"]["num_comments"],
            }
            for p in children
        ]
    return sorted(posts, key=lambda x: x["score"], reverse=True)


async def fetch_reddit_posts(query: str, limit: int = 25, subreddit: str | None = None) -> list[dict]:
    import re

    browser_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    headers = {
        "User-Agent": browser_ua,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(follow_redirects=True) as http:
        # 1. Search endpoint — most relevant results
        if subreddit:
            search_url = f"https://www.reddit.com/r/{subreddit}/search.json"
        else:
            search_url = "https://www.reddit.com/search.json"
        try:
            r = await http.get(
                search_url,
                params={"q": query, "sort": "top", "t": "all", "limit": limit, "restrict_sr": "1"},
                headers=headers,
                timeout=15,
            )
            if r.status_code == 200:
                children = r.json().get("data", {}).get("children", [])
                if children:
                    return _parse_reddit_children(children)
        except Exception:
            pass

        # 2. Fallback: top posts from subreddit — filter by keyword client-side
        if subreddit:
            try:
                r = await http.get(
                    f"https://www.reddit.com/r/{subreddit}/top.json",
                    params={"t": "all", "limit": limit},
                    headers=headers,
                    timeout=15,
                )
                if r.status_code == 200:
                    children = r.json().get("data", {}).get("children", [])
                    if children:
                        keyword_tokens = [w.lower() for w in query.split() if w.lower() not in ("game", "the", "a", "an")]
                        filtered = [
                            c for c in children
                            if any(t in c["data"]["title"].lower() for t in keyword_tokens)
                        ] or children
                        return _parse_reddit_children(filtered)
            except Exception:
                pass

        # 3. Last resort: hot posts
        if subreddit:
            try:
                r = await http.get(
                    f"https://www.reddit.com/r/{subreddit}/hot.json",
                    params={"limit": limit},
                    headers=headers,
                    timeout=15,
                )
                if r.status_code == 200:
                    children = r.json().get("data", {}).get("children", [])
                    if children:
                        return _parse_reddit_children(children)
            except Exception:
                pass

        return []


def store_posts_to_chromadb(posts: list[dict], game_name: str, category: str | None):
    if not posts:
        return
    texts = [f"{p['title']}\n{p['selftext']}" for p in posts]
    metadatas = [
        {
            "game": game_name.lower(),
            "category": category or "general",
            "subreddit": p["subreddit"],
            "url": p["url"],
            "score": p["score"],
            "source": "reddit",
        }
        for p in posts
    ]
    ids = [f"reddit_{p['id']}" for p in posts]
    try:
        vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    except Exception:
        pass


@router.post("/chat")
async def chat(request: ChatRequest):
    category = request.category or "general"

    game_name = await extract_game_name(request.question)
    game_info = await get_rawg_game_info(game_name)
    is_list = isinstance(game_info, list)
    resolved_name = game_name

    context_parts = []

    if is_list and game_info:
        resolved_name = game_name
        games_text = "\n\n".join([
            f"- {g['name']} ({g.get('released', 'N/A')}) | "
            f"Rating: {g.get('rating')} | Metacritic: {g.get('metacritic')} | "
            f"Genres: {', '.join(g.get('genres', []))} | "
            f"Platforms: {', '.join(g.get('platforms', []))}"
            for g in game_info
        ])
        context_parts.append(f"Franchise Games (RAWG):\n{games_text}")
    elif isinstance(game_info, dict) and game_info:
        resolved_name = game_info.get("name", game_name)
        context_parts.append(
            f"Game Info (RAWG):\n"
            f"Name: {resolved_name}\n"
            f"Released: {game_info.get('released')}\n"
            f"Rating: {game_info.get('rating')}\n"
            f"Metacritic: {game_info.get('metacritic')}\n"
            f"Genres: {', '.join(game_info.get('genres', []))}\n"
            f"Platforms: {', '.join(game_info.get('platforms', []))}"
        )

    if category == "updates":
        if is_list:
            games_to_check = [g["name"] for g in game_info]
            slugs = [g.get("slug") for g in game_info]
            app_ids = list(await asyncio.gather(*[get_steam_id_from_rawg_slug(slug) for slug in slugs]))
        else:
            games_to_check = [resolved_name]
            slug = game_info.get("slug") if isinstance(game_info, dict) else None
            steam_id = await resolve_steam_app_id(game_name=resolved_name, rawg_slug=slug)
            app_ids = [steam_id]

        detailed = any(w in request.question.lower() for w in ["patch notes", "detailed", "full", "changelog", "what changed", "patch"])
        news_results = await asyncio.gather(*[
            get_steam_news(aid, detailed=detailed) for aid in app_ids if aid
        ])

        all_news = []
        for game_n, news in zip([g for g, aid in zip(games_to_check, app_ids) if aid], news_results):
            for item in news:
                all_news.append(f"[{game_n}] {item['title']}\n{item['contents']}")

        if all_news:
            context_parts.append("Official Steam News & Updates:\n\n" + "\n\n".join(all_news))
        else:
            context_parts.append("No Steam news available from the API. Use your own training knowledge to provide the most recent updates, patches, and news you know about for this game or franchise.")

    elif category in REDDIT_CATEGORIES:
        subreddit, search_keywords = await asyncio.gather(
            get_game_subreddit(resolved_name),
            extract_search_keywords(request.question),
        )
        posts = await fetch_reddit_posts(search_keywords, limit=25, subreddit=subreddit)
        keyword_tokens = [w.lower() for w in search_keywords.split() if w.lower() != "game"]
        if keyword_tokens:
            relevant = [p for p in posts if any(t in p["title"].lower() or t in p["selftext"].lower() for t in keyword_tokens)]
            posts = relevant[:5] if relevant else posts[:5]
        else:
            posts = posts[:5]
        store_posts_to_chromadb(posts, resolved_name.lower(), category)

        chroma_filter = {"$and": [
            {"game": {"$eq": resolved_name.lower()}},
            {"category": {"$eq": category}},
        ]} if category != "general" else {"game": {"$eq": resolved_name.lower()}}

        docs = vector_db.similarity_search(request.question, k=request.top_k, filter=chroma_filter)
        if docs:
            context_parts.append(
                "Community Knowledge (Reddit via ChromaDB):\n" +
                "\n\n".join([doc.page_content for doc in docs])
            )

    # account & billing: RAWG platform info + OpenAI's own knowledge (no Reddit)

    if not context_parts:
        return {"answer": "Please include the game name in your question.", "sources": []}

    context = "\n\n---\n\n".join(context_parts)
    system_prompt = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["general"])

    CATEGORY_INTENTS = {
        "gameplay": f"What are the gameplay mechanics, tips, or strategies for {resolved_name}?",
        "technical": f"What are common technical issues and fixes for {resolved_name}?",
        "account": f"How do I manage my account for {resolved_name}?",
        "billing": f"What are the pricing, refund, and billing details for {resolved_name}?",
        "community": f"What is the community like for {resolved_name}? Tell me about multiplayer and social features.",
        "updates": f"What are the latest updates and patch notes for {resolved_name}?",
        "general": request.question,
    }
    is_just_game_name = len(request.question.strip().split()) <= 2
    final_question = CATEGORY_INTENTS.get(category, request.question) if is_just_game_name else request.question

    openai_messages = [{"role": "system", "content": f"{system_prompt}\n\nGame Context:\n{context}"}]
    for msg in request.messages:
        openai_messages.append({"role": msg.role, "content": msg.content})
    openai_messages.append({"role": "user", "content": final_question})

    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=4096,
        messages=openai_messages,
    )

    return {
        "answer": response.choices[0].message.content,
        "sources": [],
    }
