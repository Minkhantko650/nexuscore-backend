from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from main import vector_db
import httpx
import asyncio
import os
import re

router = APIRouter()
client = OpenAI()

RAWG_KEY = os.getenv("RAWG_API_KEY")
YOUTUBE_KEY = os.getenv("YOUTUBE_API_KEY")

YOUTUBE_CATEGORIES = {"gameplay", "technical"}


class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    top_k: int = 5
    category: str | None = None
    messages: list[ChatMessage] = []


_FORMAT = (
    "\n\nFormat your response using markdown so it is easy to read:"
    "\n- Use **bold** for key terms, item names, and important values."
    "\n- Use `## Heading` for section headers when covering multiple topics."
    "\n- Use numbered lists for step-by-step instructions."
    "\n- Use bullet points for options, features, or short facts."
    "\n- Keep paragraphs short. Avoid walls of text."
    "\n- Do not wrap the entire response in a code block."
)

CATEGORY_PROMPTS = {
    "gameplay": "You are a gameplay expert. ONLY answer questions strictly about game mechanics, strategies, bosses, combat, weapons, missions, quests, and in-game progression. If the question is not about gameplay, respond with: 'This question is not related to gameplay. Please switch to the appropriate category.' IMPORTANT: When Reddit community posts are provided in the context, they represent real player opinions. Posts with higher scores mean MORE players agree. You MUST reflect what the community actually says — do NOT override community consensus with your own training knowledge. If the community says boss X is harder than boss Y, present it that way." + _FORMAT,
    "technical": "You are a technical support specialist. ONLY answer questions strictly about crashes, bugs, errors, performance issues, drivers, installations, and system requirements. If the question is not a technical issue, respond with: 'This question is not a technical issue. Please switch to the appropriate category.'" + _FORMAT,
    "account": "You are an account support specialist. ONLY answer questions strictly about account creation, login, password reset, 2FA, account linking, saves, and account security. Use the game's platform info (Steam, GOG, PlayStation, etc.) to give accurate platform-specific account guidance. If the question is not account-related, respond with: 'This question is not account-related. Please switch to the appropriate category.'" + _FORMAT,
    "billing": "You are a billing support specialist. ONLY answer questions strictly about pricing, purchases, refunds, DLC, subscriptions, and payment issues. Use the game's platform info to give accurate platform-specific billing guidance. If the question is not billing-related, respond with: 'This question is not billing-related. Please switch to the appropriate category.'" + _FORMAT,
    "community": "You are a community manager. ONLY answer questions strictly about multiplayer, co-op, PvP, clans, community events, reporting players, and social features. If the question is not community-related, respond with: 'This question is not community-related. Please switch to the appropriate category.'" + _FORMAT,
    "updates": "You are a patch notes expert. Answer questions about game updates, patches, changelogs, new seasons, balance changes, and upcoming features. Use the official Steam news if provided. If no Steam news is available, use your own training knowledge to answer about recent updates for the game. Only reject if the question is clearly unrelated to updates at all." + _FORMAT,
    "general": "You are a helpful gaming assistant. Provide general information about the game including its overview, genre, platforms, release date, and any general facts. Do not go deep into any specific category." + _FORMAT,
}

REDDIT_CATEGORIES = {"gameplay", "technical", "community", "general"}
OFFICIAL_ONLY_CATEGORIES = {"account", "billing"}

# Cache subreddit per game so we only detect it once
_subreddit_cache: dict[str, str] = {}


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

    try:
      async with httpx.AsyncClient() as http:
        r = await http.get(
            "https://api.rawg.io/api/games",
            params={"key": RAWG_KEY, "search": game_name, "page_size": page_size},
            headers={"User-Agent": "NexusCore/1.0"},
            timeout=20,
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
                    timeout=20,
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
    except Exception:
        return {} if not franchise else []


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
        max_tokens=30,
        messages=[
            {"role": "system", "content": (
                "Extract 1-4 Reddit search keywords from the question for searching inside a game's subreddit. "
                "If there are specific names (boss names, item names, quest names, character names, error messages), use those exactly. "
                "Strip the game title itself and filler words (how, do, I, a, an, the, in, for, of, to, can, does, what, why, where, when). "
                "Do NOT append 'game'. Do NOT use overly generic words like 'tips' or 'help' unless nothing specific exists. "
                "Examples: "
                "'how do I beat the toad prince in elden ring' -> 'toad prince', "
                "'best sword build elden ring' -> 'best sword build', "
                "'elden ring bosses' -> 'bosses', "
                "'game crashes on launch' -> 'crash launch', "
                "'enemy ai attack patterns' -> 'enemy ai attack', "
                "'how to defeat the ender dragon' -> 'ender dragon defeat', "
                "'how to get golden oriole potion' -> 'golden oriole potion'. "
                "Return only the keywords, nothing else."
            )},
            {"role": "user", "content": question},
        ],
    )
    return res.choices[0].message.content.strip()


async def get_game_subreddit(game_name: str) -> str:
    browser_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def _norm(text: str) -> str:
        return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()

    raw_words = game_name.replace(":", " ").replace("-", " ").replace(".", " ").split()
    norm_name = _norm(game_name)
    game_words = [w for w in norm_name.split() if len(w) >= 2]

    arabic_nums = re.findall(r'\d+', game_name)
    roman_to_arabic = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
                       "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
    roman_nums = [roman_to_arabic[w.lower()] for w in raw_words if w.lower() in roman_to_arabic]
    game_numbers = list(set(arabic_nums + roman_nums))

    clean_slug = re.sub(r'\s+', '', norm_name)
    stop_words = {"the", "and", "for", "of", "in", "at", "to", "a", "an"}

    def rank_subreddit(s: dict) -> float:
        score = float(s.get("subscribers", 0))
        name_slug = re.sub(r'[^a-z0-9]', '', s["display_name"].lower())
        if name_slug == clean_slug:
            score *= 10000
        elif clean_slug and clean_slug in name_slug:
            score *= 1000
        elif name_slug and name_slug in clean_slug and len(name_slug) >= 4:
            score *= 500
        if game_numbers and any(num in name_slug for num in game_numbers):
            score *= 100
        word_hits = sum(1 for w in game_words if len(w) >= 4 and w in name_slug)
        score *= (50 if word_hits >= 2 else 10 if word_hits == 1 else 1)
        return score

    def is_candidate(c_data: dict) -> bool:
        name_slug = re.sub(r'[^a-z0-9]', '', c_data["display_name"].lower())
        title_norm = _norm(c_data.get("title", ""))
        if any(w in name_slug for w in game_words if len(w) >= 3):
            return True
        if any(w in title_norm for w in game_words if len(w) >= 4):
            return True
        if clean_slug and len(name_slug) >= 3 and (name_slug in clean_slug or clean_slug in name_slug):
            return True
        return False

    meaningful = [w for w in game_words if w not in stop_words]
    search_queries = list(dict.fromkeys([
        game_name,
        " ".join(meaningful),
        *[w for w in meaningful if len(w) >= 4],
    ]))

    try:
        async with httpx.AsyncClient(follow_redirects=True) as http:
            for query in search_queries:
                r = await http.get(
                    "https://www.reddit.com/subreddits/search.json",
                    params={"q": query, "limit": 25, "include_over_18": "on"},
                    headers={"User-Agent": browser_ua},
                    timeout=10,
                )
                if r.status_code != 200:
                    continue
                children = r.json().get("data", {}).get("children", [])
                candidates = [c["data"] for c in children if is_candidate(c["data"])]
                if candidates:
                    best = max(candidates, key=rank_subreddit)
                    print(f"[DEBUG] Subreddit found: r/{best['display_name']} ({best.get('subscribers', 0)} subs) via query '{query}'")
                    return best["display_name"]
    except Exception:
        pass

    fallback = clean_slug if clean_slug else (game_words[0] if game_words else game_name.lower())
    print(f"[DEBUG] Subreddit not found, using fallback: r/{fallback}")
    return fallback


def _sanitize(text: str) -> str:
    """Strip null bytes and control characters that break OpenAI JSON parsing."""
    return "".join(c for c in (text or "") if c >= " " or c in "\n\t")


def _parse_reddit_children(children: list) -> list[dict]:
    posts = [
        {
            "id": p["data"]["id"],
            "title": _sanitize(p["data"]["title"]),
            "subreddit": p["data"]["subreddit"],
            "score": p["data"]["score"],
            "url": f"https://reddit.com{p['data']['permalink']}",
            "selftext": _sanitize(p["data"].get("selftext", ""))[:2000],
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
    texts = [
        f"[Score: {p['score']} | Comments: {p['num_comments']}]\n{p['title']}\n{p['selftext']}"
        for p in posts
    ]
    metadatas = [
        {
            "game": game_name.lower(),
            "category": category or "general",
            "subreddit": p["subreddit"],
            "url": p["url"],
            "score": p["score"],
            "num_comments": p["num_comments"],
            "source": "reddit",
        }
        for p in posts
    ]
    ids = [f"reddit_{p['id']}" for p in posts]
    try:
        vector_db.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    except Exception:
        pass


async def get_youtube_videos(game_name: str, question: str) -> list[dict]:
    if not YOUTUBE_KEY:
        return []
    query = f"{game_name} {question}"
    # Keywords from game name used to verify results are actually about this game
    game_keywords = [w.lower() for w in game_name.replace(":", "").replace("-", " ").split() if len(w) >= 4]
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "key": YOUTUBE_KEY,
                    "q": query,
                    "part": "snippet",
                    "type": "video",
                    "maxResults": 5,
                    "relevanceLanguage": "en",
                    "videoCategoryId": "20",
                },
                timeout=10,
            )
            r.raise_for_status()
            items = r.json().get("items", [])
            results = []
            for item in items:
                if not item.get("id", {}).get("videoId"):
                    continue
                title = item["snippet"]["title"].lower()
                channel = item["snippet"]["channelTitle"].lower()
                # Only keep videos where at least one game keyword appears in title or channel
                if any(kw in title or kw in channel for kw in game_keywords):
                    results.append({
                        "title": item["snippet"]["title"],
                        "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                        "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                        "channel": item["snippet"]["channelTitle"],
                    })
                if len(results) == 2:
                    break
            return results
    except Exception:
        return []



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

    youtube_keywords = request.question

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
        CATEGORY_FALLBACK_KEYWORDS = {
            "gameplay": "best gameplay tips strategies bosses",
            "technical": "common bugs technical issues fixes crashes",
            "community": "community multiplayer social events",
            "general": "overview guide tips review",
        }
        is_just_game_name = len(request.question.strip().split()) <= 2
        if is_just_game_name:
            keyword_question = CATEGORY_FALLBACK_KEYWORDS.get(category, "tips guide")
        else:
            keyword_question = request.question
        youtube_keywords = keyword_question  # keep YouTube in sync with actual topic

        # Use game_name (from extract_game_name, stable) not resolved_name (from RAWG, can vary)
        cache_key = game_name.lower()
        if cache_key in _subreddit_cache:
            subreddit = _subreddit_cache[cache_key]
            print(f"[DEBUG] Using cached subreddit for {game_name}: r/{subreddit}")
            search_keywords = await extract_search_keywords(keyword_question)
        else:
            subreddit, search_keywords = await asyncio.gather(
                get_game_subreddit(resolved_name),
                extract_search_keywords(keyword_question),
            )
            # Only cache if it was a real Reddit-found subreddit, not the fallback
            if subreddit and subreddit != game_name.replace(" ", "").lower() and len(subreddit) > 2:
                _subreddit_cache[cache_key] = subreddit
        posts = await fetch_reddit_posts(search_keywords, limit=25, subreddit=subreddit)
        print(f"[DEBUG] Reddit subreddit: r/{subreddit}, keywords: {search_keywords}, posts found: {len(posts)}")
        store_posts_to_chromadb(posts, resolved_name.lower(), category)

        if posts:
            top_posts = sorted(posts, key=lambda p: p["score"], reverse=True)[:5]
            context_parts.append(
                "Community Knowledge (Reddit — sorted by popularity, higher score = more players agree):\n" +
                "\n\n".join([
                    f"[Score: {p['score']} | Comments: {p['num_comments']}]\n{p['title']}\n{p['selftext']}"
                    for p in top_posts
                ])
            )

    # account & billing: RAWG platform info + OpenAI's own knowledge (no Reddit)

    if not context_parts:
        return {"answer": "Please include the game name in your question.", "sources": [], "videos": []}

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

    has_reddit = any("Community Knowledge" in p for p in context_parts)
    community_trust = (
        "\n\nIMPORTANT: Reddit community posts are provided above. Posts with higher scores mean more players agree. "
        "You MUST base your answer on what the community actually says. "
        "Do NOT override community consensus with your own training knowledge."
    ) if has_reddit else ""

    overview_hint = (
        f"\n\nThe user has opened the {category} support tab for {resolved_name}. "
        f"Give them a helpful {category} overview of {resolved_name}. This IS a valid {category} question."
    ) if is_just_game_name else ""

    game_lock = (
        f"\n\nYou are locked to answering ONLY about {resolved_name} for this entire conversation. "
        f"Every question in this chat is about {resolved_name}. Never switch to or mention a different game as the subject."
    )

    openai_messages = [{"role": "system", "content": f"{system_prompt}{community_trust}{overview_hint}{game_lock}\n\nGame Context:\n{context}"}]
    for msg in request.messages:
        openai_messages.append({"role": msg.role, "content": msg.content})
    openai_messages.append({"role": "user", "content": final_question})

    gpt_task = asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o",
        max_tokens=4096,
        messages=openai_messages,
    )

    if category in YOUTUBE_CATEGORIES:
        response, videos = await asyncio.gather(gpt_task, get_youtube_videos(resolved_name, youtube_keywords))
    else:
        response = await gpt_task
        videos = []

    return {
        "answer": response.choices[0].message.content,
        "sources": [],
        "videos": videos,
    }
