from fastapi import APIRouter
from pydantic import BaseModel
from openai import OpenAI
from main import vector_db
import httpx
import asyncio
import os
import re
import time
import datetime

router = APIRouter(prefix="/community", tags=["community"])
client = OpenAI()

RAWG_KEY = os.getenv("RAWG_API_KEY")
YOUTUBE_KEY = os.getenv("YOUTUBE_API_KEY")

YOUTUBE_CATEGORIES = {"gameplay", "technical"}

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
    "billing": "You are a billing support specialist. ONLY answer questions strictly about pricing, purchases, refunds, and payment issues. IMPORTANT: Many games are free-to-play (F2P) and have no traditional DLC — for these games focus on in-game currency (V-Bucks, Apex Coins, Lattice, Chrono Crystals, etc.), battle passes, cosmetic bundles, skin prices, the in-game shop, starter packs, and season pass value. Use the community posts provided to reflect real player sentiment on whether purchases are worth it. If the question is not billing-related, respond with: 'This question is not billing-related. Please switch to the appropriate category.'" + _FORMAT,
    "community": "You are a community manager. ONLY answer questions strictly about multiplayer, co-op, PvP, clans, community events, reporting players, and social features. If the question is not community-related, respond with: 'This question is not community-related. Please switch to the appropriate category.'" + _FORMAT,
    "updates": "You are a patch notes expert. Answer questions about game updates, patches, changelogs, new seasons, balance changes, and upcoming features. CRITICAL: When Steam news or Reddit update posts are provided in the context, that is REAL-TIME data fetched today — NEVER say 'as of my knowledge cutoff', 'my information is up to [date]', or any variation of a training cutoff disclaimer. If real-time data is provided, present it directly as current information. Only reject if the question is clearly unrelated to updates at all." + _FORMAT,
    "general": "You are a helpful gaming assistant. Provide general information about the game including its overview, genre, platforms, release date, and any general facts. Do not go deep into any specific category." + _FORMAT,
}

REDDIT_CATEGORIES = {"gameplay", "technical", "community", "general", "account", "billing", "updates"}

# Cache subreddit per game so we only detect it once
_subreddit_cache: dict[str, str] = {}

# Cache trending posts per game+category — entries expire after CACHE_TTL seconds
_trending_cache: dict[str, dict] = {}
CACHE_TTL = 1800  # 30 minutes — keeps results fresh without hammering Reddit

# Used for generic category browsing (bare game name typed).
# No game name prefix — inside the subreddit posts don't repeat the game name,
# so searching "elden ring guide" inside r/eldenring returns 0.
REDDIT_CATEGORY_KEYWORDS: dict[str, str] = {
    "gameplay": "guide build tips how",
    "technical": "crash bug fix error",
    # "lfg" / "anyone" / "discord" match how community posts are actually written
    "community": "community fan discussion lore theory lfg discord",
    "general": "review beginner overview worth",
    # Players post "lost save", "progress wiped", "cloud sync" not "account login password"
    "account": "save lost ban progress data",
    "billing": "battle pass worth buy price refund shop bundle skin",
    "updates": "update patch news announcement season changelog",
}

# Appended to topic keywords for specific queries so each tab returns different posts.
# e.g. "elden ring bosses" on Gameplay → "bosses guide tips"
#                           on Technical → "bosses crash bug fix"
CATEGORY_FLAVOR: dict[str, str] = {
    "gameplay": "guide tips how",
    "technical": "crash bug fix error",
    "community": "community fan discussion",
    "general": "",
    "account": "save ban progress lost",
    "billing": "battle pass price worth shop bundle",
    "updates": "update news patch",
}

# Broad filter keywords — post must contain at least one in title+body snippet
REDDIT_FILTER_KEYWORDS: dict[str, list[str]] = {
    "gameplay": ["how", "tips", "guide", "build", "help", "best", "strategy", "boss",
                 "quest", "weapon", "skill", "level", "farm", "craft", "beat", "win",
                 "unlock", "find", "kill", "damage", "difficulty", "class", "loadout",
                 "mission", "achievement", "tutorial", "mechanic", "combo", "exploit"],
    "technical": ["crash", "bug", "error", "fix", "lag", "fps", "freeze", "performance",
                  "issue", "problem", "help", "not working", "broken", "stuttering",
                  "glitch", "stutter", "driver", "load", "install", "update failed"],
    # Natural LFG / community language gamers actually use
    "community": ["lfg", "looking for", "anyone", "want to play", "need help", "summon",
                  "co-op", "coop", "discord", "multiplayer", "pvp", "together", "group",
                  "server", "clan", "guild", "join", "team", "partner", "play with"],
    "general":   ["review", "guide", "overview", "worth", "beginner", "start", "first",
                  "recommend", "opinion", "thoughts", "experience", "impression"],
    # Account posts talk about saves, bans, progress — not "account login password"
    "account":   ["save", "ban", "banned", "progress", "lost", "data", "account", "login",
                  "password", "cloud", "sync", "recover", "corrupt", "deleted", "missing",
                  "linked", "profile", "suspended", "reset", "wiped", "gone"],
    "billing":   ["dlc", "refund", "expansion", "pass", "purchase", "buy", "price",
                  "worth", "discount", "sale", "microtransaction", "payment", "store",
                  "battle pass", "battlepass", "bundle", "skin", "cosmetic", "currency",
                  "v-bucks", "vbucks", "apex coins", "lattice", "chrono crystals",
                  "shop", "item shop", "starter pack", "season pass", "free to play",
                  "f2p", "premium", "subscription", "spending", "money", "cost"],
    "updates":   ["update", "patch", "changelog", "season", "notes", "new content",
                  "hotfix", "nerf", "buff", "balance", "release", "added", "fixed"],
}


CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "gameplay": (
        "gameplay guides, strategies, tips, builds, boss fights, game mechanics, "
        "quests, weapons, skills, progression, how-to help, combat advice"
    ),
    "technical": (
        "technical issues — bugs, crashes, errors, performance problems, FPS drops, "
        "freezes, installation issues, driver problems, game not launching, stuttering"
    ),
    "community": (
        "community interaction — fan discussions, lore theories, community events, "
        "looking for group (LFG), co-op requests, finding teammates, multiplayer sessions, "
        "discord servers, friend requests, fan art with discussion, community milestones"
    ),
    "general": (
        "general game discussion — reviews, first impressions, opinions, "
        "recommendations, overall thoughts, whether the game is worth playing"
    ),
    "account": (
        "account problems — lost save data, progress reset or wiped, banned accounts, "
        "cloud sync failures, login issues, missing progress, corrupted saves"
    ),
    "billing": (
        "purchases and money — DLC worth buying, refund requests, pricing, "
        "microtransactions, battle pass value, in-game currency, cosmetic bundles, "
        "skin prices, item shop, starter packs, season pass, free-to-play spending, "
        "whether purchases are worth it, game passes, sales, payment issues"
    ),
    "updates": (
        "game updates — patches, changelogs, new seasons, balance changes, "
        "hotfixes, new content drops, what changed in recent updates"
    ),
}


async def classify_posts_by_category(
    posts: list[dict], category: str, game_name: str, strict: bool = False
) -> list[dict]:
    """GPT-mini pass: keep only posts genuinely relevant to the category."""
    if not posts or category not in CATEGORY_DESCRIPTIONS:
        return posts

    lines = []
    for i, p in enumerate(posts):
        snippet = p.get("selftext", "").strip()[:150].replace("\n", " ")
        line = f"{i}. {p['title']}"
        if snippet:
            line += f" | {snippet}"
        lines.append(line)

    description = CATEGORY_DESCRIPTIONS[category]
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=300,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a content filter for Reddit posts from the {game_name} gaming community. "
                        f"The category is: {description}. "
                        + (
                            # view_all: strictest — category must be the primary topic
                            "STRICT MODE: Only include posts where this category is the PRIMARY and EXPLICIT topic. "
                            "Reject if: category is a side mention, post is mostly about something else, "
                            "it's fan art, memes, pure appreciation posts, or vague enough to fit multiple categories. "
                            "When in doubt, REJECT. "
                            if strict else
                            # view_recent: firm but not paranoid — reject clearly off-topic, keep anything genuinely related
                            "Include posts where this category is clearly the main topic OR directly related to it "
                            "(questions, guides, tips, achievements, builds, discussions about that category). "
                            "REJECT: pure fan art, memes with no useful content, 'this game is beautiful' appreciation posts, "
                            "completely off-topic posts. "
                            "Accept anything a player in this category would genuinely find useful or relevant. "
                        ) +
                        "Return ONLY a comma-separated list of numbers like '0,3,5'. "
                        "If none qualify, return exactly: none"
                    ),
                },
                {"role": "user", "content": "\n".join(lines)},
            ],
        )
        raw = res.choices[0].message.content.strip().lower()
        print(f"[DEBUG] classify_posts: category={category!r}, GPT={raw!r}, input={len(posts)}")
        if raw == "none" or not raw:
            return []
        indices = [int(x.strip()) for x in raw.replace(".", ",").split(",") if x.strip().isdigit()]
        result = [posts[i] for i in indices if i < len(posts)]
        print(f"[DEBUG] classify_posts: {len(posts)} → {len(result)} posts kept")
        return result
    except Exception as e:
        print(f"[DEBUG] classify_posts error: {e} — skipping filter")
        return posts  # never block content on GPT failure


class ChatMessage(BaseModel):
    role: str
    content: str

class CommunityChatRequest(BaseModel):
    question: str
    game_name: str
    category: str | None = None
    messages: list[ChatMessage] = []


async def extract_game_name(question: str) -> str:
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=20,
        messages=[
            {"role": "system", "content": (
                "Extract the full official game title from the text. "
                "CRITICAL: If a character name and a game title are both present, return the GAME TITLE — "
                "the character is being played inside that game, not their own game. "
                "'how to play spiderman in marvel rivals' -> 'Marvel Rivals', "
                "'spiderman tips marvel rivals' -> 'Marvel Rivals', "
                "'iron man build marvel rivals' -> 'Marvel Rivals', "
                "'wraith tips apex legends' -> 'Apex Legends', "
                "'mirage guide apex legends' -> 'Apex Legends', "
                "'jett guide valorant' -> 'Valorant', "
                "'how to beat malenia elden ring' -> 'Elden Ring', "
                "'warhammer 3' -> 'Total War: Warhammer III', "
                "'fifa 24' -> 'EA Sports FC 24'. "
                "Return just the game name, nothing else."
            )},
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


async def get_steam_id_from_rawg_slug(slug: str | None) -> int | None:
    return await resolve_steam_app_id(game_name="", rawg_slug=slug)


async def get_steam_app_id(game_name: str) -> int | None:
    return await resolve_steam_app_id(game_name=game_name)


def strip_html(text: str) -> str:
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{[^}]+\}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return text


async def get_rawg_subreddit(game_name: str) -> str | None:
    """Return the official subreddit name from RAWG's reddit_url field, or None."""
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                "https://api.rawg.io/api/games",
                params={"key": RAWG_KEY, "search": game_name, "page_size": 1},
                headers={"User-Agent": "NexusCore/1.0"},
                timeout=10,
            )
            if r.status_code != 200:
                return None
            results = r.json().get("results", [])
            if not results:
                return None
            slug = results[0].get("slug")
            if not slug:
                return None
            detail = await http.get(
                f"https://api.rawg.io/api/games/{slug}",
                params={"key": RAWG_KEY},
                headers={"User-Agent": "NexusCore/1.0"},
                timeout=10,
            )
            if detail.status_code != 200:
                return None
            reddit_url = detail.json().get("reddit_url") or ""
            if reddit_url:
                m = re.search(r"reddit\.com/r/([A-Za-z0-9_]+)", reddit_url)
                if m:
                    sub = m.group(1)
                    print(f"[DEBUG] RAWG subreddit for {game_name!r}: r/{sub}")
                    return sub
    except Exception as e:
        print(f"[DEBUG] get_rawg_subreddit error for {game_name!r}: {e}")
    return None


async def _get_sub_subscribers(subreddit: str) -> int:
    """Fetch subscriber count for a subreddit — used to pick the largest community."""
    browser_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    try:
        async with httpx.AsyncClient(follow_redirects=True) as http:
            r = await http.get(
                f"https://www.reddit.com/r/{subreddit}/about.json",
                headers={"User-Agent": browser_ua},
                timeout=8,
            )
            if r.status_code == 200:
                return r.json().get("data", {}).get("subscribers", 0)
    except Exception:
        pass
    return 0


async def best_subreddit(game_name: str) -> str:
    """Return the largest subreddit for a game.
    Runs RAWG and Reddit search in parallel, checks subscriber counts,
    and always picks the one with more members."""
    rawg_sub, reddit_sub = await asyncio.gather(
        get_rawg_subreddit(game_name),
        get_game_subreddit(game_name),
    )
    if not rawg_sub:
        return reddit_sub
    if not reddit_sub:
        return rawg_sub
    if rawg_sub.lower() == reddit_sub.lower():
        return rawg_sub
    # Both found different subs — pick the bigger community
    rawg_count, reddit_count = await asyncio.gather(
        _get_sub_subscribers(rawg_sub),
        _get_sub_subscribers(reddit_sub),
    )
    chosen = rawg_sub if rawg_count >= reddit_count else reddit_sub
    print(f"[DEBUG] Sub size check: r/{rawg_sub}({rawg_count:,}) vs r/{reddit_sub}({reddit_count:,}) → r/{chosen}")
    return chosen


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
                "CRITICAL RULE: Always preserve character names, hero names, champion names, agent names, operator names, "
                "and legend names exactly as written (e.g. Spider-Man, Spiderman, Iron Man, Wraith, Bloodhound, Jett, Reyna). "
                "These are the most important search terms — NEVER strip them. "
                "Strip ONLY the game title itself and pure filler words (how, do, I, a, an, the, in, for, of, to, can, does, what, why, where, when). "
                "Do NOT strip: character names, item names, ability names, weapon names, boss names, skin names, battle pass, currency names. "
                "Do NOT append 'game'. Do NOT use overly generic words like 'tips' or 'help' unless nothing specific exists. "
                "Examples: "
                "'how do I beat the toad prince in elden ring' -> 'toad prince', "
                "'best sword build elden ring' -> 'best sword build', "
                "'spider-man guide marvel rivals' -> 'spider-man guide', "
                "'how to play spiderman in marvel rivals' -> 'spiderman guide', "
                "'best spiderman tips marvel rivals' -> 'spiderman tips', "
                "'is the battle pass worth it in fortnite' -> 'battle pass worth', "
                "'wraith tips apex legends' -> 'wraith tips', "
                "'game crashes on launch' -> 'crash launch', "
                "'how to defeat the ender dragon' -> 'ender dragon defeat', "
                "'how to get golden oriole potion' -> 'golden oriole potion'. "
                "Return only the keywords, nothing else."
            )},
            {"role": "user", "content": question},
        ],
    )
    return res.choices[0].message.content.strip()


_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5",
          "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
_STOP_WORDS = {"the", "and", "for", "of", "in", "at", "to", "a", "an"}


async def get_game_subreddit(game_name: str) -> str:
    browser_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    def _norm(text: str) -> str:
        # Strip apostrophes, special chars — keeps alphanumeric + spaces
        return re.sub(r'[^a-z0-9\s]', ' ', text.lower()).strip()

    raw_words = game_name.replace(":", " ").replace("-", " ").replace(".", " ").split()
    norm_name = _norm(game_name)
    # game_words come from the normalized name so apostrophes in "Baldur's" don't break matching
    game_words = [w for w in norm_name.split() if len(w) >= 2]

    arabic_nums = re.findall(r'\d+', game_name)
    roman_nums = [_ROMAN[w.lower()] for w in raw_words if w.lower() in _ROMAN]
    game_numbers = list(set(arabic_nums + roman_nums))

    # Compact slug: "eldenring", "baldursgate3", "counterstrike2"
    clean_slug = re.sub(r'\s+', '', norm_name)

    def rank_subreddit(s: dict) -> float:
        score = float(s.get("subscribers", 0))
        name_slug = re.sub(r'[^a-z0-9]', '', s["display_name"].lower())
        if name_slug == clean_slug:
            score *= 10000                            # perfect match: r/eldenring
        elif clean_slug and clean_slug in name_slug:
            score *= 1000                             # game slug inside sub: r/eldenringlore
        elif name_slug and name_slug in clean_slug and len(name_slug) >= 4:
            score *= 500                              # sub slug inside game slug: r/halo for "Halo Infinite"
        if game_numbers and any(num in name_slug for num in game_numbers):
            score *= 100                              # version match: r/thewitcher3
        word_hits = sum(1 for w in game_words if len(w) >= 4 and w in name_slug)
        score *= (50 if word_hits >= 2 else 10 if word_hits == 1 else 1)
        return score

    def is_candidate(c_data: dict) -> bool:
        name_slug = re.sub(r'[^a-z0-9]', '', c_data["display_name"].lower())
        # Also check the subreddit's full title — catches r/GlobalOffensive for "Counter-Strike 2"
        # because its title contains "Counter-Strike"
        title_norm = _norm(c_data.get("title", ""))
        if any(w in name_slug for w in game_words if len(w) >= 3):
            return True
        if any(w in title_norm for w in game_words if len(w) >= 4):
            return True
        # Sub slug is a substring of game slug or vice-versa (e.g. "cs2" inside "counterstrike2")
        if clean_slug and len(name_slug) >= 3 and (name_slug in clean_slug or clean_slug in name_slug):
            return True
        return False

    meaningful = [w for w in game_words if w not in _STOP_WORDS]
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
            "selftext": _sanitize(p["data"].get("selftext", ""))[:4000],
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


async def fetch_reddit_posts(
    query: str,
    limit: int = 25,
    subreddit: str | None = None,
    time_filter: str = "all",
    strict: bool = False,
    skip_search: bool = False,
) -> list[dict]:
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
    keyword_tokens = [w.lower() for w in query.split() if w.lower() not in ("game", "the", "a", "an")]

    async with httpx.AsyncClient(follow_redirects=True) as http:
        # Tier 1: Reddit relevance search inside the subreddit.
        # sort=relevance returns posts matching the query terms, not just highest-scored.
        # Game name is NOT included in query — posts inside the subreddit don't repeat it.
        if not skip_search:
            search_url = (
                f"https://www.reddit.com/r/{subreddit}/search.json"
                if subreddit else "https://www.reddit.com/search.json"
            )
            # Tier 1: full multi-keyword search
            try:
                r = await http.get(
                    search_url,
                    params={"q": query, "sort": "relevance", "t": time_filter, "limit": limit, "restrict_sr": "1"},
                    headers=headers,
                    timeout=15,
                )
                if r.status_code == 200:
                    children = r.json().get("data", {}).get("children", [])
                    if children:
                        # Apply keyword filter — Reddit's relevance with broad terms
                        # like "how" matches unrelated posts easily.
                        filtered = [
                            c for c in children
                            if any(
                                t in (c["data"]["title"] + " " + c["data"].get("selftext", "")[:500]).lower()
                                for t in keyword_tokens
                            )
                        ]
                        if filtered:
                            return _parse_reddit_children(filtered)
                        # No keyword match — fall through to Tier 1.5 with a more
                        # specific single keyword rather than returning unrelated posts.
                        print(f"[DEBUG] fetch_reddit_posts Tier 1 keyword filter matched 0/{len(children)}, falling to Tier 1.5")
                else:
                    print(f"[DEBUG] fetch_reddit_posts search HTTP {r.status_code} for query={query!r}")
            except Exception as e:
                print(f"[DEBUG] fetch_reddit_posts search exception: {e}")

            # Tier 1.5: retry with the single most specific keyword (longest non-generic token).
            # Picks e.g. "spiderman" over "guide", "wraith" over "tips".
            _GENERIC_TOKENS = {"guide", "tips", "help", "how", "best", "fix", "issue",
                               "problem", "question", "anyone", "need", "want", "good"}
            _tokens = query.split()
            _specific = [t for t in _tokens if t.lower() not in _GENERIC_TOKENS]
            primary = max(_specific, key=len) if _specific else (max(_tokens, key=len) if _tokens else "")
            if primary and primary.lower() != query.lower():
                try:
                    r = await http.get(
                        search_url,
                        params={"q": primary, "sort": "relevance", "t": time_filter, "limit": limit, "restrict_sr": "1"},
                        headers=headers,
                        timeout=15,
                    )
                    if r.status_code == 200:
                        children = r.json().get("data", {}).get("children", [])
                        if children:
                            filtered = [
                                c for c in children
                                if any(
                                    t in (c["data"]["title"] + " " + c["data"].get("selftext", "")[:500]).lower()
                                    for t in keyword_tokens
                                )
                            ]
                            if filtered:
                                print(f"[DEBUG] fetch_reddit_posts Tier 1.5 succeeded: {primary!r}")
                                return _parse_reddit_children(filtered)
                            # Still no match — fall through to Tier 2
                            print(f"[DEBUG] fetch_reddit_posts Tier 1.5 keyword filter matched 0, falling to Tier 2")
                except Exception as e:
                    print(f"[DEBUG] fetch_reddit_posts Tier 1.5 exception: {e}")

            # Tier 1.6: time-filter fallback — old/inactive games have no recent patches or
            # LFG posts, so year-limited search returns nothing. Retry with all-time so we
            # can surface older-but-relevant posts instead of falling to unrelated hot posts.
            if time_filter != "all":
                try:
                    r = await http.get(
                        search_url,
                        params={"q": query, "sort": "relevance", "t": "all", "limit": limit, "restrict_sr": "1"},
                        headers=headers,
                        timeout=15,
                    )
                    if r.status_code == 200:
                        children = r.json().get("data", {}).get("children", [])
                        if children:
                            filtered = [
                                c for c in children
                                if any(
                                    t in (c["data"]["title"] + " " + c["data"].get("selftext", "")[:500]).lower()
                                    for t in keyword_tokens
                                )
                            ]
                            if filtered:
                                print(f"[DEBUG] fetch_reddit_posts Tier 1.6 all-time fallback succeeded")
                                return _parse_reddit_children(filtered)
                except Exception as e:
                    print(f"[DEBUG] fetch_reddit_posts Tier 1.6 exception: {e}")

        # Tier 2: top posts from subreddit, filtered by keyword tokens in title+selftext.
        # Checks selftext too — many posts have the topic in body not title.
        # Never returns unfiltered posts: that's what causes all tabs to look identical.
        if subreddit:
            try:
                r = await http.get(
                    f"https://www.reddit.com/r/{subreddit}/top.json",
                    params={"t": time_filter, "limit": limit},
                    headers=headers,
                    timeout=15,
                )
                if r.status_code == 200:
                    children = r.json().get("data", {}).get("children", [])
                    if children:
                        filtered = [
                            c for c in children
                            if any(
                                t in (c["data"]["title"] + " " + c["data"].get("selftext", "")[:500]).lower()
                                for t in keyword_tokens
                            )
                        ]
                        if filtered:
                            return _parse_reddit_children(filtered)
                        # Don't fall back to unfiltered — that makes every tab look the same.
                        # Fall through to Tier 3 (hot) which is at least different content.
                else:
                    print(f"[DEBUG] fetch_reddit_posts top HTTP {r.status_code} for r/{subreddit}")
            except Exception as e:
                print(f"[DEBUG] fetch_reddit_posts top exception: {e}")

        # Tier 3: hot posts (only in non-strict mode)
        if subreddit and not strict:
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
    # Build keyword filter — include words ≥3 chars; if none qualify (e.g. "cs2", "wow")
    # skip the filter entirely since the search query is already game-specific.
    game_keywords = [w.lower() for w in game_name.replace(":", "").replace("-", " ").split() if len(w) >= 3]
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                "https://www.googleapis.com/youtube/v3/search",
                params={
                    "key": YOUTUBE_KEY,
                    "q": query,
                    "part": "snippet",
                    "type": "video",
                    "maxResults": 10,
                    "relevanceLanguage": "en",
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
                # If we have keywords, at least one must appear in title or channel name.
                # Skip this check entirely for very short game names (keywords list is empty).
                if game_keywords and not any(kw in title or kw in channel for kw in game_keywords):
                    continue
                results.append({
                    "title": item["snippet"]["title"],
                    "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                    "channel": item["snippet"]["channelTitle"],
                })
                if len(results) == 2:
                    break
            print(f"[DEBUG] get_youtube_videos: game={game_name!r}, returned {len(results)} videos (from {len(items)} results)")
            return results
    except Exception as e:
        print(f"[DEBUG] get_youtube_videos error: {e}")
        return []


@router.post("/chat")
async def community_chat(request: CommunityChatRequest):
    category = request.category or "general"

    _QUESTION_WORDS = {
        "how", "what", "where", "why", "when", "which", "who",
        "is", "are", "was", "were", "will", "would", "should", "could",
        "does", "did", "has", "have", "do", "can",
        "beat", "defeat", "kill", "fight", "farm", "grind", "unlock",
        "get", "find", "obtain", "acquire", "collect", "earn", "win",
        "survive", "escape", "avoid", "counter", "reach", "access",
        "trigger", "spawn", "drop",
        "craft", "build", "make", "combine", "equip", "upgrade",
        "level", "max", "stack", "boost",
        "fix", "solve", "use", "start", "complete", "finish",
        "open", "buy", "purchase", "reset", "reload", "save", "load",
        "increase", "decrease", "deal",
        "need", "needs", "needed", "require", "requires", "required",
        "want", "trying",
        "best", "fastest", "easiest", "optimal", "efficient", "good",
        "better", "tips", "guide", "help", "recommend",
    }

    clean_game_name = await extract_game_name(request.game_name)
    game_info = await get_rawg_game_info(clean_game_name)
    is_list = isinstance(game_info, list)
    resolved_name = clean_game_name

    context_parts = []

    if is_list and game_info:
        games_text = "\n\n".join([
            f"- {g['name']} ({g.get('released', 'N/A')}) | "
            f"Rating: {g.get('rating')} | Metacritic: {g.get('metacritic')} | "
            f"Genres: {', '.join(g.get('genres', []))} | "
            f"Platforms: {', '.join(g.get('platforms', []))}"
            for g in game_info
        ])
        context_parts.append(f"Franchise Games (RAWG):\n{games_text}")
    elif isinstance(game_info, dict) and game_info:
        resolved_name = game_info.get("name", clean_game_name)
        context_parts.append(
            f"Game Info (RAWG):\n"
            f"Name: {resolved_name}\n"
            f"Released: {game_info.get('released')}\n"
            f"Rating: {game_info.get('rating')}\n"
            f"Metacritic: {game_info.get('metacritic')}\n"
            f"Genres: {', '.join(game_info.get('genres', []))}\n"
            f"Platforms: {', '.join(game_info.get('platforms', []))}"
        )

    reddit_posts: list[dict] = []
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

        today_str = datetime.date.today().strftime("%B %d, %Y")
        if all_news:
            context_parts.append(
                f"Official Steam News & Updates (real-time data fetched {today_str}):\n\n" +
                "\n\n".join(all_news)
            )
        else:
            # Steam App ID not found or Steam returned no news — search Reddit for
            # patch/update posts as a fallback rather than falling back to GPT training data.
            print(f"[DEBUG] updates: no Steam news for {resolved_name}, falling back to Reddit")
            fallback_sub = _subreddit_cache.get(clean_game_name.lower()) or await best_subreddit(resolved_name)
            if fallback_sub and len(fallback_sub) > 2:
                _subreddit_cache[clean_game_name.lower()] = fallback_sub
            reddit_updates = await fetch_reddit_posts(
                "update patch changelog news announcement season",
                limit=20, subreddit=fallback_sub, time_filter="all"
            )
            if reddit_updates:
                top_updates = sorted(reddit_updates, key=lambda p: p["score"], reverse=True)[:5]
                context_parts.append(
                    f"Community Update Posts from Reddit (fetched {today_str}, no Steam API data available):\n\n" +
                    "\n\n".join([
                        f"[Score: {p['score']} | Comments: {p['num_comments']}]\n{p['title']}\n{p['selftext']}"
                        for p in top_updates
                    ])
                )
            else:
                context_parts.append(
                    f"No real-time update data found via Steam or Reddit for {resolved_name}. "
                    f"Today's date is {today_str}. Answer based on any available game context above, "
                    f"and be transparent if specific patch details are unavailable."
                )

    elif category in REDDIT_CATEGORIES:
        CATEGORY_FALLBACK_KEYWORDS = {
            "gameplay": "best gameplay tips strategies bosses",
            "technical": "common bugs technical issues fixes crashes",
            "community": "community multiplayer social events",
            "general": "overview guide tips review",
            "account": "save lost progress ban login account",
            "billing": "battle pass worth buy price shop bundle skin",
        }
        _q = re.sub(r'[^\w\s]', '', request.question.strip().lower())
        _g = re.sub(r'[^\w\s]', '', request.game_name.strip().lower())
        # Also treat a literal "?" anywhere in the query as a question signal
        _has_action = (
            bool(set(request.question.lower().split()) & _QUESTION_WORDS)
            or "?" in request.question
        )
        is_just_game_name = len(request.question.strip().split()) <= 2 or (_q == _g and not _has_action)
        if is_just_game_name:
            keyword_question = CATEGORY_FALLBACK_KEYWORDS.get(category, "tips guide")
        else:
            keyword_question = request.question
        youtube_keywords = keyword_question  # keep YouTube in sync with actual topic

        cache_key = clean_game_name.lower()
        if cache_key in _subreddit_cache:
            subreddit = _subreddit_cache[cache_key]
            print(f"[DEBUG] Using cached subreddit for {clean_game_name}: r/{subreddit}")
            search_keywords = await extract_search_keywords(keyword_question)
        else:
            subreddit, search_keywords = await asyncio.gather(
                best_subreddit(resolved_name),
                extract_search_keywords(keyword_question),
            )
            if subreddit and len(subreddit) > 2:
                _subreddit_cache[cache_key] = subreddit
        reddit_posts = await fetch_reddit_posts(search_keywords, limit=25, subreddit=subreddit)
        print(f"[DEBUG] Reddit subreddit: r/{subreddit}, keywords: {search_keywords}, posts found: {len(reddit_posts)}")
        store_posts_to_chromadb(reddit_posts, resolved_name.lower(), category)

        if reddit_posts:
            top_posts = sorted(reddit_posts, key=lambda p: p["score"], reverse=True)[:5]
            context_parts.append(
                "Community Knowledge (Reddit — sorted by popularity, higher score = more players agree):\n" +
                "\n\n".join([
                    f"[Score: {p['score']} | Comments: {p['num_comments']}]\n{p['title']}\n{p['selftext']}"
                    for p in top_posts
                ])
            )

    # account & billing: RAWG platform info + OpenAI's own knowledge

    if not context_parts:
        return {"answer": "Couldn't find information for this game.", "posts": [], "videos": []}

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
    _q2 = re.sub(r'[^\w\s]', '', request.question.strip().lower())
    _g2 = re.sub(r'[^\w\s]', '', request.game_name.strip().lower())
    _has_action2 = (
        bool(set(request.question.lower().split()) & _QUESTION_WORDS)
        or "?" in request.question
    )
    is_just_game_name = len(request.question.strip().split()) <= 2 or (_q2 == _g2 and not _has_action2)
    final_question = CATEGORY_INTENTS.get(category, request.question) if is_just_game_name else request.question

    today_str = datetime.date.today().strftime("%B %d, %Y")
    date_injection = f"\n\nToday's date is {today_str}. All data provided in the context is real-time and current."

    has_reddit = any("Community Knowledge" in p for p in context_parts)
    community_trust = (
        "\n\nIMPORTANT: Reddit community posts are provided above. Posts with higher scores mean more players agree. "
        "You MUST base your answer on what the community actually says. "
        "Do NOT override community consensus with your own training knowledge."
    ) if has_reddit else ""

    game_lock = (
        f"\n\nYou are locked to answering ONLY about {resolved_name} for this entire conversation. "
        f"Every question in this chat is about {resolved_name}. Never switch to or mention a different game as the subject."
    )

    if is_just_game_name:
        # User just typed the game name and opened a tab — they want a category overview.
        # The strict "ONLY answer X or reject" prompt causes GPT to reject a bare game name
        # because it doesn't look like a specific question. Use a plain overview prompt instead.
        CATEGORY_OVERVIEW_LABELS = {
            "gameplay": "gameplay mechanics, tips, strategies, and what to expect",
            "technical": "common technical issues, known bugs, performance tips, and fixes",
            "account": "account management, save data, platform-specific account info, and common issues",
            "billing": "DLC, pricing, what's worth buying, and purchase/refund info",
            "community": "the multiplayer scene, community features, how to find groups, and social aspects",
            "updates": "recent patches, balance changes, new content, and what changed",
            "general": "overview, genre, what makes it unique, and whether it's worth playing",
        }
        label = CATEGORY_OVERVIEW_LABELS.get(category, category)
        system_content = (
            f"You are a gaming assistant for {resolved_name}. "
            f"Give the user a helpful, well-structured overview of {resolved_name} focused on: {label}. "
            f"Use the game context provided below as your primary source. "
            f"{_FORMAT}{community_trust}{game_lock}{date_injection}\n\n"
            f"Game Context:\n{context}"
        )
    else:
        system_prompt = CATEGORY_PROMPTS.get(category, CATEGORY_PROMPTS["general"])
        system_content = (
            f"{system_prompt}{community_trust}{game_lock}{date_injection} "
            f"You have access to official game data about {resolved_name}. Combine all sources into one complete answer.\n\n"
            f"Game Context:\n{context}"
        )

    openai_messages = [{"role": "system", "content": system_content}]
    for msg in request.messages:
        openai_messages.append({"role": msg.role, "content": msg.content})
    openai_messages.append({"role": "user", "content": final_question})

    gpt_task = asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4o",
        max_tokens=1024,
        messages=openai_messages,
    )

    if category in YOUTUBE_CATEGORIES:
        response, videos = await asyncio.gather(gpt_task, get_youtube_videos(resolved_name, youtube_keywords))
    else:
        response = await gpt_task
        videos = []

    return {
        "answer": response.choices[0].message.content,
        "posts": reddit_posts,
        "videos": videos,
    }


@router.delete("/cache")
async def clear_cache():
    _trending_cache.clear()
    _subreddit_cache.clear()
    return {"cleared": True}


@router.get("/trending")
async def get_trending(game: str, category: str = "general", view_all: bool = False, raw: bool = False):
    if not game.strip() or category not in REDDIT_CATEGORIES:
        return {"posts": [], "videos": []}

    # view_all=False → top posts from last year (recent & relevant)
    # view_all=True  → top posts from all time (most upvoted ever)
    time_filter = "all" if view_all else "year"
    result_limit = 12 if view_all else 10
    cache_key = f"{game.lower()}:{category}:{'all' if view_all else 'year'}"

    cached = _trending_cache.get(cache_key)
    if cached and time.time() - cached["ts"] < CACHE_TTL:
        return {"posts": cached["posts"], "videos": cached.get("videos", [])}

    clean_game = re.sub(r'[^\w\s]', ' ', game).strip()
    is_specific = len(game.strip().split()) > 2
    if is_specific:
        # Extract game name (for subreddit lookup) and topic keywords in parallel.
        game_name_clean, topic_keywords = await asyncio.gather(
            extract_game_name(game),
            extract_search_keywords(game),
        )
        # For specific queries the topic keyword IS the search — don't dilute it with
        # generic flavor words ("tips how guide") that pull in unrelated posts.
        # Add only the single most category-identifying word so Reddit can still
        # differentiate tabs (e.g. gameplay→"guide", technical→"crash").
        CATEGORY_HINT = {
            "gameplay": "guide",
            "technical": "crash",
            "community": "community",
            "general": "",
            "account": "save",
            "billing": "price",
            "updates": "update",
        }
        hint = CATEGORY_HINT.get(category, "")
        # Only append hint if it isn't already present in the extracted keywords
        if hint and hint not in topic_keywords.lower().split():
            search_keywords = f"{topic_keywords} {hint}".strip()
        else:
            search_keywords = topic_keywords
    else:
        # Always GPT-extract the proper title for subreddit lookup even for short inputs.
        # "witcher 3" → "The Witcher 3: Wild Hunt" → RAWG finds r/Witcher3 not r/thewitcher3.
        game_name_clean = await extract_game_name(game)
        search_keywords = REDDIT_CATEGORY_KEYWORDS[category]

    # Always pick the subreddit with the most members — avoids dead official subs.
    subreddit_key = game_name_clean.lower()
    cached_sub = _subreddit_cache.get(subreddit_key)
    if cached_sub:
        subreddit = cached_sub
    else:
        subreddit = await best_subreddit(game_name_clean)
        if subreddit and len(subreddit) > 2:
            _subreddit_cache[subreddit_key] = subreddit

    # strict=True blocks Tier 3 (hot posts, unfiltered) from running.
    # Applied when: specific topic query OR view_all.
    # view_all = "best relevant posts of all time" — if Tier 1/2 find nothing, return
    # empty rather than dumping random hot posts. view_recent is more lenient because
    # year-limited top posts are focused enough that Tier 3 is rarely reached anyway.
    posts, videos = await asyncio.gather(
        fetch_reddit_posts(search_keywords, limit=100, subreddit=subreddit, time_filter=time_filter, strict=is_specific or view_all),
        get_youtube_videos(game_name_clean, search_keywords),
    )

    # GPT classification: semantically filter posts to only those relevant to the tab.
    # Replaces brittle keyword matching — understands context, not just word presence.
    # e.g. "Finally beat Malenia!" → Gameplay ✓   "HOW is my PC crashing" → Technical ✓
    if posts and not raw:
        raw_posts = posts  # keep pre-GPT list as fallback
        # Always use non-strict classification — strict mode rejects too many valid posts
        # (clips, character highlights, discussion posts) leaving view_all nearly empty.
        # fetch_reddit_posts already uses strict=True for view_all to block Tier 3 junk;
        # the classification layer should be permissive so good posts aren't thrown away.
        posts = await classify_posts_by_category(posts, category, game_name_clean, strict=False)
        # If GPT wiped everything (too aggressive), surface the top pre-GPT posts
        # so the panel never shows completely empty for a valid game+category.
        if not posts and raw_posts:
            posts = sorted(raw_posts, key=lambda p: p["score"], reverse=True)[:5]
            print(f"[DEBUG] GPT returned 0 — falling back to top {len(posts)} keyword-filtered posts")

    print(f"[DEBUG] get_trending: game={game!r}, category={category}, subreddit=r/{subreddit}, keywords={search_keywords!r}, posts={len(posts)}, videos={len(videos)}")
    store_posts_to_chromadb(posts, game.lower(), category)

    top = sorted(posts, key=lambda p: p["score"], reverse=True)[:result_limit]
    result = [
        {
            "title": p["title"],
            "score": p["score"],
            "url": p["url"],
            "num_comments": p["num_comments"],
            "selftext": p.get("selftext", ""),
        }
        for p in top
    ]
    _trending_cache[cache_key] = {"posts": result, "videos": videos, "ts": time.time()}
    return {"posts": result, "videos": videos}


async def fetch_post_comments(post_url: str, limit: int = 15) -> list[str]:
    import re as _re
    browser_ua = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    # Extract post ID from URL: /r/<sub>/comments/<id>/...
    match = _re.search(r"/comments/([A-Za-z0-9]+)", post_url)
    if not match:
        print(f"[DEBUG] fetch_post_comments: could not extract post ID from {post_url}")
        return []
    post_id = match.group(1)
    # Use the proper Reddit API comments endpoint — not the page-URL+.json scraping trick
    api_url = f"https://www.reddit.com/comments/{post_id}.json"
    print(f"[DEBUG] fetch_post_comments: fetching {api_url}")
    try:
        async with httpx.AsyncClient(follow_redirects=True) as http:
            r = await http.get(
                api_url,
                params={"limit": limit, "sort": "top", "depth": 1},
                headers={"User-Agent": browser_ua, "Accept": "application/json"},
                timeout=15,
            )
            print(f"[DEBUG] fetch_post_comments: status={r.status_code}")
            if r.status_code != 200:
                return []
            data = r.json()
            if not isinstance(data, list) or len(data) < 2:
                return []
            children = data[1].get("data", {}).get("children", [])
            print(f"[DEBUG] fetch_post_comments: {len(children)} comments found")
            comments = []
            for child in children:
                if child.get("kind") != "t1":
                    continue
                body = _sanitize(child["data"].get("body", "")).strip()
                if body and body not in ("[deleted]", "[removed]"):
                    score = child["data"].get("score", 0)
                    comments.append((score, body))
            comments.sort(key=lambda x: x[0], reverse=True)
            return [body for _, body in comments[:limit]]
    except Exception as e:
        print(f"[DEBUG] fetch_post_comments: exception {e}")
        return []


class PostAnswerRequest(BaseModel):
    title: str
    selftext: str
    url: str
    category: str
    game_name: str


def _html_to_text(html: str) -> str:
    text = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL)
    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'</tr>', '\n', text)
    text = re.sub(r'</td>|</th>', '\t', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def fetch_external_content(url: str) -> str:
    """Fetch readable text from external URLs found in post bodies."""
    browser_ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    try:
        async with httpx.AsyncClient(follow_redirects=True) as http:
            # Google Sheets — HTML export includes ALL sheet tabs
            gs_match = re.search(r'docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)', url)
            if gs_match:
                sheet_id = gs_match.group(1)
                for export_url in [
                    f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=html",
                    f"https://docs.google.com/spreadsheets/d/{sheet_id}/pub?output=html",
                ]:
                    r = await http.get(export_url, headers={"User-Agent": browser_ua}, timeout=15)
                    if r.status_code == 200 and "html" in r.headers.get("content-type", ""):
                        text = _html_to_text(r.text)
                        print(f"[DEBUG] Google Sheet fetched: {len(text)} chars")
                        return f"[Google Sheets content]\n{text[:8000]}"
                print(f"[DEBUG] Google Sheet not publicly accessible: {url}")
                return f"[Note: This post links to a Google Sheet at {url} — the full sheet content requires opening it directly in a browser. The description above summarizes what it contains.]"

            # Pastebin — fetch raw
            if "pastebin.com" in url:
                raw_url = re.sub(r'pastebin\.com/(?!raw/)', 'pastebin.com/raw/', url)
                r = await http.get(raw_url, headers={"User-Agent": browser_ua}, timeout=10)
                if r.status_code == 200:
                    return f"[Paste content]\n{r.text[:6000]}"

            # Generic page — strip HTML
            r = await http.get(url, headers={"User-Agent": browser_ua}, timeout=10)
            if r.status_code == 200 and "text" in r.headers.get("content-type", ""):
                text = _html_to_text(r.text)
                return f"[Linked content]\n{text[:4000]}"
    except Exception as e:
        print(f"[DEBUG] fetch_external_content failed for {url}: {e}")
    return ""


@router.post("/post-answer")
async def post_answer(request: PostAnswerRequest):
    import re as _re
    # Find all URLs in the post body
    urls_in_body = _re.findall(r'https?://\S+', request.selftext)

    comments, external_texts = await asyncio.gather(
        fetch_post_comments(request.url),
        asyncio.gather(*[fetch_external_content(u) for u in urls_in_body[:3]]),
    )

    parts = []
    if request.selftext:
        parts.append(request.selftext)
    for ext in external_texts:
        if ext:
            parts.append(ext)
    if comments:
        parts.append("Community responses:\n" + "\n\n".join(comments))
    source_material = "\n\n---\n\n".join(parts) if parts else "(no content available)"
    print(f"[DEBUG] post_answer: title={request.title!r}, selftext_len={len(request.selftext)}, comments={len(comments)}, external_fetched={sum(1 for e in external_texts if e)}, source_len={len(source_material)}")

    openai_messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert on {request.game_name}. "
                f"You will receive raw post content and community replies. "
                f"Your response MUST be built entirely from the provided text — extract every useful piece of advice, opinion, tip, and fact that appears in it. "
                f"Do NOT generate information that is not present in the source material. "
                f"Do NOT write a generic guide about the game. "
                f"If the source mentions specific DLCs, mention those DLCs. If it mentions a specific bug, address that bug. Stay locked to what the text actually says. "
                f"Do NOT reference Reddit, 'the post', 'users', or 'comments'. Present everything as your own expert knowledge. "
                f"{_FORMAT}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Extract and present all useful information from the following community discussion:\n\n"
                f"---\n{source_material}\n---\n\n"
                f"Topic: {request.title}"
            ),
        },
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=1500,
        messages=openai_messages,
    )
    return {"answer": response.choices[0].message.content}
