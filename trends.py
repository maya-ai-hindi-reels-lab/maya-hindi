"""Top-performing captions for your niche hashtags (Hashtag Search API, Facebook Login).
Requires App Review for 'Instagram Public Content Access'. Limit: 30 unique hashtags / 7 days."""
import re
import time
from datetime import datetime

import requests

import config

BASE = f"https://graph.facebook.com/{config.API_VERSION}"
MAX_HASHTAGS = 10  # stays well under the 30-per-week cap


def _get(path, **params):
    params["access_token"] = config.FB_ACCESS_TOKEN
    r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def trending_captions(state):
    if not (config.FB_ACCESS_TOKEN and config.FB_IG_USER_ID and config.NICHE_HASHTAGS):
        return []
    cache = state.setdefault("trends", {})
    if cache.get("fetched_at", 0) > time.time() - 86400:  # refresh once a day
        return cache.get("captions", [])

    ids = cache.setdefault("hashtag_ids", {})
    items = []
    for tag in config.NICHE_HASHTAGS[:MAX_HASHTAGS]:
        try:
            if tag not in ids:
                ids[tag] = _get("ig_hashtag_search", user_id=config.FB_IG_USER_ID, q=tag)["data"][0]["id"]
            media = _get(f"{ids[tag]}/top_media", user_id=config.FB_IG_USER_ID,
                         fields="caption,media_type,like_count,comments_count", limit=50)
            items += [m for m in media.get("data", []) if m.get("media_type") == "VIDEO" and m.get("caption")]
        except (requests.HTTPError, KeyError, IndexError) as e:
            print(f"Trend lookup failed for #{tag}: {e}")

    items.sort(key=lambda m: m.get("like_count", 0) + 3 * m.get("comments_count", 0), reverse=True)
    captions = [m["caption"][:300] for m in items[:15]]
    cache.update(fetched_at=time.time(), captions=captions)
    return captions


# ---------- Trending hashtags via Gemini (+ Google Search grounding) ----------

TAG_PROMPT = """Find hashtags that are trending on Instagram Reels right now ({date}) for audiences in {region}:
both (a) trending in this niche: {topic}, and (b) trending across Reels overall this week
(current events, memes, seasonal moments) that a short funny AI video could plausibly use.
Return ONLY a comma-separated list of 30 hashtags, niche ones first, then overall ones.
No explanations, no numbering. Exclude spammy or banned tags (follow4follow, like4like, f4f, etc.)."""

SPAM = {"follow4follow", "like4like", "f4f", "l4l", "followforfollow", "likeforlike", "followme"}


def _gemini_text(prompt, use_search):
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if use_search:
        body["tools"] = [{"google_search": {}}]
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"},
        json=body, timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:300]}")
    parts = (r.json().get("candidates") or [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts if not p.get("thought"))


def _parse_tags(text):
    tags, seen = [], set()
    for token in re.split(r"[,\n]+", text):
        token = token.strip().strip(".*-•` ").lstrip("#")
        if re.fullmatch(r"[A-Za-z0-9_]{2,40}", token) and token.lower() not in seen | SPAM:
            seen.add(token.lower())
            tags.append(token)
    return tags[:30]


def trending_hashtags(state):
    """Today's trending hashtags for the niche, researched once a day and cached in state.json."""
    if not config.GEMINI_API_KEY:
        return []
    today = datetime.now(config.LOCAL_TZ).date().isoformat()
    cache = state.setdefault("trend_tags", {})
    if cache.get("date") == today and cache.get("tags"):
        return cache["tags"]

    prompt = TAG_PROMPT.format(date=today, topic=config.TREND_TOPIC, region=config.TREND_REGION)
    tags, source = [], None
    if config.TREND_SEARCH:
        try:
            tags, source = _parse_tags(_gemini_text(prompt, use_search=True)), "web search"
        except Exception as e:
            print(f"Trending tags: web search unavailable ({e}); using AI knowledge instead.")
    if not tags:
        try:
            tags, source = _parse_tags(_gemini_text(prompt, use_search=False)), "AI knowledge"
        except Exception as e:
            print(f"Trending tags: failed ({e}); continuing without.")
            return cache.get("tags", [])  # yesterday's list is better than nothing

    cache.update(date=today, tags=tags, source=source)
    print(f"Trending tags ({source}): {' '.join('#' + t for t in tags)}")
    return tags


# ---------- Trending caption / title formats ----------

STYLE_PROMPT = """What caption and title formats are trending on Instagram Reels right now ({date})
for audiences in {region}: both in this niche ({topic}) and across Reels overall this week.
Think of viral hook patterns, meme formats, catchphrases and current moments that top Reels use this week
(for example formats like "POV: ...", "Nobody: ... / Me: ...", "Tell me ... without telling me").
Return ONLY a list of 12 formats, niche ones first, one per line, each starting with "- ", written as a reusable
template with ... for the variable part. No explanations."""


def _parse_lines(text):
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip("*").strip()
        if 5 <= len(line) <= 120 and not line.endswith(":") and line not in out:
            out.append(line)
    return out[:12]


def trending_caption_styles(state):
    """Today's trending caption/title formats, researched once a day and cached in state.json."""
    if not config.GEMINI_API_KEY:
        return []
    today = datetime.now(config.LOCAL_TZ).date().isoformat()
    cache = state.setdefault("trend_styles", {})
    if cache.get("date") == today and cache.get("styles"):
        return cache["styles"]

    prompt = STYLE_PROMPT.format(date=today, topic=config.TREND_TOPIC, region=config.TREND_REGION)
    styles, source = [], None
    if config.TREND_SEARCH:
        try:
            styles, source = _parse_lines(_gemini_text(prompt, use_search=True)), "web search"
        except Exception as e:
            print(f"Trending caption styles: web search unavailable ({e}); using AI knowledge instead.")
    if not styles:
        try:
            styles, source = _parse_lines(_gemini_text(prompt, use_search=False)), "AI knowledge"
        except Exception as e:
            print(f"Trending caption styles: failed ({e}); continuing without.")
            return cache.get("styles", [])

    cache.update(date=today, styles=styles, source=source)
    print(f"Trending caption styles ({source}): " + " | ".join(styles))
    return styles
