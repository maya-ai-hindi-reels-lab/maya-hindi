"""AI packaging: picks the cover frame and writes hook, caption and hashtags.
Providers: Gemini (GEMINI_API_KEY) or Claude (ANTHROPIC_API_KEY). Chosen by AI_PROVIDER,
or automatically: Gemini if its key is set, else Claude, else no AI (fallback)."""
import base64
import json
import re

import requests

import config

PROMPT = """You are an Instagram Reels growth strategist.
The images are {n} frames from ONE video, in time order (Frame 0 to Frame {last}).
{notes}
{trends}
{styles}
{trend_tags}
Return ONLY a JSON object:
{{
  "cover_frame": <index of the most eye-catching frame: sharp, well lit, expressive face or the key moment>,
  "title": "<the caption's first line, max 70 characters: a scroll-stopping hook built on one of today's trending formats, adapted to what happens in this video. Only if none of them can fit, write an original hook in the style of those trends. May include 1 emoji.>",
  "caption": "<1-2 short lines after the title that add context or a punchline, ending with a question or call to action that invites comments>",
  "hashtags": ["<exactly 8 hashtags without #: take as many as possible from today's trending list, but ONLY ones that genuinely match this video's content (an unrelated trending tag sends the reel to the wrong audience); fill the rest with specific tags for what is in the video>"]
}}
Rules: describe only what is actually in this video. Adapt trending formats to this video; never copy another creator's caption word for word. Language: {lang}."""


def hashtag_line(ai_tags=()):
    """ALWAYS_HASHTAGS first, then the AI's tags; no duplicates (case-insensitive), max MAX_HASHTAGS."""
    seen, out = set(), []
    for tag in list(config.ALWAYS_HASHTAGS) + [str(t) for t in ai_tags]:
        clean = tag.strip().lstrip("#").replace(" ", "")
        if clean and clean.lower() not in seen:
            seen.add(clean.lower())
            out.append("#" + clean)
    return " ".join(out[: config.MAX_HASHTAGS])


def provider():
    choice = config.AI_PROVIDER
    if choice == "auto":
        if config.GEMINI_API_KEY:
            return "gemini"
        if config.ANTHROPIC_API_KEY:
            return "claude"
        return None
    return choice if choice in ("gemini", "claude") else None


def enabled():
    return provider() is not None


def _prompt(frames, notes, trending, trend_tags=None, styles=None):
    trend_block = ""
    if trending:
        trend_block = "Top-performing captions in this niche right now (reference only):\n" + \
                      "\n".join(f"- {c}" for c in trending[:12])
    return PROMPT.format(
        n=len(frames), last=len(frames) - 1, lang=config.CAPTION_LANGUAGE,
        notes=f"Creator notes about this video: {notes}" if notes else "",
        trends=trend_block,
        styles=("Caption/title formats trending on Reels today (templates; adapt, don't copy):\n"
                + "\n".join(f"- {x}" for x in styles)) if styles else "",
        trend_tags=("Hashtags trending in this niche today (use only those that match this video): "
                    + ", ".join("#" + t for t in trend_tags)) if trend_tags else "",
    )


def _images(frames):
    for i, (_, path) in enumerate(frames):
        with open(path, "rb") as f:
            yield i, base64.b64encode(f.read()).decode()


def _call_gemini(frames, prompt):
    parts = []
    for i, data in _images(frames):
        parts += [{"text": f"Frame {i}"}, {"inline_data": {"mime_type": "image/jpeg", "data": data}}]
    parts.append({"text": prompt})
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": config.GEMINI_API_KEY, "Content-Type": "application/json"},
        json={"contents": [{"role": "user", "parts": parts}],
              "generationConfig": {"responseMimeType": "application/json", "maxOutputTokens": 4096}},
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Gemini {r.status_code}: {r.text[:500]}")
    candidates = r.json().get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini returned no answer: {r.text[:500]}")
    return "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])
                   if not p.get("thought"))


def _call_claude(frames, prompt):
    content = []
    for i, data in _images(frames):
        content += [{"type": "text", "text": f"Frame {i}"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}]
    content.append({"type": "text", "text": prompt})
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": config.ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": config.CLAUDE_MODEL, "max_tokens": 1000,
              "messages": [{"role": "user", "content": content}]},
        timeout=120,
    )
    if not r.ok:
        raise RuntimeError(f"Claude {r.status_code}: {r.text[:500]}")
    return "".join(b.get("text", "") for b in r.json()["content"] if b["type"] == "text")


def generate(frames, notes="", trending=None, trend_tags=None, styles=None):
    prompt = _prompt(frames, notes, trending, trend_tags, styles)
    text = _call_gemini(frames, prompt) if provider() == "gemini" else _call_claude(frames, prompt)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise RuntimeError(f"AI reply had no JSON: {text[:300]}")
    data = json.loads(match.group(0))

    index = max(0, min(int(data.get("cover_frame", 0)), len(frames) - 1))
    tags = hashtag_line(data.get("hashtags", []))
    title = str(data.get("title") or data.get("hook") or "").strip()
    body = str(data.get("caption", "")).strip()
    if title and body.lower().startswith(title.lower()):
        body = body[len(title):].strip()  # model repeated the title inside the caption
    text = "\n".join(part for part in (title, body) if part)
    return {
        "hook": title[:60],  # on-screen text, only used when PROCESS_VIDEO=true
        "title": title,
        "caption": f"{text}\n\n{tags}".strip()[:2200],
        "cover_time": frames[index][0],
    }


class AIUnavailableError(Exception):
    """AI caption could not be generated and no DEFAULT_CAPTION is set: post later instead."""


def fallback(frames, notes=""):
    """Used when no AI key is set or the AI call fails."""
    text = (config.DEFAULT_CAPTION or notes).strip()  # DEFAULT_CAPTION first; .txt notes only as last resort
    missing = [t for t in config.ALWAYS_HASHTAGS if f"#{t.lower()}" not in text.lower()]
    if missing:
        text = f"{text}\n\n{' '.join('#' + t for t in missing)}".strip()
    return {"hook": None, "caption": text[:2200], "cover_time": frames[len(frames) // 3][0]}
