"""Dubbing core: identical to the functions in video_dubbing_live.ipynb (Colab).
Expects these globals to be set by the caller: client, types, MODEL, LANG_NAMES, TARGET_LANGUAGE_CODE, TARGET_LANGUAGE,
TTS_LANGUAGE, ECHO_TARGET_LANGUAGE, TRANSLATION_STYLE, HARYANVI_STYLE, VIDEO_CONTEXT, CONFIDENT_TONE, WHISPER_MODEL,
PAUSE_SPLIT, USE_SILENCE, MAX_EARLY_START, FIT_BY_REWRITING, PACE_TOLERANCE, MUSIC_VOLUME, DUCKING."""

import re, json, hashlib, shutil, subprocess, os
import numpy as np, soundfile as sf

class ToolError(RuntimeError):
    pass

def sh(*args):
    """Run a command; on failure raise with the tool's own error message (not just 'exit status 1')."""
    r = subprocess.run(list(args), capture_output=True, text=True)
    if r.returncode != 0:
        raise ToolError(f"{args[0]} failed: {(r.stderr or r.stdout).strip()[-1500:]}")
    return r

def ffmpeg(*args):
    sh("ffmpeg", "-y", "-loglevel", "error", *args)

def duration_of(path):
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                          "-of", "csv=p=0", path]).decode().strip())

def fingerprint(path):
    """Short content hash (size + first/last 4 MB): same file name with different content gets a different ID."""
    h = hashlib.md5(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(4 << 20))
        f.seek(max(0, os.path.getsize(path) - (4 << 20)))
        h.update(f.read())
    return h.hexdigest()[:8]

def has_audio(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return bool(out.strip())

def load_source(source: str) -> str:
    """Returns a local video path from an upload, Drive link, URL or Colab path."""
    import requests
    if not source:
        from google.colab import files
        uploaded = files.upload()
        return os.path.abspath(next(iter(uploaded)))
    if os.path.exists(source):
        return source
    m = re.search(r"/d/([\w-]{20,})|[?&]id=([\w-]{20,})", source)
    url = (f"https://drive.usercontent.google.com/download?id={m.group(1) or m.group(2)}&export=download&confirm=t"
           if "drive.google.com" in source and m else source)
    dest = "/content/source_" + hashlib.md5(source.encode()).hexdigest()[:8] + ".mp4"
    if not os.path.exists(dest):
        print(f"[Info] Downloading {source}")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            if "text/html" in r.headers.get("Content-Type", ""):
                raise SystemExit("❌ That link returned a web page, not a video. For Drive, set sharing to 'Anyone with the link'.")
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
    return dest

def separate(work: str, video: str):
    """Demucs: returns (vocals.wav, no_vocals.wav)."""
    vocals, bg = f"{work}/sep/htdemucs/audio/vocals.wav", f"{work}/sep/htdemucs/audio/no_vocals.wav"
    if not os.path.exists(vocals):
        ffmpeg("-i", video, "-vn", "-ac", "2", "-ar", "44100", f"{work}/audio.wav")
        sh("python", "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs", "-o", f"{work}/sep", f"{work}/audio.wav")
    return vocals, bg

SENT_END = re.compile(r"[.!?…。]['\")\]]*$")

def transcribe_phrases(vocals: str):
    """Whisper with word timestamps → phrases split at pauses / sentence ends. Returns (phrases, source_lang)."""
    import whisper, torch
    wm = whisper.load_model(WHISPER_MODEL, device="cuda")
    result = wm.transcribe(vocals, word_timestamps=True)
    del wm; torch.cuda.empty_cache()
    # Skip segments Whisper itself marks as probably not speech (it can invent text over music)
    segments = [s for s in result["segments"]
                if not (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0)]
    words = [w for s in segments for w in s.get("words", []) if w["word"].strip()]
    phrases, cur, sent = [], [], 0

    def close(end_of_sentence):
        nonlocal cur, sent
        if cur:
            phrases.append({"start": cur[0]["start"], "end": cur[-1]["end"], "sent": sent,
                            "src": "".join(w["word"] for w in cur).strip()})
            cur = []
            if end_of_sentence:
                sent += 1

    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            length = cur[-1]["end"] - cur[0]["start"]
            prev_ends_sentence = bool(SENT_END.search(cur[-1]["word"].strip()))
            if prev_ends_sentence:
                close(True)
            elif (gap >= PAUSE_SPLIT and length >= 1.0) or length >= 8.0 or (length >= 5.0 and cur[-1]["word"].strip().endswith(",")):
                close(False)
        cur.append(w)
    close(True)
    return phrases, result.get("language", "en")

def gemini_json(prompt: str, attempts: int = 3):
    """Ask Gemini for a JSON array; retries on empty/blocked/invalid replies."""
    import time
    last = None
    for k in range(attempts):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            m = re.search(r"\[.*\]", resp.text or "", re.S)
            if m:
                return json.loads(m.group(0))
            last = f"no JSON array in reply: {(resp.text or '')[:200]!r}"
        except Exception as e:
            last = e
        time.sleep(2 * (k + 1))
    raise SystemExit(f"❌ Gemini failed after {attempts} tries: {last}")

DEVANAGARI = {"hi", "bgc", "mr"}
PROMPT_VERSION = 6   # bump when translation instructions change, so saved translations are redone

def rules(source_lang: str) -> str:
    src = LANG_NAMES.get(source_lang, source_lang)
    keep_words = (f"Keep everyday {src} words people normally say as-is (brand names, lips, nails, perfume, phone, "
                  "subscribe), written in Devanagari.")
    if TARGET_LANGUAGE_CODE == "bgc" and HARYANVI_STYLE == "Full Haryanvi":
        style = ("Write authentic everyday Haryanvi exactly as people in Haryana speak it, in Devanagari. Use real Haryanvi "
                 "grammar and words (e.g. सै/सैं for है/हैं, थारा/म्हारा, के for क्या, घणा for बहुत, न्यूँ, इब, करै सै, "
                 "जावैगा), not Hindi with a few words changed. Keep it natural and respectful, not exaggerated. " + keep_words)
    elif TARGET_LANGUAGE_CODE == "bgc" and HARYANVI_STYLE == "Hindi with light Haryanvi touch":
        style = ("Write normal everyday spoken HINDI (Devanagari) with standard Hindi grammar, pronouns and verb forms "
                 "(मैं…हूँ, मुझे, तुम्हें/तुझे, को, है, नहीं, करता/करती हूँ). The Haryanvi touch is ONLY a direct, desi, "
                 "confident tone plus at most ONE common Haryanvi word in some sentences (घणा, थारा/म्हारा). "
                 "NEVER use Haryanvi grammar or these words: मन्ने, तने, थमने, सूं, सै, सैं, कोनी, कोन्या, ढाळ, काढण, "
                 "गेल, गेल्या, बरगी, कती, जमा, जे (for अगर), बढ़ण, खाण, दिखण, लगाण, पाणी, बालक, दूँ हूँ, करूँ हूँ, "
                 "or 'ने' in place of 'को'. Use: मुझे, तुम्हें, हूँ, है, नहीं, की तरह, निकालने, के साथ, जैसी, अगर, पानी, बच्चे. "
                 "Example: 'Drink me if you're stressed.' → GOOD: 'टेंशन में हो तो मुझे पियो।' "
                 "TOO HEAVY: 'अगर तू टेंशन में हो, तो मन्ने पी।' "
                 "Never make fun of the dialect. " + keep_words)
    elif TARGET_LANGUAGE_CODE == "bgc":
        style = ("Write HINDI with a Haryanvi flavour (Devanagari). Roughly 80% should be plain everyday Hindi that any "
                 "Hindi speaker in India understands instantly; the Haryanvi feel comes from a direct, confident desi tone "
                 "and a FEW well-known Haryanvi touches: घणा (बहुत), थारा/म्हारा (तुम्हारा/हमारा), इब (अब), के (क्या), "
                 "and at most one Haryanvi verb ending per sentence (सै, जावैगा, करो सो). "
                 "Do NOT use heavy dialect that many Hindi speakers don't know, e.g. गेल/गेल्या, कोन्या, बरगी, कती, जमा, "
                 "जे (for अगर), लाग ज्यांगे, दिखण, लगाण, पाणी, बालक. Use the Hindi words instead (के साथ, नहीं, जैसी, "
                 "अगर, दिखने लगेंगे, पानी, बच्चे). "
                 "Example: 'Your lips will look soft and pink.' → GOOD: 'थारे लिप्स घणे सॉफ्ट और गुलाबी दिखेंगे।' "
                 "TOO HEAVY: 'थारे लिप्स कती मुलायम अर गुलाबी दिखण लाग ज्यांगे।' "
                 "Keep it natural and respectful, never exaggerated or mocking. " + keep_words)
    elif TRANSLATION_STYLE == "Formal":
        style = f"Use correct, proper {TARGET_LANGUAGE} vocabulary; keep only brand names in their original form."
    else:
        style = (f"Use the natural {TARGET_LANGUAGE} a normal native speaker actually speaks: keep everyday {src} words "
                 f"people normally say as-is (brand names, lips, nails, perfume, phone, subscribe), written in {TARGET_LANGUAGE} script.")
    confident_rule = ""
    if CONFIDENT_TONE:
        confident_rule = (
            "- CONFIDENT TONE IN EVERY SENTENCE: make every sentence sound sure and direct, as far as possible. "
            "This is intended and is NOT a change of meaning:\n"
            "  • Hedged results (may, might, can, could, possibly, probably, likely, should help) → sure statements: "
            "'can make them appear fuller' → 'घनी दिखने लगेंगी'; 'it should help' → 'फ़ायदा देगा'.\n"
            "  • Soft instructions (you can use, try applying, you might want to, consider) → direct instructions: "
            "'you can apply it daily' → 'रोज़ लगाओ'; 'try using' → 'इस्तेमाल करो'.\n"
            "  • Fillers that weaken (probably, hopefully, kind of, a bit, I think, maybe) → removed, unless they are a "
            "real quantity ('a little sugar' stays 'थोड़ी चीनी').\n"
            "  • Never use शायद, हो सकता है, सकता/सकती/सकते है, लगता है, may be/maybe words in any language.\n"
            "  Example: 'Putting it on your eyelashes can make them appear fuller, so you might stop needing mascara "
            "after a while' → 'पलकों पर लगाने से वे घनी दिखने लगेंगी, जिससे कुछ समय बाद तुम्हें मस्कारा की ज़रूरत नहीं "
            "पड़ेगी' (NOT '...घनी दिख सकती हैं, जिससे शायद...').\n"
            "  • Keep every fact, ingredient and quantity exactly. Only exception: warnings, side effects and safety "
            "advice keep their original caution ('may cause irritation' stays cautious).\n")
    script_rule = ""
    if TARGET_LANGUAGE_CODE in DEVANAGARI:
        script_rule = ("\n- Write EVERYTHING in Devanagari. Never use Latin letters: English words must be written in "
                       "Devanagari (lips → लिप्स, eyelashes → आइलैशेज़, nails → नेल्स, lip scrub → लिप स्क्रब).")
    return f"""Rules:
- COMPLETE and EXACT: every fact, ingredient, body part, quantity, condition, result and instruction must be present.
  Do NOT drop, shorten, summarise, add or change anything.
- Work like a simultaneous interpreter: translate each SENTENCE as a whole so grammar and meaning are correct, then split
  your translation into the SAME phrases as the original: each phrase must carry the meaning of its own source phrase
  (as closely as {TARGET_LANGUAGE} word order allows), and the phrases of a sentence joined together must read as one
  correct, complete {TARGET_LANGUAGE} sentence.
- Keep each phrase about as long to say as the source phrase (its seconds are given).
- Match the tone of the original. No added slang, jokes or fillers.
{confident_rule}- Address the viewer the same way throughout the whole script (don't mix तुम / आप / थारा forms within a sentence).
- {style}{script_rule}
- Set "keep_original": true (and "text": "") ONLY for phrases that are pure non-speech sounds or interjections
  (laughs, "wow", "hmm", singing) → "keep_reason": "sound", or that are already spoken in {TARGET_LANGUAGE}
  → "keep_reason": "target_language". Otherwise "keep_original": false and "keep_reason": "".
{("Context: " + VIDEO_CONTEXT) if VIDEO_CONTEXT else ""}"""

LATIN = re.compile(r"[A-Za-z]")

# Dialect words not allowed at each Haryanvi level (safety net after translation and after shortening)
HEAVY_HARYANVI = ["कोन्या", "कोनी", "गेल", "गेल्या", "बरगी", "कती", "जमा", "ढाळ", "काढण", "दिखण", "लगाण", "पाणी", "बालक",
                  "बढ़ण", "खाण", "छिलणा"]
BANNED_WORDS = {
    "Hindi with light Haryanvi touch": HEAVY_HARYANVI + ["मन्ने", "तने", "थमने", "सूं", "सै", "सैं", "जावैगा", "जावैंगे",
                                                          "लागैगी", "रहवैगी", "करै", "दूँ हूँ", "करूँ हूँ"],
    "Hindi with Haryanvi flavour": HEAVY_HARYANVI + ["मन्ने", "तने", "थमने"],
}

_PUNCT = re.compile(r"[।,!?.;:()\"'-]")

def banned_in(text):
    words = BANNED_WORDS.get(HARYANVI_STYLE, []) if TARGET_LANGUAGE_CODE == "bgc" else []
    padded = " " + " ".join(_PUNCT.sub(" ", text or "").split()) + " "
    return [w for w in words if " " + w + " " in padded]

def fix_dialect(phrases, idxs):
    """Safety net: replace dialect words that are too heavy for the chosen Haryanvi level (no other changes)."""
    bad = {i: banned_in(phrases[i].get("text")) for i in idxs}
    bad = {i: w for i, w in bad.items() if w}
    if not bad:
        return
    fixed = gemini_json(f"""In each line, replace ONLY the listed dialect words (and any grammar that goes with them)
with normal everyday Hindi, keeping the meaning, tone and everything else unchanged
(e.g. मन्ने → मुझे, तने → तुम्हें, सूं/सै → हूँ/है, कोनी → नहीं, ढाळ → तरह, काढण → निकालने, पाणी → पानी).
Return ONLY a JSON array of {{"i": <same i>, "text": "<line>"}}.
Lines: {json.dumps([{"i": i, "text": phrases[i]["text"], "replace": w} for i, w in bad.items()], ensure_ascii=False)}""")
    for o in fixed:
        i = int(o.get("i", -1))
        if i in bad and o.get("text") and len(banned_in(o["text"])) < len(bad[i]):
            phrases[i]["text"] = o["text"]

def fix_script(phrases, idxs):
    """Safety net: rewrite any Latin-letter words in Devanagari (no other changes)."""
    if TARGET_LANGUAGE_CODE not in DEVANAGARI:
        return
    bad = [i for i in idxs if LATIN.search(phrases[i].get("text") or "")]
    if not bad:
        return
    fixed = gemini_json(f"""In each line, rewrite every word written in Latin letters in Devanagari script, the way it is
pronounced (e.g. lips → लिप्स, lip scrub → लिप स्क्रब, eyelashes → आइलैशेज़). Change NOTHING else.
Return ONLY a JSON array of {{"i": <same i>, "text": "<line>"}}.
Lines: {json.dumps([{"i": i, "text": phrases[i]["text"]} for i in bad], ensure_ascii=False)}""")
    for o in fixed:
        i = int(o.get("i", -1))
        if i in bad and o.get("text") and not LATIN.search(o["text"]):
            phrases[i]["text"] = o["text"]

def translate_phrases(phrases, source_lang):
    items = [{"i": i, "sentence": p["sent"], "seconds": round(p["end"] - p["start"], 1), "source": p["src"]}
             for i, p in enumerate(phrases)]
    first = gemini_json(f"""Translate this video's speech from {LANG_NAMES.get(source_lang, source_lang)} to {TARGET_LANGUAGE} for dubbing.
{rules(source_lang)}
Return ONLY a JSON array of {{"i": <same i>, "text": "<translation>", "keep_original": <true|false>, "keep_reason": "<sound|target_language|empty>"}}, same count and order.
Phrases: {json.dumps(items, ensure_ascii=False)}""")
    out = {int(o["i"]): o for o in first if "i" in o}

    # Gemini occasionally skips a phrase: ask again for just those
    missing = [it for it in items if not (out.get(it["i"], {}).get("text") or out.get(it["i"], {}).get("keep_original"))]
    if missing:
        print(f"[Info] Re-translating {len(missing)} phrase(s) that came back empty...")
        again = gemini_json(f"""Translate these phrases from {LANG_NAMES.get(source_lang, source_lang)} to {TARGET_LANGUAGE} for dubbing.
{rules(source_lang)}
Return ONLY a JSON array of {{"i": <same i>, "text": "<translation>", "keep_original": <true|false>, "keep_reason": "<sound|target_language|empty>"}}.
Whole script for context: {json.dumps([it["source"] for it in items], ensure_ascii=False)}
Phrases: {json.dumps(missing, ensure_ascii=False)}""")
        out.update({int(o["i"]): o for o in again if "i" in o})

    pairs = [dict(items[i], translation=out.get(i, {}).get("text", ""),
                  keep_original=bool(out.get(i, {}).get("keep_original")),
                  keep_reason=out.get(i, {}).get("keep_reason", "")) for i in range(len(items))]
    checked = gemini_json(f"""You are checking a {TARGET_LANGUAGE} dubbing translation against the original, sentence by sentence.
For each phrase: if ANYTHING is missing, added, changed or grammatically wrong (read the phrases of a sentence together),
or it breaks the style/script rules below, return a corrected translation; otherwise return it unchanged.
Do NOT make the style stronger or more dialectal than the rules ask for; if a line is already heavier than the rules
allow, make it lighter.
{rules(source_lang)}
Return ONLY a JSON array of {{"i": <same i>, "text": "<final translation>", "keep_original": <true|false>,
"keep_reason": "<sound|target_language|empty>", "fixed": <true|false>, "note": "<what was fixed, or empty>"}}.
Phrases: {json.dumps(pairs, ensure_ascii=False)}""")
    for o in checked:
        if "i" in o and (o.get("text") or o.get("keep_original")):   # never replace a good line with an empty one
            out[int(o["i"])] = o
    for i, p in enumerate(phrases):
        o = out.get(i, {})
        p["keep_original"] = bool(o.get("keep_original"))
        p["keep_reason"] = (o.get("keep_reason") or "sound") if p["keep_original"] else ""
        p["text"] = "" if p["keep_original"] else str(o.get("text") or "")
        p["note"] = o.get("note", "") if o.get("fixed") else ""
        if not p["text"] and not p["keep_original"]:
            print(f"⚠️ Phrase {i} has no translation; add it with EDITS: {{{i}: \"...\"}}")
    fix_script(phrases, range(len(phrases)))
    fix_dialect(phrases, range(len(phrases)))
    return phrases

_TTS = None
def tts_model():
    global _TTS
    if _TTS is None:
        import torch
        from omnivoice import OmniVoice
        _TTS = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    return _TTS

TRIM = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
        "silenceremove=start_periods=1:start_threshold=-45dB,areverse")
SR = 44100

def voice_samples(work, vocals, phrases):
    """One voice sample per sentence (its own audio, 3–10 s; short ones looped). Returns {sent: (path, text)}."""
    voc, vsr = sf.read(vocals)
    voc = voc.mean(axis=1) if voc.ndim > 1 else voc
    refs = {}
    for sent in sorted({p["sent"] for p in phrases}):
        ps = [p for p in phrases if p["sent"] == sent]
        a, b = ps[0]["start"], min(ps[-1]["end"], ps[0]["start"] + 10)
        clip = voc[int(max(0, a - 0.05) * vsr): int((b + 0.05) * vsr)]
        if len(clip) < int(0.3 * vsr):                    # too short to hold a voice: use the nearest 4 s of speech
            c = int(a * vsr); clip = voc[max(0, c - 2 * vsr): c + 2 * vsr]
        text = " ".join(p["src"] for p in ps if p["end"] <= b + 0.01) or ps[0]["src"]
        reps = int(np.ceil(4 * vsr / len(clip))) if 0 < len(clip) < 3 * vsr else 1
        path = f"{work}/voice_{sent}.wav"; sf.write(path, np.tile(clip, reps), vsr)
        refs[sent] = (path, " ".join([text] * reps))
    return refs, voc, vsr

def speak(text, ref, duration=None):
    tts = tts_model()
    kw = dict(text=text, ref_audio=ref[0], ref_text=ref[1])
    if duration:
        kw["duration"] = float(duration)
    try:
        audio = tts.generate(language_id=TTS_LANGUAGE, **kw)
    except TypeError:
        audio = tts.generate(**kw)
    return np.asarray(audio[0]), 24000

def voice_line(work, i, text, ref, duration=None):
    """Speak one phrase, trim edge silence. Returns (wav path, seconds)."""
    audio, sr = speak(text, ref, duration)
    raw, trim = f"{work}/tts_{i}.wav", f"{work}/trim_{i}.wav"
    sf.write(raw, audio, sr)
    ffmpeg("-i", raw, "-af", TRIM, trim)
    if sf.info(trim).duration < 0.15 <= sf.info(raw).duration:   # trimming removed (almost) everything: keep raw
        shutil.copy(raw, trim)
    return trim, sf.info(trim).duration

def plan_slots(phrases, lengths, total):
    """Decide where each phrase can go. Uses the silence AFTER a phrase first (until the next phrase starts),
    then the silence BEFORE it (after the previous phrase has finished, up to MAX_EARLY_START seconds early).
    Returns {i: (start, earliest, latest_end)}. Phrases that fit start exactly at their original time."""
    slots, prev_end = {}, 0.0
    for i, p in enumerate(phrases):
        if i not in lengths:
            continue
        nxt = phrases[i + 1]["start"] if i + 1 < len(phrases) else total
        if p.get("keep_original"):                       # echoed original audio stays exactly where it was
            slots[i] = (p["start"], p["start"], p["end"])
            prev_end = p["end"]
            continue
        early = MAX_EARLY_START if USE_SILENCE else 0.0
        earliest = max(prev_end + 0.08, p["start"] - early, 0.0)
        earliest = min(earliest, p["start"])             # never forced later than the original start
        latest_end = (nxt - 0.08) if USE_SILENCE else min(nxt - 0.08, p["end"] + 0.05)
        length = lengths[i]
        if p["start"] + length <= latest_end:            # fits using its own time (+ silence after): start on time
            start = p["start"]
        else:                                            # borrow silence before: start as late as possible
            start = max(earliest, latest_end - length)
        slots[i] = (start, earliest, latest_end)
        prev_end = min(start + length, latest_end)
    return slots

def fit_by_rewriting(phrases, too_long, source_lang):
    """Gemini rephrases the too-long phrases shorter with the SAME meaning; a second call checks the meaning.
    Only rewrites that keep the complete meaning are accepted. Returns {i: new_text}."""
    items = []
    for i, (length, window) in too_long.items():
        p = phrases[i]
        cps = len(p["text"]) / max(length, 0.1)                     # this voice's real speaking rate
        sentence = [q for q in phrases if q["sent"] == p["sent"]]
        items.append({"i": i, "source": p["src"], "current": p["text"], "seconds": round(window, 1),
                      "max_chars": max(4, int(cps * window * 0.95)),
                      "whole_source_sentence": " ".join(q["src"] for q in sentence),
                      "other_phrases_of_sentence": [q.get("text", "") for q in sentence if q is not p]})
    rewritten = gemini_json(f"""These {TARGET_LANGUAGE} dubbing phrases are too long to say naturally in the time available.
Rewrite each one so it fits in about max_chars characters and can be spoken naturally in the given seconds,
WITHOUT changing or losing ANY meaning.
Allowed: shorter synonyms, simpler or more compact grammar, removing repeated words and fillers, digits instead of number
words, everyday words people actually say if they are shorter.
NOT allowed: dropping any fact, ingredient, quantity, body part, condition, instruction or result; adding anything;
changing the meaning or the tone; making the language more dialectal to save characters (keep exactly the same
language level as the current line; never swap in dialect words such as सूं, सै, मन्ने, कोनी). The phrases of a sentence must still join into one correct sentence.
{rules(source_lang)}
Return ONLY a JSON array of {{"i": <same i>, "text": "<shorter phrase>"}}.
Phrases: {json.dumps(items, ensure_ascii=False)}""")
    proposals = {int(o["i"]): o.get("text", "").strip() for o in rewritten if o.get("text", "").strip()}
    if not proposals:
        return {}
    check_items = [{"i": i, "source": phrases[i]["src"], "rewritten": t,
                    "whole_source_sentence": " ".join(q["src"] for q in phrases if q["sent"] == phrases[i]["sent"])}
                   for i, t in proposals.items()]
    verdicts = gemini_json(f"""Check each shortened {TARGET_LANGUAGE} dubbing phrase against its source phrase.
Is the COMPLETE meaning preserved: every fact, ingredient, quantity, body part, condition, instruction and result,
with nothing dropped, changed or added? Minor wording differences are fine.{" Confident wording (sure statements and direct instructions instead of may/might/can/try, without शायद / हो सकता है) is intended and counts as preserved meaning." if CONFIDENT_TONE else ""}
Return ONLY a JSON array of {{"i": <same i>, "ok": <true|false>, "missing": "<what is lost, or empty>"}}.
Phrases: {json.dumps(check_items, ensure_ascii=False)}""")
    accepted = {}
    for v in verdicts:
        i = int(v["i"])
        if v.get("ok") and i in proposals:
            accepted[i] = proposals[i]
        elif i in proposals:
            print(f"   ↩️ {i:2d} shorter version rejected (would lose: {v.get('missing') or 'meaning'}); keeping the complete one")
    return accepted

def dub(work, vocals, phrases, total):
    refs, voc, vsr = voice_samples(work, vocals, phrases)
    source_lang = phrases[0].get("lang", "en")
    speaking = [i for i, p in enumerate(phrases) if p.get("text") and not p["keep_original"]]
    # Sounds/interjections always keep the original; speech already in the target language only if ECHO is on
    echo = [i for i, p in enumerate(phrases)
            if p["keep_original"] and (p.get("keep_reason") != "target_language" or ECHO_TARGET_LANGUAGE)]

    # 1) Voice every phrase at natural pace
    clips = {i: voice_line(work, i, phrases[i]["text"], refs[phrases[i]["sent"]]) for i in speaking}
    def lengths():
        out = {i: clips[i][1] for i in speaking}
        out.update({i: phrases[i]["end"] - phrases[i]["start"] for i in echo})
        return out
    def window(slot):
        return max(slot[2] - slot[1], 0.25)

    # 2) Still too long after using the silence before/after? Rephrase shorter, same meaning (up to 3 rounds)
    if FIT_BY_REWRITING:
        for rnd in (1, 2, 3):
            slots = plan_slots(phrases, lengths(), total)
            too_long = {i: (clips[i][1], window(slots[i])) for i in speaking
                        if clips[i][1] > window(slots[i]) * PACE_TOLERANCE and phrases[i].get("note") != "edited by you"}
            if not too_long:
                break
            print(f"✂️ Round {rnd}: {len(too_long)} phrase(s) too long even using the silence around them; "
                  f"rephrasing with the same meaning...")
            accepted = fit_by_rewriting(phrases, too_long, source_lang)
            for i, new_text in accepted.items():
                old = phrases[i]["text"]
                phrases[i].setdefault("full_text", old)
                phrases[i]["text"] = new_text
                fix_script(phrases, [i])
                fix_dialect(phrases, [i])
                new_text = phrases[i]["text"]
                clips[i] = voice_line(work, i, new_text, refs[phrases[i]["sent"]])
                print(f"   {i:2d} {len(old)} → {len(new_text)} chars: {new_text}")
        print()

    # 3) Final placement
    slots = plan_slots(phrases, lengths(), total)
    track = np.zeros(int((total + 5) * SR))
    for i, p in enumerate(phrases):
        label = f"[Source ({source_lang}) {p['start']:5.1f}–{p['end']:5.1f}s] {p['src']}"
        if i in echo:
            a = int(p["start"] * SR)
            seg = voc[int(p["start"] * vsr): int(p["end"] * vsr)]
            tmp = f"{work}/echo_{i}.wav"; sf.write(tmp, seg, vsr)
            ffmpeg("-i", tmp, "-ar", str(SR), "-ac", "1", f"{work}/echo_{i}_r.wav")
            y, _ = sf.read(f"{work}/echo_{i}_r.wav")
            b = min(a + len(y), len(track)); track[a:b] += y[: b - a]
            print(f"{label}\n[Echo] original audio kept\n")
            continue
        if i not in clips:
            why = ("already in the target language (ECHO_TARGET_LANGUAGE is off)" if p["keep_original"]
                   else "no translation; add one via EDITS")
            print(f"{label}\n[Muted] {why}\n")
            continue

        start, earliest, latest_end = slots[i]
        room = max(latest_end - earliest, 0.25)
        path, length = clips[i]
        pace = length / room
        if pace > 1.0:                                   # last resort: OmniVoice re-speaks it to fill all the room
            path, length = voice_line(work, i, p["text"], refs[p["sent"]], duration=room)
            start = earliest
        ffmpeg("-i", path, "-af", f"atempo={min(max(length / room, 1.0), 2.0):.3f}", "-ar", str(SR), "-ac", "1",
               f"{work}/fit_{i}.wav")
        y, _ = sf.read(f"{work}/fit_{i}.wav")
        a = int(start * SR)
        b = min(a + len(y), int(latest_end * SR), len(track))   # never runs into the next phrase
        track[a:b] += y[: b - a]
        p["pace"], p["placed_at"] = pace, start

        if pace <= 1.0:
            status = "✅ natural pace"
        elif pace <= PACE_TOLERANCE:
            status = "✅ natural pace (tiny adjustment)"
        elif p.get("note") == "edited by you":
            status = f"⏩ {pace:.2f}× faster (your edit is longer than the time available)"
        else:
            status = f"⏩ {pace:.2f}× faster (couldn't shorten without losing meaning)"
        used = []
        if start < p["start"] - 0.05:
            used.append(f"starts {p['start'] - start:.1f}s early in the silence before")
        end_at = start + min(length, room)
        if end_at > p["end"] + 0.05:
            used.append(f"uses {end_at - p['end']:.1f}s of the silence after")
        extra = f"\n  ⏱️ {'; '.join(used)}" if used else ""
        if p.get("full_text") and p["full_text"] != p["text"]:
            extra += f"\n  ✂️ rephrased to fit (same meaning). Full version: {p['full_text']}"
        if p.get("note"):
            extra += f"\n  ✔️ {p['note']}"
        print(f"{label}\n[Translation ({TARGET_LANGUAGE_CODE})] {p['text']}   {status}{extra}\n")
    out = f"{work}/dub_voice.wav"
    sf.write(out, np.clip(track[: int(total * SR)], -1, 1), SR)
    return out

def srt_time(t):
    h, rem = divmod(int(t * 1000), 3600000); m, rem = divmod(rem, 60000); s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def write_srt(path, phrases, key):
    """One subtitle per sentence."""
    blocks = []
    for sent in sorted({p["sent"] for p in phrases}):
        ps = [p for p in phrases if p["sent"] == sent]
        text = " ".join((p["src"] if key == "src" or p["keep_original"] else p.get("text", "")) for p in ps).strip()
        if text:
            blocks.append(f"{len(blocks) + 1}\n{srt_time(ps[0]['start'])} --> {srt_time(ps[-1]['end'])}\n{text}\n")
    open(path, "w", encoding="utf-8").write("\n".join(blocks))

def mix(video, voice, bg, out, total):
    """Voice + background music. Music is turned down (MUSIC_VOLUME) and, with DUCKING, dips further while the voice
    speaks. If a filter isn't available in this ffmpeg build, it falls back to simpler mixes instead of failing."""
    ducked = ("[1:a]aformat=channel_layouts=stereo,asplit=2[v][key];"
              f"[2:a]volume={MUSIC_VOLUME}[m];"
              "[m][key]sidechaincompress=threshold=0.02:ratio=6:attack=15:release=350[bg];")
    plain = f"[1:a]aformat=channel_layouts=stereo[v];[2:a]volume={MUSIC_VOLUME}[bg];"
    graphs = ([ducked] if DUCKING else []) + [plain]
    tails = ["[v][bg]amix=inputs=2:normalize=0,loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]",
             "[v][bg]amix=inputs=2:normalize=0,aresample=48000[a]"]
    errors = []
    for graph in graphs:
        for tail in tails:
            common = ["-i", video, "-i", voice, "-i", bg, "-filter_complex", graph + tail, "-map", "0:v", "-map", "[a]",
                      "-c:a", "aac", "-b:a", "192k", "-t", f"{total:.2f}", "-movflags", "+faststart"]
            for vcodec in (["-c:v", "copy"],                      # keep the original picture untouched
                           ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p"]):
                try:
                    ffmpeg(*common, *vcodec, out)
                    if errors:
                        print(f"ℹ️ Mixed with a simpler method (ffmpeg said: {errors[0][-300:]})")
                    return
                except ToolError as e:
                    errors.append(str(e))
    raise ToolError("Could not build the final video. ffmpeg errors:\n" + "\n---\n".join(errors[:3]))

def dub_file(video, work):
    """Headless version of run_translation(): one video in, dubbed MP4 out. Returns (out_path, phrases)."""
    os.makedirs(work, exist_ok=True)
    if not has_audio(video):
        raise RuntimeError("video has no audio track")
    local = f"{work}/input.mp4"
    if not os.path.exists(local):
        shutil.copy(video, local)
    total = duration_of(local)
    vocals, bg = separate(work, local)
    phrases, lang = transcribe_phrases(vocals)
    if not phrases:
        raise RuntimeError("no speech found")
    for p in phrases:
        p["lang"] = lang
    phrases = translate_phrases(phrases, lang)
    voice = dub(work, vocals, phrases, total)
    out = f"{work}/dubbed.mp4"
    mix(local, voice, bg, out, total)
    return out, phrases
