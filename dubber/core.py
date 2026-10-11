"""Dubbing core, shared by the Kaggle worker and the Colab test notebook (colab/dub_test.ipynb).
Call configure() first: it sets the globals used below (client, types, MODEL, TARGET_LANGUAGE_CODE, TARGET_LANGUAGE,
TTS_LANGUAGE, ECHO_TARGET_LANGUAGE, TRANSLATION_STYLE, HARYANVI_STYLE, VIDEO_CONTEXT, CONFIDENT_TONE, WHISPER_MODEL,
PAUSE_SPLIT, USE_SILENCE, MAX_EARLY_START, FIT_BY_REWRITING, PACE_TOLERANCE, MUSIC_VOLUME, DUCKING, and optionally
VOICE_ENGINE and HINDI_VOICE)."""

import re, json, hashlib, shutil, subprocess, os
import numpy as np, soundfile as sf

LANG_NAMES = {"hi": "Hindi", "bgc": "Haryanvi", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "ta": "Tamil",
              "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "en": "English"}

def configure(settings, gemini_client, gemini_types):
    """Apply dubber/settings.json-style values (and the Gemini client) to this module."""
    g = globals()
    g.update({k: v for k, v in settings.items() if k != "GEMINI_MODEL"})
    g["MODEL"] = settings.get("GEMINI_MODEL", "gemini-3.5-flash")
    code = settings["TARGET_LANGUAGE_CODE"]
    target = LANG_NAMES.get(code, code)
    if code == "bgc" and settings.get("HARYANVI_STYLE") != "Full Haryanvi":
        target = "Hindi with a Haryanvi style"
    g["TARGET_LANGUAGE"], g["TTS_LANGUAGE"] = target, {"bgc": "hi"}.get(code, code)
    g["client"], g["types"] = gemini_client, gemini_types

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
    elif TARGET_LANGUAGE_CODE == "hi" and TRANSLATION_STYLE != "Formal":
        style = ("Write pure, standard everyday spoken HINDI (Devanagari) that any Hindi speaker in India understands "
                 "instantly, with standard grammar, pronouns and verb forms (मैं…हूँ, मुझे, तुम्हें, तुम्हारा, है, नहीं). "
                 "Never use Haryanvi, Rajasthani, Bhojpuri or other dialect words such as घणा, थारा, म्हारा, सै, मन्ने, तने, "
                 "कोनी, इब: use बहुत, तुम्हारा, हमारा, है, मुझे, तुम्हें, नहीं, अब. " + keep_words)
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
{confident_rule}- Address the viewer the same way throughout the whole script (don't mix तुम / आप{' / थारा' if TARGET_LANGUAGE_CODE == 'bgc' else ''} forms within a sentence).
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
# Pure Hindi (TARGET_LANGUAGE_CODE "hi"): no dialect words at all ("जमा" is normal Hindi, so it is allowed)
PURE_HINDI_BANNED = [w for w in BANNED_WORDS["Hindi with light Haryanvi touch"] if w != "जमा"] + [
    "घणा", "घणी", "घणे", "थारा", "थारी", "थारे", "म्हारा", "म्हारी", "म्हारे", "इब"]

_PUNCT = re.compile(r"[।,!?.;:()\"'-]")

def banned_in(text):
    if TARGET_LANGUAGE_CODE == "hi":
        words = PURE_HINDI_BANNED
    else:
        words = BANNED_WORDS.get(HARYANVI_STYLE, []) if TARGET_LANGUAGE_CODE == "bgc" else []
    padded = " " + " ".join(_PUNCT.sub(" ", text or "").split()) + " "
    return [w for w in words if " " + w + " " in padded]

def fix_dialect(phrases, idxs):
    """Safety net: replace dialect words not allowed for the chosen language/style (no other changes)."""
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

def f0_median(y, sr):
    """Median pitch (Hz) of the voiced parts of a signal; 0.0 if no voice is found."""
    import librosa
    if sr != 16000:
        y = librosa.resample(np.asarray(y, dtype=np.float32), orig_sr=sr, target_sr=16000)
    if len(y) < 1600:
        return 0.0
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=800, sr=16000, frame_length=1024)
    return float(np.nanmedian(f0[voiced])) if np.any(voiced) else 0.0

_PITCH = {}
def median_pitch(path, cache=True):
    """f0_median of a wav file (cached per file version unless cache=False)."""
    key = (path, os.path.getmtime(path))
    if not cache or key not in _PITCH:
        y, sr = sf.read(path)
        _PITCH[key] = f0_median(y.mean(axis=1) if y.ndim > 1 else y, sr)
    return _PITCH[key]

def semitones(a, b):
    return 12 * np.log2(a / b) if a > 0 and b > 0 else 0.0

def voice_samples(work, vocals, phrases):
    """One voice sample per sentence, 3–10 s. A short sentence is extended with nearby sentences in a similar pitch
    (most likely the same character) instead of being looped, so the cloned voice stays stable.
    Returns ({sent: (path, text)}, vocals, sample rate)."""
    voc, vsr = sf.read(vocals)
    voc = voc.mean(axis=1) if voc.ndim > 1 else voc
    sents = sorted({p["sent"] for p in phrases})
    span = {}
    for sent in sents:
        ps = [p for p in phrases if p["sent"] == sent]
        a, b = ps[0]["start"], min(ps[-1]["end"], ps[0]["start"] + 10)
        text = " ".join(p["src"] for p in ps if p["end"] <= b + 0.01) or ps[0]["src"]
        clip = voc[int(max(0, a - 0.05) * vsr): int((b + 0.05) * vsr)]
        span[sent] = (a, b, text, clip, f0_median(clip, vsr))
    refs, gap = {}, np.zeros(int(0.15 * vsr))
    for k, sent in enumerate(sents):
        own_pitch, use, length = span[sent][4], [sent], len(span[sent][3]) / vsr
        neighbours = sorted((s for s in sents if s != sent), key=lambda s: abs(sents.index(s) - k))
        for s in neighbours:
            if length >= 3:
                break
            clip, pitch = span[s][3], span[s][4]
            if own_pitch and pitch and abs(semitones(pitch, own_pitch)) <= 3 and length + len(clip) / vsr <= 10:
                use.append(s); length += len(clip) / vsr
        use.sort()
        clip = np.concatenate([x for s in use for x in (span[s][3], gap)][:-1])
        text = " ".join(span[s][2] for s in use)
        if len(clip) < int(0.3 * vsr):                    # too short to hold a voice: use the nearest 4 s of speech
            c = int(span[sent][0] * vsr); clip = voc[max(0, c - 2 * vsr): c + 2 * vsr]
        reps = int(np.ceil(4 * vsr / len(clip))) if 0 < len(clip) < 3 * vsr else 1   # last resort: loop
        path = f"{work}/voice_{sent}.wav"; sf.write(path, np.tile(clip, reps), vsr)
        refs[sent] = (path, " ".join([text] * reps))
    return refs, voc, vsr

def speak_omnivoice(text, ref, duration=None):
    tts = tts_model()
    kw = dict(text=text, ref_audio=ref[0], ref_text=ref[1])
    if duration:
        kw["duration"] = float(duration)
    try:
        audio = tts.generate(language_id=TTS_LANGUAGE, **kw)
    except TypeError:
        audio = tts.generate(**kw)
    return np.asarray(audio[0]), 24000

# ---- Voice engines (VOICE_ENGINE). Cloning from the English speaker carries an English accent, so:
#   omnivoice         : OmniVoice clones the original speaker directly (original voice, English accent)
#   chatterbox        : Chatterbox Multilingual clones the original speaker with cfg_weight=0 (less accent bleed)
#   hindi_then_seedvc : OmniVoice speaks in a native Hindi voice (dubber/voices), then Seed-VC changes only the
#                       timbre to the original speaker (native pronunciation, close to the original voice)

VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(globals().get("__file__", "."))), "voices")
SEEDVC_SHA = "51383efd921027683c89e5348211d93ff12ac2a8"   # pinned Plachtaa/seed-vc commit
SEEDVC_DIR = "/tmp/seed-vc"

_CHATTERBOX = None
def chatterbox_model():
    global _CHATTERBOX
    if _CHATTERBOX is None:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        print("[Info] Loading Chatterbox (first time in a session it downloads ~3 GB; can take a few minutes)...", flush=True)
        _CHATTERBOX =ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
    return _CHATTERBOX

def speak_chatterbox(text, ref):
    """No fixed-duration mode: the last-resort fit in dub() speeds it up with atempo instead."""
    import warnings
    model = chatterbox_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)   # torch sdp_kernel deprecation, printed on every line
        wav = model.generate(text, language_id=TTS_LANGUAGE, audio_prompt_path=ref[0], cfg_weight=0.0)
    return wav.squeeze(0).detach().cpu().numpy(), model.sr

_SEEDVC = None
def seedvc_model():
    global _SEEDVC
    if _SEEDVC is None:
        import sys, tarfile, urllib.request, torch, yaml
        print("[Info] Loading Seed-VC (first time in a session it downloads ~2 GB of models; can take a few minutes)...",
              flush=True)
        if not os.path.exists(f"{SEEDVC_DIR}/configs/v2/vc_wrapper.yaml"):
            urllib.request.urlretrieve(f"https://codeload.github.com/Plachtaa/seed-vc/tar.gz/{SEEDVC_SHA}", "/tmp/seed-vc.tgz")
            with tarfile.open("/tmp/seed-vc.tgz") as t:
                t.extractall("/tmp")
            shutil.rmtree(SEEDVC_DIR, ignore_errors=True)
            os.rename(f"/tmp/seed-vc-{SEEDVC_SHA}", SEEDVC_DIR)
        # huggingface_hub 1.x no longer passes/accepts proxies and resume_download; Seed-VC's BigVGAN still requires them
        bigvgan = f"{SEEDVC_DIR}/modules/bigvgan/bigvgan.py"
        src = open(bigvgan).read()
        patched = (src.replace("proxies: Optional[Dict],", "proxies: Optional[Dict] = None,")
                      .replace("resume_download: bool,", "resume_download: bool = False,")
                      .replace("proxies=proxies,", "").replace("resume_download=resume_download,", ""))
        if patched != src:
            open(bigvgan, "w").write(patched)
        if SEEDVC_DIR not in sys.path:
            sys.path.insert(0, SEEDVC_DIR)
        from hydra.utils import instantiate
        from omegaconf import DictConfig
        cwd = os.getcwd()
        os.chdir(SEEDVC_DIR)                             # Seed-VC downloads its checkpoints into ./checkpoints
        try:
            model = instantiate(DictConfig(yaml.safe_load(open("configs/v2/vc_wrapper.yaml"))))
            model.load_checkpoints()
        finally:
            os.chdir(cwd)
        model.to(torch.device("cuda")).eval()
        model.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=torch.float16, device=torch.device("cuda"))
        _SEEDVC = model
    return _SEEDVC

def seedvc_convert(source_wav, target_wav):
    """Timbre-only conversion (convert_style=False): keeps the source's words, timing and pronunciation."""
    import torch
    full = None
    for _, out in seedvc_model().convert_voice_with_streaming(
            source_audio_path=source_wav, target_audio_path=target_wav, diffusion_steps=30, length_adjust=1.0,
            intelligebility_cfg_rate=0.7, similarity_cfg_rate=0.7, top_p=0.9, temperature=1.0, repetition_penalty=1.0,
            convert_style=False, anonymization_only=False, device=torch.device("cuda"), dtype=torch.float16,
            stream_output=True):
        full = out if out is not None else full
    if full is None:
        raise ToolError("Seed-VC returned no audio")
    sr, audio = full
    return np.asarray(audio, dtype=np.float32), sr

_HINDI_REF = {}
def hindi_reference(ref):
    """Native Hindi voice (path, text) for the original speaker: HINDI_VOICE male/female, or auto by pitch."""
    choice = globals().get("HINDI_VOICE", "auto")
    if choice == "auto":
        if ref[0] not in _HINDI_REF:
            pitch = median_pitch(ref[0])
            _HINDI_REF[ref[0]] ="female" if pitch >= 165 else "male"
            print(f"   🎙️ {os.path.basename(ref[0])}: median pitch {pitch:.0f} Hz → {_HINDI_REF[ref[0]]} Hindi voice")
        choice = _HINDI_REF[ref[0]]
    voices = json.load(open(f"{VOICES_DIR}/voices.json", encoding="utf-8"))
    return f"{VOICES_DIR}/{voices[choice]['file']}", voices[choice]["text"]

def speak_hindi_then_seedvc(text, ref, duration=None):
    audio, sr = speak_omnivoice(text, hindi_reference(ref), duration)
    tmp = ref[0].rsplit(".", 1)[0] + "_native_tmp.wav"
    sf.write(tmp, audio, sr)
    return seedvc_convert(tmp, ref[0])

def speak(text, ref, duration=None):
    engine = globals().get("VOICE_ENGINE", "omnivoice")
    if engine == "chatterbox":
        return speak_chatterbox(text, ref)
    if engine == "hindi_then_seedvc":
        return speak_hindi_then_seedvc(text, ref, duration)
    return speak_omnivoice(text, ref, duration)

def voice_line(work, i, text, ref, duration=None):
    """Speak one phrase, trim edge silence. Returns (wav path, seconds).
    PITCH_GUARD: if the line's pitch is more than PITCH_TOLERANCE semitones away from the original speaker (the voice
    'jumped'), it is spoken again (up to 2 more times) and the take closest to the original pitch is kept."""
    guard = globals().get("PITCH_GUARD", True)
    tolerance = float(globals().get("PITCH_TOLERANCE", 4.0))
    target = median_pitch(ref[0]) if guard else 0.0
    takes = []
    for attempt in range(3 if guard and target else 1):
        if attempt:
            import torch
            torch.manual_seed(1000 * attempt + i)          # a different take
        audio, sr = speak(text, ref, duration)
        raw, trim = f"{work}/tts_{i}_{attempt}.wav", f"{work}/trim_{i}_{attempt}.wav"
        sf.write(raw, audio, sr)
        ffmpeg("-i", raw, "-af", TRIM, trim)
        if sf.info(trim).duration < 0.15 <= sf.info(raw).duration:   # trimming removed (almost) everything: keep raw
            shutil.copy(raw, trim)
        off = semitones(median_pitch(trim, cache=False), target) if target else 0.0
        takes.append((abs(off), attempt, trim))
        if abs(off) <= tolerance:
            break
        print(f"   🎚️ line {i}: voice {off:+.1f} semitones off the original, speaking it again...")
    off, attempt, best = min(takes)
    if len(takes) > 1:
        print(f"   🎚️ line {i}: kept take {attempt + 1} ({off:.1f} semitones off)")
    final = f"{work}/trim_{i}.wav"
    shutil.copy(best, final)
    return final, sf.info(final).duration

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

def very_words():
    """Example words for "very" in prompts: the Haryanvi घणा only when a Haryanvi style is chosen."""
    return "very/बहुत/घणा" if TARGET_LANGUAGE_CODE == "bgc" else "very/बहुत"

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
language level as the current line; never swap in dialect words such as सूं, सै, मन्ने, कोनी); changing how the viewer
is addressed or the verb form (पियो stays पियो, never पी; तुम्हें never becomes तुझे); dropping words that sharpen the
meaning (suddenly/अचानक, {very_words()}, instantly/तुरंत, numbers). The phrases of a sentence must still join into one correct sentence.
{rules(source_lang)}
Return ONLY a JSON array of {{"i": <same i>, "text": "<shorter phrase>"}}.
Phrases: {json.dumps(items, ensure_ascii=False)}""")
    proposals = {int(o["i"]): str(o.get("text") or "").strip() for o in rewritten
                 if "i" in o and str(o.get("text") or "").strip()}
    if not proposals:
        return {}
    check_items = [{"i": i, "source": phrases[i]["src"], "rewritten": t,
                    "whole_source_sentence": " ".join(q["src"] for q in phrases if q["sent"] == phrases[i]["sent"])}
                   for i, t in proposals.items()]
    verdicts = gemini_json(f"""Check each shortened {TARGET_LANGUAGE} dubbing phrase against its source phrase.
Is the COMPLETE meaning preserved: every fact, ingredient, quantity, body part, condition, instruction and result,
with nothing dropped, changed or added? Minor wording differences are fine. Answer ok=false if the form of address or
verb form changed (e.g. पियो → पी, तुम्हें → तुझे) or a word that sharpens the meaning was dropped (suddenly/अचानक,
{very_words()}, instantly/तुरंत, numbers).{" Confident wording (sure statements and direct instructions instead of may/might/can/try, without शायद / हो सकता है) is intended and counts as preserved meaning." if CONFIDENT_TONE else ""}
Return ONLY a JSON array of {{"i": <same i>, "ok": <true|false>, "missing": "<what is lost, or empty>"}}.
Phrases: {json.dumps(check_items, ensure_ascii=False)}""")
    accepted = {}
    for v in verdicts:
        if "i" not in v:
            continue
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
        p["pace"], p["placed_at"], p["spoken_end"] = pace, start, b / SR

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

# ---- Remix: make the reel look like a new video (full-screen picture, Hindi captions, hook) ----
# Bought bundle clips sit in a template (black frame + header text) that many pages post; Instagram limits reach for
# such duplicates. REMIX_VIDEO crops back to the moving picture, fills 9:16, burns in the dubbed captions and a hook.

FONT_URL = "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSansDevanagari/hinted/ttf/NotoSansDevanagari-Bold.ttf"
FONT_DIR = "/tmp/maya-fonts"
FONT_NAME = "Noto Sans Devanagari"

def video_size(path):
    w, h = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                    "stream=width,height", "-of", "csv=p=0", path]).decode().strip().split(",")[:2]
    return int(w), int(h)

def detect_content_box(video, samples=24):
    """The moving picture inside a static template (black bars, header text): (x, y, w, h) in source pixels.
    None when there is no frame to remove or the detection isn't confident (then nothing is cropped)."""
    W, H = video_size(video)
    sw = 180; sh = max(2, int(round(H * sw / W / 2)) * 2)
    total, frames = duration_of(video), []
    for k in range(samples):
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{total * (k + 0.5) / samples:.2f}", "-i", video,
                              "-frames:v", "1", "-vf", f"scale={sw}:{sh},format=gray", "-f", "rawvideo", "-"],
                             capture_output=True).stdout
        if len(raw) == sw * sh:
            frames.append(np.frombuffer(raw, np.uint8).reshape(sh, sw).astype(np.float32))
    if len(frames) < 8:
        return None
    std = np.stack(frames).std(axis=0)                  # how much each pixel changes over time
    rows, cols = std.mean(axis=1), std.mean(axis=0)
    if np.percentile(std, 90) < 2:                      # nearly static video: can't tell frame from picture
        return None
    r = np.where(rows > 0.2 * np.percentile(rows, 90))[0]
    c = np.where(cols > 0.2 * np.percentile(cols, 90))[0]
    if not len(r) or not len(c):
        return None
    y0, y1, x0, x1 = r[0], r[-1] + 1, c[0], c[-1] + 1
    fw, fh = (x1 - x0) / sw, (y1 - y0) / sh
    if fw > 0.97 and fh > 0.97:                         # no template around the picture
        return None
    mask = np.ones_like(std, bool); mask[y0:y1, x0:x1] = False
    if fw < 0.4 or fh < 0.3 or std[mask].mean() > 0.35 * std[~mask].mean():
        return None                                     # not a clear static frame around a moving picture
    # Only a real template counts: it takes up a good part of the frame and is mostly dark (black bars around the
    # picture). A full-screen video whose edges just happen to be still is left alone.
    still = np.median(np.stack(frames), axis=0)
    if fw * fh > 0.85 or (still[mask] < 40).mean() < 0.5:
        return None
    # Parts of the picture that never move (a still background) look like frame: grow each side outward until the
    # next line is mostly dark, i.e. the template's black border.
    def dark(line):
        return (line < 40).mean() >= 0.5
    while y0 > 0 and not dark(still[y0 - 1, x0:x1]): y0 -= 1
    while y1 < sh and not dark(still[y1, x0:x1]): y1 += 1
    while x0 > 0 and not dark(still[y0:y1, x0 - 1]): x0 -= 1
    while x1 < sw and not dark(still[y0:y1, x1]): x1 += 1
    sx, sy = W / sw, H / sh
    x, y = int(x0 * sx) + 2, int(y0 * sy) + 2           # 2 px inside, away from the frame's edge line
    w, h = int((x1 - x0) * sx) - 4, int((y1 - y0) * sy) - 4
    return x // 2 * 2, y // 2 * 2, w // 2 * 2, h // 2 * 2

def fill_916(label_in, label_out, w, h):
    """Filter that fills 1080x1920: crop to fill when the shape is close, else fit over a blurred copy."""
    if 0.8 <= (w / h) / (9 / 16) <= 1.25:
        return f"{label_in}scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1{label_out}"
    return (f"{label_in}split=2[bg][fg];[bg]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
            f"boxblur=30:5[bgb];[fg]scale=1080:1920:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1{label_out}")

def caption_font():
    """Downloads the Devanagari font once; None if captions can't be rendered here."""
    path = f"{FONT_DIR}/NotoSansDevanagari-Bold.ttf"
    if not os.path.exists(path):
        try:
            import urllib.request
            os.makedirs(FONT_DIR, exist_ok=True)
            urllib.request.urlretrieve(FONT_URL, path)
        except Exception as e:
            print(f"⚠️ Caption font download failed ({e}); no captions.")
            return None
    filters = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return FONT_DIR if " subtitles " in filters else None

_SCREEN_TEXT = {}
def screen_texts(phrases):
    """On-screen texts written from the dubbed script (cached per script): a hook for the first second and a closing
    line for the end (a question that invites comments or a reason to save). Returns {"hook": ..., "ending": ...}."""
    script = " ".join(p.get("text") or "" for p in phrases).strip()
    if not script:
        return {"hook": "", "ending": ""}
    if script not in _SCREEN_TEXT:
        script_rule = " in Devanagari script" if TARGET_LANGUAGE_CODE in DEVANAGARI else ""
        try:
            out = gemini_json(f"""Write two short on-screen texts for this {TARGET_LANGUAGE} reel about talking objects, in
{TARGET_LANGUAGE}{script_rule}, no emoji, no hashtags, saying only what the script supports:
- "hook": shown in the first second, at most 24 characters, spoken by the object: make viewers curious or feel something
  (a warning, a surprising truth).
- "ending": shown in the last 2 seconds, at most 40 characters: a question that makes viewers comment (e.g. which one
  they need most) or a reason to save the reel.
Return ONLY a JSON array: [{{"hook": "<hook>", "ending": "<ending>"}}]
Script: {script}""")
            item = (out or [{}])[0]
            texts = {"hook": str(item.get("hook", "")).strip(), "ending": str(item.get("ending", "")).strip()}
        except (Exception, SystemExit) as e:
            print(f"⚠️ No on-screen texts ({e})")
            texts = {"hook": "", "ending": ""}
        for key, limit in (("hook", 40), ("ending", 60)):
            if banned_in(texts[key]) or len(texts[key]) > limit:
                texts[key] = ""
        _SCREEN_TEXT[script] = texts
    return _SCREEN_TEXT[script]

def ass_time(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"

def write_ass(path, phrases, hook, total, ending=""):
    """Bold captions for each dubbed line (timed to the Hindi voice), hook at the top for the first ~1.8 s,
    closing question in the middle for the last ~2.5 s."""
    def clean(text):
        return (text or "").replace("{", "(").replace("}", ")").replace("\n", " ").strip()
    events = []
    if globals().get("BURN_CAPTIONS", True):
        lines = sorted((p for p in phrases if p.get("text") and not p.get("keep_original") and p.get("spoken_end")),
                       key=lambda p: p["placed_at"])
        for k, p in enumerate(lines):
            end = min(p["spoken_end"] + 0.15, total)
            if k + 1 < len(lines):
                end = min(end, lines[k + 1]["placed_at"] - 0.02)   # never on screen together with the next line
            events.append(f"Dialogue: 0,{ass_time(p['placed_at'])},{ass_time(max(end, p['placed_at'] + 0.3))},"
                          f"Caption,,0,0,0,,{clean(p['text'])}")
    if hook:
        events.append(f"Dialogue: 1,{ass_time(0)},{ass_time(min(1.8, total))},Hook,,0,0,0,,{clean(hook)}")
    if ending and total > 6:
        events.append(f"Dialogue: 1,{ass_time(total - 2.5)},{ass_time(total)},End,,0,0,0,,{clean(ending)}")
    if not events:
        return None
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,{FONT_NAME},70,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,6,2,2,90,90,620,1
Style: Hook,{FONT_NAME},88,&H0000E5FF,&H0000E5FF,&H50000000,&H50000000,-1,0,0,0,100,100,0,0,3,20,0,8,80,80,280,1
Style: End,{FONT_NAME},76,&H00FFFFFF,&H00FFFFFF,&H50000000,&H50000000,-1,0,0,0,100,100,0,0,3,20,0,5,90,90,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""" + "\n".join(events) + "\n")
    return path

def remix_video(src, out, work, phrases, total):
    """Crop the template away, fill 9:16, burn captions + hook. Audio is copied unchanged."""
    W, H = video_size(src)
    box = detect_content_box(src) if globals().get("CROP_FRAME", True) else None
    x, y, w, h = box or (0, 0, W, H)
    if box:
        print(f"🖼️ Template frame removed: keeping {w}x{h} at ({x},{y}) of {W}x{H}")
    reshape = bool(box) or abs(W / H - 9 / 16) > 0.01
    fonts, ass = caption_font(), None
    if fonts:
        texts = screen_texts(phrases) if (globals().get("HOOK_TEXT", True) or globals().get("END_TEXT", True)) else {}
        hook = texts.get("hook", "") if globals().get("HOOK_TEXT", True) else ""
        ending = texts.get("ending", "") if globals().get("END_TEXT", True) else ""
        if hook:
            print(f"🪝 Hook: {hook}")
        if ending:
            print(f"🔚 Ending: {ending}")
        ass = write_ass(f"{work}/captions.ass", phrases, hook, total, ending)
    if not reshape and not ass:
        shutil.copy(src, out)
        return
    crop = f"crop={w}:{h}:{x}:{y}," if box else ""
    base = fill_916(f"[0:v]{crop}", "[base]", w, h) if reshape else "[0:v]null[base]"
    encode = ["-map", "0:a?", "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
              "-c:a", "copy", "-movflags", "+faststart"]
    if ass:
        try:
            ffmpeg("-i", src, "-filter_complex", f"{base};[base]subtitles=filename={ass}:fontsdir={fonts}[v]",
                   "-map", "[v]", *encode, out)
            return
        except ToolError as e:
            print(f"⚠️ Captions couldn't be burned in ({str(e)[-300:]}); keeping the crop only.")
    ffmpeg("-i", src, "-filter_complex", base, "-map", "[base]", *encode, out)

def finish_video(work, video, voice, bg, phrases, total, out):
    """Mix the dubbed voice with the music, then (REMIX_VIDEO) crop + captions + hook. Never fails because of the remix."""
    if not globals().get("REMIX_VIDEO", True):
        mix(video, voice, bg, out, total)
        return out
    mixed = f"{work}/mixed.mp4"
    mix(video, voice, bg, mixed, total)
    try:
        remix_video(mixed, out, work, phrases, total)
    except Exception as e:
        print(f"⚠️ Remix failed, using the dubbed video without crop/captions: {e}")
        shutil.copy(mixed, out)
    return out

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
    finish_video(work, local, voice, bg, phrases, total, out)
    return out, phrases
