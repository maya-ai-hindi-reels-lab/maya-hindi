"""All settings come from environment variables (GitHub Secrets / Variables)."""
import os
from zoneinfo import ZoneInfo


def _env(name, default=""):
    # GitHub passes unset variables as "", so treat empty as "not set"
    value = os.environ.get(name, "").strip()
    return value if value else default


# --- Credentials (secrets) ---
IG_USER_ID = os.environ["IG_USER_ID"]
IG_ACCESS_TOKEN = os.environ["IG_ACCESS_TOKEN"]
GOOGLE_SA_JSON = os.environ["GOOGLE_SA_JSON"]
GEMINI_API_KEY = _env("GEMINI_API_KEY")                # AI captions + cover pick (Gemini)
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY")          # AI captions + cover pick (Claude)
FB_ACCESS_TOKEN = _env("FB_ACCESS_TOKEN")              # optional: trending hashtag research
FB_IG_USER_ID = _env("FB_IG_USER_ID")                  # IG business ID under Facebook Login

# --- Source ---
# The "Maya Hindi dubbed" folder (GitHub variable DUBBED_FOLDER_ID). No default: falling back to the
# English originals folder would post undubbed videos.
DRIVE_FOLDER_ID = _env("DRIVE_FOLDER_ID") or _env("DUBBED_FOLDER_ID")
if not DRIVE_FOLDER_ID:
    raise SystemExit("DUBBED_FOLDER_ID is not set (repo Settings → Secrets and variables → Actions → Variables).")
BACKGROUND_MUSIC_ID = _env("BACKGROUND_MUSIC_ID")      # Drive file used only for videos with no audio
# Subfolders with these names are ignored (put videos you already posted manually here)
SKIP_FOLDERS = {f.strip().lower() for f in _env("SKIP_FOLDERS", "posted,done,skip").split(",") if f.strip()}

# --- Posting behaviour ---
POSTS_PER_DAY = int(_env("POSTS_PER_DAY", "3"))
LOCAL_TZ = ZoneInfo(_env("LOCAL_TZ", "Asia/Kolkata"))
ACTIVE_HOURS = range(int(_env("ACTIVE_START", "8")), int(_env("ACTIVE_END", "24")))
MIN_GAP_HOURS = int(_env("MIN_GAP_HOURS", "3"))
DEFAULT_SLOTS = _env("DEFAULT_SLOTS", "13:00,19:00,21:30")
DEFAULT_CAPTION = _env("DEFAULT_CAPTION").replace("\\n", "\n")  # typed \n also means a new line
EXPLORE_RATE = float(_env("EXPLORE_RATE", "0.2"))
SLOT_GRACE_HOURS = 2
MEASURE_AFTER_HOURS = 48
MAX_RETRIES = 3

# --- Video processing ---
# false = upload the original file untouched (default); true = re-render to 9:16 + hook overlay
PROCESS_VIDEO = _env("PROCESS_VIDEO", "false").lower() == "true"
REQUIRE_AUDIO = _env("REQUIRE_AUDIO", "true").lower() == "true"  # skip videos with no sound track
MAX_FILE_MB = 300  # Instagram Reels upload limit
FIT_MODE = _env("FIT_MODE", "blur")            # blur = fit + blurred fill, crop = centre crop
HOOK_OVERLAY = _env("HOOK_OVERLAY", "intro")   # only used when PROCESS_VIDEO=true
MAX_DURATION = float(_env("MAX_DURATION", "90"))

# --- AI captions ---
AI_PROVIDER = _env("AI_PROVIDER", "auto").lower()   # auto | gemini | claude
GEMINI_MODEL = _env("GEMINI_MODEL", "gemini-3.5-flash")
CLAUDE_MODEL = _env("CLAUDE_MODEL", "claude-sonnet-5")
CAPTION_LANGUAGE = _env("CAPTION_LANGUAGE",
                        "Hinglish - Hindi written in English (Roman) letters mixed with common English words, "
                        "the way young Indians type on Instagram. Casual and catchy, no Devanagari script.")
# Tags added to every reel (before the AI's tags); total capped at MAX_HASHTAGS
ALWAYS_HASHTAGS = [h.strip().lstrip("#") for h in _env("ALWAYS_HASHTAGS").replace(" ", ",").split(",") if h.strip().lstrip("#")]
MAX_HASHTAGS = int(_env("MAX_HASHTAGS", "8"))
# Daily trending-hashtag research via Gemini + Google Search (falls back to model knowledge)
TREND_TOPIC = _env("TREND_TOPIC", "Hindi and Haryanvi AI videos of talking objects, desi funny and tips reels")
TREND_REGION = _env("TREND_REGION", "India")
TREND_SEARCH = _env("TREND_SEARCH", "true").lower() == "true"
NICHE_HASHTAGS = [h.strip().lstrip("#") for h in _env("NICHE_HASHTAGS").split(",") if h.strip()]

# --- Upload method ---
# url       = Instagram downloads the original file from a public Drive link (Instagram Login; default)
# resumable = file is uploaded from the runner (requires Facebook Login token; needed for PROCESS_VIDEO)
UPLOAD_MODE = _env("UPLOAD_MODE", "url")
VIDEO_URL_TEMPLATE = _env("VIDEO_URL_TEMPLATE",
                          "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t")

# --- Instagram API ---
API_VERSION = _env("IG_API_VERSION", "v22.0")
GRAPH_HOST = "https://graph.facebook.com" if UPLOAD_MODE == "resumable" else "https://graph.instagram.com"
RUPLOAD_HOST = "https://rupload.facebook.com/ig-api-upload"
# Meta reports online_followers hours in Pacific time, whatever the audience's country
ONLINE_FOLLOWERS_TZ = ZoneInfo(_env("ONLINE_FOLLOWERS_TZ", "America/Los_Angeles"))

STATE_FILE = _env("STATE_FILE", "state.json")
