# Maya Hindi: Automatic Dubbing & Instagram Posting

Fully automatic pipeline that takes the English "talking object" videos of **Maya AI** (@maya.ai.reels), dubs them into
**Hindi with a light Haryanvi touch** in the **original speaker's cloned voice**, and posts them to a separate Hindi
Instagram account: **3 reels a day, at the best times for an Indian audience, with trending Hindi captions**.

Everything runs on **free services**: GitHub Actions, Kaggle (free GPU), Google Drive and the Instagram API. The only
running cost is Gemini, which comes to cents per month.

**Test translation/dubbing on one video in Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maya-ai-hindi-reels-lab/maya-hindi/blob/main/colab/dub_test.ipynb) (see [section 18](#18-the-colab-notebook))

---

## Contents

1. [How it works](#1-how-it-works)
2. [The dubbing pipeline, step by step](#2-the-dubbing-pipeline-step-by-step)
3. [The posting system](#3-the-posting-system)
4. [Accounts and services used](#4-accounts-and-services-used)
5. [Project files](#5-project-files)
6. [Workflows (automations)](#6-workflows-automations)
7. [Setup from scratch](#7-setup-from-scratch)
8. [Secrets and variables reference](#8-secrets-and-variables-reference)
9. [Dubbing settings reference (`dubber/settings.json`)](#9-dubbing-settings-reference-dubbersettingsjson)
10. [Daily operation](#10-daily-operation)
11. [Monitoring and reading the logs](#11-monitoring-and-reading-the-logs)
12. [Common tasks](#12-common-tasks)
13. [Updating the code](#13-updating-the-code)
14. [Maintenance calendar](#14-maintenance-calendar)
15. [Troubleshooting](#15-troubleshooting)
16. [Limits, quotas and costs](#16-limits-quotas-and-costs)
17. [Security](#17-security)
18. [The Colab notebook](#18-the-colab-notebook)
19. [FAQ](#19-faq)

---

## 1. How it works

```
 ┌──────────────────────────┐
 │ English videos           │  Google Drive (English account)
 │ folder 1pMEvp8X…         │  shared READ-ONLY with this project's service account
 └────────────┬─────────────┘
              │ 01:00 IST daily: "dub" workflow (GitHub) starts a Kaggle run
              ▼
 ┌──────────────────────────┐
 │ Kaggle free GPU (T4)     │  Demucs → Whisper → Gemini → OmniVoice → ffmpeg
 │ maya-hindi-dubber        │  same code as the Colab test notebook
 └────────────┬─────────────┘
              │ uploads video + Hindi script (.txt)
              ▼
 ┌──────────────────────────┐
 │ "Maya Hindi dubbed"      │  Google Drive (this project's account), viewable by link
 │ folder                   │
 └────────────┬─────────────┘
              │ every 30 min: "instagram-autopost" workflow (GitHub)
              ▼
 ┌──────────────────────────┐
 │ Hindi Instagram account  │  3 reels/day at India prime time, trending Hindi captions
 └──────────────────────────┘
```

**Key ideas**

- **Buffer:** the dubbing run keeps **6 dubbed videos ready**, dubbing at most **4 per night**. Posting takes 3 a day,
  so a failed night doesn't interrupt posting.
- **Same code everywhere:** the Colab test notebook (`colab/dub_test.ipynb`) downloads and runs `dubber/core.py`
  from this repo, so what you test in Colab is exactly what runs automatically.
- **Never twice:** a video is never dubbed twice (tracked by Drive file ID and content fingerprint) and never posted twice.

---

## 2. The dubbing pipeline, step by step

For each English video, Kaggle runs:

| # | Step | Tool | What happens |
|---|---|---|---|
| 1 | Download | Google Drive API (service account) | The original is downloaded from the English folder |
| 2 | Separate | **Demucs** | Voice and background music are split into two tracks |
| 3 | Transcribe | **Whisper** (`medium`) | Speech is transcribed with a timestamp for every word; invented text over music is filtered out |
| 4 | Split into phrases | Interpreter-style | Speech is cut into phrases at natural pauses (`PAUSE_SPLIT`) and sentence ends |
| 5 | Translate | **Gemini** | Each sentence is translated **completely and exactly**, split into the same phrases as the original |
| 6 | Verify | Gemini (2nd pass) | Each translation is checked against the English; anything missing or wrong is fixed |
| 7 | Safety checks | Local + Gemini | Any English words in Latin letters are rewritten in Devanagari; any dialect words too heavy for the chosen style are replaced with Hindi |
| 8 | Voice | **OmniVoice** | Each phrase is spoken in a **clone of the original speaker's voice**, cloned from their own sentence |
| 9 | Fit timing | see below | Each phrase is fitted into the original's time without overlaps |
| 10 | Mix | **ffmpeg** | Hindi voice + original music (turned down, with ducking) → final MP4, original picture untouched |
| 11 | Upload | Google Drive API (your account) | Video + `.txt` with the Hindi script go to "Maya Hindi dubbed" |

### Translation rules (what Gemini is told)

- **Complete and exact:** every fact, ingredient, body part, quantity, condition, instruction and result is kept. Nothing
  dropped, shortened or added.
- **Natural spoken style:** everyday words people actually say; common English words (lips, nails, perfume, subscribe,
  brand names) stay English but are written in Devanagari (लिप्स, नेल्स).
- **Hindi with a light Haryanvi touch** (default): standard Hindi grammar (मैं…हूँ, मुझे, तुम्हें, है, नहीं); the Haryanvi
  feel comes from a direct, desi, confident tone and at most one common word like घणा or थारा. Heavy dialect is banned
  (मन्ने, तने, सूं, सै, कोनी, ढाळ, काढण…) and a safety check replaces it if it slips through.
- **Confident tone:** "may / might / can / try" become sure statements and direct instructions
  (e.g. *"घनी दिखने लगेंगी"*, not *"शायद घनी दिख सकती हैं"*). Warnings and safety advice keep their caution.
- **Consistent address:** the viewer is addressed the same way throughout.
- **Echo:** laughs, "wow", sounds and interjections keep the **original audio** instead of being translated.

### Timing: how a Hindi line is fitted

Hindi is often longer than English. For each phrase, in this order:

1. **Fits its own time** → starts exactly when the English phrase starts, at natural pace. ✅
2. **Too long** → uses the **silence after** it (until the next phrase starts).
3. **Still too long** → starts a little **earlier, in the silence before** it (up to `MAX_EARLY_START` seconds, never
   overlapping the previous phrase).
4. **Still too long** → Gemini **rephrases it shorter with the same meaning**, using a character budget based on the voice's
   real speaking rate. A second Gemini call checks the meaning, and rewrites that lose anything are rejected. Up to
   3 rounds; dialect words may not be used to save characters.
5. **Last resort** → spoken slightly faster, to exactly fill the available time. ⏩

Phrases never overlap each other.

### Mixing

- Background music at `MUSIC_VOLUME` (0.25 = 25% of original).
- **Ducking:** music dips a further ~18 dB while the voice speaks and comes back in pauses.
- Loudness normalized to Instagram's level (−14 LUFS).
- If a filter isn't available, the mix automatically falls back to a simpler version instead of failing.

---

## 3. The posting system

Runs every 30 minutes on GitHub (`instagram-autopost`). Each run:

1. **Plans the day** (first run each day): 3 posting slots, based on when your followers are online (`online_followers`)
   and which hours performed best for past reels (views, likes, comments, shares, saves). Until there's data it uses
   `DEFAULT_SLOTS` (13:00, 19:00, 21:30 India time). About 1 day in 5 it tests a new hour so it keeps learning.
2. **Posts when a slot is due:** takes the next dubbed video (natural name order, so `(2)` comes before `(10)`).
3. **Writes the caption with Gemini:** it looks at 8 frames of the video plus the Hindi script (`.txt`) and writes a
   **trending-format title** (first line), a short caption ending with a question, and **up to 8 hashtags**, in
   **Hinglish** (Hindi in English letters mixed with English words) by default.
4. **Uses today's trends:** once a day Gemini searches the web for trending Reels **hashtags** and **caption formats**
   in India for this niche. Only trends that genuinely fit the video are used.
5. **Picks the cover:** Gemini picks the most eye-catching frame as the reel cover.
6. **Publishes:** Instagram downloads the video straight from its Drive link (uploaded as-is, no re-encoding).
7. **Learns:** 48 hours after each post, it saves the reel's stats to improve future posting times.

**Duplicate and crash protection**

- Posted videos are recorded by Drive file ID and content fingerprint (MD5) in `state.json`.
- Before publishing, the upload is saved as "in flight". If a run crashes mid-publish, the next run asks Instagram
  whether it went live before trying again.
- Videos Instagram won't accept (wrong format, no audio, over 300 MB) are skipped with the reason logged.
- If the AI is unavailable, `DEFAULT_CAPTION` is used; if that's not set, it waits and retries.

---

## 4. Accounts and services used

All belong to **this project** (separate from the English page), except the read-only English folder.

| Service | Account | Used for | Cost |
|---|---|---|---|
| GitHub | `maya-ai-hindi-reels-lab` | Code, schedules, secrets, posting | Free (private repo, 2,000 min/month) |
| GitHub Pages | `maya-ai-hindi-reels-lab.github.io` (public) | Homepage + privacy policy required by Google | Free |
| Google Cloud project | new Google account | Drive API, service account, OAuth app | Free |
| Google Drive | new Google account | "Maya Hindi dubbed" folder | Free (15 GB) |
| Google Drive | **English** account | English videos folder, **shared read-only** | Free |
| Gemini API (AI Studio) | new Google account | Translation, captions, trends | Cents/month (billing enabled for live web trends) |
| Kaggle | new account (phone-verified) | Free T4 GPU for dubbing | Free (~30 GPU h/week) |
| Meta for Developers | new app | Instagram API access | Free |
| Instagram | Hindi account (Creator) | Publishing | Free |

---

## 5. Project files

```
maya-hindi/
├── .github/workflows/
│   ├── dub.yml               Daily: decide how many to dub, start Kaggle, report failures
│   ├── autopost.yml          "instagram-autopost": every 30 min: plan day, post, collect stats
│   ├── kaggle-secrets.yml    Manual: copy keys into a private Kaggle dataset
│   └── refresh-token.yml     "refresh-ig-token": weekly, renews the Instagram token
├── dubber/                   Everything that runs on Kaggle
│   ├── settings.json         Dubbing settings (same names as the notebook)
│   ├── bootstrap.py          Installs PyTorch 2.8, OmniVoice, Demucs, Whisper, current ffmpeg
│   ├── core.py               Dubbing functions (used by Kaggle and the Colab notebook)
│   ├── worker.py             Picks next videos, dubs them, uploads to Drive
│   ├── build_kernel.py       Assembles the single-file Kaggle script + metadata
│   ├── buffer_count.py       Decides how many to dub (keeps 6 ready)
│   └── kaggle_status.py      Detects a failed previous Kaggle run and prints its log
├── colab/
│   └── dub_test.ipynb        Test translation/dubbing on one video in Colab (uses dubber/core.py)
├── tools/
│   └── setup_drive_upload.py One-time Colab script: creates the dubbed folder + Drive token
├── main.py                   Posting: schedule, pick next video, publish, stats
├── scheduler.py              Chooses daily posting times
├── instagram.py              Instagram API calls
├── drive.py                  Reads Drive folders (recursive, duplicates, order)
├── ai_caption.py             Caption, title, hashtags, cover (Gemini)
├── trends.py                 Daily trending hashtags + caption formats
├── media.py                  Reads video info/frames, checks Instagram requirements
├── config.py                 All posting settings (from GitHub variables)
├── refresh_token.py          Renews the Instagram token
├── requirements.txt
└── state.json                Created automatically: posted videos, stats, schedule. Never delete.
```

---

## 6. Workflows (automations)

Find them in the repo's **Actions** tab.

| Workflow | Runs | What it does | Manual options |
|---|---|---|---|
| **dub** | Daily 19:30 UTC (01:00 IST) | Checks the previous Kaggle run; counts ready videos; starts a Kaggle run to top up to 6 (max 4) | **count**: dub exactly this many now |
| **instagram-autopost** | Every 30 min | Plans the day, posts when a slot is due, saves `state.json` | **post now**: publish the next video immediately |
| **refresh-ig-token** | Mondays 03:00 UTC | Renews `IG_ACCESS_TOKEN` (valid 60 days) and saves it back as a secret | – |
| **kaggle-secrets** | Only manually | Copies `GEMINI_API_KEY`, `GOOGLE_SA_JSON`, `DRIVE_OAUTH_JSON` into the private Kaggle dataset `maya-dub-secrets` | – |

To run any of them manually: **Actions → workflow name → Run workflow → Run workflow**.

---

## 7. Setup from scratch

Use the **new** accounts everywhere, except step B3.

### A. Instagram account and Meta app

1. Create the Instagram account → **Settings → Account type and tools → Switch to professional account → Creator**.
2. **developers.facebook.com → My Apps → Create App** → Instagram use case → "I don't want to connect a business
   portfolio yet".
3. App dashboard → **Instagram → API setup with Instagram login → Add account** → log in with the Hindi account. If a tester
   invite appears: Instagram → Settings → Website permissions → Apps and websites → Tester invites → Accept.
4. **Generate token** → copy → `IG_ACCESS_TOKEN`.
5. Open `https://graph.instagram.com/me?fields=user_id,username&access_token=TOKEN` → copy `user_id` → `IG_USER_ID`.

### B. Google Cloud and Drive

1. **console.cloud.google.com** → **New Project** → **APIs & Services → Library → Google Drive API → Enable**.
2. **Service accounts → Create service account** (no roles) → **Keys → Add key → JSON**. The file contents are
   `GOOGLE_SA_JSON`; note its `client_email`.
3. **In the English Google account:** English videos folder → **Share** → the service-account email → **Viewer**.
4. **Homepage + privacy page** (required by Google): public repo `maya-ai-hindi-reels-lab.github.io` with `index.html`
   and `privacy.html` → **Settings → Pages → Deploy from branch → main → /(root)**.
5. **Google Auth Platform** (OAuth consent screen):
   - **Branding:** app name `Maya dubber`, support email, home page `https://maya-ai-hindi-reels-lab.github.io/`,
     privacy policy `https://maya-ai-hindi-reels-lab.github.io/privacy.html`, authorized domain
     `maya-ai-hindi-reels-lab.github.io`, developer email → **Save**.
   - **Audience:** External → **Publish app** → status **In production** (in Testing, tokens expire after 7 days).
   - **Data Access → Add or remove scopes** → `.../auth/drive.file` → **Update → Save**.
6. **Clients → Create client → Desktop app** → copy **Client ID** and **Client secret**.
7. **colab.research.google.com** (new account) → new notebook → paste `tools/setup_drive_upload.py` → fill
   `CLIENT_ID`, `CLIENT_SECRET`, `SERVICE_ACCOUNT_EMAIL` → run → open link → allow ("Advanced → Go to Maya dubber" if
   warned) → copy the full `http://localhost/?…` address back into Colab. It prints `DUBBED_FOLDER_ID` and
   `DRIVE_OAUTH_JSON`, and creates the **Maya Hindi dubbed** folder (viewable by link, shared with the service account).

### C. Gemini key
**aistudio.google.com → Get API key → Create API key** (on a project with billing for live web trends) → `GEMINI_API_KEY`.

### D. Kaggle
1. Sign up at **kaggle.com** → **Settings → Phone verification** (required for GPU and internet).
2. **Settings → API → Create New Token** → `kaggle.json` → `username` = `KAGGLE_USERNAME`, `key` = `KAGGLE_KEY`.

### E. GitHub repository
1. Create the **private** repo `maya-hindi` (no README).
2. Push token: **Settings → Developer settings → Fine-grained tokens → Generate** → only `maya-hindi` →
   **Contents** + **Workflows**: Read and write.
3. Push the code:
   ```bash
   cd /path/to/maya-hindi
   git init && git add . && git commit -m "init" && git branch -M main
   git remote add origin https://maya-ai-hindi-reels-lab@github.com/maya-ai-hindi-reels-lab/maya-hindi.git
   git push -u origin main        # password = the token
   ```
4. Repo **Settings → Actions → General → Workflow permissions → Read and write** → Save.
5. Second fine-grained token, only `maya-hindi`, **Secrets: Read and write** → `GH_PAT`.

### F. Secrets and variables
Add everything from [section 8](#8-secrets-and-variables-reference).

### G. First run
1. **Actions → kaggle-secrets → Run workflow** → ends with `✅ Private dataset ready`.
2. **Actions → dub → Run workflow**, count `1` → follow the Kaggle link → ends with `✅ Uploaded` and `[Done] 1 dubbed`
   (first run ~10–20 min). Check the video in "Maya Hindi dubbed".
3. **Actions → instagram-autopost → Run workflow**, tick **post now** → log ends with `Posted … -> media …`.

---

## 8. Secrets and variables reference

Repo → **Settings → Secrets and variables → Actions**.

### Secrets (required)

| Name | What it is | Where it comes from |
|---|---|---|
| `IG_USER_ID` | Hindi Instagram account ID | Setup A5 |
| `IG_ACCESS_TOKEN` | Instagram API token (auto-renewed weekly) | Setup A4 |
| `GOOGLE_SA_JSON` | Service account key (reads Drive folders) | Setup B2, whole JSON file |
| `DRIVE_OAUTH_JSON` | Token that lets Kaggle upload to your Drive (`client_id`, `client_secret`, `refresh_token`) | Setup B7 |
| `GEMINI_API_KEY` | Gemini API key | Setup C |
| `KAGGLE_USERNAME` | Kaggle username | Setup D2 |
| `KAGGLE_KEY` | Kaggle API key | Setup D2 |
| `GH_PAT` | GitHub token that saves the renewed Instagram token | Setup E5 |

### Variables (required)

| Name | Value |
|---|---|
| `SOURCE_FOLDER_ID` | English videos folder: `1pMEvp8XsHOS6mJRQFh04Yj3ViSrBK8QY` |
| `DUBBED_FOLDER_ID` | "Maya Hindi dubbed" folder ID (Setup B7). Posting stops with an error if it's missing |
| `DEFAULT_CAPTION` | Hindi caption used only if the AI fails, e.g. `आखिर तक देखो 😂 ऐसे और मज़ेदार वीडियो के लिए फॉलो करो!` |

### Variables (optional: defaults shown)

| Name | Default | Meaning |
|---|---|---|
| `POSTS_PER_DAY` | `3` | Reels per day (max 16) |
| `DEFAULT_SLOTS` | `13:00,19:00,21:30` | Posting times until the system has data |
| `LOCAL_TZ` | `Asia/Kolkata` | Timezone for all times |
| `ACTIVE_START` / `ACTIVE_END` | `8` / `24` | Only post between these hours |
| `MIN_GAP_HOURS` | `3` | Minimum hours between posts (≈ 16 ÷ posts per day) |
| `CAPTION_LANGUAGE` | Hinglish (Hindi in English letters + English words) | Language/style of Instagram title, caption and hashtags (the dubbed audio stays Hindi) |
| `ALWAYS_HASHTAGS` | *(none)* | Tags forced onto every reel (e.g. `mayaai`) |
| `SKIP_FOLDERS` | `posted,done,skip` | Subfolder names ignored by both dubbing and posting |
| `TREND_TOPIC` | Hindi/Haryanvi AI talking-object reels | What trends to research |
| `TREND_REGION` | `India` | Where trends are researched |
| `BUFFER` | `6` | Dubbed videos to keep ready |
| `MAX_PER_RUN` | `4` | Max videos dubbed per night |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Gemini model for captions |

---

## 9. Dubbing settings reference (`dubber/settings.json`)

Same names and meaning as the Colab notebook. Edit on GitHub (file → ✏️ → **Commit changes**); the next dubbing run uses them.

| Setting | Current | Options / meaning |
|---|---|---|
| `TARGET_LANGUAGE_CODE` | `bgc` | `bgc` = Haryanvi options below; `hi` = plain Hindi; also `bn`, `mr`, `gu`, `pa`, `ur`, `ta`, `te`, `kn`, `ml` |
| `HARYANVI_STYLE` | `Hindi with light Haryanvi touch` | `Hindi with light Haryanvi touch` · `Hindi with Haryanvi flavour` · `Full Haryanvi` |
| `TRANSLATION_STYLE` | `Natural spoken` | `Natural spoken` (everyday words) · `Formal` |
| `CONFIDENT_TONE` | `true` | Sure statements and direct instructions; no शायद / हो सकता है |
| `ECHO_TARGET_LANGUAGE` | `true` | Keep parts already spoken in the target language (sounds are always kept) |
| `VIDEO_CONTEXT` | `""` | One line about the videos to guide translation, e.g. `talking objects giving health tips` |
| `WHISPER_MODEL` | `medium` | `small` (faster) · `medium` · `large-v3` (most accurate, slower) |
| `PAUSE_SPLIT` | `0.35` | Pause (s) that splits phrases. Smaller = tighter sync |
| `USE_SILENCE` | `true` | Let long lines use silence before/after them |
| `MAX_EARLY_START` | `1.5` | How early (s) a line may start in the silence before it |
| `FIT_BY_REWRITING` | `true` | Rephrase too-long lines shorter with the same meaning |
| `PACE_TOLERANCE` | `1.08` | How much faster than natural (8%) before rephrasing |
| `MUSIC_VOLUME` | `0.25` | Background music level (1.0 = original) |
| `DUCKING` | `true` | Music dips while the voice speaks |
| `GEMINI_MODEL` | `gemini-3.5-flash` | Model used for translation |

Videos already dubbed are **not** re-dubbed after a settings change. To redo one, see [section 12](#12-common-tasks).

---

## 10. Daily operation

Nothing to do. A normal day:

| Time (IST) | What happens |
|---|---|
| 01:00 | **dub** checks the buffer and starts Kaggle if fewer than 6 videos are ready |
| ~01:05–01:40 | Kaggle dubs 1–4 videos and uploads them to "Maya Hindi dubbed" |
| First run after midnight | **instagram-autopost** plans today's 3 slots |
| ~13:00, ~19:00, ~21:30 | A reel is posted at each slot (times improve as it learns) |
| Monday 08:30 | **refresh-ig-token** renews the Instagram token |

**Your only regular jobs:** keep adding English videos to the English folder, and glance at the Actions tab or
emails for failures.

---

## 11. Monitoring and reading the logs

### Where to look
- **GitHub → Actions:** every run; green ✓ = OK, red ✗ = failed (GitHub also emails you).
- **Kaggle → Your Work → Code → maya-hindi-dubber → Logs:** full dubbing log.
- **`state.json`** in the repo: every posted reel (file, media ID, time, caption, stats after 48 h), skipped videos and why.
- **Drive → "Maya Hindi dubbed":** dubbed videos and scripts.

### Healthy Kaggle log
```
[Setup] Installing PyTorch 2.8 + dubbing tools (a few minutes)...
[Setup] ffmpeg version 7.x-static … | all needed features: yes
[Info] 697 originals, 12 already dubbed, 685 to go. Dubbing 3 now.
===== 200/NEW VIDS (13).mp4 =====
✂️ Round 1: 2 phrase(s) too long even using the silence around them; rephrasing with the same meaning...
[Source (en)   0.0–  1.1s] I'm ginger tea.
[Translation (bgc)] मैं अदरक की चाय हूँ।   ✅ natural pace
  ⏱️ uses 0.1s of the silence after
✅ Uploaded: 200 - NEW VIDS (13).mp4
[Done] 3 dubbed, 0 failed.
```

### Healthy posting log (`Run python main.py`)
```
AI captions: gemini
Today's slots: ['…T13:04:00+05:30', '…T19:12:00+05:30', '…T21:30:00+05:30']
Drive: scanned 1 folder(s), 12 file(s), 6 video(s), 6 pending -> next: 200 - NEW VIDS (7).mp4
Trending tags (web search): #…
Trending caption styles (web search): POV: … | …
Title: …
Posted 200 - NEW VIDS (7).mp4 -> media 1789…
```
Between slots it just says `No slot due.`

### Symbols in the dubbing transcript
| Symbol | Meaning |
|---|---|
| ✅ natural pace | Fits at normal speed |
| ⏱️ | Used silence before/after to fit |
| ✂️ rephrased to fit | Shortened with the same meaning (full version shown) |
| ⏩ N× faster | Couldn't be shortened without losing meaning, so it's spoken faster |
| ✔️ | Corrected by the verification pass |
| [Echo] | Original sound kept (laugh, "wow"…) |

---

## 12. Common tasks

| Task | How |
|---|---|
| **Post the next reel now** | Actions → instagram-autopost → Run workflow → tick **post now** |
| **Dub more now** | Actions → dub → Run workflow → count e.g. `3` |
| **Change posts per day** | Variable `POSTS_PER_DAY` (and `MIN_GAP_HOURS` ≈ 16 ÷ posts). Applies from the next day; to apply today, edit `state.json` → `"plan_date": null` → commit |
| **Change posting times** | Variable `DEFAULT_SLOTS` (used until it has learned) |
| **Change dubbing style / music** | Edit `dubber/settings.json` → commit |
| **Re-dub a video** | Delete it (and its `.txt`) from "Maya Hindi dubbed"; the next run dubs it again |
| **Skip an English video** | Move it into a subfolder named `posted`, `done` or `skip` in the English folder (names set by `SKIP_FOLDERS`) |
| **Stop posting temporarily** | Actions → instagram-autopost → **⋯ → Disable workflow** (same for dub). Enable again later |
| **Change Gemini key / Drive token / service account** | Update the secret, then run **kaggle-secrets** |
| **Test a setting before automating** | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maya-ai-hindi-reels-lab/maya-hindi/blob/main/colab/dub_test.ipynb) → try it on one video ([section 18](#18-the-colab-notebook)) |

---

## 13. Updating the code

Local project folder: `/Users/mac/Desktop/idris/ig-autopost-repo-hindi/maya-hindi-repo` (linked to GitHub).

```bash
cd /Users/mac/Desktop/idris/ig-autopost-repo-hindi/maya-hindi-repo
git pull                                                  # first: get the bot's latest state.json
rsync -a --exclude .git --exclude state.json /path/to/new/maya-hindi/ ./   # copy in the new files (never the old history)
git add -A
git commit -m "Describe the change"
git pull                                                  # always before push (the bot commits state.json)
git push
```

- **Order is always pull → add → commit → pull → push.**
- **Never `git push --force`:** it would delete `state.json` (the posted-video history).
- The password prompt wants a **fine-grained token** (Contents + Workflows), not your GitHub password.
- After changing anything in `dubber/`, the next **dub** run uses it automatically.

---

## 14. Maintenance calendar

| When | What | Action needed |
|---|---|---|
| Weekly | Instagram token renewed automatically | None (needs `GH_PAT` valid) |
| Before `GH_PAT` expires | Token for saving the renewed Instagram token | Create a new one, update the `GH_PAT` secret |
| Every 90 days (or your chosen expiry) | Push token on your Mac | Create a new one when `git push` fails |
| When the English folder runs out | Queue empty | Add more English videos |
| Monthly | GitHub Actions minutes | Settings → Billing → check usage stays under 2,000 |
| If you revoke Google access | Drive token | Redo Setup B7, update `DRIVE_OAUTH_JSON`, run kaggle-secrets |

---

## 15. Troubleshooting

### dub workflow / Kaggle

| Message | Cause | Fix |
|---|---|---|
| `Secret … is missing` (kaggle-secrets) | A secret isn't set or is misspelled | Add it (section 8) |
| `Secret file '…' not found` | Kaggle dataset missing or outdated | Run **kaggle-secrets**, then **dub** |
| GPU / internet not allowed | Kaggle phone not verified | Kaggle → Settings → Phone verification |
| `invalid_grant` | OAuth app in Testing, or access revoked | Publish app (B5), redo B7, update `DRIVE_OAUTH_JSON`, run kaggle-secrets |
| `File not found` for originals | English folder not shared with the service account | Setup B3 |
| `ffmpeg failed: …` | Video tool problem (message shows ffmpeg's reason) | Check the `[Setup] ffmpeg …` line; send the message for help |
| `machine_shape` error on push | Kaggle CLI changed | Delete the `"machine_shape"` line in `dubber/build_kernel.py` |
| Red ✗ at "Previous run failed" | Last night's Kaggle run failed | Open "Check the previous dubbing run" for its log; a new run was still started |
| Kaggle quota exceeded | Weekly GPU hours used up | Wait for the weekly reset; lower `MAX_PER_RUN` |

### instagram-autopost

| Message | Cause | Fix |
|---|---|---|
| `Drive: scanned … 0 file(s)` | Nothing dubbed yet, or wrong `DUBBED_FOLDER_ID` | Run **dub**; check the variable |
| `Queue empty` | All dubbed videos posted | Dubbing catches up at 01:00, or run **dub** manually |
| `Video is not publicly downloadable` | Dubbed folder not "Anyone with the link" | Drive → folder → Share → General access → Anyone with the link → Viewer |
| `Skipped …: no audio track` | Video has no sound | Check the dubbed file |
| `Skipped …: Instagram won't accept it as-is` | Format, codec or size (>300 MB) | Check the original video |
| `AI caption unavailable … Will retry` | Gemini failed and no `DEFAULT_CAPTION` | Check `GEMINI_API_KEY`; set `DEFAULT_CAPTION` |
| `OAuthException` / code 190 | Instagram token expired | Generate a new token (A4), update `IG_ACCESS_TOKEN`; check refresh-ig-token runs |
| `Publishing quota exhausted` | Instagram's 24 h limit | Retries automatically |
| `Could not save the state file` | Push conflict | Re-run; if repeated, check workflow permissions (E4) |

### git on your Mac

| Message | Fix |
|---|---|
| `not a git repository` | You're not in the linked folder; `cd` to the project folder |
| `Invalid username or token` | Use a fine-grained token, not your password; clear the keychain: `printf "protocol=https\nhost=github.com\nusername=maya-ai-hindi-reels-lab\n" \| git credential-osxkeychain erase` |
| `rejected … (fetch first)` | `git pull`, then `git push` |
| `cannot pull with rebase: unstaged changes` | `git add -A && git commit -m "…"` first |
| `refusing to allow … workflow` | Token needs **Workflows: Read and write** |

---

## 16. Limits, quotas and costs

| Item | Limit | This project's usage |
|---|---|---|
| Kaggle GPU | ~30 h/week | ~15–25 min per nightly run |
| GitHub Actions (private) | 2,000 min/month | ~1,500 (posting every 30 min + nightly dub). If close: set the autopost schedule to `0 * * * *` (hourly) |
| Instagram API publishing | 100 posts / 24 h | 3/day |
| Google Drive (new account) | 15 GB | Dubbed videos; delete old ones if it fills up (posting history is kept in `state.json`) |
| Gemini | Pay-as-you-go | Translation + verification + captions + 2 daily trend searches: cents per month; web-search grounding is within the free monthly allowance |

---

## 17. Security

- **Never paste tokens, keys or JSON files into chats, emails or code.** Store them only as GitHub secrets.
- If a secret leaks:
  - **Drive token:** myaccount.google.com/permissions → remove **Maya dubber**; reset the client secret; redo B7; update
    `DRIVE_OAUTH_JSON`; run kaggle-secrets.
  - **GitHub token:** Developer settings → delete it → create a new one.
  - **Gemini key:** AI Studio → delete it → create a new one → update the secret → run kaggle-secrets.
  - **Instagram token:** generate a new one → update `IG_ACCESS_TOKEN`.
- The Kaggle dataset `maya-dub-secrets` is **private**; keep it that way.
- The Drive app only has `drive.file` access: it can see only the files it created, not the rest of your Drive.
- The "Maya Hindi dubbed" folder is viewable by link (Instagram needs this); the link isn't listed anywhere.
- The English folder is shared **read-only** with the service account.

---

## 18. The Colab notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/maya-ai-hindi-reels-lab/maya-hindi/blob/main/colab/dub_test.ipynb)

`colab/dub_test.ipynb` runs the dubbing pipeline on **one video** in Colab. It downloads `dubber/core.py` from this repo,
so it is exactly the code the nightly Kaggle run uses. Use it to:
- **test translation settings** (language, Haryanvi level, style, tone, context) before changing `dubber/settings.json`;
- **preview** the full dub and fix individual lines with `EDITS`;
- **adjust the music** instantly with the remix step.

### One-time setup
1. Click the badge. The repo is private, so Colab asks to **authorize GitHub** the first time: allow it (if the badge
   says the notebook isn't found: Colab → **File → Open notebook → GitHub** → tick **Include private repos** → pick
   `maya-ai-hindi-reels-lab/maya-hindi` → `colab/dub_test.ipynb`).
2. **Runtime → Change runtime type → T4 GPU**.
3. 🔑 **Secrets** (left bar) → add with *Notebook access* on:
   - `GEMINI_API_KEY`: the same Gemini key as the GitHub secret;
   - `GH_TOKEN`: GitHub **Settings → Developer settings → Fine-grained tokens → Generate**, only `maya-hindi`,
     **Contents: Read-only**. (Without it, step 2 asks you to upload `dubber/core.py` and `dubber/settings.json`.)

Secrets stay in your Colab account; they are not saved in the notebook.

### Using it
| Step | What it does |
|---|---|
| 1 · Install | Installs the same versions as Kaggle (~3 min). The session restarts once; that's expected |
| 2 · Get code | Downloads the repo (`BRANCH` = `main`, or a branch you're testing) and reads `settings.json` |
| 3 · Transcribe | Pick a video (upload, Drive/web link, or a Drive path with `MOUNT_DRIVE`) → Demucs + Whisper |
| 4 · Translate | Gemini translation with the form's settings. **Change settings and re-run only this step** to compare |
| 5 · Edits | Optional: replace individual lines by number |
| 6 · Dub | Cloned voice, timing and mix → plays the video (tick `DOWNLOAD` to save it) |
| 7 · Remix | Optional: change only the music level, instantly |
| 8 · Settings | Prints the settings you used: paste into `dubber/settings.json` on GitHub to make them the default |

The form defaults match the current `dubber/settings.json`; if you change that file, update the defaults in the notebook too.

---

## 19. FAQ

**Does it change the English account?** No. It only reads the English videos folder.

**What if I add new English videos?** They're picked up automatically at the next nightly run, in name order.

**Will it re-dub videos after I change settings?** No, only new ones. Delete a dubbed video to redo it.

**Why is the voice slightly accented?** The voice is cloned from the English speaker to keep the character's voice, and
a little of the original accent can carry over.

**Why no lip sync?** Open-source lip-sync models only work on human faces, not talking objects.

**Can I post more than 3 a day?** Yes: set `POSTS_PER_DAY` (and possibly `BUFFER` / `MAX_PER_RUN`). More posts can
lower reach per reel; increase gradually and watch views per reel.

**What happens if Kaggle fails one night?** The 6-video buffer keeps posting going for 2 days, and GitHub emails you.

**Where is the posting history?** `state.json` in the repo. Never delete it or force-push over it.
