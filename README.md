# Maya Hindi: automatic dubbing + Instagram posting (standalone, all free)

A separate project for the Hindi/Haryanvi Instagram account. It runs in its **own GitHub account** with its own
Instagram, Kaggle, Google Cloud and Gemini accounts. The only link to the English page: it **reads** the English
videos from their Drive folder.

```
English videos (Drive, read-only) ─► Kaggle free GPU (daily) ─► "Maya Hindi dubbed" folder (this account's Drive)
                                     dub = same code as the Colab notebook        │ video + Hindi script (.txt)
                                                                                  ▼
                                               Hindi Instagram account: 3 reels/day at India times, Hindi captions
```

- **`dub` workflow** (daily 01:00 IST): starts a Kaggle run that dubs just enough videos to keep **6 ready**
  (max 4 per day), with the settings in `dubber/settings.json`.
- **`instagram-autopost` workflow** (every 30 min): posts the dubbed videos at the best times, with trending
  Hindi captions and hashtags, and never posts a video twice.
- **`refresh-token`** (weekly) keeps the Instagram token alive; **`kaggle-secrets`** (manual) copies keys to Kaggle.

## Setup

### 1. Hindi Instagram account and Meta app
1. Create the Instagram account → Settings → **Switch to professional account → Creator**.
2. developers.facebook.com (log in with the new account's Facebook/Meta login) → **Create App** → Instagram use case
   → skip the business portfolio.
3. **Instagram → API setup with Instagram login → Add account** → log in with the Hindi account → **Generate token**.
4. User ID: open `https://graph.instagram.com/me?fields=user_id,username&access_token=TOKEN`.

### 2. Google Cloud (new account)
1. console.cloud.google.com → **New Project** → **APIs & Services → Library → Google Drive API → Enable**.
2. **Service accounts → Create** → **Keys → Add key → JSON** (download). Note its email (`…@….iam.gserviceaccount.com`).
3. **In the English Drive account**, share the English videos folder with that service-account email as **Viewer**.
4. **OAuth consent screen**: External, app name `Maya dubber`, scope `.../auth/drive.file` → **Publish app**
   (status *In production*; in *Testing* the token expires after 7 days). `drive.file` needs no Google review.
5. **Credentials → Create credentials → OAuth client ID → Desktop app** → copy **Client ID** and **Client secret**.
6. In a Colab notebook (signed in with the new Google account), paste `tools/setup_drive_upload.py` into a cell,
   fill in `CLIENT_ID`, `CLIENT_SECRET`, `SERVICE_ACCOUNT_EMAIL`, run it and follow the prompts.
   It creates the **"Maya Hindi dubbed"** folder and prints `DUBBED_FOLDER_ID` and `DRIVE_OAUTH_JSON`.

### 3. Gemini and Kaggle
1. aistudio.google.com → **Get API key** (on a project with billing enabled, for live trending research).
2. kaggle.com → sign up → **Settings → Phone verification** (needed for GPU) → **API → Create New Token** (`kaggle.json`).

### 4. GitHub (new account)
1. Create a **private** repo, push this project, then **Settings → Actions → General → Workflow permissions →
   Read and write**.
2. Create a fine-grained token (this repo only, **Secrets: read/write**) for `GH_PAT`.

**Secrets**

| Secret | Value |
|---|---|
| `IG_USER_ID` | step 1.4 |
| `IG_ACCESS_TOKEN` | step 1.3 |
| `GOOGLE_SA_JSON` | full contents of the JSON key (step 2.2) |
| `DRIVE_OAUTH_JSON` | printed in step 2.6 |
| `GEMINI_API_KEY` | step 3.1 |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | from `kaggle.json` |
| `GH_PAT` | step 4.2 |

**Variables**

| Variable | Value |
|---|---|
| `SOURCE_FOLDER_ID` | the English videos folder ID (`1pMEvp8XsHOS6mJRQFh04Yj3ViSrBK8QY`) |
| `DUBBED_FOLDER_ID` | printed in step 2.6 |
| `DEFAULT_CAPTION` | Hindi fallback caption (used only if the AI fails) |

Optional (defaults in brackets): `POSTS_PER_DAY` (3), `DEFAULT_SLOTS` (`13:00,19:00,21:30`), `LOCAL_TZ`
(`Asia/Kolkata`), `CAPTION_LANGUAGE` (Hindi with a light Haryanvi touch), `TREND_TOPIC`, `TREND_REGION` (India),
`BUFFER` (6 ready videos), `MAX_PER_RUN` (4), `GEMINI_MODEL`.

### 5. Start
1. **Actions → kaggle-secrets → Run workflow** (keys go into a *private* Kaggle dataset).
2. **Actions → dub → Run workflow** with count `1` → follow the Kaggle link in the log (~10–15 min first time).
   The dubbed video + script appear in "Maya Hindi dubbed".
3. **Actions → instagram-autopost → Run workflow** with **post now** ticked → first reel.

Then it runs by itself.

## Day to day
- **Dubbing settings** (Haryanvi style, music volume, confident tone…): edit `dubber/settings.json` and push.
- **Re-dub a video:** delete it from "Maya Hindi dubbed".
- **Keys changed:** run **kaggle-secrets** again.
- **Failures:** a failed Kaggle run makes the next `dub` run red with the error log; GitHub emails you.

## Limits (all free)
- Kaggle: ~30 GPU hours/week; each daily run uses ~15–25 min.
- GitHub Actions (private repo): 2,000 min/month, and this repo uses ~1,500. If it gets close, change the
  `instagram-autopost` schedule to `0 * * * *` (hourly).
- If `kaggle kernels push` complains about `machine_shape`, delete that line in `dubber/build_kernel.py`.
