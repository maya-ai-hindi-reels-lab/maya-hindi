"""How many videos Kaggle should dub today: enough to keep BUFFER dubbed-but-not-yet-posted videos ready."""
import json, os, sys
from google.oauth2 import service_account
from googleapiclient.discovery import build

BUFFER = int(os.environ.get("HI_BUFFER") or 6)
MAX_PER_RUN = int(os.environ.get("HI_MAX_PER_RUN") or 4)
manual = (os.environ.get("COUNT_INPUT") or "").strip()
if manual:
    print(f"count={int(manual)}"); sys.exit(0)

creds = service_account.Credentials.from_service_account_info(
    json.loads(os.environ["GOOGLE_SA_JSON"]), scopes=["https://www.googleapis.com/auth/drive.readonly"])
svc = build("drive", "v3", credentials=creds, cache_discovery=False)
ids, token = set(), None
while True:
    resp = svc.files().list(q=f"'{os.environ['HI_DRIVE_FOLDER_ID']}' in parents and trashed=false and mimeType contains 'video/'",
                            fields="nextPageToken, files(id)", pageSize=1000, pageToken=token,
                            supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
    ids |= {f["id"] for f in resp.get("files", [])}
    token = resp.get("nextPageToken")
    if not token:
        break
STATE = os.environ.get("STATE_FILE") or "state.json"
MAX_RETRIES = 3  # same as config.MAX_RETRIES: videos that failed this often are never posted
state = json.load(open(STATE)) if os.path.exists(STATE) else {}
used = {p["file_id"] for p in state.get("posts", [])} | set(state.get("skipped", {}))
used |= {fid for fid, n in state.get("failed", {}).items() if n >= MAX_RETRIES}
ready = len(ids - used)
count = max(0, min(MAX_PER_RUN, BUFFER - ready))
print(f"Ready to post: {ready}, buffer target: {BUFFER} -> dub {count}", file=sys.stderr)
print(f"count={count}")
