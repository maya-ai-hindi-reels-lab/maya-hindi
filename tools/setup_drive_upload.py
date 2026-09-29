# ONE-TIME SETUP — paste this whole file into a Google Colab cell and run it.
# Creates the "Maya Hindi dubbed" folder in YOUR Google Drive and the token that lets Kaggle upload into it.
CLIENT_ID = ""              # Google Cloud → APIs & Services → Credentials → your OAuth client (Desktop app)
CLIENT_SECRET = ""
SERVICE_ACCOUNT_EMAIL = ""  # "client_email" in your service-account JSON (…@….iam.gserviceaccount.com)
FOLDER_NAME = "Maya Hindi dubbed"

import os, json
assert CLIENT_ID and CLIENT_SECRET and SERVICE_ACCOUNT_EMAIL, "Fill in CLIENT_ID, CLIENT_SECRET and SERVICE_ACCOUNT_EMAIL first."
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"   # allows the http://localhost redirect used by desktop apps
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

flow = Flow.from_client_config(
    {"installed": {"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                   "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                   "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": ["http://localhost"]}},
    scopes=["https://www.googleapis.com/auth/drive.file"], redirect_uri="http://localhost")
url, _ = flow.authorization_url(access_type="offline", prompt="consent")
print("1) Open this link, pick your Google account and click Allow")
print("   (if you see 'Google hasn't verified this app': Advanced → Go to … → Continue):\n\n", url, "\n")
print("2) The browser then shows 'This site can't be reached / localhost refused to connect'. That is expected.")
redirect = input("3) Copy the FULL address from the browser's address bar and paste it here: ").strip()
flow.fetch_token(authorization_response=redirect)
creds = flow.credentials
if not creds.refresh_token:
    raise SystemExit("❌ No refresh token returned. Remove the app's access at myaccount.google.com/permissions and run again.")

drive = build("drive", "v3", credentials=creds)
folder = drive.files().create(body={"name": FOLDER_NAME, "mimeType": "application/vnd.google-apps.folder"},
                              fields="id").execute()["id"]
drive.permissions().create(fileId=folder, body={"type": "anyone", "role": "reader"}).execute()   # Instagram can fetch
drive.permissions().create(fileId=folder, sendNotificationEmail=False,
                           body={"type": "user", "role": "reader", "emailAddress": SERVICE_ACCOUNT_EMAIL}).execute()

print("\n✅ Done. Add these to GitHub (Settings → Secrets and variables → Actions):\n")
print("Variable  DUBBED_FOLDER_ID   =", folder)
print("Secret    DRIVE_OAUTH_JSON   =", json.dumps({"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                                                    "refresh_token": creds.refresh_token}))
