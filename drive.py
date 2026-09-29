"""Google Drive access via a service account (root folder must be shared with its email).
Scans the root folder and all subfolders recursively, follows shortcuts, and returns each
file's MD5 checksum so identical videos can be detected even under different names/IDs."""
import json
import re

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

import config

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
FILE_FIELDS = "id,name,mimeType,md5Checksum,size,shortcutDetails"

_svc = None


def _service():
    global _svc
    if _svc is None:
        creds = service_account.Credentials.from_service_account_info(
            json.loads(config.GOOGLE_SA_JSON),
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )
        _svc = build("drive", "v3", credentials=creds, cache_discovery=False)
    return _svc


def _list_children(folder_id):
    files, token = [], None
    while True:
        resp = _service().files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields=f"nextPageToken, files({FILE_FIELDS})",
            pageSize=1000,
            pageToken=token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files += resp.get("files", [])
        token = resp.get("nextPageToken")
        if not token:
            return files


def _resolve_shortcut(f):
    """Return the shortcut's target as a normal file dict (with md5), or None if unreadable."""
    target_id = f.get("shortcutDetails", {}).get("targetId")
    if not target_id:
        return None
    try:
        return _service().files().get(fileId=target_id, fields=FILE_FIELDS,
                                      supportsAllDrives=True).execute()
    except HttpError as e:
        print(f"Drive: cannot open shortcut '{f['name']}': {e}")
        return None


def _walk(root_id):
    """All files under root_id with 'path' (e.g. 'Batch1/001_x.mp4') and 'md5'."""
    results, queue, seen = [], [(root_id, "")], {root_id}
    while queue:
        folder_id, prefix = queue.pop(0)
        try:
            children = _list_children(folder_id)
        except HttpError as e:
            print(f"Drive: cannot read folder '{prefix or 'root'}': {e}")
            continue

        for f in children:
            name = f["name"]
            if f["mimeType"] == SHORTCUT:
                target = _resolve_shortcut(f)
                if not target:
                    continue
                f = target  # keep the shortcut's display name, use the target's data

            if f["mimeType"] == FOLDER:
                if name.strip().lower() in config.SKIP_FOLDERS:
                    print(f"Drive: skipping folder '{prefix}{name}/'")
                elif f["id"] not in seen:
                    seen.add(f["id"])
                    queue.append((f["id"], f"{prefix}{name}/"))
            else:
                results.append({"id": f["id"], "name": name, "mimeType": f["mimeType"],
                                "md5": f.get("md5Checksum"), "path": f"{prefix}{name}"})
    return results, len(seen)


def natural_key(path):
    """Human ordering: 'NEW VIDS (2)' before 'NEW VIDS (10)', and 'x (1).mp4' before 'x (1) copy.mp4'."""
    stem = path.rsplit(".", 1)[0].lower()
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", stem)], path.lower()


def pending_videos(exclude_ids):
    """Unposted videos sorted by full path (name folders/files 01_, 02_... to control order),
    plus caption .txt files keyed by folder + base name."""
    files, folder_count = _walk(config.DRIVE_FOLDER_ID)
    captions = {f["path"].rsplit(".", 1)[0]: f for f in files if f["name"].lower().endswith(".txt")}

    videos = []
    for f in files:
        if f["mimeType"].startswith("video/"):
            f["caption_key"] = f["path"].rsplit(".", 1)[0]
            videos.append(f)
    pending = sorted((v for v in videos if v["id"] not in exclude_ids), key=lambda v: natural_key(v["path"]))

    print(f"Drive: scanned {folder_count} folder(s), {len(files)} file(s), "
          f"{len(videos)} video(s), {len(pending)} pending"
          + (f" -> next: {pending[0]['path']}" if pending else ""))
    if not videos and files:
        print("Drive: non-video files seen: "
              + ", ".join(f"{f['path']} ({f['mimeType']})" for f in files[:15]))
    return pending, captions


def download(file_id, dest):
    request = _service().files().get_media(fileId=file_id, supportsAllDrives=True)
    with open(dest, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return dest


def read_text(file_id):
    raw = _service().files().get_media(fileId=file_id, supportsAllDrives=True).execute()
    return raw.decode("utf-8").strip()
