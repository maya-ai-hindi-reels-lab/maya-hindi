# ---- Kaggle worker: pick the next originals from Drive, dub them, upload to the Hindi Drive folder ----
import io, json, re, shutil, time, traceback

def main():
    S = PARAMS["settings"]
    g = globals()
    g.update({k: v for k, v in S.items() if k != "GEMINI_MODEL"})
    g["MODEL"] = S.get("GEMINI_MODEL", "gemini-3.5-flash")
    g["LANG_NAMES"] = {"hi": "Hindi", "bgc": "Haryanvi", "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "ta": "Tamil",
                       "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu", "en": "English"}
    code = S["TARGET_LANGUAGE_CODE"]
    target = g["LANG_NAMES"].get(code, code)
    if code == "bgc" and S.get("HARYANVI_STYLE") != "Full Haryanvi":
        target = "Hindi with a Haryanvi style"
    g["TARGET_LANGUAGE"], g["TTS_LANGUAGE"] = target, {"bgc": "hi"}.get(code, code)

    from google import genai
    from google.genai import types as gtypes
    g["client"] = genai.Client(api_key=kaggle_secret("gemini_api_key.txt"))
    g["types"] = gtypes

    from google.oauth2 import service_account
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseUpload, MediaIoBaseDownload
    sa = service_account.Credentials.from_service_account_info(
        json.loads(kaggle_secret("google_sa.json")), scopes=["https://www.googleapis.com/auth/drive.readonly"])
    oauth = json.loads(kaggle_secret("drive_oauth.json"))
    user = Credentials(None, refresh_token=oauth["refresh_token"], client_id=oauth["client_id"],
                       client_secret=oauth["client_secret"], token_uri="https://oauth2.googleapis.com/token",
                       scopes=["https://www.googleapis.com/auth/drive.file"])
    ro = build("drive", "v3", credentials=sa, cache_discovery=False)      # read the English originals
    rw = build("drive", "v3", credentials=user, cache_discovery=False)    # write dubbed videos to your Drive

    originals = list_videos(ro, PARAMS["orig_folder"])
    done_ids, done_md5 = set(), set()
    token = None
    while True:
        resp = rw.files().list(q=f"'{PARAMS['dub_folder']}' in parents and trashed=false", pageSize=1000, pageToken=token,
                               fields="nextPageToken, files(id,name,appProperties)").execute()
        for f in resp.get("files", []):
            ap = f.get("appProperties") or {}
            done_ids.add(ap.get("source_id")); done_md5.add(ap.get("source_md5"))
        token = resp.get("nextPageToken")
        if not token:
            break
    todo = [v for v in originals if v["id"] not in done_ids and (not v.get("md5") or v["md5"] not in done_md5)]
    print(f"[Info] {len(originals)} originals, {len(originals) - len(todo)} already dubbed, "
          f"{len(todo)} to go. Dubbing {min(PARAMS['count'], len(todo))} now.\n", flush=True)

    ok = failed = 0
    for v in todo[: PARAMS["count"]]:
        work = f"/kaggle/working/dub_{v['id']}"
        try:
            os.makedirs(work, exist_ok=True)
            ext = os.path.splitext(v["name"])[1] or ".mp4"
            src = f"{work}/src{ext}"
            with open(src, "wb") as fh:
                dl = MediaIoBaseDownload(fh, ro.files().get_media(fileId=v["id"], supportsAllDrives=True), chunksize=16 << 20)
                done = False
                while not done:
                    _, done = dl.next_chunk()
            print(f"===== {v['path']} =====", flush=True)
            out, phrases = dub_file(src, work)

            base = re.sub(r"\.[^.]+$", "", v["path"]).replace("/", " - ")
            props = {"source_id": v["id"], "source_md5": v.get("md5") or "", "lang": code}
            rw.files().create(body={"name": base + ".mp4", "parents": [PARAMS["dub_folder"]], "appProperties": props},
                              media_body=MediaFileUpload(out, mimetype="video/mp4", resumable=True), fields="id").execute()
            # Notes for the caption AI on the posting side: the dubbed script
            script = " ".join(p.get("text") or "" for p in phrases).strip()
            if script:
                rw.files().create(body={"name": base + ".txt", "parents": [PARAMS["dub_folder"]], "appProperties": props},
                                  media_body=MediaIoBaseUpload(io.BytesIO(script.encode()), mimetype="text/plain"),
                                  fields="id").execute()
            ok += 1
            print(f"✅ Uploaded: {base}.mp4\n", flush=True)
        except Exception:
            failed += 1
            print(f"❌ Failed: {v['path']}\n{traceback.format_exc()}", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    print(f"[Done] {ok} dubbed, {failed} failed.")
    if failed and not ok:
        raise SystemExit(1)

def natural_key(path):
    stem = path.rsplit(".", 1)[0].lower()
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", stem)], path.lower()

def list_videos(svc, root):
    """All videos under root (subfolders included, shortcuts followed, 'posted/done/skip' folders ignored)."""
    FOLDER, SHORTCUT = "application/vnd.google-apps.folder", "application/vnd.google-apps.shortcut"
    fields = "nextPageToken, files(id,name,mimeType,md5Checksum,shortcutDetails)"
    out, queue, seen = [], [(root, "")], {root}
    while queue:
        fid, prefix = queue.pop(0)
        token = None
        while True:
            resp = svc.files().list(q=f"'{fid}' in parents and trashed=false", fields=fields, pageSize=1000,
                                    pageToken=token, supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
            for f in resp.get("files", []):
                name = f["name"]
                if f["mimeType"] == SHORTCUT:
                    tid = (f.get("shortcutDetails") or {}).get("targetId")
                    try:
                        f = svc.files().get(fileId=tid, fields="id,name,mimeType,md5Checksum", supportsAllDrives=True).execute()
                    except Exception:
                        continue
                if f["mimeType"] == FOLDER:
                    if name.strip().lower() not in {"posted", "done", "skip"} and f["id"] not in seen:
                        seen.add(f["id"]); queue.append((f["id"], f"{prefix}{name}/"))
                elif f["mimeType"].startswith("video/"):
                    out.append({"id": f["id"], "name": name, "path": prefix + name, "md5": f.get("md5Checksum")})
            token = resp.get("nextPageToken")
            if not token:
                break
    return sorted(out, key=lambda v: natural_key(v["path"]))

main()
