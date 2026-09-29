"""Instagram API: publish reels (in two phases so crashes can't cause double posts),
read quota, audience activity and reel insights."""
import os
import time
from collections import defaultdict

import requests

import config

BASE = f"{config.GRAPH_HOST}/{config.API_VERSION}"


def _get(path, **params):
    params["access_token"] = config.IG_ACCESS_TOKEN
    r = requests.get(f"{BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _post(path, **data):
    data["access_token"] = config.IG_ACCESS_TOKEN
    r = requests.post(f"{BASE}/{path}", data=data, timeout=60)
    if not r.ok:
        raise RuntimeError(f"POST {path} failed: {r.status_code} {r.text}")
    return r.json()


def publishing_quota_left():
    data = _get(f"{config.IG_USER_ID}/content_publishing_limit", fields="config,quota_usage")["data"][0]
    return data["config"]["quota_total"] - data["quota_usage"]


# ---------- Phase 1: container + upload + processing ----------

def create_container(caption, thumb_offset_ms=None, video_url=None):
    """video_url set -> Instagram fetches the file itself; otherwise a resumable session is opened."""
    params = dict(media_type="REELS", caption=caption, share_to_feed="true")
    if video_url:
        params["video_url"] = video_url
    else:
        params["upload_type"] = "resumable"
    if thumb_offset_ms is not None:
        params["thumb_offset"] = str(int(thumb_offset_ms))  # cover frame
    return _post(f"{config.IG_USER_ID}/media", **params)["id"]


def upload_video(container_id, file_path):
    size = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        r = requests.post(
            f"{config.RUPLOAD_HOST}/{config.API_VERSION}/{container_id}",
            headers={"Authorization": f"OAuth {config.IG_ACCESS_TOKEN}",
                     "offset": "0", "file_size": str(size)},
            data=f,
            timeout=900,
        )
    if not r.ok or not r.json().get("success"):
        raise RuntimeError(f"Upload failed: {r.status_code} {r.text}")


def container_status(container_id):
    """IN_PROGRESS, FINISHED, PUBLISHED, ERROR or EXPIRED."""
    return _get(container_id, fields="status_code,status")


def check_public_url(url):
    """Make sure Instagram will be able to download the file (i.e. Drive sharing is public)."""
    r = requests.get(url, headers={"Range": "bytes=0-1023"}, timeout=30, allow_redirects=True)
    content_type = r.headers.get("Content-Type", "")
    if r.status_code >= 400 or "text/html" in content_type:
        raise RuntimeError(
            f"Video is not publicly downloadable (HTTP {r.status_code}, {content_type or 'no type'}). "
            "Set the Drive folder sharing to 'Anyone with the link: Viewer'.")


def wait_until_ready(container_id, max_wait=900, interval=15):
    waited = 0
    while waited < max_wait:
        data = container_status(container_id)
        code = data.get("status_code")
        if code in ("FINISHED", "PUBLISHED"):
            return code
        if code in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Processing failed: {data}")
        time.sleep(interval)
        waited += interval
    raise TimeoutError("Video processing timed out")


# ---------- Phase 2: publish ----------

def publish_container(container_id):
    return _post(f"{config.IG_USER_ID}/media_publish", creation_id=container_id)["id"]


def find_recent_media_by_caption(caption, limit=15):
    """Find our own recently published media with this exact caption (crash recovery)."""
    data = _get(f"{config.IG_USER_ID}/media", fields="id,caption,timestamp", limit=limit)
    for m in data.get("data", []):
        if (m.get("caption") or "").strip() == caption.strip():
            return m["id"]
    return None


# ---------- Insights ----------

def online_followers_by_hour():
    """Average online followers per hour (hour keys in ONLINE_FOLLOWERS_TZ).
    Returns {} if unavailable (e.g. account under 100 followers)."""
    until = int(time.time())
    try:
        data = _get(f"{config.IG_USER_ID}/insights", metric="online_followers",
                    period="lifetime", since=until - 29 * 86400, until=until)
    except requests.HTTPError:
        return {}
    totals, counts = defaultdict(float), defaultdict(int)
    for series in data.get("data", []):
        for day in series.get("values", []):
            for hour, value in (day.get("value") or {}).items():
                totals[int(hour)] += value
                counts[int(hour)] += 1
    return {h: totals[h] / counts[h] for h in totals}


def media_metrics(media_id):
    for metrics in ("views,reach,likes,comments,shares,saved,total_interactions",
                    "views,likes,comments,shares,saved"):
        try:
            data = _get(f"{media_id}/insights", metric=metrics)
            return {m["name"]: m["values"][0]["value"] for m in data.get("data", [])}
        except requests.HTTPError:
            continue
    raise RuntimeError(f"Could not read insights for {media_id}")
