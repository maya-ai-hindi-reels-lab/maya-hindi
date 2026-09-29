"""Entry point, run every 30 min by GitHub Actions.
Plans the day once, posts when a slot is due, and collects insights for learning.

Duplicate protection:
  1. Drive file ID      - a file that was posted is never picked again
  2. Content MD5        - identical video under another name/folder is skipped
  3. In-flight record   - if a run dies mid-publish, the next run checks Instagram
                          before retrying, so a crash can't cause a double post
  4. Skip folders       - videos in 'posted'/'done'/'skip' subfolders are ignored
"""
import json
import os
import tempfile
import traceback
from datetime import datetime, timedelta

import ai_caption
import config
import drive
import instagram as ig
import media
import scheduler
import trends


def log(msg):
    print(f"[{datetime.now(config.LOCAL_TZ):%Y-%m-%d %H:%M}] {msg}", flush=True)


# ---------- State ----------

def load_state():
    state = {}
    if os.path.exists(config.STATE_FILE):
        with open(config.STATE_FILE) as f:
            state = json.load(f)
    for key, default in (("posts", []), ("failed", {}), ("skipped", {}), ("plan", []),
                         ("plan_date", None), ("done_slots", []), ("in_flight", None)):
        state.setdefault(key, default)
    return state


def save_state(state):
    tmp = config.STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, config.STATE_FILE)  # atomic: never leaves a half-written file


def record_post(state, video, media_id, caption):
    now = datetime.now(config.LOCAL_TZ)
    state["posts"].append({
        "file_id": video["id"], "name": video["path"], "md5": video.get("md5"),
        "media_id": media_id, "posted_at": now.isoformat(), "local_hour": now.hour,
        "caption": caption, "metrics": None,
    })
    state["failed"].pop(video["id"], None)


def skip(state, video, reason):
    state["skipped"][video["id"]] = {"name": video["path"], "reason": reason}
    log(f"Skipped {video['path']}: {reason}")


# ---------- Crash recovery ----------

def resolve_in_flight(state):
    """If the previous run stopped between upload and saving, find out whether it was published."""
    job = state.get("in_flight")
    if not job:
        return
    log(f"Checking unfinished upload of {job['video']['path']} ...")
    try:
        code = ig.container_status(job["container_id"]).get("status_code")
    except Exception as e:
        code = None
        log(f"Could not read container status: {e}")

    media_id = None
    if code == "PUBLISHED":
        media_id = ig.find_recent_media_by_caption(job["caption"]) or "unknown"
    elif code is None:
        # Container unreadable: fall back to checking our recent posts for the caption
        try:
            media_id = ig.find_recent_media_by_caption(job["caption"])
        except Exception:
            media_id = None

    if media_id:
        record_post(state, job["video"], media_id, job["caption"])
        log(f"It was published (media {media_id}); recorded, will not post again.")
    else:
        log("It was not published; the video will be retried normally.")
    state["in_flight"] = None
    save_state(state)


# ---------- Metrics ----------

def collect_metrics(state):
    cutoff = datetime.now(config.LOCAL_TZ) - timedelta(hours=config.MEASURE_AFTER_HOURS)
    for p in state["posts"]:
        if p.get("metrics") is None and p.get("media_id") not in (None, "unknown") \
                and datetime.fromisoformat(p["posted_at"]) <= cutoff:
            try:
                p["metrics"] = ig.media_metrics(p["media_id"])
                log(f"Insights for {p['name']}: {p['metrics']}")
            except Exception as e:
                log(f"Insights not available yet for {p['name']}: {e}")


# ---------- Posting ----------

def build_reel(video, captions, trending, tmp, trend_tags=None, styles=None):
    """Download -> check -> AI caption + cover pick -> (optional) render.
    Returns (path_to_upload, caption, cover_ms)."""
    ext = os.path.splitext(video["name"])[1] or ".mp4"
    src = drive.download(video["id"], os.path.join(tmp, "src" + ext))
    meta = media.probe(src)

    if not config.PROCESS_VIDEO:
        problems = media.check_as_is(src, meta)
        if problems:
            raise media.UnsupportedVideoError("; ".join(problems))
        if abs(meta["width"] / meta["height"] - 9 / 16) > 0.02:
            log(f"Note: {video['path']} is {meta['width']}x{meta['height']}, not 9:16; "
                "Instagram will crop it in the Reels feed.")

    music = None
    if not meta["has_audio"]:
        if config.PROCESS_VIDEO and config.BACKGROUND_MUSIC_ID:
            music = drive.download(config.BACKGROUND_MUSIC_ID, os.path.join(tmp, "music"))
        elif config.REQUIRE_AUDIO:
            raise media.NoAudioError(video["path"])

    key = video["caption_key"]
    notes = drive.read_text(captions[key]["id"]) if key in captions else ""
    frames = media.extract_frames(src, meta, tmp)  # read-only: the video itself is not changed

    package = ai_caption.fallback(frames, notes)
    if not ai_caption.enabled() and not (notes or config.DEFAULT_CAPTION):
        raise ai_caption.AIUnavailableError("no AI key set and no DEFAULT_CAPTION")
    if ai_caption.enabled():
        try:
            package = ai_caption.generate(frames, notes, trending, trend_tags, styles)
            log(f"Title: {package.get('title', '')}")
        except Exception as e:
            if not (notes or config.DEFAULT_CAPTION):
                raise ai_caption.AIUnavailableError(str(e))
            log(f"AI caption failed, using fallback: {e}")

    if not config.PROCESS_VIDEO:
        return src, package["caption"], package["cover_time"] * 1000  # original file, untouched

    out = media.render(src, meta, os.path.join(tmp, "final.mp4"), hook=package["hook"], music=music)
    return out, package["caption"], package["cover_time"] * 1000


def publish(state, video, path, caption, cover_ms):
    if config.UPLOAD_MODE == "url":
        url = config.VIDEO_URL_TEMPLATE.format(id=video["id"])
        ig.check_public_url(url)
        container_id = ig.create_container(caption, cover_ms, video_url=url)
    else:
        container_id = ig.create_container(caption, cover_ms)
        ig.upload_video(container_id, path)
    ig.wait_until_ready(container_id)

    # Save the in-flight record BEFORE publishing, so a crash right after publish is recoverable
    state["in_flight"] = {"video": video, "container_id": container_id, "caption": caption}
    save_state(state)

    media_id = ig.publish_container(container_id)
    record_post(state, video, media_id, caption)
    state["in_flight"] = None
    save_state(state)
    return media_id


def post_next(state):
    """Returns True if the slot is consumed (posted, or nothing to post)."""
    if ig.publishing_quota_left() <= 0:
        log("Publishing quota exhausted; will retry next run.")
        return False

    posted_ids = {p["file_id"] for p in state["posts"]}
    posted_md5 = {p["md5"] for p in state["posts"] if p.get("md5")}
    exclude = posted_ids | set(state["skipped"])
    exclude |= {fid for fid, n in state["failed"].items() if n >= config.MAX_RETRIES}

    videos, captions = drive.pending_videos(exclude)
    if not videos:
        log("Queue empty: add more videos to the Drive folder.")
        return True

    trending = trends.trending_captions(state)
    use_gemini = ai_caption.provider() == "gemini"
    trend_tags = trends.trending_hashtags(state) if use_gemini else []
    styles = trends.trending_caption_styles(state) if use_gemini else []

    for video in videos:
        if video.get("md5") and video["md5"] in posted_md5:
            original = next(p["name"] for p in state["posts"] if p.get("md5") == video["md5"])
            skip(state, video, f"duplicate of already-posted '{original}'")
            continue

        with tempfile.TemporaryDirectory() as tmp:
            try:
                path, caption, cover_ms = build_reel(video, captions, trending, tmp, trend_tags, styles)
                media_id = publish(state, video, path, caption, cover_ms)
            except ai_caption.AIUnavailableError as e:
                log(f"AI caption unavailable ({e}); not posting a generic caption. Will retry next run.")
                return False
            except media.NoAudioError:
                skip(state, video, "no audio track (set REQUIRE_AUDIO=false to post anyway)")
                continue
            except media.UnsupportedVideoError as e:
                skip(state, video, f"Instagram won't accept it as-is: {e}")
                continue
            except Exception:
                state["failed"][video["id"]] = state["failed"].get(video["id"], 0) + 1
                log(f"Failed to post {video['path']} "
                    f"(attempt {state['failed'][video['id']]}/{config.MAX_RETRIES}):\n"
                    f"{traceback.format_exc()}")
                return False

        log(f"Posted {video['path']} -> media {media_id}")
        return True

    log("No postable videos left in queue.")
    return True


# ---------- Main ----------

def main():
    if config.PROCESS_VIDEO and config.UPLOAD_MODE == "url":
        log("PROCESS_VIDEO needs UPLOAD_MODE=resumable (Facebook Login); uploading originals as-is.")
        config.PROCESS_VIDEO = False
    log(f"AI captions: {ai_caption.provider() or 'off (no GEMINI_API_KEY / ANTHROPIC_API_KEY)'}")
    state = load_state()
    try:
        resolve_in_flight(state)

        now = datetime.now(config.LOCAL_TZ)
        today = now.date().isoformat()

        if state["plan_date"] != today:
            state["plan"] = scheduler.plan_day(state, ig.online_followers_by_hour())
            state["plan_date"], state["done_slots"] = today, []
            log(f"Today's slots: {state['plan']}")

        collect_metrics(state)

        if os.environ.get("POST_NOW", "").lower() == "true":  # manual test from Actions
            log("POST_NOW: posting immediately.")
            post_next(state)
            return

        due = [s for s in state["plan"]
               if s not in state["done_slots"] and datetime.fromisoformat(s) <= now]
        for missed in due[:-1]:  # never burst-post several at once
            state["done_slots"].append(missed)
            log(f"Skipped missed slot {missed}")
        if due:
            slot = due[-1]
            if now - datetime.fromisoformat(slot) > timedelta(hours=config.SLOT_GRACE_HOURS):
                state["done_slots"].append(slot)
                log(f"Skipped stale slot {slot}")
            elif post_next(state):
                state["done_slots"].append(slot)
        else:
            log("No slot due.")
    finally:
        save_state(state)


if __name__ == "__main__":
    main()
