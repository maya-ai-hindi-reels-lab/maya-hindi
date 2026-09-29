"""ffmpeg processing: 9:16 1080x1920, H.264/AAC, loudness-normalised audio, hook text overlay."""
import json
import os
import subprocess
import textwrap

import config

W, H = 1080, 1920
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class NoAudioError(Exception):
    pass


class UnsupportedVideoError(Exception):
    pass


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed:\n{result.stderr[-2000:]}")
    return result.stdout


def probe(path):
    info = json.loads(_run(["ffprobe", "-v", "error", "-print_format", "json",
                            "-show_streams", "-show_format", path]))
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    width, height = int(video["width"]), int(video["height"])

    # Phone videos often store portrait as landscape + rotation metadata
    rotation = int(video.get("tags", {}).get("rotate", 0))
    for side in video.get("side_data_list", []):
        rotation = int(side.get("rotation", rotation))
    if abs(rotation) in (90, 270):
        width, height = height, width

    audio = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    real_duration = float(info["format"]["duration"])
    return {
        "width": width,
        "height": height,
        "has_audio": audio is not None,
        "video_codec": video.get("codec_name", ""),
        "audio_codec": audio.get("codec_name", "") if audio else "",
        "format": info["format"].get("format_name", ""),
        "real_duration": real_duration,
        "duration": min(real_duration, config.MAX_DURATION),  # used for frame sampling / rendering
    }


def check_as_is(path, meta):
    """Problems that would make Instagram reject the original file. Empty list = OK to upload."""
    problems = []
    if not any(fmt in meta["format"] for fmt in ("mp4", "mov")):
        problems.append(f"container '{meta['format']}' (needs MP4/MOV)")
    if meta["video_codec"] not in ("h264", "hevc"):
        problems.append(f"video codec '{meta['video_codec']}' (needs H.264/HEVC)")
    if meta["has_audio"] and meta["audio_codec"] != "aac":
        problems.append(f"audio codec '{meta['audio_codec']}' (needs AAC)")
    if meta["real_duration"] < 3:
        problems.append(f"too short ({meta['real_duration']:.1f}s, min 3s)")
    size_mb = os.path.getsize(path) / 1024 / 1024
    if size_mb > config.MAX_FILE_MB:
        problems.append(f"file too large ({size_mb:.0f} MB, max {config.MAX_FILE_MB} MB)")
    return problems


def extract_frames(path, meta, out_dir, count=8):
    """Evenly spaced candidate frames (skipping the very start/end) for cover + AI context."""
    frames = []
    for i in range(count):
        t = meta["duration"] * (i + 0.5) / count
        out = os.path.join(out_dir, f"frame_{i}.jpg")
        _run(["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1",
              "-vf", "scale=540:-2", "-q:v", "3", out])
        frames.append((t, out))
    return frames


def _video_filter(meta):
    if abs(meta["width"] / meta["height"] - 9 / 16) < 0.01:
        return f"[0:v]scale={W}:{H},setsar=1,fps=30[base]"
    if config.FIT_MODE == "crop":
        return (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},setsar=1,fps=30[base]")
    # Fit the whole frame, fill the rest with a blurred copy (no black bars)
    return (f"[0:v]split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:5[bgb];"
            f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[fgs];"
            f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,fps=30[base]")


def render(src, meta, dest, hook=None, music=None):
    graph = _video_filter(meta)
    video_out = "[base]"

    if hook and config.HOOK_OVERLAY != "off":
        text_file = os.path.join(os.path.dirname(dest), "hook.txt")
        with open(text_file, "w") as f:
            f.write("\n".join(textwrap.wrap(hook, 18)))
        enable = ":enable='lt(t,3)'" if config.HOOK_OVERLAY == "intro" else ""
        graph += (f";[base]drawtext=fontfile={FONT}:textfile={text_file}:fontsize=72:"
                  f"fontcolor=white:line_spacing=14:box=1:boxcolor=black@0.55:boxborderw=28:"
                  f"x=(w-text_w)/2:y=h*0.14{enable}[v]")
        video_out = "[v]"

    cmd = ["ffmpeg", "-y", "-i", src]
    if meta["has_audio"]:
        audio_in = "[0:a:0]"
    elif music:
        cmd += ["-stream_loop", "-1", "-i", music]
        audio_in = "[1:a:0]"
    else:
        raise NoAudioError("Video has no audio track")
    # Instagram-style loudness target so the reel isn't quiet when sound is on
    graph += f";{audio_in}loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[a]"

    cmd += [
        "-filter_complex", graph, "-map", video_out, "-map", "[a]",
        "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20", "-maxrate", "10M", "-bufsize", "20M",
        "-c:a", "aac", "-b:a", "192k", "-ac", "2",
        "-t", f"{meta['duration']:.2f}", "-movflags", "+faststart", dest,
    ]
    _run(cmd)
    return dest
