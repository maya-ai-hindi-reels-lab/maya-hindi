# ---- Kaggle bootstrap: runs BEFORE anything imports torch, so the right version loads (no restart needed) ----
import os, sys, subprocess, glob

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

print("[Setup] Installing PyTorch 2.8 + dubbing tools (a few minutes)...", flush=True)
_pip("torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0", "--index-url", "https://download.pytorch.org/whl/cu126")
_pip("omnivoice", "demucs", "openai-whisper", "soundfile", "google-genai", "google-api-python-client", "google-auth", "requests")
def _install_ffmpeg():
    """Kaggle's built-in ffmpeg is old/limited; install a current static build (same features as Colab)."""
    import tarfile, urllib.request, shutil as _sh
    url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
    urllib.request.urlretrieve(url, "/tmp/ffmpeg.tar.xz")
    with tarfile.open("/tmp/ffmpeg.tar.xz") as t:
        for m in t.getmembers():
            if m.name.endswith(("/ffmpeg", "/ffprobe")):
                m.name = os.path.basename(m.name)
                t.extract(m, "/usr/local/bin")
    os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")

def _ffmpeg_ok():
    f = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True)
    e = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True)
    return f.returncode == 0 and all(x in f.stdout for x in ("sidechaincompress", "loudnorm", "amix")) and "libx264" in e.stdout

try:
    _install_ffmpeg()
except Exception as e:
    print(f"[Setup] Couldn't download static ffmpeg ({e}); using the built-in one.")
_v = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
print(f"[Setup] {_v} | all needed features: {'yes' if _ffmpeg_ok() else 'NO'}", flush=True)

def kaggle_secret(filename):
    """Secrets live in the private Kaggle dataset created by the 'kaggle-secrets' GitHub workflow."""
    for path in glob.glob(f"/kaggle/input/**/{filename}", recursive=True):
        return open(path).read().strip()
    raise SystemExit(f"❌ Secret file '{filename}' not found. Run the 'kaggle-secrets' workflow on GitHub first.")
