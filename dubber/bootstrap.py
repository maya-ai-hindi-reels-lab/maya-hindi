# ---- Kaggle bootstrap: runs BEFORE anything imports torch, so the right version loads (no restart needed) ----
import os, sys, subprocess, glob

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

print("[Setup] Installing PyTorch 2.8 + dubbing tools (a few minutes)...", flush=True)
_pip("torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0", "--index-url", "https://download.pytorch.org/whl/cu126")
_pip("omnivoice", "demucs", "openai-whisper", "soundfile", "google-genai", "google-api-python-client", "google-auth", "requests")
if subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0:
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)

def kaggle_secret(filename):
    """Secrets live in the private Kaggle dataset created by the 'kaggle-secrets' GitHub workflow."""
    for path in glob.glob(f"/kaggle/input/**/{filename}", recursive=True):
        return open(path).read().strip()
    raise SystemExit(f"❌ Secret file '{filename}' not found. Run the 'kaggle-secrets' workflow on GitHub first.")
