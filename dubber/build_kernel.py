"""Assembles the single-file Kaggle script (bootstrap + dubbing core + worker) and its metadata."""
import argparse, json, os
here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--count", type=int, required=True)
ap.add_argument("--out", default="build/kernel")
a = ap.parse_args()

user = os.environ["KAGGLE_USERNAME"]
skip = [f.strip().lower() for f in (os.environ.get("SKIP_FOLDERS") or "posted,done,skip").split(",") if f.strip()]
params = {"count": a.count, "orig_folder": os.environ["ORIG_FOLDER_ID"], "dub_folder": os.environ["HI_DRIVE_FOLDER_ID"],
          "skip_folders": skip, "settings": json.load(open(f"{here}/settings.json"))}
if params["settings"].get("VOICE_ENGINE") == "hindi_then_seedvc":
    # Kaggle gets a single script, so the native Hindi reference voices travel inside it
    import base64
    params["voices"] = {name: base64.b64encode(open(f"{here}/voices/{name}", "rb").read()).decode()
                        for name in sorted(os.listdir(f"{here}/voices")) if name.endswith((".wav", ".json"))}
os.makedirs(a.out, exist_ok=True)
src = ["import json\n", f"PARAMS = json.loads({json.dumps(json.dumps(params, ensure_ascii=False))})\n"]
for part in ("bootstrap.py", "core.py", "worker.py"):
    src.append(f"\n# {'=' * 20} {part} {'=' * 20}\n" + open(f"{here}/{part}", encoding="utf-8").read())
open(f"{a.out}/kernel.py", "w", encoding="utf-8").write("".join(src))
json.dump({"id": f"{user}/maya-hindi-dubber", "title": "maya-hindi-dubber", "code_file": "kernel.py",
           "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
           "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
           "dataset_sources": [f"{user}/maya-dub-secrets"], "competition_sources": [], "kernel_sources": []},
          open(f"{a.out}/kernel-metadata.json", "w"), indent=2)
print(f"Kernel built: dub {a.count} video(s)")
