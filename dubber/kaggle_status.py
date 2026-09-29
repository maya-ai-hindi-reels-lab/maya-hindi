"""Reports whether the previous Kaggle dubbing run failed (and shows its log), so GitHub emails you."""
import glob, os, subprocess, sys
kid = f"{os.environ['KAGGLE_USERNAME']}/maya-hindi-dubber"
r = subprocess.run(["kaggle", "kernels", "status", kid], capture_output=True, text=True)
text = (r.stdout + r.stderr).lower()
print(r.stdout.strip() or r.stderr.strip())
failed = "error" in text and "404" not in text and "not found" not in text
if failed:
    os.makedirs("prev_run", exist_ok=True)
    subprocess.run(["kaggle", "kernels", "output", kid, "-p", "prev_run"], capture_output=True)
    for log in glob.glob("prev_run/*.log"):
        print("----- last lines of the previous Kaggle run -----")
        print(open(log, errors="ignore").read()[-4000:])
with open(os.environ.get("GITHUB_OUTPUT", "/dev/null"), "a") as f:
    f.write(f"failed={'true' if failed else 'false'}\n")
