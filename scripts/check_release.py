"""T15 release check: the repository must contain code, manifests, configs, results and our own annotations only.

Command: python scripts/check_release.py     (exit code 1 on any violation)
Fails if `git ls-files` lists anything under data/ or models/, any audio / pitch / tonic / model-weight / array file,
or any file larger than 20 MB. Prints the largest tracked files.
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAD_SUFFIX = (".npz", ".npy", ".wav", ".mp3", ".flac", ".m4a", ".mp4", ".pitch", ".tonic", ".tonicFine", ".joblib", ".pt", ".pth", ".zip")
BAD_PREFIX = ("data/", "models/")
MAX_BYTES = 20 * 2**20

files = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split("\n")
files = [f for f in files if f]
bad = [f for f in files if f.startswith(BAD_PREFIX) or f.endswith(BAD_SUFFIX) or ".pitch" in os.path.basename(f)]
big = [(os.path.getsize(f"{ROOT}/{f}"), f) for f in files if os.path.exists(f"{ROOT}/{f}")]
big.sort(reverse=True)
too_big = [f for s, f in big if s > MAX_BYTES]
print(f"{len(files)} tracked files; largest: " + ", ".join(f"{f} ({s / 2**20:.2f} MB)" for s, f in big[:3]))
if bad or too_big:
    print("VIOLATIONS (raw data / weights / large files must not be released):")
    for f in bad + too_big:
        print("  ", f)
    sys.exit(1)
print("OK: no Saraga/CMD audio, pitch, tonic files, model weights or arrays are tracked.")
