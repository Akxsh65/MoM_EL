"""Parallel, resumable range downloader (Zenodo throttles single connections).
Usage: python scripts/pdl.py URL OUT [workers=16] [chunk_mb=8]"""
import os, sys, json, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

url, out = sys.argv[1], sys.argv[2]
workers = int(sys.argv[3]) if len(sys.argv) > 3 else 16
chunk = int(sys.argv[4]) * 2**20 if len(sys.argv) > 4 else 8 * 2**20
req = urllib.request.Request(url, method="HEAD")
size = int(urllib.request.urlopen(req).headers["Content-Length"])
state_p = out + ".state"
done = set(json.load(open(state_p))) if os.path.exists(state_p) else set()
if not os.path.exists(out):
    with open(out, "wb") as f: f.truncate(size)
lock = threading.Lock(); got = [len(done) * chunk]

def fetch(i):
    if i in done: return
    s, e = i * chunk, min(size, (i + 1) * chunk) - 1
    for _ in range(8):
        try:
            r = urllib.request.Request(url, headers={"Range": f"bytes={s}-{e}"})
            data = urllib.request.urlopen(r, timeout=60).read()
            if len(data) != e - s + 1: raise IOError("short")
            break
        except Exception:
            time.sleep(2)
    else:
        return
    with lock:
        with open(out, "r+b") as f: f.seek(s); f.write(data)
        done.add(i); got[0] += len(data)
        json.dump(sorted(done), open(state_p, "w"))
        print(f"{got[0]/size*100:.1f}%", flush=True)

n = -(-size // chunk)
with ThreadPoolExecutor(workers) as ex: list(ex.map(fetch, range(n)))
print("COMPLETE" if len(done) == n else "INCOMPLETE", flush=True)
