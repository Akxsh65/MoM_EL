"""Extract only non-audio files (pitch, tonic, sections, metadata) from the remote Saraga zip."""
import os, sys, zipfile
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
from remote_zip import HttpRangeFile

URL = "https://zenodo.org/record/4301737/files/saraga1.5_carnatic.zip?download=1"
OUT = "data/raw/saraga"
import threading, copy
import time
for _t in range(10):
    try: Z0 = zipfile.ZipFile(HttpRangeFile(URL)); break
    except Exception: time.sleep(30)
tl = threading.local()
names = [i for i in Z0.infolist()
         if not i.is_dir() and not i.filename.startswith("__MACOSX")
         and os.path.splitext(i.filename)[1].lower() not in (".mp3", ".ds_store")]
print(len(names), "files", flush=True)

import struct, zlib, urllib.request

def rng(s, e):
    r = urllib.request.Request(URL, headers={"Range": f"bytes={s}-{e}"})
    return urllib.request.urlopen(r, timeout=120).read()

def get(info):
    dest = os.path.join(OUT, info.filename)
    if os.path.exists(dest) and os.path.getsize(dest) == info.file_size: return 1
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for _ in range(5):
        try:
            o = info.header_offset
            buf = rng(o, o + 30 + 1024 + info.compress_size - 1)
            n, m = struct.unpack("<HH", buf[26:30])
            raw = buf[30 + n + m: 30 + n + m + info.compress_size]
            data = raw if info.compress_type == 0 else zlib.decompress(raw, -15)
            assert len(data) == info.file_size
            open(dest, "wb").write(data); return 1
        except Exception:
            import time; time.sleep(5 * (_ + 1))
    return 0

with ThreadPoolExecutor(4) as ex: r = list(ex.map(get, names))
print("ok", sum(r), "of", len(r), flush=True)
