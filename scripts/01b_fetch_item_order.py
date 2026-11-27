"""Fetch Saraga concert track order from MusicBrainz (public API, 1 req/s, generic User-Agent).
Command: python scripts/01b_fetch_item_order.py   -> data/raw/musicbrainz/{release_mbid}.json (cache, gitignored)"""
import glob, json, os, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/data/raw/musicbrainz"
os.makedirs(OUT, exist_ok=True)
rel = {}
for j in glob.glob(f"{ROOT}/data/raw/saraga/saraga1.5_carnatic/*/*/*.json"):
    c = (json.load(open(j, encoding="utf8")).get("concert") or [{}])[0]
    if c.get("mbid"): rel[c["mbid"]] = c["title"]
for m, title in sorted(rel.items()):
    dest = f"{OUT}/{m}.json"
    if os.path.exists(dest): continue
    req = urllib.request.Request(f"https://musicbrainz.org/ws/2/release/{m}?inc=recordings&fmt=json",
                                 headers={"User-Agent": "raga-tracking-research/0.1"})
    for k in range(4):
        try:
            open(dest, "wb").write(urllib.request.urlopen(req, timeout=30).read()); print("ok", title); break
        except Exception as e:
            print("retry", title, e); time.sleep(5 * (k + 1))
    time.sleep(1.1)
