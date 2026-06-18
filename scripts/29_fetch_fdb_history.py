"""Step 28B/C: match squad clubs to fdb slugs, fetch per-club monthly history.

Each /clubs-ranking/{slug} page embeds data_array = [{Date, Points, Ranking}...]
monthly from 2010 — point-in-time club strength with ONE fetch per club, no
Wayback. Output: data/external/fdb_history.json {slug: {"YYYY-MM": points}} and
data/external/fdb_clubmap.json {squad_club_name: slug or null}.
"""
import re, json, glob, time, ssl, unicodedata, urllib.request
from pathlib import Path
from difflib import get_close_matches
import pandas as pd

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
HIST = Path("data/external/fdb_hist")
master = json.load(open("data/external/fdb_master.json"))

DROP = {"fc", "cf", "sc", "ac", "afc", "cd", "sv", "bk", "fk", "sk", "if", "ks", "club", "de", "s", "l", "city"}
ALIAS = {"copenhagen": "kobenhavn", "fc copenhagen": "kobenhavn"}  # squad-name -> hint


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in DROP)


mnorm = {}
for slug, v in master.items():
    mnorm.setdefault(norm(v["name"]), slug)
mkeys = list(mnorm)


def match(club):
    n = ALIAS.get(norm(club), norm(club))
    if n in mnorm:
        return mnorm[n]
    for k in mkeys:
        if k and (k in n or n in k) and abs(len(k) - len(n)) <= 4:
            return mnorm[k]
    h = get_close_matches(n, mkeys, n=1, cutoff=0.86)
    return mnorm[h[0]] if h else None


clubs = set()
for f in glob.glob("data/external/squads_*.csv"):
    clubs |= set(pd.read_csv(f).club.dropna())
clubmap = {c: match(c) for c in sorted(clubs)}
json.dump(clubmap, open("data/external/fdb_clubmap.json", "w"), ensure_ascii=False, indent=0)
slugs = sorted({s for s in clubmap.values() if s})
print(f"{len(clubs)} clubs -> {len(slugs)} distinct fdb slugs to fetch")


def fetch_hist(slug):
    f = HIST / f"{slug}.json"
    if f.exists():
        return json.load(open(f))
    req = urllib.request.Request(f"https://footballdatabase.com/clubs-ranking/{slug}", headers=UA)
    html = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                html = r.read().decode("utf-8", "ignore")
            if "data_array" in html:
                break
        except Exception:
            pass
        time.sleep(4)
    recs = re.findall(r"Date:\s*new Date\('(\d{4})-(\d{2})-\d{2}'\),\s*Points:\s*(\d+)", html or "")
    series = {f"{y}-{m}": int(p) for y, m, p in recs}
    json.dump(series, open(f, "w"))
    time.sleep(1.1)
    return series


import os as _os
hist = json.load(open("data/external/fdb_history.json")) if _os.path.exists("data/external/fdb_history.json") else {}  # merge existing (don't clobber manual slugs)
try:
    _man = pd.read_csv("data/external/fdb_manual_slugs.csv", comment="#")
    slugs = sorted(set(slugs) | {x for x in _man.slug.dropna() if x})
except FileNotFoundError:
    pass
for i, slug in enumerate(slugs, 1):
    hist[slug] = fetch_hist(slug)
    if i % 50 == 0:
        print(f"  ...{i}/{len(slugs)} ({sum(1 for v in hist.values() if v)} non-empty)")
json.dump(hist, open("data/external/fdb_history.json", "w"))
ok = sum(1 for v in hist.values() if v)
print(f"history fetched: {ok}/{len(slugs)} slugs with a series")
