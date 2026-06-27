"""Step 32: country-page fallback for clubs not in the top-1500 world list.

Simone's method: for an unmatched club, fetch footballdatabase's national
ranking (e.g. /ranking/uzbekistan/1) — which lists every club of that country
incl. weak ones — and find the slug there. We infer a club's country from the
national teams of its players (correct for domestic clubs of weaker leagues,
which is exactly the gap clubelo/the world-top-1500 miss). Extends
fdb_clubmap.json and fetches the newly found slugs' histories.
"""
import re, json, glob, time, ssl, unicodedata, urllib.request
from pathlib import Path
from difflib import get_close_matches
import pandas as pd
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
CR = Path("data/external/fdb_country"); CR.mkdir(parents=True, exist_ok=True)
HIST = Path("data/external/fdb_hist")
clubmap = json.load(open("data/external/fdb_clubmap.json"))
DROP = {"fc", "cf", "sc", "ac", "afc", "cd", "sv", "bk", "fk", "sk", "if", "ks", "club", "de", "s", "l", "city"}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(w for w in s.split() if w not in DROP)


def cslug(nation):
    fix = {"United States": "usa", "South Korea": "south-korea", "Saudi Arabia": "saudi-arabia",
           "Czech Republic": "czech-republic", "Ivory Coast": "ivory-coast", "DR Congo": "dr-congo",
           "Cape Verde": "cape-verde", "South Africa": "south-africa", "New Zealand": "new-zealand",
           "Bosnia and Herzegovina": "bosnia-herzegovina"}
    if nation in fix:
        return fix[nation]
    return re.sub(r"\s+", "-", norm(nation))


def fetch_country(slug, page):
    f = CR / f"{slug}_{page}.html"
    if not f.exists():
        try:
            req = urllib.request.Request(f"https://footballdatabase.com/ranking/{slug}/{page}", headers=UA)
            with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                html = r.read().decode("utf-8", "ignore")
        except Exception:
            return {}
        f.write_text(html); time.sleep(1.0)
    soup = BeautifulSoup(f.read_text(), "lxml")
    out = {}
    for tr in soup.select("tr"):
        a = tr.select_one("a[href*='clubs-ranking']")
        if a and a.get_text(strip=True):
            out[norm(a.get_text(strip=True))] = a.get("href").split("clubs-ranking/")[-1].strip("/")
    return out


# unmatched clubs -> candidate nations (from players' national teams)
clubs_nat = {}
for f in glob.glob("data/external/squads_*.csv"):
    sq = pd.read_csv(f)
    for c, t in zip(sq.club, sq.team):
        if clubmap.get(c) is None and pd.notna(c):
            clubs_nat.setdefault(c, set()).add(t)

nations = sorted({n for ns in clubs_nat.values() for n in ns})
country_idx = {}
for nat in nations:
    sl = cslug(nat)
    idx = {}
    for p in (1, 2):
        idx.update(fetch_country(sl, p))
    if idx:
        country_idx[nat] = idx

found = 0
for club, nats in clubs_nat.items():
    cands = {}
    for n in nats:
        cands.update(country_idx.get(n, {}))
    if not cands:
        continue
    nc = norm(club)
    slug = cands.get(nc)
    if not slug:
        for k, v in cands.items():
            if k and (k in nc or nc in k) and abs(len(k) - len(nc)) <= 6:
                slug = v; break
    if not slug:
        h = get_close_matches(nc, list(cands), n=1, cutoff=0.82)
        slug = cands[h[0]] if h else None
    if slug:
        clubmap[club] = slug; found += 1

json.dump(clubmap, open("data/external/fdb_clubmap.json", "w"), ensure_ascii=False, indent=0)
print(f"country fallback recovered {found} clubs; matched total now "
      f"{sum(v is not None for v in clubmap.values())}/{len(clubmap)}")

# fetch histories for any newly mapped slugs
hist = json.load(open("data/external/fdb_history.json"))
newslugs = sorted({s for s in clubmap.values() if s and s not in hist})
print(f"fetching {len(newslugs)} new club histories...")
for i, slug in enumerate(newslugs, 1):
    f = HIST / f"{slug}.json"
    if f.exists():
        hist[slug] = json.load(open(f)); continue
    try:
        req = urllib.request.Request(f"https://footballdatabase.com/clubs-ranking/{slug}", headers=UA)
        with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
            html = r.read().decode("utf-8", "ignore")
        recs = re.findall(r"Date:\s*new Date\('(\d{4})-(\d{2})-\d{2}'\),\s*Points:\s*(\d+)", html)
        hist[slug] = {f"{y}-{m}": int(p) for y, m, p in recs}
        json.dump(hist[slug], open(f, "w")); time.sleep(1.0)
    except Exception:
        hist[slug] = {}
json.dump(hist, open("data/external/fdb_history.json", "w"))
print(f"history now covers {sum(1 for v in hist.values() if v)} slugs")
