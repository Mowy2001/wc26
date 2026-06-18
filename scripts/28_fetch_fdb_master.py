"""Step 28A: build the footballdatabase master club list (name->slug->points).

Fetch the top N world-ranking pages (current), parse each row's club name,
country, canonical slug and current points. This single list (a) gives current
club strength for the 2026 deployment and (b) the canonical slugs needed to pull
per-club point-in-time history (step 28B). Cloudflare is bypassed with a browser
UA; pages cached under data/external/fdb_rank/.
"""
import re, time, ssl, urllib.request, json
from pathlib import Path
from bs4 import BeautifulSoup

EXT = Path("data/external/fdb_rank"); EXT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
NPAGES = 30  # top ~1500 clubs; squad clubs below that are genuinely weak (floor ok)


def fetch(page):
    f = EXT / f"world_{page}.html"
    if not f.exists():
        req = urllib.request.Request(f"https://footballdatabase.com/ranking/world/{page}", headers=UA)
        for _ in range(4):
            try:
                with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
                    html = r.read().decode("utf-8", "ignore")
                if "clubs-ranking" in html:
                    f.write_text(html); break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise RuntimeError(f"page {page} failed")
        time.sleep(1.2)
    return f.read_text()


master = {}  # slug -> {name, country, points}
for p in range(1, NPAGES + 1):
    soup = BeautifulSoup(fetch(p), "lxml")
    for a in soup.select("a[href*='clubs-ranking']"):
        slug = a.get("href").split("clubs-ranking/")[-1].strip("/")
        name = a.get_text(strip=True)
        if not name or slug in master:
            continue
        # points: the row's last numeric cell; find enclosing row
        tr = a.find_parent("tr")
        pts = None; country = ""
        if tr:
            cells = [c.get_text(" ", strip=True) for c in tr.select("td")]
            nums = [int(x) for c in cells for x in re.findall(r"\b(1[0-9]{3}|20[0-9]{2})\b", c)]
            pts = nums[0] if nums else None
            cc = tr.select_one(".flag, .country") or tr
        master[slug] = {"name": name, "points": pts}
    if p % 10 == 0:
        print(f"  ...{p} pages, {len(master)} clubs")

json.dump(master, open("data/external/fdb_master.json", "w"), ensure_ascii=False, indent=0)
print(f"master list: {len(master)} clubs, {sum(1 for v in master.values() if v['points'])} with points")
print("top:", list(master.items())[:3])
