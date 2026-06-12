"""Step 17: fetch FBref Big-5 player season stats via the Wayback Machine.

FBref itself sits behind Cloudflare; archive.org serves the raw HTML of
heavily-visited pages. Snapshots are chosen point-in-time: each backtest
fold uses a page captured BEFORE its tournament (2021-22 season captured
2022-08, months before Qatar). The 2025-26 snapshot (2026-03-09) covers
~3/4 of the season — the freshest data that existed before the tournament.

Output per season: data/external/fbref_{slug}.csv with player, nation,
squad, age, minutes, goals, npxG (npxG is NaN before 2017-18).
"""
import sys, time
from pathlib import Path
import pandas as pd, requests
from bs4 import BeautifulSoup, Comment

EXT = Path("data/external")
UA = {"User-Agent": "wc26-research/0.1 (simone.moawad@gmail.com)"}
PAGES = [
    ("fbref_2017_18", "20260306151902", "https://fbref.com/en/comps/Big5/2017-2018/stats/players/2017-2018-Big-5-European-Leagues-Stats"),
    ("fbref_2021_22", "20220821162035", "https://fbref.com/en/comps/Big5/2021-2022/stats/players/2021-2022-Big-5-European-Leagues-Stats"),
    ("fbref_2024_25", "20251026074147", "https://fbref.com/en/comps/Big5/2024-2025/stats/players/2024-2025-Big-5-European-Leagues-Stats"),
    ("fbref_2025_26", "20260309231038", "https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats"),
]


def fetch(slug, ts, url):
    f = EXT / f"{slug}.html"
    if not f.exists():
        wb = f"https://web.archive.org/web/{ts}/{url}"
        for attempt in range(5):
            try:
                r = requests.get(wb, headers=UA, timeout=300)
                if r.status_code == 200 and len(r.text) > 500_000:
                    f.write_text(r.text)
                    break
                print(f"  {slug}: HTTP {r.status_code}, {len(r.text)}B — retry")
            except Exception as e:
                print(f"  {slug}: {type(e).__name__} — retry")
            time.sleep(20)
        else:
            raise RuntimeError(f"could not fetch {slug}")
        time.sleep(5)
    return f.read_text()


def parse(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", id="stats_standard")
    if table is None:  # FBref wraps tables in HTML comments
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if 'id="stats_standard"' in c:
                table = BeautifulSoup(c, "lxml").find("table", id="stats_standard")
                break
    if table is None:
        raise RuntimeError("stats_standard table not found")
    rows = []
    for tr in table.select("tbody tr"):
        if "thead" in (tr.get("class") or []):
            continue
        cell = {td.get("data-stat"): td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])}
        if not cell.get("player"):
            continue
        rows.append({
            "player": cell["player"],
            "nation": (cell.get("nationality") or "").split()[-1] if cell.get("nationality") else "",
            "squad": cell.get("team") or cell.get("squad") or "",
            "age": (cell.get("age") or "").split("-")[0],
            "minutes": (cell.get("minutes") or "0").replace(",", ""),
            "goals": cell.get("goals") or "0",
            "npg": cell.get("goals_pens") or "",
            "npxg": cell.get("npxg") or "",
        })
    df = pd.DataFrame(rows)
    for c in ["minutes", "goals"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["npxg", "npg"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # players who changed club mid-season appear once per club: aggregate.
    # min_count=1 keeps all-NaN sums as NaN (a 0 would fake availability).
    df = df.groupby(["player", "nation"], as_index=False).agg(
        minutes=("minutes", "sum"), goals=("goals", "sum"),
        npg=("npg", lambda x: x.sum(min_count=1)),
        npxg=("npxg", lambda x: x.sum(min_count=1)),
        age=("age", "first"))
    return df


for slug, ts, url in PAGES:
    html = fetch(slug, ts, url)
    df = parse(html)
    df.to_csv(EXT / f"{slug}.csv", index=False)
    print(f"{slug}: {len(df)} players, minutes>0: {(df.minutes > 0).sum()}, "
          f"npxg available: {df.npxg.notna().sum()}")
print("done")
