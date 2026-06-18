"""Step 11: build the football-capital feature (backlog #5, first residual block).

Sources (cached in data/external/):
  * Wikipedia tournament squads pages -> player, caps, club per team
  * api.clubelo.com daily snapshots   -> club Elo (European clubs only)

Feature: capital(team) = squad mean of club Elo, with non-European/unmatched
clubs imputed at the snapshot's 10th percentile ("outside top club football"
is itself the signal), z-scored within each tournament. Point-in-time: each
tournament uses the clubelo snapshot of its opening day; squads are the
official pre-tournament lists.

Output: outputs/capital.csv (tournament, team, capital_z, coverage).
"""
import re, sys, unicodedata
from pathlib import Path
import pandas as pd, requests
from bs4 import BeautifulSoup
sys.path.insert(0, "src")
from wc26.capital import norm, capital_table

EXT = Path("data/external"); EXT.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "wc26-research/0.1 (simone.moawad@gmail.com)"}

TOURNAMENTS = [
    ("wc2026",   "2026_FIFA_World_Cup_squads", "2026-06-11"),
    ("wc2022",   "2022_FIFA_World_Cup_squads", "2022-11-20"),
    ("wc2018",   "2018_FIFA_World_Cup_squads", "2018-06-14"),
    ("wc2014",   "2014_FIFA_World_Cup_squads", "2014-06-12"),
    ("euro2016", "UEFA_Euro_2016_squads",      "2016-06-10"),
    ("euro2020", "UEFA_Euro_2020_squads",      "2021-06-11"),
    ("euro2024", "UEFA_Euro_2024_squads",      "2024-06-14"),
]
# Minutes-weighting (capital v2, admitted by scripts/21): use club-season
# minutes where a point-in-time FBref page exists; equal-weight otherwise.
FBREF_SEASON = {"wc2018": "fbref_2017_18", "wc2022": "fbref_2021_22",
                "euro2024": "fbref_2023_24", "wc2026": "fbref_2025_26"}
# Wikipedia section names -> results.csv team names
TEAM_ALIASES = {"Türkiye": "Turkey", "Czechia": "Czech Republic",
                "Korea Republic": "South Korea", "IR Iran": "Iran", "Cabo Verde": "Cape Verde"}




def fetch(path: Path, url: str) -> str:
    if not path.exists():
        r = requests.get(url, headers=UA, timeout=60)
        r.raise_for_status()
        path.write_text(r.text)
    return path.read_text()


def parse_squads(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for h3 in soup.select("h3"):
        team = TEAM_ALIASES.get(h3.get_text(" ", strip=True), h3.get_text(" ", strip=True))
        # the squad table is the first sortable table after this heading
        node = h3.parent if h3.parent.name == "div" else h3
        table = node.find_next("table", class_="sortable")
        if table is None:
            continue
        for tr in table.select("tr.nat-fs-player"):
            th = tr.find("th")
            tds = tr.find_all("td")
            if th is None or len(tds) < 5:
                continue
            club_links = tds[-1].find_all("a")
            club = club_links[-1].get_text(" ", strip=True) if club_links else ""
            caps_txt = tds[-3].get_text(strip=True)
            caps = int(caps_txt) if caps_txt.isdigit() else 0
            bday = tr.find("span", class_="bday")
            name = re.sub(r"\s*\(.*?\)\s*", " ", th.get_text(" ", strip=True)).strip()
            rows.append({"team": team, "player": name,
                         "caps": caps, "club": club,
                         "birth": bday.get_text(strip=True) if bday else ""})
    df = pd.DataFrame(rows)
    return df[df["club"] != ""].drop_duplicates(subset=["team", "player"])




all_caps = []
for slug, page, date in TOURNAMENTS:
    html = fetch(EXT / f"squads_{slug}.html", f"https://en.wikipedia.org/wiki/{page}")
    csv = fetch(EXT / f"clubelo_{slug}.csv", f"http://api.clubelo.com/{date}")
    clubelo = pd.read_csv(EXT / f"clubelo_{slug}.csv")
    squads = parse_squads(html)
    squads.to_csv(EXT / f"squads_{slug}.csv", index=False)
    minutes = None
    if slug in FBREF_SEASON and (EXT / f"{FBREF_SEASON[slug]}.csv").exists():
        fb = pd.read_csv(EXT / f"{FBREF_SEASON[slug]}.csv")
        minutes = {norm(p): m for p, m in zip(fb.player, fb.minutes)}
    cap = capital_table(squads, clubelo, minutes=minutes)
    cap.insert(0, "tournament", slug)
    all_caps.append(cap)
    lo, hi = cap.nsmallest(1, "capital_z"), cap.nlargest(1, "capital_z")
    print(f"{slug}: {len(cap)} teams, {squads.shape[0]} players, "
          f"coverage {cap.coverage.mean():.0%} | top {hi.team.iloc[0]} {hi.capital_z.iloc[0]:+.2f} "
          f"| bottom {lo.team.iloc[0]} {lo.capital_z.iloc[0]:+.2f}")

pd.concat(all_caps).round(4).to_csv("outputs/capital_clubelo.csv", index=False)  # legacy; deployed capital = worldelo via scripts/30
print("outputs/capital.csv written")
