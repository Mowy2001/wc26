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
# Wikipedia section names -> results.csv team names
TEAM_ALIASES = {"Türkiye": "Turkey", "Czechia": "Czech Republic",
                "Korea Republic": "South Korea", "IR Iran": "Iran", "Cabo Verde": "Cape Verde"}
# Wikipedia club names -> clubelo names (top offenders; the rest is fuzzy)
CLUB_ALIASES = {
    "manchester city": "man city", "manchester united": "man united",
    "paris saint germain": "paris sg", "inter milan": "inter", "internazionale": "inter",
    "bayern munich": "bayern", "atletico madrid": "atletico", "sporting cp": "sporting",
    "sporting lisbon": "sporting", "tottenham hotspur": "tottenham",
    "borussia dortmund": "dortmund", "borussia monchengladbach": "gladbach",
    "bayer leverkusen": "leverkusen", "ac milan": "milan", "as roma": "roma",
    "real betis": "betis", "athletic bilbao": "bilbao", "real sociedad": "sociedad",
    "newcastle united": "newcastle", "west ham united": "west ham",
    "wolverhampton wanderers": "wolves", "nottingham forest": "forest",
    "brighton hove albion": "brighton", "psv eindhoven": "psv",
    "rb leipzig": "leipzig", "red bull salzburg": "salzburg", "rangers": "rangers",
    "celtic": "celtic", "olympique lyonnais": "lyon", "olympique de marseille": "marseille",
    "as monaco": "monaco", "dinamo zagreb": "din zagreb", "red star belgrade": "crvena zvezda",
}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    drop = {"fc", "cf", "sc", "ac", "afc", "cd", "sv", "bk", "fk", "sk", "if", "ks",
            "club", "de", "futbol", "futebol", "calcio", "1", "04", "05", "09", "1899", "1900"}
    return " ".join(w for w in s.split() if w not in drop)


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
            rows.append({"team": team, "player": th.get_text(" ", strip=True),
                         "caps": caps, "club": club})
    df = pd.DataFrame(rows)
    return df[df["club"] != ""].drop_duplicates(subset=["team", "player"])


def capital_table(squads: pd.DataFrame, clubelo: pd.DataFrame) -> pd.DataFrame:
    elo_by_norm = {}
    for r in clubelo.itertuples(index=False):
        elo_by_norm.setdefault(norm(r.Club), float(r.Elo))
    names = list(elo_by_norm)
    floor = clubelo["Elo"].quantile(0.10)
    from difflib import get_close_matches

    def club_elo(club: str) -> float | None:
        n = CLUB_ALIASES.get(norm(club), norm(club))
        if n in elo_by_norm:
            return elo_by_norm[n]
        hit = get_close_matches(n, names, n=1, cutoff=0.88)
        return elo_by_norm[hit[0]] if hit else None

    out = []
    for team, g in squads.groupby("team"):
        elos = g["club"].map(club_elo)
        matched = elos.notna()
        out.append({"team": team, "capital_raw": float(elos.fillna(floor).mean()),
                    "coverage": float(matched.mean()), "n_players": len(g)})
    df = pd.DataFrame(out)
    df["capital_z"] = (df.capital_raw - df.capital_raw.mean()) / df.capital_raw.std(ddof=0)
    return df


all_caps = []
for slug, page, date in TOURNAMENTS:
    html = fetch(EXT / f"squads_{slug}.html", f"https://en.wikipedia.org/wiki/{page}")
    csv = fetch(EXT / f"clubelo_{slug}.csv", f"http://api.clubelo.com/{date}")
    clubelo = pd.read_csv(EXT / f"clubelo_{slug}.csv")
    squads = parse_squads(html)
    squads.to_csv(EXT / f"squads_{slug}.csv", index=False)
    cap = capital_table(squads, clubelo)
    cap.insert(0, "tournament", slug)
    all_caps.append(cap)
    lo, hi = cap.nsmallest(1, "capital_z"), cap.nlargest(1, "capital_z")
    print(f"{slug}: {len(cap)} teams, {squads.shape[0]} players, "
          f"coverage {cap.coverage.mean():.0%} | top {hi.team.iloc[0]} {hi.capital_z.iloc[0]:+.2f} "
          f"| bottom {lo.team.iloc[0]} {lo.capital_z.iloc[0]:+.2f}")

pd.concat(all_caps).round(4).to_csv("outputs/capital.csv", index=False)
print("outputs/capital.csv written")
