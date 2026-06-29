"""Step 41: fetch live market odds (The Odds API) and append to a history.

Pulls two markets for the 2026 World Cup:
  - per-match 1X2 (h2h) from `soccer_fifa_world_cup`
  - outright champion odds from `soccer_fifa_world_cup_winner`

Each call's odds are turned into margin-free implied probabilities (median odds
across bookmakers, then de-vigged — proportional for 1X2, Shin for the long
outright market), and a timestamped snapshot is appended to
outputs/odds_history.jsonl. The site reads the latest snapshot for the
model-vs-market comparison and the whole history for market movement.

Key: ODDS_API_KEY env var, else .secrets/odds_api_key (gitignored). Free tier is
~500 requests/month; this uses 2 per run, so twice-daily is ~120/month.
"""
import json
import os
import sys
import time
from pathlib import Path
from statistics import median

import requests

sys.path.insert(0, "src")
from wc26.benchmark import shin_probs

KEY = os.environ.get("ODDS_API_KEY") or Path(".secrets/odds_api_key").read_text().strip()
BASE = "https://api.the-odds-api.com/v4/sports"
HIST = Path("outputs/odds_history.jsonl")

# The Odds API team names -> our canonical names (only where they differ)
NAME_FIX = {"USA": "United States", "South Korea": "Korea Republic"}


def canon(t):
    return NAME_FIX.get(t, t)


def consensus_1x2(outcomes_by_book):
    """Median decimal odds per outcome across books -> de-vigged probs (home,draw,away)."""
    # outcomes_by_book: list of {name: price}
    keys = set().union(*[set(b) for b in outcomes_by_book])
    med = {k: median([b[k] for b in outcomes_by_book if k in b]) for k in keys}
    inv = {k: 1.0 / v for k, v in med.items()}
    s = sum(inv.values())
    return {k: round(v / s, 4) for k, v in inv.items()}  # proportional de-vig


def fetch(url):
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json(), r.headers.get("x-requests-remaining")


def main():
    snap = {"fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "matches": {}, "outright": {}}

    # ---- per-match 1X2 ----
    data, rem = fetch(f"{BASE}/soccer_fifa_world_cup/odds/?apiKey={KEY}&regions=eu&markets=h2h&oddsFormat=decimal")
    for m in data:
        home, away = canon(m["home_team"]), canon(m["away_team"])
        books = []
        for bk in m.get("bookmakers", []):
            mk = next((x for x in bk["markets"] if x["key"] == "h2h"), None)
            if mk:
                books.append({canon(o["name"]): o["price"] for o in mk["outcomes"]})
        if not books:
            continue
        probs = consensus_1x2(books)
        snap["matches"][f"{home}|{away}"] = {
            "home": home, "away": away, "commence": m["commence_time"], "n_books": len(books),
            "pH": probs.get(home, 0.0), "pD": probs.get("Draw", 0.0), "pA": probs.get(away, 0.0)}

    # ---- outright champion ----
    odata, rem = fetch(f"{BASE}/soccer_fifa_world_cup_winner/odds/?apiKey={KEY}&regions=eu&markets=outrights&oddsFormat=decimal")
    am = {}  # team -> median decimal odds
    for ev in odata:
        for bk in ev.get("bookmakers", []):
            for mk in bk["markets"]:
                if mk["key"] == "outrights":
                    for o in mk["outcomes"]:
                        am.setdefault(canon(o["name"]), []).append(o["price"])
    med = {t: median(v) for t, v in am.items()}
    # Shin de-vig over the whole field (long market, favourite-longshot bias)
    if med:
        american = {t: round((o - 1) * 100, 1) for t, o in med.items()}  # decimal->american for shin_probs
        snap["outright"] = {t: round(p, 4) for t, p in shin_probs(american).items()}

    HIST.parent.mkdir(exist_ok=True)
    with open(HIST, "a") as f:
        f.write(json.dumps(snap) + "\n")
    print(f"odds snapshot appended: {len(snap['matches'])} matches, {len(snap['outright'])} outright teams "
          f"| requests remaining: {rem}")
    # show a couple for sanity
    for k, v in list(snap["matches"].items())[:3]:
        print(f"  {v['home']} v {v['away']}: 1X2 {v['pH']:.0%}/{v['pD']:.0%}/{v['pA']:.0%} ({v['n_books']} books)")
    top = sorted(snap["outright"].items(), key=lambda x: -x[1])[:4]
    print("  outright top:", {t: f"{p:.0%}" for t, p in top})


if __name__ == "__main__":
    main()
