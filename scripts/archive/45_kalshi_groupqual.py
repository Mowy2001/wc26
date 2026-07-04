"""Benchmark: our pre-tournament group-qualification calls vs the Kalshi market.

The one per-team "does X reach the knockouts" market that existed BEFORE kickoff
with a verifiable point-in-time price: Kalshi's KXWCGROUPQUAL series (one binary
market per team, open since 2025-12-12). Prices are the last daily candle close
strictly before the opening match (2026-06-11 17:00 UTC), fetched from Kalshi's
public candlestick API and frozen in data/external/kalshi_groupqual.csv - a
single provider, no vig to remove (prices are traded probabilities), and anyone
can re-pull the same candles to audit. The playoff losers (Jamaica, New
Caledonia) never traded and are excluded; the priced markets sum to 32.0, the
exact number of advancing teams.

Our side: the eve-of-tournament P(qualify) per team, replay.json snapshot k=0.

Result (recorded 2026-07-04, groups fully decided):
    log-loss  model 0.5286   Kalshi 0.4261   (Brier 0.183 vs 0.140)
    paired t = 2.83 over the 48 calls; model closer on only 16/48 teams.
A clear point to the market: it priced squad news and form we deliberately
exclude (Panama: model 72% / market 32% -> out; Ghana: 16% / 47% -> through).
Consistent with the design bet - our edge, if any, is mechanistic transparency,
not information.

Standalone; run from the repo root:  python scripts/archive/45_kalshi_groupqual.py
(Re-fetch candles yourself with --refetch to audit the CSV.)
"""
import csv
import json
import math
import statistics
import sys

REFETCH = "--refetch" in sys.argv
CSV = "data/external/kalshi_groupqual.csv"
KICKOFF_TS = 1781283600  # 2026-06-11 17:00 UTC, before the opening match


def refetch() -> None:
    """Re-pull the pre-tournament closes from Kalshi's public API (audit path)."""
    import datetime as dt
    import time
    import urllib.request

    base = "https://api.elections.kalshi.com/trade-api/v2"

    def get(u):
        req = urllib.request.Request(u, headers={"User-Agent": "wc26-research/0.1"})
        return json.load(urllib.request.urlopen(req, timeout=20))

    name = {"Turkiye": "Turkey", "USA": "United States", "Korea Republic": "South Korea",
            "IR Iran": "Iran", "Congo DR": "DR Congo", "Czechia": "Czech Republic",
            "Curacao": "Curaçao"}
    start = int(dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc).timestamp())
    rows = []
    for m in get(f"{base}/markets?series_ticker=KXWCGROUPQUAL&limit=200&status=settled")["markets"]:
        team = m.get("yes_sub_title") or m["title"]
        cs = get(f"{base}/series/KXWCGROUPQUAL/markets/{m['ticker']}/candlesticks"
                 f"?start_ts={start}&end_ts={KICKOFF_TS}&period_interval=1440").get("candlesticks", [])
        px = None
        for cd in reversed(cs):
            p = cd.get("price", {}).get("close_dollars")
            if p:
                px = float(p)
            else:
                b = cd.get("yes_bid", {}).get("close_dollars")
                a = cd.get("yes_ask", {}).get("close_dollars")
                if b and a:
                    px = (float(b) + float(a)) / 2
            if px is not None:
                break
        if px is None:
            continue  # playoff losers never traded
        rows.append([name.get(team, team), m["ticker"], round(px, 4), m["result"]])
        time.sleep(0.12)
    with open(CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["team", "ticker", "p_qualify_pre", "result"])
        w.writerows(sorted(rows))
    print(f"{CSV} refetched: {len(rows)} teams")


def main() -> None:
    if REFETCH:
        refetch()
    kal = {r["team"]: r for r in csv.DictReader(open(CSV))}
    eve = json.load(open("outputs/history/replay.json"))["snapshots"][0]["qualify"]

    diffs, ll_m, ll_k, br_m, br_k, n = [], 0.0, 0.0, 0.0, 0.0, 0
    for team, r in sorted(kal.items()):
        y = 1 if r["result"] == "yes" else 0
        pm = min(max(eve[team], 1e-4), 1 - 1e-4)
        pk = min(max(float(r["p_qualify_pre"]), 1e-4), 1 - 1e-4)
        lm = -(y * math.log(pm) + (1 - y) * math.log(1 - pm))
        lk = -(y * math.log(pk) + (1 - y) * math.log(1 - pk))
        ll_m += lm; ll_k += lk
        br_m += (pm - y) ** 2; br_k += (pk - y) ** 2
        diffs.append(lm - lk); n += 1

    t = (sum(diffs) / n) / (statistics.stdev(diffs) / math.sqrt(n))
    print(f"n={n} pre-tournament qualification calls (Kalshi close before kickoff)")
    print(f"log-loss  model {ll_m/n:.4f}   Kalshi {ll_k/n:.4f}   delta {ll_m/n - ll_k/n:+.4f}")
    print(f"Brier     model {br_m/n:.4f}   Kalshi {br_k/n:.4f}")
    print(f"paired t = {t:.2f}; model closer on {sum(1 for x in diffs if x < 0)}/{n} teams")
    print("Verdict: the market wins this panel - it prices information we deliberately exclude.")


if __name__ == "__main__":
    main()
