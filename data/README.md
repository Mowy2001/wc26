# Data — provenance & attribution

All inputs are **public** and used **point-in-time** (never dated after the moment
being predicted). The code in this repo is MIT-licensed; the third-party datasets
below keep their own licenses.

## Core — feeds the deployed live model

| Path | Content | Source / license |
|---|---|---|
| `raw/results.csv` | ~49k international matches, 1872→today, incl. the 2026 schedule | [martj42/international_results](https://github.com/martj42/international_results) — CC0 |
| `raw/goalscorers.csv` | goalscorers per match (player layer / Golden Boot) | martj42/international_results — CC0 |
| `raw/shootouts.csv` | historical penalty shootouts (calibrates the shootout model) | martj42/international_results — CC0 |
| `external/squads_wc2026.csv` | 2026 squads (player layer) | Wikipedia squad pages |
| `external/altitude.json` | venue + habitual-altitude metres (altitude venue tilt **and** altitude-in-Elo correction) | public elevation data |
| `external/geocode.json` | venue coordinates | public geocoding |

## Model lab — feeds **archived / rejected / shadow** experiments only

These are **not** read by the live pipeline or the site at runtime. They are kept
for **reproducibility**: anyone can re-run a rejected block and watch it fail the
out-of-sample gate (that transparency is the point — see `docs/DECISIONS.md`).

| Path | Fed into | Verdict |
|---|---|---|
| `external/clubelo_*.csv` | football capital (club strength of each squad) | REMOVED from the model (Europe bias); kept as shadow |
| `external/fdb_*.json` | capital v2 with a fair global source (footballdatabase) | tested, still failed the gate |
| `external/fbref_*.csv` | player layer v3 (top-5-league minutes / np-goals) | not deployed (subsumed by squad+age) |
| `external/squads_{euro,wc}20xx.csv` | past-tournament squads to **gate** the player / capital / cohesion blocks | — |
| `external/tmax_*.json` (120 files) | climate / heat block (per country & host city max-temperature series) | REJECTED ×2 |
| `external/fifa_ranking.csv` | official FIFA ranking as the strength input (vs our Elo) | REJECTED (+0.07 log-loss worse, WC 1994-2022); `scripts/archive/44` |
| `external/kalshi_groupqual.csv` | Kalshi pre-tournament P(qualify) per team (public candlesticks) | benchmark: market beat us 0.426 vs 0.529; `scripts/archive/45` |

> Why keep ~28 MB of rejected-experiment data? Because "we tested it and it didn't
> earn its place" is only credible if the test is reproducible. See the gating
> philosophy in the top-level `README.md` and the full ledger in `docs/DECISIONS.md`.
