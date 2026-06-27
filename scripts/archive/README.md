# scripts/archive/ — completed experiments (not in the live pipeline)

Every script here reached a **verdict** and is kept for the record, not re-run.
Their committed artifacts in `outputs/` remain (the site still reads some of
them), so archiving the script changes nothing live. Run from the **repo root**
if you ever need to regenerate (`python scripts/archive/NN_name.py`). The full
narrative is in `docs/METHODOLOGY.md`; this is the index.

## Baselines & hyperparameters
| # | Script | Verdict |
|---|--------|---------|
| 02 | `fit_backtest` | One-off WC2022 backtest (LL 1.06 vs 1.099 uniform). |
| 03 | `simulate_groups` | Group-stage v0 — **superseded** by `04`. |
| 06 | `tune_xi` | Time-decay ξ — **flat plateau**, not identified; chosen at centre (`xi_tuning.csv`). |
| 37 | `gridsearch_elo_weights` | K-factors / home-adv — **flat plateau**, kept eloratings convention. |
| 40 | `broadened_gate` | Cross-confederation robustness gate — model **generalises** (better outside UEFA). |

## Capital — REMOVED from the model
| # | Script | Verdict |
|---|--------|---------|
| 11,12 | `build_capital`, `backtest_capital` | clubelo squad strength — admitted, but signal was **Europe bias**. |
| 21 | `backtest_capital_v2` | minutes-weighting — near-null. |
| 28,29,30,32,33,34 | fdb pipeline (`fetch_fdb_*`, `build_capital_fdb`, `fdb_country_fallback`, `gate_capital_source`, `build_capital_hybrid`) | fair global source **fails the gate** → capital **removed**; fdb kept for the model lab. |

## Climate / heat — REJECTED
| # | Script | Verdict |
|---|--------|---------|
| 13,14 | `build_climate`, `backtest_climate` | national-country climate — **rejected** (Qatar-2022 AC). |
| 27 | `backtest_climate_clubcountry` | club-country re-test — **rejected again**; CIES academies dropped. |

## Player layer
| # | Script | Verdict |
|---|--------|---------|
| 16,18 | `backtest_players`, `backtest_players_xg` | squad-filter + age α=0.1 **admitted, deployed in `09`**; npxG blend subsumed. |
| 17 | `fetch_fbref` | FBref season data scrape (`data/external/fbref_*`). |
| 20 | `backtest_penalty` | penalty-taker bonus — **rejected** (double-counts). |

## Residual blocks — shadow / excluded
| # | Script | Verdict |
|---|--------|---------|
| 24 | `build_diaspora` | US-ancestry support — **shadow** (un-gateable, no past US WC). |
| 26 | `backtest_cohesion` | club-concentration proxy — **excluded** (proportionality). |
| 39 | `backtest_cohesion_continuity` | squad continuity — **rejected** (OOS +0.0029). |

## Deployed via the library, gate archived
| # | Script | Verdict |
|---|--------|---------|
| 35 | `backtest_altitude_elo` | altitude-aware Elo home advantage — **admitted, DEPLOYED** via `elo.py:ALT_GAMMA` + `01`. |

## One-off report
| # | Script | Verdict |
|---|--------|---------|
| 15 | `benchmark_report` | three-way model/market/Klement comparison (market via `benchmark.shin_probs`, used live by `05`). |
