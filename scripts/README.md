# scripts/ — the live pipeline

These are the scripts that **reproduce the current forecast and site**. One-off
experiments (gates that reached a verdict, rejected/shadow features) live in
[`archive/`](archive/) with their verdicts. Every script is standalone — run it
from the **repo root** (`python scripts/NN_name.py`); paths are relative to the
working directory, not the script.

## Reproduce from scratch, in order
| # | Script | Produces |
|---|--------|----------|
| 01 | `build_and_validate` | `elo_history.parquet` — Elo from scratch, **altitude-aware** (`ALT_GAMMA`) |
| 08 | `bootstrap_params` | `dc_bootstrap.json` — DC parameter bootstrap (B=100) |
| 19 | `backtest_fatigue` | `fatigue.csv` / `fatigue_beta.json` — **deployed** fatigue tilt |
| 22 | `backtest_altitude` | `altitude_beta.json` — altitude gate (CONMEBOL qualifiers) |
| 23 | `build_altitude_2026` | `altitude_tilt.csv` — **deployed** per-(team,venue) altitude tilt |
| 04 | `simulate_tournament` | `tournament_probs_v1.csv`, `bracket.csv`, `goal_samples.parquet` (eve, unconditioned) |
| 09 | `player_layer` | `golden_boot.csv`, `distinct_scorers.csv` |
| 36 | `export_match_dists` | `match_dists.json` — per-match score heatmaps |
| 38 | `team_drivers` | `team_drivers.json` — strength vs draw vs tilts breakdown |
| 07 | `ablations` | `ablations.json` — host-advantage / xi / fatigue / altitude ablations (site cards) |
| 25 | `shadow_scores` | `shadow_scores.csv` — model-lab shadow board (diaspora, cohesion) |
| 31 | `replay_history` | `history/replay.json` — per-match slider snapshots (incl. bracket `adv`) |
| 10 | `live_update` | **CANONICAL live refresh**: conditions on real results (`fixed_results` + KO), re-sims, appends timeline + replay snapshot, re-exports |
| 05 | `export_site_data` | `site/data.js` (also invoked by 10) |

## Live refresh (during the tournament)
Run **`10_live_update`** on each new result — it conditions on the played
matches, refreshes the headline numbers, and re-exports the site. Periodically
re-run **`31_replay_history`** to rebuild clean slider history. The frozen
`outputs/history/baseline_eve.csv` (June-11 information set) is **immutable** —
never overwrite it.

## Deployed model = Elo → Dixon-Coles + two tilts
The tilts actually in the model are **fatigue** (19) and **altitude** (22/23 +
the in-Elo correction `ALT_GAMMA` in `elo.py`). Everything else tested is in
`archive/` — see its README for the ledger of verdicts.
