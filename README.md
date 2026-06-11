# WC26 — Probabilistic World Cup 2026 Forecasting

End-to-end probabilistic model for the 2026 FIFA World Cup: group tables, qualifiers, knockout bracket, Golden Boot, most-distinct-scorers team and most-goals-in-group-stage team.

**Philosophy:** we don't predict *who wins*, we predict *distributions*. Every output is an empirical frequency over tens of thousands of Monte Carlo simulations. The model is evaluated on log-loss and calibration, never on a single outcome.

## Architecture (3 layers)

```
[Match history 1872→today] ──► Proprietary Elo (src/wc26/elo.py)
                                      │
                                      ▼
                    Dixon-Coles with Elo covariates + residual blocks
                              (src/wc26/dixon_coles.py)
                                      │  λ_home, λ_away per fixture
                                      ▼
                    Tournament Monte Carlo (src/wc26/simulate.py)
                    groups → best thirds → R32 bracket → final
                                      │
                                      ▼
                    Player layer (TODO): goal allocation → Golden Boot
```

1. **Proprietary Elo** — a from-scratch replica of eloratings.net computed over the full history: gives *point-in-time* ratings at any date (zero look-ahead bias in backtests).
2. **Dixon-Coles** — Poisson goals for each side, with the ρ low-score correction and exponential time-downweighting. Team strengths are driven by Elo (no free per-team parameters: international data is too sparse). The `extra_cols` hook accepts residual blocks (climate, football capital, diaspora, fatigue, cohesion).
3. **Monte Carlo** — samples every scoreline from the DC score matrix, applies FIFA tiebreakers, ranks the 8 best third-placed teams, propagates the bracket. Supports `fixed_results` to condition on matches already played (live-updating forecasts during the tournament).

## Quick start

```bash
pip install -r requirements.txt
python scripts/01_build_and_validate.py   # Elo + group reconstruction (sanity check)
python scripts/02_fit_backtest.py         # DC fit + out-of-sample backtest on WC2022
python scripts/03_simulate_groups.py      # 20k group-stage simulations
```

## Data

| File | Content | Used for |
|---|---|---|
| `data/raw/results.csv` | 49k international matches 1872→today, **including the 2026 schedule** | Elo, DC fit, tournament structure |
| `data/raw/goalscorers.csv` | 47k goals with scorer, minute, penalty flag | Player layer |
| `data/raw/shootouts.csv` | Historical penalty shootouts | Knockout shootout calibration |

Source: github.com/martj42/international_results (CC0). The 2026 groups are **reconstructed algorithmically** from the fixture graph (connected components) — the draw is never hardcoded.

## Current results (model v0, Elo-only)

WC2022 backtest (64 matches, point-in-time pre-tournament fit):
- Log-loss 1.062 vs 1.099 uniform; favourite calibration: 54.4% predicted vs 56.2% observed.

External benchmarks recorded on 2026-06-11 (tournament eve): BetMGM outright odds (Spain +450, France +500, England +700...), Golden Boot (Mbappé +550/600, Kane +650/700) and Klement's forecast (Netherlands champions over Portugal). Three-way comparison after the tournament.

## Known issues / roadmap

See `CLAUDE.md` for the detailed operational backlog. Highlights:
1. **Host home advantage**: verified active (raw data flags host home matches correctly; a defensive invariant now enforces it in `data.py`). The residual USA gap vs the market is a rating question, not a venue one — see `docs/METHODOLOGY.md`.
2. **Group labels**: composition is verified, but letters (A..L) are assigned by kickoff order and must be reconciled with official FIFA labels.
3. **Knockout bracket**: needs the FIFA third-place allocation table (Annex C of the regulations).

## Repo layout

```
src/wc26/        modules (elo, dixon_coles, data, simulate) — every file has
                 a module docstring with the full mathematical model
scripts/         numbered pipeline, run in order
docs/            METHODOLOGY.md — design decisions and their rationale
outputs/         elo_history.parquet, simulated probabilities (CSV)
```
