# CLAUDE.md — operating context for Claude Code

## Project
Probabilistic model for the 2026 World Cup (see README.md for architecture).
Owner: Simone. Working language: English everywhere (code, comments, docs).
Documentation is an explicit project priority: every new module needs a
docstring with the mathematical model, every design choice goes in docs/METHODOLOGY.md.

## Current state
- **Deployed model:** proprietary Elo → Elo-driven Dixon-Coles → tournament Monte
  Carlo, plus three gated tilts: **fatigue**, **altitude venue tilt**, and the
  **altitude-in-Elo** correction. Everything else tested was cut or kept as shadow.
- **Every selected variable now has a verdict.** The full decision ledger
  (admitted / probation / shadow / rejected / null, with OOS numbers) lives in
  **docs/DECISIONS.md**; the mathematical narrative in **docs/METHODOLOGY.md**.
- **Gate:** PRIMARY = WC+Euro (decides) + ROBUST = Copa/AFCON/Asian/Gold Cup (veto
  only). The engine generalises (better OOS outside Europe); the confederation gap
  is narrowly about *cross*-confederation matchups, which barely exist to learn from.
- **Live since 2026-06-13:** results.csv refreshed from upstream; group + KO outcomes
  conditioned. Canonical refresh = `scripts/10_live_update.py` (re-sims, re-grades,
  re-exports; also fetches live odds via 41/42/43, non-fatal without ODDS_API_KEY).
- **Site v2 (broadcast rewrite):** live track record up top, next-match 1X2/12 vs
  market, custom "climb to the trophy" bracket + pick-winner sandbox, per-match score
  heatmaps, per-team driver breakdown, dynamic model-vs-market with movement. Refresh
  via `scripts/05_export_site_data.py` after every simulation.

## Backlog (numbering stable; closed items are in docs/DECISIONS.md)
6b. **Player layer v3:** npxG once a gateable pre-2018 source exists (penalty bonus
   already REJECTED; squad+age already deployed; FBref np-goals subsumed).
7. **Cohesion graph** (most expensive, last): shared career minutes per pair
   (Transfermarkt), slow decay; feature = mean density over the expected XI. Two
   cheaper cohesion proxies already failed (see DECISIONS).
9. **Report:** three-way comparison (model vs BetMGM 11/06 vs Klement) + live
   calibration, after the final.
10. **Elite-exposure gate (2030 candidate):** live-2026 diagnostic (scripts/archive/46)
   shows teams with few trailing-4y matches vs 1800+ opponents underperform the model
   (r=+0.29 with live residual; Turkey/Uzbekistan/Jordan archetype). Gate an
   exposure-based rating shrinkage on WC 1994-2022, same protocol as archive/44.

## Repo layout (reorganised 2026-06-28)
- `scripts/` = the LIVE PIPELINE only (~17 scripts that reproduce the forecast + site):
  01 Elo, 08 bootstrap, 19 fatigue, 22/23 altitude, 04 sim, 09 golden boot, 36 heatmaps,
  38 drivers, 07 ablations, 25 shadow, 31 replay, 10 live-refresh (CANONICAL), 05 export,
  41 odds-fetch, 42 next-matches, 43 bracket-grids, 15 benchmark (called by 10).
  See scripts/README.md for the run order.
- `scripts/archive/` = completed one-off experiments (rejected/shadow/superseded gates),
  kept for the record; their outputs/ artifacts stay (site still reads some). Index +
  verdicts in scripts/archive/README.md. All scripts are standalone — run from repo root.
- Numbering is stable; an "scripts/NN" reference may now live in scripts/archive/.

## Conventions
- Point-in-time ALWAYS: never use ratings/data later than the prediction date.
- Hyperparameters (xi, ...) chosen on predictive backtest log-loss, never in-sample.
- Outputs are distributions; evaluating the model on single correct picks is forbidden.
- Fixed seeds in simulations for reproducibility.
- BENCHMARKS ARE NOT INPUTS (Simone, 2026-06-25): the market (BetMGM/Shin) and Klement
  are yardsticks we score AGAINST — never model ingredients. Anchoring to them is unfair:
  (1) circular (can't be scored against what you copied), (2) un-auditable oracle vs our
  transparent mechanisms, (3) un-gateable (no archived historical odds). Market divergence
  is a diagnostic only. Ingredients must be mechanistic, gateable, and independent of the
  benchmarks. (Full rationale in docs/DECISIONS.md; the retired player-Elo design
  note lives in git history.)
- COMM RULE (Simone, 2026-06-16): football metaphors only in site copy (trials/scouting,
  earning a place in the squad) — not weather-forecast / courtroom analogies.
- SECRETS: the Odds API key lives in `.secrets/odds_api_key` (gitignored) — NEVER commit
  it. Scripts read `os.environ.get("ODDS_API_KEY") or Path(".secrets/odds_api_key")...`.

## External benchmarks (recorded 2026-06-11, do NOT overwrite)
- BetMGM outright: Spain +450, France +500, England +700, Brazil +800,
  Portugal/Argentina +900, Germany 14-1, Netherlands 20-1, USA 50-1
- Golden Boot: Mbappé ~+550/600, Kane ~+650/700, Messi ~+1100/1200, Haaland ~+1300/1400
- Kalshi: USA ~51% to win group D
- Klement: Netherlands champions (final vs Portugal), England and Spain out in semis
