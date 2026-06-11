# CLAUDE.md — operating context for Claude Code

## Project
Probabilistic model for the 2026 World Cup (see README.md for architecture).
Owner: Simone. Working language: English everywhere (code, comments, docs).
Documentation is an explicit project priority: every new module needs a
docstring with the mathematical model, every design choice goes in docs/METHODOLOGY.md.

## Status (2026-06-11)
- [x] Proprietary Elo, validated (Spain 2216 #1, consistent with the market)
- [x] Elo-driven Dixon-Coles, WC2022 backtest: LL 1.062 vs 1.099 uniform, well calibrated
- [x] Group-stage Monte Carlo (FIFA tiebreakers + best thirds), 20k sims, output in outputs/
- [x] Host home advantage: `WC2026_HOSTS` invariant in data.py; was already active
      (raw data correct). USA stays at 23% P1: the gap vs Kalshi's 51% is in the Elo
      (USA 1824, last in its group; eloratings.net agrees: USA #39). See METHODOLOGY.md.
      → residual block #5
- [x] Official group labels: anchored to the FIFA draw via OFFICIAL_GROUP_ANCHORS
      (kickoff order would swap C/D; USA is in group D, matching Kalshi)
- [x] Knockout bracket: official matches 73-104, thirds allocated by deterministic
      matching on published slot constraints (literal 495-row Annex C not transcribed,
      see METHODOLOGY.md), ET ~Poisson/3, shootout logit fitted on shootouts.csv.
      v1: Spain 24.8% champion, Argentina 18.8%, France 10.1% (vs BetMGM 18/17/12 raw)
- [x] Results site: site/index.html (static, self-contained; data via
      scripts/05_export_site_data.py — re-run it after every new simulation)
- [x] xi tuned (scripts/06): 0.0027, but the curve is FLAT (pooled LL 0.9952-0.9965,
      paired t=0.45 between grid extremes) — xi not identified, chosen at plateau
      centre. Honest null result documented in METHODOLOGY.md. Backtest LL now 1.060.
- [x] Ablations (scripts/07) + data-first site v2: host advantage doubles USA group
      odds (12.5%->23.7%), xi a rounding error; site leads with data choices,
      methodology demoted to collapsibles
- [ ] Backlog below

## Backlog (in order; numbering stable, #1-4 closed — see Status)
5. **Residual blocks** (scraping — needs full network access, hence Claude Code):
   - football capital: FBref top-5-league minutes, clubelo.com club Elo weighted by
     minutes, CIES academy ranking
   - climate: open-meteo history for venue+kickoff vs players' club cities (mismatch
     score); venue altitude (CDMX 2240m, Guadalajara 1566m)
   - diaspora: US census ancestry per metro area → de-facto home support
   - fatigue: club-season minutes per player (FBref)
   They enter as extra_cols in the DC (hierarchical-residual design: they predict
   Elo residuals). Rule: a feature that doesn't improve backtest log-loss is dropped.
6. **Player layer**: historical per-player goal share + penalty takers from
   goalscorers.csv; xG/90 and expected minutes from FBref. Multinomial allocation
   conditioned on simulated team goals → Golden Boot and most-distinct-scorers
   distributions.
7. **Cohesion graph** (most expensive feature, last): shared career minutes per pair
   (Transfermarkt), slow decay; feature = mean density over the expected XI.
8. **Parameter uncertainty**: bootstrap DC parameters, sample them at the start of
   each simulation.
9. **Report**: three-way comparison (model vs BetMGM 11/06 odds vs Klement) + live
   calibration. Played matches enter via `fixed_results` in simulate_group_stage.

## Conventions
- Point-in-time ALWAYS: never use ratings/data later than the prediction date.
- Hyperparameters (xi, ...) chosen on predictive backtest log-loss, never in-sample.
- Outputs are distributions; evaluating the model on single correct picks is forbidden.
- Fixed seeds in simulations for reproducibility.

## External benchmarks (recorded 2026-06-11, do NOT overwrite)
- BetMGM outright: Spain +450, France +500, England +700, Brazil +800,
  Portugal/Argentina +900, Germany 14-1, Netherlands 20-1, USA 50-1
- Golden Boot: Mbappé ~+550/600, Kane ~+650/700, Messi ~+1100/1200, Haaland ~+1300/1400
- Kalshi: USA ~51% to win group D
- Klement: Netherlands champions (final vs Portugal), England and Spain out in semis
