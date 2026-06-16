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
- [x] Parameter bootstrap (#8, scripts/08): B=100, effect on P(champion) ±0.07pp =
      MC noise. Kept on (correct integral); honest reading: model risk is in the data,
      not the fit
- [x] Player layer v1 (#6, src/players.py + scripts/09): decayed scorer shares +
      17.3% debutant bucket. Mbappé 15.9%/Kane 13.6% vs market 14.8%/12.9%. Declared
      artifact: Valencia 17.7% top (no age/minutes data — waits for FBref block)
- [x] Live-update scaffold (scripts/10): fixed_results from refreshed results.csv,
      frozen ratings/params; KO conditioning TODO. Site v3: lineage diagram, bootstrap
      card, Golden Boot section
- [x] Capital block #5a (scripts/11-12): squads (wiki) x clubelo, LOTO backtest
      -0.0012 OOS LL (t=-0.32), ADMITTED ON PROBATION, beta=0.0328 tilt in all sims.
      Europe-only source bias declared (taxes Brazil/Mexico/USA). Site v4 card.
- [x] Climate block #5b (scripts/13-14): REJECTED — OOS LL +0.0218 worse (t=+2.61),
      Qatar 2022 fold explodes (AC + winter + passport-is-not-a-climate). Coefficient
      zero; city-aware tilt mechanism kept. Benchmark harness (#9 partial): Shin
      de-vig + live match scorer (scripts/15); site market section now margin-free
- [x] Player-layer gates (scripts/16, 18): REVERSED after fixing a metric bug (full
      bucket mass per unknown scorer -> degenerate). Corrected: squad filter + age
      alpha=0.1 ADMITTED (2.7266 vs 2.9122, 3/3 folds), DEPLOYED in 09. FBref club
      np-goals blend helps vs v1 but is subsumed by squad+age (+0.004) -> not
      deployed. Golden Boot now: Mbappe 30.4%, Kane 16.6%, Messi 1.2% (age 39) vs
      market 8% — boldest disagreement on record
- [x] Fatigue block #5d (scripts/19): squad-mean club minutes on top of capital;
      b=-0.028 stable across both folds, OOS -0.0012 -> ADMITTED on probation.
      France pays most (-2.4pp). Tilts centralised in src/wc26/tilts.py
- [x] LIVE since 2026-06-13: results.csv refreshed from upstream; Mexico 2-0 RSA
      (P=80%) and KOR 2-1 CZE (P=45%) conditioned; running LL 0.51 vs 1.10 uniform.
      Canonical refresh = scripts/10
- [x] Penalty-taker bonus (scripts/20): REJECTED — double-counts penalties already
      in the goal shares (+0.0054 OOS). Live refresh AUTOMATED: cloud routine every
      4h through 07-19 (trig_01EtNmHHV3iS3iuvCFa1R6zp), pushes to main on new results
- [x] KO live conditioning (deadline beaten): real knockout outcomes imposed at the
      pair level (wc2026_played_ko; shootout winners from shootouts.csv); R32 routine
      keeps working through the final. thirds_override ready for FIFA's real
      allocation after 06-27
- [x] Baseline frozen + forecast timeline (2026-06-13): outputs/history/
      baseline_eve.csv = final model (v3) on the June-11 information set, immutable;
      every live refresh appends a snapshot to outputs/history/timeline.csv.
      Site: white baseline notch + movement deltas on the champion race
- [x] Capital v2 minutes-weighting (scripts/21): ADMITTED but near-null (corr v1/v2
      0.997, OOS -0.0013 vs -0.0008). Deployed, beta 0.0328->0.0353; build logic
      shared in src/wc26/capital.py. Squads are mostly regular starters -> ~no effect
- [x] Predicted KO bracket on the site (collect_bracket in simulate): modal occupant
      + probability per slot, live-conditioned, converges to champion. 16/72 played
- [ ] DECISION 2026-06-16 (revised same day): FINISH testing the selected variables
      after all — diaspora and cohesion go through the gate (may be rejected; that's
      fine). Then present data + their impact on the site.
- [ ] WANT (Simone, 2026-06-16): a time SLIDER on groups + bracket showing the
      situation match by match (data already in outputs/history/timeline.csv +
      per-snapshot bracket — need to start saving bracket history too).
- [ ] COMM RULE (Simone, 2026-06-16): football metaphors ONLY. Replace the
      weather-forecast / bouncer / courtroom analogies with football ones
      (trials/scouting, earning a place in the squad, etc.).
- [ ] Backlog below

## Backlog (in order; numbering stable, #1-4, 5a, 6, 8 closed — see Status)
5. **Residual blocks** (scraping — needs full network access, hence Claude Code):
   - football capital v2: FBref top-5-league minutes to weight the clubelo tilt
     within squads (v1 admitted on probation, see Status); CIES academy ranking
   - climate v2 (v1 REJECTED, see Status): club-city acclimatization instead of
     home-country; needs the FBref/club-city block first
   - diaspora: US census ancestry per metro area → de-facto home support
   They enter as extra_cols in the DC (hierarchical-residual design: they predict
   Elo residuals). Rule: a feature that doesn't improve backtest log-loss is dropped.
6b. **Player layer v3** (rest of #6): npxG once a gateable pre-2018 source exists
   (penalty bonus already REJECTED). Capital v2 minutes-weighting done (near-null).
7. **Cohesion graph** (most expensive feature, last): shared career minutes per pair
   (Transfermarkt), slow decay; feature = mean density over the expected XI.
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
