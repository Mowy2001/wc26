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
- [x] Altitude block (scripts/22-23): reframed by Simone as habitual-altitude
      acclimatisation, tested on CONMEBOL qualifiers (where altitude varies). OOS
      -0.0225, t=-2.42 (STRONGEST block), b=0.134/km, ADMITTED. Deployed as city tilt
      at Mexico City/Zapopan: Mexico 2.1->3.2% champion. tilts.load_city_tilt.
- [x] Diaspora (scripts/24): DECLARED SHADOW BET, never in official model (can't be
      gated — no past US World Cup). sqrt(US foreign-born) index, city tilt 0.06 at US
      venues, Mexico strongest. Shadow scoreboard (scripts/25): all variants within
      0.019 LL over 20 matches — honest "one tournament can't decide". Site: model lab.
- [x] Cohesion (scripts/26): club-concentration proxy. Passes OOS by a hair (-0.0047)
      but t=-0.60 AND deployment swings Spain -9.6pp (non-sig feature dominating the
      favourite) -> EXCLUDED from official model, kept as SHADOW (scripts/25) like
      diaspora. Lesson: OOS-improvement necessary not sufficient; needs proportionality.
- [x] Climate club-country re-test (scripts/27): REJECTED again (+0.0165 OOS, t=+2.09);
      Qatar-2022 AC fold poisons it, non-Qatar a wash. CIES academies dropped (fuzzy,
      collinear). Heat closed: doesn't show even with the correct proxy.
- [x] Altitude-IN-ELO correction (scripts/35, 2026-06-25): altitude-aware Elo home
      advantage so home wins at altitude earn fewer rating points (the Ecuador
      residual, distinct from the venue tilt). Canonical 6-tournament gate FLAT
      (blind, like the tilt); CONMEBOL-at-sea-level power cut (n=1226) cleanly
      U-shaped, optimum gamma=75 Elo/km, OOS -0.0042, paired t=-1.91 -> ADMITTED
      ON PROBATION. DEPLOYED 2026-06-25 (Simone): elo.py ALT_GAMMA=75, canonical
      build (scripts/01) is altitude-aware; full live chain re-run (08/10/09/36/31/05).
      Effect at 2026-06-11: Ecuador -18, Colombia -5, Brazil +10, Uruguay +13
      (Bolivia -89 but not in WC26). Deliberately moves the live model off the
      frozen June-11 baseline (deltas now mix model-change + results) -- accepted.
      Honest caveat: only -18 for Ecuador, most of their rating is genuine; their
      collapse is variance/squad like Turkey's.
- [x] Per-match score-distribution HEATMAP (scripts/36 + site): every group fixture
      exports its pre-match DC score matrix (frozen ratings + deployed tilts), shown
      as a clickable goals x goals heatmap with modal scoreline, W/D/L split and the
      actual result ringed. Wired on the live recent-matches rows; showHeat(h,a) is
      reusable for more entry points (slider/groups TODO).
- [x] Elo-weight grid search (scripts/37, 2026-06-25): tuned HOME_ADV, global K mult,
      friendly K xi-style on the 6-tournament pooled OOS LL. FLAT PLATEAU, no axis
      beats default beyond noise (home_adv best 40 t=-0.69; k_mult flat t=-0.19;
      friendly_k best 30 t=-0.75). Only significant signal is a negative control:
      friendly K too LOW (5-10) clearly hurts (t=+2.1) -> friendlies carry signal,
      don't zero them. Verdict: keep eloratings convention, deploy nothing.
- [x] Bracket DUAL VIEW (2026-06-25): collect_bracket now also emits `adv` = P(modal
      team WINS that tie); threaded through replay (31) + live (10) snapshots + export.
      Site toggle "Reach slot | Win the tie" on the predicted bracket, mode-aware legend.
      Full replay rebuilt so every slider snapshot carries adv. (Watch: a leftover `}`
      from the legend edit silently killed the forecast IIFE -> all downstream renderers
      blank; caught by dump-dom counts. Always re-verify renderer counts after app.js edits.)
- [x] Market MORE VISIBLE + flags (#5, 2026-06-25): market section shows model-vs-market
      divergence chips (>=5pp), a "Sharpest disagreements" headline (Argentina model-loves,
      France model-cold), "toss-up" badges on coin-flip fixtures (no W/D/L > 42%, from
      match_dists), and "wide-open" badges on tight groups. All client-side from data.js.
- [x] Player-Elo cross-confederation DESIGN NOTE + staged ladder (#7,
      docs/player_elo_design.md, 2026-06-25): opponent-adjusted player rating bridged
      across leagues to fix the confederation-isolation bias (Turkey/Ecuador). Market-
      anchoring REJECTED on principle (benchmarks aren't inputs — see Conventions).
      §8 ladder, each rung independently gated: Stage 0 squad-value floor probe ->
      Stage 1 self-built global club-Elo (players inherit club, bridged FREE via
      continental cups) -> Stage 2 minutes+transfer-continuity+recency player-Elo ->
      Stage 3 opponent-adjusted event contributions (the expensive rung) -> Stage 4
      role-aware + uncertainty. Rule: build Stage 0/1 first; escalate only on a
      positive, proportional gate; shadow until admitted.
- [x] Player-Elo FEASIBILITY check (2026-06-25, docs §9): BLOCKED. Investigated the best
      open global club dataset (xgabora, 230k matches, 38 divisions incl. ARG/BRA/CHN/
      JAP/MEX/USA) — it's DOMESTIC-ONLY, leagues are disconnected pools, and its own Elo
      leaves the non-European leagues NaN. The only global club-strength source (fdb) is
      the already-REJECTED capital. The cross-confederation bridge does not exist in open
      data because confederations barely play each other at ANY level (same root cause as
      the national bias). Verdict: do NOT build now; bias is largely irreducible from match
      evidence; keep the spec, surface as diagnostic. Market prices it via opinion (why
      it's a benchmark, not an input).
- [x] Per-team DRIVER breakdown (#4, scripts/38 + site, 2026-06-25): each team's
      P(top 2) split ADDITIVELY into intrinsic strength (P in an average group via a
      per-team mini-sim vs 3 field-mean sides), draw luck (actual vs average group),
      and tilts (deployed altitude+fatigue) — strength+draw+tilt=total exactly. Eve
      (unconditioned). Reveals Mexico 96% = 66 strength +24 draw +7 venue; Canada 92%
      = 53 +37 +2 (easiest group); Argentina 94% almost all quality. Site section
      "What's earning each team its place": stacked bars (quality/draw/venue&freshness)
      + hatched drag for tough groups. team_drivers.json -> data.js.
- [ ] All selected variables now have verdicts. NEXT: match-by-match SLIDER
      (groups+bracket) + football-metaphor site rewrite (see direction memo).
- [ ] WANT (Simone, 2026-06-16): a time SLIDER on groups + bracket showing the
      situation match by match (data already in outputs/history/timeline.csv +
      per-snapshot bracket — need to start saving bracket history too).
- [ ] COMM RULE (Simone, 2026-06-16): football metaphors ONLY. Replace the
      weather-forecast / bouncer / courtroom analogies with football ones
      (trials/scouting, earning a place in the squad, etc.).
- [ ] Backlog below

- [x] Cohesion attempt #2 — squad CONTINUITY (scripts/39, 2026-06-26): fraction of a
      team's squad retained from its previous tournament (France 18/26 settled, Spain
      10/26 rebuilt). Gated on current Elo-DC, 6 tournaments: REJECTED, OOS +0.0029
      worse, t=+1.08, b=-0.025. Turkey 2nd-most-settled yet collapsed. Two cohesion
      proxies now both fail; only unscrapeable Transfermarkt pairwise-minutes untested.
- [x] CIES academies SETTLED (2026-06-26, docs METHODOLOGY): investigated — public CIES
      tables list only ~top 15 exporters, recent snapshot, no per-country history ->
      un-gateable point-in-time; self-built expatriation proxy blocked (clubelo Europe-
      only, fdb has no country); signal collinear with Elo anyway. Stays OUT, evidenced.

- [x] Broadened gate (scripts/40, 2026-06-28): added a ROBUST cross-confederation gate
      (Copa/AFCON/Asian Cup/Gold Cup, n=1213) beside the WC+Euro PRIMARY. Big finding:
      the model GENERALISES — base OOS LL is BETTER outside Europe (ROBUST 0.9315 vs
      PRIMARY 0.9856); Euro is the hardest tournament, Gold/Asian cups easiest. So the
      confederation issue is narrowly about CROSS-confederation matchups, not non-European
      competence. xi plateau holds; altitude-in-Elo not vetoed (flat on ROBUST, n=166).
      New rule: Elo/DC features pass PRIMARY (decides) + not contradicted by ROBUST (veto);
      squad blocks can't extend (no continental squad files).

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

- [x] Capital REMOVED from the model (2026-06-18, Simone): its signal was clubelo's
      Europe bias (floor = signal); fair global source (footballdatabase, 54->89% cover)
      fails the gate; 'no capital' is better AND unbiased. Deployed tilts = fatigue +
      altitude only. fdb pipeline kept for the model lab (capital now a shadow).
      Routine -> twice daily. Golden Boot subtitle de-staled.

## Repo layout (reorganised 2026-06-28)
- `scripts/` = the LIVE PIPELINE only (14 scripts that reproduce the forecast + site):
  01 Elo, 08 bootstrap, 19 fatigue, 22/23 altitude, 04 sim, 09 golden boot, 36 heatmaps,
  38 drivers, 07 ablations, 25 shadow, 31 replay, 10 live-refresh (CANONICAL), 05 export.
  See scripts/README.md for the run order.
- `scripts/archive/` = 26 completed one-off experiments (rejected/shadow/superseded gates),
  kept for the record; their outputs/ artifacts stay (site still reads some). Index +
  verdicts in scripts/archive/README.md. All scripts are standalone — run from repo root.
- Numbering in this file/backlog is stable; an "scripts/NN" reference may now live in
  scripts/archive/ (see that README to locate it).

## Conventions
- Point-in-time ALWAYS: never use ratings/data later than the prediction date.
- Hyperparameters (xi, ...) chosen on predictive backtest log-loss, never in-sample.
- Outputs are distributions; evaluating the model on single correct picks is forbidden.
- Fixed seeds in simulations for reproducibility.
- BENCHMARKS ARE NOT INPUTS (Simone, 2026-06-25): the market (BetMGM/Shin) and Klement
  are yardsticks we score AGAINST — never model ingredients. Anchoring to them is unfair:
  (1) circular (can't be scored against what you copied), (2) un-auditable oracle vs our
  transparent mechanisms, (3) un-gateable (no archived historical odds). Market divergence
  is a diagnostic only (the #5 flags). Ingredients must be mechanistic, gateable, and
  independent of the benchmarks. See docs/player_elo_design.md §6.

## External benchmarks (recorded 2026-06-11, do NOT overwrite)
- BetMGM outright: Spain +450, France +500, England +700, Brazil +800,
  Portugal/Argentina +900, Germany 14-1, Netherlands 20-1, USA 50-1
- Golden Boot: Mbappé ~+550/600, Kane ~+650/700, Messi ~+1100/1200, Haaland ~+1300/1400
- Kalshi: USA ~51% to win group D
- Klement: Netherlands champions (final vs Portugal), England and Spain out in semis

## IMPORTANT: Sound Notification

After finishing responding to my request or running a command, run this command to notify me by sound:

```bash
afplay /System/Library/Sounds/Funk.aiff
```