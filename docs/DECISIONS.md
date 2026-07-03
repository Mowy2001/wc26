# Decision log — what earned its place, and what didn't

This is the ledger of every modelling decision, with its verdict. The rule
throughout: an ingredient is **admitted only if it improves predictive
(out-of-sample) log-loss by an amount proportionate to its statistical support** —
and benchmarks (market, rival forecasts) are *never* inputs, only yardsticks.

The gate has two panels:
- **PRIMARY** — World Cups + Euros, point-in-time. Decides.
- **ROBUST** — Copa América / AFCON / Asian Cup / Gold Cup. Can veto, can't admit
  (no continental squad files). Big finding: the engine is actually *better* OOS
  outside Europe (ROBUST 0.9315 vs PRIMARY 0.9856) — the confederation issue is
  narrowly about *cross*-confederation matchups, not non-European competence.

Tags: **ADMITTED** (deployed) · **PROBATION** (deployed, near-null) · **SHADOW**
(in the model lab only, never official) · **REJECTED** · **NULL** (honest non-result).

---

## The engine — validated

- **Proprietary Elo** — from-scratch eloratings.net replica over the full history;
  gives us point-in-time ratings at any date. Spain #1 (2216), consistent with the
  market and eloratings.net.
- **Elo-driven Dixon-Coles** — Poisson goals as *functions of Elo* (no free per-team
  parameters; international data is too sparse), with the ρ low-score correction.
  WC2022 backtest, point-in-time: **log-loss 1.060 vs 1.099 uniform**, well calibrated.
- **Tournament Monte Carlo** — 20k sims (fixed seed), FIFA tiebreakers, 8 best
  thirds, official bracket, extra time (~Poisson λ/3), shootout logit fitted on
  historical shootouts. Parameter bootstrap integrates over fit uncertainty
  (effect on P(champion) ≈ ±0.07pp — model risk is in the data, not the fit).

## Deployed tilts (residual nudges on top of Elo-DC)

- **Fatigue** — squad-mean club minutes. b ≈ −0.028, stable across folds, OOS
  −0.0012 → **ADMITTED** (probation). France pays the most (−2.4pp).
- **Altitude venue tilt** — habitual-altitude acclimatisation, tested on CONMEBOL
  qualifiers where altitude varies. OOS −0.0225, t = −2.42 (the strongest single
  block), b = 0.134/km → **ADMITTED**. Applied at Mexico City / Zapopan; Mexico
  2.1 → 3.2% champion.
- **Altitude-in-Elo correction** — altitude-aware Elo home advantage so home wins
  in thin air earn fewer rating points (distinct from the venue tilt). Canonical
  6-tournament gate is flat (blind); CONMEBOL-at-sea-level power cut (n=1226) is
  cleanly U-shaped, optimum γ = 75 Elo/km, OOS −0.0042 → **ADMITTED** (probation).
- **Capital v2 minutes-weighting** — **ADMITTED** but near-null (corr v1/v2 0.997);
  squads are mostly regular starters, so ~no effect. (Capital itself: see below.)

## Tested and cut

- **Football capital** (clubelo club strength of each squad) — the signal was
  clubelo's **Europe bias** (the coverage floor *was* the signal); the fair global
  source (footballdatabase, 54→89% coverage) fails the gate. "No capital" is both
  better and unbiased → **REMOVED** from the model, kept as **SHADOW**.
- **Climate / heat** — **REJECTED ×2**. Home-country version: OOS +0.0218 worse
  (t = +2.61), the Qatar-2022 fold explodes (AC + winter + a passport is not a
  climate). Re-tested with club-city acclimatisation: still +0.0165 worse
  (t = +2.09). Heat closed even with the correct proxy.
- **Penalty-taker bonus** — **REJECTED**: double-counts penalties already in the
  goal shares (+0.0054 OOS).
- **Official FIFA ranking** (as the strength input, in place of our Elo) —
  **REJECTED**. Same DC model, same point-in-time training rows, only the rating
  swapped; tested on every World Cup 1994-2022. Our Elo wins all eight: pooled 493
  matches, **0.9745 vs 1.0485** (uniform 1.0986), paired t = 4.18, p ~ 3.5e-5. The
  fitted coefficient holds at ~0.75-0.82 for Elo but decays for FIFA from ~0.5 to
  ~0.14 — the goal model learns the ranking barely tracks match margins. This is
  why FIFA rankings sit on the *deliberately-excluded* list, not just as taste.
  `scripts/archive/44_fifa_vs_elo.py`, `data/external/fifa_ranking.csv`.
- **Cohesion** — **EXCLUDED ×2**. Club-concentration proxy passes OOS by a hair but
  swings Spain −9.6pp on a non-significant feature (t = −0.60) → fails
  proportionality. Squad-continuity proxy (retained from the previous tournament)
  is also worse (+0.0029, t = +1.08); Turkey was 2nd-most-settled and still
  collapsed. Both kept as **SHADOW**.
- **Diaspora** — **SHADOW** by construction: can't be gated (no past US World Cup),
  so it never enters the official model. sqrt(US foreign-born) index as a city tilt.
- **CIES academies** — **un-gateable**: public tables are a recent top-15 snapshot
  with no per-country history; self-built proxy blocked; signal collinear with Elo
  anyway. Stays out, evidenced.

## Honest non-results

- **Time-decay ξ** — tuned on predictive log-loss; the curve is **flat** (pooled LL
  0.9952–0.9965, paired t = 0.45 between grid extremes). ξ is *not identified*;
  chosen at the plateau centre. **NULL**, documented.
- **Elo-weight grid** (home advantage, global K, friendly K) — **flat plateau**, no
  axis beats the eloratings convention beyond noise. The only signal is a negative
  control: friendlies set too low clearly hurt (t = +2.1) → they carry signal,
  don't zero them.

## The data wall we did not cross

- **Cross-confederation player-Elo** — the principled fix for the confederation bias
  (Turkey, Ecuador). **BLOCKED** in open data: the only global club-strength source
  is the already-rejected capital; the best open club dataset is domestic-only with
  disconnected league pools. Confederations barely play each other at *any* level —
  the bridge doesn't exist to learn from. We keep the design spec
  (`docs/player_elo_design.md`) and surface the bias as a diagnostic instead of
  faking a fix. Market-anchoring was **rejected on principle** (benchmarks are not
  inputs).

---

*For the full mathematical narrative behind each line, see `docs/METHODOLOGY.md`.
For the experiment scripts and their outputs, see `scripts/archive/README.md`.*
