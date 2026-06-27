# Player-Elo as a cross-confederation bridge — design note

Author: design draft for Simone, 2026-06-25. **No code yet** — this is the
feasibility/architecture note requested before committing the effort (backlog
task #7). Decision gate at the end.

## 1. The problem it targets

National-team Elo is **confederation-isolated**. A side's rating is built almost
entirely against opponents from its own continent (UEFA teams play UEFA
qualifiers, CONMEBOL plays CONMEBOL). The only bridges Elo has across
confederations are sparse, low-K friendlies and the occasional intercontinental
tournament. So the rating answers "how good are you *within your pool*", which is
weak evidence for a neutral-site match against a side from another pool.

This is exactly the residual we could not fix for **Turkey** (a UEFA-internal
résumé, then lost neutral-site games to Australia/AFC and Paraguay/CONMEBOL) and
part of why the market never trusted **Ecuador** as highly as our Elo did. The
altitude-in-Elo fix addressed Ecuador's venue inflation (−18); it did nothing for
the confederation-comparability gap, which is the bigger, unaddressed bias.

## 2. The idea (Simone's): an opponent-adjusted player rating, bridged across leagues

Club football is a **global mixing bowl**. A Korean, an Ecuadorian and a German
all play, week in week out, against common opponents — and the leagues themselves
are linked by continental club cups (UCL, Libertadores, …) and by transfers. So a
**squad-aggregate player rating** carries information that national-team Elo
structurally cannot: a continent-bridging measure of how good these specific
players are on a common scale.

Two hard requirements distinguish this from the **football-capital** block we
already **removed** (clubelo squad strength), whose only signal turned out to be
clubelo's Europe bias:

1. **Opponent-adjusted *within* league.** A player's raw match rating must be
   discounted for the strength of the opponent he faced (and credited for facing
   strong ones). "Plays in a big league" must not by itself mean "good".
2. **Fairly bridged *across* leagues.** The per-league scales must be put on a
   common footing using real cross-league signal, not an assumed hierarchy.

If either is missing, the feature re-imports the capital bias and fails the gate.

## 3. Architecture

### 3.1 Per-match player contribution
Source candidates (in order of preference / effort):
- **FBref / StatsBomb**: per-match xG, xA, np-xG, progressive actions, minutes.
  Build a per-90 contribution score from these (objective, not a pundit "rating").
- **WhoScored / SofaScore ratings**: a ready composite, but opaque and partly
  subjective — usable as a cross-check, not the primary.
- **Transfermarkt market value**: not a performance measure but a strong,
  continent-spanning *prior* — useful for the bridge (§3.3) and as a shrinkage
  target, never as the signal itself.

### 3.2 Opponent adjustment (the within-league core)
Treat each player's per-match contribution as a noisy observation and fit an
**opponent-adjusted player rating** — the football analogue of adjusted
plus-minus / a Bradley–Terry on contributions:

    contribution_{p, match} ≈ skill_p − defense_{opponent} + home + ε

solved as a ridge-regularised linear model (or an online Elo update per match,
which is what makes it a "player Elo"). Regularisation shrinks low-minutes players
to a league-mean prior. Output: `skill_p` on a per-league scale, plus a
`defense_team` by-product.

### 3.3 Cross-league bridge (the part that must not cheat)
Put the per-league `skill` scales onto one axis using only real linkage:
- **Continental club competitions**: UCL/Europa, Copa Libertadores/Sudamericana,
  AFC/CAF Champions Leagues — matches where teams (hence players) from different
  leagues meet directly. These are the anchor edges of the bridge graph.
- **Transfers**: a player's skill should be ~continuous across a mid-season or
  summer move (Player X = same person in Ligue 1 and the Premier League). Each
  transfer is a soft equality constraint linking two league scales.
- **Market value** as a weak global prior to stabilise leagues that are poorly
  connected (e.g. some AFC/CAF leagues) — explicitly flagged as a prior, with a
  sensitivity check that the result doesn't just reproduce it.

Estimate per-league offsets jointly (one global least-squares / random-effects
solve over the bridge graph). **Coverage honesty**: report, per national squad,
the share of expected-XI minutes that sit in well-bridged leagues; a squad whose
players are mostly in thinly-linked leagues gets a wider error bar (or is shrunk
to Elo), the same discipline we applied to capital's coverage.

### 3.4 Aggregation → national-team feature
For each 2026 squad, take the **expected XI** (minutes-weighted, reuse the
players.py squad logic), map club→bridged player skill, aggregate to a squad
index `squad_skill_z`. It enters the Dixon-Coles as an **extra_col** in the
hierarchical-residual design (predicting Elo residuals), exactly like fatigue.

## 4. Gating — the same bar as everything else
- Point-in-time, six-tournament pooled OOS log-loss vs the no-feature model.
- Because the payoff is specifically **cross-confederation**, add a power cut:
  intercontinental matches only (WC + Confederations Cup + intercontinental
  friendlies), where national Elo is weakest and a bridged player signal should
  help most — analogous to how the altitude block was gated on CONMEBOL qualifiers.
- **Proportionality check** (the cohesion lesson): it must not swing a favourite
  by a large amount on a non-significant coefficient. OOS improvement is necessary,
  not sufficient.
- Rule stands: if it doesn't beat the backtest, it's dropped (or kept as a shadow).

## 5. Risks / why this might still fail
- **Thin bridge.** CONMEBOL↔UEFA club matches are rare (Libertadores vs UCL never
  meet except the defunct Intercontinental/Club World Cup). The bridge may lean too
  hard on transfers + market value, sliding back toward the capital bias.
- **xG history depth.** FBref event data thins out pre-2017 and is sparse outside
  the top leagues → the backtest folds (2014, 2016) may be under-powered, like the
  player-layer's pre-2018 gating problem.
- **Cost.** This is the most expensive feature we'd attempt (FBref + Transfermarkt
  scraping, a joint estimation), in the territory of the backlog's cohesion graph.

## 6. NOT a shortcut: market-anchoring shrinkage (rejected on principle)
An earlier draft proposed shrinking each team toward the Shin-de-vigged bookmaker
prior as a cheap benchmark. **Rejected — it is methodologically unfair in this
project** (Simone, 2026-06-25). Three reasons:
1. **Circularity.** "Model vs market" is one of our headline *benchmarks*. Feeding
   the market in makes that comparison meaningless — you cannot be honestly scored
   against something you copied; the log-loss gap would close for a fake reason.
2. **Auditability.** Every other ingredient is a transparent mechanism that earns
   its place by a gate. The market is an opaque aggregate of other people's models
   and money — an oracle we cannot audit. Importing it betrays the project's thesis
   (a clean, first-principles model).
3. **Un-gateable.** The admission rule needs point-in-time improvement on six past
   tournaments, but we have no archived bookmaker odds for 2014/2016/2018. No
   historical snapshots → no gate → by the diaspora precedent it could only ever be
   a shadow, and a shadow scored against the market is still circular.

The market stays strictly a **benchmark and a divergence diagnostic** (the #5
flags) — never an input. The honest cheap proxy for cross-confederation signal is a
*mechanistic* one (e.g. continental-club-competition results as bridge edges, §3.3),
not the bookmaker.

## 7. Recommendation
1. Build a **player-Elo v0** using the cheapest *mechanistic* opponent-adjusted
   signal (FBref np-xG + a ridge opponent adjustment; market value only as a weak,
   disclosed bridge prior, never as the signal). Gate it on the intercontinental
   cut **and** the six-tournament pooled OOS log-loss.
2. Keep it a **shadow** (scripts/25) until it clears the gate with *proportional*
   movement — capital and cohesion are the cautionary precedents (OOS improvement is
   necessary, not sufficient).
3. Only if v0 survives does the full §3.3 league-bridge earn its (expensive) build.

Net: the idea is sound and aims at our one unaddressed bias, but the path is a
minimal *mechanistic* player-Elo gated honestly — explicitly NOT market-anchoring,
which fails on circularity, auditability and gateability. Do not start with the
expensive scrape, and do not reach for the bookmaker as a feature.

## 8. Staged construction ladder (simple → elaborate)

Build the ladder one rung at a time; **each rung is independently gated** (point-in-
time, six-tournament pooled OOS log-loss + an intercontinental power cut + the
proportionality check). Climb only when a rung shows real, proportional signal —
stop at the first that pays, or abandon if an early rung already says "no player
aggregate survives." Effort is cumulative; a later rung reuses the earlier scaffold.

### Stage 0 — Squad-value floor probe  (~½ day)
- **Construction**: squad aggregate = minutes-weighted mean of a per-player static
  index (Transfermarkt market value, log-scaled). No opponent adjustment, no time.
- **Bridge across leagues**: none — value is already a global currency (its weakness).
- **Data**: Transfermarkt squad pages (already partly scraped for `players.py`).
- **Gate expectation**: likely REJECTED/shadow, same Europe-bias failure mode as the
  removed capital block. Purpose is a *floor*: if even this shows nothing, the whole
  idea is probably dead; if it shows biased signal, we know the bias to beat.
- **Note**: market *value* (a player attribute) ≠ the betting market (our benchmark) —
  using it as a feature is allowed, but it is the least auditable rung; treat as a probe.

### Stage 1 — Self-computed club-Elo, players inherit their club  (~2 days)
- **Construction**: run our Elo machinery on **club** matches → one global club rating
  pool. Each player gets their current club's Elo; squad = mean over the expected XI.
- **Bridge across leagues**: FREE and *mechanistic* — a single global pool fed by
  domestic leagues + continental cups (UCL, Libertadores, AFC/CAF CL) + the Club World
  Cup links every league through the results graph. No market value needed.
- **Data**: club match results (same shape as our national results.csv; scrapeable).
- **Why it's the real floor**: it is "clubelo, but self-built and honestly bridged,"
  so it directly tests whether the *capital* idea works once the Europe-bias floor is
  removed by a fair global results graph rather than clubelo's coverage.

### Stage 2 — Player-Elo: minutes, transfer continuity, recency  (~3 days)
- **Construction**: a player's rating = minutes-weighted, time-decayed blend of the
  club Elos they actually played for; rating **follows the player across transfers**
  (continuity) and decays so recent form dominates. Squad = expected-XI aggregate.
- **Bridge**: inherited from Stage 1's global club pool; transfers add soft continuity
  edges (same player = same skill across a move).
- **Data**: club results + per-player appearances/minutes (FBref/Transfermarkt).
- **Step up**: distinguishes a fringe player at a big club from its starter, and a
  player's *current* level from his club's historical pedigree.

### Stage 3 — Opponent-adjusted contributions (the true "player Elo")  (~1–2 weeks)
- **Construction**: per-match player contribution from event data (np-xG, xA,
  progressive actions per 90), fit the adjusted model `contrib ≈ skill_p −
  defense_opp + home` (ridge / online Elo). This is §3.2 — credit for facing strong
  defences, not just for playing in a strong league.
- **Bridge**: §3.3 — continental-cup matches as anchor edges + transfer continuity;
  market value only as a weak, disclosed stabiliser for thinly-linked leagues.
- **Data**: FBref/StatsBomb event data (thin pre-2017 → weak early backtest folds).
- **This is the expensive rung** — only build if Stage 1/2 already beat no-feature.

### Stage 4 — Role-aware + uncertainty-propagating  (research)
- **Construction**: separate **attack vs defence** player ratings, weight by position
  (a striker and a keeper move λ differently), and propagate per-player uncertainty
  (low minutes → wide) into the squad aggregate and into the existing parameter
  bootstrap, widening forecast tails honestly.
- **Most elaborate**: effectively a small hierarchical player model feeding the DC.
- Build only if Stage 3 is admitted and the marginal gain looks real.

### Decision rule across the ladder
1. Stage 0 + Stage 1 first (cheap). If neither beats no-feature on the gate, **stop** —
   document the null like capital, keep as shadow. The confederation bias may simply
   not be recoverable from squad data at our backtest depth.
2. If Stage 1 shows proportional signal, do Stage 2 (cheap refinement).
3. Escalate to Stage 3 only on a positive Stage 1/2 — never speculatively.
4. Stage 4 is a luxury, contingent on Stage 3.

Throughout: keep it a **shadow** (scripts/25) until admitted with proportional movement,
and never let it dominate a favourite on a non-significant coefficient (the cohesion lesson).

## 9. Feasibility verdict on the data (investigated 2026-06-25) — BLOCKED

Before building, we checked what bridging evidence actually exists. The verdict: the
**cross-confederation club bridge does not exist in open data**, so the rung that
would fix the Turkey/Ecuador bias cannot be built honestly today.

- **Self-built route (Stage 1).** The best open global club dataset
  (xgabora/Club-Football-Match-Data-2000-2025: 230,557 matches, 38 divisions across
  every confederation — ARG, BRA, CHN, JAP, MEX, USA + Europe) is **domestic-only**
  (football-data.co.uk). No continental/intercontinental matches → the 38 leagues are
  **disconnected pools**; clubs from different countries never meet, so a self-built Elo
  cannot put them on one scale. Its own pre-computed Elo proves the point: the European
  pyramid is bridged sensibly (E0 2062 → SP1 2000 → … → lower leagues) but **ARG, BRA,
  CHN, JAP, MEX, USA are NaN — unrated.** The exact confederations we needed are blank.
- **Rating route (also Stage 1).** The one source that DOES rate non-European clubs
  globally is footballdatabase (fdb, 89% coverage) — and we already tested it: it is the
  **removed capital block** (failed the gate, signal was coverage/geography bias). So the
  global club-strength signal has been tried and rejected.
- **Opponent-adjusted route (Stage 3).** Needs global per-match event data; FBref is
  Europe-only and season-level. A Korean/Ecuadorian starter outside the top-5 leagues has
  no event row, so the coverage gap is exactly where the bias lives.

**Why this is fundamental, not a sourcing gap.** Confederations barely play each other at
*any* level — that is the same root cause as the national-team bias, reproduced at club
level. There is simply little match evidence linking CONMEBOL/AFC/CONCACAF to UEFA. The
**market** prices the cross-confederation gap via opinion and money, not match evidence —
which is precisely why it is a benchmark and (correctly, §6) not an input.

**Recommendation: do NOT build the cross-confederation player-Elo now.** Keep this note as
the approved spec. Revisit only if a genuinely bridged, gateable global source appears
(e.g. a clean Club World Cup + continental-cup results corpus, still a *thin* bridge per
§5). The honest current stance: the confederation bias is **largely irreducible from match
evidence**, and we surface it as a divergence/diagnostic rather than pretend to fix it.
A within-Europe squad-quality refinement (FBref) remains possible but does **not** address
the bias and would, like capital, likely be subsumed — low priority.
