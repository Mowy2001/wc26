# Methodology — decisions and rationale

## Why Dixon-Coles and not pure ML
International football = few matches per team, heterogeneous opponents, noisy
friendlies. A well-specified statistical model with strengths anchored to Elo is more
robust than a GBM with many features (high overfitting risk). The literature (Groll
et al.) shows hybrids win precisely because they are anchored to a ranking. A possible
XGBoost variant is kept only if it beats the DC in the backtest.

## Why the Elo is computed from scratch
Point-in-time discipline: the backtest needs ratings AS THEY WERE at a given date.
Scraping eloratings.net would only give the current state. Our Elo replicates their
formula (importance K, margin multiplier, home adv 100) over the same history.

## Handling luck (central design decision)
1. Aleatoric: Poisson sampling in the Monte Carlo IS football's variance.
2. Epistemic in the data: prefer xG to realised goals where possible (mean reversion).
3. Epistemic in the parameters: bootstrap over DC parameters (backlog #8) to widen the tails.
4. Shootouts ≈ coin flip (calibrate goalkeeper margin on shootouts.csv).
Evaluation ONLY on log-loss and aggregate calibration.

## Hierarchical-residual design for the original features
Elo + squad value = baseline. The original blocks (climate, football capital, diaspora,
fatigue, cohesion) predict the RESIDUALS: "holding nominal strength fixed, what moves
the needle?". Selection: domain pre-screening + elastic net; only what improves
out-of-sample log-loss survives. Honest caveat: climate and diaspora are hard to
backtest (Qatar 2022 was air-conditioned; the diaspora effect is 2026-specific) →
conservative priors, declared as domain bets.

## 2026 host home advantage (verified 2026-06-11)
Backlog hypothesis: the USA gap between model (23% group win) and market (Kalshi ~51%)
was caused by `neutral=True` on host home matches. Falsified: the raw data already
flags the 9 host home matches (USA/Mexico/Canada at home) as `neutral=False`, and the
simulator already passed the flag to the DC. The override in `data.py` (`WC2026_HOSTS`)
is therefore a no-op today, kept as a defensive invariant against dataset refreshes.
Re-running 03 (seed 26) gives identical probabilities → home advantage was already
active, and visibly so (USA last in its group by Elo but second by P1; +9-10pp per
match vs neutral). The residual gap vs the market lives in the rating, not the venue:
our Elo has USA at 1824 (Turkey 1967, Paraguay 1915, Australia 1898); eloratings.net
on 2026-06-11 has USA #39 at 1726 with a Turkey-USA gap of 185 (ours: 143) → our Elo
is already kinder to the USA than the reference. The market prices what Elo cannot see
(squad value, host boost beyond the average home effect, diaspora): closing the gap is
expected from the residual block (backlog #5), accepted only if it improves backtest
log-loss.

## Tiebreakers and best thirds
FIFA Art. 13: points > goal difference > goals scored > head-to-head (points, GD, GF)
> fair play > drawing of lots. Fair play is unobservable pre-match → random as proxy
(equivalent in expectation). Thirds: points > GD > GF > random; the best 8 advance;
bracket allocation follows the official FIFA table (495 combinations — backlog #3).

## Knockout bracket (implemented 2026-06-11)
Official schedule, matches 73-104. Three modelling decisions:
1. **Third-place allocation**: FIFA's regulations fix one assignment per combination
   of qualified thirds (495-row annex). We could not transcribe the literal table;
   instead we compute a deterministic perfect matching against the slot constraints
   printed in the schedule (3rd A/B/C/D/F etc.), most-constrained slot first,
   alphabetical tie-break. Every assignment satisfies the published constraints; it
   may differ from FIFA's row in which *allowed* third lands on which slot. The
   approximation only shuffles thirds among slots they could legally occupy.
2. **Extra time**: independent Poisson at 1/3 of the 90' lambdas (30' at the same
   scoring intensity, DC low-score correction not re-applied — tau is a 90'
   phenomenon).
3. **Shootouts**: logit fit on the 677 historical shootouts merged with point-in-time
   Elo: P(A wins) = sigmoid(0.309·host_A + 0.677·Δelo/400). Log-likelihood -462.5 vs
   -469.3 for a pure coin flip (both terms ≈2.5σ): shootouts are *almost* a coin
   flip, with a small skill+host tilt. Knockout venues carry the host home effect
   (e.g. Mexico plays R32/R16 in Mexico City if it wins group A).

## Time-decay tuning (xi) — and an honest null result (2026-06-11)
Grid 0.0005-0.005/day backtested point-in-time on WC2014/2018/2022 + Euro
2016/2021/2024 (345 matches, scripts/06). Pooled log-loss spans 0.9952-0.9965 across
the whole grid; per-tournament optima sit at opposite edges; paired t between the
extremes = 0.45. Conclusion: xi is NOT identified by this backtest — a 10x change in
memory length moves Spain's title probability by ~1pp (scripts/07). We set xi=0.0027
(pooled argmin, plateau centre) and report the flatness rather than claim an
improvement. This is the project's evaluation philosophy working as intended: a
hyperparameter that cannot hurt much cannot help much either.

## Ablation studies (scripts/07)
Counterfactual tournament re-runs quantify each data choice for the site's
"what moves the forecast" section. Host advantage is the headline: switching every
2026 match to neutral ground drops USA group-win from 23.7% to 12.5% (and title odds
8x, 0.8%->0.1%), Mexico 67.7%->53.5%, Canada 51.6%->38.4%. Memory length (xi at the
grid extremes) is statistical noise by comparison. Rule of thumb established: venue
data >> recency weighting, and both << the rating itself.

## Parameter bootstrap (backlog #8, 2026-06-11)
B=100 nonparametric refits on resampled training history; each simulated tournament
draws one parameter set. Measured spread: sd(beta_elo)=0.017, sd(beta_home)=0.031,
sd(rho)=0.029. Effect on P(champion): mean |delta| = 0.07pp vs the point estimate —
within Monte Carlo noise at 20k sims. Verdict: with ~17k matches behind 5 parameters,
epistemic parameter uncertainty is negligible; the bootstrap stays on (it is the
correct integral) but the honest reading is that model risk lives in WHICH data
enters, not in the optimizer. Completes the luck-handling triad of the design.

## Player layer v1 (backlog #6, 2026-06-11)
Goals from the tournament Monte Carlo (groups + KO incl. extra time, same sims/seed)
are split per team via Multinomial(G, p): p = exponentially decayed scorer shares from
goalscorers.csv (half-life 900 days; squad membership proxied by >=1 goal in the last
900 days), with a 17.3% "new faces" bucket — the measured share of WC2014/18/22 goals
scored by players with no prior international goal; the bucket cannot win the Boot.
Result vs market (raw implied): Mbappé 15.9% vs 14.8%, Kane 13.6% vs 12.9%, Messi
10.7% vs 8.0%, Haaland 5.6% vs 6.9% — close on all four priced names. Declared
artifact: Enner Valencia tops the table (17.7%) on a genuine recent record plus
Ecuador's high simulated totals, with no age/minutes signal to discount him — the
exact blind spot the FBref block (backlog #5) should fix; until then it stands as a
documented limitation, not silently patched.

## Live updating (during the tournament)
scripts/10: played group matches (refreshed results.csv) enter as fixed_results;
ratings and parameters stay frozen at tournament eve — live updates change what is
KNOWN, never what the model believes. Knockout conditioning is a follow-up.

## Football-capital block (backlog #5a, admitted on probation, 2026-06-12)
First external residual block. Feature: squad-mean club Elo (official squad lists
from Wikipedia x api.clubelo.com snapshot at each tournament's opening day),
unmatched/non-European clubs imputed at the snapshot's 10th percentile, z-scored
within tournament. Enters as a multiplicative tilt on the DC lambdas:
lh' = lh*exp(+b*capdiff), la' = la*exp(-b*capdiff). Admission test (scripts/12),
leave-one-tournament-out on the same 6 tournaments as the xi tuning: pooled OOS
log-loss 0.9952 -> 0.9939 (-0.0012 over 345 matches), b* stable and positive in all
folds (0.023-0.041), b=0.0328 on all six. Paired t = -0.32: the gain is within
noise. Verdict: ADMITTED by the letter of the rule (it improves), explicitly ON
PROBATION — re-judged on WC2026 itself in the final report. Two declared caveats:
(1) clubelo covers Europe only, so the feature taxes squads based elsewhere
(Brazil, Mexico, USA move AWAY from the market while European squads move toward
it — a structural source bias, not a football statement); (2) one global b, no
within-squad weighting (caps/minutes) yet. Effect on the headline: Spain 24.5->26.8,
France 10.2->12.1, England 6.6->8.0; Brazil 5.3->4.3, Mexico 2.6->1.5. Side effect:
Ecuador's totals drop, Mbappe overtakes Valencia atop the Golden Boot table.

## Football-capital v2 — minutes-weighting (admitted, near-null, 2026-06-14)
Refinement: weight each player's club Elo by his club-season minutes (FBref via
Wayback, point-in-time), so a fringe player at a giant counts less than a regular.
Gated v1-vs-v2 on the three folds with minutes data (WC2018/22, Euro2024,
scripts/21): pooled OOS log-loss v1 -0.0008, v2 -0.0013 vs no-capital -> v2 wins,
but the two features correlate at 0.997 and the 2026 z-scores shift by 0.05 on
average (max 0.27, Ecuador). Deployed (it is the gate winner and the theoretically
correct weighting; beta moves 0.0328 -> 0.0353), reported as a near-null: national
squads are overwhelmingly regular starters, so who-plays-where barely changes once
you know who's called up. Build code shared in src/wc26/capital.py so v1 and v2 use
identical club matching.

## Altitude block (admitted, the strongest signal, 2026-06-17)
Reframed from "non-backtestable" (Simone): test it where altitude actually varies —
CONMEBOL World Cup qualifying, a home/away round robin from La Paz (3782 m) and Quito
(2854 m) to sea level (611 matches 2000-2025, scripts/22). Feature: a team is
acclimatised to its HABITUAL altitude (mean elevation of its home cities); a side is
hurt by max(0, venue_alt - habitual)/1000 km; tilt lh*=exp(b*(suffer_away-suffer_home)).
6 time-ordered folds: pooled OOS log-loss -0.0225, paired t=-2.42 (by far our most
significant block — capital was t=-0.32), b=0.134/km stable in every fold (0.129-0.141).
ADMITTED. Deployment (scripts/23): per-(team, 2026-venue) city tilt via the city_log_tilt
mechanism (built originally for the rejected heat block — reused). Only Mexico City
(2240 m) and Zapopan (1565 m) carry it. Effect: Mexico's title odds 2.1%->3.2% (it plays
in the air it lives in), Ecuador/South Africa gain; lowland teams drawn to altitude
(group A's Czech Republic, South Korea) drop. Declared simplification: habitual altitude
from the national team's home cities, not players' club cities (a possible v2); oxygen-mask
training is unmodellable and ignored. Altitude tilts are centralised via
src/wc26/tilts.py:load_city_tilt.

## Altitude-in-Elo: the rating-inflation correction (scripts/35, 2026-06-25)
A DISTINCT effect from the venue tilt above, prompted by Ecuador's live collapse.
The venue tilt is a match-time handicap; it does nothing about the fact that a
habitual-altitude side *accumulates* Elo by winning home qualifiers in the thin air,
which the flat 100-pt Elo home advantage cannot tell apart from genuine strength. That
inflation then travels to a sea-level neutral tournament. Fix under test: an
altitude-AWARE Elo home advantage, dr += gamma*(suffer_away - suffer_home) in Elo points,
so a home win at altitude raises the expected score and earns fewer rating points.
Two gates: (A) the canonical 6-tournament pooled OOS LL is FLAT/slightly worse (best
gamma=0) — WC/Euro avoid altitude, so it is blind, exactly as the venue tilt was. (B) the
power cut — CONMEBOL sides playing at sea-level-ish venues (alt<1000 m), forward folds,
n=1226 — is cleanly U-shaped with an interior optimum at gamma=75 Elo pts/km: OOS LL
0.8971->0.8930 (-0.0042), paired t=-1.91. The interior optimum (monotone approach on both
sides) is the evidence it is a real effect, not monotone noise. Proportionality is good:
at 2026-06-11 it deflates exactly the right sides and only them — Bolivia -89 (La Paz),
Ecuador -19, Colombia -6 (Bogotá), while lowland visitors Brazil/Argentina nudge +10/+5
(their altitude away-results cost less). Verdict: ADMITTED ON PROBATION (effect size beats
capital/fatigue, t weaker than the venue tilt). DEPLOYED 2026-06-25 (Simone's call):
elo.py carries ALT_GAMMA=75 and the canonical build (scripts/01) is altitude-aware, so every
downstream consumer of elo_history.parquet inherits it and the full live chain was re-run.
This deliberately moves the live model off the frozen 2026-06-11 baseline (baseline_eve.csv
stays immutable, so on-site movement deltas now blend the model change with results — an
accepted cost). Honest caveat for the Ecuador question: the correction is only ~-18
Elo for Ecuador (Guayaquil at sea level dilutes Quito in their habitual mean), nowhere near
enough to drop them out of the top 10 — most of Ecuador's high rating is genuine, so their
WC collapse is, like Turkey's, mostly variance/squad rather than a fixable rating bias.

## Are the Elo weights identified? No -- a flat plateau (scripts/37, 2026-06-25)
The Elo update weights (HOME_ADV=100, the 60/50/40/30/20 K tiers, the goal-diff
multiplier) were inherited from eloratings.net and never tuned on our own backtest. We
tuned them xi-style: recompute the whole Elo history per setting, refit DC point-in-time
before each of six tournaments, score pooled OOS outcome log-loss (n=345), one factor at a
time around the deployed config (home_adv=100, global K mult=1.0, friendly K=20; altitude
term fixed). Result: a FLAT PLATEAU, like xi -- no axis beats the default beyond noise.
home_adv: best 40 (delta -0.0012, t=-0.69), and >=130 is mildly worse (t~+1.3). k_mult:
totally flat (best 1.25, delta -0.0004, t=-0.19) -- it trades off against the refitted
beta_elo exactly as predicted. friendly_k: best 30 (delta -0.0018, t=-0.75), NOT significant.
The one significant signal is a negative control: pushing friendly K DOWN to 5-10 clearly
HURTS (t=+2.10 / +1.78) -- friendlies carry real information and must not be discounted to
near-zero (a common modelling shortcut our data rejects). Verdict: keep the inherited
convention; deploy nothing. The eloratings weights sit at/near our predictive optimum.
Full arc: capital was admitted on clubelo (Europe-only). Investigation (scripts/28-34)
showed its predictive value came ENTIRELY from clubelo's geographic bias — the floor
penalising non-European-league players WAS the signal. Head-to-head gate: clubelo+floor
-0.0015 OOS, global footballdatabase +0.0016, hybrid +0.0017. So a fair, unbiased club
rating (footballdatabase, cross-confederation Elo, coverage 54%->89%) makes capital fail
the admission gate — and 'no capital' both predicts better than the fair version AND is
unbiased. We briefly deployed worldelo on principle, then REMOVED the block entirely:
it cannot earn its place without a bias we won't endorse, and dropping it costs only
+0.0015 log-loss (noise, like xi/bootstrap). Completing club coverage would NOT help —
the floor is the signal, so more fair ratings = less signal; the unmatched ~11% are
genuinely weak clubs where the floor is ~correct anyway. The footballdatabase pipeline
and capital_fdb/hybrid CSVs are kept for the record and the model lab (capital is now a
shadow). Deployed team tilts = fatigue only. Lesson: a feature can pass the gate for the
wrong reason; interrogate WHY it predicts, not just THAT it does.


## Climate re-test with club-country acclimatisation (still REJECTED, 2026-06-18)
Simone's approved cheap re-do of the heat block with the correct proxy: a team's
climate = mean climate of its players' CLUB countries (clubelo; non-European clubs
fall back to the national country), not the passport country. Same gate, on top of
capital. Verdict: REJECTED again, pooled OOS +0.0165 (t=+2.09). Autopsy: the wc2022
fold alone explodes (+0.089) — Qatar's air-conditioned stadiums actively invert the
signal; the other five folds are a wash (mostly micro-improvements). So the proxy fix
DID help (no more "Canada suffers in Doha"), but heat still can't be admitted: the only
hot World Cup in the test era was air-conditioned, and outside it there is no
exploitable heat variation in WC/Euro data. Honest closure: heat doesn't show even done
right. (Contrast altitude, which had huge CONMEBOL variation and no "AC" equivalent.)

## Climate block (backlog #5b, REJECTED, 2026-06-12)
Hypothesis: heat mismatch (venue climatology minus home-country climatology, same
month-day window, 10 prior years, point-in-time) tilts goal rates. Same LOTO gate
as the capital block, on top of it. Verdict: REJECTED, and not marginally — pooled
OOS log-loss +0.0218 WORSE (paired t=+2.61), per-fold coefficients flip sign, and
the Qatar 2022 fold explodes (+0.11): air-conditioned stadiums, winter scheduling,
and the design flaw that a passport is not a climate (Canada "suffers" 38.8C in
Doha while its players live in European club cities). The admission rule did its
job on a plausible-sounding feature. Coefficient stays at zero; the city-aware tilt
mechanism stays in simulate.py for any future venue-dependent block that passes
(e.g. club-city-based acclimatization, v2). Altitude remains non-backtestable
(no high-altitude venue since 1986) and is therefore also out.

## Player layer gate — a bug in the judge, and a reversed verdict (2026-06-12)
The allocation gate (scripts/16) scores the REALISED within-team scorer splits of
WC2014/18/22 with the per-goal multinomial log-likelihood, point-in-time weights,
debutant share from the three prior WCs. FIRST RUN (buggy metric): goals by
scorers without a named share were scored at the FULL new-faces bucket mass, as if
the bucket were one pseudo-player — a metric that rewards diffuse named lists and,
in the limit, scores an all-bucket model perfectly (LL = 0, caught when an empty
variant "won"). Under that metric every challenger lost and we wrongly announced
that rosters carry no information. CORRECTED METRIC: the bucket mass is split
among the K squad members without a named share. The verdict FLIPS, decisively:
official-squad filter + age discount exp(-0.1*max(0, age-30)) beats v1 in all
three folds (pooled 2.7266 vs 2.9122, LOTO alpha stable at 0.1-0.15). DEPLOYED.
Sequential test on top (scripts/18): FBref club-season non-penalty goals as a
blended component — helps vs v1 alone (-0.056 OOS) but adds nothing on top of the
squad+age base (+0.004): club form is subsumed; NOT deployed (npxG would be nicer
but the 2017-18 archive page predates xG, and we deploy only what we gate).
Consequences for 2026: Mbappe 30.4% Golden Boot (France's mass concentrates once
non-selected past scorers leave), Kane 16.6%, Messi 1.2% vs market ~8% (age 39
discount — our boldest market disagreement, on the record), Valencia 2.5%.
Lesson, on the record: the gate protects you only if the metric is right; when a
variant scores impossibly well, suspect the judge before the contestant.

## Fatigue block (backlog #5d, admitted on probation, 2026-06-13)
Feature: squad-mean club-season minutes (FBref via Wayback, pre-tournament
season; for the November 2022 fold the previous full season — declared), matched
players only (coverage 38-48%, Big-5 bias as with capital), z-scored within
tournament. Gated on top of the capital tilt so quality is already priced; what
is left is load. Two folds only (minutes pages exist point-in-time for 2018 and
2022): b* = -0.027/-0.029 (stable, theory-consistent sign: more minutes, fewer
goals), mean OOS -0.0012. Admitted on probation; b = -0.0280 on both folds.
Effect on 2026: France (load +1.79 sigma, heaviest squad) -2.4pp on the title,
Spain -1.4, England -1.3; Morocco/Ecuador/Mexico gain. Residual tilts are now
centralised in src/wc26/tilts.py (capital + fatigue).

## Penalty-taker bonus (player v3a, REJECTED, 2026-06-13)
Multiplying the designated taker's weight by (1+k) — taker = most penalty goals in
the 3 prior years — loses for every k > 0 on the corrected gate (mean OOS +0.0054).
Reading: penalties already live inside the taker's historical goal share; an
explicit bonus double-counts them. Rejected without regret.

## Live automation (2026-06-13)
Cloud routine `WC26 live refresh` (claude.ai/code/routines), every 4 hours through
2026-07-19: pulls upstream results.csv, and only if new 2026 matches were played
reruns 01 -> 10 -> 15 and pushes data/outputs/site. Beliefs stay frozen at June 11;
the routine may not touch model code. Guard: self-disables instruction after the
final; group-reconstruction canary aborts the commit on data corruption.

## Cohesion — admitted by the letter, excluded on proportionality (2026-06-18)
Last selected variable. Transfermarkt career-shared-minutes is unscrapeable; proxy =
squad club-concentration (Herfindahl of club shares), gated ON TOP of capital so it
tests squad STRUCTURE net of club LEVEL. LOTO on 6 tournaments: pooled OOS log-loss
-0.0047 — it passes the bar — but paired t=-0.60 (not significant) and the coefficient
is NEGATIVE (b=-0.067): net of club quality, more concentrated squads slightly
UNDERperform (an "insular squad" signal, the opposite of the popular club-core story).
The decisive problem is deployment: Spain (cohesion_z=1.70) would take a -0.114 log-rate
hit EVERY match, compounding to -9.6pp on its title odds — a non-significant feature
swinging the favourite by ten points. The backtest measures the AVERAGE effect over 345
mostly-moderate-z matches; deployment is dominated by an extreme-z favourite on a deep
run, exactly where the evidence is thinnest. So cohesion is NOT in the official model:
it lives as a shadow (scripts/25), scored live alongside diaspora. (It nominally tops
the 20-match live board — which is precisely why the live board never decides admission;
the backtest does.) Lesson: "improves OOS log-loss" is necessary, not sufficient — the
effect must also be proportionate to the evidence.

## Cohesion attempt #2 — squad continuity, REJECTED (scripts/39, 2026-06-26)
A second, independent realisation of the cohesion idea (Simone): continuity = the
fraction of a team's tournament squad that also appeared in its most recent PREVIOUS
tournament squad (point-in-time, from the wc2014/euro2016/wc2018/euro2020/wc2022/euro2024
squad files). "A settled team that has played together" — e.g. France brought 18/26 of its
WC2022 squad to Euro2024 (settled) vs Spain's 10/26 (rebuilt). Gated on the CURRENT model
(plain Elo-DC; capital is gone) over the 6 tournaments. Verdict: REJECTED — pooled OOS
log-loss +0.0029 WORSE (0.9962 -> 0.9992), paired t=+1.08, 4 of 6 folds worse; fitted
b=-0.025 (settled squads if anything slightly worse, not significant). Telling: Turkey was
the 2nd-most-settled wc2026 squad (z=1.54) and still collapsed. Two different cohesion
proxies (club-concentration, continuity) now both fail — cohesion does not show in our
data; only the unscrapeable Transfermarkt pairwise-minutes graph remains untested.

## Broadened gate — does the model generalise beyond UEFA? YES (scripts/40, 2026-06-28)
The admission gate was always WC+Euro (target-resemblance), but that pool is UEFA-heavy,
so we never checked generalisation. Added a ROBUST gate of the four continental cups
(Copa América/CONMEBOL, AFCON/CAF, Asian Cup/AFC, Gold Cup/CONCACAF; editions auto-detected,
point-in-time, n=1213; Nations League excluded as too noisy). Findings:
- BASE CALIBRATION, counter to the worry, is BETTER outside Europe. Pooled OOS LL ROBUST
  0.9315 vs PRIMARY 0.9856 (lower=better; uniform 1.0986). Per confederation, gain over
  uniform: Gold Cup +0.239, Asian Cup +0.209, Copa +0.150, World Cup +0.134, AFCON +0.108,
  Euro +0.083. The Euro is the HARDEST tournament we predict (compressed, all-good field);
  Gold/Asian cups the easiest (big favourite-vs-minnow Elo gaps the model nails). So the
  confederation problem is NARROW — it is specifically about CROSS-confederation matchups
  (UEFA-vs-AFC etc., which barely exist to learn from), NOT general non-European competence.
  Within each confederation's own tournament the model is well calibrated.
- xi PLATEAU HOLDS on ROBUST too (0.0010->0.9310 ... 0.0050->0.9320, a 0.001 swing) — not
  identified anywhere; the WC+Euro null generalises.
- ALTITUDE-IN-ELO not vetoed: on ROBUST gamma=75 vs 0 is -0.0000 (t=-0.02, n=166) — the
  continental finals are sea-level/single-host so blind to altitude (like WC+Euro); the
  CONMEBOL qualifiers stay the only power test, and the broad set does not contradict it.
New convention: Elo/DC-level features should pass PRIMARY (decides) and not be contradicted
by ROBUST (veto); squad-based blocks still can't extend (no continental-cup squad files).

## CIES football academies — settled as un-gateable + collinear (2026-06-26)
Long flagged as "fuzzy, collinear" and dropped without a gate; Simone asked to settle it.
Investigated the data directly. CIES publishes a country talent-EXPORT index (expatriates
trained: Brazil 1289, France 1033, Argentina 905, England 535, Spain 458, Colombia 448,
Nigeria 385 ...), conceptually distinct from capital (development, not current club level).
But it is NOT cleanly gateable: the public reports list only ~top 15 exporters (smaller WC
nations absent), as a recent snapshot with no clean per-country point-in-time history. A
self-built point-in-time proxy (squad-expatriation = share of squad at foreign clubs) is
blocked too: clubelo gives club-country for European clubs only, and fdb_master carries no
country — so non-European domestic players can't be classified. And the obtainable signal
is collinear with Elo by construction (the same big nations top both export and rating), so
even if gated it would most likely be subsumed — the original worry, now evidenced. Verdict:
stays OUT, un-gateable from available data; documented rather than deployed. Revisit only if
a full, historical, per-country CIES table becomes available.

## Diaspora — a declared shadow bet, never in the official model (2026-06-17)
There is no past World Cup on US soil, so de-facto home support from US diaspora
CANNOT be backtested — it can never pass the gate. Per Simone's call it lives ONLY as
a registered shadow challenger (scripts/24-25), not in the official forecast. Feature
(deliberately simple, declared): support_index = sqrt(US foreign-born population from
the country), normalised to the max; city tilt = 0.06 * index at US venues only,
excluding the USA. Mexico dominates (+0.060 log-lambda, huge US presence), then Korea
/ Colombia / Haiti / England. DIA_SCALE=0.06 is a chosen, not fitted, number.

## The model lab — shadow scoreboard (2026-06-17)
scripts/25 scores every variant (Elo-only, full model, minus each admitted block,
plus the diaspora shadow) on the played matches, beliefs frozen at June 11, accumulated
live. After 20 matches all six sit within 0.019 log-loss of each other (Elo-only
nominally ahead, full model mid-pack) — exactly as predicted: one tournament cannot
separate tilts this small (our 345-match backtest reached |t|=2.4 only for altitude).
The scoreboard's value is honesty + accrual over many tournaments, NOT today's leader;
the site shows it with that caveat in bold. The diaspora bet is visible here and
nowhere else.

## Three-way benchmark
After the tournament: log-loss and calibration of our model vs bookmaker implied
probabilities (margin removed, Shin's method) vs Klement's forecasts (GDP/population/
climate/ranking/host). Note on Klement: 3 correct champions out of 3 but always
near-favourites (survivorship of mediatised models); Netherlands 2026 is his first
true contrarian test.
