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

## Capital source switched to worldelo / footballdatabase (2026-06-18, declared override)
The capital block was originally built on clubelo (Europe-only), which — we later found —
made it predictive ONLY through a geographic bias: the floor penalising non-European-league
players carried the signal (scripts/33 head-to-head: clubelo+floor -0.0015 OOS; global
footballdatabase +0.0016; hybrid +0.0017). On Simone's call we switched the deployed source
to the global footballdatabase Elo (a proper cross-confederation Elo via the Club World Cup;
point-in-time per-club monthly history, scripts/28-32; coverage 54%->89%). This is a DECLARED
OVERRIDE of the admission rule: footballdatabase capital fails the OOS gate (REJECTED, b=0.0115
in-sample), but it is the unbiased measure we would have built from the start — and starting
there we would simply have rejected the block, never silently shipped a worse model (the gate
compares to no-capital). We keep it in, with a small weight, as a forward-looking choice: club
strength should matter, and the clubelo edge was an artefact of an era when non-European leagues
were weaker. Effect: European powers down (Spain -2.7pp, France -1.5pp), the Americas up (Mexico
+1.2pp, Brazil/Colombia/Ecuador up). Honest caveat on record: 'no capital' actually predicts
slightly better than worldelo-capital and is equally unbiased.

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
