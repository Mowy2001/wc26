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

## Three-way benchmark
After the tournament: log-loss and calibration of our model vs bookmaker implied
probabilities (margin removed, Shin's method) vs Klement's forecasts (GDP/population/
climate/ranking/host). Note on Klement: 3 correct champions out of 3 but always
near-favourites (survivorship of mediatised models); Netherlands 2026 is his first
true contrarian test.
