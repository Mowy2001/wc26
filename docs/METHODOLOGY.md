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

## Three-way benchmark
After the tournament: log-loss and calibration of our model vs bookmaker implied
probabilities (margin removed, Shin's method) vs Klement's forecasts (GDP/population/
climate/ranking/host). Note on Klement: 3 correct champions out of 3 but always
near-favourites (survivorship of mediatised models); Netherlands 2026 is his first
true contrarian test.
