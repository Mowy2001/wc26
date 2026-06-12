"""Three-way benchmark utilities (backlog #9): model vs bookmaker vs Klement.

Shin's method for removing the bookmaker margin
-----------------------------------------------
Raw inverse odds overstate probabilities because they sum to 1 + margin.
Proportional scaling assumes the margin is spread evenly; empirically
bookmakers shade longshots more (favourite-longshot bias). Shin (1992/93)
models a fraction z of bettors as insiders, which yields implied
probabilities

    p_i = ( sqrt(z^2 + 4 (1 - z) pi_i^2 / B) - z ) / (2 (1 - z))

where pi_i are the normalised inverse odds and B = sum(pi_i). z is found
by requiring sum(p_i) = 1 (bisection — z is typically 1-3%). For outright
markets with many runners this materially de-shades the longshots vs
proportional scaling.

Scoring: average negative log-likelihood of the realised outcome, the same
metric used everywhere in this project. For the champion market a single
realised outcome cannot rank forecasters reliably; we therefore also score
match-level forecasts as results accumulate (the honest, higher-n test).
"""

from __future__ import annotations

import numpy as np


def implied_raw(odds_american: dict[str, float]) -> dict[str, float]:
    """Inverse American odds (contain the bookmaker margin)."""
    return {k: 1.0 / (1.0 + o / 100.0) for k, o in odds_american.items()}


def shin_probs(odds_american: dict[str, float], residual_mass: float = 0.0) -> dict[str, float]:
    """Margin-free implied probabilities via Shin's method.

    residual_mass: probability reserved for runners NOT quoted in the input
    (an outright market rarely lists all 48 teams). The Shin solve then
    targets sum(p) = 1 - residual_mass.
    """
    target = 1.0 - residual_mass
    pi = np.array(list(implied_raw(odds_american).values()))
    B = pi.sum()
    pi_n = pi / B

    def total(z: float) -> float:
        p = (np.sqrt(z * z + 4.0 * (1.0 - z) * pi_n * pi_n / B) - z) / (2.0 * (1.0 - z))
        return p.sum()

    lo, hi = 0.0, 0.5
    if total(0.0) <= target:  # no margin to remove beyond normalisation
        z = 0.0
    else:
        for _ in range(60):
            mid = (lo + hi) / 2.0
            lo, hi = (lo, mid) if total(mid) < target else (mid, hi)
        z = (lo + hi) / 2.0
    p = (np.sqrt(z * z + 4.0 * (1.0 - z) * pi_n * pi_n / B) - z) / (2.0 * (1.0 - z))
    p = p * (target / p.sum())
    return dict(zip(odds_american.keys(), p.tolist()))


def log_score(prob_of_realised: float) -> float:
    """Negative log-likelihood of the realised outcome (lower = better)."""
    return float(-np.log(max(prob_of_realised, 1e-12)))
