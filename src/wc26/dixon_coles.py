"""Dixon-Coles match model driven by Elo covariates.

Model
-----
Goals scored by each side follow Poisson distributions:

    X_home ~ Poisson(lambda_h),  X_away ~ Poisson(lambda_a)

    log lambda_h = beta0 + beta_home * home + beta_elo * elodiff_h
    log lambda_a = beta0 + beta_away * home - beta_elo * elodiff_h

where `elodiff_h = (elo_home - elo_away) / 400` (scaled to keep the
coefficient O(1)) and `home` is 1 for non-neutral venues. Attack/defence
strengths are therefore *functions of Elo* rather than free per-team
parameters — essential with international data, where each team plays
only ~10 matches/year (a free-parameter DC would overfit badly).

Two Dixon-Coles refinements over naive double-Poisson:

1. Low-score dependence correction tau(x, y; rho)
   Empirically 0-0/1-1 are more frequent and 1-0/0-1 less frequent than
   independence implies. The DC tau adjusts only those four cells:

       tau(0,0) = 1 - lambda_h * lambda_a * rho
       tau(0,1) = 1 + lambda_h * rho
       tau(1,0) = 1 + lambda_a * rho
       tau(1,1) = 1 - rho
       tau(x,y) = 1 otherwise

2. Exponential time-downweighting
   Match i played `d_i` days before the fit date receives weight
   exp(-xi * d_i / 365). `xi` is a hyperparameter chosen on the backtest
   (predictive log-loss), NOT by in-sample likelihood — in-sample always
   prefers xi = 0.

Estimation: weighted maximum likelihood via scipy L-BFGS-B.

Extension hooks
---------------
`extra_cols` lets you add residual-block covariates (climate mismatch,
football capital, diaspora support, fatigue, cohesion graph density, ...).
Each extra column c enters as `+ beta_c * c_h` in log lambda_h and
`- beta_c * c_h` in log lambda_a (differential parametrisation). This is
the hierarchical-residual design: Elo carries baseline strength, extras
explain over/under-performance relative to Elo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10  # truncation for score-grid computations


def tau(x: np.ndarray, y: np.ndarray, lh: np.ndarray, la: np.ndarray, rho: float) -> np.ndarray:
    """Dixon-Coles low-score adjustment factor (vectorised)."""
    out = np.ones_like(lh, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lh * la * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lh * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + la * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return np.clip(out, 1e-10, None)


@dataclass
class DixonColes:
    """Elo-covariate Dixon-Coles model.

    Attributes (after fit)
    ----------------------
    params_ : dict with beta0, beta_home, beta_away, beta_elo, rho and one
              beta per extra column.
    """

    xi: float = 0.0018  # time-decay/day; ~half-life 13 months. Tuned in backtest.
    extra_cols: list[str] = field(default_factory=list)
    params_: dict | None = None

    # ------------------------------------------------------------------ #
    def _lambdas(self, theta: np.ndarray, X: dict) -> tuple[np.ndarray, np.ndarray]:
        b0, bh, ba, belo = theta[0], theta[1], theta[2], theta[3]
        extras = theta[4 : 4 + len(self.extra_cols)]
        lin = belo * X["elodiff"]
        for coef, col in zip(extras, self.extra_cols):
            lin = lin + coef * X[col]
        log_lh = b0 + bh * X["home"] + lin
        log_la = b0 + ba * X["home"] - lin
        return np.exp(np.clip(log_lh, -10, 3)), np.exp(np.clip(log_la, -10, 3))

    def _neg_loglik(self, theta: np.ndarray, X: dict) -> float:
        rho = theta[-1]
        lh, la = self._lambdas(theta, X)
        ll = (
            poisson.logpmf(X["hg"], lh)
            + poisson.logpmf(X["ag"], la)
            + np.log(tau(X["hg"], X["ag"], lh, la, rho))
        )
        return -float(np.sum(X["w"] * ll))

    # ------------------------------------------------------------------ #
    def fit(self, matches: pd.DataFrame, fit_date: pd.Timestamp) -> "DixonColes":
        """Weighted ML fit.

        `matches` needs: date, home_score, away_score, neutral, elo_home_pre,
        elo_away_pre, plus any `extra_cols`. Only matches before `fit_date`
        are used (point-in-time discipline).
        """
        df = matches[matches["date"] < fit_date].dropna(
            subset=["home_score", "away_score", "elo_home_pre", "elo_away_pre"]
        )
        days = (fit_date - df["date"]).dt.days.to_numpy(float)
        X = {
            "hg": df["home_score"].to_numpy(int),
            "ag": df["away_score"].to_numpy(int),
            "home": (~df["neutral"].astype(bool)).to_numpy(float),
            "elodiff": ((df["elo_home_pre"] - df["elo_away_pre"]) / 400.0).to_numpy(float),
            "w": np.exp(-self.xi * days),
        }
        for c in self.extra_cols:
            X[c] = df[c].to_numpy(float)

        theta0 = np.concatenate(
            [[0.1, 0.25, -0.15, 0.8], np.zeros(len(self.extra_cols)), [-0.05]]
        )
        bounds = [(-3, 3)] * (4 + len(self.extra_cols)) + [(-0.9, 0.9)]
        res = minimize(self._neg_loglik, theta0, args=(X,), method="L-BFGS-B", bounds=bounds)
        names = ["beta0", "beta_home", "beta_away", "beta_elo"] + [
            f"beta_{c}" for c in self.extra_cols
        ] + ["rho"]
        self.params_ = dict(zip(names, res.x))
        self.params_["converged"] = bool(res.success)
        self.params_["n_matches"] = int(len(df))
        return self

    # ------------------------------------------------------------------ #
    def predict_lambdas(
        self, elo_home: float, elo_away: float, neutral: bool = True, extras: dict | None = None
    ) -> tuple[float, float]:
        """Expected goals (lambda_h, lambda_a) for one fixture."""
        p = self.params_
        lin = p["beta_elo"] * (elo_home - elo_away) / 400.0
        for c in self.extra_cols:
            lin += p[f"beta_{c}"] * (extras or {}).get(c, 0.0)
        home = 0.0 if neutral else 1.0
        lh = np.exp(p["beta0"] + p["beta_home"] * home + lin)
        la = np.exp(p["beta0"] + p["beta_away"] * home - lin)
        return float(lh), float(la)

    def score_matrix(self, lh: float, la: float) -> np.ndarray:
        """Joint pmf over scorelines (0..MAX_GOALS)², tau-corrected, renormalised."""
        gx = np.arange(MAX_GOALS + 1)
        ph = poisson.pmf(gx, lh)[:, None]
        pa = poisson.pmf(gx, la)[None, :]
        M = ph * pa
        rho = self.params_["rho"]
        M[0, 0] *= 1 - lh * la * rho
        M[0, 1] *= 1 + lh * rho
        M[1, 0] *= 1 + la * rho
        M[1, 1] *= 1 - rho
        return M / M.sum()

    def outcome_probs(self, lh: float, la: float) -> tuple[float, float, float]:
        """(P_home_win, P_draw, P_away_win) from the score matrix."""
        M = self.score_matrix(lh, la)
        return float(np.tril(M, -1).sum()), float(np.trace(M)), float(np.triu(M, 1).sum())

    def sample_score(self, lh: float, la: float, rng: np.random.Generator) -> tuple[int, int]:
        """Draw one scoreline — the Monte Carlo 'luck' injection point."""
        M = self.score_matrix(lh, la)
        idx = rng.choice(M.size, p=M.ravel())
        return int(idx // (MAX_GOALS + 1)), int(idx % (MAX_GOALS + 1))
