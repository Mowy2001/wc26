"""Rejected candidate: the official FIFA ranking as the strength input, in place
of our point-in-time Elo. (Backlog answer to "is our Elo actually better than the
FIFA ranking?" — the ranking is on the deliberately-excluded list; this is the
backtest that earns it that place.)

Design (same shape as the other admission gates, PRIMARY = World Cups)
----------------------------------------------------------------------
For each World Cup 1994-2022 we fit the SAME Dixon-Coles goal model on the matches
BEFORE the tournament (point-in-time, no look-ahead), predict that tournament's
matches, and score the log-loss of the realised 1X2 result. We do this twice —
once with our Elo as the rating, once with the FIFA ranking — and both models fit
on the IDENTICAL match set (only rows where both ratings exist), so the only thing
that varies is the rating source, not the training size or the goal-model calibration.

FIFA points come from the public historical ranking (data/external/fifa_ranking.csv,
monthly snapshots 1992-2024, Dato-Futbol/fifa-ranking). Point-in-time join = the
latest snapshot STRICTLY before each match. FIFA changed methodology in 2018
(adopted an Elo-like formula), so its raw points are on two different scales; we
z-score points within each snapshot date (neutralising the scale break) and put
them on an Elo-like scale so the optimiser sees comparable magnitudes. The verdict
is unchanged on raw post-2018 points (2022: Elo 1.079 vs FIFA 1.088).

Verdict (see docs/DECISIONS.md): REJECTED. Our Elo wins every one of the eight
tournaments; pooled 493 matches, Elo log-loss 0.9745 vs FIFA 1.0485 (uniform 1.0986),
paired t = 4.18, p ~ 3.5e-5. The fitted rating coefficient stays ~0.75-0.82 for Elo
but shrinks for FIFA from ~0.5 (1990s) to ~0.14 (2018-22): the goal model learns the
FIFA numbers carry little information about match margins.

Standalone; run from the repo root:  python scripts/archive/44_fifa_vs_elo.py
"""
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy import stats

from wc26.data import load_results
from wc26.dixon_coles import DixonColes

# FIFA ranking name -> results.csv name (only the World-Cup-relevant mismatches).
FIFA_TO_RESULTS = {
    "China PR": "China", "Côte d'Ivoire": "Ivory Coast", "IR Iran": "Iran",
    "Korea DPR": "North Korea", "Korea Republic": "South Korea",
    "USA": "United States", "Congo DR": "DR Congo", "Cabo Verde": "Cape Verde",
    "Cape Verde Islands": "Cape Verde", "Czechia": "Czech Republic",
    "Türkiye": "Turkey", "FYR Macedonia": "North Macedonia",
}
WORLD_CUPS = [1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022]


def fifa_ratings() -> pd.DataFrame:
    """Point-in-time FIFA rating, z-scored per snapshot (kills the 2018 scale break)
    and mapped onto an Elo-like scale so the DC optimiser is well conditioned."""
    f = pd.read_csv("data/external/fifa_ranking.csv", parse_dates=["date"])
    f["team"] = f["team"].replace(FIFA_TO_RESULTS)
    z = f.groupby("date")["total_points"].transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    f["fifa_r"] = 1500.0 + 150.0 * z
    return f[["date", "team", "fifa_r"]].sort_values("date")


def load_matches() -> pd.DataFrame:
    """Played matches with point-in-time Elo (from elo_history) and the point-in-time
    FIFA rating (latest snapshot strictly before the match) attached to each side."""
    res = load_results()
    elo = pd.read_parquet("outputs/elo_history.parquet")
    df = res.dropna(subset=["home_score", "away_score"]).copy()
    key = ["date", "home_team", "away_team"]
    df["dup"] = df.groupby(key).cumcount()
    elo["dup"] = elo.groupby(key).cumcount()
    df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    fifa = fifa_ratings()
    for side in ("home", "away"):
        ft = fifa.rename(columns={"team": f"{side}_team", "fifa_r": f"fifa_{side}"})
        df[f"fifa_{side}"] = pd.merge_asof(
            df.sort_values("date"), ft.sort_values("date"), on="date",
            by=f"{side}_team", direction="backward", allow_exact_matches=False,
        )[f"fifa_{side}"].values
    return df


def loglosses(model: DixonColes, sub: pd.DataFrame, hcol: str, acol: str) -> np.ndarray:
    out = []
    for r in sub.itertuples(index=False):
        lh, la = model.predict_lambdas(getattr(r, hcol), getattr(r, acol), neutral=bool(r.neutral))
        pH, pD, pA = model.outcome_probs(lh, la)
        p = pH if r.home_score > r.away_score else (pA if r.home_score < r.away_score else pD)
        out.append(-np.log(max(p, 1e-12)))
    return np.array(out)


def main() -> None:
    df = load_matches()
    wc = df[df["tournament"] == "FIFA World Cup"].copy()
    both = ["fifa_home", "fifa_away", "elo_home_pre", "elo_away_pre"]
    elo_ll, fifa_ll = [], []
    print("World Cup   n    Elo LL  (beta)   FIFA LL (beta)    delta(FIFA-Elo)")
    for year in WORLD_CUPS:
        tourn = wc[wc.date.dt.year == year]
        start = tourn.date.min()
        test = tourn.dropna(subset=both)
        train = df[df.date < start].dropna(subset=both).copy()  # identical rows for both models

        m_elo = DixonColes().fit(train, start)
        tf = train.copy()
        tf["elo_home_pre"], tf["elo_away_pre"] = tf["fifa_home"], tf["fifa_away"]
        m_fifa = DixonColes().fit(tf, start)

        e = loglosses(m_elo, test, "elo_home_pre", "elo_away_pre")
        f = loglosses(m_fifa, test, "fifa_home", "fifa_away")
        elo_ll += list(e)
        fifa_ll += list(f)
        print(f"  {year}    {len(test):3d}   {e.mean():.4f} ({m_elo.params_['beta_elo']:.2f})   "
              f"{f.mean():.4f} ({m_fifa.params_['beta_elo']:.2f})     {f.mean() - e.mean():+.4f}")

    e, f = np.array(elo_ll), np.array(fifa_ll)
    t, p = stats.ttest_rel(f, e)
    print(f"\nPOOLED  n={len(e)}   our Elo {e.mean():.4f}   FIFA {f.mean():.4f}   "
          f"delta {f.mean() - e.mean():+.4f}   paired t={t:.2f} p={p:.2g}   "
          f"(uniform 1/3 = {-np.log(1/3):.4f})")
    print("Verdict: REJECTED — the FIFA ranking is a materially weaker predictor of "
          "World Cup results than our point-in-time Elo.")


if __name__ == "__main__":
    main()
