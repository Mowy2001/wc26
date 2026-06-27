"""Step 20: admission gate for the penalty-taker bonus (player layer v3a).

Hypothesis: the designated penalty taker collects extra goals beyond his
open-play share. Variant: multiply the taker's weight by (1+k) inside the
named mass, renormalise. Taker = scorer with most penalty goals for the
team in the 3 years before the tournament (tie -> most recent penalty).
Gate: corrected per-goal multinomial LL on realised WC2014/18/22 splits,
on top of the deployed base (squad filter + age alpha=0.1), k by LOTO.
"""
import sys
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from wc26.data import load_goalscorers
from wc26.players import estimate_debutant_share, squad_weights, DEBUTANT_KEY, _norm_name

WCS = [("wc2014", "2014-06-12", "2014-07-13"),
       ("wc2018", "2018-06-14", "2018-07-15"),
       ("wc2022", "2022-11-20", "2022-12-18")]
PRIOR = {"wc2014": [("2002-05-31", "2002-06-30"), ("2006-06-09", "2006-07-09"), ("2010-06-11", "2010-07-11")],
         "wc2018": [("2006-06-09", "2006-07-09"), ("2010-06-11", "2010-07-11"), ("2014-06-12", "2014-07-13")],
         "wc2022": [("2010-06-11", "2010-07-11"), ("2014-06-12", "2014-07-13"), ("2018-06-14", "2018-07-15")]}
KS = [0.0, 0.25, 0.5, 1.0, 1.5]

gs = load_goalscorers()


def taker(team: str, asof: pd.Timestamp) -> str | None:
    p = gs[(gs["team"] == team) & (gs["date"] < asof)
           & (gs["date"] >= asof - pd.Timedelta(days=1095))
           & (gs["penalty"].astype(bool))].dropna(subset=["scorer"])
    if p.empty:
        return None
    counts = p.groupby("scorer").agg(n=("date", "size"), last=("date", "max"))
    return counts.sort_values(["n", "last"], ascending=False).index[0]


def with_bonus(w: dict, tk: str | None, k: float) -> dict:
    if not tk or k == 0:
        return w
    deb = w.get(DEBUTANT_KEY, 0.0)
    named = {p: v for p, v in w.items() if p != DEBUTANT_KEY}
    tk_key = next((p for p in named if _norm_name(p) == _norm_name(tk)), None)
    if tk_key is None:
        return w
    named[tk_key] *= 1.0 + k
    z = sum(named.values())
    out = {p: v / z * (1.0 - deb) for p, v in named.items()}
    out[DEBUTANT_KEY] = deb
    return out


def ll_per_goal(k: float, slug, start, end, squad):
    asof = pd.Timestamp(start)
    deb = estimate_debutant_share(gs, PRIOR[slug])
    wc = gs[(gs["date"] >= start) & (gs["date"] <= end) & (~gs["own_goal"].astype(bool))].dropna(subset=["scorer"])
    out, n = 0.0, 0
    for team, g in wc.groupby("team"):
        w = squad_weights(gs, squad, team, asof, deb, age_alpha=0.1, drop_to_bucket=False)
        w = with_bonus(w, taker(team, asof), k)
        norm_w = {_norm_name(p): v for p, v in w.items() if p != DEBUTANT_KEY}
        roster = {_norm_name(p) for p in squad[squad["team"] == team]["player"]}
        p_new = w.get(DEBUTANT_KEY, 1e-6) / max(1, len(roster - set(norm_w)))
        for scorer, goals in g["scorer"].value_counts().items():
            out -= goals * np.log(max(norm_w.get(_norm_name(scorer), p_new), 1e-9))
            n += goals
    return out / n


df = pd.DataFrame({
    slug: {f"k{k}": ll_per_goal(k, slug, s, e, pd.read_csv(f"data/external/squads_{slug}.csv"))
           for k in KS}
    for slug, s, e in WCS}).T
print(df.round(4).to_string())

deltas = []
print("\nLOTO: held WC | k from others | LL vs base (k=0)")
for held in df.index:
    best = df.drop(held).mean().idxmin()
    d = float(df.loc[held, best] - df.loc[held, "k0.0"])
    deltas.append(d)
    print(f"  {held}: {best}  ({d:+.4f})")
verdict = "ADMITTED" if np.mean(deltas) < 0 else "REJECTED"
print(f"\nMean OOS delta: {np.mean(deltas):+.4f} -> {verdict}")
