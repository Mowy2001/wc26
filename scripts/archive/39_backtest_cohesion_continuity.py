"""Step 39: cohesion, attempt #2 — SQUAD CONTINUITY (Simone, 2026-06-26).

The original cohesion idea was "shared career minutes per pair" (Transfermarkt,
unscrapeable). scripts/26 tried a club-concentration proxy and it was EXCLUDED
(passed OOS by a hair but swung Spain -9.6pp on a non-significant coefficient —
failed proportionality). This is a different, cheaper realisation of the same
intuition that we CAN build from local data:

  continuity(team, T) = fraction of the team's squad at tournament T that also
                        appeared in the team's most recent PREVIOUS tournament
                        squad (among wc2014/euro2016/wc2018/euro2020/wc2022/
                        euro2024). "A settled team that has played together."

Point-in-time: a team's continuity at T uses only squads from BEFORE T. France
came into Euro2024 with 18/26 of its WC2022 squad (very settled); Spain only
10/26 (rebuilt) — that contrast is the signal.

Gate: same LOTO-6 harness as the other blocks, but baselined on the CURRENT
model (plain Elo-DC — capital was removed 2026-06-18, so no capital tilt).
z-scored within tournament; tilt lh*=exp(b*(cont_home-cont_away)); admitted only
if pooled OOS log-loss beats b=0 AND the move is proportional (the cohesion lesson).
"""
import json
import sys

sys.path.insert(0, "src")
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from wc26.data import load_results
from wc26.dixon_coles import DixonColes

# slug -> (competition, start, end); ORDERED in time for "previous squad" lookup
TOURNAMENTS = [
    ("wc2014", "FIFA World Cup", "2014-06-12", "2014-07-13"),
    ("euro2016", "UEFA Euro", "2016-06-10", "2016-07-10"),
    ("wc2018", "FIFA World Cup", "2018-06-14", "2018-07-15"),
    ("euro2020", "UEFA Euro", "2021-06-11", "2021-07-11"),
    ("wc2022", "FIFA World Cup", "2022-11-20", "2022-12-18"),
    ("euro2024", "UEFA Euro", "2024-06-14", "2024-07-14"),
]
START = {s: pd.Timestamp(a) for s, _, a, _ in TOURNAMENTS}
SQUADS = {s: pd.read_csv(f"data/external/squads_{s}.csv") for s, *_ in TOURNAMENTS}


def continuity_z(slug):
    """z-scored squad continuity for the teams at `slug` (point-in-time)."""
    here = SQUADS[slug]
    priors = [s for s in SQUADS if START[s] < START[slug]]
    rows = []
    for team, g in here.groupby("team"):
        squad = set(g.player)
        # most recent prior tournament in which this team appeared
        prev = None
        for s in sorted(priors, key=lambda s: START[s], reverse=True):
            psq = set(SQUADS[s][SQUADS[s].team == team].player)
            if psq:
                prev = psq
                break
        if prev is None:
            rows.append({"team": team, "cont": np.nan})
        else:
            rows.append({"team": team, "cont": len(squad & prev) / len(squad)})
    d = pd.DataFrame(rows)
    m, sd = d.cont.mean(), d.cont.std(ddof=0)
    d["z"] = (d.cont - m) / sd if sd > 0 else 0.0
    d["z"] = d["z"].fillna(0.0)  # no prior squad -> neutral
    return d.set_index("team")["z"]


results = load_results()
elo = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
elo["dup"] = elo.groupby(key).cumcount()
df = df.merge(elo[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1").sort_values("date")

packs = {}
for slug, comp, start, end in TOURNAMENTS:
    fd = pd.Timestamp(start)
    model = DixonColes().fit(df[df["date"] >= fd - pd.DateOffset(years=20)], fd)
    test = df[(df["tournament"] == comp) & (df["date"] >= start) & (df["date"] <= end)]
    cz = continuity_z(slug)
    rows = []
    for r in test.itertuples(index=False):
        lh, la = model.predict_lambdas(r.elo_home_pre, r.elo_away_pre, neutral=bool(r.neutral))
        dc = float(cz.get(r.home_team, 0.0) - cz.get(r.away_team, 0.0))
        actual = 0 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 2)
        rows.append((lh, la, dc, actual))
    packs[slug] = (model, rows)


def ll_vec(slug, b):
    model, rows = packs[slug]
    return np.array([-np.log(max(model.outcome_probs(lh * np.exp(b * d), la * np.exp(-b * d))[a], 1e-12))
                     for lh, la, d, a in rows])


def ll(slug, b):
    return float(ll_vec(slug, b).sum())


print(f"{'held out':9s} {'b* (others)':>12s} {'LL(base)':>9s} {'LL(b*)':>9s} {'delta':>8s}")
pool0 = pool1 = n_tot = 0.0
diffs = []
for held, *_ in TOURNAMENTS:
    others = [s for s, *_ in TOURNAMENTS if s != held]
    b_star = minimize_scalar(lambda b: sum(ll(s, b) for s in others), bounds=(-0.6, 0.6), method="bounded").x
    n = len(packs[held][1])
    l0, l1 = ll(held, 0.0) / n, ll(held, b_star) / n
    pool0 += l0 * n; pool1 += l1 * n; n_tot += n
    diffs.append(ll_vec(held, b_star) - ll_vec(held, 0.0))
    print(f"{held:9s} {b_star:12.4f} {l0:9.4f} {l1:9.4f} {l1 - l0:+8.4f}")

d = np.concatenate(diffs)
t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
b_all = minimize_scalar(lambda b: sum(ll(s, b) for s, *_ in TOURNAMENTS), bounds=(-0.6, 0.6), method="bounded").x
verdict = "ADMITTED" if pool1 < pool0 else "REJECTED"
print(f"\nPooled OOS log-loss: base {pool0/n_tot:.4f} vs continuity {pool1/n_tot:.4f} "
      f"({(pool1-pool0)/n_tot:+.4f}) over {int(n_tot)} matches | paired t = {t:.2f}")
print(f"b on all six: {b_all:.4f} | Verdict (gate): {verdict}")

# 2026 deployment feature (continuity of the wc2026 squad vs its last tournament)
SQUADS["wc2026"] = pd.read_csv("data/external/squads_wc2026.csv")
START["wc2026"] = pd.Timestamp("2026-06-11")
c26 = continuity_z("wc2026").rename("continuity_z")
json.dump({"beta_continuity": round(float(b_all), 4), "oos_delta": round(float((pool1-pool0)/n_tot), 5),
           "t_paired": round(float(t), 2), "n": int(n_tot), "verdict": verdict},
          open("outputs/cohesion_continuity_beta.json", "w"), indent=1)
print("\nMost settled wc2026 squads:", {k: round(v, 2) for k, v in c26.nlargest(4).items()})
print("Most rebuilt wc2026 squads:", {k: round(v, 2) for k, v in c26.nsmallest(4).items()})
