"""Step 7: ablation studies — how much does each data choice move the forecast?

Four counterfactual tournament runs (same 20k sims, same seed as the headline):
  - no_host_advantage    : every match on neutral ground (groups + knockout);
                           keeps bootstrap draws like the headline run
  - xi_short_memory      : xi = 0.005  (half-life ~4.6 months), point estimate
  - xi_long_memory       : xi = 0.0005 (half-life ~3.8 years), point estimate
  - no_param_uncertainty : point estimate instead of bootstrap draws
  - no_capital           : without the football-capital tilt (admitted block)

Output: outputs/ablations.json with full P_champion vectors per scenario
plus host-nation group numbers, consumed by the site exporter (step 05).
"""
import json, sys, time
sys.path.insert(0, "src")
import pandas as pd
from wc26.data import load_results, wc2026_group_fixtures, reconstruct_groups
from wc26.elo import ratings_asof
from wc26.dixon_coles import DixonColes
from wc26.simulate import simulate_tournament

results = load_results()
elo_hist = pd.read_parquet("outputs/elo_history.parquet")
df = results.dropna(subset=["home_score", "away_score"]).copy()
key = ["date", "home_team", "away_team"]
df["dup"] = df.groupby(key).cumcount()
e = elo_hist.copy(); e["dup"] = e.groupby(key).cumcount()
df = df.merge(e[key + ["dup", "elo_home_pre", "elo_away_pre"]], on=key + ["dup"], validate="1:1")
train = df[df["date"] >= "2005-01-01"]
FIT = pd.Timestamp("2026-06-11")

elo_now = ratings_asof(elo_hist, "2026-06-11")
gfx = wc2026_group_fixtures(results)
groups = reconstruct_groups(gfx)

base_model = DixonColes().fit(train, FIT)
try:
    draws = json.load(open("outputs/dc_bootstrap.json"))
except FileNotFoundError:
    draws = None
try:
    _cap = pd.read_csv("outputs/capital.csv").query("tournament == 'wc2026'")
    _beta = json.load(open("outputs/capital_beta.json"))["beta_capital"]
    tilt = {r.team: _beta * r.capital_z for r in _cap.itertuples(index=False)}
except FileNotFoundError:
    tilt = None
scenarios = {
    "no_host_advantage": dict(model=base_model, host_advantage=False, param_draws=draws, team_log_tilt=tilt),
    "xi_short_memory": dict(model=DixonColes(xi=0.005).fit(train, FIT), host_advantage=True, param_draws=None, team_log_tilt=tilt),
    "xi_long_memory": dict(model=DixonColes(xi=0.0005).fit(train, FIT), host_advantage=True, param_draws=None, team_log_tilt=tilt),
    "no_param_uncertainty": dict(model=base_model, host_advantage=True, param_draws=None, team_log_tilt=tilt),
    "no_capital": dict(model=base_model, host_advantage=True, param_draws=draws, team_log_tilt=None),
}

out = {}
for name, cfg in scenarios.items():
    t0 = time.time()
    tbl = simulate_tournament(groups, gfx, cfg["model"], elo_now, n_sims=20000,
                              host_advantage=cfg["host_advantage"],
                              param_draws=cfg["param_draws"],
                              team_log_tilt=cfg["team_log_tilt"])["teams"]
    out[name] = {
        "P_champion": {t: round(float(r["P_champion"]), 4) for t, r in tbl.iterrows()},
        "hosts": {t: {"P1": round(float(tbl.loc[t, "P1"]), 4),
                      "P_qualify": round(float(tbl.loc[t, "P_qualify"]), 4)}
                  for t in ["United States", "Mexico", "Canada"]},
    }
    print(f"{name}: done in {time.time() - t0:.0f}s")

with open("outputs/ablations.json", "w") as f:
    json.dump(out, f, indent=1)
print("outputs/ablations.json written")
