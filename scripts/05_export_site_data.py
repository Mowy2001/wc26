"""Step 5: export model outputs as site/data.js for the static results site.

The site is fully self-contained (file:// friendly): data is embedded as a
JS constant instead of fetched, so no server or CORS setup is needed.
Re-run this script after every new simulation (steps 04, 06, 07).
"""
import json, sys
from datetime import date
sys.path.insert(0, "src")
import pandas as pd
from wc26.benchmark import shin_probs
from wc26.elo import ratings_asof

BETMGM = {"Spain": 450, "France": 500, "England": 700, "Brazil": 800,
          "Portugal": 900, "Argentina": 900, "Germany": 1400,
          "Netherlands": 2000, "United States": 5000}
tbl = pd.read_csv("outputs/tournament_probs_v1.csv", index_col=0)
elo = ratings_asof(pd.read_parquet("outputs/elo_history.parquet"), "2026-06-11")

teams = []
for t, r in tbl.iterrows():
    teams.append({
        "team": t, "group": r["group"], "elo": round(float(elo[t])),
        **{c: float(r[c]) for c in ["P1", "P2", "P3", "P4", "P_qualify", "P_R32",
                                    "P_R16", "P_QF", "P_SF", "P_final", "P_champion",
                                    "P_top_scoring_team"]},
    })

# Optional research artifacts (steps 06-07); the site degrades gracefully
# if they are missing.
try:
    xi = pd.read_csv("outputs/xi_tuning.csv")
    xi_pooled = (
        xi.assign(tot=xi.ll * xi.n).groupby("xi")
        .apply(lambda g: g.tot.sum() / g.n.sum(), include_groups=False)
        .round(4)
    )
    xi_tuning = {"pooled": {f"{k:g}": v for k, v in xi_pooled.items()},
                 "chosen": 0.0027, "paired_t_extremes": 0.45,
                 "n_matches": int(xi.groupby("tournament").n.first().sum())}
except FileNotFoundError:
    xi_tuning = None
try:
    ablations = json.load(open("outputs/ablations.json"))
except FileNotFoundError:
    ablations = None
try:
    gb = pd.read_csv("outputs/golden_boot.csv").head(12)
    golden_boot = gb.round(4).to_dict("records")
    distinct = pd.read_csv("outputs/distinct_scorers.csv", index_col=0)
    distinct_top = distinct.head(6).iloc[:, 0].round(4).to_dict()
except FileNotFoundError:
    golden_boot, distinct_top = None, None
try:
    _cz = pd.read_csv("outputs/capital.csv").query("tournament == 'wc2026'")
    capital = {"meta": json.load(open("outputs/capital_beta.json")),
               "coverage": round(float(_cz.coverage.mean()), 3),
               "z": {r.team: round(float(r.capital_z), 2) for r in _cz.itertuples(index=False)}}
except FileNotFoundError:
    capital = None
try:
    climate = json.load(open("outputs/climate_beta.json"))
except FileNotFoundError:
    climate = None
try:
    altitude = json.load(open("outputs/altitude_beta.json"))
except FileNotFoundError:
    altitude = None
try:
    shadow = pd.read_csv("outputs/shadow_scores.csv").to_dict("records")
except FileNotFoundError:
    shadow = None
try:
    _bk = pd.read_csv("outputs/bracket.csv")
    bracket = {}
    for r in _bk.itertuples(index=False):
        bracket.setdefault(int(r.match), {})[r.slot] = {"team": r.team, "p": float(r.p)}
except FileNotFoundError:
    bracket = None
try:
    _b = pd.read_csv("outputs/history/baseline_eve.csv", index_col=0)
    baseline = {t: {"P_champion": round(float(r.P_champion), 4),
                    "P_qualify": round(float(r.P_qualify), 4)} for t, r in _b.iterrows()}
except FileNotFoundError:
    baseline = None
try:
    _fz = pd.read_csv("outputs/fatigue.csv", index_col=0)["fatigue_z"]
    fatigue = {"meta": json.load(open("outputs/fatigue_beta.json")),
               "z": {k: round(float(v), 2) for k, v in _fz.items()}}
except FileNotFoundError:
    fatigue = None
try:
    bd = pd.DataFrame(json.load(open("outputs/dc_bootstrap.json")))
    bootstrap = {"B": int(len(bd)), "sd": bd.std().round(4).to_dict()}
except FileNotFoundError:
    bootstrap = None

data = {
    "generated": str(date.today()),
    "n_sims": 20000,
    "seed": 26,
    "model_version": "v3 (Elo-driven Dixon-Coles, tuned xi, official bracket, calibrated shootouts, parameter bootstrap, player layer, capital block on probation)",
    "backtest": {
        "tournament": "World Cup 2022 (64 matches, point-in-time fit)",
        "log_loss_model": 1.060, "log_loss_uniform": 1.099,
        "calib_predicted": 0.541, "calib_observed": 0.562,
    },
    "shootout": {"b_home": 0.309, "b_elo": 0.677, "n": 677,
                 "ll_model": -462.5, "ll_coin": -469.3},
    # External benchmarks recorded 2026-06-11 (tournament eve) — do not overwrite.
    "betmgm_outright": BETMGM,
    "betmgm_shin": {k: round(v, 4) for k, v in
                    shin_probs(BETMGM, residual_mass=float(
                        1.0 - tbl.loc[list(BETMGM), "P_champion"].sum())).items()},
    "kalshi_usa_group": 0.51,
    "klement": "Netherlands champions (final vs Portugal); England and Spain out in the semis",
    "xi_tuning": xi_tuning,
    "ablations": ablations,
    "bootstrap": bootstrap,
    "capital": capital,
    "climate": climate,
    "altitude": altitude,
    "shadow_scores": shadow,
    "uniform_logloss": 1.0986,
    "fatigue": fatigue,
    "baseline_eve": baseline,
    "bracket": bracket,
    "golden_boot": golden_boot,
    "golden_boot_market": {"Kylian Mbappé": 575, "Harry Kane": 675,
                           "Lionel Messi": 1150, "Erling Haaland": 1350},
    "debutant_share": 0.173,
    "distinct_scorers": distinct_top,
    "teams": teams,
}

with open("site/data.js", "w") as f:
    f.write("// Generated by scripts/05_export_site_data.py — do not edit by hand.\n")
    f.write("const WC26 = " + json.dumps(data, indent=1) + ";\n")
print(f"site/data.js written: {len(teams)} teams, xi_tuning={xi_tuning is not None}, ablations={ablations is not None}")
