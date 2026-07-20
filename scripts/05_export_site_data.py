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
    replay = json.load(open("outputs/history/replay.json"))
except FileNotFoundError:
    replay = None
try:
    scoring = json.load(open("outputs/history/scoring.json"))
except FileNotFoundError:
    scoring = None
try:
    standings = json.load(open("outputs/history/standings.json"))
except FileNotFoundError:
    standings = None
try:
    _bk = pd.read_csv("outputs/bracket.csv")
    bracket = {}
    for r in _bk.itertuples(index=False):
        bracket.setdefault(int(r.match), {})[r.slot] = {
            "team": r.team, "p": float(r.p), "adv": float(getattr(r, "adv", 0.0))}
except FileNotFoundError:
    bracket = None
try:
    _b = pd.read_csv("outputs/history/baseline_eve.csv", index_col=0)
    baseline = {t: {"P_champion": round(float(r.P_champion), 4),
                    "P_qualify": round(float(r.P_qualify), 4),
                    "P1": round(float(r.P1), 4)} for t, r in _b.iterrows()}
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
# Tournament outcome, only once the whole thing is played (all 104): powers the
# site's retrospective "how it ended" state. Final = last fixture, third-place =
# the one before it (the schedule always closes final, then play-off the day prior).
try:
    _rr = pd.read_csv("data/raw/results.csv")
    _wcp = (_rr[(_rr.tournament == "FIFA World Cup") & (_rr.date >= "2026-06-11")]
            .dropna(subset=["home_score"]).sort_values("date"))
    if len(_wcp) >= 104:
        _fin, _thd = _wcp.iloc[-1], _wcp.iloc[-2]
        _mkres = lambda r: {"home": r.home_team, "away": r.away_team,
                            "hg": int(r.home_score), "ag": int(r.away_score)}
        outcome = {"final": _mkres(_fin), "third": _mkres(_thd),
                   "champion": _fin.home_team if _fin.home_score > _fin.away_score else _fin.away_team}
    else:
        outcome = None
except FileNotFoundError:
    outcome = None
try:
    match_dists = json.load(open("outputs/match_dists.json"))
except FileNotFoundError:
    match_dists = None
try:
    team_drivers = json.load(open("outputs/team_drivers.json"))
except FileNotFoundError:
    team_drivers = None
try:
    next_matches = json.load(open("outputs/next_matches.json"))
except FileNotFoundError:
    next_matches = None
try:
    bracket_dists = json.load(open("outputs/bracket_dists.json"))
except FileNotFoundError:
    bracket_dists = None
try:  # goal-model point estimate -> client-side score grids for any pairing
    dc_params = json.load(open("outputs/dc_params.json"))
except FileNotFoundError:
    dc_params = None
# knockout venue map + altitude city tilt, so the client-side goal model can
# price host-at-home ties exactly like the Monte Carlo (venue-aware).
from wc26.simulate import (FINAL_MATCH, KO_CITY, QF_MATCHES, R16_MATCHES,
                           R32_MATCHES, SF_MATCHES)
ko_venues = {int(mn): {"v": v, "c": KO_CITY.get(mn)} for mn, _, _, v in
             [*R32_MATCHES, *R16_MATCHES, *QF_MATCHES, *SF_MATCHES, FINAL_MATCH]}
try:
    _at = pd.read_csv("outputs/altitude_tilt.csv")
    city_tilt = {f"{r.team}|{r.city}": round(float(r.log_tilt), 4)
                 for r in _at.itertuples(index=False) if abs(float(r.log_tilt)) > 1e-6}
except FileNotFoundError:
    city_tilt = None
# venue metadata (country + altitude) for every 2026 host city, so the site can
# label each fixture's venue with the host-nation flag and its elevation.
try:
    _alt = json.load(open("data/external/altitude.json"))
    _fx = pd.read_csv("data/raw/results.csv")
    _fx = _fx[(_fx["tournament"] == "FIFA World Cup") & (_fx["date"] >= "2026-06-11")]
    city_meta = {c: {"country": g["country"].iloc[0],
                     "alt": int(_alt.get(c) or 0)}
                 for c, g in _fx.dropna(subset=["city"]).groupby("city")}
except (FileNotFoundError, KeyError):
    city_meta = None
try:  # live market: latest outright snapshot + a short history for movement
    _oh = [json.loads(l) for l in open("outputs/odds_history.jsonl") if l.strip()]
    market_now = {"fetched": _oh[-1]["fetched"], "outright": _oh[-1]["outright"]} if _oh else None
    market_hist = [{"t": s["fetched"], "o": s["outright"]} for s in _oh[-30:]] if _oh else None
except FileNotFoundError:
    market_now, market_hist = None, None

data = {
    "generated": str(date.today()),
    "n_sims": 20000,
    "seed": 26,
    "model_version": "v3 (Elo-driven Dixon-Coles, tuned xi, official bracket, calibrated shootouts, parameter bootstrap, player layer, deployed tilts: fatigue + altitude)",
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
    "replay": replay,
    "scoring": scoring,
    "standings": standings,
    "uniform_logloss": 1.0986,
    "fatigue": fatigue,
    "baseline_eve": baseline,
    "bracket": bracket,
    "golden_boot": golden_boot,
    "golden_boot_market": {"Kylian Mbappé": 575, "Harry Kane": 675,
                           "Lionel Messi": 1150, "Erling Haaland": 1350},
    "debutant_share": 0.173,
    "distinct_scorers": distinct_top,
    "match_dists": match_dists,
    "outcome": outcome,
    "dc_params": dc_params,
    "ko_venues": ko_venues,
    "city_tilt": city_tilt,
    "city_meta": city_meta,
    "team_drivers": team_drivers,
    "next_matches": next_matches,
    "bracket_dists": bracket_dists,
    "market_now": market_now,
    "market_hist": market_hist,
    "teams": teams,
}

# Kalshi pre-tournament P(qualify) per team (public candlesticks, frozen in
# data/external; see scripts/archive/45) — powers the gold market verdicts on
# the group scorecards.
try:
    import csv as _csv
    data["kalshi_qualify"] = {r["team"]: float(r["p_qualify_pre"])
                              for r in _csv.DictReader(open("data/external/kalshi_groupqual.csv"))}
except FileNotFoundError:
    pass

with open("site/data.js", "w") as f:
    f.write("// Generated by scripts/05_export_site_data.py — do not edit by hand.\n")
    f.write("const WC26 = " + json.dumps(data, indent=1) + ";\n")
print(f"site/data.js written: {len(teams)} teams, xi_tuning={xi_tuning is not None}, ablations={ablations is not None}")

# Cache-busting: stamp asset URLs with the export time so browsers pick up fresh
# data/JS/CSS within the page's own 10-minute cache window (GitHub Pages serves
# every file with max-age=600; without a version query, returning visitors keep
# stale assets long after a refresh has deployed).
import re as _re
_v = pd.Timestamp.utcnow().strftime("%Y%m%d%H%M")
for _page in ("site/index.html", "site/under-the-hood.html"):
    _h = open(_page).read()
    _h = _re.sub(r'(href="style\.css)(\?v=\d+)?(")', rf"\g<1>?v={_v}\g<3>", _h)
    _h = _re.sub(r'(src="(?:data|app)\.js)(\?v=\d+)?(")', rf"\g<1>?v={_v}\g<3>", _h)
    open(_page, "w").write(_h)
print(f"asset URLs stamped ?v={_v}")
