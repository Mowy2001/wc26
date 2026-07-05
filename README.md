# WC26 — Probabilistic World Cup 2026 Forecasting

**Live site: <https://mowy2001.github.io/wc26/>** — the forecast, graded in public.

An end-to-end, **point-in-time** probabilistic model for the 2026 FIFA World Cup:
group tables, knockout bracket, champion odds, and the Golden Boot — refreshed
live as real results come in, and graded against them.

**Philosophy.** We don't predict *who wins*, we predict *distributions*. Every
number is an empirical frequency over 20,000 Monte Carlo tournaments. Four rules
make it trustworthy:
- **Point-in-time always** — no rating or feature uses information dated after the
  moment being predicted.
- **Earn your place out-of-sample** — every optional ingredient is gated on
  predictive log-loss across past tournaments, never on in-sample fit.
- **Proportionality** — improving log-loss is necessary but not sufficient; the
  effect must be proportionate to its statistical support.
- **Benchmarks are not inputs** — the market and rival forecasts are yardsticks we
  score against, never fed back in.

## Architecture

```
[Match history 1872→today]
        │
        ▼
Proprietary Elo  (src/wc26/elo.py)        ← altitude-aware home advantage
        │  point-in-time pre-match ratings
        ▼
Elo-driven Dixon-Coles  (src/wc26/dixon_coles.py)
   log λ = β0 + β_home·home + β_elo·ΔElo/400  (+ residual tilts)
        │  + deployed tilts: fatigue, altitude  (src/wc26/tilts.py)
        ▼
Tournament Monte Carlo  (src/wc26/simulate.py)
   groups (FIFA tiebreakers, 8 best thirds) → R32 bracket
   → extra time (Poisson λ/3) → shootout logit → champion
        │
        ├──► Player layer  (src/wc26/players.py)  → Golden Boot
        └──► Live conditioning (fixed_results + KO)  → graded vs reality
```

1. **Proprietary Elo** — a from-scratch replica of the eloratings.net method over
   the full history, so we own point-in-time ratings at any date. The home
   advantage is **altitude-aware** (`ALT_GAMMA`): a home win in thin air earns
   fewer points, so altitude fortresses don't inflate ratings that won't travel to
   a sea-level neutral tournament.
2. **Elo-driven Dixon-Coles** — Poisson goals whose rates are *functions of Elo*
   (no free per-team parameters — international data is too sparse), with the ρ
   low-score correction and exponential time-decay ξ. Residual tilts enter as a
   small multiplicative nudge `λ ·= exp(tilt_h − tilt_a)`.
3. **Monte Carlo** — 20,000 tournaments (fixed seed): samples every scoreline,
   applies FIFA tiebreakers, ranks the 8 best thirds, plays the official bracket
   with extra time and a calibrated penalty-shootout model. A parameter bootstrap
   integrates over fit uncertainty.

## The deployed model: what's in, what's out

**In** (each survived the gate): Elo · Dixon-Coles · **fatigue** tilt · **altitude**
venue tilt · **altitude-in-Elo** correction.

**Out** (tested, didn't make it — see `scripts/archive/` and `docs/METHODOLOGY.md`):
football capital (Europe-bias), climate/heat (rejected ×2), penalty bonus
(double-counts), diaspora (shadow), cohesion (excluded ×2), CIES academies
(un-gateable), player-Elo (cross-confederation data wall).

A **broadened gate** (Copa/AFCON/Asian Cup/Gold Cup) confirms the engine
generalises — it predicts non-European tournaments at least as well as European
ones; the confederation gap is specifically about *cross*-confederation matches,
which barely exist to learn from.

## Quick start

The live pipeline lives in [`scripts/`](scripts/README.md) (run from the repo root).
Completed experiments are in [`scripts/archive/`](scripts/archive/README.md).

```bash
pip install -r requirements.txt
# reproduce the forecast from scratch (see scripts/README.md for the full order)
python scripts/01_build_and_validate.py    # Elo (altitude-aware) + group reconstruction
python scripts/08_bootstrap_params.py      # DC parameter bootstrap
python scripts/04_simulate_tournament.py   # 20k full-tournament sims → P(champion)
python scripts/09_player_layer.py          # Golden Boot
python scripts/36_export_match_dists.py    # per-match score heatmaps
python scripts/38_team_drivers.py          # strength vs draw vs tilts breakdown
python scripts/05_export_site_data.py      # write site/data.js → open site/index.html

# during the tournament — the canonical live refresh:
python scripts/10_live_update.py           # condition on real results, re-sim, re-export
```

## The site (`site/`)

A self-contained static site (`index.html` + `app.js` + `style.css` + generated
`data.js`) that leads with the **live track record** and tells the story match by
match:
- a **time slider** over the whole tournament (champion race, bracket, groups, Golden Boot);
- a **predicted bracket** with a *Reach slot / Win the tie* toggle;
- clickable **per-match score heatmaps** (the modal scoreline, with the real result ringed);
- a **driver breakdown** — each team's odds split into squad quality vs the draw vs venue & freshness;
- **model-vs-market** divergence flags and a live, graded scoreboard.

## Data

| File | Content |
|---|---|
| `data/raw/results.csv` | ~49k international matches 1872→today, incl. the 2026 schedule |
| `data/raw/goalscorers.csv`, `shootouts.csv` | scorers (Golden Boot) and historical shootouts (the penalty model) |
| `data/external/` | squads, clubelo/fdb, altitude, tmax (climate) — mostly feeding archived experiments |

Source: github.com/martj42/international_results (CC0). The 2026 groups are
**reconstructed algorithmically** from the fixture graph (connected components) —
the draw is never hardcoded.

## Results so far

- WC2022 backtest (point-in-time): log-loss **1.060 vs 1.099** uniform, well calibrated.
- Live 2026 (all played matches, groups + knockouts): running log-loss
  **0.87 vs 1.099** uniform after 88 matches — updated automatically on the
  [live site](https://mowy2001.github.io/wc26/).
- External benchmarks frozen on 2026-06-11 (BetMGM, Klement) for a fair three-way
  comparison after the final — scored, never used as inputs.

## Documentation

- **`docs/DECISIONS.md`** — the decision ledger: every block with its verdict (admitted / probation / shadow / rejected / null) and out-of-sample number.
- **`docs/METHODOLOGY.md`** — the full narrative: every layer, every gated block, every honest null.
- **`scripts/README.md`** / **`scripts/archive/README.md`** — the live pipeline and the experiment ledger.
- **`data/README.md`** — data provenance and attribution (core vs model-lab inputs).
- **`CLAUDE.md`** — operating context and conventions for the AI agent that builds this.

Every `src/wc26/*.py` module carries a docstring with its full mathematical model.

## License

Code and documentation: **MIT** (see `LICENSE`). Third-party datasets under `data/`
keep their own licenses — match data is CC0 (martj42/international_results); full
provenance in `data/README.md`.
