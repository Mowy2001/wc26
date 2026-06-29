# WC26 site — redesign brief (v2), 2026-06-29

Full requirements from Simone, to be designed in Claude Design then built.
Audience: LinkedIn + non-technical, but rigorous. Identity: keep Direction B
("Living Forecast", broadcast/neon).

## Framing
- Describe the project as an **open-source, AI-built probabilistic prediction of
  the tournament using public data**. (New lede / about.)

## Structure (new order)
1. **Next matches — FIRST thing on the site.** Upcoming fixtures with their outcome
   probabilities (1X2 for group games, 1-2 for knockout). The hook.
2. **Hero — show the winner immediately, but otherwise fully redesigned.** Big champion
   flag (much larger). Keep "who wins" up top; rethink everything else.
3. Live record — results so far, but the probability shown should be the **match
   OUTCOME** (1X2 groups / 1-2 knockout), not the exact-scoreline prob. Clicking a
   match → expected-goals heatmap (already exists, keep).
4. **Round-by-round table** (who's likely to reach each stage) — restored, visible,
   placed **before the bracket**. [done: un-collapsed]
5. **Bracket — rethought.** Either the classic **two halves facing each other**
   (mirrored, converging on the final) or a distinctive custom graphic. Keep the
   **head-to-head** win-the-tie probabilities. Clicking a tie → **expected-goals
   heatmap** (new — same modal as group matches).
6. Drivers, model-vs-market, under-the-hood (Act 2) — keep, but see market below.

## Interactivity (the big theme)
- **Follow each team's path**: click a team → highlight/trace its most likely route
  through groups + bracket, with per-round probabilities.
- **What-if simulation**: let the user play with the **bracket or groups** — force a
  result / pick a winner and see the probabilities update (a sandbox).
- Time-travel already works (slider changes the past) — keep and lean into it.

## Market — make it DYNAMIC
- Today it's a single frozen snapshot (BetMGM outright, June 11). Make it:
  - **time-aware**: as you scrub the past, both OUR probabilities and the MARKET's
    move — show how each has changed.
  - **per-match** if possible: market odds on individual fixtures, not just outright.
  - handle **updated odds** (current, not only June-11): compare how our numbers moved
    vs how the market moved.

## DATA DEPENDENCIES (flag before building)
- **Per-match market odds** + **live/updated odds over time**: we currently only have
  the frozen June-11 outright champion odds. Per-match and time-series market data
  need a SOURCE (scrape / API) and a stored history. This is the main blocker for the
  dynamic-market asks — needs decision on source.
- Next-match outcome probs: derivable now (match_dists for groups; bracket head-to-head
  for KO). No blocker.
- Team-path tracing: derivable from the per-snapshot bracket + group probs. No blocker.
- What-if simulation: needs a client-side mini-simulator OR pre-computed conditionals;
  scope TBD.

## Open questions for Simone
1. Market data: is there a source we can use for per-match + updated odds (and may we
   scrape it)? Without it, the dynamic/per-match market is not buildable.
2. "Simulation": force-a-result sandbox that re-computes probabilities live, or simpler
   "pick the winner and see the bracket fill"?
3. Bracket: mirrored two-halves vs a custom graphic — preference?
