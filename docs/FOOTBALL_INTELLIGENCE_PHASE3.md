# Independent football intelligence: Phase 3

## Integration result

The existing FPL engine remains the selected projection source. This phase adds a time-safe context adapter that can place eligible football insights beside a player's FPL baseline, official FPL ep_next benchmark, and (only after the deadline) actual points. It does not alter fpl_engine.py, transfer scoring, HOLD comparisons, captaincy, or the selected points.

Every result has intelligence_adjusted_points=null, selected_points equal to the existing baseline, and recommendation_changed=false. There is no candidate adjustment model or calibration artifact to apply.

## Why numeric integration is gated

The audit found:

- fpl_engine.py already uses lagged FPL xGI, minutes, starts/appearance rates, fixtures, and a current availability override;
- Phase 2 xG/xA, minutes, and start trends overlap those inputs and could double-count them;
- the implemented source has selected historical competition coverage rather than a complete current Premier League feed;
- there is no verified crosswalk from its player identities to FPL IDs by season, nor a time-aligned current FPL outcome cohort;
- the existing historical FPL backtest is a proxy and cannot test an intelligence-adjusted candidate against the baseline with the required point-in-time event features.

Without those matched features and a chronological walk-forward comparison, a point adjustment or transfer reranking would invent evidence. Phase 3 therefore exposes only evidence context and an explicit baseline-only gate.

## Context gates

An insight appears in FPL context only when it has:

1. an exact, positive FPL player ID match;
2. an eligible OBSERVED or INFERRED classification;
3. HIGH or MEDIUM sample quality;
4. timezone-aware source and retrieval timestamps no later than the target deadline;
5. source player and fixture references.

Post-deadline, unverified, incomplete, unmapped, or untimestamped evidence is excluded with a reason. Indicators already represented in baseline xGI/minutes/start features are labeled as overlaps. Other indicators remain descriptive and cannot change the forecast.

Actual points are accepted only with an explicit availability timestamp strictly after the target deadline. They are outcome diagnostics and are never used in the selected forecast.

## Gate before any future adjustment

Before the adapter can return an adjusted forecast, a later change must provide:

- a rights-cleared, versioned player crosswalk and source coverage audit;
- point-in-time insight availability aligned to historical FPL deadlines;
- a candidate model with a recorded data cutoff and reproducible artifact hash;
- chronological walk-forward results over multiple seasons comparing baseline, candidate, and official ep_next on common players/Gameweeks;
- improvement in error and decision metrics, including transfer-versus-HOLD outcomes, without evidence leakage or double-counting;
- a safe fallback and rollback path.

This phase does not claim that these requirements are satisfied. The Streamlit dashboard and optimizer remain unchanged; surfacing the context in a user-facing section belongs to Phase 4 after the integration gate has a real, validated signal.
