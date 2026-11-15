# Independent football intelligence: Phase 5

## Audit and scope

| Capability | Existing implementation | Phase 5 result |
|---|---|---|
| FPL baseline model | Frozen v3.0 ridge artifact and heuristic fallback | Reused unchanged |
| Historical FPL validation | `historical_backtest.py` reconstructs conservative lagged features; `fpl_validation.py` reports error and ranking metrics | Reused as context; historical results remain explicitly proxy evaluations |
| Intelligence indicators | Phase 2 descriptive, evidence-linked match trends | Reused as context only |
| Intelligence to FPL bridge | Phase 3 attaches time-safe context but always selects the baseline | Remains baseline-only |
| Candidate intelligence forecast | No candidate model, calibration artifact, or numeric adjustment exists | Not fabricated or trained |
| Point-in-time Premier League evidence | Current source is selected historical coverage without a verified FPL crosswalk or aligned deadline snapshots | Insufficient for an intelligence-adjusted backtest |

This phase adds a reusable out-of-sample evaluator for a future candidate artifact. It compares a candidate and the existing FPL baseline on the same player/Gameweek cohort, and compares both with official `ep_next` where that value has a pre-deadline timestamp. It reports MAE, RMSE, bias, Spearman rank correlation, top-10/top-20 capture, season summaries, and per-Gameweek results.

Point-error metrics pool the common player/Gameweek rows. Rank and top-capture metrics are calculated within each Gameweek and macro-averaged across Gameweeks, so repeated player IDs across different weeks are not collapsed into one cross-week ranking.

## Leakage and cohort rules

Each validation row must identify the FPL player, season, Gameweek, target deadline, feature snapshot time, candidate training cutoff, baseline prediction, candidate prediction, actual score, actual availability time, and candidate artifact hash. Optional `ep_next` values also require a pre-deadline availability timestamp.

- Features must be available no later than the target deadline.
- Candidate training data must end before the feature snapshot.
- Actual points must become available strictly after the target deadline.
- The evaluator fails on duplicate player/Gameweek rows, mixed candidate versions or hashes, missing required scores, invalid timestamps, and invalid artifact hashes. It does not silently shrink the candidate/baseline cohort.
- All metrics use the same complete candidate/baseline cohort. The `ep_next` comparison uses only rows where a valid pre-deadline `ep_next` value exists, and reports that cohort separately.

The default screening policy requires two seasons, six evaluated Gameweeks in every season, at least 20 common players in every Gameweek, MAE improvement in each season and at least two thirds of Gameweeks, no pooled RMSE regression, and no regression in pooled rank or top-player capture. These thresholds are explicit and configurable for research. A passing result is `ELIGIBLE_FOR_HUMAN_REVIEW`, never an automatic promotion.

## Controlled learning result

No intelligence-adjusted predictions or point-in-time evaluation records are currently available. The evaluator therefore returns `NO_CANDIDATE_DATA` until a candidate is supplied. The existing v3.0 production artifact and forecasts are not changed, and no candidate is trained or promoted in this phase.

Passing this screening gate is not sufficient for production. Before any promotion, create a rights-cleared prospective or historically aligned dataset; verify player/fixture identity and feature cutoffs; evaluate calibration, minutes/start probabilities, transfer-versus-HOLD and captaincy outcomes; inspect uncertainty and per-position/Gameweek cohorts; and approve a reproducible, versioned artifact with rollback. No statistical significance claim is made by the screening summary.

## Remaining dependency

The next data step is a source and capture design that can preserve Premier League observations and FPL projections exactly as they were available before each deadline, with a verified season-aware player crosswalk and usage rights for the intended analysis. Until that exists, the engine must continue to select the existing baseline and report that intelligence validation has not run.
