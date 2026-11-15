# Current implementation audit

Audit base: `main` at `4e92509` (PR #70 merged). This note separates existing behavior from the focused lineup work in this branch.

| Feature | Existing implementation on audit base | Status | Required work |
|---|---|---|---|
| Squad synchronization | `fpl_squad.py` loads verified target-GW picks, or reconstructs a prior roster from validated picks and transfer/chip history. Incomplete reconstruction requires confirmation. | Partial | Preserve source and target-GW status in every recommendation; current-XI comparison must only use verified target-GW picks. |
| Bank and account verification | `fpl_account.py` validates bank, free transfers, roster and known prices. `app.py` reads public profile/history and supports manager-confirmed bank, FT and selling-price inputs. | Partial | Public data can lag the current deadline; manual selling prices and team-value reconciliation may be needed. Never treat a prior roster as a current published XI. |
| Transfer optimization | `fpl_transfer.py` searches legal complete rosters with exact-tenth budget checks, free transfers/hits, hold comparison, and total/incremental gates. Search is bounded (30 incoming candidates, 120 beam states) and is approximate. | Partial | Expand decision-quality evaluation only with historical evidence; keep budget and account readiness as hard action gates. |
| Formation and XI optimization | `fpl_lineup.py` selected one legal XI by projected points plus captain points from the 15-player roster. | Partial on audit base; improved in this branch | Score all eight legal formations, expose the runner-up and margin, reject club-cap violations, and use deterministic tie-breaking. |
| Bench and captaincy | One selected XI is reused by the pitch, captain planner and chip calculations. The goalkeeper reserve is first; outfield reserves are ordered by projected points. Captain and vice are highest and second-highest projected starters. | Partial | Add clear reasons, minutes/availability diagnostics, and compare with the actual published XI only when available. Bench ordering remains a projection heuristic. |
| Availability | `fpl_availability.py` supplies one shared status/chance adjustment. The fallback projection applies minutes and availability; v3.0 uses lagged historical minutes features and a current-status override for the nearest GW. | Partial | Factors are heuristics, not calibrated start probabilities. Keep them visible and do not apply a second multiplier in lineup selection. |
| Chip strategy | `fpl_chip_strategy.py` and dashboard logic reuse the current selected XI, bench and per-GW projections. | Partial | Scores are heuristic comparisons over visible fixtures; no learned chip-value calibration is claimed. |
| Forecast validation | `fpl_validation.py` reports point error, bias, rank correlation and top-player capture; live and historical-proxy cohorts are separated in the dashboard. | Partial | Historical proxy is not an exact point-in-time replay. Continue preserving leakage controls and cohort labels. |
| Self-learning | The scheduled worker captures immutable snapshots and reconciles actual points/minutes. The manual historical workflow fits expanding-season holdouts and refits a deployable v3.0 artifact. | Partial | Daily worker does not train a candidate. Historical publication does not require a measured improvement over the current production model and has no promotion history/rollback gate. |
| Dashboard integration | Four tabs use the shared roster, forecast and lineup; bank and recommendation status are surfaced. | Partial | Before this branch, the optimized XI replaced the reference lineup in the UI; there was no current-published-XI versus optimized-XI comparison. |

## Workflow and data-health notes

- `fpl-auto-learning.yml` is restricted to `main`, runs tests, captures/reconciles snapshots, publishes changed snapshot files, and appends the forecast log. A successful workflow run does not mean a model was trained or promoted.
- `fpl-historical-backtest.yml` is manually dispatched on `main`, trains on prior seasons for expanding holdouts, then refits on all configured complete seasons and publishes a v3.0 artifact only after a complete run. It does not compare the candidate to the currently deployed artifact and enforce an improvement threshold before replacement.
- The repo has immutable snapshot storage, append-only diagnostics, a reconciliation index and guarded publishing code. Workflow status for the latest scheduled runs was not available through the connected GitHub tools during this audit; no run result is inferred here.

## Work in this branch

- Validate the three-player club limit in public picks and lineup selection.
- Score the eight legal formations using the same target-GW projection map and stable tie-breaking; report the runner-up and points gap.
- Explain XI, bench and captain choices with factual projection-based reasons and show expected-minutes/availability diagnostics without double-adjusting points.
- Compare the model XI to the verified current published XI. If only prior/reconstructed picks exist, show an explicit unavailable status rather than treating them as current.

## Remaining recommended phases

1. Add an evidence-based candidate-model promotion gate and rollback record around the frozen production artifact.
2. Add decision-outcome tracking for recommendations versus HOLD, actual captain/XI outcomes, transfer hits and saved transfers.
3. Improve calibrated minutes/start probabilities only after enough point-in-time data exists; keep current heuristics clearly labeled until then.
4. Extend future planning only after confirming the same verified account snapshot and per-Gameweek XI are used throughout.
