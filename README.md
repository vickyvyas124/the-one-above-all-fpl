# The-One-Above-All FPL Manager Hub — v2.9 Validation Release

A Streamlit FPL dashboard for squad review, fixture planning, captain/vice-captain advice, transfer HOLD/ACT guidance, chip timing and validation of player-point forecasts. It uses public FPL endpoints and does not ask for FPL passwords or submit transfers.

## Why v2.9 exists

This is a **validation release**, not a claim that the model is already more accurate. It freezes model weights while comparing our forecasts to a baseline and FPL's live `ep_next` benchmark, records expected minutes separately, and adds a leakage-conscious retrospective proxy backtest over prior seasons.

## What's new in v2.9

- **Benchmark against `ep_next`:** the scheduled worker stores FPL's live `ep_next` value from the same pre-deadline snapshot as the model and baseline. Historical scraped `xP` is not used as a benchmark because the archive timing may include post-match updates.
- **One- and five-Gameweek validation:** the worker stores one-GW projections and five-GW projected totals, and reconciles each with actual results only after the relevant events finish.
- **Ranking metrics:** dashboard reports Spearman rank correlation, top-10/top-20 capture, MAE, bias and `ep_next` comparison on the matched benchmark cohort. Top-k metrics are useful diagnostics, not guarantees of fantasy points. Captain regret is measured only when a complete public squad was available pre-deadline, so the recorded XI/captain decision does not use hindsight.
- **Minutes diagnostic:** records an explicit expected-minutes forecast and reports expected-minutes MAE/bias separately from points error. Expected minutes are scaled to scheduled fixture count, so a blank Gameweek is zero and a Double Gameweek can exceed 90 minutes.
- **Consistent feature coverage:** the scheduled worker fetches player history for the full player pool with bounded concurrency and retries. Each row records whether the history signal was successfully enriched and the sample count.
- **Historical replay:** `historical_backtest.py` evaluates archived 2023–24, 2024–25 and 2025–26 Gameweeks using lagged-only inputs. Where archived fixture/team identifiers are available, the replay distinguishes blanks from single and double Gameweeks using team fixture schedule counts. It deliberately excludes archived same-GW `xP`/`ep_next` values that can contain timing leakage. The replay is a historical proxy, not an exact recreation of what the live API showed before each old deadline.
- **Previous-season context:** manager panel displays past-season final points/rank alongside the current partial season. A final-season score must not be compared directly to this year's partial score as if it were a like-for-like checkpoint.
- **Frozen weights:** online bias correction is paused during validation so metrics are not changing as the model is being evaluated. Any future calibration must be supported by held-out results.
- **Automated storage:** scheduled GitHub Actions persists forecasts and results on the separate `fpl-data` branch. CSV download is optional backup only.

## Deploy through the GitHub web interface

1. Download and extract the v2.9 ZIP on your laptop.
2. Open `https://github.com/vickyvyas124/the-one-above-all-fpl` and stay on `main`.
3. Replace the following root files with the package versions: `app.py`, `fpl_engine.py`, `fpl_background_worker.py`, `fpl_validation.py`, `historical_backtest.py`, `requirements.txt`, and `README.md`. Leave the repository-root `forecast_log.csv` alone; the package does not contain a replacement. Never replace `fpl-data/forecast_log.csv`.
4. Add or replace `tests/test_fpl_validation.py` at the exact path `tests/test_fpl_validation.py`. In the web editor, use **Add file → Create new file** and enter that full path if the `tests` folder does not exist.
5. Replace `.github/workflows/fpl-auto-learning.yml` with the package version. It keeps the existing **FPL Auto Learning Log** workflow name and runs the unit tests before updating data.
6. Add `.github/workflows/fpl-historical-backtest.yml` from the package. In the browser, choose **Add file → Create new file** and enter that exact path.
7. Commit the changes to `main`. Streamlit should redeploy.
8. Open **Actions → Historical FPL Backtest → Run workflow**. This manual job downloads the configured archives, runs the point-in-time proxy replay, and saves `historical_backtest_summary.csv`, `historical_backtest_gameweeks.csv`, and `historical_backtest_metadata.json` on `fpl-data`.
9. Open **Actions → FPL Auto Learning Log** and ensure scheduled runs remain green. Open Streamlit and refresh; the historical summary should appear after it has been committed to `fpl-data`.

**Data safety:** the `fpl-data` branch is the source of truth for the persistent forecast log. Leave its existing `forecast_log.csv` in place; this package intentionally does not include a replacement seed CSV. Both workflows use the same GitHub Actions concurrency group so they do not push updates to `fpl-data` simultaneously. The worker migrates old headers by adding missing columns and marks missing model versions as `legacy-unknown`; those rows are preserved but excluded from v2.9 metrics.

## Interpretation guidance

- An individual Gameweek is noisy. Do not change the model based on one lucky or unlucky result.
- Compare model, baseline and live `ep_next` for the same player/Gameweek set and forecast timestamp.
- MAE measures point error; Spearman measures ranking; top-k capture measures overlap with the highest actual scorers; minutes error diagnoses likely playing-time mistakes. No one metric should decide the model.
- Current-season five-GW validation only appears after the full horizon has completed. Older-season historical replay is a separate proxy and should not be compared directly to the live `ep_next` benchmark.
- FPL archived gameweek data is not a perfect snapshot of what was known before every historic deadline. Features are lagged, archived post-match expected-points fields are excluded, and all limitations are written to the metadata report.
- The model remains a heuristic. It does not automatically rewrite its entire strategy and does not guarantee rank gains or a top-10 finish.

## Scheduled worker

`fpl_background_worker.py` captures one snapshot during the six-hour pre-deadline window and reconciles one-GW and five-GW actuals when FPL marks the associated events finished. The worker will migrate older CSV schemas without erasing the stored rows. GitHub Actions requires repository `contents: write` permission to persist updates on `fpl-data`.

## Files

- `app.py` — Streamlit UI, projections, season context and validation report.
- `fpl_engine.py` — shared heuristic point projection and expected-minutes calculation.
- `fpl_validation.py` — MAE, bias, Spearman, top-k capture, expected-minutes metrics and captain regret where the required pre-deadline lineup data exists.
- `fpl_background_worker.py` — scheduled snapshot/reconciliation process.
- `historical_backtest.py` — lagged-feature replay across prior seasons.
- `.github/workflows/fpl-auto-learning.yml` — scheduled prospective forecast log.
- `.github/workflows/fpl-historical-backtest.yml` — manual historical backtest and report persistence.
- `tests/test_fpl_validation.py` — basic metric, leakage and fixture-count unit tests.
- The repository-root `forecast_log.csv` may remain as a legacy local fallback; do not overwrite the authoritative `fpl-data` branch log.
