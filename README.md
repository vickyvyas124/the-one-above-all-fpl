# The-One-Above-All FPL Manager Hub v3.0

This release moves the historical data from being only a validation report to a usable future-Gameweek prediction model. The Historical FPL Backtest workflow trains a small standardized ridge-regression model from lagged player/Gameweek statistics, evaluates it using previous-season-only holdouts, and writes `trained_fpl_model.json` to the persistent `fpl-data` branch.

The dashboard and scheduled worker consume the same JSON model artifact. The live model uses prior-season-available/current-season-to-date features only: position, player point averages and recent points, recent minutes/appearance/start rates, xGI per 90, and the target Gameweek fixture count and home share. FPL's official fixture difficulty is applied after the learned Gameweek prediction as a modest fixture adjustment; the currently published injury/chance-of-playing status is an explicit availability override. The old heuristic remains as a fallback until a model artifact is present.

## Deploy on GitHub

Replace these root-level files on `main`: `app.py`, `fpl_engine.py`, `fpl_background_worker.py`, `historical_backtest.py`, `fpl_validation.py`, `requirements.txt`. Add the new root-level file `fpl_learning.py`. Replace both workflows with the versions under `.github/workflows/`. Keep your existing `forecast_log.csv` untouched and do not alter the `fpl-data` branch manually.

Then run **Actions → Historical FPL Backtest → Run workflow**. It should test the code, replay the historical seasons using expanding-season holdouts, and publish `trained_fpl_model.json` plus summary reports to `fpl-data`. Confirm the run is green and inspect `historical_backtest_summary.csv`; the 2023–24 row is a fallback diagnostic because no earlier season is configured for training, while 2024–25 and 2025–26 are the primary trained-model holdouts. After the artifact is committed, Streamlit should load it on refresh. The Auto Learning workflow then records v3.0 future predictions.

## Model caveats

This is a transparent, regularized regression model—not a guarantee of better FPL rank. Historical data files are not exact point-in-time copies of the public API, so the holdout evaluation reduces but does not eliminate possible dataset artifacts. The backtest must show improvement versus the baseline before we increase model complexity or make it the sole decision-maker. `ep_next` is collected prospectively in the live snapshot but excluded from historical training because the archives do not guarantee it was captured before kickoff/deadline.
