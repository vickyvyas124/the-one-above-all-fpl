# The-One-Above-All FPL Manager Hub — v2.8

A Streamlit dashboard for team analysis, fixture planning, captain/vice-captain advice, transfer HOLD/ACT guidance, chip timing and forecast validation. It uses public FPL endpoints and does not ask for FPL passwords or submit transfers to your account.

## What's new in v2.8

- **Transfer discipline:** defaults to HOLD. A move must clear a multi-Gameweek projected gain threshold, incoming-player minutes and availability reliability gates, legality/affordability checks, and a larger threshold when taking a hit or moving a captaincy-relevant player. Low ownership only changes ranking between candidates; it cannot override the points gate.
- **Timing guidance:** if a promising move is seen more than 24 hours before the deadline, the app advises monitoring rather than transferring immediately. Nearer the deadline, it asks for a final news/price check.
- **Budget sync:** attempts to derive bank and free transfers from the public manager profile, history, transfers and chip history. The free-transfer total is reconstructed, not guaranteed by a documented live API field. A manual override remains available. Confirm bank and FTs against the official FPL app before a transfer.
- **Selling-price estimate:** uses public pick purchase-price data where available and estimates the FPL sell-on price. Verify the official selling price on the Transfers page.
- **Chip timing:** compares current projected opportunities with later Gameweeks whose fixtures are already published. Future fixtures and player projections can change; use as decision support.
- **One shared XI/captain model:** the captain shortlist uses the model-selected legal XI rather than a separate old lineup when all 15 squad members are matched.
- **Scheduled forecast learning:** GitHub Actions runs every 15 minutes, captures one pre-deadline forecast in the six-hour window, and reconciles actual points/minutes after an event is marked finished. The persistent log is stored on a separate `fpl-data` branch so normal dashboard redeploys don't erase it.
- **Capped online calibration:** after at least four earlier completed Gameweeks of v2.8 forecasts, the model can apply small, position-level bias corrections. This is simple online calibration, not a trained machine-learning model. The model must be judged by backtests before relying on it.

## Deploy from the GitHub web interface

1. Extract the ZIP on your laptop.
2. Open your repository: `https://github.com/vickyvyas124/the-one-above-all-fpl`.
3. On the default branch (`main`), replace `app.py`, `requirements.txt` and `README.md` with the files from this package. Add `fpl_engine.py`, `fpl_background_worker.py` and the seed `forecast_log.csv` to the repository root.
4. Create the workflow file: choose **Add file → Create new file**, name it exactly `.github/workflows/fpl-auto-learning.yml`, then paste the contents of this package's `.github/workflows/fpl-auto-learning.yml` and commit it to `main`.
5. Ensure the workflow has write permission: in GitHub, open **Settings → Actions → General → Workflow permissions**, select **Read and write permissions**, and save if needed. The workflow needs permission to create/update the `fpl-data` branch.
6. Open the **Actions** tab, select **FPL Auto Learning Log**, and click **Run workflow** once to test it. Check the run logs and confirm it succeeds. The first successful run creates `fpl-data`; later runs update only the CSV branch, not the Streamlit app code.
7. Open the Streamlit app and refresh. Once the `fpl-data` branch exists and contains `forecast_log.csv`, the app reads that persistent log automatically. CSV downloads are optional backups, not weekly tasks.

## How the scheduled job works

- The workflow must be committed to the repository's default branch (`main`). GitHub scheduled workflows run from the default branch and may be delayed. If the workflow is disabled or fails, forecasts will not be captured reliably.
- It only creates a *prospective* snapshot in the six hours before a deadline. It never fabricates a pre-deadline prediction after a deadline has passed.
- When FPL marks a Gameweek finished, it fills actual player points and minutes for previously saved forecasts.
- The first GW8 forecast supplied by the user is preserved as seed data. Because it was created by v2.7, it has no v2.8 model-version tag and is not used to calibrate v2.8.
- The worker logs public-data errors to the GitHub Actions run. If the FPL API changes or becomes unavailable, inspect the failed run before assuming the backtest is up to date.

## Model and safety notes

The points model is currently a transparent heuristic with capped recency influence, not an official FPL projection and not a validated ML model. The online calibration only makes small, regularized adjustments after sufficient prior v2.8 observations. It does **not** automatically change formula weights every week or guarantee better points/rank. We keep a baseline and report MAE so improvements can be measured.

The auto bank/free-transfer values and selling prices can be wrong if public endpoint fields lag or omit transactions/chips. Verify them against the official FPL UI. All transfer and chip decisions are recommendations only; no changes are submitted to the user's FPL account.

## Files

- `app.py`: Streamlit dashboard.
- `fpl_engine.py`: shared point-projection and calibration functions used by the dashboard and worker.
- `fpl_background_worker.py`: scheduled forecast snapshot and result reconciliation.
- `.github/workflows/fpl-auto-learning.yml`: GitHub Actions schedule and persistence logic.
- `forecast_log.csv`: the user's existing GW8 forecast seed.
- `requirements.txt`: dashboard dependencies.
