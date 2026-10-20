# The-One-Above-All FPL Optimizer v1

A free Streamlit dashboard using public, read-only Fantasy Premier League endpoints. No paid app, API key, FPL password, or account cookie is required.

## Run locally

1. Install Python 3.10 or newer from https://www.python.org/downloads/
2. Unzip this folder.
3. Open a terminal/Command Prompt inside the folder.
4. Create a virtual environment (recommended):

   Windows:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
6. Start dashboard:
   ```bash
   streamlit run app.py
   ```
7. Open the local URL Streamlit prints (usually http://localhost:8501).

## What is included

- Public player/fixture data retrieval and 30-minute cache.
- Manager profile lookup for team ID 2311669 when public endpoint is available.
- Squad seeded from the GW8 screenshot shared in ChatGPT.
- A transparent, heuristic expected-points baseline over a selectable 1–8 Gameweek horizon.
- Affordable replacement shortlist by position using current public prices plus bank.
- Status/form/PPG/minutes/ownership columns.

## Important limitations

- This is a v1 prototype. The public FPL API is unofficial and may change.
- Public API endpoints do not reveal the exact current private squad, selling prices, bench order, or private transfer state. Those remain seeded from screenshots and should be verified manually.
- Current public prices are not the same as your personal selling prices.
- Forecasts are not official predictions or guarantees. The heuristic blends form and points per game and adjusts for fixture difficulty; it is not a trained model.
- The dashboard does not log in to FPL, use cookies, or submit transfers.
- It is not yet a full 15-player mathematical squad optimizer. Next upgrades: xG/xA-based forecasts, predicted minutes/injury news, complete legal-transfer search, captain/bench optimization, saved forecast history and scheduled refresh/deployment.

## Scheduled refresh

The app refreshes its public API cache when data is older than 30 minutes and the dashboard is rerun/reloaded. It does not run in the background when your computer is off or the app is closed. For scheduled runs, deploy to a free hosting tier that supports Streamlit or run a local scheduled task; provider limits and uptime can change. Avoid adding FPL login credentials to a public deployment.

## Privacy

Only the public team ID is used to request the public manager profile. No FPL password or private account data is requested.
