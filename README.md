The-One-Above-All FPL Manager Hub v2.7
A phone-friendly Streamlit dashboard for public FPL manager data, fixture-aware projections, transfer shortlists, chip guidance and forecast validation.
Run / deploy
Main file: `app.py`
Python requirements: `requirements.txt`
Streamlit Community Cloud entrypoint: `app.py`
No FPL password or private cookie is requested.
v2.7 additions
Fetches cached per-player match history for the recorded squad and a bounded pool of leading transfer candidates.
Blends a capped recent-points signal and recent minutes/appearance trend into the existing fixture-adjusted forecast.
Exhaustively checks legal formations from the current 15-player squad and recommends the starting XI, captain, vice-captain and bench order for the next Gameweek.
Keeps the prior heuristic as a baseline.
Automatically freezes one forecast per upcoming Gameweek when the app is opened before the deadline; once saved, it is not overwritten.
After a Gameweek is officially marked finished by FPL, retrieves actual player points/minutes and compares MAE/bias for both projections.
CSV export/restore lets you back up results because hosted app filesystems can be ephemeral.
Important limits
FPL's JSON API is public but unofficial/undocumented and may change.
Forecast logging runs when the dashboard is visited; it is not a scheduled background job. It needs at least one visit before the Gameweek deadline to record a pre-deadline forecast.
History enrichment is intentionally bounded to avoid hundreds of player API calls. Players outside this pool still receive the baseline projection until included in the pool.
Forecasts are heuristic, not a validated ML model, and do not guarantee rankings or points.
A global transfer optimizer that rebuilds the full 15-player squad in one search, and properly calibrated probability/confidence, are future steps. The current transfer shortlist tests one transfer at a time.
