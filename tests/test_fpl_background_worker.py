import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import fpl_background_worker


class BackgroundWorkerArtifactTests(unittest.TestCase):
    def test_missing_model_skips_new_snapshot_but_reconciles_existing_rows(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            log_path = Path(temp) / "forecast_log.csv"
            old = {column: None for column in fpl_background_worker.COLUMNS}
            old.update({
                "season": "2026-27", "gameweek": 1, "player_id": 7, "player": "Test",
                "position": "MID", "club": "TST", "model_version": "3.0",
                "forecast_created_utc": "2026-08-01T00:00:00+00:00",
                "actual_points_5gw": 0, "actual_minutes_5gw": 0,
            })
            pd.DataFrame([old], columns=fpl_background_worker.COLUMNS).to_csv(log_path, index=False)
            deadline = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
            bootstrap = {
                "elements": [{"id": 7, "team": 1, "element_type": 3, "status": "a", "minutes": 90}],
                "teams": [{"id": 1, "short_name": "TST"}],
                "element_types": [{"id": 3, "singular_name_short": "MID"}],
                "events": [
                    {"id": 1, "finished": True},
                    {"id": 2, "is_next": True, "finished": False, "deadline_time": deadline},
                ],
            }

            def fake_get_json(path, *args, **kwargs):
                if path == "bootstrap-static/":
                    return bootstrap
                if path == "fixtures/":
                    return []
                if path == "event/1/live/":
                    return {"elements": [{"id": 7, "stats": {"total_points": 6, "minutes": 90}}]}
                raise AssertionError(f"Unexpected API path: {path}")

            with patch.object(fpl_background_worker, "get_json", side_effect=fake_get_json), \
                 patch.object(fpl_background_worker, "season_label", return_value="2026-27"), \
                 patch.object(fpl_background_worker, "fetch_history_signals") as history_fetch, \
                 patch.object(sys, "argv", ["fpl_background_worker.py", "--csv-path", str(log_path)]):
                self.assertEqual(fpl_background_worker.main(), 0)
                history_fetch.assert_not_called()

            result = pd.read_csv(log_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(set(result["gameweek"]), {1})
            self.assertEqual(float(result.loc[0, "actual_points"]), 6.0)

    def test_unreadable_existing_log_is_preserved_and_worker_fails_closed(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            log_path = Path(temp) / "forecast_log.csv"
            original = b"existing-but-corrupt-forecast-data\n"
            log_path.write_bytes(original)
            with patch.object(sys, "argv", ["fpl_background_worker.py", "--csv-path", str(log_path)]):
                with self.assertRaisesRegex(RuntimeError, "preserving it"):
                    fpl_background_worker.main()
            self.assertEqual(log_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
