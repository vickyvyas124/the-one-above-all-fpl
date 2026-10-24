import sys
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import fpl_background_worker
import fpl_snapshot_store


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
            deadline = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
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
                if path == "element-summary/7/":
                    return {"history": []}
                if path == "entry/2311669/event/2/picks/":
                    return {"picks": []}
                if path == "entry/2311669/":
                    return {"bank": 0, "free_transfers": None}
                if path == "event/1/live/":
                    return {"elements": [{"id": 7, "stats": {"total_points": 6, "minutes": 90}}]}
                raise AssertionError(f"Unexpected API path: {path}")

            def fake_record(path, *args, **kwargs):
                payload = fake_get_json(path)
                body = __import__("json").dumps(payload).encode()
                return payload, fpl_snapshot_store.payload_record(
                    body=body, url=f"https://example.test/{path}", status_code=200,
                    fetched_at_utc=datetime.now(timezone.utc),
                )

            snapshot_dir = Path(temp) / "snapshots"
            with patch.dict(os.environ, {"FPL_SNAPSHOT_DIR": str(snapshot_dir)}), \
                 patch.object(fpl_background_worker, "get_json", side_effect=fake_get_json), \
                 patch.object(fpl_background_worker, "get_json_with_record", side_effect=fake_record), \
                 patch.object(fpl_background_worker, "season_label", return_value="2026-27"), \
                 patch.object(sys, "argv", ["fpl_background_worker.py", "--csv-path", str(log_path)]):
                self.assertEqual(fpl_background_worker.main(), 0)

            result = pd.read_csv(log_path)
            self.assertEqual(len(result), 1)
            self.assertEqual(set(result["gameweek"]), {1})
            self.assertEqual(float(result.loc[0, "actual_points"]), 6.0)
            manifest_path = snapshot_dir / "2026-27" / "gw02" / "6h" / "manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = __import__("json").loads(manifest_path.read_text())
            self.assertFalse(manifest["forecast"]["included"])
            self.assertEqual(manifest["model"]["status"], "missing")

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
