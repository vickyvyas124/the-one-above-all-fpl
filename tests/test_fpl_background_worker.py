import contextlib
import io
import sys
import os
import gzip
import hashlib
import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

import fpl_background_worker
import fpl_learning
import fpl_snapshot_store


class BackgroundWorkerArtifactTests(unittest.TestCase):
    def setUp(self):
        # Worker tests use mocked API payloads and temporary snapshot stores.
        # Do not let GitHub runner environment variables enable remote reads
        # or persistent-workflow behavior during this isolated unit suite.
        self._github_actions = patch.dict(os.environ, {"GITHUB_ACTIONS": "false"})
        self._github_actions.start()
        self.addCleanup(self._github_actions.stop)

    def test_invalid_picks_failure_metadata_discards_raw_response_and_entry_id(self):
        raw_secret = b'{"manager":"PRIVATE_MANAGER_NAME"}'
        raw_record = fpl_snapshot_store.payload_record(
            body=raw_secret,
            url="https://fantasy.premierleague.com/api/entry/2311669/event/2/picks/",
            status_code=500,
            fetched_at_utc=datetime.now(timezone.utc),
            error="request failed for 2311669 PRIVATE_MANAGER_NAME",
        )

        safe_record = fpl_background_worker.sanitized_public_picks_record(
            None, raw_record, gameweek=2
        )

        self.assertIsNone(safe_record["body"])
        safe_metadata = safe_record["metadata"]
        self.assertNotIn("2311669", safe_metadata["url"])
        self.assertNotIn("PRIVATE_MANAGER_NAME", safe_metadata["error"])
        self.assertIsNone(safe_metadata["sha256"])

    def test_workflow_failure_artifact_is_sanitized_and_non_masking(self):
        workflow = (Path(__file__).parents[1] / ".github/workflows/fpl-auto-learning.yml").read_text()
        preparation = workflow.split("- name: Prepare sanitized snapshot failure diagnostic", 1)[1]
        preparation = preparation.split("- name: Upload sanitized snapshot failure diagnostic", 1)[0]
        upload_step = workflow.split("- name: Upload sanitized snapshot failure diagnostic", 1)[1]
        upload_step = upload_step.split("- name: Persist log updates", 1)[0]

        self.assertIn("diagnostic_type=snapshot_failure", preparation)
        self.assertIn("run_id=%s", preparation)
        self.assertIn("run_attempt=%s", preparation)
        self.assertIn("capture_outcome=%s", preparation)
        self.assertIn("publish_outcome=%s", preparation)
        self.assertNotIn("FPL_SNAPSHOT_GITHUB_TOKEN", preparation)
        self.assertNotIn("FPL_DATA_DIR", preparation)
        self.assertNotIn("fpl-snapshots", preparation)
        self.assertIn("steps.snapshot-run.outcome == 'failure'", upload_step)
        self.assertIn("steps.snapshot-publish.outcome == 'failure'", upload_step)
        self.assertIn("actions/upload-artifact@v4", upload_step)
        self.assertIn("path: ${{ runner.temp }}/fpl-safe-recovery/diagnostic.txt", upload_step)
        self.assertIn("if-no-files-found: error", upload_step)
        self.assertIn("retention-days: 3", upload_step)
        self.assertIn("continue-on-error: true", upload_step)
        self.assertIn("fpl-safe-recovery/diagnostic.txt", workflow)
        self.assertNotIn("fpl-snapshots", upload_step)
        self.assertNotIn("fpl-data", upload_step)
        self.assertNotIn("FPL_SNAPSHOT_GITHUB_TOKEN", upload_step)

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
            raw_picks = {
                "active_chip": "wildcard",
                "entry_history": {
                    "manager_name": "PRIVATE_MANAGER_NAME",
                    "bank": 7,
                    "event_transfers": 3,
                },
                "picks": [{
                    "element": 7, "position": 1, "multiplier": 2,
                    "is_captain": True, "is_vice_captain": False,
                    "element_type": 3, "purchase_price": 55,
                    "private_unrelated_field": "PRIVATE_PICK_DATA",
                }],
            }

            def fake_get_json(path, *args, **kwargs):
                if path == "bootstrap-static/":
                    return bootstrap
                if path == "fixtures/":
                    return []
                if path == "element-summary/7/":
                    return {"history": []}
                if path == "entry/2311669/event/2/picks/":
                    return raw_picks
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
            self.assertNotIn("public-entry", manifest["sources"])
            picks_metadata = manifest["sources"]["public-picks"]
            self.assertNotIn("2311669", picks_metadata["url"])
            picks_body = gzip.decompress(
                (manifest_path.parent / "payloads/public-picks.json.gz").read_bytes()
            )
            expected_picks = json.dumps(
                {"event": 2, "picks": [{
                    "element": 7, "position": 1, "multiplier": 2,
                    "is_captain": True, "is_vice_captain": False,
                }]},
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode()
            self.assertEqual(picks_body, expected_picks)
            self.assertEqual(picks_metadata["sha256"], hashlib.sha256(expected_picks).hexdigest())
            stored_manifest = manifest_path.read_text()
            for sensitive in ("PRIVATE_MANAGER_NAME", "PRIVATE_PICK_DATA", "event_transfers", "purchase_price", "2311669"):
                self.assertNotIn(sensitive, stored_manifest)
            self.assertEqual(fpl_snapshot_store.verify_snapshot(manifest_path.parent), [])

    def test_unreadable_existing_log_is_preserved_and_worker_fails_closed(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            log_path = Path(temp) / "forecast_log.csv"
            original = b"existing-but-corrupt-forecast-data\n"
            log_path.write_bytes(original)
            with patch.object(sys, "argv", ["fpl_background_worker.py", "--csv-path", str(log_path)]):
                with self.assertRaisesRegex(RuntimeError, "preserving it"):
                    fpl_background_worker.main()
            self.assertEqual(log_path.read_bytes(), original)


def _valid_model_json() -> str:
    """A small but contract-valid v3.0 artifact trained on deterministic synthetic rows."""
    rng = np.random.RandomState(7)
    frame = pd.DataFrame({name: rng.rand(600) for name in fpl_learning.FEATURES})
    for name in fpl_learning.POSITION_FEATURES:
        frame[name] = 0.0
    frame.loc[:, "position_MID"] = 1.0
    frame["actual_points"] = rng.rand(600) * 6
    model = fpl_learning.fit_ridge_model(frame, source_seasons=["2023-24"])
    assert fpl_learning.validate_model(model)
    return json.dumps(model, sort_keys=True)


class WorkerModelArtifactBehaviourTests(unittest.TestCase):
    """The worker may only create forecasts from a valid model, and must otherwise
    preserve history, keep reconciling actuals and say which kind of failure occurred."""

    def setUp(self):
        self._github_actions = patch.dict(os.environ, {"GITHUB_ACTIONS": "false"})
        self._github_actions.start()
        self.addCleanup(self._github_actions.stop)

    def _run_worker(self, temp, *, model_text, bootstrap_available=True, hours_to_deadline=5.9):
        """Run main() with mocked FPL payloads. GW1 is final with one unreconciled legacy row."""
        log_path = Path(temp) / "forecast_log.csv"
        legacy = {column: None for column in fpl_background_worker.COLUMNS}
        legacy.update({
            "season": "2026-27", "gameweek": 1, "player_id": 7, "player": "Test", "position": "MID",
            "club": "TST", "model_version": "legacy-unknown",
            "forecast_created_utc": "2026-08-01T00:00:00+00:00", "projection_model": 4.2,
            "actual_points_5gw": 0, "actual_minutes_5gw": 0,
        })
        pd.DataFrame([legacy], columns=fpl_background_worker.COLUMNS).to_csv(log_path, index=False)
        model_path = Path(temp) / "trained_fpl_model.json"
        if model_text is not None:
            model_path.write_text(model_text, encoding="utf-8")
        deadline = (datetime.now(timezone.utc) + timedelta(hours=hours_to_deadline)).isoformat()
        bootstrap = {
            "elements": [{"id": 7, "web_name": "Test", "team": 1, "element_type": 3, "status": "a",
                          "minutes": 900, "points_per_game": "4.0", "form": "4.0"}],
            "teams": [{"id": 1, "short_name": "TST"}, {"id": 2, "short_name": "OPP"}],
            "element_types": [{"id": 3, "singular_name_short": "MID"}],
            "events": [{"id": 1, "finished": True},
                       {"id": 2, "is_next": True, "finished": False, "deadline_time": deadline}],
        }
        fixtures = [{"id": 201, "event": 2, "team_h": 1, "team_a": 2, "team_h_difficulty": 2,
                     "team_a_difficulty": 4, "finished": False,
                     "kickoff_time": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()}]

        def fake_get_json(path, *args, **kwargs):
            if path == "bootstrap-static/":
                return bootstrap if bootstrap_available else None
            if path == "fixtures/":
                return fixtures
            if path == "element-summary/7/":
                return {"history": []}
            if path == "entry/2311669/event/2/picks/":
                return {"picks": [{"element": 7, "position": 1, "multiplier": 1,
                                   "is_captain": False, "is_vice_captain": False}]}
            if path == "event/1/live/":
                return {"elements": [{"id": 7, "stats": {"total_points": 6, "minutes": 90}}]}
            raise AssertionError(f"Unexpected API path: {path}")

        def fake_record(path, *args, **kwargs):
            payload = fake_get_json(path)
            return payload, fpl_snapshot_store.payload_record(
                body=json.dumps(payload).encode(), url=f"https://example.test/{path}",
                status_code=200 if payload is not None else 503,
                fetched_at_utc=datetime.now(timezone.utc),
            )

        snapshot_dir = Path(temp) / "snapshots"
        stdout = io.StringIO()
        with patch.dict(os.environ, {"FPL_SNAPSHOT_DIR": str(snapshot_dir)}), \
             patch.object(fpl_background_worker, "get_json", side_effect=fake_get_json), \
             patch.object(fpl_background_worker, "get_json_with_record", side_effect=fake_record), \
             patch.object(fpl_background_worker, "season_label", return_value="2026-27"), \
             patch.object(sys, "argv", ["fpl_background_worker.py", "--csv-path", str(log_path)]), \
             contextlib.redirect_stdout(stdout):
            code = fpl_background_worker.main()
        manifest_path = snapshot_dir / "2026-27" / "gw02" / "6h" / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        return code, stdout.getvalue(), log_path, model_path, manifest

    def test_valid_model_creates_forecasts_and_preserves_existing_history(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            code, out, log_path, _, manifest = self._run_worker(temp, model_text=_valid_model_json())
            self.assertEqual(code, 0)
            result = pd.read_csv(log_path)
            new = result[result["model_version"].astype(str) == "3.0"]
            self.assertEqual(list(new["gameweek"]), [2])
            self.assertEqual(list(new["player_id"]), [7])
            self.assertTrue(new["projection_model"].notna().all())
            legacy = result[result["model_version"].astype(str) == "legacy-unknown"]
            self.assertEqual(len(legacy), 1)
            self.assertEqual(float(legacy.iloc[0]["projection_model"]), 4.2)
            self.assertEqual(float(legacy.iloc[0]["actual_points"]), 6.0)
            self.assertEqual(manifest["model"]["status"], "valid")
            self.assertTrue(manifest["forecast"]["included"])
            self.assertNotIn("DIAGNOSTIC[model-artifact]", out)

    def test_unusable_model_blocks_new_forecasts_but_history_and_reconciliation_continue(self):
        valid = json.loads(_valid_model_json())
        cases = {
            "malformed": "{truncated",
            "incompatible": json.dumps(dict(valid, model_version="2.9")),
            "invalid": json.dumps(dict(valid, training_rows=12)),
        }
        for expected_status, model_text in cases.items():
            with self.subTest(expected_status), TemporaryDirectory(dir=Path.cwd()) as temp:
                code, out, log_path, model_path, manifest = self._run_worker(temp, model_text=model_text)
                self.assertEqual(code, 0)
                result = pd.read_csv(log_path)
                self.assertEqual(set(result["model_version"].astype(str)), {"legacy-unknown"})
                self.assertEqual(list(result["gameweek"]), [1])
                self.assertEqual(float(result.iloc[0]["projection_model"]), 4.2)  # history untouched
                self.assertEqual(float(result.iloc[0]["actual_points"]), 6.0)     # reconciliation still ran
                self.assertEqual(model_path.read_text(encoding="utf-8"), model_text)  # never overwritten
                self.assertIn("DIAGNOSTIC[model-artifact]", out)
                self.assertIn(f"status={expected_status}", out)
                self.assertNotIn("DIAGNOSTIC[data-source]", out)
                self.assertEqual(manifest["model"]["status"], "invalid_or_incompatible")
                self.assertFalse(manifest["forecast"]["included"])
                self.assertIn(expected_status, manifest["model"]["reason"])

    def test_missing_model_is_reported_as_missing_with_remediation(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            code, out, log_path, model_path, manifest = self._run_worker(temp, model_text=None)
            self.assertEqual(code, 0)
            self.assertFalse(model_path.exists())  # a failed load must not create an artifact
            self.assertIn("DIAGNOSTIC[model-artifact]: status=missing", out)
            self.assertIn("Historical FPL Backtest", out)
            self.assertEqual(manifest["model"]["status"], "missing")
            self.assertEqual(set(pd.read_csv(log_path)["model_version"].astype(str)), {"legacy-unknown"})

    def test_data_source_failure_is_categorized_and_leaves_log_untouched(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            code, out, log_path, _, _ = self._run_worker(
                temp, model_text=_valid_model_json(), bootstrap_available=False)
            self.assertEqual(code, 1)
            self.assertIn("DIAGNOSTIC[data-source]", out)
            self.assertNotIn("DIAGNOSTIC[model-artifact]", out)
            result = pd.read_csv(log_path)
            self.assertEqual(len(result), 1)
            self.assertTrue(pd.isna(result.iloc[0]["actual_points"]))  # nothing reconciled or invented

    def test_diagnostic_emits_github_annotation_only_inside_actions(self):
        message = "line one\nwith 100% detail"
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}), contextlib.redirect_stdout(io.StringIO()) as captured:
            fpl_background_worker.diagnostic("model-artifact", message)
        lines = captured.getvalue().splitlines()
        self.assertEqual(lines[0], "DIAGNOSTIC[model-artifact]: line one")
        annotation = [line for line in lines if line.startswith("::warning")]
        self.assertEqual(annotation, ["::warning title=Model artifact unavailable::line one%0Awith 100%25 detail"])
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "false"}), contextlib.redirect_stdout(io.StringIO()) as captured:
            fpl_background_worker.diagnostic("data-source", "x")
        self.assertNotIn("::warning", captured.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
