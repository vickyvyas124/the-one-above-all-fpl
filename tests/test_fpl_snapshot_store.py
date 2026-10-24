import gzip
import hashlib
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import fpl_snapshot_store as snapshots


UTC = timezone.utc


def record(body, name="fixture"):
    return snapshots.payload_record(
        body=body, url=f"https://example.test/{name}", status_code=200,
        fetched_at_utc=datetime(2026, 9, 1, 12, tzinfo=UTC),
    )


class SnapshotStoreTests(unittest.TestCase):
    def test_capture_windows_are_deadline_relative_and_bounded(self):
        deadline = datetime(2026, 9, 2, 12, tzinfo=UTC)
        self.assertEqual(snapshots.window_for_deadline(deadline - timedelta(hours=24), deadline), "24h")
        self.assertEqual(snapshots.window_for_deadline(deadline - timedelta(hours=6), deadline), "6h")
        self.assertIsNone(snapshots.window_for_deadline(deadline - timedelta(hours=20), deadline))
        self.assertIsNone(snapshots.window_for_deadline(deadline + timedelta(seconds=1), deadline))

    def test_snapshot_is_immutable_and_preserves_exact_source_bytes(self):
        with TemporaryDirectory() as temp:
            captured = datetime(2026, 9, 1, 12, tzinfo=UTC)
            deadline = captured + timedelta(hours=6)
            original = b'{ "source" : "exact bytes" }\n'
            kwargs = dict(
                season="2026-27", gameweek=1, window_label="6h",
                deadline_utc=deadline, captured_at_utc=captured,
                sources={"bootstrap-static": record(original)},
                histories={7: record(b'{"history":[]}\n', "history/7")},
                model_status="unavailable", model_reason="artifact missing",
            )
            path, created = snapshots.capture_snapshot(temp, **kwargs)
            self.assertTrue(created)
            self.assertEqual(gzip.decompress((path / "payloads/bootstrap-static.json.gz").read_bytes()), original)
            self.assertEqual(snapshots.verify_snapshot(path), [])
            replacement = {**kwargs, "sources": {"bootstrap-static": record(b"replacement")}}
            same_path, created_again = snapshots.capture_snapshot(temp, **replacement)
            self.assertFalse(created_again)
            self.assertEqual(path, same_path)
            self.assertEqual(gzip.decompress((path / "payloads/bootstrap-static.json.gz").read_bytes()), original)

    def test_deadline_passed_capture_is_rejected(self):
        with TemporaryDirectory() as temp:
            captured = datetime(2026, 9, 1, 12, tzinfo=UTC)
            with self.assertRaisesRegex(ValueError, "at or after"):
                snapshots.capture_snapshot(
                    temp, season="2026-27", gameweek=1, window_label="6h",
                    deadline_utc=captured, captured_at_utc=captured,
                    sources={}, histories={}, model_status="unavailable",
                )

    def test_partial_sources_and_history_gaps_are_explicit(self):
        with TemporaryDirectory() as temp:
            captured = datetime(2026, 9, 1, 12, tzinfo=UTC)
            path, _ = snapshots.capture_snapshot(
                temp, season="2026-27", gameweek=1, window_label="24h",
                deadline_utc=captured + timedelta(hours=24), captured_at_utc=captured,
                sources={"bootstrap-static": record(b"{}"),
                         "fixtures": snapshots.payload_record(
                             body=None, url="https://example.test/fixtures", status_code=503,
                             fetched_at_utc=captured, error="service unavailable")},
                histories={10: record(b"{}"), 11: None}, model_status="unavailable",
                model_reason="invalid artifact",
            )
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["sources"]["fixtures"]["fetch_outcome"], "error")
            self.assertEqual(manifest["player_history"]["requested"], 2)
            self.assertEqual(manifest["player_history"]["available"], 1)
            self.assertEqual(manifest["model"]["reason"], "invalid artifact")
            self.assertIn("history.11", manifest["missing_fields"])

    def test_forecast_requires_valid_pinned_artifact_and_replay_inputs_verify(self):
        with TemporaryDirectory() as temp:
            captured = datetime(2026, 9, 1, 12, tzinfo=UTC)
            model = b'{"weights": [1, 2]}\n'
            digest = hashlib.sha256(model).hexdigest()
            base = dict(
                season="2026-27", gameweek=1, window_label="6h",
                deadline_utc=captured + timedelta(hours=6), captured_at_utc=captured,
                sources={"bootstrap-static": record(b"{}"), "fixtures": record(b"[]")},
                histories={7: record(b'{"history":[]}')},
                forecast_rows=[{"player_id": 7, "projection_model": 4.5}],
            )
            with self.assertRaisesRegex(ValueError, "valid model"):
                snapshots.capture_snapshot(temp, **base, model_status="unavailable")
            path, created = snapshots.capture_snapshot(
                temp, **base, model_status="valid", model_sha256=digest,
                model_artifact_body=model, code_commit_sha="abc123",
            )
            self.assertTrue(created)
            self.assertEqual(snapshots.verify_snapshot(path), [])
            self.assertEqual(gzip.decompress((path / "model.json.gz").read_bytes()), model)
            self.assertEqual(gzip.decompress((path / "forecast.json.gz").read_bytes()),
                             b'[{"player_id":7,"projection_model":4.5}]')

    def test_invalid_model_is_hashed_but_never_stored_as_a_forecast_artifact(self):
        with TemporaryDirectory() as temp:
            captured = datetime(2026, 9, 1, 12, tzinfo=UTC)
            invalid_model = b'{"wrong_version": true}'
            digest = hashlib.sha256(invalid_model).hexdigest()
            path, _ = snapshots.capture_snapshot(
                temp, season="2026-27", gameweek=1, window_label="6h",
                deadline_utc=captured + timedelta(hours=6), captured_at_utc=captured,
                sources={}, histories={}, model_status="invalid_or_incompatible",
                model_reason="incompatible artifact", model_sha256=digest,
            )
            manifest = json.loads((path / "manifest.json").read_text())
            self.assertEqual(manifest["model"]["sha256"], digest)
            self.assertFalse(manifest["model"]["artifact_captured"])
            self.assertIn("model.artifact", manifest["missing_fields"])
            self.assertFalse((path / "model.json.gz").exists())
            self.assertFalse((path / "forecast.json.gz").exists())
            self.assertEqual(snapshots.verify_snapshot(path), [])

    def test_corrupt_gzip_and_hash_are_reported(self):
        with TemporaryDirectory() as temp:
            captured = datetime(2026, 9, 1, 12, tzinfo=UTC)
            path, _ = snapshots.capture_snapshot(
                temp, season="2026-27", gameweek=1, window_label="6h",
                deadline_utc=captured + timedelta(hours=6), captured_at_utc=captured,
                sources={"fixtures": record(b"[]")}, histories={}, model_status="unavailable",
            )
            (path / "payloads/fixtures.json.gz").write_bytes(b"invalid gzip")
            self.assertTrue(any("fixtures: unreadable gzip" in error for error in snapshots.verify_snapshot(path)))
            (path / "payloads/fixtures.json.gz").write_bytes(gzip.compress(b"different valid body"))
            self.assertIn("fixtures: SHA-256 mismatch", snapshots.verify_snapshot(path))

    def test_deadline_move_between_windows_is_diagnostic_and_snapshots_stay_immutable(self):
        with TemporaryDirectory() as temp:
            first_deadline = datetime(2026, 9, 2, 12, tzinfo=UTC)
            first_capture = first_deadline - timedelta(hours=24)
            first_path, _ = snapshots.capture_snapshot(
                temp, season="2026-27", gameweek=3, window_label="24h",
                deadline_utc=first_deadline, captured_at_utc=first_capture,
                sources={}, histories={}, model_status="unavailable",
            )
            moved_deadline = first_deadline + timedelta(hours=1)
            second_capture = moved_deadline - timedelta(hours=6)
            snapshots.capture_snapshot(
                temp, season="2026-27", gameweek=3, window_label="6h",
                deadline_utc=moved_deadline, captured_at_utc=second_capture,
                sources={}, histories={}, model_status="unavailable",
            )
            self.assertTrue(snapshots.record_deadline_move(
                temp, season="2026-27", gameweek=3, current_window="6h",
                deadline_utc=moved_deadline, at_utc=second_capture,
            ))
            self.assertEqual(json.loads((first_path / "manifest.json").read_text())["deadline_utc"],
                             first_deadline.isoformat(timespec="seconds"))
            diagnostic = [json.loads(line) for line in (Path(temp) / "diagnostics.jsonl").read_text().splitlines()]
            self.assertEqual(diagnostic[0]["kind"], "deadline_moved")

    def test_deadline_move_after_first_capture_in_same_window_is_detected(self):
        with TemporaryDirectory() as temp:
            old_deadline = datetime(2026, 9, 2, 12, tzinfo=UTC)
            captured = old_deadline - timedelta(hours=24)
            snapshots.capture_snapshot(
                temp, season="2026-27", gameweek=5, window_label="24h",
                deadline_utc=old_deadline, captured_at_utc=captured,
                sources={}, histories={}, model_status="unavailable",
            )
            self.assertTrue(snapshots.record_deadline_move(
                temp, season="2026-27", gameweek=5, current_window="24h",
                deadline_utc=old_deadline + timedelta(hours=2), at_utc=captured,
            ))

    def test_missed_capture_windows_are_reported_once(self):
        with TemporaryDirectory() as temp:
            now = datetime(2026, 9, 1, 12, tzinfo=UTC)
            self.assertEqual(snapshots.record_missed_windows(
                temp, season="2026-27", gameweek=4, hours_to_deadline=22, at_utc=now), ["24h"])
            self.assertEqual(snapshots.record_missed_windows(
                temp, season="2026-27", gameweek=4, hours_to_deadline=4, at_utc=now), ["6h"])
            self.assertEqual(snapshots.record_missed_windows(
                temp, season="2026-27", gameweek=4, hours_to_deadline=4, at_utc=now), [])

    def test_double_gameweek_announced_later_does_not_rewrite_earlier_fixture_context(self):
        with TemporaryDirectory() as temp:
            deadline = datetime(2026, 9, 2, 12, tzinfo=UTC)
            early = deadline - timedelta(hours=24)
            late = deadline - timedelta(hours=6)
            early_fixtures = b'[{"id":10,"event":2}]'
            later_fixtures = b'[{"id":10,"event":2},{"id":11,"event":2}]'
            path_24, _ = snapshots.capture_snapshot(
                temp, season="2026-27", gameweek=2, window_label="24h",
                deadline_utc=deadline, captured_at_utc=early,
                sources={"fixtures": record(early_fixtures)}, histories={}, model_status="unavailable",
            )
            snapshots.capture_snapshot(
                temp, season="2026-27", gameweek=2, window_label="6h",
                deadline_utc=deadline, captured_at_utc=late,
                sources={"fixtures": record(later_fixtures)}, histories={}, model_status="unavailable",
            )
            self.assertEqual(gzip.decompress((path_24 / "payloads/fixtures.json.gz").read_bytes()), early_fixtures)

    def test_workflow_persistent_publish_is_main_only_and_uses_snapshot_branch(self):
        root = Path(__file__).parents[1]
        worker_workflow = (root / ".github/workflows/fpl-auto-learning.yml").read_text()
        backtest_workflow = (root / ".github/workflows/fpl-historical-backtest.yml").read_text()
        self.assertIn("if: github.ref == 'refs/heads/main'", worker_workflow)
        self.assertIn("Persist new snapshots to dedicated branch", worker_workflow)
        self.assertIn("HEAD:fpl-snapshots", worker_workflow)
        self.assertIn("steps.snapshot-run.outputs.changed == 'true'", worker_workflow)
        self.assertIn("if: github.ref == 'refs/heads/main'", backtest_workflow)

    def test_remote_snapshot_metadata_uses_injected_api_responses(self):
        from unittest.mock import patch
        import fpl_background_worker

        manifest = {"identity": {"season": "2026-27", "gameweek": 9, "window_label": "24h"},
                    "deadline_utc": "2026-10-01T12:00:00+00:00"}

        class Response:
            status_code = 200
            content = json.dumps(manifest).encode()

            def raise_for_status(self):
                return None

        def fake_get(url, **kwargs):
            if "/24h/manifest.json" in url:
                return Response()
            missing = Response()
            missing.status_code = 404
            missing.content = b""
            return missing

        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "vickyvyas124/the-one-above-all-fpl"}), \
             patch.object(fpl_background_worker.requests, "get", side_effect=fake_get):
            manifests, diagnostics, errors = fpl_background_worker.fetch_remote_snapshot_metadata("2026-27", 9)
        self.assertEqual(manifests["24h"], manifest)
        self.assertEqual(diagnostics, [])
        self.assertEqual(errors, {})

    def test_remote_actual_ledger_restore_uses_injected_api_response(self):
        from unittest.mock import patch
        import fpl_background_worker

        body = b'{"record_type":"actual_score","player_id":7}\n'

        class Response:
            status_code = 200
            content = body

            def raise_for_status(self):
                return None

        with TemporaryDirectory() as temp:
            with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REPOSITORY": "vickyvyas124/the-one-above-all-fpl"}), \
                 patch.object(fpl_background_worker.requests, "get", return_value=Response()):
                ok, error = fpl_background_worker.restore_remote_actual_ledger(temp, "2026-27", 2)
            ledger = Path(temp) / "actuals/2026-27/gw02.jsonl"
            self.assertTrue(ok)
            self.assertIsNone(error)
            self.assertEqual(ledger.read_bytes(), body)

    def test_finished_is_not_final_until_data_checked_when_available(self):
        self.assertFalse(snapshots.event_scores_final({"finished": True, "data_checked": False}))
        self.assertTrue(snapshots.event_scores_final({"finished": True, "data_checked": True}))
        self.assertTrue(snapshots.event_scores_final({"finished": True}))
        self.assertFalse(snapshots.event_scores_final({"finished": False, "data_checked": True}))

    def test_finalized_actual_score_changes_are_append_only_revisions(self):
        with TemporaryDirectory() as temp:
            first_time = datetime(2026, 9, 1, 12, tzinfo=UTC)
            with self.assertRaisesRegex(ValueError, "data_checked"):
                snapshots.append_finalized_actuals(
                    temp, season="2026-27", gameweek=1, actuals={7: (5, 90)},
                    data_checked=False, reconciled_at_utc=first_time,
                )
            self.assertEqual(snapshots.append_finalized_actuals(
                temp, season="2026-27", gameweek=1, actuals={7: (5, 90)},
                data_checked=True, reconciled_at_utc=first_time,
            ), 1)
            forecast_path = Path(temp) / "2026-27/gw01/6h/manifest.json"
            forecast_path.parent.mkdir(parents=True)
            forecast_path.write_text('{"original_forecast":true}')
            original_forecast = forecast_path.read_bytes()
            self.assertEqual(snapshots.append_finalized_actuals(
                temp, season="2026-27", gameweek=1, actuals={7: (5, 90)},
                data_checked=True, reconciled_at_utc=first_time + timedelta(hours=24),
            ), 0)
            self.assertEqual(snapshots.append_finalized_actuals(
                temp, season="2026-27", gameweek=1, actuals={7: (6, 90)},
                data_checked=True, reconciled_at_utc=first_time + timedelta(hours=48),
            ), 1)
            ledger = Path(temp) / "actuals/2026-27/gw01.jsonl"
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            revisions = [row for row in rows if row["record_type"] == "actual_score"]
            self.assertEqual([row["revision"] for row in revisions], [1, 2])
            self.assertEqual([row["actual_points"] for row in revisions], [5, 6])
            self.assertEqual([row["previous_actual_points"] for row in revisions], [None, 5])
            self.assertEqual(len([row for row in rows if row["record_type"] == "reconciliation"]), 3)
            self.assertFalse(snapshots.actual_reconciliation_due(
                temp, season="2026-27", gameweek=1,
                now=first_time + timedelta(hours=71),
            ))
            self.assertTrue(snapshots.actual_reconciliation_due(
                temp, season="2026-27", gameweek=1,
                now=first_time + timedelta(hours=72),
            ))
            self.assertEqual(forecast_path.read_bytes(), original_forecast)

    def test_reconciliation_index_survives_ephemeral_worker_directories(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "source"
            runner = Path(temp) / "runner"
            first_time = datetime(2026, 9, 1, 12, tzinfo=UTC)
            snapshots.append_finalized_actuals(
                source, season="2026-27", gameweek=8, actuals={7: (4, 90)},
                data_checked=True, reconciled_at_utc=first_time,
            )
            runner.mkdir()
            (runner / "reconciliation-index.json").write_bytes(
                (source / "reconciliation-index.json").read_bytes()
            )
            self.assertFalse(snapshots.actual_reconciliation_due(
                runner, season="2026-27", gameweek=8,
                now=first_time + timedelta(hours=23),
            ))
            self.assertTrue(snapshots.actual_reconciliation_due(
                runner, season="2026-27", gameweek=8,
                now=first_time + timedelta(hours=24),
            ))


if __name__ == "__main__":
    unittest.main(verbosity=2)
