from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from fpl_intelligence.contract import CONTRACT_VERSION, EvidenceQuality, PlayerMatchObservation
from fpl_intelligence.dashboard import load_dashboard_snapshot
from fpl_intelligence.sources.statsbomb_open_data import MatchBundle, RawDocument
from fpl_intelligence.storage import persist_bundle


def _observation() -> PlayerMatchObservation:
    return PlayerMatchObservation(
        schema_version=CONTRACT_VERSION,
        source_id="statsbomb_open_data",
        source_player_id="101",
        fpl_player_id=None,
        season="2024/2025",
        gameweek=None,
        competition_round="Matchweek 1",
        fixture_id="9001",
        match_date="2024-08-16",
        player_name="Example Player",
        club="Example FC",
        opponent="Other FC",
        position="Forward",
        minutes=90,
        started=True,
        substitute=False,
        shots=2,
        shots_on_target=1,
        xg=0.3,
        xa=0.0,
        key_passes=0,
        chances_created=0,
        touches_attacking_area=3,
        touches_in_box=2,
        progressive_carries=1,
        progressive_passes=0,
        defensive_actions=0,
        role_segments=(),
        source_timestamp="2024-08-16T22:00:00Z",
        retrieved_at_utc="2024-08-17T00:00:00Z",
        source_url="https://example.test/9001",
        raw_bundle_sha256="a" * 64,
        completeness=EvidenceQuality.HIGH,
    )


def _persist_fixture(root: Path) -> None:
    raw_body = b'{"fixture":9001}'
    raw = RawDocument(
        url="https://example.test/9001",
        body=raw_body,
        payload={"fixture": 9001},
        retrieved_at_utc="2024-08-17T00:00:00Z",
        source_timestamp="2024-08-16T22:00:00Z",
        sha256=hashlib.sha256(raw_body).hexdigest(),
    )
    bundle = MatchBundle(
        source_id="statsbomb_open_data",
        competition_id=2,
        season_id=27,
        match_id=9001,
        match=None,
        events=(),
        lineups=(),
        documents={"events": raw},
        source_timestamp="2024-08-16T22:00:00Z",
    )
    persist_bundle(root, bundle, [_observation()], status="data_successfully_collected")


class FootballIntelligenceDashboardTests(unittest.TestCase):
    def test_missing_dataset_returns_explicit_empty_state_without_fetching(self):
        with tempfile.TemporaryDirectory() as directory:
            snapshot = load_dashboard_snapshot(Path(directory) / "not-created")
        self.assertEqual(snapshot.status, "NO_LOCAL_DATASET")
        self.assertIsNone(snapshot.analysis)
        self.assertEqual(snapshot.fixture_count, 0)

    def test_reads_latest_local_fixture_and_reports_provenance_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _persist_fixture(root)
            snapshot = load_dashboard_snapshot(root)
        self.assertEqual(snapshot.status, "READY")
        self.assertEqual(snapshot.fixture_count, 1)
        self.assertEqual(snapshot.observation_count, 1)
        self.assertEqual(snapshot.analyzed_player_count, 0)
        self.assertEqual(snapshot.collection_status_counts, {"data_successfully_collected": 1})
        self.assertEqual(snapshot.last_data_retrieved_utc, "2024-08-17T00:00:00Z")
        self.assertIsNotNone(snapshot.analysis)

    def test_rejects_paths_that_escape_the_local_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _persist_fixture(root)
            manifest = next((root / "derived").rglob("*.manifest.json"))
            data = manifest.read_text(encoding="utf-8").replace(
                "raw/statsbomb_open_data/", "../outside/"
            )
            manifest.write_text(data, encoding="utf-8")
            snapshot = load_dashboard_snapshot(root)
        self.assertEqual(snapshot.status, "NO_ANALYZABLE_OBSERVATIONS")
        self.assertTrue(any("escapes the configured store" in warning for warning in snapshot.warnings))


if __name__ == "__main__":
    unittest.main()

