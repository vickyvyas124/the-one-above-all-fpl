from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest

from fpl_intelligence.contract import CollectionStatus
from fpl_intelligence.ingest import collect_match
from fpl_intelligence.sources.statsbomb_open_data import (
    SourceDataError,
    StatsBombOpenDataAdapter,
)
from fpl_intelligence.storage import list_fixture_revisions, persist_bundle


MATCHES = [{
    "match_id": 9001,
    "match_status": "available",
    "match_date": "2025-01-01",
    "match_week": 19,
    "season": {"season_name": "2024/2025"},
    "home_team": {"home_team_id": 10, "home_team_name": "Home FC"},
    "away_team": {"away_team_id": 20, "away_team_name": "Away FC"},
    "last_updated": "2025-01-02T00:00:00Z",
}]
EVENTS = [
    {
        "id": "shot-1", "type": {"name": "Shot"}, "player": {"id": 101, "name": "Player One"},
        "minute": 20, "second": 0, "location": [110, 40],
        "shot": {"statsbomb_xg": 0.4, "outcome": {"name": "Goal"}},
    },
    {
        "id": "pass-1", "type": {"name": "Pass"}, "player": {"id": 102, "name": "Player Two"},
        "minute": 19, "second": 0, "location": [90, 40],
        "pass": {"shot_assist": True, "goal_assist": True, "assisted_shot_id": "shot-1"},
    },
]
LINEUPS = [
    {"team_id": 10, "lineup": [
        {"player_id": 101, "player_name": "Player One", "positions": [{
            "position": "Forward", "position_id": 23, "from": "00:00", "to": "90:00",
            "start_reason": "Starting XI", "end_reason": "Final Whistle",
        }]},
        {"player_id": 102, "player_name": "Player Two", "positions": [{
            "position": "Midfielder", "position_id": 15, "from": "00:00", "to": "90:00",
            "start_reason": "Starting XI", "end_reason": "Final Whistle",
        }]},
    ]},
    {"team_id": 20, "lineup": [
        {"player_id": 201, "player_name": "Player Three", "positions": [{
            "position": "Goalkeeper", "position_id": 1, "from": "00:00", "to": "90:00",
            "start_reason": "Starting XI", "end_reason": "Final Whistle",
        }]},
    ]},
]


class FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self.content = json.dumps(payload).encode("utf-8")

    def raise_for_status(self):
        return None


class FakeSession:
    def __init__(self, *, events=EVENTS, lineups=LINEUPS, matches=MATCHES):
        self.payloads = {
            "matches/1/2.json": matches,
            "events/9001.json": events,
            "lineups/9001.json": lineups,
        }

    def get(self, url, **kwargs):
        path = url.split("/data/", 1)[-1]
        if path not in self.payloads or self.payloads[path] is None:
            response = FakeResponse({})
            response.status_code = 404
            return response
        return FakeResponse(self.payloads[path])


def adapter_for(**kwargs):
    return StatsBombOpenDataAdapter(
        session=FakeSession(**kwargs), base_url="https://example.test/data",
        player_id_map={"101": 501},
    )


class FootballIntelligencePhase1Tests(unittest.TestCase):
    def test_normalizes_measured_metrics_and_keeps_fpl_gameweek_unknown(self):
        adapter = adapter_for()
        bundle = adapter.fetch_match_bundle(1, 2, 9001)
        observations = adapter.normalize_match(bundle)
        by_source_id = {row.source_player_id: row for row in observations}

        shooter = by_source_id["101"]
        self.assertEqual(shooter.fpl_player_id, 501)
        self.assertIsNone(shooter.gameweek)
        self.assertEqual(shooter.xg, 0.4)
        self.assertEqual(shooter.shots_on_target, 1)
        self.assertEqual(shooter.minutes, 90)
        self.assertEqual(shooter.club, "Home FC")
        self.assertEqual(shooter.opponent, "Away FC")
        self.assertIsNone(shooter.progressive_carries)
        self.assertIn("progressive_carries", shooter.missing_fields)

        assister = by_source_id["102"]
        self.assertEqual(assister.key_passes, 1)
        self.assertEqual(assister.chances_created, 1)
        self.assertEqual(assister.xa, 0.4)

    def test_missing_event_payload_is_incomplete_and_not_zero_filled(self):
        adapter = StatsBombOpenDataAdapter(
            session=FakeSession(events=None), base_url="https://example.test/data",
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            result = collect_match(adapter, directory, competition_id=1, season_id=2, match_id=9001)
            self.assertEqual(result.status, CollectionStatus.DATA_INCOMPLETE)
            self.assertEqual(result.observations, 0)

    def test_duplicate_event_id_fails_closed(self):
        events = EVENTS + [dict(EVENTS[0])]
        adapter = adapter_for(events=events)
        bundle = adapter.fetch_match_bundle(1, 2, 9001)
        with self.assertRaisesRegex(SourceDataError, "event ID is duplicated"):
            adapter.normalize_match(bundle)

    def test_duplicate_season_match_id_fails_closed(self):
        adapter = adapter_for(matches=MATCHES + [dict(MATCHES[0])])
        with self.assertRaisesRegex(SourceDataError, "is duplicated"):
            adapter.fetch_match_bundle(1, 2, 9001)

    def test_unknown_match_status_is_not_treated_as_unplayed(self):
        match = dict(MATCHES[0])
        match.pop("match_status")
        adapter = adapter_for(matches=[match])
        with self.assertRaisesRegex(SourceDataError, "unknown or missing completion status"):
            adapter.fetch_match_bundle(1, 2, 9001)

    def test_same_bundle_persists_as_same_immutable_revision(self):
        adapter = adapter_for()
        bundle = adapter.fetch_match_bundle(1, 2, 9001)
        observations = adapter.normalize_match(bundle)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            # Persist the exact same snapshot twice to assert content-addressed idempotence.
            second = persist_bundle(
                directory, bundle, observations, status=CollectionStatus.DATA_INCOMPLETE.value,
            )
            first = persist_bundle(
                directory, bundle, observations, status=CollectionStatus.DATA_INCOMPLETE.value,
            )
            self.assertFalse(second["duplicate"])
            self.assertTrue(first["duplicate"])
            revisions = list_fixture_revisions(
                directory, competition_id=1, season_id=2, fixture_id=9001,
            )
            self.assertEqual(len(revisions), 1)
            manifest = revisions[0]
            raw_path = Path(directory) / manifest["raw_documents"]["events"]["store_path"]
            self.assertEqual(json.loads(gzip.decompress(raw_path.read_bytes())), EVENTS)


if __name__ == "__main__":
    unittest.main()
