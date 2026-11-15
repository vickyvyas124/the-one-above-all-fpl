from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

from fpl_intelligence.contract import CONTRACT_VERSION, EvidenceQuality, PlayerMatchObservation
from fpl_intelligence.copilot import (
    evaluate_user_observation,
    combine_assessments,
    list_evaluations,
    list_user_observations,
    record_user_observation,
    record_user_assessment,
    record_verified_mapping,
    resolve_verified_mapping,
)


def observation(*, source_id="statsbomb_open_data", source_player_id="101", season="2024/2025",
                shots=2, missing=(), fixture="9001"):
    return PlayerMatchObservation(
        schema_version=CONTRACT_VERSION, source_id=source_id, source_player_id=source_player_id,
        fpl_player_id=None, season=season, gameweek=None, competition_round="1",
        fixture_id=fixture, match_date="2024-08-16", player_name="Source Name",
        club="Club", opponent="Opponent", position="Forward", minutes=90,
        started=True, substitute=False, shots=shots, shots_on_target=1, xg=0.3, xa=0.1,
        key_passes=0, chances_created=0, touches_attacking_area=1, touches_in_box=1,
        progressive_carries=None, progressive_passes=None, defensive_actions=0,
        role_segments=(), source_timestamp="2024-08-16T22:00:00Z",
        retrieved_at_utc="2024-08-17T00:00:00Z", source_url="https://example.test/source",
        raw_bundle_sha256="a" * 64, completeness=EvidenceQuality.HIGH,
        missing_fields=tuple(missing),
    )


class FootballIntelligenceCopilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def mapping(self, *, season="2024/2025", fpl_id=501):
        return record_verified_mapping(
            self.root, source_id="statsbomb_open_data", season=season,
            source_player_id="101", fpl_player_id=fpl_id,
            source_player_name="Source Name", fpl_player_name="Human Confirmed Player",
            verified_by="manager", evidence="Manually checked official player identity",
            confirmed=True,
        )

    def note(self, **kwargs):
        return record_user_observation(
            self.root, text="I expect at least two shots in each observed match.",
            author="manager", season="2024/2025", fpl_player_id=501,
            metric="shots", operator="at_least", threshold=2, **kwargs,
        )

    def test_mapping_requires_explicit_confirmation_and_is_season_scoped(self):
        with self.assertRaisesRegex(ValueError, "explicitly confirmed"):
            record_verified_mapping(
                self.root, source_id="statsbomb_open_data", season="2024/2025",
                source_player_id="101", fpl_player_id=501, source_player_name="A",
                fpl_player_name="B", verified_by="manager", evidence="manual check", confirmed=False,
            )
        self.mapping()
        self.assertEqual(resolve_verified_mapping(
            self.root, source_id="statsbomb_open_data", season="2024/2025", source_player_id="101"), 501)
        self.assertIsNone(resolve_verified_mapping(
            self.root, source_id="statsbomb_open_data", season="2025/2026", source_player_id="101"))
        with self.assertRaisesRegex(ValueError, "Conflicting mapping"):
            self.mapping(fpl_id=502)
        with self.assertRaisesRegex(ValueError, "different source identity"):
            record_verified_mapping(
                self.root, source_id="statsbomb_open_data", season="2024/2025",
                source_player_id="202", fpl_player_id=501, source_player_name="Other Source Name",
                fpl_player_name="Human Confirmed Player", verified_by="manager",
                evidence="Manually checked official player identity", confirmed=True,
            )

    def test_note_is_immutable_and_unmapped_player_yields_insufficient_data(self):
        note_id = self.note()
        result = evaluate_user_observation(self.root, note_id, [observation()])
        self.assertEqual(result["verdict"], "INSUFFICIENT_DATA")
        self.assertEqual(result["channel"], "AI_RULE_BASED")
        self.assertEqual(len(list_user_observations(self.root)), 1)
        self.assertEqual(len(list_evaluations(self.root)), 1)

    def test_rule_verdicts_use_only_exact_verified_season_mapping(self):
        self.mapping()
        note_id = self.note()
        agreed = evaluate_user_observation(self.root, note_id, [observation(shots=3)])
        self.assertEqual(agreed["verdict"], "AGREE")
        disagreed = evaluate_user_observation(self.root, note_id, [observation(shots=1)])
        self.assertEqual(disagreed["verdict"], "DISAGREE")
        partial = evaluate_user_observation(self.root, note_id, [
            observation(shots=3), observation(shots=1, fixture="9002"),
        ])
        self.assertEqual(partial["verdict"], "PARTIALLY_AGREE")
        wrong_season = evaluate_user_observation(self.root, note_id, [observation(season="2023/2024")])
        self.assertEqual(wrong_season["verdict"], "INSUFFICIENT_DATA")

    def test_unknown_metric_does_not_become_zero_or_a_false_disagreement(self):
        self.mapping()
        note_id = self.note()
        result = evaluate_user_observation(
            self.root, note_id, [replace(observation(shots=None), missing_fields=("shots",))],
        )
        self.assertEqual(result["verdict"], "INSUFFICIENT_DATA")
        self.assertIsNone(result["evidence"][0]["value"])

    def test_unstructured_notes_are_stored_but_not_claimed_as_evaluated(self):
        note_id = record_user_observation(
            self.root, text="Looks more advanced today", author="manager", season="2024/2025",
        )
        result = evaluate_user_observation(self.root, note_id, [observation()])
        self.assertEqual(result["verdict"], "INSUFFICIENT_DATA")
        self.assertIsNone(list_user_observations(self.root)[0]["claim"])

    def test_user_ai_and_combined_assessments_remain_separate_and_traceable(self):
        note_id = self.note()
        user_id = record_user_assessment(self.root, note_id=note_id, verdict="AGREE")
        ai_result = evaluate_user_observation(self.root, note_id, [])
        combined = combine_assessments(
            self.root, note_id=note_id, user_verdict="AGREE", ai_verdict="INSUFFICIENT_DATA",
            user_assessment_id=user_id, ai_assessment_id=ai_result["evaluation_id"],
        )
        self.assertEqual(combined["channel"], "COMBINED")
        self.assertEqual(combined["user_verdict"], "AGREE")
        self.assertEqual(combined["ai_verdict"], "INSUFFICIENT_DATA")
        self.assertEqual(combined["ai_assessment_id"], ai_result["evaluation_id"])
        self.assertEqual(combined["verdict"], "INSUFFICIENT_DATA")
        channels = {row["channel"] for row in list_evaluations(self.root)}
        self.assertEqual(channels, {"USER", "AI_RULE_BASED", "COMBINED"})


if __name__ == "__main__":
    unittest.main()
