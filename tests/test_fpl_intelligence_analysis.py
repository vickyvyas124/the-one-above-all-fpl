from __future__ import annotations

from dataclasses import replace
import unittest

from fpl_intelligence.analysis import analyze_observations
from fpl_intelligence.contract import EvidenceClass, EvidenceQuality, PlayerMatchObservation


def observation(
    index: int, *, player_id: str = "source-1", name: str = "Player One",
    club: str = "North FC", position: str = "Forward",
    minutes: float | None = 90.0, started: bool | None = True,
    xg: float | None = 0.1, shots: int | None = 1,
    xa: float | None = 0.0, key_passes: int | None = 0,
    box_events: int | None = 1, defensive: int | None = 1,
    fpl_id: int | None = 101,
) -> PlayerMatchObservation:
    missing = ["progressive_carries", "progressive_passes"]
    for field, value in (
        ("minutes", minutes), ("started", started), ("xg", xg), ("shots", shots),
        ("xa", xa), ("key_passes", key_passes), ("touches_in_box", box_events),
        ("defensive_actions", defensive),
    ):
        if value is None:
            missing.append(field)
    return PlayerMatchObservation(
        schema_version="1.0.0", source_id="statsbomb_open_data",
        source_player_id=player_id, fpl_player_id=fpl_id, season="2024/2025",
        gameweek=None, competition_round=str(index + 1), fixture_id=f"fixture-{index}",
        match_date=f"2025-01-{index + 1:02d}", player_name=name, club=club,
        opponent="South FC", position=position, minutes=minutes, started=started,
        substitute=False, shots=shots, shots_on_target=0, xg=xg, xa=xa,
        key_passes=key_passes, chances_created=key_passes,
        touches_attacking_area=box_events, touches_in_box=box_events,
        progressive_carries=None, progressive_passes=None, defensive_actions=defensive,
        role_segments=(), source_timestamp="2025-01-20T00:00:00Z",
        retrieved_at_utc="2025-01-20T01:00:00+00:00",
        source_url="https://example.test/source.json", raw_bundle_sha256="a" * 64,
        completeness=EvidenceQuality.INCOMPLETE,
        missing_fields=tuple(sorted(set(missing))),
        metric_definitions={"shots": "Test fixture measurement."},
    )


def sample_player(*, name="Player One", player_id="source-1", fpl_id=101):
    rows = []
    for index in range(6):
        recent = index >= 3
        rows.append(observation(
            index, name=name, player_id=player_id, fpl_id=fpl_id,
            position="Defender" if index < 3 else "Midfielder",
            minutes=75 if recent else 65,
            xg=0.3 if recent else 0.1,
            shots=3 if recent else 1,
            xa=0.2 if recent else 0.0,
            key_passes=2 if recent else 0,
            box_events=4 if recent else 1,
        ))
    return rows


class FootballIntelligenceAnalysisTests(unittest.TestCase):
    def test_emits_evidence_linked_inferred_trends_and_explicit_opportunity_rule(self):
        result = analyze_observations(sample_player())
        metric = {item.measurements["indicator"]: item for item in result.insights}
        xg = metric["xg_per90"]
        self.assertEqual(xg.classification, EvidenceClass.INFERRED)
        self.assertEqual(xg.measurements["baseline_match_count"], 3)
        self.assertEqual(xg.measurements["recent_match_count"], 3)
        self.assertAlmostEqual(xg.measurements["absolute_change"], 0.3 * 90 / 75 - 0.1 * 90 / 65, places=5)
        self.assertEqual(xg.fixture_ids, tuple(f"fixture-{i}" for i in range(6)))
        self.assertEqual(xg.source_player_ids, ("source-1",))
        self.assertEqual(xg.fpl_player_id, 101)
        self.assertIsNone(xg.measurements.get("points"))
        self.assertEqual(result.emerging_opportunities[0].improving_dimensions, (
            "box_involvement", "chance_creation", "goal_threat",
        ))
        self.assertEqual(result.emerging_opportunities[0].evidence_quality, EvidenceQuality.HIGH)
        opportunity = result.emerging_opportunities[0]
        self.assertEqual(opportunity.classification, EvidenceClass.INFERRED)
        self.assertEqual(opportunity.season, "2024/2025")
        self.assertEqual(opportunity.fixture_ids, tuple(f"fixture-{i}" for i in range(6)))
        self.assertTrue(opportunity.source_timestamp)
        self.assertIn("not a transfer or points recommendation", opportunity.explanation)

    def test_position_label_shift_is_reported_without_claiming_tactical_role(self):
        result = analyze_observations(sample_player())
        role = next(item for item in result.insights
                    if item.measurements["indicator"] == "listed_position_shift")
        self.assertEqual(role.measurements["baseline_dominant_position"], "Defender")
        self.assertEqual(role.measurements["recent_dominant_position"], "Midfielder")
        self.assertIn("does not establish a tactical role change", role.explanation)

    def test_minutes_variability_is_descriptive_and_not_fatigue_claim(self):
        rows = sample_player()
        for index, minutes in enumerate((60, 70, 80, 45, 75, 90)):
            rows[index] = replace(rows[index], minutes=minutes)
        result = analyze_observations(rows)
        insight = next(item for item in result.insights
                       if item.measurements["indicator"] == "minutes_variability")
        self.assertEqual(insight.classification, EvidenceClass.INFERRED)
        self.assertGreater(insight.measurements["change_in_standard_deviation"], 0)
        self.assertIn("does not establish injury, fatigue", insight.explanation)

    def test_missing_measurements_remain_unknown_and_reduce_metric_coverage(self):
        rows = sample_player()
        rows[3] = replace(rows[3], xg=None, missing_fields=tuple(sorted(
            set(rows[3].missing_fields) | {"xg"},
        )))
        result = analyze_observations(rows)
        indicators = {item.measurements["indicator"] for item in result.insights}
        self.assertIn("xg_per90", indicators)
        self.assertAlmostEqual(result.metric_coverage["xg_per90"], 5 / 6, places=4)
        self.assertIn("xa_per90", indicators)
        xg = next(item for item in result.insights if item.measurements["indicator"] == "xg_per90")
        self.assertEqual(xg.measurements["recent_match_count"], 2)
        self.assertEqual(xg.evidence_quality, EvidenceQuality.MEDIUM)

    def test_insufficient_player_history_is_skipped_not_zero_filled(self):
        result = analyze_observations(sample_player()[:3])
        self.assertEqual(result.insights, ())
        self.assertEqual(result.emerging_opportunities, ())
        self.assertEqual(result.analyzed_player_count, 0)
        self.assertTrue(result.skipped_players)

    def test_two_distinct_attacking_domains_are_required_for_emerging_shortlist(self):
        rows = sample_player()
        for index in range(3, 6):
            rows[index] = replace(
                rows[index], xa=0.0, key_passes=0, chances_created=0,
                touches_attacking_area=0, touches_in_box=0,
            )
        result = analyze_observations(rows)
        self.assertEqual(result.emerging_opportunities, ())

    def test_opportunity_order_uses_unweighted_indicator_count_and_stable_name_ties(self):
        top = sample_player(name="Zed")
        for index in range(3, 6):
            top[index] = replace(top[index], touches_attacking_area=1, touches_in_box=1)
        rows = top + sample_player(name="Ari", player_id="source-2", fpl_id=102)
        result = analyze_observations(rows)
        self.assertEqual([item.player_name for item in result.emerging_opportunities], ["Ari", "Zed"])
        self.assertEqual([len(item.improving_dimensions) for item in result.emerging_opportunities], [3, 2])
        self.assertEqual([item.rank for item in result.emerging_opportunities], [1, 2])

    def test_duplicate_observation_fails_closed(self):
        rows = sample_player()
        rows.append(rows[0])
        with self.assertRaisesRegex(ValueError, "Duplicate source/player/fixture"):
            analyze_observations(rows)

    def test_conflicting_fpl_mapping_fails_closed(self):
        rows = sample_player()
        rows[0] = replace(rows[0], fpl_player_id=999)
        with self.assertRaisesRegex(ValueError, "maps to multiple FPL IDs"):
            analyze_observations(rows)

    def test_analysis_does_not_infer_gameweek_from_competition_round(self):
        result = analyze_observations(sample_player())
        self.assertTrue(result.insights)
        self.assertTrue(all(item.measurements["season"] == "2024/2025" for item in result.insights))
        self.assertTrue(all(item.fixture_ids for item in result.insights))


if __name__ == "__main__":
    unittest.main()
