from __future__ import annotations

import unittest

from fpl_intelligence.contract import EvidenceClass, EvidenceQuality, IntelligenceInsight
from fpl_intelligence.integration import build_fpl_decision_context


DEADLINE = "2025-02-01T00:00:00Z"


def insight(
    insight_id="xg-trend", *, fpl_id=101, classification=EvidenceClass.INFERRED,
    quality=EvidenceQuality.HIGH, source_time="2025-01-20T12:00:00Z",
    retrieved_time="2025-01-20T12:30:00Z", indicator="xg_per90",
    higher_is_better=True, delta=0.2,
):
    return IntelligenceInsight(
        schema_version="1.0.0", insight_id=insight_id,
        classification=classification, evidence_quality=quality,
        source_id="statsbomb_open_data", source_timestamp=source_time,
        retrieved_at_utc=retrieved_time, fixture_ids=("fixture-1", "fixture-2"),
        source_player_ids=("source-player-1",), fpl_player_id=fpl_id,
        measurements={
            "player_name": "Player One", "season": "2024/2025",
            "indicator": indicator, "absolute_change": delta,
            "higher_is_better": higher_is_better, "domain": "goal_threat",
        },
        comparison_period="recent matches versus baseline",
        explanation="Measured source data indicates a change.",
    )


class FplIntelligenceIntegrationTests(unittest.TestCase):
    def test_eligible_context_never_changes_baseline_forecast_or_decision(self):
        context = build_fpl_decision_context(
            fpl_player_id=101, target_deadline_utc=DEADLINE, baseline_points=3.6,
            official_ep_next=3.8, actual_points=7,
            actual_available_at_utc="2025-02-02T00:00:00Z",
            insights=[insight()],
        )
        self.assertEqual(context.status, "CONTEXT_ONLY_UNCALIBRATED")
        self.assertEqual(context.baseline_points, 3.6)
        self.assertIsNone(context.intelligence_adjusted_points)
        self.assertEqual(context.selected_points, 3.6)
        self.assertEqual(context.official_ep_next, 3.8)
        self.assertEqual(context.actual_points, 7.0)
        self.assertFalse(context.recommendation_changed)
        self.assertEqual(len(context.signals), 1)
        self.assertEqual(context.signals[0].trend, "positive_trend")
        self.assertEqual(context.signals[0].baseline_overlap, ("xGI_per90",))
        self.assertFalse(context.signals[0].can_change_forecast)
        self.assertIn("repeatable chronological", context.gate_reason)

    def test_unique_context_is_visible_but_does_not_change_transfer_or_points(self):
        candidate = insight(indicator="box_events_per90", higher_is_better=True)
        context = build_fpl_decision_context(
            fpl_player_id=101, target_deadline_utc=DEADLINE,
            baseline_points=4.25, insights=[candidate],
        )
        self.assertEqual(context.selected_points, 4.25)
        self.assertFalse(context.recommendation_changed)
        self.assertEqual(context.signals[0].baseline_overlap, ())

    def test_postdeadline_or_unmatched_evidence_is_excluded(self):
        items = [
            insight("wrong-player", fpl_id=202),
            insight("source-after-deadline", source_time="2025-02-02T00:00:00Z"),
            insight("retrieved-after-deadline", retrieved_time="2025-02-02T00:00:00Z"),
            insight("unverified", classification=EvidenceClass.UNVERIFIED),
            insight("reported", classification=EvidenceClass.REPORTED),
            insight("incomplete", quality=EvidenceQuality.INCOMPLETE),
        ]
        context = build_fpl_decision_context(
            fpl_player_id=101, target_deadline_utc=DEADLINE, baseline_points=2.0,
            insights=items,
        )
        self.assertEqual(context.signals, ())
        self.assertEqual(len(context.excluded_signals), len(items))
        self.assertEqual(context.status, "BASELINE_ONLY_NO_ELIGIBLE_CONTEXT")

    def test_missing_or_naive_evidence_timestamps_are_not_time_safe(self):
        item = insight(source_time="2025-01-20", retrieved_time="2025-01-20T12:30:00Z")
        context = build_fpl_decision_context(
            fpl_player_id=101, target_deadline_utc=DEADLINE, baseline_points=1.0,
            insights=[item],
        )
        self.assertEqual(context.signals, ())
        self.assertIn("time is missing or unverified", context.excluded_signals[0].reason)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            build_fpl_decision_context(
                fpl_player_id=101, target_deadline_utc="2025-02-01T00:00:00",
                baseline_points=1.0, insights=[],
            )

    def test_actual_points_must_be_available_after_prediction_deadline(self):
        with self.assertRaisesRegex(ValueError, "strictly after"):
            build_fpl_decision_context(
                fpl_player_id=101, target_deadline_utc=DEADLINE, baseline_points=2.0,
                insights=[], actual_points=8, actual_available_at_utc=DEADLINE,
            )

    def test_invalid_fpl_id_and_duplicate_insight_ids_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "positive, verified"):
            build_fpl_decision_context(
                fpl_player_id=101.5, target_deadline_utc=DEADLINE,
                baseline_points=2.0, insights=[],
            )
        repeated = insight()
        with self.assertRaisesRegex(ValueError, "Duplicate intelligence insight ID"):
            build_fpl_decision_context(
                fpl_player_id=101, target_deadline_utc=DEADLINE, baseline_points=2.0,
                insights=[repeated, repeated],
            )

    def test_non_monotonic_defensive_metric_is_not_called_positive_or_negative(self):
        item = insight(indicator="defensive_actions_per90", higher_is_better=False, delta=0.8)
        context = build_fpl_decision_context(
            fpl_player_id=101, target_deadline_utc=DEADLINE, baseline_points=2.0,
            insights=[item],
        )
        self.assertEqual(context.signals[0].trend, "increased")


if __name__ == "__main__":
    unittest.main()
