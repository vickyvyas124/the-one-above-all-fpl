from __future__ import annotations

from dataclasses import replace
import unittest

from fpl_intelligence.validation import (
    WalkForwardForecast,
    evaluate_walk_forward,
)


ARTIFACT = "a" * 64


def forecasts(*, one_season_degrades=False):
    rows = []
    for season_index, season in enumerate(("2023-24", "2024-25")):
        for gameweek in range(1, 7):
            deadline = f"{2023 + season_index}-09-{gameweek + 1:02d}T12:00:00Z"
            features_at = f"{2023 + season_index}-09-{gameweek:02d}T12:00:00Z"
            trained_through = f"{2023 + season_index}-08-01T00:00:00Z"
            actual_available = f"{2023 + season_index}-09-{gameweek + 3:02d}T12:00:00Z"
            for player_id in range(1, 21):
                actual = float(player_id % 13 + gameweek % 4)
                candidate_error = 2.0 if one_season_degrades and season_index == 1 else 0.1
                rows.append(WalkForwardForecast(
                    season=season,
                    gameweek=gameweek,
                    fpl_player_id=player_id,
                    deadline_utc=deadline,
                    feature_available_at_utc=features_at,
                    candidate_trained_through_utc=trained_through,
                    baseline_points=actual + 1.0,
                    candidate_points=actual + candidate_error,
                    actual_points=actual,
                    actual_available_at_utc=actual_available,
                    candidate_version="candidate-1",
                    candidate_artifact_sha256=ARTIFACT,
                    official_ep_next=actual + 1.5,
                    ep_next_available_at_utc=f"{2023 + season_index}-09-{gameweek:02d}T10:00:00Z",
                ))
    return rows


class FootballIntelligenceValidationTests(unittest.TestCase):
    def test_no_candidate_data_fails_closed_and_never_promotes(self):
        report = evaluate_walk_forward([])
        self.assertEqual(report.status, "NO_CANDIDATE_DATA")
        self.assertFalse(report.promotion_allowed)
        self.assertIsNone(report.candidate_metrics)

    def test_repeatable_walk_forward_gain_is_only_eligible_for_human_review(self):
        report = evaluate_walk_forward(forecasts())
        self.assertEqual(report.status, "ELIGIBLE_FOR_HUMAN_REVIEW")
        self.assertFalse(report.promotion_allowed)
        self.assertEqual(report.season_count, 2)
        self.assertEqual(report.gameweek_count, 12)
        self.assertEqual(report.player_gameweek_rows, 240)
        self.assertLess(report.candidate_metrics["MAE"], report.baseline_metrics["MAE"])
        self.assertLessEqual(report.candidate_metrics["RMSE"], report.baseline_metrics["RMSE"])
        self.assertEqual(report.official_ep_next_metrics["Players"], 240)
        self.assertLess(
            report.official_ep_next_metrics["Candidate MAE on ep_next cohort"],
            report.official_ep_next_metrics["Baseline MAE on ep_next cohort"],
        )
        self.assertIn("automatic promotion is disabled", report.gate_reasons[0])

    def test_improvement_that_does_not_repeat_across_seasons_is_rejected(self):
        report = evaluate_walk_forward(forecasts(one_season_degrades=True))
        self.assertEqual(report.status, "INSUFFICIENT_OR_FAILED_EVIDENCE")
        self.assertFalse(report.promotion_allowed)
        self.assertTrue(any("every evaluated season" in reason for reason in report.gate_reasons))

    def test_post_deadline_features_training_or_actuals_are_rejected(self):
        row = forecasts()[0]
        with self.assertRaisesRegex(ValueError, "after the target deadline"):
            evaluate_walk_forward([replace(row, feature_available_at_utc="2023-09-02T12:01:00Z")])
        with self.assertRaisesRegex(ValueError, "training cutoff"):
            evaluate_walk_forward([replace(row, candidate_trained_through_utc=row.feature_available_at_utc)])
        with self.assertRaisesRegex(ValueError, "strictly after"):
            evaluate_walk_forward([replace(row, actual_available_at_utc=row.deadline_utc)])

    def test_post_deadline_ep_next_and_duplicate_rows_are_rejected(self):
        row = forecasts()[0]
        with self.assertRaisesRegex(ValueError, "ep_next became available after"):
            evaluate_walk_forward([replace(row, ep_next_available_at_utc="2023-09-02T12:01:00Z")])
        with self.assertRaisesRegex(ValueError, "Duplicate season/Gameweek/player"):
            evaluate_walk_forward([row, row])

    def test_validation_requires_one_candidate_artifact(self):
        rows = forecasts()[:2]
        rows[1] = replace(rows[1], candidate_version="candidate-2")
        with self.assertRaisesRegex(ValueError, "one candidate version"):
            evaluate_walk_forward(rows)

    def test_small_or_single_season_evidence_never_passes_the_gate(self):
        report = evaluate_walk_forward(forecasts()[:20])
        self.assertEqual(report.status, "INSUFFICIENT_OR_FAILED_EVIDENCE")
        self.assertTrue(any("at least 2 are required" in reason for reason in report.gate_reasons))


if __name__ == "__main__":
    unittest.main()
