import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from fpl_validation import evaluate_log, score_gameweek_group, spearman, top_k_capture
import historical_backtest
import fpl_engine
import fpl_learning


class ValidationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            "season": ["2026-27"] * 4,
            "gameweek": [9] * 4,
            "player_id": [11, 12, 13, 14],
            "projection_model": [9.0, 8.0, 2.0, 1.0],
            "projection_baseline": [5.0, 6.0, 4.0, 3.0],
            "projection_ep_next": [10.0, 7.0, None, 1.0],
            "projection_expected_minutes": [85.0, 80.0, 10.0, 25.0],
            "actual_points": [10.0, 8.0, 0.0, 3.0],
            "actual_minutes": [90.0, 90.0, 0.0, 20.0],
            "history_enriched": [True, True, False, True],
        })

    def test_spearman_perfect_order(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_top_k_capture(self):
        value = top_k_capture([10, 9, 1, 0], [9, 10, 0, 1], [1, 2, 3, 4], k=2)
        self.assertAlmostEqual(value, 1.0)

    def test_ep_metrics_use_same_ep_cohort(self):
        metrics = score_gameweek_group(self.frame)
        self.assertEqual(metrics["Players"], 4)
        self.assertAlmostEqual(metrics["EP next coverage %"], 75.0)
        self.assertAlmostEqual(metrics["Model MAE · EP cohort"], 1.0)
        self.assertAlmostEqual(metrics["EP next MAE"], 1.0)
        self.assertAlmostEqual(metrics["Expected-minutes MAE"], 7.5)
        self.assertEqual(metrics["Minutes scored players"], 4)

    def test_evaluate_log_groups_by_gameweek(self):
        two = self.frame.copy()
        two["gameweek"] = 10
        two["actual_points"] = two["actual_points"] + 1
        all_rows = pd.concat([self.frame, two], ignore_index=True)
        scored = evaluate_log(all_rows)
        self.assertEqual(len(scored), 2)
        self.assertTrue((scored["Players"] == 4).all())
        self.assertIn("History coverage %", scored.columns)

    def test_captain_regret_only_when_eligible_xi_is_logged(self):
        # A full legal XI is represented by the rows flagged as XI-eligible.
        frame = pd.DataFrame({
            "player_id": list(range(1, 13)),
            "projection_model": [8.0] * 12,
            "projection_baseline": [7.0] * 12,
            "actual_points": [5.0, 3.0, 4.0, 2.0, 6.0, 1.0, 7.0, 0.0, 2.0, 3.0, 4.0, 9.0],
            "actual_minutes": [90.0] * 12,
            "xi_eligible": [True] * 11 + [False],
            "model_captain_pick": [True] + [False] * 11,
        })
        metrics = score_gameweek_group(frame)
        self.assertEqual(metrics["Captain regret coverage"], 1)
        self.assertAlmostEqual(metrics["Model captain actual pts"], 5.0)
        self.assertAlmostEqual(metrics["Best XI actual captain pts"], 7.0)
        self.assertAlmostEqual(metrics["Captain regret (points)"], 2.0)

    def test_horizon_played_only_uses_horizon_minutes(self):
        frame = pd.DataFrame({
            "season": ["2026-27"] * 3, "gameweek": [9] * 3,
            "player_id": [1, 2, 3],
            "projection_model_5gw": [10.0, 8.0, 4.0],
            "projection_baseline_5gw": [9.0, 7.0, 5.0],
            "actual_points_5gw": [12.0, 0.0, 6.0],
            "actual_minutes": [0.0, 90.0, 0.0],
            "actual_minutes_5gw": [180.0, 0.0, 90.0],
        })
        metrics = evaluate_log(
            frame, actual_col="actual_points_5gw", model_col="projection_model_5gw",
            baseline_col="projection_baseline_5gw", benchmark_col="__none__",
            include_minutes_metrics=False, include_captain_metrics=False,
        )
        # Horizon players 1 and 3 played, despite their one-GW minutes being 0.
        self.assertEqual(int(metrics.loc[0, "Players who played"]), 2)

    def test_horizon_metrics_can_exclude_one_gameweek_diagnostics(self):
        frame = self.frame.copy()
        frame["actual_points_5gw"] = [20, 15, 5, 2]
        frame["projection_model_5gw"] = [21, 14, 4, 3]
        frame["projection_baseline_5gw"] = [19, 14, 6, 2]
        frame["xi_eligible"] = [True, True, True, True]
        frame["model_captain_pick"] = [True, False, False, False]
        metrics = evaluate_log(
            frame, actual_col="actual_points_5gw", model_col="projection_model_5gw",
            baseline_col="projection_baseline_5gw", benchmark_col="__no_ep_next__",
            include_minutes_metrics=False, include_captain_metrics=False,
        )
        self.assertIn("Model MAE", metrics.columns)
        self.assertNotIn("Expected-minutes MAE", metrics.columns)
        self.assertNotIn("Captain regret (points)", metrics.columns)

    def test_captain_regret_not_invented_without_complete_squad_flags(self):
        frame = self.frame.copy()
        frame["xi_eligible"] = None
        frame["model_captain_pick"] = None
        metrics = score_gameweek_group(frame)
        self.assertEqual(metrics["Captain regret coverage"], 0)
        self.assertTrue(pd.isna(metrics["Captain regret (points)"]))

    def test_expected_minutes_scales_with_blank_and_double_gameweeks(self):
        base = {
            "minutes": 900, "starts": 10, "substitute_appearances": 0,
            "appearances": 10, "status": "a", "chance_of_playing_next_round": 100,
        }
        single = fpl_engine.estimate_expected_minutes(base, fixture_count=1)
        double = fpl_engine.estimate_expected_minutes(base, fixture_count=2)
        blank = fpl_engine.estimate_expected_minutes(base, fixture_count=0)
        self.assertAlmostEqual(double, single * 2)
        self.assertEqual(blank, 0.0)
        fixtures = [
            {"event": 9, "team_h": 1, "team_a": 2, "finished": False},
            {"event": 9, "team_h": 3, "team_a": 1, "finished": False},
            {"event": 9, "team_h": 4, "team_a": 5, "finished": False},
        ]
        self.assertEqual(fpl_engine.team_fixture_count(fixtures, 1, 9), 2)
        self.assertEqual(fpl_engine.team_fixture_count(fixtures, 1, 10), 0)

    def test_expected_minutes_reflects_availability(self):
        base = {
            "minutes": 900, "starts": 10, "substitute_appearances": 0,
            "appearances": 10, "status": "a", "chance_of_playing_next_round": 100,
        }
        doubtful = {**base, "status": "d", "chance_of_playing_next_round": 50}
        self.assertGreater(fpl_engine.estimate_expected_minutes(base), fpl_engine.estimate_expected_minutes(doubtful))


class HistoricalBacktestTests(unittest.TestCase):
    def synthetic_raw(self):
        rows = []
        # Each team has two scheduled fixture IDs in GW2. Each player's points are
        # deliberately changed in one test to verify the forecast itself is lagged.
        for player_id, team_code, position in [(1, "A", "FWD"), (2, "A", "MID"), (3, "B", "DEF"), (4, "B", "GKP")]:
            for gw in range(1, 5):
                fixture_ids = [f"{team_code}-{gw}-1", f"{team_code}-{gw}-2"] if gw == 2 else [f"{team_code}-{gw}-1"]
                for fixture_no, fixture_id in enumerate(fixture_ids, start=1):
                    rows.append({
                        "element": player_id, "round": gw,
                        "total_points": float(player_id + gw + fixture_no),
                        "minutes": 90 if (player_id + gw + fixture_no) % 3 else 0,
                        "starts": 1 if (player_id + gw + fixture_no) % 3 else 0,
                        "element_type": position, "expected_goals": .1 * player_id,
                        "expected_assists": .05 * player_id, "was_home": fixture_no % 2 == 1,
                        "fixture": fixture_id, "team_code": team_code, "web_name": f"P{player_id}",
                        "kickoff_time": (datetime(2025, 8, 1, tzinfo=timezone.utc)
                                         + timedelta(days=(gw - 1) * 7 + fixture_no - 1)).isoformat(),
                    })
        return pd.DataFrame(rows)

    def test_team_schedule_fixture_count_not_individual_player_row_count(self):
        raw = self.synthetic_raw()
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            grouped = historical_backtest.season_frame("2025-26", "synthetic")
        gw2 = grouped[grouped["gameweek"] == 2]
        self.assertTrue((gw2["fixture_count"] == 2).all())
        self.assertEqual(gw2["player_id"].nunique(), 4)
        panel = historical_backtest.add_point_in_time_features(grouped)
        self.assertFalse((historical_backtest._single_fixture_rows(panel)["gameweek"] == 2).any())

    def test_known_schedule_blank_gameweek_is_not_assumed_single_fixture(self):
        raw = self.synthetic_raw()
        # Team A has no fixture in GW3; the archive still has player rows for it,
        # but their fixture ID is blank. This must produce a zero fixture multiplier.
        blank_mask = (raw["team_code"] == "A") & (raw["round"] == 3)
        raw.loc[blank_mask, "fixture"] = None
        raw.loc[blank_mask, "total_points"] = 0
        raw.loc[blank_mask, "minutes"] = 0
        raw.loc[blank_mask, "expected_goals"] = 0
        raw.loc[blank_mask, "expected_assists"] = 0
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            grouped = historical_backtest.season_frame("2025-26", "synthetic")
            panel = historical_backtest.add_point_in_time_features(grouped)
        blank = panel[(panel["team_key"] == "A") & (panel["gameweek"] == 3)]
        self.assertTrue((blank["fixture_count"] == 0).all())
        # The proxy does not leak the final target schedule into a prediction.
        self.assertTrue((blank["prediction_model"] >= 0).all())
        self.assertTrue((blank["prediction_expected_minutes"] >= 0).all())
        self.assertTrue((blank["fixture_count_feature"] == 1).all())

    def test_fixture_code_lookup_supplies_missing_gameweek_column(self):
        raw = self.synthetic_raw().rename(columns={"fixture": "fixture_code"})
        fixture_table = (raw[["fixture_code", "round", "kickoff_time"]].drop_duplicates()
                         .rename(columns={"fixture_code": "code", "round": "event"}))
        raw = raw.drop(columns=["round", "kickoff_time"])
        # The 2025-26 archive is missing round/event; its companion fixture table
        # maps each fixture_code to the official FPL event/Gameweek and kickoff.
        with patch.object(historical_backtest, "fetch_csv", side_effect=[raw, fixture_table]):
            grouped = historical_backtest.season_frame("2025-26", "synthetic-player-file")
        self.assertEqual(set(grouped["gameweek"].unique()), {1, 2, 3, 4})
        gw2 = grouped[grouped["gameweek"] == 2]
        self.assertTrue((gw2["fixture_count"] == 2).all())
        self.assertEqual(gw2["player_id"].nunique(), 4)

    def test_target_gw_points_do_not_leak_into_target_prediction(self):
        with patch.object(historical_backtest, "fetch_csv", return_value=self.synthetic_raw()):
            grouped = historical_backtest.season_frame("2025-26", "synthetic")
        original = historical_backtest.add_point_in_time_features(grouped)
        altered = grouped.copy()
        target = (altered["player_id"] == 1) & (altered["gameweek"] == 3)
        altered.loc[target, ["actual_points", "minutes", "xg", "xa"]] = [999, 90, 9, 9]
        altered.loc[target, ["fixture_count", "was_home"]] = [3, 1.0]
        altered_features = historical_backtest.add_point_in_time_features(altered)
        original_prediction = original.loc[(original["player_id"] == 1) & (original["gameweek"] == 3), "prediction_model"].iloc[0]
        altered_prediction = altered_features.loc[(altered_features["player_id"] == 1) & (altered_features["gameweek"] == 3), "prediction_model"].iloc[0]
        self.assertAlmostEqual(float(original_prediction), float(altered_prediction))

    def test_proxy_report_compares_common_cohort_and_exposes_gameweek_bootstrap_intervals(self):
        with patch.object(historical_backtest, "fetch_csv", return_value=self.synthetic_raw()):
            grouped = historical_backtest.season_frame("2025-26", "synthetic")
        panel = historical_backtest.add_point_in_time_features(grouped)
        summary, gameweeks = historical_backtest._score_panel(
            "2025-26", panel, None, "heuristic_fallback_no_prior_seasons", "synthetic"
        )
        row = summary.iloc[0]
        self.assertEqual(row["evaluation_type"], historical_backtest.EVALUATION_TYPE)
        self.assertEqual(row["player_gameweeks_evaluated"], int(gameweeks["Players"].sum()))
        for method in ("trained_model", "season_average", "last5_gameweek_average", "production_fallback_proxy"):
            self.assertIn(f"{method}_model_mae_mean_by_gw", row.index)
            self.assertIn(f"{method}_model_mae_ci95_low", row.index)
            self.assertIn(f"{method}_model_mae_ci95_high", row.index)
        self.assertIn("production_fallback_formula", row.index)
        self.assertTrue((gameweeks["Players"] > 0).all())

    def test_training_and_live_features_match_for_same_deadline_safe_history(self):
        raw = self.synthetic_raw()
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            grouped = historical_backtest.season_frame("2025-26", "synthetic")
        panel = historical_backtest.add_point_in_time_features(grouped)
        target = panel[(panel["player_id"] == 1) & (panel["gameweek"] == 4)].iloc[0]
        target_cutoff = target["_target_cutoff"]
        historical_records = raw[(raw["element"] == 1) & (raw["round"] < 4)].to_dict("records")
        live_signal = fpl_engine.make_history_signal({"history": historical_records}, cutoff_utc=target_cutoff)
        position = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[target["position_prev"]]
        live_features = fpl_learning.make_feature_row(
            position=position, history_signal=live_signal, fixture_count=1, home_share=.5
        )
        for feature in fpl_learning.FEATURES:
            self.assertAlmostEqual(float(live_features[feature]), float(target[feature]), places=8, msg=feature)

    def test_late_prior_fixture_stats_are_excluded_until_available(self):
        raw = self.synthetic_raw()
        late = (raw["round"] == 2) & (raw["fixture"] == "A-2-2")
        raw.loc[late, "kickoff_time"] = datetime(2025, 8, 20, tzinfo=timezone.utc).isoformat()
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            grouped = historical_backtest.season_frame("2025-26", "synthetic")
        first = historical_backtest.add_point_in_time_features(grouped)
        changed = grouped.copy()
        changed.loc[(changed["player_id"] == 1) & (changed["gameweek"] == 2), "actual_points"] += 500
        second = historical_backtest.add_point_in_time_features(changed)
        p1 = first.loc[(first["player_id"] == 1) & (first["gameweek"] == 3), "prediction_model"].iloc[0]
        p2 = second.loc[(second["player_id"] == 1) & (second["gameweek"] == 3), "prediction_model"].iloc[0]
        self.assertAlmostEqual(float(p1), float(p2))

    def test_fixtureless_nonzero_stats_fail_closed(self):
        raw = self.synthetic_raw()
        raw.loc[0, "fixture"] = None
        raw.loc[0, "total_points"] = 3
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            with self.assertRaisesRegex(ValueError, "without a fixture identifier"):
                historical_backtest.season_frame("2025-26", "synthetic")

    def test_missing_player_identifier_fails_closed(self):
        raw = self.synthetic_raw()
        raw.loc[0, "element"] = None
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            with self.assertRaisesRegex(ValueError, "player identifiers"):
                historical_backtest.season_frame("2025-26", "synthetic")

    def test_missing_team_identifier_fails_closed(self):
        raw = self.synthetic_raw()
        raw.loc[0, "team_code"] = None
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            with self.assertRaisesRegex(ValueError, "team identifiers"):
                historical_backtest.season_frame("2025-26", "synthetic")

    def test_duplicate_player_fixture_fails_closed(self):
        raw = self.synthetic_raw()
        raw = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            with self.assertRaisesRegex(ValueError, "duplicate player/fixture"):
                historical_backtest.season_frame("2025-26", "synthetic")

    def test_out_of_range_gameweek_fails_closed(self):
        raw = self.synthetic_raw()
        raw.loc[0, "round"] = 39
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            with self.assertRaisesRegex(ValueError, "outside 1..38"):
                historical_backtest.season_frame("2025-26", "synthetic")

    def test_duplicate_fixture_lookup_fails_closed(self):
        raw = self.synthetic_raw().rename(columns={"fixture": "fixture_code"}).drop(columns=["round"])
        fixture_table = pd.DataFrame({"code": ["A-1-1", "A-1-1"], "event": [1, 1]})
        with patch.object(historical_backtest, "fetch_csv", side_effect=[raw, fixture_table]):
            with self.assertRaisesRegex(ValueError, "duplicate fixture identifiers"):
                historical_backtest.season_frame("2025-26", "synthetic")

    def test_unassigned_postponed_fixture_fails_closed(self):
        raw = self.synthetic_raw().rename(columns={"fixture": "fixture_code"}).drop(columns=["round"])
        fixture_table = pd.DataFrame({"code": ["A-1-1"], "event": [None]})
        with patch.object(historical_backtest, "fetch_csv", side_effect=[raw, fixture_table]):
            with self.assertRaisesRegex(ValueError, "unassigned/postponed"):
                historical_backtest.season_frame("2025-26", "synthetic")

    def test_missing_gw_without_verified_lookup_fails_closed(self):
        raw = self.synthetic_raw().drop(columns=["round"])
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            with self.assertRaisesRegex(ValueError, "no fixture lookup"):
                historical_backtest.season_frame("2024-25", "synthetic")

    def test_missing_kickoff_timestamp_fails_closed(self):
        raw = self.synthetic_raw().drop(columns=["kickoff_time"])
        with patch.object(historical_backtest, "fetch_csv", return_value=raw):
            with self.assertRaisesRegex(ValueError, "kickoff timestamps"):
                historical_backtest.season_frame("2024-25", "synthetic")

    def test_incomplete_run_preserves_existing_model_and_reports(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            out = Path(temp)
            model_path = out / "trained_fpl_model.json"
            summary_path = out / "historical_backtest_summary.csv"
            model_path.write_text("known-good-model", encoding="utf-8")
            summary_path.write_text("known-good-summary", encoding="utf-8")
            with patch("sys.argv", ["historical_backtest.py", "--out-dir", temp, "--seasons", "2023-24", "2024-25"]), \
                 patch.object(historical_backtest, "fetch_csv", side_effect=RuntimeError("network down")):
                self.assertEqual(historical_backtest.main(), 1)
            self.assertEqual(model_path.read_text(encoding="utf-8"), "known-good-model")
            self.assertEqual(summary_path.read_text(encoding="utf-8"), "known-good-summary")
            manifest = pd.read_json(out / "historical_backtest_failure.json", typ="series")
            self.assertEqual(manifest["status"], "incomplete")
            self.assertEqual(manifest["skipped_seasons"], ["2023-24", "2024-25"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
