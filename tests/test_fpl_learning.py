import unittest
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd

import fpl_learning
import fpl_engine


class FutureModelTests(unittest.TestCase):
    def training_frame(self, n=1500):
        rng = np.random.default_rng(77)
        positions = rng.integers(0, 4, size=n)
        fixture_count = rng.integers(0, 3, size=n)
        recent = rng.uniform(0, 8, size=n)
        season = rng.uniform(0, 6, size=n)
        minutes = rng.uniform(0, 90, size=n)
        appearance = rng.uniform(0, 1, size=n)
        start_rate = rng.uniform(0, 1, size=n)
        season_xgi = rng.uniform(0, 1.5, size=n)
        recent_xgi = rng.uniform(0, 1.5, size=n)
        home = rng.uniform(0, 1, size=n)
        points = (0.10 * season + 0.18 * recent + 0.015 * minutes + 0.5 * appearance
                  + 0.3 * start_rate + 0.5 * season_xgi + 0.4 * recent_xgi
                  + 1.3 * fixture_count + 0.35 * home + positions * 0.1 + rng.normal(0, 0.4, size=n))
        frame = pd.DataFrame({
            "season_mean_prev": season,
            "recent_mean_prev": recent,
            "recent_minutes_per_fixture_prev": minutes,
            "recent_appearance_rate_prev": appearance,
            "recent_start_rate_prev": start_rate,
            "season_xgi_per90_prev": season_xgi,
            "recent_xgi_per90_prev": recent_xgi,
            "fixture_count": fixture_count,
            "home_share": home,
            "actual_points": points,
        })
        for idx, pos in enumerate(["GKP", "DEF", "MID", "FWD"]):
            frame[f"position_{pos}"] = (positions == idx).astype(float)
        return frame

    def test_model_trains_and_produces_versioned_artifact(self):
        model = fpl_learning.fit_ridge_model(self.training_frame(), source_seasons=["2023-24", "2024-25"])
        self.assertEqual(model["model_version"], "3.0")
        self.assertGreater(model["training_rows"], 1000)
        row = {key: float(self.training_frame(1).iloc[0].get(key, 0)) for key in fpl_learning.FEATURES}
        prediction = fpl_learning.predict_features(model, row)
        self.assertIsNotNone(prediction)
        self.assertGreaterEqual(prediction, 0.0)

    def test_live_features_match_trained_schema(self):
        row = fpl_learning.make_feature_row(
            position=4,
            history_signal={
                "season_points_per_gw": 4.1, "recent_points_per_gw": 5.0,
                "recent_minutes_per_fixture": 78, "recent_appearance_rate": 0.8,
                "recent_start_rate": 0.6, "season_xgi_per90": 0.7, "recent_xgi_per90": 0.9,
            },
            fixture_count=2, home_share=0.5,
        )
        self.assertEqual(set(row), set(fpl_learning.FEATURES))
        self.assertEqual(row["fixture_count"], 1.0)
        self.assertEqual(row["home_share"], 0.5)
        self.assertEqual(row["position_FWD"], 1.0)

    def test_new_player_missing_history_uses_training_missing_value_policy(self):
        row = fpl_learning.make_feature_row(
            position="MID", history_signal={}, fixture_count=1, home_share=.5,
            fallback_season_mean=8.0, fallback_recent_mean=9.0, fallback_xgi_per90=2.0,
        )
        for name in ("season_mean_prev", "recent_mean_prev", "season_xgi_per90_prev", "recent_xgi_per90_prev"):
            self.assertEqual(row[name], 0.0)

    def test_reported_production_fallback_matches_live_single_fixture_output(self):
        original = fpl_engine.TRAINED_MODEL
        try:
            fpl_engine.TRAINED_MODEL = None
            player = {"id": 8, "team": 1, "element_type": 3, "points_per_game": 4.2,
                      "form": 5.0, "minutes": 900, "starts": 10, "appearances": 10,
                      "substitute_appearances": 0, "expected_goals": 4.0, "expected_assists": 2.0,
                      "status": "a", "chance_of_playing_next_round": 100}
            history = {8: {"recent_points_per_appearance": 5.5, "played_matches": 4,
                           "recent_minutes_per_fixture": 80, "sample_matches": 4}}
            fixtures = [{"event": 9, "team_h": 1, "team_a": 2, "team_h_difficulty": 2,
                         "team_a_difficulty": 4, "finished": False}]
            live, _, count = fpl_engine.estimate_points(
                player, "TST", fixtures, 1, start_gw=9, history_signals=history
            )
            expected = round(fpl_engine.production_fallback_per_fixture(player, history[8]) * 1.06, 1)
            self.assertEqual(count, 1)
            self.assertEqual(live, expected)
        finally:
            fpl_engine.TRAINED_MODEL = original

    def test_invalid_model_falls_back_cleanly(self):
        self.assertIsNone(fpl_learning.predict_features({"model_version": "2.9"}, {}))

    def test_schedule_features_are_neutralized_in_training_and_inference(self):
        frame = self.training_frame()
        model = fpl_learning.fit_ridge_model(frame, source_seasons=["2023-24"])
        base = fpl_learning.make_feature_row(
            position="MID", history_signal={"season_points_per_gw": 4, "recent_points_per_gw": 4,
            "recent_minutes_per_fixture": 75, "recent_appearance_rate": .9,
            "recent_start_rate": .8, "season_xgi_per90": .5, "recent_xgi_per90": .6},
            fixture_count=1, home_share=0.5)
        double = dict(base, fixture_count=2.0, home_share=1.0)
        self.assertEqual(fpl_learning.predict_features(model, double), fpl_learning.predict_features(model, base))
        self.assertEqual(model["feature_policy"], fpl_learning.FEATURE_POLICY)

    def test_old_artifact_without_feature_policy_is_rejected(self):
        model = fpl_learning.fit_ridge_model(self.training_frame(), source_seasons=["2023-24"])
        model.pop("feature_policy")
        self.assertIsNone(fpl_learning.predict_features(model, {}))
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / "model.json"
            path.write_text(json.dumps(model), encoding="utf-8")
            self.assertIsNone(fpl_learning.load_model(path))

    def test_model_artifact_missing_invalid_incompatible_and_valid(self):
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / "trained_fpl_model.json"
            self.assertIsNone(fpl_learning.load_model(path))
            path.write_text("{broken", encoding="utf-8")
            self.assertIsNone(fpl_learning.load_model(path))
            model = fpl_learning.fit_ridge_model(self.training_frame(), source_seasons=["2023-24"])
            incompatible = dict(model, model_version="2.9")
            path.write_text(json.dumps(incompatible), encoding="utf-8")
            self.assertIsNone(fpl_learning.load_model(path))
            incompatible = dict(model)
            incompatible["coefficients"] = {"wrong": 1.0}
            path.write_text(json.dumps(incompatible), encoding="utf-8")
            self.assertIsNone(fpl_learning.load_model(path))
            incompatible = dict(model, model_type="unknown")
            path.write_text(json.dumps(incompatible), encoding="utf-8")
            self.assertIsNone(fpl_learning.load_model(path))
            path.write_text(json.dumps(model), encoding="utf-8")
            self.assertTrue(fpl_learning.validate_model(fpl_learning.load_model(path)))

    def test_history_aggregates_fixture_records_into_gameweeks(self):
        base = datetime(2025, 8, 1, tzinfo=timezone.utc)
        rows = [
            {"round": 1, "fixture": 101, "kickoff_time": base.isoformat(), "total_points": 4, "minutes": 90,
             "starts": 1, "expected_goals": .2, "expected_assists": .1},
            {"round": 2, "fixture": 201, "kickoff_time": (base + timedelta(days=7)).isoformat(), "total_points": 2,
             "minutes": 90, "starts": 1, "expected_goals": .1, "expected_assists": 0},
            {"round": 2, "fixture": 202, "kickoff_time": (base + timedelta(days=8)).isoformat(), "total_points": 3,
             "minutes": 30, "starts": 0, "expected_goals": .2, "expected_assists": .1},
        ]
        signal = fpl_engine.make_history_signal({"history": rows}, cutoff_utc=base + timedelta(days=20))
        self.assertEqual(signal["history_total_gws"], 2)
        self.assertEqual(signal["sample_matches"], 3)
        self.assertEqual(signal["recent_points_per_gw"], 4.5)
        self.assertEqual(signal["season_points_per_gw"], 4.5)
        self.assertAlmostEqual(signal["recent_minutes_per_fixture"], 75.0)
        self.assertEqual(signal["recent_appearance_rate"], 1.0)
        self.assertEqual(signal["recent_start_rate"], 1.0)

    def test_history_cutoff_excludes_recent_or_unverifiable_gameweeks(self):
        cutoff = datetime(2025, 9, 10, tzinfo=timezone.utc)
        rows = [
            {"round": 1, "fixture": 1, "kickoff_time": "2025-08-01T12:00:00Z", "total_points": 5, "minutes": 90},
            {"round": 2, "fixture": 2, "kickoff_time": "2025-09-08T12:00:00Z", "total_points": 99, "minutes": 90},
            {"round": 3, "fixture": 3, "total_points": 99, "minutes": 90},
        ]
        signal = fpl_engine.make_history_signal({"history": rows}, cutoff_utc=cutoff)
        self.assertEqual(signal["history_total_gws"], 1)
        self.assertEqual(signal["season_points_per_gw"], 5.0)

    def test_engine_uses_model_for_future_gameweeks_and_returns_fixture_total(self):
        model = fpl_learning.fit_ridge_model(self.training_frame(), source_seasons=["2023-24"])
        original = fpl_engine.TRAINED_MODEL
        try:
            fpl_engine.TRAINED_MODEL = model
            player = {"id": 50, "team": 1, "element_type": 4, "status": "a",
                      "chance_of_playing_next_round": 100, "minutes": 900,
                      "expected_goals": 4.0, "expected_assists": 2.0,
                      "points_per_game": 5.0, "form": 4.0}
            history = {50: {"season_points_per_gw": 4.0, "recent_points_per_gw": 4.5,
                            "recent_minutes_per_fixture": 80.0, "recent_appearance_rate": 0.9,
                            "recent_start_rate": 0.8, "season_xgi_per90": 0.6,
                            "recent_xgi_per90": 0.8}}
            fixtures = [
                {"event": 9, "team_h": 1, "team_a": 2, "team_h_difficulty": 2,
                 "team_a_difficulty": 4, "finished": False},
                {"event": 10, "team_h": 3, "team_a": 1, "team_h_difficulty": 3,
                 "team_a_difficulty": 3, "finished": False},
            ]
            one, _, one_count = fpl_engine.estimate_points(player, "A", fixtures, 1, start_gw=9,
                history_signals=history, use_history=True)
            two, _, two_count = fpl_engine.estimate_points(player, "A", fixtures, 2, start_gw=9,
                history_signals=history, use_history=True)
            self.assertEqual(one_count, 1)
            self.assertEqual(two_count, 2)
            self.assertGreaterEqual(one, 0)
            self.assertGreaterEqual(two, one)
        finally:
            fpl_engine.TRAINED_MODEL = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
