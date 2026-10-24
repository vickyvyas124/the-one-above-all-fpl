import unittest
import json
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

    def test_invalid_model_falls_back_cleanly(self):
        self.assertIsNone(fpl_learning.predict_features({"model_version": "2.9"}, {}))

    def test_schedule_features_are_neutralized_in_training_and_inference(self):
        frame = self.training_frame()
        model = fpl_learning.fit_ridge_model(frame)
        base = fpl_learning.make_feature_row(
            position="MID", history_signal={"season_points_per_gw": 4, "recent_points_per_gw": 4,
            "recent_minutes_per_fixture": 75, "recent_appearance_rate": .9,
            "recent_start_rate": .8, "season_xgi_per90": .5, "recent_xgi_per90": .6},
            fixture_count=1, home_share=0.5)
        double = dict(base, fixture_count=2.0, home_share=1.0)
        self.assertEqual(fpl_learning.predict_features(model, double), fpl_learning.predict_features(model, base))
        self.assertEqual(model["feature_policy"], fpl_learning.FEATURE_POLICY)

    def test_old_artifact_without_feature_policy_is_rejected(self):
        model = fpl_learning.fit_ridge_model(self.training_frame())
        model.pop("feature_policy")
        self.assertIsNone(fpl_learning.predict_features(model, {}))
        with TemporaryDirectory(dir=Path.cwd()) as temp:
            path = Path(temp) / "model.json"
            path.write_text(json.dumps(model), encoding="utf-8")
            self.assertIsNone(fpl_learning.load_model(path))

    def test_engine_uses_model_for_future_gameweeks_and_returns_fixture_total(self):
        model = fpl_learning.fit_ridge_model(self.training_frame())
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
