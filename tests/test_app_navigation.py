import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import requests

try:
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise
    AppTest = None


SQUAD = [
    ("David", "Raya", "Raya", 1),
    ("Panagiotis", "Tzolakis", "Tzolakis", 1),
    ("Egan", "Egan", "Egan", 2),
    ("Lewis", "Hall", "Hall", 2),
    ("Riccardo", "Calafiori", "Calafiori", 2),
    ("Maxim", "De Cuyper", "De Cuyper", 2),
    ("Leif", "Davis", "Davis", 2),
    ("Cole", "Palmer", "Palmer", 3),
    ("Marcus", "Tavernier", "Tavernier", 3),
    ("Pascal", "Groß", "Groß", 3),
    ("Morgan", "Rogers", "Rogers", 3),
    ("Kevin", "Schade", "Schade", 3),
    ("Charalampos", "Kostoulas", "Kostoulas", 4),
    ("João", "Pedro", "João Pedro", 4),
    ("Erling", "Haaland", "Haaland", 4),
]


class MockResponse:
    def __init__(self, payload=None, *, status_code=200, text=None):
        self.payload = payload
        self.status_code = status_code
        self.text = text if text is not None else (
            json.dumps(payload) if payload is not None else ""
        )

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"mock HTTP {self.status_code}")

    def json(self):
        if self.payload is None:
            raise ValueError("mock response has no JSON payload")
        return self.payload


def mocked_fpl_get(url, **_kwargs):
    if url.endswith("/trained_fpl_model.json"):
        return MockResponse(status_code=404)
    if url.endswith("/forecast_log.csv"):
        return MockResponse(status_code=200, text="season,gameweek,player_id\n")
    if url.endswith("/historical_backtest_summary.csv"):
        return MockResponse(status_code=404)
    if url.endswith("/bootstrap-static/"):
        players = []
        for player_id, (first, second, web_name, position) in enumerate(SQUAD, 1):
            players.append(_player(player_id, first, second, web_name, position, player_id))
        players.extend([
            _player(16, "Candidate", "Keeper", "Candidate GK", 1, 16, premium=True),
            _player(17, "Candidate", "Defender", "Candidate DEF", 2, 17, premium=True),
            _player(18, "Candidate", "Midfielder", "Candidate MID", 3, 18, premium=True),
            _player(19, "Candidate", "Forward", "Candidate FWD", 4, 19, premium=True),
        ])
        now = datetime.now(timezone.utc)
        events = [
            {"id": gameweek, "name": f"Gameweek {gameweek}",
             "finished": gameweek == 1, "is_current": gameweek == 1,
             "is_next": gameweek == 2,
             "deadline_time": (now + timedelta(days=2)).isoformat() if gameweek == 2 else None}
            for gameweek in range(1, 9)
        ]
        return MockResponse({
            "elements": players,
            "teams": [
                {"id": team_id, "name": f"Team {team_id}", "short_name": f"T{team_id:02d}"}
                for team_id in range(1, 21)
            ],
            "element_types": [
                {"id": 1, "singular_name_short": "GKP"},
                {"id": 2, "singular_name_short": "DEF"},
                {"id": 3, "singular_name_short": "MID"},
                {"id": 4, "singular_name_short": "FWD"},
            ],
            "events": events,
        })
    if url.endswith("/fixtures/"):
        fixtures = []
        for gameweek in range(2, 9):
            for team_id in range(1, 11):
                fixtures.append({
                    "id": gameweek * 100 + team_id,
                    "event": gameweek,
                    "team_h": team_id,
                    "team_a": team_id + 10,
                    "team_h_difficulty": 3,
                    "team_a_difficulty": 3,
                    "kickoff_time": f"2026-10-{gameweek + 1:02d}T15:00:00Z",
                    "finished": False,
                })
        return MockResponse(fixtures)
    if "/element-summary/" in url:
        return MockResponse({"history": [], "fixtures": []})
    if "/entry/2311669/event/" in url and url.endswith("/picks/"):
        picks = []
        for position, player_id in enumerate(range(1, 16), 1):
            pick = {
                "element": player_id,
                "position": position,
                "multiplier": 2 if player_id == 15 else (0 if position > 11 else 1),
                "is_captain": player_id == 15,
                "is_vice_captain": player_id == 8,
                "purchase_price": 60,
            }
            picks.append(pick)
        return MockResponse({"picks": picks})
    if url.endswith("/entry/2311669/history/"):
        return MockResponse({
            "current": [{
                "event": 1, "points": 54, "rank": 1000, "overall_rank": 5000,
                "bank": 100, "event_transfers": 0, "event_transfers_cost": 0,
            }],
            "past": [], "chips": [],
        })
    if url.endswith("/entry/2311669/transfers/"):
        return MockResponse([])
    if url.endswith("/entry/2311669/"):
        return MockResponse({
            "summary_overall_points": 54, "summary_overall_rank": 5000,
            "summary_event_points": 54, "summary_event_rank": 1000,
            "last_deadline_bank": 100, "last_deadline_value": 1000,
        })
    raise AssertionError(f"AppTest attempted an unmocked network request: {url}")


def _player(player_id, first, second, web_name, position, team_id, *, premium=False):
    return {
        "id": player_id,
        "first_name": first,
        "second_name": second,
        "web_name": web_name,
        "team": team_id,
        "element_type": position,
        "now_cost": 50 if premium else 60,
        "form": 12.0 if premium else 3.0,
        "points_per_game": 12.0 if premium else 3.0,
        "minutes": 3000 if premium else 1000,
        "starts": 34 if premium else 11,
        "appearances": 34 if premium else 12,
        "substitute_appearances": 0,
        "expected_goals": 15.0 if premium else 2.0,
        "expected_assists": 10.0 if premium else 2.0,
        "status": "a",
        "chance_of_playing_next_round": 100,
        "selected_by_percent": 3.0 if premium else 15.0,
        "ep_next": "5.0",
    }


def _values(elements):
    return [str(getattr(item, "value", "")) for item in elements]


@unittest.skipIf(AppTest is None, "Streamlit is not installed in this environment")
class DashboardNavigationAppTests(unittest.TestCase):
    def test_dashboard_tabs_recommendation_and_existing_controls_render(self):
        app_path = Path(__file__).parents[1] / "app.py"
        with patch("requests.get", side_effect=mocked_fpl_get):
            app = AppTest.from_file(str(app_path), default_timeout=60).run()

            self.assertFalse(app.exception, [item.message for item in app.exception])
            self.assertEqual(
                [tab.label for tab in app.tabs],
                ["Decide", "Squad", "Fixtures", "Model"],
            )

            rendered_text = "\n".join(
                value
                for collection in ("info", "success", "warning", "markdown", "caption", "subheader")
                for value in _values(getattr(app, collection))
            )
            self.assertTrue(
                any("Decision: " in value for value in (
                    *_values(app.info), *_values(app.success), *_values(app.warning)
                )),
                "The automatic transfer recommendation verdict was not rendered.",
            )
            self.assertIn("Automatic transfer recommendations", rendered_text)

            team_id = next(item for item in app.number_input if item.label == "FPL Team ID")
            self.assertEqual(team_id.value, 2311669)
            projection_horizon = next(item for item in app.slider if item.label == "Projection horizon (Gameweeks)")
            self.assertEqual(projection_horizon.value, 5)
            budget_source = next(item for item in app.selectbox if item.label == "Budget & transfer source")
            self.assertEqual(budget_source.value, "Auto-sync from FPL history")
            risk = next(item for item in app.select_slider if item.label == "Risk appetite")
            self.assertEqual(risk.value, "Moderate")

            budget_source.set_value("Manual override")
            app.run()
            self.assertFalse(app.exception, [item.message for item in app.exception])
            self.assertTrue(any(item.label == "Bank (£m)" for item in app.number_input))
            free_transfers = next(item for item in app.selectbox if item.label == "Free transfers available")
            self.assertIn(free_transfers.value, [0, 1, 2, 3, 4, 5])

            next(item for item in app.slider if item.label == "Projection horizon (Gameweeks)").set_value(3)
            next(item for item in app.checkbox if item.label == "Allow a -4 hit if model sees a gain").set_value(True)
            next(item for item in app.select_slider if item.label == "Risk appetite").set_value("Aggressive")
            app.run()
            self.assertFalse(app.exception, [item.message for item in app.exception])
            captions = "\n".join(_values(app.caption))
            self.assertIn("Risk mode: Aggressive", captions)
            self.assertIn("Horizon: 3 GWs", captions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
