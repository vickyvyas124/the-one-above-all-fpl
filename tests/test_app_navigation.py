import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import requests
import streamlit as st
from fpl_lineup import score_starting_xi, select_model_lineup

try:
    import streamlit as st
    from streamlit.testing.v1 import AppTest
except ModuleNotFoundError as exc:
    if exc.name != "streamlit":
        raise
    AppTest = None
    st = None


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
            error = requests.HTTPError(f"mock HTTP {self.status_code}")
            error.response = self
            raise error

    def json(self):
        if self.payload is None:
            raise ValueError("mock response has no JSON payload")
        return self.payload


def mocked_fpl_get(url, **_kwargs):
    if url.endswith("/trained_fpl_model.json"):
        return MockResponse(status_code=404)
    if url.endswith("/forecast_log.csv"):
        return MockResponse(
            status_code=200,
            text="season,gameweek,player_id,model_version,projection_model\n2026-27,2,1,legacy-unknown,4.2\n",
        )
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


def mocked_fpl_get_invalid_picks(url, **kwargs):
    response = mocked_fpl_get(url, **kwargs)
    if "/entry/2311669/event/" in url and url.endswith("/picks/"):
        picks = response.payload["picks"]
        picks[-1]["position"] = picks[-2]["position"]
        return MockResponse({"picks": picks})
    return response


def mocked_fpl_get_predeadline_picks(url, **kwargs):
    if "/entry/2311669/event/2/picks/" in url:
        return MockResponse(status_code=404)
    return mocked_fpl_get(url, **kwargs)


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
    def test_app_lineup_imports_are_available_from_the_checked_out_module(self):
        self.assertTrue(callable(score_starting_xi))
        self.assertTrue(callable(select_model_lineup))
        app_source = (Path(__file__).parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn("from fpl_lineup import score_starting_xi, select_model_lineup", app_source)

    def test_dashboard_tabs_recommendation_and_existing_controls_render(self):
        app_path = Path(__file__).parents[1] / "app.py"
        with patch("requests.get", side_effect=mocked_fpl_get) as network:
            app = AppTest.from_file(str(app_path), default_timeout=60).run()
            initial_network_call_count = network.call_count

            self.assertFalse(app.exception, [item.message for item in app.exception])
            self.assertEqual(
                [tab.label for tab in app.tabs],
                ["Decide", "Squad", "Fixtures", "Model"],
            )
            model_tab = app.tabs[3]
            xi_table = next(
                item for item in model_tab.dataframe
                if {"Position", "Player", "Captain", "Vice"}.issubset(item.value.columns)
            )
            bench_table = next(
                item for item in model_tab.dataframe
                if {"Order", "Player", "Position"}.issubset(item.value.columns)
            )
            xi_names = xi_table.value["Player"].tolist()
            bench_names = bench_table.value["Player"].tolist()
            self.assertEqual(len(xi_names), 11)
            self.assertEqual(len(bench_names), 4)
            self.assertEqual(len(set(xi_names + bench_names)), 15)
            self.assertCountEqual(xi_names + bench_names, [row[2] for row in SQUAD])
            self.assertEqual(xi_table.value["Captain"].eq("C").sum(), 1)
            self.assertEqual(xi_table.value["Vice"].eq("VC").sum(), 1)
            self.assertIn("Kostoulas", xi_names + bench_names)
            _captain_xi_row = xi_table.value.loc[xi_table.value["Captain"].eq("C")].iloc[0]
            _xi_base_total = float(xi_table.value["GW2 pts"].sum())
            _xi_with_captain = next(
                item for item in model_tab.metric
                if item.label == "XI + captain projected points"
            )
            self.assertAlmostEqual(float(_xi_with_captain.value), _xi_base_total + float(_captain_xi_row["GW2 pts"]))
            _captain_table = next(
                item for item in app.tabs[1].dataframe
                if {"Projected GW points", "Captain 2× estimate"}.issubset(item.value.columns)
            )
            _captain_candidate = _captain_table.value.loc[
                _captain_table.value["Player"].eq(_captain_xi_row["Player"])
            ].iloc[0]
            self.assertAlmostEqual(
                float(_captain_candidate["Captain 2× estimate"]),
                2 * float(_captain_candidate["Projected GW points"]),
            )
            _formation_metric = next(item for item in model_tab.metric if item.label == "Model-best formation")
            _defenders, _midfielders, _forwards = map(int, _formation_metric.value.split("-"))
            self.assertTrue(3 <= _defenders <= 5 and 2 <= _midfielders <= 5 and 1 <= _forwards <= 3)
            _forecast_metric = next(item for item in model_tab.metric if item.label == "Forecast record")
            self.assertEqual(_forecast_metric.value, "Not saved")
            fixture_text = "\n".join(str(item.value) for item in app.tabs[2].markdown)
            self.assertIn("one-Gameweek estimate", fixture_text)
            for _, bench_row in bench_table.value.iterrows():
                self.assertIn(f"{bench_row['Player']} {float(bench_row['GW2 pts']):.1f}", fixture_text)
            squad_markup = "\n".join(str(item.value) for item in app.tabs[1].markdown)
            _captain_name = xi_table.value.loc[xi_table.value["Captain"].eq("C"), "Player"].iloc[0]
            _vice_name = xi_table.value.loc[xi_table.value["Vice"].eq("VC"), "Player"].iloc[0]
            self.assertIn(f"{_captain_name} · C", squad_markup)
            self.assertIn(f"{_vice_name} · VC", squad_markup)
            status_strip = next(
                element.value
                for element in app.markdown
                if "class='status-strip'" in str(element.value)
            )
            self.assertIn("<strong>Free transfers</strong> 1 · history-derived", status_strip)

            decide_tab = app.tabs[0]
            verdict_markdown = next(
                element
                for element in decide_tab.markdown
                if "verdict-card" in str(element.value)
            )
            self.assertIn("HOLD FOR NOW", verdict_markdown.value)
            self.assertIn("Provisional", verdict_markdown.value)
            decision_summary = next(
                item for item in decide_tab.markdown
                if "Captain and vice-captain" in str(item.value)
            )
            self.assertIn(f"C: {_captain_name} · VC: {_vice_name}", decision_summary.value)
            default_table = next(
                element
                for element in decide_tab.dataframe
                if list(element.value.columns) == ["Sell", "Buy", "Gain", "Needed", "Reliability"]
            )
            child_positions = {
                id(element): position
                for position, element in decide_tab.children.items()
            }
            self.assertLess(
                child_positions[id(verdict_markdown)],
                child_positions[id(default_table)],
                "The verdict must render before the default transfer table.",
            )
            _detail_select = next(
                item for item in decide_tab.selectbox
                if item.label == "Transfer to inspect"
            )
            self.assertTrue(_detail_select.options)
            _budget_table = next(
                item for item in decide_tab.dataframe
                if {"Buy price (£m)", "Budget ceiling (£m)"}.issubset(item.value.columns)
            )
            self.assertTrue(
                (_budget_table.value["Buy price (£m)"] <= _budget_table.value["Budget ceiling (£m)"]).all(),
                "Every automatic transfer candidate must fit within official sale price plus bank.",
            )
            _comparison = next(
                item for item in decide_tab.dataframe
                if {"Hold current player", "Transfer candidate", "Cumulative net"}.issubset(item.value.columns)
            )
            self.assertEqual(len(_comparison.value), 5)
            self.assertTrue(any(item.label == "Squad is correct" for item in app.checkbox))
            self.assertTrue(any(item.label == "Team news checked" for item in app.checkbox))

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
            self.assertEqual(network.call_count, initial_network_call_count, "UI control reruns must not add network requests.")
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

    def test_invalid_public_roster_fails_closed_without_sample_squad(self):
        st.cache_data.clear()
        app_path = Path(__file__).parents[1] / "app.py"
        with patch("requests.get", side_effect=mocked_fpl_get_invalid_picks):
            app = AppTest.from_file(str(app_path), default_timeout=60).run()
        self.assertFalse(app.exception, [item.message for item in app.exception])
        status_strip = next(item.value for item in app.markdown if "class='status-strip'" in str(item.value))
        self.assertIn("unavailable · check public picks", status_strip)
        verdict = next(item.value for item in app.tabs[0].markdown if "verdict-card" in str(item.value))
        self.assertIn("VERIFY DATA", verdict)
        self.assertIn("No example or saved squad is substituted", verdict)
        self.assertFalse(any(
            {"Sell", "Buy", "Gain", "Needed", "Reliability"}.issubset(item.value.columns)
            for item in app.tabs[0].dataframe
        ))

    def test_predeadline_404_reconstructs_prior_squad_without_routine_confirmation(self):
        st.cache_data.clear()
        app_path = Path(__file__).parents[1] / "app.py"
        with patch("requests.get", side_effect=mocked_fpl_get_predeadline_picks):
            app = AppTest.from_file(str(app_path), default_timeout=60).run()
            self.assertFalse(app.exception, [item.message for item in app.exception])
            _status_strip = next(item.value for item in app.markdown if "class='status-strip'" in str(item.value))
            self.assertIn("auto-reconstructed from GW1 · provisional until GW2 picks publish", _status_strip)
            self.assertTrue(any(
                {"Player", "Club"}.issubset(item.value.columns)
                for item in app.tabs[2].dataframe
            ), "The verified prior roster should restore the fixture matrix.")
            self.assertTrue(any("Model-selected starting XI" in str(item.value)
                                for item in app.tabs[1].markdown))
            _reference_markup = "\n".join(str(item.value) for item in app.tabs[1].markdown)
            # The pitch renders the model-selected XI and the bench as player tiles;
            # only the XI should be marked as starters.
            self.assertEqual(_reference_markup.count("class='player-tile'"), 15)
            self.assertEqual(_reference_markup.count("Starting XI ·"), 11)
            self.assertEqual(_reference_markup.count("<span class='tile-status'>Bench</span>"), 4)
            self.assertTrue(any(
                {"Position", "Player", "Captain", "Vice"}.issubset(item.value.columns)
                for item in app.tabs[3].dataframe
            ), "A complete prior squad plus complete empty public history should generate a provisional model XI automatically.")
            self.assertFalse(any(
                item.label == "Confirm or correct current 15-player roster"
                for item in app.sidebar.multiselect
            ))
            self.assertFalse(any(
                item.label == "I confirm this roster includes all transfers and chip changes"
                for item in app.sidebar.checkbox
            ))
            verdict = next(item.value for item in app.tabs[0].markdown if "verdict-card" in str(item.value))
            self.assertIn("Provisional", verdict)
            self.assertIn("HOLD FOR NOW", verdict)
            plan_table = next(
                item for item in app.tabs[0].dataframe
                if {"Transfers", "Free used", "Paid", "Hit", "Affordability", "By Gameweek"}.issubset(item.value.columns)
            )
            self.assertTrue(any("Hold / roll" in value for value in plan_table.value["Transfers"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
