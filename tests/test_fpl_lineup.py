import unittest

from fpl_lineup import (
    FORMATION_ORDER,
    is_legal_formation,
    published_lineup_from_picks,
    score_starting_xi,
    select_model_lineup,
    validate_public_picks,
)


def make_squad():
    roles = ["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    return [{"id": index, "Player": f"Player {index}", "Position": role,
             "Club": f"Club {(index - 1) // 3 + 1}"}
            for index, role in enumerate(roles, 1)]


class PublicPickValidationTests(unittest.TestCase):
    def setUp(self):
        self.picks = [{"element": index, "position": index} for index in range(1, 16)]
        self.known = set(range(1, 16))

    def test_accepts_complete_unique_ordered_roster(self):
        self.assertEqual(validate_public_picks(self.picks, self.known), (True, ""))

    def test_rejects_missing_duplicate_unknown_player_and_bad_positions(self):
        for changed in (
            self.picks[:-1],
            self.picks[:-1] + [{"element": 14, "position": 15}],
            self.picks[:-1] + [{"element": 99, "position": 15}],
            self.picks[:-1] + [{"element": 15}],
            self.picks[:-1] + [{"element": 15, "position": 14}],
        ):
            with self.subTest(picks=changed):
                self.assertFalse(validate_public_picks(changed, self.known)[0])

    def test_rejects_roster_with_wrong_position_counts(self):
        players = {index: {"element_type": role} for index, role in enumerate(
            [1, 1] + [2] * 5 + [3] * 4 + [4] * 4, 1
        )}
        self.assertFalse(validate_public_picks(self.picks, players)[0])

    def test_rejects_roster_over_three_players_from_a_club(self):
        roles = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3
        players = {index: {"element_type": role, "team": 1}
                   for index, role in enumerate(roles, 1)}
        valid, reason = validate_public_picks(self.picks, players)
        self.assertFalse(valid)
        self.assertIn("maximum of three", reason)


class ModelLineupTests(unittest.TestCase):
    def test_evaluates_all_eight_legal_formations_and_reports_the_runner_up(self):
        squad = make_squad()
        projections = {player["id"]: float(player["id"] % 7) for player in squad}
        selection = select_model_lineup(squad, projections)
        self.assertEqual(selection["formations_evaluated"], [f"{d}-{m}-{f}" for d, m, f in FORMATION_ORDER])
        self.assertEqual(len(selection["formation_scores"]), 8)
        self.assertEqual(selection["formation"], selection["formation_scores"][0]["formation"])
        self.assertEqual(selection["second_best_formation"], selection["formation_scores"][1]["formation"])
        self.assertAlmostEqual(
            selection["formation_gap_points"],
            selection["formation_scores"][0]["projected_points"] - selection["formation_scores"][1]["projected_points"],
        )
        self.assertEqual(selection["projected_points"], max(row["projected_points"] for row in selection["formation_scores"]))

    def test_ties_are_deterministic_and_selection_reasons_use_projection_evidence(self):
        squad = make_squad()
        projections = {player["id"]: 5.0 for player in squad}
        first = select_model_lineup(squad, projections)
        second = select_model_lineup(list(reversed(squad)), projections)
        self.assertEqual(first["formation"], "3-4-3")
        self.assertEqual([p["id"] for p in first["starters"]], [p["id"] for p in second["starters"]])
        self.assertIn("projected", first["captain"]["Selection reason"])
        self.assertEqual(first["formation_gap_points"], 0.0)

    def test_rejects_squad_over_three_from_one_club(self):
        squad = make_squad()
        for player in squad[:4]:
            player["Club"] = "Too many"
        self.assertIsNone(select_model_lineup(squad, {player["id"]: 5.0 for player in squad}))

    def test_fast_starting_xi_score_matches_full_lineup_with_captain_bonus(self):
        squad = make_squad()
        projections = {index: float(index) for index in range(1, 16)}
        selection = select_model_lineup(squad, projections)
        self.assertEqual(score_starting_xi(squad, projections), selection["projected_points"])

    def test_selects_eleven_and_four_unique_members_from_all_fifteen(self):
        squad = make_squad()
        projections = {index: float(index) for index in range(1, 16)}
        selection = select_model_lineup(squad, projections)
        self.assertIsNotNone(selection)
        starters, bench = selection["starters"], selection["bench"]
        all_ids = [player["id"] for player in starters + bench]
        self.assertEqual(len(starters), 11)
        self.assertEqual(len(bench), 4)
        self.assertEqual(len(set(all_ids)), 15)
        self.assertEqual(set(all_ids), {player["id"] for player in squad})
        self.assertTrue(is_legal_formation(starters))
        self.assertIn(selection["captain"]["id"], {p["id"] for p in starters})
        self.assertIn(selection["vice"]["id"], {p["id"] for p in starters})
        self.assertNotEqual(selection["captain"]["id"], selection["vice"]["id"])
        base_total = sum(player["GW projected pts"] for player in starters)
        self.assertEqual(selection["base_xi_projected_points"], base_total)
        self.assertEqual(selection["captain_extra_points"], selection["captain"]["GW projected pts"])
        self.assertEqual(selection["projected_points"], base_total + selection["captain"]["GW projected pts"])

    def test_rejects_duplicate_or_invalid_squad_shape(self):
        squad = make_squad()
        projections = {player["id"]: 1.0 for player in squad}
        squad[-1] = dict(squad[0])
        self.assertIsNone(select_model_lineup(squad, projections))

    def test_legal_formation_rejects_illegal_role_counts(self):
        self.assertFalse(is_legal_formation(make_squad()[:11]))
        legal = [{"Position": role} for role in ["GK", "DEF", "DEF", "DEF", "MID", "MID", "MID", "MID", "MID", "FWD", "FWD"]]
        self.assertTrue(is_legal_formation(legal))
        self.assertFalse(is_legal_formation(legal[:-1] + [{"Position": "GK"}]))

    def test_published_lineup_uses_only_verified_current_picks_and_checks_formation(self):
        squad = make_squad()
        position_order = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15, 2, 6, 7, 12]
        picks = [{"element": player_id, "position": position,
                  "is_captain": player_id == 15, "is_vice_captain": player_id == 11}
                 for position, player_id in enumerate(position_order, 1)]
        projections = {player["id"]: float(player["id"]) for player in squad}
        self.assertIsNone(published_lineup_from_picks(picks, squad, projections, verified=False))
        current = published_lineup_from_picks(picks, squad, projections, verified=True)
        self.assertIsNotNone(current)
        self.assertEqual(current["formation"], "3-4-3")
        self.assertEqual(len(current["starters"]), 11)
        self.assertEqual(len(current["bench"]), 4)
        self.assertEqual(current["captain"]["id"], 15)
        self.assertEqual(current["vice"]["id"], 11)
        self.assertIsNone(published_lineup_from_picks(picks[:-1], squad, projections, verified=True))


if __name__ == "__main__":
    unittest.main()
