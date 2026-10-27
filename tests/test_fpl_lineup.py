import unittest

from fpl_lineup import is_legal_formation, score_starting_xi, select_model_lineup, validate_public_picks


def make_squad():
    roles = ["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    return [{"id": index, "Player": f"Player {index}", "Position": role}
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


class ModelLineupTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
