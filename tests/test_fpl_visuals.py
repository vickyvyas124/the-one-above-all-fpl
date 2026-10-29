import unittest

from fpl_visuals import club_badge_url, player_headshot_url, player_initials


class SquadVisualAssetTests(unittest.TestCase):
    def test_headshot_uses_photo_identifier_not_fpl_element_id(self):
        player = {"id": 411, "code": 223094, "photo": "223094.jpg"}
        self.assertEqual(
            player_headshot_url(player),
            "https://resources.premierleague.com/premierleague/photos/players/110x140/p223094.png",
        )

    def test_headshot_missing_or_malformed_metadata_uses_fallback(self):
        self.assertIsNone(player_headshot_url({"id": 411}))
        self.assertIsNone(player_headshot_url({"photo": "../bad.jpg"}))
        self.assertIsNone(player_headshot_url(None))
        self.assertEqual(player_initials({"first_name": "Erling", "second_name": "Haaland"}), "EH")
        self.assertEqual(player_initials({"web_name": "Saka"}), "SA")
        self.assertEqual(player_initials({}), "?")

    def test_badge_is_resolved_from_team_id_then_official_team_code(self):
        teams = {1: {"id": 1, "name": "Arsenal", "code": 3}}
        self.assertEqual(
            club_badge_url(teams, 1),
            "https://resources.premierleague.com/premierleague/badges/50/t3.png",
        )
        self.assertIsNone(club_badge_url(teams, 99))
        self.assertIsNone(club_badge_url({1: {"name": "Arsenal"}}, 1))
        self.assertIsNone(club_badge_url(teams, "not-an-id"))


if __name__ == "__main__":
    unittest.main()
