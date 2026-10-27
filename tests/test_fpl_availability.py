import unittest

import fpl_engine
from fpl_availability import availability_adjustment


class AvailabilityAdjustmentTests(unittest.TestCase):
    def test_explicitly_available_player_without_chance_is_unadjusted(self):
        self.assertEqual(availability_adjustment({"status": "a"}), 1.0)

    def test_doubtful_player_uses_official_chance_field(self):
        player = {"status": "d", "chance_of_playing_next_round": 80}
        self.assertEqual(availability_adjustment(player), 0.8)
        self.assertEqual(fpl_engine._availability_override(player), 0.8)
        self.assertEqual(fpl_engine._estimate_minutes_parts(player)[1], 0.8)

    def test_doubtful_player_without_chance_uses_conservative_fallback(self):
        self.assertEqual(availability_adjustment({"status": "d"}), 0.55)

    def test_unavailable_statuses_share_conservative_factor(self):
        for status in ("i", "s", "u"):
            with self.subTest(status=status):
                self.assertEqual(availability_adjustment({"status": status}), 0.2)

    def test_chance_field_is_clamped_and_invalid_values_use_unknown_factor(self):
        self.assertEqual(availability_adjustment({"status": "a", "chance_of_playing_next_round": 0}), 0.2)
        self.assertEqual(availability_adjustment({"status": "a", "chance_of_playing_next_round": 125}), 1.0)
        self.assertEqual(availability_adjustment({"status": "a", "chance_of_playing_next_round": "bad"}), 1.0)

    def test_missing_status_evidence_is_conservative(self):
        self.assertEqual(availability_adjustment({}), 0.9)

    def test_later_gameweeks_do_not_carry_current_status_factor_forward(self):
        player = {"status": "i", "chance_of_playing_next_round": 0}
        self.assertEqual(fpl_engine._availability_override(player, weeks_ahead=1), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
