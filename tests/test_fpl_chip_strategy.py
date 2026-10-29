import unittest

from fpl_chip_strategy import chip_half, chip_history_for_half, chip_planning_gameweeks


class ChipStrategyTests(unittest.TestCase):
    def test_chip_sets_reset_at_gameweek_20(self):
        self.assertEqual(chip_half(19), "first_half")
        self.assertEqual(chip_half(20), "second_half")

    def test_history_tracks_each_chip_in_the_current_half_only(self):
        rows = [
            {"name": "wildcard", "event": 6},
            {"name": "wildcard", "event": 22},
            {"name": "bboost", "event": 18},
            {"name": "freehit", "event": 21},
            {"name": "3xc", "event": 11},
        ]
        first_half = chip_history_for_half(rows, 19)
        second_half = chip_history_for_half(rows, 20)
        self.assertEqual(first_half["wildcard"], [6])
        self.assertEqual(first_half["bench_boost"], [18])
        self.assertEqual(first_half["triple_captain"], [11])
        self.assertEqual(second_half["wildcard"], [22])
        self.assertEqual(second_half["free_hit"], [21])
        self.assertEqual(second_half["bench_boost"], [])

    def test_chip_planning_horizon_stops_at_half_boundary(self):
        self.assertEqual(chip_planning_gameweeks(17, 5), [17, 18, 19])
        self.assertEqual(chip_planning_gameweeks(19, 5), [19])
        self.assertEqual(chip_planning_gameweeks(20, 5), [20, 21, 22, 23, 24])
        self.assertEqual(chip_planning_gameweeks(37, 5), [37, 38])


if __name__ == "__main__":
    unittest.main()
