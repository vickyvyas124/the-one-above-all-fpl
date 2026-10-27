import unittest

from fpl_transfer import (
    gameweek_projection_series,
    is_affordable_tenths,
    official_selling_price_tenths,
    transfer_budget_tenths,
    transfer_hit_cost,
    transfer_projection_comparison,
)


class TransferProjectionTests(unittest.TestCase):
    def test_official_selling_price_uses_half_profit_rounded_down(self):
        self.assertEqual(official_selling_price_tenths(100, 107), 103)
        self.assertEqual(official_selling_price_tenths(100, 101), 100)
        self.assertEqual(official_selling_price_tenths(107, 100), 100)

    def test_budget_uses_exact_fpl_tenths_and_includes_sale_plus_bank(self):
        budget = transfer_budget_tenths(103, 7)
        self.assertEqual(budget, 110)
        self.assertTrue(is_affordable_tenths(110, budget))
        self.assertFalse(is_affordable_tenths(111, budget))

    def test_hit_only_applies_when_enabled_and_no_free_transfer_exists(self):
        self.assertEqual(transfer_hit_cost(0, False), 0)
        self.assertEqual(transfer_hit_cost(0, True), 4)
        self.assertEqual(transfer_hit_cost(1, True), 0)

    def test_series_uses_one_gameweek_per_row_including_blanks(self):
        def estimate(_player, _team, _fixtures, *, horizon, start_gw):
            self.assertEqual(horizon, 1)
            return ({2: 2.5, 3: 0, 4: 4.0}[start_gw], 3, 0)

        rows = gameweek_projection_series({}, "T", [], horizon=3, start_gw=2, estimate_points=estimate)
        self.assertEqual(rows, [
            {"gameweek": 2, "points": 2.5},
            {"gameweek": 3, "points": 0.0},
            {"gameweek": 4, "points": 4.0},
        ])

    def test_comparison_matches_total_gain_and_charges_hit_once(self):
        outgoing = [{"gameweek": gw, "points": pts} for gw, pts in zip(range(2, 5), [3, 1, 4])]
        incoming = [{"gameweek": gw, "points": pts} for gw, pts in zip(range(2, 5), [4, 4, 5])]
        rows = transfer_projection_comparison(outgoing, incoming, transfer_hit=4)
        self.assertEqual([row["Cumulative net"] for row in rows], [-3, 0, 1])
        self.assertEqual(rows[-1]["Cumulative net"], sum(r["points"] for r in incoming) - sum(r["points"] for r in outgoing) - 4)

    def test_rejects_incomparable_horizons(self):
        with self.assertRaises(ValueError):
            transfer_projection_comparison([{"gameweek": 1, "points": 1}], [], 0)


if __name__ == "__main__":
    unittest.main()
