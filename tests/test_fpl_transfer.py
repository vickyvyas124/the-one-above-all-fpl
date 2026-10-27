import unittest

from fpl_transfer import (
    gameweek_projection_series,
    is_affordable_tenths,
    official_selling_price_tenths,
    transfer_budget_tenths,
    transfer_hit_cost,
    transfer_projection_comparison,
    optimize_transfer_plans,
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

    def test_optimizer_includes_hold_one_two_and_three_with_one_hit(self):
        positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        squad = [{"id": i + 1, "position": pos, "club": f"C{i // 3}",
                  "sell_tenths": 50, "price_tenths": 50, "projections": [2, 2]}
                 for i, pos in enumerate(positions)]
        candidates = [{"id": 100 + i, "position": pos, "club": f"N{i}",
                       "price_tenths": 50, "projections": [8, 8]}
                      for i, pos in enumerate(positions)]
        plans = optimize_transfer_plans(squad, candidates, 2, 0, horizon=2,
                                        beam_width=200, candidate_limit=30)
        self.assertEqual(set(plans), {0, 1, 2, 3, 4})
        self.assertEqual(plans[3]["paid_transfers"], 1)
        self.assertEqual(plans[3]["hit_cost"], 4)
        self.assertEqual(plans[3]["gameweeks"][0]["hit_cost"], 4)
        self.assertEqual(plans[3]["gameweeks"][1]["hit_cost"], 0)
        self.assertEqual(plans[4]["paid_transfers"], 2)
        self.assertEqual(plans[4]["hit_cost"], 8)
        one_ft = optimize_transfer_plans(squad, candidates, 1, 0, horizon=2,
                                         beam_width=200, candidate_limit=30)
        self.assertEqual(one_ft[2]["paid_transfers"], 1)
        self.assertEqual(one_ft[2]["hit_cost"], 4)
        self.assertEqual(one_ft[3]["paid_transfers"], 2)
        self.assertEqual(one_ft[3]["hit_cost"], 8)

    def test_optimizer_respects_combined_budget_and_roster_club_limit(self):
        positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        squad = [{"id": i + 1, "position": pos, "club": f"C{i // 3}",
                  "sell_tenths": 100, "price_tenths": 100, "projections": [1]}
                 for i, pos in enumerate(positions)]
        candidates = [{"id": 100 + i, "position": pos, "club": "C0",
                       "price_tenths": 101, "projections": [10]}
                      for i, pos in enumerate(positions)]
        plans = optimize_transfer_plans(squad, candidates, 2, 0, horizon=1,
                                        beam_width=50, candidate_limit=20)
        self.assertNotIn(1, plans)  # C0 already has three players; buys there are illegal.
        affordable = optimize_transfer_plans(squad, [dict(candidates[2], club="NEW", price_tenths=201)],
                                             2, 0, horizon=1)
        self.assertNotIn(1, affordable)  # Exact combined sale budget is only 200 tenths.
        exact_pair = [dict(candidates[0], id=201, club="NEW1", price_tenths=100),
                      dict(candidates[2], id=202, club="NEW2", price_tenths=100)]
        exact_plans = optimize_transfer_plans(squad, exact_pair, 2, 0, horizon=1)
        self.assertIn(2, exact_plans)
        self.assertEqual(exact_plans[2]["budget_after_tenths"], 0)
        self.assertEqual(exact_plans[2]["budget_before_tenths"], 200)
        original_ids = {player["id"] for player in squad}
        self.assertTrue(all(move["out"]["id"] in original_ids
                            for move in exact_plans[2]["transfers"]))

    def test_acquired_player_cannot_become_outgoing_or_fund_a_later_move(self):
        positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        squad = [{"id": i + 1, "position": pos, "club": f"C{i // 3}",
                  "sell_tenths": 0, "price_tenths": 100, "projections": [1]}
                 for i, pos in enumerate(positions)]
        candidates = [
            {"id": 100, "position": "GK", "club": "NEW1", "price_tenths": 0,
             "sell_tenths": 1000, "projections": [10]},
            {"id": 101, "position": "GK", "club": "NEW2", "price_tenths": 100,
             "projections": [20]},
        ]
        plans = optimize_transfer_plans(squad, candidates, 2, 0, horizon=1)
        self.assertNotIn(2, plans)
        self.assertTrue(all(move["out"]["id"] <= 15
                            for plan in plans.values() for move in plan["transfers"]))

    def test_unverified_bank_or_sales_keep_projections_but_not_affordability_claim(self):
        positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        squad = [{"id": i + 1, "position": pos, "club": f"C{i // 3}",
                  "price_tenths": 100, "projections": [1]}
                 for i, pos in enumerate(positions)]
        candidates = [{"id": 100, "position": "GK", "club": "NEW",
                       "price_tenths": 100, "projections": [5]}]
        plans = optimize_transfer_plans(squad, candidates, 1, None, horizon=1,
                                        budget_verified=True)
        self.assertIn(1, plans)
        self.assertIsNone(plans[1]["affordable"])
        self.assertIsNone(plans[1]["budget_before_tenths"])
        self.assertIsNone(plans[1]["budget_after_tenths"])
        fractional_bank = optimize_transfer_plans(squad, candidates, 1, 0.5, horizon=1)
        self.assertIsNone(fractional_bank[1]["affordable"])
        missing_sales = [dict(player, sell_tenths=100) for player in squad]
        missing_sales[0].pop("sell_tenths")
        missing_price = optimize_transfer_plans(missing_sales, candidates, 1, 10, horizon=1)
        self.assertIsNone(missing_price[1]["affordable"])

    def test_position_and_club_constraints_hold_after_each_move(self):
        positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        squad = [{"id": i + 1, "position": pos, "club": "C0" if i < 3 else f"C{i}",
                  "sell_tenths": 100, "price_tenths": 100, "projections": [1]}
                 for i, pos in enumerate(positions)]
        candidates = [
            {"id": 100, "position": "GK", "club": "C0", "price_tenths": 100,
             "projections": [5]},
            {"id": 101, "position": "MID", "club": "C0", "price_tenths": 100,
             "projections": [5]},
        ]
        plans = optimize_transfer_plans(squad, candidates, 2, 0, horizon=1)
        self.assertIn(1, plans)
        self.assertNotIn(2, plans)  # The second incoming would exceed C0's three-player limit.
        wrong_position = [{"id": 102, "position": "WING", "club": "NEW",
                           "price_tenths": 100, "projections": [10]}]
        wrong_position_plans = optimize_transfer_plans(squad, wrong_position, 1, 0, horizon=1)
        self.assertEqual(set(wrong_position_plans), {0})

    def test_hold_remains_when_every_transfer_is_unaffordable_or_not_an_upgrade(self):
        positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        squad = [{"id": i + 1, "position": pos, "club": f"C{i // 3}",
                  "sell_tenths": 100, "price_tenths": 100, "projections": [5]}
                 for i, pos in enumerate(positions)]
        unaffordable = [{"id": 100, "position": "GK", "club": "NEW",
                         "price_tenths": 201, "projections": [20]}]
        plans = optimize_transfer_plans(squad, unaffordable, 1, 0, horizon=1)
        self.assertEqual(set(plans), {0})
        self.assertEqual(plans[0]["cumulative_net_gain"], 0)
        inferior = [{"id": 101, "position": "GK", "club": "NEW",
                     "price_tenths": 100, "projections": [1]}]
        plans = optimize_transfer_plans(squad, inferior, 1, 0, horizon=1)
        self.assertEqual(plans[0]["cumulative_net_gain"], 0)
        self.assertLessEqual(plans[1]["cumulative_net_gain"], plans[0]["cumulative_net_gain"])

    def test_zero_free_transfers_include_hold_and_multiple_hits(self):
        positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        squad = [{"id": i + 1, "position": pos, "club": f"C{i // 3}",
                  "sell_tenths": 100, "price_tenths": 100, "projections": [1]}
                 for i, pos in enumerate(positions)]
        candidates = [{"id": 100 + i, "position": pos, "club": f"N{i}",
                       "price_tenths": 100, "projections": [5]}
                      for i, pos in enumerate(positions)]
        plans = optimize_transfer_plans(squad, candidates, 0, 0, horizon=1,
                                        beam_width=50, candidate_limit=20)
        self.assertIn(0, plans)
        self.assertEqual(plans[0]["cumulative_net_gain"], 0)
        self.assertEqual(plans[1]["hit_cost"], 4)
        self.assertEqual(plans[2]["hit_cost"], 8)

    def test_starting_xi_score_can_reject_a_bench_only_upgrade(self):
        positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
        squad = [{"id": i + 1, "position": pos, "club": f"C{i // 3}",
                  "sell_tenths": 100, "price_tenths": 100, "projections": [1]}
                 for i, pos in enumerate(positions)]
        candidates = [{"id": 100 + i, "position": pos, "club": f"N{i}",
                       "price_tenths": 100, "projections": [10]}
                      for i, pos in enumerate(positions)]
        plans = optimize_transfer_plans(squad, candidates, 1, 0, horizon=1,
                                        beam_width=50, candidate_limit=20,
                                        score_roster=lambda _roster, _week: 60)
        self.assertIn(1, plans)
        self.assertEqual(plans[1]["cumulative_net_gain"], 0)


if __name__ == "__main__":
    unittest.main()
