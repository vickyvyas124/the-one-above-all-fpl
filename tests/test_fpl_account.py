import unittest

from fpl_account import build_account_snapshot, parse_selling_price_import


def legal_players():
    roles = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3
    return {index + 1: {"element_type": role, "team": index + 1}
            for index, role in enumerate(roles)}


class AccountSnapshotTests(unittest.TestCase):
    def test_verified_snapshot_is_shared_and_actionable(self):
        players = legal_players()
        ids = list(players)
        snapshot = build_account_snapshot(
            manager_id=2311669, gameweek=9, squad_ids=ids, players_by_id=players,
            squad_verified=True, bank_value=20, bank_source="official",
            free_transfers=2, free_transfers_verified=True,
            prices_by_id={pid: 100 for pid in ids},
        )
        self.assertTrue(snapshot.recommendations_ready)
        self.assertTrue(snapshot.actionable_transfers_ready)
        self.assertEqual(snapshot.bank_tenths, 20)

    def test_unknown_purchase_price_keeps_budget_unverified(self):
        players = legal_players()
        ids = list(players)
        prices = {pid: 100 for pid in ids}
        prices.pop(ids[0])
        snapshot = build_account_snapshot(
            manager_id=1, gameweek=1, squad_ids=ids, players_by_id=players,
            squad_verified=True, bank_value=20, bank_source="official",
            free_transfers=1, free_transfers_verified=True, prices_by_id=prices,
        )
        self.assertEqual(snapshot.budget_status, "UNVERIFIED")
        self.assertFalse(snapshot.actionable_transfers_ready)

    def test_missing_bank_or_unconfirmed_free_transfers_is_not_actionable(self):
        players = legal_players()
        ids = list(players)
        snapshot = build_account_snapshot(
            manager_id=1, gameweek=1, squad_ids=ids, players_by_id=players,
            squad_verified=True, bank_value=None, bank_source="official unavailable",
            free_transfers=1, free_transfers_verified=False,
            prices_by_id={pid: 100 for pid in ids},
        )
        self.assertIsNone(snapshot.bank_tenths)
        self.assertFalse(snapshot.actionable_transfers_ready)

    def test_price_import_requires_all_ids_exact_prices_and_value_reconciliation(self):
        players = legal_players()
        for player in players.values():
            player["now_cost"] = 100
        ids = list(players)
        text = "player_id,sell_tenths\n" + "\n".join(f"{pid},100" for pid in ids)
        prices, reason = parse_selling_price_import(text, ids, players, 20, 1520)
        self.assertEqual(set(prices), set(ids))
        self.assertIn("reconciled", reason)
        self.assertIsNone(parse_selling_price_import(text.replace(",100", ",100.5", 1), ids, players, 20, 1520)[0])
        self.assertIsNone(parse_selling_price_import(text, ids, players, 20, 1518)[0])

    def test_price_import_rejects_duplicate_or_out_of_squad_player(self):
        players = legal_players()
        for player in players.values():
            player["now_cost"] = 100
        ids = list(players)
        text = "player_id,sell_tenths\n" + "\n".join(f"{pid},100" for pid in ids)
        duplicate = text + f"\n{ids[0]},100"
        self.assertIsNone(parse_selling_price_import(duplicate, ids, players, 20, 1520)[0])


if __name__ == "__main__":
    unittest.main()
