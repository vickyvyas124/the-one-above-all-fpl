import unittest
from datetime import datetime, timezone
import json
from types import SimpleNamespace

from fpl_squad import apply_public_transfers, load_public_squad, validate_squad_ids


def player_map():
    roles = [1, 1] + [2] * 5 + [3] * 5 + [4] * 3
    roles.append(4)
    return {pid: {"id": pid, "element_type": role} for pid, role in enumerate(roles, 1)}


def picks():
    return {"picks": [
        {"element": pid, "position": pid, "purchase_price": 60,
         "is_captain": pid == 15, "is_vice_captain": pid == 8}
        for pid in range(1, 16)
    ]}


def http_error(status):
    return RuntimeErrorWithResponse(SimpleNamespace(status_code=status))


class RuntimeErrorWithResponse(Exception):
    def __init__(self, response):
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


class PreDeadlineSquadTests(unittest.TestCase):
    def setUp(self):
        self.players = player_map()
        self.events = [
            {"id": 8, "finished": True},
            {"id": 9, "is_next": True, "finished": False,
             "deadline_time": "2026-10-31T11:00:00Z"},
        ]
        self.now = datetime(2026, 10, 27, tzinfo=timezone.utc)

    def load(self, *, rows=None, transfer_available=True, prior=None, target_error=None):
        responses = {8: prior or picks()}

        def fetch(event):
            if event == 9:
                if target_error:
                    raise target_error
                return picks()
            return responses[event]

        return load_public_squad(
            9, self.events, fetch, rows, transfer_available, self.players, now=self.now,
        )

    def test_upcoming_404_uses_valid_prior_picks_and_applies_verified_transfer(self):
        self.players[16] = {"id": 16, "element_type": 4}
        result = self.load(
            rows=[{"event": 9, "time": "2026-10-29T10:00:00Z", "element_out": 13,
                   "element_in": 16, "element_in_cost": 65}],
            target_error=http_error(404),
        )
        self.assertTrue(result["available"])
        self.assertEqual(result["source"], "prior")
        self.assertEqual(result["baseline_gw"], 8)
        self.assertNotIn(13, result["player_ids"])
        self.assertIn(16, result["player_ids"])
        self.assertTrue(result["transfer_history_reconstructed"])
        self.assertFalse(result["needs_confirmation"])
        self.assertIn("Auto-reconstructed from GW8 picks", result["reason"])
        self.assertIn("provisional until GW9 picks publish", result["reason"])

    def test_transactions_are_applied_in_chronological_order(self):
        self.players[16] = {"id": 16, "element_type": 4}
        self.players[17] = {"id": 17, "element_type": 4}
        roster, applied, valid, _ = apply_public_transfers(
            list(range(1, 16)),
            [
                {"event": 9, "time": "2026-10-29T11:00:00Z", "element_out": 16,
                 "element_in": 17, "element_in_cost": 66},
                {"event": 9, "time": "2026-10-29T10:00:00Z", "element_out": 13,
                 "element_in": 16, "element_in_cost": 65},
            ],
            8, 9, self.players,
        )
        self.assertTrue(valid)
        self.assertEqual([row["in"] for row in applied], [16, 17])
        self.assertNotIn(13, roster)
        self.assertNotIn(16, roster)
        self.assertIn(17, roster)

    def test_successful_empty_transfer_history_proves_no_public_moves(self):
        empty = self.load(rows=[], transfer_available=True, target_error=http_error(404))
        failed = self.load(rows=None, transfer_available=False, target_error=http_error(404))
        self.assertTrue(empty["available"])
        self.assertTrue(failed["available"])
        self.assertTrue(empty["transfer_history_reconstructed"])
        self.assertFalse(failed["transfer_history_reconstructed"])
        self.assertFalse(empty["needs_confirmation"])
        self.assertTrue(failed["needs_confirmation"])
        self.assertIn("Auto-reconstructed", empty["reason"])
        self.assertIn("could not be loaded", failed["reason"])

    def test_invalid_prior_roster_never_becomes_available(self):
        invalid = picks()
        invalid["picks"] = invalid["picks"][:-1]
        result = self.load(prior=invalid, target_error=http_error(404))
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "unavailable")
        self.assertIn("invalid", result["reason"])

    def test_429_5xx_and_network_failures_have_distinct_recovery_messages(self):
        for error, expected in (
            (http_error(403), "denied access"),
            (http_error(429), "rate-limited"),
            (http_error(503), "503"),
            (TimeoutError("connection timed out"), "timed out"),
            (json.JSONDecodeError("bad JSON", "{", 1), "malformed"),
        ):
            with self.subTest(error=error):
                result = self.load(target_error=error)
                self.assertFalse(result["available"])
                self.assertIn(expected, result["reason"])

    def test_current_picks_are_verified_without_manual_confirmation(self):
        result = self.load()
        self.assertEqual(result["source"], "current")
        self.assertTrue(result["target_verified"])
        self.assertFalse(result["needs_confirmation"])

    def test_manual_roster_correction_requires_complete_legal_squad(self):
        ids = list(range(1, 16))
        valid, _ = validate_squad_ids(ids, self.players)
        self.assertTrue(valid)
        corrected = ids[:-1] + [16]
        self.players[16] = {"id": 16, "element_type": 4}
        self.assertTrue(validate_squad_ids(corrected, self.players)[0])
        self.assertFalse(validate_squad_ids(corrected[:-1], self.players)[0])

    def test_bad_public_transfer_is_not_applied(self):
        roster, applied, valid, reason = apply_public_transfers(
            list(range(1, 16)),
            [{"event": 9, "time": "2026-10-29", "element_out": 99,
              "element_in": 16, "element_in_cost": 65}],
            8, 9, self.players,
        )
        self.assertEqual(roster, list(range(1, 16)))
        self.assertEqual(applied, [])
        self.assertFalse(valid)
        self.assertIn("missing", reason)

    def test_free_hit_chip_makes_prior_picks_ambiguous(self):
        result = load_public_squad(
            9, self.events, lambda event: (_ for _ in ()).throw(http_error(404)) if event == 9 else picks(),
            [], True, self.players, chip_rows=[{"event": 9, "name": "freehit"}], now=self.now,
        )
        self.assertTrue(result["available"])
        self.assertTrue(result["needs_confirmation"])
        self.assertFalse(result["transfer_history_reconstructed"])
        self.assertIn("Free Hit", result["reason"])


if __name__ == "__main__":
    unittest.main()
