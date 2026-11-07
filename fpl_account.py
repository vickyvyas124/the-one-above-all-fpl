"""Verified account snapshot shared by all optimizer recommendations."""

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import io

from fpl_squad import validate_squad_ids
from fpl_transfer import exact_fpl_tenths


@dataclass(frozen=True)
class AccountSnapshot:
    manager_id: int
    gameweek: int
    squad_ids: tuple
    squad_status: str
    bank_tenths: int | None
    bank_source: str
    budget_status: str
    free_transfers: int | None
    free_transfers_status: str
    synced_at_utc: str | None

    @property
    def recommendations_ready(self):
        return self.squad_status in {"VERIFIED", "PROVISIONAL"} and len(self.squad_ids) == 15

    @property
    def actionable_transfers_ready(self):
        return (self.recommendations_ready and self.budget_status == "VERIFIED"
                and self.bank_tenths is not None and self.free_transfers_status == "VERIFIED")


def build_account_snapshot(*, manager_id, gameweek, squad_ids, players_by_id,
                           squad_verified, squad_provisional=False, bank_value, bank_source,
                           free_transfers, free_transfers_verified,
                           prices_by_id, selling_prices_by_id=None, synced_at=None):
    """Validate identity, budget and FT inputs without inventing missing prices."""
    ids = tuple(int(pid) for pid in (squad_ids or ()))
    legal, _ = validate_squad_ids(ids, players_by_id)
    squad_status = ("VERIFIED" if squad_verified and legal and not squad_provisional else
                    "PROVISIONAL" if squad_verified and legal else "UNVERIFIED")
    bank = exact_fpl_tenths(bank_value)
    bank_status = "VERIFIED" if bank is not None else "UNVERIFIED"
    selling_prices_by_id = selling_prices_by_id or {}
    prices_complete = len(ids) == 15 and all(
        ((exact_fpl_tenths(prices_by_id.get(pid)) or
          exact_fpl_tenths(selling_prices_by_id.get(pid))) is not None)
        and (exact_fpl_tenths(prices_by_id.get(pid)) or
             exact_fpl_tenths(selling_prices_by_id.get(pid))) > 0 for pid in ids
    )
    budget_status = "VERIFIED" if bank_status == "VERIFIED" and prices_complete else "UNVERIFIED"
    try:
        ft = int(free_transfers)
        ft_valid = 0 <= ft <= 5
    except (TypeError, ValueError, OverflowError):
        ft, ft_valid = None, False
    ft_status = "VERIFIED" if free_transfers_verified and ft_valid else "UNVERIFIED"
    stamp = synced_at
    if isinstance(stamp, datetime):
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        stamp = stamp.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    return AccountSnapshot(int(manager_id), int(gameweek), ids, squad_status, bank,
                           str(bank_source), budget_status, ft if ft_valid else None,
                           ft_status, str(stamp) if stamp is not None else None)


def parse_selling_price_import(text, squad_ids, players_by_id, bank_tenths, team_value_tenths):
    """Validate pasted CSV: player_id,sell_tenths; never infer omitted prices."""
    try:
        rows = list(csv.DictReader(io.StringIO(str(text or ""))))
    except (csv.Error, TypeError, ValueError):
        return None, "Could not parse the pasted price table."
    if not rows or not {"player_id", "sell_tenths"}.issubset(set(rows[0])):
        return None, "Expected CSV headers: player_id,sell_tenths (prices in £0.1m units)."
    expected = {int(pid) for pid in squad_ids}
    prices = {}
    for row in rows:
        try:
            pid = int(row["player_id"])
            sale = verified_price = exact_fpl_tenths(row["sell_tenths"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None, "Every row needs an exact player ID and integer-tenths selling price."
        current = exact_fpl_tenths(players_by_id.get(pid, {}).get("now_cost"))
        if pid not in expected or pid in prices or verified_price is None or verified_price <= 0:
            return None, "Player IDs must be unique current squad members, with positive exact-tenth prices."
        if current is None or sale > current:
            return None, f"Selling price for player {pid} exceeds or cannot be checked against current official price."
        prices[pid] = sale
    if set(prices) != expected:
        return None, "Import must include one selling price for every player in the current 15-player squad."
    bank = exact_fpl_tenths(bank_tenths)
    team_value = exact_fpl_tenths(team_value_tenths)
    market_total = sum(exact_fpl_tenths(players_by_id[pid].get("now_cost")) or 0 for pid in expected)
    if bank is None or team_value is None or abs(bank + market_total - team_value) > 1:
        return None, "Bank plus current official squad market prices does not reconcile to team value within £0.1m."
    return prices, "Verified manager-confirmed selling prices reconciled to bank and team value."
