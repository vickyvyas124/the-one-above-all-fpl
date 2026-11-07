"""Verified account snapshot shared by all optimizer recommendations."""

from dataclasses import dataclass
from datetime import datetime, timezone

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
                           prices_by_id, synced_at=None):
    """Validate identity, budget and FT inputs without inventing missing prices."""
    ids = tuple(int(pid) for pid in (squad_ids or ()))
    legal, _ = validate_squad_ids(ids, players_by_id)
    squad_status = ("VERIFIED" if squad_verified and legal and not squad_provisional else
                    "PROVISIONAL" if squad_verified and legal else "UNVERIFIED")
    bank = exact_fpl_tenths(bank_value)
    bank_status = "VERIFIED" if bank is not None else "UNVERIFIED"
    prices_complete = len(ids) == 15 and all(
        exact_fpl_tenths(prices_by_id.get(pid)) is not None
        and exact_fpl_tenths(prices_by_id.get(pid)) > 0 for pid in ids
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
