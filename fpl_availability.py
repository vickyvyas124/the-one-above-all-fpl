"""Shared, explicit availability adjustments from FPL's structured player fields.

The factor is a projection heuristic, not a calibrated probability. Free-text
news is deliberately not parsed into an injury classification.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


def availability_adjustment(player: Mapping[str, Any] | None) -> float:
    """Return the common live availability multiplier used by projections and decisions.

    Official unavailable statuses retain the existing conservative 0.20 factor.
    Doubtful players use FPL's chance field when valid, otherwise the existing
    0.55 estimate. Explicitly available players without a chance field are 1.0;
    missing/unknown status evidence uses the conservative 0.90 estimate.
    """
    player = player or {}
    status = str(player.get("status") or "").strip().lower()
    if status in {"i", "s", "u"}:
        return 0.20

    raw_chance = player.get("chance_of_playing_next_round")
    chance = None
    if raw_chance is not None:
        try:
            parsed = float(raw_chance)
            if math.isfinite(parsed):
                chance = max(0.20, min(1.0, parsed / 100.0))
        except (TypeError, ValueError):
            pass

    if status == "d":
        return chance if chance is not None else 0.55
    if chance is not None:
        return chance
    if status == "a":
        return 1.0
    return 0.90
