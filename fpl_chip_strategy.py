"""Pure helpers for FPL's two-set 2026/27 chip rules."""

CHIP_ALIASES = {
    "wildcard": "wildcard", "wc": "wildcard",
    "bboost": "bench_boost", "benchboost": "bench_boost", "bench_boost": "bench_boost",
    "freehit": "free_hit", "free_hit": "free_hit",
    "3xc": "triple_captain", "triplecaptain": "triple_captain", "triple_captain": "triple_captain",
}

CHIP_TYPES = ("wildcard", "bench_boost", "free_hit", "triple_captain")


def normalize_chip_name(value):
    """Normalize the public history chip names without assuming a fixed spelling."""
    normalized = "".join(char for char in str(value or "").lower() if char.isalnum() or char == "_")
    return CHIP_ALIASES.get(normalized)


def chip_half(gameweek):
    """Return the chip set that can be used in this Gameweek."""
    return "first_half" if int(gameweek) < 20 else "second_half"


def chip_history_for_half(chip_rows, gameweek):
    """Return used event numbers by chip for the half containing ``gameweek``."""
    half = chip_half(gameweek)
    used = {chip: [] for chip in CHIP_TYPES}
    if not isinstance(chip_rows, list):
        return used
    for row in chip_rows:
        if not isinstance(row, dict):
            continue
        chip = normalize_chip_name(row.get("name"))
        try:
            event = int(row.get("event"))
        except (TypeError, ValueError):
            continue
        if chip and chip_half(event) == half:
            used[chip].append(event)
    return used


def chip_planning_gameweeks(current_gameweek, horizon, max_gameweek=38):
    """Return the selected projection horizon, stopping at the chip-set boundary."""
    current = int(current_gameweek)
    half_end = 19 if current < 20 else int(max_gameweek)
    end = min(int(max_gameweek), half_end, current + max(1, int(horizon)) - 1)
    return list(range(current, end + 1))
