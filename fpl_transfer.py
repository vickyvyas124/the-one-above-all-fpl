"""Transparent per-Gameweek transfer projection arithmetic."""


def official_selling_price_tenths(purchase_price, current_price):
    """FPL sale price in tenths: initial cost plus half profit, rounded down."""
    purchase = int(purchase_price)
    current = int(current_price)
    return current if current <= purchase else purchase + (current - purchase) // 2


def transfer_hit_cost(free_transfers, allow_hit):
    """Cost for the one proposed transfer under the current hit setting."""
    return 4 if bool(allow_hit) and int(free_transfers) == 0 else 0


def gameweek_projection_series(player, team, fixtures, *, horizon, start_gw, estimate_points):
    """Return one engine projection per assigned Gameweek, including blanks."""
    result = []
    for gameweek in range(int(start_gw), int(start_gw) + int(horizon)):
        points, _, _ = estimate_points(player, team, fixtures, horizon=1, start_gw=gameweek)
        result.append({"gameweek": gameweek, "points": float(points)})
    return result


def transfer_projection_comparison(outgoing, incoming, transfer_hit=0):
    """Compare holding with a transfer; charge any hit once in the first GW."""
    if len(outgoing) != len(incoming):
        raise ValueError("Outgoing and incoming projection horizons must match.")
    cumulative = 0.0
    rows = []
    for index, (sell, buy) in enumerate(zip(outgoing, incoming)):
        if int(sell["gameweek"]) != int(buy["gameweek"]):
            raise ValueError("Outgoing and incoming Gameweeks must match.")
        raw_delta = float(buy["points"]) - float(sell["points"])
        hit = float(transfer_hit) if index == 0 else 0.0
        net_delta = raw_delta - hit
        cumulative += net_delta
        rows.append({
            "Gameweek": int(sell["gameweek"]),
            "Hold current player": float(sell["points"]),
            "Transfer candidate": float(buy["points"]),
            "Raw difference": raw_delta,
            "Transfer hit": hit,
            "Net difference": net_delta,
            "Cumulative net": cumulative,
        })
    return rows
