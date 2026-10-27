"""Transparent per-Gameweek transfer projection arithmetic."""

def official_selling_price_tenths(purchase_price, current_price):
    """FPL sale price in tenths: initial cost plus half profit, rounded down."""
    purchase = int(purchase_price)
    current = int(current_price)
    return current if current <= purchase else purchase + (current - purchase) // 2


def transfer_budget_tenths(selling_price, bank):
    """Available one-player purchase budget in exact FPL tenths of a million."""
    return int(selling_price) + int(bank)


def is_affordable_tenths(player_price, budget):
    """Whether an FPL price in tenths fits within the exact transfer budget."""
    return int(player_price) <= int(budget)


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


def _exact_tenths(value):
    """Return an exact integer tenths value, rejecting fractional prices."""
    try:
        if isinstance(value, bool):
            return None
        integer = int(value)
        return integer if float(value) == integer and integer >= 0 else None
    except (TypeError, ValueError, OverflowError):
        return None


def optimize_transfer_plans(squad, candidates, free_transfers, bank_tenths, *,
                            horizon=5, start_gw=1, beam_width=500,
                            candidate_limit=40, paid_transfer_limit=2,
                            score_roster=None, budget_verified=True):
    """Bounded beam search for legal, budgeted transfer plans.

    Players are mappings with ``id``, ``position``, ``club``, ``price_tenths``,
    and a ``projections`` sequence (one value per horizon GW). Squad members
    also need ``sell_tenths``. ``score_roster(roster, gw_index)`` may provide
    expected starting-XI points; otherwise projected points are summed.
    Returns the best plan at each transfer count, including hold.
    """
    squad = [dict(p) for p in squad]
    candidates = [dict(p) for p in candidates]
    if len(squad) != 15 or len({int(p["id"]) for p in squad}) != 15:
        return {}
    role_counts = {}
    club_counts = {}
    for player in squad:
        role = _transfer_position(player)
        role_counts[role] = role_counts.get(role, 0) + 1
        club = player.get("club")
        club_counts[club] = club_counts.get(club, 0) + 1
    if role_counts != {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3} or any(n > 3 for n in club_counts.values()):
        return {}
    # Treat malformed bank or missing manager-specific sale prices as
    # unverified. Keep projections available, but never label them affordable.
    verified_bank = _exact_tenths(bank_tenths)
    budget_verified = bool(budget_verified and verified_bank is not None)
    bank_tenths = verified_bank if budget_verified else None
    for player in squad:
        sale_price = _exact_tenths(player.get("sell_tenths"))
        if sale_price is None:
            budget_verified = False
            player["sell_tenths"] = 0
        else:
            player["sell_tenths"] = sale_price
    ft = max(0, min(5, int(free_transfers)))
    horizon = max(1, int(horizon))
    max_transfers = min(len(squad), ft + max(0, int(paid_transfer_limit)))
    existing = {int(p["id"]) for p in squad}
    valid_candidates = []
    seen_candidate_ids = set()
    for player in candidates:
        try:
            player_id = int(player["id"])
            price = _exact_tenths(player.get("price_tenths"))
            projections = [float(value) for value in player["projections"][:horizon]]
        except (KeyError, TypeError, ValueError):
            continue
        position = _transfer_position(player)
        if (position not in {"GK", "DEF", "MID", "FWD"}
                or player_id in existing or player_id in seen_candidate_ids
                or len(projections) != horizon or player.get("club") is None):
            continue
        if price is None:
            budget_verified = False
            price = 0
        player["id"] = player_id
        player["position"] = position
        player["price_tenths"] = price
        player["projections"] = projections
        seen_candidate_ids.add(player_id)
        valid_candidates.append(player)
    candidates = valid_candidates
    candidates.sort(key=lambda p: sum(map(float, p.get("projections", [])[:horizon])), reverse=True)
    candidates = candidates[:max(1, int(candidate_limit))]
    def score(roster, transfers):
        rows = []
        hit = max(0, transfers - ft) * 4
        for gw_i in range(horizon):
            if score_roster:
                hold_pts = float(score_roster(squad, gw_i))
                plan_pts = float(score_roster(roster, gw_i))
            else:
                hold_pts = sum(float(p.get("projections", [0] * horizon)[gw_i]) for p in squad)
                plan_pts = sum(float(p.get("projections", [0] * horizon)[gw_i]) for p in roster)
            rows.append({"gameweek": int(start_gw) + gw_i, "hold_points": hold_pts,
                         "plan_points": plan_pts, "hit_cost": hit if gw_i == 0 else 0,
                         "net_gain": plan_pts - hold_pts - (hit if gw_i == 0 else 0)})
        return rows

    def make_plan(roster, moves):
        rows = score(roster, len(moves))
        moved_sales = sum(int(m["out"].get("sell_tenths", 0)) for m in moves)
        moved_buys = sum(int(m["in"]["price_tenths"]) for m in moves)
        return {"transfers": list(moves), "transfer_count": len(moves),
                "free_transfers_used": min(ft, len(moves)),
                "paid_transfers": max(0, len(moves) - ft),
                "hit_cost": max(0, len(moves) - ft) * 4,
                "budget_before_tenths": (bank_tenths + moved_sales) if budget_verified else None,
                "budget_after_tenths": (bank_tenths + moved_sales - moved_buys) if budget_verified else None,
                "affordable": True if budget_verified else None, "gameweeks": rows,
                "cumulative_net_gain": sum(r["net_gain"] for r in rows), "roster": roster}

    results = {0: make_plan(squad, [])}
    # State is (roster, moves, outgoing IDs, incoming IDs); each depth is
    # ranked by projected net value and capped to a fixed beam width.
    beam = [(squad, [], set(), set())]
    for depth in range(1, max_transfers + 1):
        expanded = []
        for roster, moves, outs, ins in beam:
            # Each transaction must sell a distinct member of the original
            # squad. An acquired player cannot be sold in this same plan.
            for out in squad:
                oid = int(out["id"])
                if oid in outs:
                    continue
                for incoming in candidates:
                    iid = int(incoming["id"])
                    if iid in ins or _transfer_position(incoming) != _transfer_position(out):
                        continue
                    if int(incoming["id"]) not in {int(p["id"]) for p in roster}:
                        new_roster = [p for p in roster if int(p["id"]) != oid] + [incoming]
                    else:
                        continue
                    clubs = {}
                    for p in new_roster:
                        clubs[p["club"]] = clubs.get(p["club"], 0) + 1
                    if any(n > 3 for n in clubs.values()):
                        continue
                    new_moves = moves + [{"out": out, "in": incoming}]
                    total_sales = sum(int(m["out"].get("sell_tenths", 0)) for m in new_moves)
                    total_buys = sum(int(m["in"]["price_tenths"]) for m in new_moves)
                    if budget_verified and total_sales + bank_tenths < total_buys:
                        continue
                    expanded.append((new_roster, new_moves, outs | {oid}, ins | {iid}))
        if not expanded:
            break
        # A plan can be reached in several transaction orders; keep one state
        # for each final set of moved players before applying the beam cap.
        unique_states = {}
        for state in expanded:
            key = frozenset(int(player["id"]) for player in state[0])
            unique_states.setdefault(key, state)
        expanded = list(unique_states.values())
        expanded.sort(key=lambda state: sum(r["net_gain"] for r in score(state[0], depth)), reverse=True)
        beam = expanded[:max(1, int(beam_width))]
        best_state = beam[0]
        results[depth] = make_plan(best_state[0], best_state[1])
    return results


def _transfer_position(player):
    position = str(player.get("position", player.get("Position", ""))).upper()
    return "GK" if position == "GKP" else position
