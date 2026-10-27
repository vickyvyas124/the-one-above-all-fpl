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
        role = str(player.get("position", player.get("Position", ""))).upper()
        role = "GK" if role == "GKP" else role
        role_counts[role] = role_counts.get(role, 0) + 1
        club = player.get("club")
        club_counts[club] = club_counts.get(club, 0) + 1
    if role_counts != {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3} or any(n > 3 for n in club_counts.values()):
        return {}
    ft = max(0, min(5, int(free_transfers)))
    horizon = max(1, int(horizon))
    max_transfers = min(len(squad), ft + max(1, int(paid_transfer_limit)))
    existing = {int(p["id"]) for p in squad}
    candidates = [p for p in candidates if int(p["id"]) not in existing]
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
                "budget_before_tenths": (int(bank_tenths or 0) + moved_sales) if budget_verified else None,
                "budget_after_tenths": (int(bank_tenths or 0) + moved_sales - moved_buys) if budget_verified else None,
                "affordable": True if budget_verified else None, "gameweeks": rows,
                "cumulative_net_gain": sum(r["net_gain"] for r in rows), "roster": roster}

    results = {0: make_plan(squad, [])}
    # State is (roster, moves, outgoing IDs, incoming IDs); each depth is
    # ranked by projected net value and capped to a fixed beam width.
    beam = [(squad, [], set(), set())]
    for depth in range(1, max_transfers + 1):
        expanded = []
        for roster, moves, outs, ins in beam:
            for out in roster:
                oid = int(out["id"])
                if oid in outs:
                    continue
                for incoming in candidates:
                    iid = int(incoming["id"])
                    if iid in ins or incoming.get("position") != out.get("position"):
                        continue
                    new_roster = [p for p in roster if int(p["id"]) != oid] + [incoming]
                    clubs = {}
                    for p in new_roster:
                        clubs[p["club"]] = clubs.get(p["club"], 0) + 1
                    if any(n > 3 for n in clubs.values()):
                        continue
                    new_moves = moves + [{"out": out, "in": incoming}]
                    total_sales = sum(int(m["out"].get("sell_tenths", 0)) for m in new_moves)
                    total_buys = sum(int(m["in"]["price_tenths"]) for m in new_moves)
                    if budget_verified and total_sales + int(bank_tenths or 0) < total_buys:
                        continue
                    expanded.append((new_roster, new_moves, outs | {oid}, ins | {iid}))
        if not expanded:
            break
        # A plan can be reached in several transaction orders; keep one state
        # for each final set of moved players before applying the beam cap.
        unique_states = {}
        for state in expanded:
            key = (frozenset(state[2]), frozenset(state[3]))
            unique_states.setdefault(key, state)
        expanded = list(unique_states.values())
        expanded.sort(key=lambda state: sum(r["net_gain"] for r in score(state[0], depth)), reverse=True)
        beam = expanded[:max(1, int(beam_width))]
        best_state = beam[0]
        results[depth] = make_plan(best_state[0], best_state[1])
    return results
