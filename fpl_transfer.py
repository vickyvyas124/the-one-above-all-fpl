"""Transparent per-Gameweek transfer projection arithmetic."""
from decimal import Decimal, InvalidOperation

def official_selling_price_tenths(purchase_price, current_price):
    """FPL sale price in tenths: initial cost plus half profit, rounded down."""
    purchase = int(purchase_price)
    current = int(current_price)
    return current if current <= purchase else purchase + (current - purchase) // 2


def transfer_budget_tenths(selling_price, bank):
    """Available one-player purchase budget in exact FPL tenths of a million."""
    sale, balance = _exact_tenths(selling_price), _exact_tenths(bank)
    if sale is None or balance is None:
        raise ValueError("Selling price and bank must be exact nonnegative FPL tenths.")
    return sale + balance


def is_affordable_tenths(player_price, budget):
    """Whether an FPL price in tenths fits within the exact transfer budget."""
    price, ceiling = _exact_tenths(player_price), _exact_tenths(budget)
    return price is not None and ceiling is not None and price <= ceiling


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


def exact_fpl_tenths(value):
    """Public exact nonnegative tenths parser for FPL cost fields."""
    return _exact_tenths(value)


def pounds_millions_to_tenths(value):
    """Convert a decimal £m amount to exact tenths; reject sub-tenth values."""
    try:
        amount = Decimal(str(value))
        tenths = amount * 10
        if not amount.is_finite() or amount < 0 or tenths != tenths.to_integral_value():
            return None
        return int(tenths)
    except (InvalidOperation, TypeError, ValueError, OverflowError):
        return None


def verified_player_price_tenths(value):
    """Return a positive exact player price, or None when the price is unknown/invalid."""
    price = _exact_tenths(value)
    return price if price is not None and price > 0 else None


def transfer_plan_action_threshold(paid_transfers):
    """Minimum cumulative horizon gain, net of hits, used for dashboard action.

    Free-transfer plans require +4 points; any plan using paid transfers requires
    +6 points. This is a conservative decision margin over the configured horizon,
    not a guarantee of points or rank gain.
    """
    return 6.0 if int(paid_transfers) > 0 else 4.0


def select_actionable_transfer_plan(plans):
    """Choose the strongest plan that clears both total and incremental gates.

    A larger transfer plan must improve on the best plan with one fewer move
    by the same action margin used for the plan's free/paid marginal move.
    This prevents a weak second move from being recommended just because the
    first move already made the combined plan look attractive.
    """
    if not plans:
        return None
    hold = plans.get(0)
    selected = hold
    for count in sorted(key for key in plans if key > 0):
        plan = plans[count]
        if plan.get("affordable") is not True:
            continue
        total_gain = float(plan.get("cumulative_net_gain", float("-inf")))
        if total_gain < transfer_plan_action_threshold(plan.get("paid_transfers", 0)):
            continue
        previous_count = max((key for key in plans if key < count), default=None)
        previous = plans.get(previous_count) if previous_count is not None else None
        if previous is not None:
            incremental_gain = total_gain - float(previous.get("cumulative_net_gain", 0.0))
            marginal_paid = max(0, int(plan.get("paid_transfers", 0))
                                - int(previous.get("paid_transfers", 0)))
            if incremental_gain < transfer_plan_action_threshold(marginal_paid):
                continue
        selected = plan
    return selected


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
    budget_inputs_verified = bool(budget_verified and verified_bank is not None)
    bank_tenths = verified_bank
    for player in squad:
        sale_price = _exact_tenths(player.get("sell_tenths"))
        # A zero sale value is not a valid FPL selling price. Keep unknown values
        # unknown; only the outgoing players in a plan affect its budget status.
        player["sell_tenths"] = sale_price if sale_price and sale_price > 0 else None
    ft = max(0, min(5, int(free_transfers)))
    horizon = max(1, int(horizon))
    max_transfers = min(len(squad), ft + max(0, int(paid_transfer_limit)))
    existing = {int(p["id"]) for p in squad}
    valid_candidates = []
    seen_candidate_ids = set()
    for player in candidates:
        try:
            player_id = int(player["id"])
            price = verified_player_price_tenths(player.get("price_tenths"))
            projections = [float(value) for value in player["projections"][:horizon]]
        except (KeyError, TypeError, ValueError):
            continue
        position = _transfer_position(player)
        if (position not in {"GK", "DEF", "MID", "FWD"}
                or player_id in existing or player_id in seen_candidate_ids
                or len(projections) != horizon or player.get("club") is None):
            continue
        # Zero/malformed costs are not genuine FPL prices. Keep the projection,
        # but never turn an unknown cost into a free transfer.
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
        sale_values = [m["out"].get("sell_tenths") for m in moves]
        buy_values = [m["in"].get("price_tenths") for m in moves]
        plan_budget_verified = (budget_inputs_verified
                                and all(value is not None for value in sale_values + buy_values))
        moved_sales = sum(sale_values) if plan_budget_verified and moves else 0
        moved_buys = sum(buy_values) if plan_budget_verified and moves else 0
        budget_before = (bank_tenths + moved_sales) if plan_budget_verified and bank_tenths is not None else None
        budget_after = (budget_before - moved_buys) if budget_before is not None and moves else budget_before
        budget_reason = None
        if moves and not plan_budget_verified:
            if verified_bank is None or not budget_verified:
                budget_reason = "Bank or manager budget inputs are unverified."
            elif any(value is None for value in sale_values):
                budget_reason = "An outgoing player has no verified manager-specific selling price."
            else:
                budget_reason = "An incoming player has no valid current purchase price."
        return {"transfers": list(moves), "transfer_count": len(moves),
                "free_transfers_used": min(ft, len(moves)),
                "paid_transfers": max(0, len(moves) - ft),
                "hit_cost": max(0, len(moves) - ft) * 4,
                "budget_before_tenths": budget_before,
                "budget_after_tenths": budget_after,
                "affordable": True if not moves else (True if plan_budget_verified else None),
                "projection_only": bool(moves and not plan_budget_verified),
                "budget_reason": budget_reason, "gameweeks": rows,
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
                    sale_values = [m["out"].get("sell_tenths") for m in new_moves]
                    buy_values = [m["in"].get("price_tenths") for m in new_moves]
                    can_verify_budget = (budget_inputs_verified
                                         and all(value is not None for value in sale_values + buy_values))
                    total_sales = sum(sale_values) if can_verify_budget else 0
                    total_buys = sum(buy_values) if can_verify_budget else 0
                    if can_verify_budget and total_sales + bank_tenths < total_buys:
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
        width = max(1, int(beam_width))
        # Keep confirmed-budget search states from being crowded out by a
        # projection-only state with a misleadingly attractive unknown price.
        ranked = sorted(expanded,
                        key=lambda state: sum(r["net_gain"] for r in score(state[0], depth)),
                        reverse=True)
        verified_states, projection_states = [], []
        for state in ranked:
            plan = make_plan(state[0], state[1])
            (verified_states if plan["affordable"] is True else projection_states).append(state)
        beam = verified_states[:width]
        if projection_states and len(beam) < width:
            beam.extend(projection_states[:width - len(beam)])
        elif projection_states and width > 1:
            beam[-1] = projection_states[0]

        best_verified = next((state for state in verified_states), None)
        best_projection = next((state for state in projection_states), None)
        selected_state = best_verified or best_projection
        selected_plan = make_plan(selected_state[0], selected_state[1])
        if best_verified and best_projection:
            projection_plan = make_plan(best_projection[0], best_projection[1])
            if frozenset(int(player["id"]) for player in projection_plan["roster"]) != frozenset(
                    int(player["id"]) for player in selected_plan["roster"]):
                selected_plan["projection_only_alternatives"] = [projection_plan]
        results[depth] = selected_plan
    return results


def _transfer_position(player):
    position = str(player.get("position", player.get("Position", ""))).upper()
    return "GK" if position == "GKP" else position
