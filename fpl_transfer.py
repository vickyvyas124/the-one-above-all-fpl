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


def transfer_plan_gate_assessment(plans, transfer_count):
    """Return auditable total and marginal gate values for one plan size."""
    plan = plans.get(int(transfer_count))
    if not plan or int(transfer_count) <= 0:
        return {"total_gain": 0.0, "total_gate": 0.0, "total_pass": True,
                "incremental_gain": None, "incremental_gate": None, "incremental_pass": True}
    total = float(plan.get("cumulative_net_gain", float("nan")))
    total_gate = transfer_plan_action_threshold(plan.get("paid_transfers", 0))
    prior_count = max((count for count in plans if int(count) < int(transfer_count)), default=None)
    prior = plans.get(prior_count) if prior_count is not None else None
    incremental = None
    incremental_gate = None
    incremental_pass = True
    if prior is not None:
        incremental = total - float(prior.get("cumulative_net_gain", 0.0))
        marginal_paid = max(0, int(plan.get("paid_transfers", 0)) - int(prior.get("paid_transfers", 0)))
        incremental_gate = transfer_plan_action_threshold(marginal_paid)
        incremental_pass = incremental >= incremental_gate
    nested_incremental = plan.get("nested_marginal_gain")
    nested_gate = plan.get("nested_marginal_gate")
    nested_pass = nested_incremental is None or nested_gate is None or nested_incremental >= nested_gate
    return {"total_gain": total, "total_gate": total_gate, "total_pass": total >= total_gate,
            "incremental_gain": incremental, "incremental_gate": incremental_gate,
            "incremental_pass": incremental_pass, "nested_incremental_gain": nested_incremental,
            "nested_incremental_gate": nested_gate, "nested_incremental_pass": nested_pass}


def select_actionable_transfer_plan(plans, *, require_verified_budget=True, allow_paid_transfers=True):
    """Choose the strongest plan that clears both total and incremental gates.

    A larger transfer plan must improve on the best plan with one fewer move
    by the same action margin used for the plan's free/paid marginal move.
    This prevents a weak second move from being recommended just because the
    first move already made the combined plan look attractive. Set
    ``require_verified_budget=False`` only to build a clearly provisional
    projected suggestion while the dashboard asks the manager to verify inputs.
    """
    if not plans:
        return None
    hold = plans.get(0)
    selected = hold
    for count in sorted(key for key in plans if key > 0):
        plan = plans[count]
        if not allow_paid_transfers and int(plan.get("paid_transfers", 0)) > 0:
            continue
        if (require_verified_budget and plan.get("affordable") is not True) or plan.get("affordable") is False:
            continue
        assessment = transfer_plan_gate_assessment(plans, count)
        if not assessment["total_pass"] or not assessment["incremental_pass"] or not assessment["nested_incremental_pass"]:
            continue
        selected = plan
    return selected


def build_transfer_candidate_pool(candidates, squad, bank_tenths, *, candidate_limit=40,
                                  budget_verified=True, minimum_goalkeepers=4):
    """Retain positional and price-efficient options before bounded beam search."""
    positions = ("GK", "DEF", "MID", "FWD")
    current_ids = {int(p["id"]) for p in squad}
    budget = _exact_tenths(bank_tenths) if budget_verified else None
    result = {}
    for position in positions:
        outgoing = [p for p in squad if _transfer_position(p) == position]
        affordable_caps = []
        if budget is not None:
            affordable_caps = [budget + sale for p in outgoing
                               if (sale := verified_player_price_tenths(p.get("sell_tenths"))) is not None]
        eligible = []
        for raw in candidates:
            try:
                if int(raw["id"]) in current_ids or _transfer_position(raw) != position:
                    continue
                price = verified_player_price_tenths(raw.get("price_tenths"))
                if budget is not None and affordable_caps and price is not None and not any(price <= cap for cap in affordable_caps):
                    continue
                points = sum(float(v) for v in raw.get("projections", []))
            except (KeyError, TypeError, ValueError):
                continue
            player = dict(raw)
            player["pool_raw_points"] = points
            player["pool_points_per_million"] = points / max(0.1, price / 10) if price is not None else None
            eligible.append(player)
        raw_ranked = sorted(eligible, key=lambda p: p["pool_raw_points"], reverse=True)
        value_ranked = sorted((p for p in eligible if p["pool_points_per_million"] is not None),
                              key=lambda p: p["pool_points_per_million"], reverse=True)
        keep_n = max(1, int(candidate_limit) // len(positions))
        chosen = {int(p["id"]): p for p in raw_ranked[:keep_n]}
        chosen.update({int(p["id"]): p for p in value_ranked[:keep_n]})
        required = int(minimum_goalkeepers) if position == "GK" else 0
        if len(chosen) < required:
            for p in raw_ranked:
                chosen.setdefault(int(p["id"]), p)
                if len(chosen) >= required:
                    break
        result[position] = list(chosen.values())
    pooled = [player for position in positions for player in result[position]]
    return sorted(pooled, key=lambda p: (p["pool_points_per_million"] is not None,
                                         p["pool_points_per_million"] or 0,
                                         p["pool_raw_points"]), reverse=True)


def evaluate_transfer_plan(squad, moves, bank_tenths, free_transfers, scorer, horizon, *,
                           actionable_transfers_ready=False, start_gw=1):
    """Pure legal/financial/projection evaluation for an explicit transfer plan."""
    roster = [dict(p) for p in squad]
    ids = [int(p["id"]) for p in roster]
    result = {"legal": False, "position_valid": False, "club_limit_valid": False,
              "prices_verified": False, "bank_after_tenths": None, "hit_cost": None,
              "net_gain_vs_hold": None, "status": "PROVISIONAL", "reason": ""}
    if len(roster) != 15 or len(set(ids)) != 15:
        result["reason"] = "The current squad must contain 15 unique players."
        return result
    original = {int(p["id"]): p for p in roster}
    incoming_ids, outgoing_ids = set(), set()
    parsed_moves = []
    for move in moves:
        out_id = int(move["out_id"] if isinstance(move, dict) else move[0])
        in_player = move["in"] if isinstance(move, dict) and isinstance(move.get("in"), dict) else move[1]
        in_id = int(in_player["id"])
        if out_id not in original or out_id in outgoing_ids or in_id in ids or in_id in incoming_ids:
            result["reason"] = "A move sells a player outside the original squad or duplicates a player."
            return result
        outgoing_ids.add(out_id)
        incoming_ids.add(in_id)
        parsed_moves.append((original[out_id], dict(in_player)))
    role_counts = {}
    clubs = {}
    for out_player, in_player in parsed_moves:
        if _transfer_position(out_player) != _transfer_position(in_player):
            result["reason"] = "Transfers must replace players in the same position."
            return result
        roster = [p for p in roster if int(p["id"]) != int(out_player["id"])] + [in_player]
    for player in roster:
        role = _transfer_position(player)
        role_counts[role] = role_counts.get(role, 0) + 1
        clubs[player.get("club")] = clubs.get(player.get("club"), 0) + 1
    expected = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    position_valid = role_counts == expected
    club_valid = all(count <= 3 for count in clubs.values())
    result.update({"position_valid": position_valid, "club_limit_valid": club_valid})
    if not position_valid or not club_valid:
        result["reason"] = "The resulting squad violates position counts or the three-per-club limit."
        return result
    result["legal"] = True
    bank = _exact_tenths(bank_tenths)
    sales = [verified_player_price_tenths(out.get("sell_tenths")) for out, _ in parsed_moves]
    buys = [verified_player_price_tenths(incoming.get("price_tenths")) for _, incoming in parsed_moves]
    result["prices_verified"] = bank is not None and all(p is not None for p in sales + buys)
    if result["prices_verified"]:
        balance = bank + sum(sales) - sum(buys)
        result["bank_after_tenths"] = balance
    else:
        balance = None
    ft = max(0, min(5, int(free_transfers)))
    hit = max(0, len(parsed_moves) - ft) * 4
    result["hit_cost"] = hit
    if result["prices_verified"] and balance < 0:
        result["status"] = "UNAFFORDABLE"
        result["reason"] = "The transfer plan exceeds the verified bank balance."
        return result
    if result["legal"] and len(parsed_moves) == 0:
        result["status"] = "AFFORDABLE" if actionable_transfers_ready and result["prices_verified"] else "PROVISIONAL"
    elif result["prices_verified"] and actionable_transfers_ready:
        result["status"] = "AFFORDABLE"
    else:
        result["status"] = "PROVISIONAL"
    horizon = max(1, int(horizon))
    if scorer is not None and result["legal"]:
        hold = sum(float(scorer(squad, index)) for index in range(horizon))
        proposed = sum(float(scorer(roster, index)) for index in range(horizon))
        result["net_gain_vs_hold"] = proposed - hold - hit
    result["reason"] = "" if result["status"] == "AFFORDABLE" else "Verify account inputs and prices before acting."
    result["moves"] = parsed_moves
    result["roster"] = roster
    result["gross_gain_vs_hold"] = ((result["net_gain_vs_hold"] + hit)
                                    if result["net_gain_vs_hold"] is not None else None)
    result["bank_after_tenths"] = balance
    return result


def optimize_transfer_plans(squad, candidates, free_transfers, bank_tenths, *,
                            horizon=5, start_gw=1, beam_width=500,
                            candidate_limit=40, paid_transfer_limit=2,
                            score_roster=None, budget_verified=True, allow_hit=True):
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
    max_paid = max(0, int(paid_transfer_limit)) if allow_hit else 0
    max_transfers = min(len(squad), ft + max_paid)
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
    hold_scores = []
    for gw_i in range(horizon):
        if score_roster:
            hold_scores.append(float(score_roster(squad, gw_i)))
        else:
            hold_scores.append(sum(float(p.get("projections", [0] * horizon)[gw_i]) for p in squad))
    score_cache = {}
    def score(roster, transfers):
        cache_key = (frozenset(int(p["id"]) for p in roster), int(transfers))
        if cache_key in score_cache:
            return score_cache[cache_key]
        rows = []
        hit = max(0, transfers - ft) * 4
        for gw_i in range(horizon):
            if score_roster:
                plan_pts = float(score_roster(roster, gw_i))
            else:
                plan_pts = sum(float(p.get("projections", [0] * horizon)[gw_i]) for p in roster)
            rows.append({"gameweek": int(start_gw) + gw_i, "hold_points": hold_scores[gw_i],
                         "plan_points": plan_pts, "hit_cost": hit if gw_i == 0 else 0,
                         "net_gain": plan_pts - hold_scores[gw_i] - (hit if gw_i == 0 else 0)})
        score_cache[cache_key] = rows
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
        total_net = sum(r["net_gain"] for r in rows)
        nested_margins = []
        for move in moves:
            reduced = [p for p in roster if int(p["id"]) != int(move["in"]["id"])] + [move["out"]]
            reduced_net = sum(r["net_gain"] for r in score(reduced, len(moves) - 1))
            nested_margins.append(total_net - reduced_net)
        nested_margin = min(nested_margins) if nested_margins else None
        nested_paid = max(0, len(moves) - ft) - max(0, len(moves) - 1 - ft) if moves else 0
        return {"transfers": list(moves), "transfer_count": len(moves),
                "free_transfers_used": min(ft, len(moves)),
                "paid_transfers": max(0, len(moves) - ft),
                "hit_cost": max(0, len(moves) - ft) * 4,
                "budget_before_tenths": budget_before,
                "budget_after_tenths": budget_after,
                "affordable": True if not moves else (True if plan_budget_verified else None),
                "projection_only": bool(moves and not plan_budget_verified),
                "budget_reason": budget_reason, "gameweeks": rows,
                "cumulative_net_gain": total_net, "nested_marginal_gain": nested_margin,
                "nested_marginal_gate": transfer_plan_action_threshold(nested_paid) if moves else None,
                "roster": roster}

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
