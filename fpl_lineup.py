"""Pure validation and selection helpers for the FPL squad and model XI."""

import math


FORMATION_ORDER = ((3, 4, 3), (3, 5, 2), (4, 3, 3), (4, 4, 2),
                   (4, 5, 1), (5, 2, 3), (5, 3, 2), (5, 4, 1))
ROLE_COUNTS = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}


def _role(player):
    role = str(player.get("Position", player.get("position", ""))).upper()
    return "GK" if role == "GKP" else role


def _club(player):
    return player.get("Club", player.get("club", player.get("team")))


def _valid_squad(squad, *, require_clubs=True):
    if len(squad) != 15:
        return False
    try:
        ids = [int(player["id"]) for player in squad]
    except (KeyError, TypeError, ValueError):
        return False
    if len(set(ids)) != 15 or {_role(player) for player in squad} - set(ROLE_COUNTS):
        return False
    if {role: sum(_role(player) == role for player in squad) for role in ROLE_COUNTS} != ROLE_COUNTS:
        return False
    clubs = [_club(player) for player in squad]
    if require_clubs and any(club is None or str(club).strip() == "" for club in clubs):
        return False
    club_counts = {}
    for club in clubs:
        if club is not None:
            key = str(club)
            club_counts[key] = club_counts.get(key, 0) + 1
    return all(count <= 3 for count in club_counts.values())


def score_starting_xi(squad, projection_by_id):
    """Fast expected XI plus captain score without building lineup details."""
    if not _valid_squad(squad, require_clubs=False):
        return 0.0
    grouped = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for player in squad:
        try:
            points = float(projection_by_id.get(int(player["id"]), 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(points):
            return 0.0
        grouped[_role(player)].append(points)
    goalkeepers = sorted(grouped["GK"], reverse=True)
    best = float("-inf")
    for defenders in range(3, 6):
        for midfielders in range(2, 6):
            forwards = 10 - defenders - midfielders
            if not 1 <= forwards <= 3:
                continue
            chosen = (goalkeepers[:1] + sorted(grouped["DEF"], reverse=True)[:defenders]
                      + sorted(grouped["MID"], reverse=True)[:midfielders]
                      + sorted(grouped["FWD"], reverse=True)[:forwards])
            best = max(best, sum(chosen) + max(chosen))
    return best if best != float("-inf") else 0.0


def validate_public_picks(picks, known_player_ids):
    """Return (valid, reason) for one complete, ordered 15-player FPL squad."""
    if not isinstance(picks, list) or len(picks) != 15:
        return False, "FPL did not return exactly 15 picks."
    ids, positions = [], []
    try:
        for pick in picks:
            ids.append(int(pick["element"]))
            positions.append(int(pick["position"]))
    except (KeyError, TypeError, ValueError):
        return False, "A pick is missing a valid player or squad position."
    if len(set(ids)) != 15:
        return False, "FPL picks contain duplicate player identifiers."
    known = {int(pid) for pid in known_player_ids}
    if not set(ids).issubset(known):
        return False, "FPL picks contain a player missing from the current player data."
    if sorted(positions) != list(range(1, 16)):
        return False, "FPL picks do not have unique positions 1 through 15."
    if hasattr(known_player_ids, "get"):
        role_by_type = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
        role_counts = {role: 0 for role in role_by_type.values()}
        for player_id in ids:
            try:
                role = role_by_type[int(known_player_ids[player_id].get("element_type"))]
            except (KeyError, TypeError, ValueError):
                return False, "A selected player has an unknown FPL position."
            role_counts[role] += 1
        if role_counts != {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}:
            return False, "The selected players do not form a legal FPL 15-player squad."
        club_counts = {}
        for player_id in ids:
            club = known_player_ids[player_id].get("team")
            if club is None:
                return False, "A selected player is missing a current club assignment."
            club_counts[str(club)] = club_counts.get(str(club), 0) + 1
        if any(count > 3 for count in club_counts.values()):
            return False, "The selected players exceed the maximum of three players from one club."
    return True, ""


def is_legal_formation(players):
    """Whether an XI satisfies FPL goalkeeper and outfield formation rules."""
    if len(players) != 11:
        return False
    counts = {role: 0 for role in ("GK", "GKP", "DEF", "MID", "FWD")}
    for player in players:
        role = str(player.get("Position", player.get("position", ""))).upper()
        if role not in counts:
            return False
        counts[role] += 1
    gk = counts["GK"] + counts["GKP"]
    return (gk == 1 and 3 <= counts["DEF"] <= 5
            and 2 <= counts["MID"] <= 5 and 1 <= counts["FWD"] <= 3)


def select_model_lineup(squad, projection_by_id):
    """Select one legal XI, captain, vice-captain and four-player bench.

    `projection_by_id` is the one-GW projection map shared by every consumer.
    Returns None unless the entire squad is a valid 15-player FPL roster.
    """
    if not _valid_squad(squad):
        return None

    candidates = []
    for player in squad:
        try:
            points = float(projection_by_id.get(int(player["id"]), 0.0))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(points):
            return None
        item = dict(player)
        item["Position"] = _role(item)
        item["GW projected pts"] = points
        candidates.append(item)
    groups = {role: sorted((item for item in candidates if item["Position"] == role),
                           key=lambda item: (-item["GW projected pts"], int(item["id"])))
              for role in ROLE_COUNTS}
    formation_rows = []
    for defenders, midfielders, forwards in FORMATION_ORDER:
        starters = (groups["GK"][:1] + groups["DEF"][:defenders]
                    + groups["MID"][:midfielders] + groups["FWD"][:forwards])
        if len(starters) != 11 or not is_legal_formation(starters):
            continue
        base_points = sum(item["GW projected pts"] for item in starters)
        captain = min(starters, key=lambda item: (-item["GW projected pts"], int(item["id"])))
        vice = min((item for item in starters if int(item["id"]) != int(captain["id"])),
                   key=lambda item: (-item["GW projected pts"], int(item["id"])))
        formation_rows.append({
            "formation": f"{defenders}-{midfielders}-{forwards}",
            "starters": starters, "captain": captain, "vice": vice,
            "base_xi_projected_points": base_points,
            "captain_extra_points": captain["GW projected pts"],
            "projected_points": base_points + captain["GW projected pts"],
        })
    if len(formation_rows) != len(FORMATION_ORDER):
        return None
    formation_rank = {f"{d}-{m}-{f}": index for index, (d, m, f) in enumerate(FORMATION_ORDER)}
    formation_rows.sort(key=lambda row: (-row["projected_points"], formation_rank[row["formation"]],
                                         tuple(int(item["id"]) for item in row["starters"])))
    best = formation_rows[0]
    second = formation_rows[1]
    starters = best["starters"]
    starting_ids = {int(item["id"]) for item in starters}
    reserves = [item for item in candidates if int(item["id"]) not in starting_ids]
    bench_gk = sorted((item for item in reserves if item["Position"] == "GK"),
                      key=lambda item: (-item["GW projected pts"], int(item["id"])))
    bench_outfield = sorted((item for item in reserves if item["Position"] != "GK"),
                            key=lambda item: (-item["GW projected pts"], int(item["id"])))
    bench = bench_gk[:1] + bench_outfield[:3]
    if len(bench) != 4 or starting_ids.intersection(int(item["id"]) for item in bench):
        return None

    role_quota = {"GK": 1, "DEF": int(best["formation"].split("-")[0]),
                  "MID": int(best["formation"].split("-")[1]),
                  "FWD": int(best["formation"].split("-")[2])}
    for role, quota in role_quota.items():
        for item in starters:
            if item["Position"] == role:
                item["Selection reason"] = (
                    f"Selected among the top {quota} projected {role} options for the best-scoring legal formation."
                )
    best["captain"]["Selection reason"] = "Highest projected points among starters; captaincy adds one extra projection."
    best["vice"]["Selection reason"] = "Second-highest projected points among starters; backup captain."
    bench[0]["Selection reason"] = "Reserve goalkeeper; FPL substitution rules require a goalkeeper substitute."
    for index, item in enumerate(bench[1:], 1):
        item["Selection reason"] = f"Outfield substitute {index}, ordered by projected points among unused squad players."

    score_rows = [{key: row[key] for key in (
        "formation", "base_xi_projected_points", "captain_extra_points", "projected_points"
    )} for row in formation_rows]
    return {
        "starters": starters, "bench": bench, "captain": best["captain"], "vice": best["vice"],
        "formation": best["formation"],
        "formation_scores": score_rows,
        "formations_evaluated": [f"{d}-{m}-{f}" for d, m, f in FORMATION_ORDER],
        "second_best_formation": second["formation"],
        "second_best_projected_points": second["projected_points"],
        "formation_gap_points": best["projected_points"] - second["projected_points"],
        "selection_reasons": {int(item["id"]): item["Selection reason"] for item in starters + bench},
        "base_xi_projected_points": best["base_xi_projected_points"],
        "captain_extra_points": best["captain_extra_points"],
        # FPL scores the captain at 2x; the lineup sum already includes their base 1x.
        "projected_points": best["projected_points"],
    }


def published_lineup_from_picks(picks, squad, projection_by_id, *, verified=False):
    """Build a current published XI only from an explicitly verified target-GW pick set."""
    if not verified or not _valid_squad(squad):
        return None
    known = {}
    role_to_type = {"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}
    for player in squad:
        club = _club(player)
        if club is None:
            return None
        known[int(player["id"])] = {"element_type": role_to_type[_role(player)], "team": club}
    valid, _reason = validate_public_picks(picks, known)
    if not valid:
        return None
    player_by_id = {int(player["id"]): player for player in squad}
    try:
        ordered_picks = sorted(picks, key=lambda pick: int(pick["position"]))
        starters = []
        bench = []
        for pick in ordered_picks:
            player_id = int(pick["element"])
            item = dict(player_by_id[player_id])
            item["Position"] = _role(item)
            points = float(projection_by_id[player_id])
            if not math.isfinite(points):
                return None
            item["GW projected pts"] = points
            (starters if int(pick["position"]) <= 11 else bench).append(item)
        captains = [int(pick["element"]) for pick in ordered_picks if pick.get("is_captain")]
        vice_captains = [int(pick["element"]) for pick in ordered_picks if pick.get("is_vice_captain")]
        if len(captains) != 1 or len(vice_captains) != 1:
            return None
        captain_id, vice_id = captains[0], vice_captains[0]
    except (KeyError, TypeError, ValueError, StopIteration):
        return None
    if (len(starters) != 11 or len(bench) != 4 or not is_legal_formation(starters)
            or captain_id not in {int(p["id"]) for p in starters}
            or vice_id not in {int(p["id"]) for p in starters} or captain_id == vice_id
            or sum(item["Position"] == "GK" for item in bench) != 1):
        return None
    formation_counts = {role: sum(item["Position"] == role for item in starters) for role in ("DEF", "MID", "FWD")}
    captain = next(item for item in starters if int(item["id"]) == captain_id)
    vice = next(item for item in starters if int(item["id"]) == vice_id)
    base_points = sum(item["GW projected pts"] for item in starters)
    return {
        "starters": starters, "bench": bench, "captain": captain, "vice": vice,
        "formation": f"{formation_counts['DEF']}-{formation_counts['MID']}-{formation_counts['FWD']}",
        "base_xi_projected_points": base_points,
        "captain_extra_points": captain["GW projected pts"],
        "projected_points": base_points + captain["GW projected pts"],
        "source": "Verified current public GW picks",
    }
