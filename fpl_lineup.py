"""Pure validation and selection helpers for the FPL squad and model XI."""

from itertools import combinations


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
    if len(squad) != 15:
        return None
    ids = [int(player["id"]) for player in squad]
    if len(set(ids)) != 15:
        return None
    role_counts = {}
    for player in squad:
        role = str(player.get("Position", "")).upper()
        role_counts[role] = role_counts.get(role, 0) + 1
    if not (role_counts.get("GK", 0) + role_counts.get("GKP", 0) == 2
            and role_counts.get("DEF", 0) == 5
            and role_counts.get("MID", 0) == 5
            and role_counts.get("FWD", 0) == 3):
        return None

    candidates = []
    for player in squad:
        item = dict(player)
        item["GW projected pts"] = float(projection_by_id.get(int(player["id"]), 0.0))
        candidates.append(item)
    best = None
    best_score = float("-inf")
    for combo in combinations(candidates, 11):
        if not is_legal_formation(combo):
            continue
        captain_points = max(item["GW projected pts"] for item in combo)
        score = sum(item["GW projected pts"] for item in combo) + captain_points
        if score > best_score:
            best, best_score = list(combo), score
    if best is None:
        return None

    order = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
    best.sort(key=lambda item: (order[item["Position"].upper()], -item["GW projected pts"], int(item["id"])))
    captain = max(best, key=lambda item: (item["GW projected pts"], -int(item["id"])))
    vice = max((item for item in best if item["id"] != captain["id"]),
               key=lambda item: (item["GW projected pts"], -int(item["id"])))
    starters = {int(item["id"]) for item in best}
    reserves = [item for item in candidates if int(item["id"]) not in starters]
    bench_gk = sorted((item for item in reserves if item["Position"].upper() in {"GK", "GKP"}),
                      key=lambda item: (-item["GW projected pts"], int(item["id"])))
    bench_outfield = sorted((item for item in reserves if item["Position"].upper() not in {"GK", "GKP"}),
                            key=lambda item: (-item["GW projected pts"], int(item["id"])))
    bench = bench_gk[:1] + bench_outfield[:3]
    if len(bench) != 4 or starters.intersection(int(item["id"]) for item in bench):
        return None
    roles = {role: sum(item["Position"].upper() == role for item in best) for role in ("DEF", "MID", "FWD")}
    return {
        "starters": best, "bench": bench, "captain": captain, "vice": vice,
        "formation": f"{roles['DEF']}-{roles['MID']}-{roles['FWD']}",
        "projected_points": sum(item["GW projected pts"] for item in best),
    }
