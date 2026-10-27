"""Safe public FPL squad loading for pre-deadline Gameweeks."""

from datetime import datetime, timezone
import json

from fpl_lineup import validate_public_picks


def _status_code(error):
    response = getattr(error, "response", None)
    try:
        return int(getattr(response, "status_code", 0)) or None
    except (TypeError, ValueError):
        return None


def describe_picks_error(error):
    """Return a user-facing reason without conflating HTTP and data errors."""
    status = _status_code(error)
    if status == 403:
        return "FPL denied access to these public picks (403). Check the team ID and retry later."
    if status == 404:
        return "FPL has not published picks for this Gameweek (404)."
    if status == 429:
        return "FPL rate-limited the picks request (429). Wait briefly and refresh."
    if status is not None and status >= 500:
        return f"FPL picks service returned an error ({status}). Retry later."
    if isinstance(error, (TimeoutError, ConnectionError)) or "timeout" in type(error).__name__.lower():
        return "The public FPL picks request timed out. Check the connection and retry."
    if isinstance(error, json.JSONDecodeError) or "jsondecode" in type(error).__name__.lower():
        return "FPL returned malformed picks data. Retry later; no squad was assumed."
    if status is not None:
        return f"The public FPL picks request failed ({status}). Retry later."
    return "The public FPL picks request failed because of a network or response error. Retry later."


def validate_squad_ids(player_ids, known_player_ids):
    """Validate a manually confirmed roster's identity and FPL position counts."""
    if not isinstance(player_ids, (list, tuple)) or len(player_ids) != 15:
        return False, "Select exactly 15 players."
    try:
        ids = [int(player_id) for player_id in player_ids]
    except (TypeError, ValueError):
        return False, "Every selected player must have a valid FPL identifier."
    if len(set(ids)) != 15:
        return False, "The selected roster contains duplicate players."
    if not set(ids).issubset({int(pid) for pid in known_player_ids}):
        return False, "The selected roster includes a player missing from current FPL data."
    role_by_type = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    counts = {role: 0 for role in role_by_type.values()}
    for player_id in ids:
        try:
            role = role_by_type[int(known_player_ids[player_id].get("element_type"))]
        except (KeyError, TypeError, ValueError):
            return False, "A selected player has an unknown FPL position."
        counts[role] += 1
    if counts != {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}:
        return False, "The selected roster must contain 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards."
    return True, ""


def apply_public_transfers(player_ids, transfer_rows, baseline_gw, target_gw, known_player_ids):
    """Apply verifiable public transactions in event/time order, fail closed on bad rows."""
    if not isinstance(transfer_rows, list):
        return list(player_ids), [], False, "Transfer history is unavailable; it cannot confirm that no moves occurred."
    relevant = []
    try:
        for row in transfer_rows:
            event = int(row.get("event"))
            if int(baseline_gw) < event <= int(target_gw):
                out_id, in_id = int(row["element_out"]), int(row["element_in"])
                if out_id not in known_player_ids or in_id not in known_player_ids:
                    raise ValueError("A transfer references a player missing from current public data.")
                try:
                    cost = int(row["element_in_cost"])
                except (KeyError, TypeError, ValueError):
                    cost = None
                timestamp = datetime.fromisoformat(str(row["time"]).replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("A transfer timestamp has no timezone.")
                if cost is not None and cost <= 0:
                    cost = None
                relevant.append((event, timestamp, out_id, in_id, cost, row))
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        return list(player_ids), [], False, f"Transfer history contains an unverifiable transaction: {exc}"

    if not relevant:
        return list(player_ids), [], False, "No public transactions were returned for this window; an empty history does not prove that no moves or chips occurred."

    roster = [int(player_id) for player_id in player_ids]
    applied = []
    for _event, _time, out_id, in_id, cost, row in sorted(relevant, key=lambda item: (item[0], item[1])):
        if out_id not in roster or in_id in roster:
            return list(player_ids), [], False, "A public transfer does not match the prior roster; correct the roster manually."
        roster[roster.index(out_id)] = in_id
        applied.append({"out": out_id, "in": in_id, "purchase_price": cost, "event": _event, "raw": row})
    valid, reason = validate_squad_ids(roster, known_player_ids)
    if not valid:
        return list(player_ids), [], False, f"Applying public transfers did not produce a legal squad: {reason}"
    return roster, applied, True, ""


def load_public_squad(target_gw, events, fetch_picks, transfer_rows, transfer_history_available,
                      known_player_ids, *, now=None):
    """Load current picks, or a validated prior squad for a pre-deadline 404."""
    now = now or datetime.now(timezone.utc)
    try:
        payload = fetch_picks(int(target_gw))
    except Exception as exc:
        status = _status_code(exc)
        event = next((row for row in events if int(row.get("id", -1)) == int(target_gw)), {})
        try:
            deadline = datetime.fromisoformat(str(event.get("deadline_time")).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            deadline = None
        upcoming = bool(event.get("is_next")) or bool(deadline and deadline > now)
        if status != 404 or not upcoming or deadline is None or deadline <= now:
            return {"available": False, "reason": describe_picks_error(exc), "source": "unavailable"}

        completed = [row for row in events if row.get("finished") and int(row.get("id", 0)) < int(target_gw)]
        if not completed:
            return {"available": False, "reason": "Upcoming picks are not public yet, and no completed Gameweek is available as a squad baseline.", "source": "unavailable"}
        baseline_gw = max(int(row["id"]) for row in completed)
        try:
            baseline_payload = fetch_picks(baseline_gw)
        except Exception as baseline_error:
            return {"available": False, "reason": f"Upcoming GW{target_gw} picks are not public. The last completed squad (GW{baseline_gw}) could not be loaded: {describe_picks_error(baseline_error)}", "source": "unavailable"}
        baseline_picks = baseline_payload.get("picks", []) if isinstance(baseline_payload, dict) else []
        valid, reason = validate_public_picks(baseline_picks, known_player_ids)
        if not valid:
            return {"available": False, "reason": f"Upcoming GW{target_gw} picks are not public. Prior GW{baseline_gw} picks are invalid: {reason}", "source": "unavailable"}
        baseline_ids = [int(row["element"]) for row in sorted(baseline_picks, key=lambda item: int(item["position"]))]
        if transfer_history_available:
            roster, transactions, reconstructed, transfer_note = apply_public_transfers(
                baseline_ids, transfer_rows, baseline_gw, target_gw, known_player_ids
            )
        else:
            roster, transactions, reconstructed = baseline_ids, [], False
            transfer_note = "Public transfer history could not be loaded; it cannot confirm that no moves occurred."
        by_id = {int(row["element"]): dict(row) for row in baseline_picks}
        for position, player_id in enumerate(roster, 1):
            by_id.setdefault(player_id, {"element": player_id})["position"] = position
        picks = [by_id[player_id] for player_id in roster]
        message = (f"Last published squad: GW{baseline_gw}. GW{target_gw} picks are not public yet. "
                   f"{len(transactions)} public transfer(s) applied. Confirm this roster includes all transfers and chip changes before using recommendations." if reconstructed else
                   f"Last published squad: GW{baseline_gw}. GW{target_gw} picks are not public yet. {transfer_note} "
                   "Confirm or correct all transfers and chip changes before using recommendations.")
        return {
            "available": True, "player_ids": roster, "picks": picks, "source": "prior", "baseline_gw": baseline_gw,
            "transactions": transactions, "transfer_history_reconstructed": reconstructed,
            "needs_confirmation": True, "target_verified": False, "reason": message,
        }

    picks = payload.get("picks", []) if isinstance(payload, dict) else []
    valid, reason = validate_public_picks(picks, known_player_ids)
    if not valid:
        return {"available": False, "reason": f"Current GW{target_gw} picks are invalid: {reason} No example or saved squad is substituted.", "source": "invalid"}
    ordered = sorted(picks, key=lambda row: int(row["position"]))
    return {
        "available": True, "player_ids": [int(row["element"]) for row in ordered], "picks": ordered,
        "source": "current", "baseline_gw": int(target_gw), "transactions": [],
        "transfer_history_reconstructed": True, "needs_confirmation": False,
        "target_verified": True, "reason": f"Current squad loaded from verified public picks for GW{target_gw}.",
    }
