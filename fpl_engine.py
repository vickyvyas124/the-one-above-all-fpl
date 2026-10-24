"""Shared FPL projection core used by the dashboard and scheduled forecast worker.

This is a transparent heuristic model, not a trained or validated machine-learning model.
"""
from datetime import datetime, timezone
import math

def make_history_signal(summary):
    history = summary.get("history", []) if isinstance(summary, dict) else []
    history = sorted([r for r in history if isinstance(r, dict)], key=lambda r: int(r.get("round") or 0))
    recent = history[-5:]
    played = [r for r in recent if float(r.get("minutes") or 0) > 0]
    points = [float(r.get("total_points") or 0) for r in played]
    minutes_all = [float(r.get("minutes") or 0) for r in recent]
    starts = [1 if float(r.get("starts") or 0) > 0 else 0 for r in recent]
    return {
        "sample_matches": len(recent),
        "played_matches": len(played),
        "recent_points_per_appearance": (sum(points) / len(points)) if points else None,
        "recent_minutes_per_fixture": (sum(minutes_all) / len(minutes_all)) if minutes_all else None,
        "recent_appearance_rate": (len(played) / len(recent)) if recent else None,
        "recent_start_rate": (sum(starts) / len(starts)) if starts else None,
        "last_history_gw": int(recent[-1].get("round") or 0) if recent else None,
    }

def _estimate_minutes_parts(p, history_signal=None):
    """Return expected minutes before availability risk and a separate availability factor."""
    def num(key):
        try:
            return float(p.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    minutes, starts = num("minutes"), int(num("starts"))
    if minutes <= 0:
        expected_minutes = 35.0
    elif starts > 0:
        average_start_minutes = minutes / starts
        starts_subs = starts + num("substitute_appearances")
        start_rate = min(1.0, starts / max(1.0, starts_subs))
        expected_minutes = min(90.0, average_start_minutes) * (0.65 + 0.35 * start_rate)
    else:
        expected_minutes = min(60.0, minutes / max(1.0, num("appearances")))

    hist = history_signal or {}
    recent_minutes = hist.get("recent_minutes_per_fixture")
    recent_sample = int(hist.get("sample_matches") or 0)
    if recent_minutes is not None and recent_sample >= 3:
        expected_minutes = 0.60 * expected_minutes + 0.40 * float(recent_minutes)
    expected_minutes = max(0.0, min(90.0, expected_minutes))

    status = str(p.get("status") or "").lower()
    chance = p.get("chance_of_playing_next_round")
    if status in {"i", "s", "u"}:
        availability_factor = 0.20
    elif status == "d":
        availability_factor = 0.55
    elif chance is not None:
        try:
            availability_factor = max(0.20, min(1.0, float(chance) / 100.0))
        except (TypeError, ValueError):
            availability_factor = 0.90
    else:
        availability_factor = 0.90
    return expected_minutes, availability_factor


def team_fixture_count(fixtures, team_id, gameweek):
    """Count the team's scheduled fixtures in one Gameweek; a blank returns zero."""
    if team_id is None or gameweek is None:
        return 0
    count = 0
    try:
        team_id, gameweek = int(team_id), int(gameweek)
    except (TypeError, ValueError):
        return 0
    for fixture in fixtures or []:
        try:
            if int(fixture.get("event") or 0) != gameweek:
                continue
            if fixture.get("team_h") != team_id and fixture.get("team_a") != team_id:
                continue
            if not fixture.get("finished"):
                count += 1
        except (TypeError, ValueError, AttributeError):
            continue
    return count


def estimate_expected_minutes(p, history_signal=None, fixture_count=1):
    """Expected total minutes in a Gameweek, including availability and fixture count.

    A blank Gameweek is zero; a Double Gameweek can exceed 90 expected minutes.
    This aligns the minutes diagnostic with actual minutes summed over the event.
    """
    try:
        fixtures_n = max(0, min(3, int(fixture_count)))
    except (TypeError, ValueError):
        fixtures_n = 1
    if fixtures_n == 0:
        return 0.0
    expected_minutes, availability_factor = _estimate_minutes_parts(p, history_signal)
    return round(expected_minutes * availability_factor * fixtures_n, 1)


def estimate_points(p, team, fixtures, horizon=5, start_gw=None, use_history=True, history_signals=None, calibration_offsets=None):
    """Transparent fixture-level projection; recent player match history is a capped signal."""
    def num(key):
        try:
            return float(p.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    ppg, form = num("points_per_game"), num("form")
    minutes, starts = num("minutes"), int(num("starts"))
    status = str(p.get("status") or "").lower()
    chance = p.get("chance_of_playing_next_round")
    pos = p.get("element_type")
    baseline = {1: 3.2, 2: 3.5, 3: 3.8, 4: 4.0}.get(pos, 3.6)

    # Season xG/xA are cumulative; normalize to a conservative per-90 attacking signal.
    xg90 = num("expected_goals") / max(minutes, 90) * 90
    xa90 = num("expected_assists") / max(minutes, 90) * 90
    attacking_signal = min(1.5, (xg90 + xa90) * 2.0)
    raw = 0.50 * ppg + 0.25 * form + 0.25 * (baseline + attacking_signal)

    hist = (history_signals or {}).get(int(p.get("id", -1)), {}) if use_history else {}
    recent_pts = hist.get("recent_points_per_appearance")
    played_count = int(hist.get("played_matches") or 0)
    if recent_pts is not None and played_count > 0:
        # Limit recency influence to 30% so a short hot/cold streak cannot dominate season skill.
        recent_weight = min(0.30, 0.06 * played_count)
        raw = (1.0 - recent_weight) * raw + recent_weight * float(recent_pts)

    sample_weight = min(0.85, minutes / 900.0)
    per_match = sample_weight * raw + (1.0 - sample_weight) * baseline

    expected_minutes, availability_factor = _estimate_minutes_parts(p, hist)
    minutes_factor = max(0.35, min(1.0, expected_minutes / 80.0))

    upcoming = [f for f in fixtures if not f.get("finished")
                and f.get("event") is not None
                and (f.get("team_h") == p.get("team") or f.get("team_a") == p.get("team"))]
    if start_gw is not None:
        upcoming = [f for f in upcoming
                    if int(start_gw) <= int(f["event"]) < int(start_gw) + int(horizon)]
    else:
        event_ids = sorted({int(f["event"]) for f in upcoming})
        target_ids = set(event_ids[:int(horizon)])
        upcoming = [f for f in upcoming if int(f["event"]) in target_ids]
    upcoming.sort(key=lambda f: (f.get("event") or 99, f.get("kickoff_time") or ""))
    scores, diffs = [], []
    for f in upcoming:
        home = f.get("team_h") == p.get("team")
        diff = float(f.get("team_h_difficulty" if home else "team_a_difficulty") or 3)
        diffs.append(diff)
        fixture_factor = max(0.82, min(1.16, 1.0 + (3.0 - diff) * 0.06))
        scores.append(per_match * minutes_factor * availability_factor * fixture_factor)
    if not scores:
        return 0.0, 3.0, 0
    total = sum(scores)
    # Position-level bias correction is deliberately small and only enabled after
    # enough prior finished Gameweeks are available. It is learned from prior
    # forecasts only; the baseline forecast remains uncalibrated.
    if use_history and calibration_offsets:
        try:
            offset = float(calibration_offsets.get(str(pos), 0.0))
            if math.isfinite(offset):
                total = max(0.0, total + offset * len(scores))
        except (TypeError, ValueError):
            pass
    return round(total, 1), round(sum(diffs) / len(diffs), 2), len(scores)


def derive_calibration(forecast_rows, target_gameweek, min_completed_gws=4):
    """Learn capped position-level bias corrections from prior v2.9 snapshots.

    Input may be a list of dictionaries or a pandas DataFrame (iterable of row dicts
    is preferred). Returns (offsets, metadata). No calibration occurs before four
    distinct earlier Gameweeks and minimum per-position sample sizes.
    """
    rows = []
    try:
        source = forecast_rows.to_dict("records") if hasattr(forecast_rows, "to_dict") else list(forecast_rows or [])
    except Exception:
        source = []
    for row in source:
        try:
            gw = int(row.get("gameweek"))
            if gw >= int(target_gameweek):
                continue
            if str(row.get("model_version", "")) != "2.9":
                continue
            actual = float(row.get("actual_points"))
            predicted = float(row.get("projection_model"))
            if not (math.isfinite(actual) and math.isfinite(predicted)):
                continue
            position = str(row.get("position", ""))
            if position not in {"GKP", "DEF", "MID", "FWD"}:
                continue
            rows.append({"gameweek": gw, "position": position, "residual": actual - predicted})
        except (TypeError, ValueError, AttributeError):
            continue
    completed_gws = sorted({row["gameweek"] for row in rows})
    metadata = {"enabled": False, "completed_gws": len(completed_gws), "samples_by_position": {}, "offsets": {}}
    if len(completed_gws) < min_completed_gws:
        return {}, metadata
    offsets = {}
    for position in ("GKP", "DEF", "MID", "FWD"):
        position_rows = [row for row in rows if row["position"] == position]
        count = len(position_rows)
        metadata["samples_by_position"][position] = count
        minimum = 35 if position == "GKP" else 60
        if count < minimum:
            continue
        mean_residual = sum(row["residual"] for row in position_rows) / count
        shrinkage = count / (count + 150.0)
        offset = max(-0.60, min(0.60, mean_residual * shrinkage))
        offsets[position] = round(offset, 3)
    metadata["offsets"] = dict(offsets)
    metadata["enabled"] = bool(offsets)
    return offsets, metadata

