"""Shared FPL projection core used by the dashboard and scheduled forecast worker.

When a versioned historical model artifact is available, future Gameweek points are
predicted by that trained model. A transparent heuristic remains as a safe fallback.
"""
from datetime import datetime, timedelta, timezone
import math

import fpl_learning

# Set by the dashboard (remote artifact) or worker (persistent fpl-data branch).
TRAINED_MODEL = None

def make_history_signal(summary, cutoff_utc=None):
    """Build lagged signals from completed Gameweeks, matching training aggregation.

    API history can contain one record per fixture, so double Gameweeks must first
    be summed into one observation. If a simulated deadline is supplied, only
    events whose latest fixture is at least 72 hours old are eligible. A missing
    kickoff timestamp makes that event unusable for this point-in-time feature set.
    """
    from collections import defaultdict

    history = summary.get("history", []) if isinstance(summary, dict) else []
    grouped = defaultdict(list)
    cutoff = None
    if cutoff_utc is not None:
        cutoff = cutoff_utc if isinstance(cutoff_utc, datetime) else datetime.fromisoformat(str(cutoff_utc).replace("Z", "+00:00"))
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        cutoff = cutoff.astimezone(timezone.utc)
    for row in history:
        if isinstance(row, dict) and row.get("round") is not None:
            try:
                grouped[int(row["round"])].append(row)
            except (TypeError, ValueError):
                continue
    gameweeks = []
    for gameweek, rows in sorted(grouped.items()):
        if cutoff is not None:
            try:
                kickoffs = [datetime.fromisoformat(str(row["kickoff_time"]).replace("Z", "+00:00")) for row in rows]
                if any(value.tzinfo is None for value in kickoffs):
                    continue
                available_at = max(kickoffs).astimezone(timezone.utc) + timedelta(hours=72)
                if available_at >= cutoff:
                    continue
            except (KeyError, TypeError, ValueError):
                continue
        fixture_ids = {str(row.get("fixture")) for row in rows if row.get("fixture") is not None}
        fixture_count = max(1, len(fixture_ids))
        points = sum(float(row.get("total_points") or 0) for row in rows)
        minutes = sum(float(row.get("minutes") or 0) for row in rows)
        xgi = sum(float(row.get("expected_goals") or 0) + float(row.get("expected_assists") or 0) for row in rows)
        gameweeks.append({"round": gameweek, "points": points, "minutes": minutes,
                          "starts": max([float(row.get("starts") or 0) for row in rows] or [0]),
                          "xgi": xgi, "fixture_count": fixture_count})
    recent = gameweeks[-5:]
    played = [row for row in recent if row["minutes"] > 0]
    points_played = [row["points"] for row in played]
    recent_min_sum = sum(row["minutes"] for row in recent)
    all_min_sum = sum(row["minutes"] for row in gameweeks)
    recent_fixture_count = sum(row["fixture_count"] for row in recent)
    recent_minutes_per_gw = [row["minutes"] / row["fixture_count"] for row in recent]
    all_points = [row["points"] for row in gameweeks]
    all_xgi = [row["xgi"] for row in gameweeks]
    recent_xgi = [row["xgi"] for row in recent]
    return {
        "sample_matches": recent_fixture_count,
        "played_matches": len(played),
        "recent_points_per_appearance": (sum(points_played) / len(points_played)) if points_played else None,
        "recent_points_per_gw": (sum(row["points"] for row in recent) / len(recent)) if recent else None,
        "season_points_per_gw": (sum(all_points) / len(all_points)) if all_points else None,
        "recent_minutes_per_fixture": (sum(recent_minutes_per_gw) / len(recent_minutes_per_gw)) if recent_minutes_per_gw else None,
        "recent_appearance_rate": (len(played) / len(recent)) if recent else None,
        "recent_start_rate": (sum(1 for row in recent if row["starts"] > 0) / len(recent)) if recent else None,
        "season_xgi_per90": (sum(all_xgi) / max(90.0, all_min_sum) * 90.0) if all_xgi else None,
        "recent_xgi_per90": (sum(recent_xgi) / max(90.0, recent_min_sum) * 90.0) if recent_xgi else None,
        "last_history_gw": int(recent[-1]["round"]) if recent else None,
        "history_total_gws": len(gameweeks),
    }


def _availability_override(p, weeks_ahead=0):
    """Apply current status only to the nearest GW, not unchanged through a five-GW horizon.

    Historical minutes already learn typical absence/rotation risk. Without a separate
    injury-return model, don't assume today's status flag persists into later Gameweeks.
    """
    if int(weeks_ahead) > 0:
        return 1.0
    status = str(p.get("status") or "").lower()
    chance = p.get("chance_of_playing_next_round")
    if status in {"i", "s", "u"}:
        return 0.20
    if status == "d":
        if chance is not None:
            try:
                return max(0.20, min(1.0, float(chance) / 100.0))
            except (TypeError, ValueError):
                pass
        return 0.55
    if chance is not None:
        try:
            return max(0.20, min(1.0, float(chance) / 100.0))
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def _upcoming_fixtures(p, fixtures, horizon=5, start_gw=None):
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
    return upcoming


def _estimate_points_trained(p, fixtures, hist, horizon=5, start_gw=None):
    """Score each future Gameweek with the historical model artifact."""
    upcoming = _upcoming_fixtures(p, fixtures, horizon, start_gw)
    if not upcoming or not TRAINED_MODEL:
        return None
    from collections import defaultdict
    by_event = defaultdict(list)
    for fixture in upcoming:
        by_event[int(fixture["event"])].append(fixture)
    total, diffs, total_fixtures = 0.0, [], 0
    first_event = min(by_event) if by_event else None
    position = p.get("element_type")
    minutes = float(p.get("minutes") or 0)
    fallback_xgi90 = (float(p.get("expected_goals") or 0) + float(p.get("expected_assists") or 0)) / max(90.0, minutes) * 90.0
    for event_id, event_fixtures in sorted(by_event.items()):
        fixture_count = len(event_fixtures)
        home_count = sum(1 for f in event_fixtures if f.get("team_h") == p.get("team"))
        home_share = home_count / max(1, fixture_count)
        feature_row = fpl_learning.make_feature_row(
            position=position, history_signal=hist,
            fixture_count=fixture_count, home_share=home_share,
            fallback_season_mean=float(p.get("points_per_game") or 0),
            fallback_recent_mean=float(p.get("form") or 0),
            fallback_xgi_per90=fallback_xgi90,
        )
        predicted = fpl_learning.predict_features(TRAINED_MODEL, feature_row)
        if predicted is None:
            return None
        per_fixture_factors = []
        for f in event_fixtures:
            home = f.get("team_h") == p.get("team")
            diff = float(f.get("team_h_difficulty" if home else "team_a_difficulty") or 3)
            diffs.append(diff)
            per_fixture_factors.append(max(0.82, min(1.16, 1.0 + (3.0 - diff) * 0.06)))
        # Historical archives do not contain deadline-time fixture difficulty or
        # schedule snapshots. Model output is a single-fixture expectation; apply
        # only the live published fixture count here. Fixture difficulty remains a
        # diagnostic and cannot silently change a proxy-validated model score.
        # The trained model learns single-fixture target rows; doubles are scaled
        # using the schedule actually published for the live target Gameweek.
        # Explicit availability is a live-only override for the nearest GW.
        horizon_start = int(start_gw) if start_gw is not None else int(first_event or event_id)
        total += predicted * fixture_count * _availability_override(p, weeks_ahead=(event_id - horizon_start))
        total_fixtures += fixture_count
    if total_fixtures == 0:
        return 0.0, 3.0, 0
    return round(max(0.0, total), 1), round(sum(diffs) / len(diffs), 2), total_fixtures

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


def production_fallback_per_fixture(p, hist=None):
    """Exact v3.0 heuristic projection for one neutral-difficulty fixture."""
    def num(key):
        try:
            return float(p.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    hist = hist or {}
    ppg, form = num("points_per_game"), num("form")
    minutes, starts = num("minutes"), int(num("starts"))
    pos = p.get("element_type")
    baseline = {1: 3.2, 2: 3.5, 3: 3.8, 4: 4.0}.get(pos, 3.6)
    xg90 = num("expected_goals") / max(minutes, 90) * 90
    xa90 = num("expected_assists") / max(minutes, 90) * 90
    attacking_signal = min(1.5, (xg90 + xa90) * 2.0)
    raw = 0.50 * ppg + 0.25 * form + 0.25 * (baseline + attacking_signal)
    recent_pts = hist.get("recent_points_per_appearance")
    played_count = int(hist.get("played_matches") or 0)
    if recent_pts is not None and played_count > 0:
        recent_weight = min(0.30, 0.06 * played_count)
        raw = (1.0 - recent_weight) * raw + recent_weight * float(recent_pts)
    sample_weight = min(0.85, minutes / 900.0)
    per_match = sample_weight * raw + (1.0 - sample_weight) * baseline
    expected_minutes, availability_factor = _estimate_minutes_parts(p, hist)
    minutes_factor = max(0.35, min(1.0, expected_minutes / 80.0))
    return per_match * minutes_factor * availability_factor


def estimate_points(p, team, fixtures, horizon=5, start_gw=None, use_history=True, history_signals=None, calibration_offsets=None):
    """Use the trained historical model when available; retain a transparent fallback."""
    hist_for_model = (history_signals or {}).get(int(p.get("id", -1)), {}) if use_history else {}
    if use_history and TRAINED_MODEL:
        trained = _estimate_points_trained(p, fixtures, hist_for_model, horizon=horizon, start_gw=start_gw)
        if trained is not None:
            return trained

    hist = hist_for_model
    pos = p.get("element_type")
    per_match = production_fallback_per_fixture(p, hist)

    upcoming = _upcoming_fixtures(p, fixtures, horizon, start_gw)
    scores, diffs = [], []
    for f in upcoming:
        home = f.get("team_h") == p.get("team")
        diff = float(f.get("team_h_difficulty" if home else "team_a_difficulty") or 3)
        diffs.append(diff)
        fixture_factor = max(0.82, min(1.16, 1.0 + (3.0 - diff) * 0.06))
        scores.append(per_match * fixture_factor)
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

