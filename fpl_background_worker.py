"""Scheduled FPL forecast snapshot and backtest updater for v2.9.

Snapshots the full player pool, stores a same-deadline FPL ep_next benchmark,
records history coverage, and reconciles both single-GW and five-GW outcomes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import fpl_engine

BASE = "https://fantasy.premierleague.com/api"
MODEL_VERSION = "2.9"
COLUMNS = [
    "season", "gameweek", "player_id", "player", "position", "club",
    "model_version", "forecast_created_utc",
    "projection_model", "projection_baseline", "projection_ep_next",
    "projection_model_5gw", "projection_baseline_5gw", "forecast_horizon_gws",
    "history_enriched", "history_sample_matches", "projection_expected_minutes", "xi_eligible", "model_captain_pick",
    "actual_points", "actual_minutes", "actual_points_5gw", "actual_minutes_5gw",
]
HEADERS = {"User-Agent": "Mozilla/5.0 The-One-Above-All-FPL/2.9"}


def get_json(path: str, attempts: int = 3):
    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(f"{BASE}/{path}", timeout=25, headers=HEADERS)
            if response.status_code == 429 or response.status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.8 * (2 ** attempt))
    raise RuntimeError(f"FPL request failed for {path}: {last_error}")


def season_label(now=None) -> str:
    now = now or datetime.now(timezone.utc)
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def position_short(player, element_types) -> str:
    return next((x.get("singular_name_short", "?") for x in element_types if x.get("id") == player.get("element_type")), "?")


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        print(f"Existing forecast CSV could not be read; starting empty: {exc}")
        frame = pd.DataFrame(columns=COLUMNS)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    if "model_version" in frame.columns:
        versions = frame["model_version"].astype("string").str.strip()
        frame.loc[versions.isna() | versions.eq(""), "model_version"] = "legacy-unknown"
    return frame[COLUMNS]


def write_log(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame[COLUMNS].to_csv(temp, index=False)
    os.replace(temp, path)


def fetch_one_history(player_id: int):
    last_error = None
    for attempt in range(3):
        try:
            payload = get_json(f"element-summary/{int(player_id)}/", attempts=1)
            return int(player_id), fpl_engine.make_history_signal(payload), None
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
    return int(player_id), None, str(last_error)


def fetch_history_signals(player_ids):
    signals, failures = {}, {}
    unique_ids = sorted({int(pid) for pid in player_ids if pid is not None})
    if not unique_ids:
        return signals, failures
    # All-player coverage prevents the forecast pool receiving hidden mixtures of inputs.
    # Bounded concurrency and retries reduce burst pressure on the public endpoint.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one_history, pid) for pid in unique_ids]
        for future in as_completed(futures):
            try:
                pid, signal, error = future.result()
                if signal is not None:
                    signals[pid] = signal
                else:
                    failures[pid] = error or "unknown error"
            except Exception as exc:
                print(f"Unexpected player-history task failure: {exc}")
    return signals, failures


def get_public_picks_ids(team_id: int, gameweek: int, player_ids: set[int]) -> set[int]:
    try:
        payload = get_json(f"entry/{team_id}/event/{gameweek}/picks/")
        picks = payload.get("picks", []) if isinstance(payload, dict) else []
        ids = {int(row["element"]) for row in picks if row.get("element") is not None}
        return ids & player_ids if len(ids) == 15 else set()
    except Exception:
        return set()


def choose_legal_xi(squad_ids: set[int], player_by_id: dict[int, dict], projection_by_id: dict[int, float]):
    """Return (XI ids, captain id) for the best legal formation from a public 15-player squad."""
    from itertools import combinations

    ids = sorted(int(pid) for pid in squad_ids if int(pid) in player_by_id)
    if len(ids) != 15:
        return set(), None
    best_score = float("-inf")
    best_xi = []
    for combo in combinations(ids, 11):
        roles = [int(player_by_id[pid].get("element_type") or 0) for pid in combo]
        if roles.count(1) != 1:
            continue
        defenders, midfielders, forwards = roles.count(2), roles.count(3), roles.count(4)
        if not (3 <= defenders <= 5 and 2 <= midfielders <= 5 and 1 <= forwards <= 3):
            continue
        captain_id = max(combo, key=lambda pid: (float(projection_by_id.get(pid, 0.0)), -pid))
        score = sum(float(projection_by_id.get(pid, 0.0)) for pid in combo) + float(projection_by_id.get(captain_id, 0.0))
        if score > best_score:
            best_score, best_xi = score, list(combo)
    if not best_xi:
        return set(), None
    captain_id = max(best_xi, key=lambda pid: (float(projection_by_id.get(pid, 0.0)), -pid))
    return set(best_xi), int(captain_id)


def actual_map_for_gw(gameweek: int) -> dict[int, tuple[float, float]]:
    live = get_json(f"event/{int(gameweek)}/live/")
    result = {}
    for element in live.get("elements", []):
        stats = element.get("stats", {})
        try:
            result[int(element["id"])] = (
                float(stats.get("total_points") or 0),
                float(stats.get("minutes") or 0),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="forecast_log.csv")
    args = parser.parse_args()
    log_path = Path(args.csv_path)
    team_id = int(os.environ.get("FPL_TEAM_ID", "2311669"))
    changed = False
    log = read_log(log_path)

    bootstrap = get_json("bootstrap-static/")
    fixtures = get_json("fixtures/")
    players = [p for p in bootstrap.get("elements", []) if p.get("id") is not None]
    teams = {int(team["id"]): team for team in bootstrap.get("teams", []) if team.get("id") is not None}
    types = bootstrap.get("element_types", [])
    events = bootstrap.get("events", [])
    event_by_id = {int(e["id"]): e for e in events if e.get("id") is not None}
    next_event = next((event for event in events if event.get("is_next")), None)
    current_event = next((event for event in events if event.get("is_current")), None)
    target = next_event or (None if current_event and current_event.get("finished") else current_event)
    season = season_label()

    if target and target.get("id") is not None and target.get("deadline_time"):
        gameweek = int(target["id"])
        deadline = datetime.fromisoformat(str(target["deadline_time"]).replace("Z", "+00:00"))
        hours_to_deadline = (deadline - datetime.now(timezone.utc)).total_seconds() / 3600.0
        already_saved = (
            not log.empty
            and ((log["season"].astype(str) == season)
                 & (pd.to_numeric(log["gameweek"], errors="coerce") == gameweek)
                 & (log["model_version"].astype(str) == MODEL_VERSION)).any()
        )
        if 0 < hours_to_deadline <= 6 and not already_saved:
            # No online correction is applied during this validation release. Freeze the
            # formula while the benchmarks and rankings are accumulated.
            player_by_id = {int(player["id"]): player for player in players}
            history_ids = set(player_by_id)
            signals, history_failures = fetch_history_signals(history_ids)
            coverage = len(signals) / max(1, len(history_ids))
            print(f"History coverage for {season} GW{gameweek}: {len(signals)}/{len(history_ids)} ({coverage:.1%}); failures={len(history_failures)}")
            if history_failures:
                print("Some players have no recent-history signal; their rows are explicitly marked unenriched.")
            created = datetime.now(timezone.utc).isoformat(timespec="seconds")
            season_max_gw = max(event_by_id) if event_by_id else gameweek
            horizon_end_gw = min(gameweek + 4, season_max_gw)
            horizon_len = max(1, horizon_end_gw - gameweek + 1)
            rows = []
            for player in players:
                try:
                    player_id = int(player["id"])
                    team_code = teams.get(int(player.get("team") or -1), {}).get("short_name", "?")
                    history = signals.get(player_id, {})
                    projection_model, _, _ = fpl_engine.estimate_points(
                        player, team_code, fixtures, 1, start_gw=gameweek,
                        use_history=True, history_signals=signals, calibration_offsets={}
                    )
                    projection_baseline, _, _ = fpl_engine.estimate_points(
                        player, team_code, fixtures, 1, start_gw=gameweek,
                        use_history=False, history_signals={}, calibration_offsets={}
                    )
                    projection_model_5gw, _, _ = fpl_engine.estimate_points(
                        player, team_code, fixtures, horizon_len, start_gw=gameweek,
                        use_history=True, history_signals=signals, calibration_offsets={}
                    )
                    projection_baseline_5gw, _, _ = fpl_engine.estimate_points(
                        player, team_code, fixtures, horizon_len, start_gw=gameweek,
                        use_history=False, history_signals={}, calibration_offsets={}
                    )
                    ep_next = player.get("ep_next")
                    try:
                        ep_next = float(ep_next) if ep_next is not None and str(ep_next).strip() != "" else None
                    except (TypeError, ValueError):
                        ep_next = None
                    rows.append({
                        "season": season, "gameweek": gameweek, "player_id": player_id,
                        "player": player.get("web_name", ""), "position": position_short(player, types),
                        "club": team_code, "model_version": MODEL_VERSION, "forecast_created_utc": created,
                        "projection_model": projection_model, "projection_baseline": projection_baseline,
                        "projection_ep_next": ep_next,
                        "projection_model_5gw": projection_model_5gw,
                        "projection_baseline_5gw": projection_baseline_5gw,
                        "forecast_horizon_gws": horizon_len,
                        "history_enriched": player_id in signals,
                        "history_sample_matches": int(history.get("sample_matches") or 0) if history else None,
                        "projection_expected_minutes": fpl_engine.estimate_expected_minutes(
                            player, history, fpl_engine.team_fixture_count(fixtures, player.get("team"), gameweek)
                        ),
                        "xi_eligible": None, "model_captain_pick": None,
                        "actual_points": None, "actual_minutes": None,
                        "actual_points_5gw": None, "actual_minutes_5gw": None,
                    })
                except Exception as exc:
                    print(f"Could not forecast player {player.get('id')}: {exc}")
            if rows:
                # Captain-regret diagnostics are only recorded if the public picks endpoint
                # exposes a complete 15-player squad before the deadline. If not, leave the
                # fields empty rather than inventing a lineup from post-deadline data.
                public_squad_ids = get_public_picks_ids(team_id, gameweek, set(player_by_id))
                if len(public_squad_ids) == 15:
                    projection_by_id = {int(row["player_id"]): float(row["projection_model"]) for row in rows}
                    xi_ids, captain_id = choose_legal_xi(public_squad_ids, player_by_id, projection_by_id)
                    if len(xi_ids) == 11 and captain_id is not None:
                        for row in rows:
                            player_id = int(row["player_id"])
                            row["xi_eligible"] = player_id in xi_ids
                            row["model_captain_pick"] = player_id == captain_id
                        print(f"Captain-regret tracking enabled for {season} GW{gameweek}: legal model XI and captain captured.")
                    else:
                        print("Public squad was available but no legal XI was found; captain fields remain empty.")
                else:
                    print("Complete public squad unavailable pre-deadline; captain-regret fields remain empty for this snapshot.")
                new = pd.DataFrame(rows, columns=COLUMNS)
                log = pd.concat([log, new], ignore_index=True)
                changed = True
                print(f"Saved v{MODEL_VERSION} pre-deadline forecast for {season} GW{gameweek}: {len(rows)} players; ep_next coverage={new['projection_ep_next'].notna().mean():.1%}; history coverage={new['history_enriched'].map(bool).mean():.1%}.")
        elif already_saved:
            print(f"Forecast already exists for {season} GW{gameweek}; preserving the first snapshot.")
        elif hours_to_deadline > 6:
            print(f"GW{gameweek} is {hours_to_deadline:.1f} hours from deadline; waiting for the six-hour decision window.")
        else:
            print(f"Deadline for GW{gameweek} has passed; will not create a retrospective forecast.")
    else:
        print("No upcoming event with a published deadline; skipped new snapshot.")

    finished_gws = {int(event["id"]) for event in events if event.get("finished") and event.get("id") is not None}
    max_gw = max(event_by_id) if event_by_id else 38
    if not log.empty:
        # Reconcile single-GW actuals as soon as FPL says the event is finished.
        numeric_gws = pd.to_numeric(log["gameweek"], errors="coerce")
        for gameweek in sorted(numeric_gws.dropna().astype(int).unique()):
            mask = numeric_gws == gameweek
            if gameweek not in finished_gws or not log.loc[mask, "actual_points"].isna().any():
                continue
            try:
                actuals = actual_map_for_gw(gameweek)
                for idx in log.index[mask]:
                    try:
                        player_id = int(log.at[idx, "player_id"])
                    except (TypeError, ValueError):
                        continue
                    if player_id in actuals:
                        log.at[idx, "actual_points"] = actuals[player_id][0]
                        log.at[idx, "actual_minutes"] = actuals[player_id][1]
                        changed = True
                print(f"Reconciled one-GW actuals for GW{gameweek}.")
            except Exception as exc:
                print(f"Could not fetch actuals for GW{gameweek}: {exc}")

        # Reconcile the five-GW horizon only once every event in that horizon is finished.
        all_gws = sorted(numeric_gws.dropna().astype(int).unique())
        for gameweek in all_gws:
            mask = numeric_gws == gameweek
            if not log.loc[mask, "actual_points_5gw"].isna().any():
                continue
            horizon_end = min(int(gameweek) + 4, max_gw)
            horizon_events = set(range(int(gameweek), horizon_end + 1))
            if not horizon_events.issubset(finished_gws):
                continue
            totals: dict[int, list[float]] = {}
            all_actuals_available = True
            for hgw in sorted(horizon_events):
                try:
                    actuals = actual_map_for_gw(hgw)
                except Exception as exc:
                    print(f"Could not fetch actuals for GW{hgw} for horizon starting GW{gameweek}: {exc}")
                    all_actuals_available = False
                    break
                for pid, (points, minutes) in actuals.items():
                    totals.setdefault(pid, [0.0, 0.0])
                    totals[pid][0] += points
                    totals[pid][1] += minutes
            if not all_actuals_available:
                continue
            for idx in log.index[mask]:
                try:
                    pid = int(log.at[idx, "player_id"])
                except (TypeError, ValueError):
                    continue
                actual_points, actual_minutes = totals.get(pid, [0.0, 0.0])
                log.at[idx, "actual_points_5gw"] = actual_points
                log.at[idx, "actual_minutes_5gw"] = actual_minutes
            changed = True
            print(f"Reconciled {len(horizon_events)}-GW actuals for forecast starting GW{gameweek}.")

    if not log.empty:
        log["gameweek"] = pd.to_numeric(log["gameweek"], errors="coerce")
        log["player_id"] = pd.to_numeric(log["player_id"], errors="coerce")
        # Keep the first forecast per model version and GW; preserve actuals on upgrades.
        log = log.sort_values("forecast_created_utc").drop_duplicates(
            subset=["season", "gameweek", "player_id", "model_version"], keep="first"
        )
        log = log[COLUMNS]
    if changed or not path_exists_with_columns(log_path, COLUMNS):
        write_log(log_path, log)
        print(f"Updated persistent forecast log at {log_path} ({len(log)} rows, schema v2.9).")
    else:
        print(f"No forecast-log changes required. Existing log has {len(log)} rows.")
    return 0


def path_exists_with_columns(path: Path, columns: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        return list(pd.read_csv(path, nrows=0).columns) == columns
    except Exception:
        return False


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FPL scheduled worker failed: {exc}", file=sys.stderr)
        raise
