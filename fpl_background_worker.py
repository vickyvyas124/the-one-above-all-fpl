"""Scheduled FPL forecast snapshot and backtest updater.

Run by GitHub Actions every 15 minutes. Stores forecast_log.csv on a dedicated
branch so the Streamlit app can redeploy independently without losing history.
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import fpl_engine

BASE = "https://fantasy.premierleague.com/api"
COLUMNS = [
    "season", "gameweek", "player_id", "player", "position", "club",
    "projection_model", "projection_baseline", "forecast_created_utc",
    "actual_points", "actual_minutes", "model_version",
]
HEADERS = {"User-Agent": "Mozilla/5.0 The-One-Above-All-FPL/2.8"}


def get_json(path: str):
    response = requests.get(f"{BASE}/{path}", timeout=20, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def season_label(now=None) -> str:
    now = now or datetime.now(timezone.utc)
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def norm(value) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum())


def position_short(player, element_types) -> str:
    return next((x.get("singular_name_short", "?") for x in element_types if x.get("id") == player.get("element_type")), "?")


def history_priority(player) -> float:
    def number(key):
        try:
            return float(player.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0
    minutes = number("minutes")
    xgi90 = (number("expected_goals") + number("expected_assists")) / max(90.0, minutes) * 90.0
    return 0.45 * number("points_per_game") + 0.20 * number("form") + 0.50 * min(1.0, minutes / 900.0) + 0.20 * min(1.5, xgi90)


def read_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    try:
        frame = pd.read_csv(path)
    except Exception:
        frame = pd.DataFrame(columns=COLUMNS)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame[COLUMNS]


def write_log(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame[COLUMNS].to_csv(temp, index=False)
    os.replace(temp, path)


def fetch_history_signals(player_ids):
    signals = {}
    player_ids = sorted({int(pid) for pid in player_ids if pid is not None})
    if not player_ids:
        return signals
    def fetch_one(player_id):
        payload = get_json(f"element-summary/{player_id}/")
        return player_id, fpl_engine.make_history_signal(payload)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one, pid) for pid in player_ids]
        for future in as_completed(futures):
            try:
                player_id, signal = future.result()
                signals[player_id] = signal
            except Exception as exc:
                print(f"History unavailable for one player: {exc}")
    return signals


def get_public_picks_ids(team_id: int, gameweek: int, player_ids: set[int]) -> set[int]:
    try:
        payload = get_json(f"entry/{team_id}/event/{gameweek}/picks/")
        picks = payload.get("picks", []) if isinstance(payload, dict) else []
        ids = {int(row["element"]) for row in picks if row.get("element") is not None}
        return ids & player_ids if len(ids) == 15 else set()
    except Exception:
        return set()


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
    players = bootstrap.get("elements", [])
    teams = {int(team["id"]): team for team in bootstrap.get("teams", []) if team.get("id") is not None}
    types = bootstrap.get("element_types", [])
    events = bootstrap.get("events", [])
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
            and (log["season"].astype(str) == season).any()
            and ((log["season"].astype(str) == season) & (pd.to_numeric(log["gameweek"], errors="coerce") == gameweek)).any()
        )
        if 0 < hours_to_deadline <= 6 and not already_saved:
            calibration_offsets, calibration_meta = fpl_engine.derive_calibration(log, gameweek)
            if calibration_meta.get("enabled"):
                print(f"Online calibration active from {calibration_meta.get('completed_gws')} prior Gameweeks: {calibration_offsets}")
            else:
                print(f"Online calibration warming up ({calibration_meta.get('completed_gws', 0)}/4 prior model Gameweeks).")
            player_by_id = {int(player["id"]): player for player in players if player.get("id") is not None}
            priority_pool = sorted(players, key=history_priority, reverse=True)[:45]
            history_ids = {int(player["id"]) for player in priority_pool if player.get("id") is not None}
            history_ids |= get_public_picks_ids(team_id, gameweek, set(player_by_id))
            signals = fetch_history_signals(history_ids)
            created = datetime.now(timezone.utc).isoformat(timespec="seconds")
            rows = []
            for player in players:
                try:
                    player_id = int(player["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                team_code = teams.get(int(player.get("team") or -1), {}).get("short_name", "?")
                projected, _, _ = fpl_engine.estimate_points(player, team_code, fixtures, 1, start_gw=gameweek, use_history=True, history_signals=signals, calibration_offsets=calibration_offsets)
                baseline, _, _ = fpl_engine.estimate_points(player, team_code, fixtures, 1, start_gw=gameweek, use_history=False, history_signals={}, calibration_offsets={})
                rows.append({
                    "season": season, "gameweek": gameweek, "player_id": player_id,
                    "player": player.get("web_name", ""), "position": position_short(player, types),
                    "club": team_code, "projection_model": projected, "projection_baseline": baseline,
                    "forecast_created_utc": created, "actual_points": None, "actual_minutes": None,
                    "model_version": "2.8",
                })
            if rows:
                log = pd.concat([log, pd.DataFrame(rows, columns=COLUMNS)], ignore_index=True)
                changed = True
                print(f"Saved automatic pre-deadline forecast for {season} GW{gameweek}: {len(rows)} players.")
        elif already_saved:
            print(f"Forecast already exists for {season} GW{gameweek}; preserving the first snapshot.")
        elif hours_to_deadline > 6:
            print(f"GW{gameweek} is {hours_to_deadline:.1f} hours from deadline; waiting for the 6-hour decision window.")
        else:
            print(f"Deadline for GW{gameweek} has passed; will not create a retrospective forecast.")
    else:
        print("No upcoming event with a published deadline; skipped new snapshot.")

    finished_gws = {int(event["id"]) for event in events if event.get("finished") and event.get("id") is not None}
    if not log.empty:
        numeric_gws = pd.to_numeric(log["gameweek"], errors="coerce")
        for gameweek in sorted(numeric_gws.dropna().astype(int).unique()):
            mask = (numeric_gws == gameweek)
            if gameweek not in finished_gws or not log.loc[mask, "actual_points"].isna().any():
                continue
            try:
                live = get_json(f"event/{gameweek}/live/")
                actual_map = {}
                for element in live.get("elements", []):
                    stats = element.get("stats", {})
                    try:
                        actual_map[int(element["id"])] = (float(stats.get("total_points") or 0), float(stats.get("minutes") or 0))
                    except (KeyError, TypeError, ValueError):
                        continue
                for index in log.index[mask]:
                    try:
                        player_id = int(log.at[index, "player_id"])
                    except (TypeError, ValueError):
                        continue
                    if player_id in actual_map:
                        log.at[index, "actual_points"] = actual_map[player_id][0]
                        log.at[index, "actual_minutes"] = actual_map[player_id][1]
                        changed = True
                print(f"Reconciled actual scores for finished GW{gameweek}.")
            except Exception as exc:
                print(f"Could not fetch actuals for GW{gameweek}: {exc}")

    # Keep one stable record per season/Gameweek/player and preserve filled actuals.
    if not log.empty:
        log["gameweek"] = pd.to_numeric(log["gameweek"], errors="coerce")
        log["player_id"] = pd.to_numeric(log["player_id"], errors="coerce")
        log = log.sort_values("forecast_created_utc").drop_duplicates(subset=["season", "gameweek", "player_id"], keep="last")
        log = log[COLUMNS]
    if changed:
        write_log(log_path, log)
        print(f"Updated persistent forecast log at {log_path} ({len(log)} rows).")
    else:
        print(f"No forecast-log changes required. Existing log has {len(log)} rows.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FPL scheduled worker failed: {exc}", file=sys.stderr)
        raise
