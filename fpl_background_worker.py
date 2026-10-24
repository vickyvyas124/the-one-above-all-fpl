"""Scheduled FPL forecast snapshot and backtest updater for v3.0.

Snapshots the full player pool, stores a same-deadline FPL ep_next benchmark,
records history coverage, and reconciles both single-GW and five-GW outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

import fpl_engine
import fpl_learning
import fpl_snapshot_store

BASE = "https://fantasy.premierleague.com/api"
MODEL_VERSION = "3.0"
COLUMNS = [
    "season", "gameweek", "player_id", "player", "position", "club",
    "model_version", "forecast_created_utc",
    "projection_model", "projection_baseline", "projection_ep_next",
    "projection_model_5gw", "projection_baseline_5gw", "forecast_horizon_gws",
    "history_enriched", "history_sample_matches", "projection_expected_minutes", "xi_eligible", "model_captain_pick",
    "actual_points", "actual_minutes", "actual_points_5gw", "actual_minutes_5gw",
]
HEADERS = {"User-Agent": "Mozilla/5.0 The-One-Above-All-FPL/3.0"}


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


def get_json_with_record(path: str, attempts: int = 3):
    """Fetch parsed JSON and an exact-body provenance record for a snapshot."""
    last_error = None
    url = f"{BASE}/{path}"
    last_body = None
    last_status = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=25, headers=HEADERS)
            last_status = response.status_code
            last_body = bytes(response.content)
            response.raise_for_status()
            payload = json.loads(last_body.decode("utf-8"))
            return payload, fpl_snapshot_store.payload_record(
                body=last_body, url=url, status_code=last_status, fetched_at_utc=datetime.now(timezone.utc)
            )
        except Exception as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.8 * (2 ** attempt))
    fetched = datetime.now(timezone.utc)
    return None, fpl_snapshot_store.payload_record(
        body=last_body, url=url,
        status_code=last_status or getattr(getattr(last_error, "response", None), "status_code", None),
        fetched_at_utc=fetched, error=str(last_error),
    )


def fetch_history_capture(player_ids, cutoff_utc=None):
    """Fetch exact element-summary bodies, summaries and per-player statuses."""
    signals, failures, records = {}, {}, {}
    unique_ids = sorted({int(pid) for pid in player_ids if pid is not None})

    def fetch(player_id):
        path = f"element-summary/{player_id}/"
        payload, record = get_json_with_record(path, attempts=3)
        if not isinstance(payload, dict):
            return player_id, None, record, record["metadata"].get("error") or "invalid response"
        try:
            signal = fpl_engine.make_history_signal(payload, cutoff_utc=cutoff_utc)
            return player_id, signal, record, None
        except Exception as exc:
            record["metadata"]["error"] = str(exc)
            record["metadata"]["fetch_outcome"] = "error"
            return player_id, None, record, str(exc)

    if not unique_ids:
        return signals, failures, records
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, pid) for pid in unique_ids]
        for future in as_completed(futures):
            pid, signal, record, error = future.result()
            records[pid] = record
            if signal is not None:
                signals[pid] = signal
            else:
                failures[pid] = error or "unknown error"
    return signals, failures, records


def capture_forecast_rows(players, teams, fixtures, gameweek, horizon_len, signals):
    """Produce the exact production-engine projections stored with a capture."""
    rows, failures = [], []
    for player in players:
        try:
            player_id = int(player["id"])
            club = teams.get(int(player.get("team") or -1), {}).get("short_name", "?")
            projection, _, _ = fpl_engine.estimate_points(
                player, club, fixtures, 1, start_gw=gameweek,
                use_history=True, history_signals=signals, calibration_offsets={},
            )
            projection_5gw, _, _ = fpl_engine.estimate_points(
                player, club, fixtures, horizon_len, start_gw=gameweek,
                use_history=True, history_signals=signals, calibration_offsets={},
            )
            rows.append({"player_id": player_id, "projection_model": projection,
                         "projection_model_5gw": projection_5gw})
        except Exception as exc:
            failures.append({"player_id": player.get("id"), "error": str(exc)})
    return rows, failures


def fetch_remote_snapshot_metadata(season: str, gameweek: int):
    """Read only the two manifests and diagnostics from the persistent snapshot branch."""
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        return {}, [], {}
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if not repository or "/" not in repository:
        return {}, [], {"repository": "GITHUB_REPOSITORY is missing or invalid"}
    api_base = f"https://api.github.com/repos/{repository}/contents"
    raw_base_url = f"https://raw.githubusercontent.com/{repository}/fpl-snapshots"
    headers = {**HEADERS, "Accept": "application/vnd.github.raw+json"}
    token = os.environ.get("FPL_SNAPSHOT_GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    manifests, errors = {}, {}
    for label in fpl_snapshot_store.WINDOWS:
        path = f"{season}/gw{int(gameweek):02d}/{label}/manifest.json"
        url = f"{api_base}/{path}?ref=fpl-snapshots"
        try:
            response = requests.get(url, timeout=15, headers=headers)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            manifest = json.loads(response.content.decode("utf-8"))
            if manifest.get("identity") != {"season": season, "gameweek": int(gameweek), "window_label": label}:
                errors[label] = "Remote manifest identity does not match its path"
                continue
            manifests[label] = manifest
        except Exception as exc:
            errors[label] = str(exc)
    diagnostics = []
    local_diagnostics = Path(os.environ.get("FPL_SNAPSHOT_DIR", "fpl-snapshots")) / "diagnostics.jsonl"
    if local_diagnostics.exists():
        lines = local_diagnostics.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                diagnostics.append(json.loads(line))
            except ValueError:
                errors["diagnostics"] = "Local persistent diagnostics file contains invalid JSONL"
                break
    else:
        diagnostics_url = f"{raw_base_url}/diagnostics.jsonl"
        try:
            response = requests.get(diagnostics_url, timeout=15, headers=HEADERS)
            if response.status_code != 404:
                response.raise_for_status()
                for line in response.content.decode("utf-8").splitlines():
                    try:
                        diagnostics.append(json.loads(line))
                    except ValueError:
                        errors["diagnostics"] = "Remote diagnostics file contains invalid JSONL"
                        break
        except Exception as exc:
            errors["diagnostics"] = str(exc)
    return manifests, diagnostics, errors


def restore_remote_actual_ledger(snapshot_root: str | Path, season: str, gameweek: int):
    """Restore one score ledger only when its daily recheck is due."""
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        return True, None
    repository = os.environ.get("GITHUB_REPOSITORY", "")
    if not repository or "/" not in repository:
        return False, "GITHUB_REPOSITORY is missing or invalid"
    path = Path(snapshot_root) / "actuals" / str(season) / f"gw{int(gameweek):02d}.jsonl"
    if path.exists():
        return True, None
    api_path = f"actuals/{season}/gw{int(gameweek):02d}.jsonl"
    url = f"https://api.github.com/repos/{repository}/contents/{api_path}?ref=fpl-snapshots"
    headers = {**HEADERS, "Accept": "application/vnd.github.raw+json"}
    token = os.environ.get("FPL_SNAPSHOT_GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        response = requests.get(url, timeout=15, headers=headers)
        if response.status_code == 404:
            return True, None
        response.raise_for_status()
        for line in response.content.decode("utf-8").splitlines():
            json.loads(line)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".tmp")
        temp.write_bytes(response.content)
        os.replace(temp, path)
        return True, None
    except Exception as exc:
        return False, str(exc)


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
        raise RuntimeError(f"Existing forecast CSV could not be read; preserving it and skipping this worker run: {exc}") from exc
    required = {"season", "gameweek", "player_id"}
    if not required.issubset(frame.columns):
        raise RuntimeError("Existing forecast CSV has no recognizable forecast schema; preserving it and skipping this worker run")
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    if "model_version" in frame.columns:
        versions = frame["model_version"].astype("string").str.strip()
        frame["model_version"] = versions.fillna("legacy-unknown").replace("", "legacy-unknown").astype(str)
    return frame[COLUMNS]


def write_log(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    frame[COLUMNS].to_csv(temp, index=False)
    os.replace(temp, path)


def fetch_one_history(player_id: int, cutoff_utc=None):
    last_error = None
    for attempt in range(3):
        try:
            payload = get_json(f"element-summary/{int(player_id)}/", attempts=1)
            return int(player_id), fpl_engine.make_history_signal(payload, cutoff_utc=cutoff_utc), None
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
    return int(player_id), None, str(last_error)


def fetch_history_signals(player_ids, cutoff_utc=None):
    signals, failures = {}, {}
    unique_ids = sorted({int(pid) for pid in player_ids if pid is not None})
    if not unique_ids:
        return signals, failures
    # All-player coverage prevents the forecast pool receiving hidden mixtures of inputs.
    # Bounded concurrency and retries reduce burst pressure on the public endpoint.
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch_one_history, pid, cutoff_utc) for pid in unique_ids]
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
            points = float(stats.get("total_points") or 0)
            minutes = float(stats.get("minutes") or 0)
            if not math.isfinite(points) or not math.isfinite(minutes):
                continue
            result[int(element["id"])] = (points, minutes)
        except (KeyError, TypeError, ValueError):
            continue
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default="forecast_log.csv")
    parser.add_argument("--snapshot-dir", default=os.environ.get("FPL_SNAPSHOT_DIR"))
    args = parser.parse_args()
    log_path = Path(args.csv_path)
    model_path = log_path.with_name("trained_fpl_model.json")
    fpl_engine.TRAINED_MODEL = fpl_learning.load_model(model_path)
    if fpl_engine.TRAINED_MODEL:
        print(f"Loaded trained model v3.0: {fpl_engine.TRAINED_MODEL.get('training_rows', 0)} historical player-GWs from {fpl_engine.TRAINED_MODEL.get('source_seasons', [])}.")
    else:
        print("DIAGNOSTIC: new v3.0 forecast snapshots are disabled because the trained model artifact is missing, invalid, or incompatible. Existing forecast rows will still be reconciled.")
    team_id = int(os.environ.get("FPL_TEAM_ID", "2311669"))
    changed = False
    log = read_log(log_path)

    bootstrap, bootstrap_record = get_json_with_record("bootstrap-static/")
    fixtures, fixtures_record = get_json_with_record("fixtures/")
    if not isinstance(bootstrap, dict):
        fpl_snapshot_store.append_diagnostic(
            args.snapshot_dir or log_path.parent / "fpl-snapshots", season=season_label(),
            gameweek=None, kind="source_fetch_failure",
            message="Could not load bootstrap-static or fixtures; capture cannot be assigned to an event.",
            at_utc=datetime.now(timezone.utc),
        )
        print("DIAGNOSTIC: source payload unavailable; snapshot capture skipped. Existing forecast rows will still be reconciled only if usable bootstrap data is available.")
        return 1
    fixtures_available = isinstance(fixtures, list)
    if not fixtures_available:
        fixtures = []
    players = [p for p in bootstrap.get("elements", []) if p.get("id") is not None]
    teams = {int(team["id"]): team for team in bootstrap.get("teams", []) if team.get("id") is not None}
    types = bootstrap.get("element_types", [])
    events = bootstrap.get("events", [])
    event_by_id = {int(e["id"]): e for e in events if e.get("id") is not None}
    next_event = next((event for event in events if event.get("is_next")), None)
    current_event = next((event for event in events if event.get("is_current")), None)
    target = next_event or (None if current_event and current_event.get("finished") else current_event)
    season = season_label()

    # Capture context independently of model availability. The 24h and 6h
    # windows are separate immutable identities; six hours is the primary
    # decision snapshot and 24 hours is retained for comparison.
    snapshot_root = Path(args.snapshot_dir) if args.snapshot_dir else log_path.parent / "fpl-snapshots"
    captured_at = datetime.now(timezone.utc)
    if target and target.get("id") is not None and target.get("deadline_time"):
        gameweek = int(target["id"])
        deadline = datetime.fromisoformat(str(target["deadline_time"]).replace("Z", "+00:00"))
        window_label = fpl_snapshot_store.window_for_deadline(captured_at, deadline)
        remote_manifests, remote_diagnostics, remote_errors = fetch_remote_snapshot_metadata(season, gameweek)
        for source_name, error in remote_errors.items():
            message = f"Could not verify persistent snapshot store ({source_name}): {error}"
            if not fpl_snapshot_store.diagnostic_exists(
                    snapshot_root, season=season, gameweek=gameweek,
                    kind="snapshot_store_read_failure", message_prefix=f"Could not verify persistent snapshot store ({source_name}):",
                    existing_records=remote_diagnostics):
                fpl_snapshot_store.append_diagnostic(
                    snapshot_root, season=season, gameweek=gameweek,
                    kind="snapshot_store_read_failure", message=message,
                    at_utc=datetime.now(timezone.utc),
                )
        if window_label:
            snapshot_path = snapshot_root / season / f"gw{gameweek:02d}" / window_label
            if not (snapshot_path / "manifest.json").exists() and window_label not in remote_manifests and window_label not in remote_errors:
                cutoff = deadline - timedelta(hours=2)
                signals, history_failures, history_records = fetch_history_capture(
                    [p.get("id") for p in players], cutoff_utc=cutoff
                )
                source_records = {"bootstrap-static": bootstrap_record, "fixtures": fixtures_record}
                team_id = int(os.environ.get("FPL_TEAM_ID", "2311669"))
                picks_payload, picks_record = get_json_with_record(
                    f"entry/{team_id}/event/{gameweek}/picks/", attempts=1
                )
                source_records["public-picks"] = picks_record
                _, entry_record = get_json_with_record(f"entry/{team_id}/", attempts=1)
                source_records["public-entry"] = entry_record
                max_gw = max(event_by_id) if event_by_id else gameweek
                horizon_len = max(1, min(gameweek + 4, max_gw) - gameweek + 1)
                artifact_valid = fpl_learning.validate_model(fpl_engine.TRAINED_MODEL)
                model_sha = None
                model_bytes = None
                if model_path.exists():
                    try:
                        model_bytes = model_path.read_bytes()
                        model_sha = hashlib.sha256(model_bytes).hexdigest()
                    except OSError:
                        artifact_valid = False
                model_valid = artifact_valid and fixtures_available
                forecast_failures = []
                forecast_rows = None
                if model_valid:
                    forecast_rows, forecast_failures = capture_forecast_rows(
                        players, teams, fixtures, gameweek, horizon_len, signals
                    )
                    if forecast_failures:
                        forecast_rows = None
                if artifact_valid and not model_sha:
                    artifact_valid = False
                    model_valid = False
                    forecast_rows = None
                model_reason = None if model_valid and not forecast_failures else (
                    f"Projection generation failed for {len(forecast_failures)} players; context captured without a forecast."
                    if forecast_failures else
                    "Fixture source unavailable; context captured without projections."
                    if not fixtures_available else
                    ("Model artifact is missing; context captured without projections."
                     if not model_path.exists() else
                     "Model artifact is invalid, incompatible or unreadable; context captured without projections.")
                )
                capture_completed_at = datetime.now(timezone.utc)
                if capture_completed_at >= deadline:
                    fpl_snapshot_store.append_diagnostic(
                        snapshot_root, season=season, gameweek=gameweek,
                        kind="deadline_passed_before_capture",
                        message="Source collection completed at or after the deadline; no purported pre-deadline snapshot was written.",
                        at_utc=capture_completed_at,
                    )
                    forecast_rows = None
                    model_reason = "Source collection completed after deadline; snapshot rejected."
                    print(f"DIAGNOSTIC: {model_reason}")
                    continue_snapshot = False
                else:
                    continue_snapshot = True
                if continue_snapshot:
                    snapshot_path, created_snapshot = fpl_snapshot_store.capture_snapshot(
                        snapshot_root, season=season, gameweek=gameweek,
                        window_label=window_label, deadline_utc=deadline,
                        captured_at_utc=capture_completed_at, sources=source_records,
                        histories=history_records,
                        model_status="valid" if artifact_valid else ("missing" if not model_path.exists() else "invalid_or_incompatible"),
                        model_reason=model_reason, model_sha256=model_sha,
                        model_artifact_body=model_bytes if artifact_valid else None,
                        code_commit_sha=os.environ.get("GITHUB_SHA"),
                        forecast_rows=forecast_rows,
                        provenance={"deadline": "derived",
                                    "fixture_schedule": "captured",
                                    **({"public_picks": "captured"} if picks_payload is not None else {}),
                                    **({"public_entry": "captured"} if entry_record.get("metadata", {}).get("fetch_outcome") == "success" else {})},
                    )
                    if created_snapshot:
                        print(f"Captured immutable {window_label} context snapshot for {season} GW{gameweek}: {len(history_records) - len(history_failures)}/{len(players)} histories available; model={'valid' if model_valid else 'unavailable'}.")
                    else:
                        print(f"Snapshot already exists for {season} GW{gameweek} {window_label}; preserving it.")
                failed_sources = [name for name, record in source_records.items()
                                  if not record or record.get("metadata", {}).get("fetch_outcome") != "success"]
                if failed_sources:
                    fpl_snapshot_store.append_diagnostic(
                        snapshot_root, season=season, gameweek=gameweek,
                        kind="partial_api_fetch",
                        message=f"Source fetch failed or was partial: {', '.join(sorted(failed_sources))}.",
                        at_utc=capture_completed_at,
                    )
                if not artifact_valid:
                    fpl_snapshot_store.append_diagnostic(
                        snapshot_root, season=season, gameweek=gameweek,
                        kind="model_failure", message=model_reason or "Model artifact unavailable.",
                        at_utc=capture_completed_at,
                    )
                if history_failures:
                    fpl_snapshot_store.append_diagnostic(
                        snapshot_root, season=season, gameweek=gameweek,
                        kind="history_gaps", message=f"{len(history_failures)} player history fetches failed.",
                        at_utc=capture_completed_at,
                    )
                if forecast_failures:
                    fpl_snapshot_store.append_diagnostic(
                        snapshot_root, season=season, gameweek=gameweek,
                        kind="forecast_failure",
                        message=("Projection generation failed for player IDs: "
                                 + ", ".join(str(item["player_id"]) for item in forecast_failures[:25]) + "."),
                        at_utc=capture_completed_at,
                    )
            elif window_label in remote_manifests and not (snapshot_path / "manifest.json").exists():
                print(f"Snapshot already exists in persistent branch for {season} GW{gameweek} {window_label}; preserving it.")
            elif window_label in remote_errors:
                print(f"DIAGNOSTIC: could not verify the persistent {window_label} snapshot identity; skipping capture.")
        # Check on every workflow run, including when the current immutable
        # window already exists or a moved deadline is outside both windows.
        fpl_snapshot_store.record_deadline_move(
            snapshot_root, season=season, gameweek=gameweek,
            current_window=window_label or "outside-window", deadline_utc=deadline,
            at_utc=datetime.now(timezone.utc), remote_manifests=remote_manifests,
            existing_diagnostics=remote_diagnostics,
        )
        hours = (deadline - captured_at).total_seconds() / 3600.0
        fpl_snapshot_store.record_missed_windows(
            snapshot_root, season=season, gameweek=gameweek,
            hours_to_deadline=hours, at_utc=captured_at,
            existing_windows=set(remote_manifests), existing_diagnostics=remote_diagnostics,
        )

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
        if 0 < hours_to_deadline <= 6 and not already_saved and fpl_learning.validate_model(fpl_engine.TRAINED_MODEL) and fixtures_available:
            # No online correction is applied during this validation release. Freeze the
            # formula while the benchmarks and rankings are accumulated.
            player_by_id = {int(player["id"]): player for player in players}
            history_ids = set(player_by_id)
            history_cutoff = deadline - timedelta(hours=2)
            signals, history_failures = fetch_history_signals(history_ids, cutoff_utc=history_cutoff)
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
        elif 0 < hours_to_deadline <= 6 and not fpl_engine.TRAINED_MODEL:
            print(f"DIAGNOSTIC: skipped new forecast for {season} GW{gameweek}; trained v3.0 artifact is unavailable or invalid. Existing rows are preserved and will still be reconciled.")
        elif 0 < hours_to_deadline <= 6 and not fixtures_available:
            print(f"DIAGNOSTIC: skipped new forecast for {season} GW{gameweek}; fixture source is unavailable. Existing rows are preserved.")
        elif already_saved:
            print(f"Forecast already exists for {season} GW{gameweek}; preserving the first snapshot.")
        elif hours_to_deadline > 6:
            print(f"GW{gameweek} is {hours_to_deadline:.1f} hours from deadline; waiting for the six-hour decision window.")
        else:
            print(f"Deadline for GW{gameweek} has passed; will not create a retrospective forecast.")
    else:
        print("No upcoming event with a published deadline; skipped new snapshot.")

    finalized_events = {
        int(event["id"]): event for event in events
        if event.get("id") is not None and fpl_snapshot_store.event_scores_final(event)
    }
    finalized_gws = set(finalized_events)
    max_gw = max(event_by_id) if event_by_id else 38
    if not log.empty:
        # Final score checks use data_checked when the API exposes it. The
        # append-only revision ledger is polled daily for late score changes.
        numeric_gws = pd.to_numeric(log["gameweek"], errors="coerce")
        current_season = log["season"].astype(str) == season
        current_gws = sorted(numeric_gws[current_season].dropna().astype(int).unique())
        for gameweek in current_gws:
            mask = (numeric_gws == gameweek) & current_season
            if gameweek not in finalized_gws:
                continue
            needs_revision_check = fpl_snapshot_store.actual_reconciliation_due(
                snapshot_root, season=season, gameweek=gameweek, now=datetime.now(timezone.utc)
            )
            needs_forecast_log_update = log.loc[mask, "actual_points"].isna().any()
            if not needs_revision_check and not needs_forecast_log_update:
                continue
            ledger_ready = True
            if needs_revision_check:
                ledger_ready, ledger_error = restore_remote_actual_ledger(
                    snapshot_root, season, gameweek
                )
                if not ledger_ready:
                    message = f"Could not restore finalized actual-score history: {ledger_error}"
                    if not fpl_snapshot_store.diagnostic_exists(
                            snapshot_root, season=season, gameweek=gameweek,
                            kind="actual_ledger_restore_failure",
                            message_prefix="Could not restore finalized actual-score history:"):
                        fpl_snapshot_store.append_diagnostic(
                            snapshot_root, season=season, gameweek=gameweek,
                            kind="actual_ledger_restore_failure", message=message,
                            at_utc=datetime.now(timezone.utc),
                        )
            try:
                actuals = actual_map_for_gw(gameweek)
                if needs_revision_check and ledger_ready:
                    changed_scores = fpl_snapshot_store.append_finalized_actuals(
                        snapshot_root, season=season, gameweek=gameweek,
                        actuals=actuals,
                        data_checked=bool(finalized_events[gameweek].get("data_checked", True)),
                        reconciled_at_utc=datetime.now(timezone.utc),
                    )
                    print(f"Final-score reconciliation checkpoint for GW{gameweek}; changed player scores={changed_scores}.")
                if needs_forecast_log_update:
                    for idx in log.index[mask]:
                        try:
                            player_id = int(log.at[idx, "player_id"])
                        except (TypeError, ValueError):
                            continue
                        if player_id in actuals:
                            log.at[idx, "actual_points"] = actuals[player_id][0]
                            log.at[idx, "actual_minutes"] = actuals[player_id][1]
                            changed = True
                    print(f"Reconciled one-GW forecast-log actuals for GW{gameweek}.")
            except Exception as exc:
                print(f"Could not fetch actuals for GW{gameweek}: {exc}")

        # Reconcile the five-GW horizon only once every event in that horizon is finished.
        all_gws = current_gws
        for gameweek in all_gws:
            mask = (numeric_gws == gameweek) & current_season
            if not log.loc[mask, "actual_points_5gw"].isna().any():
                continue
            horizon_end = min(int(gameweek) + 4, max_gw)
            horizon_events = set(range(int(gameweek), horizon_end + 1))
            if not horizon_events.issubset(finalized_gws):
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
