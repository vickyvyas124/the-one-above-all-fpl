"""Immutable, content-hashed storage for pre-deadline FPL captures.

The store is filesystem-backed for local runs and tests. Main-branch CI
publishes completed bundles to the dedicated ``fpl-snapshots`` Git branch.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "1.0"
FEATURE_POLICY_VERSION = "point-in-time-capture-v1"
ACTUAL_RECHECK_HOURS = 24
WINDOWS = {"24h": (23.0, 25.0), "6h": (5.0, 7.0)}


def utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Snapshot times must include a timezone")
    return value.astimezone(timezone.utc)


def window_for_deadline(now: datetime, deadline: datetime) -> str | None:
    hours = (utc_datetime(deadline) - utc_datetime(now)).total_seconds() / 3600
    for label, (lower, upper) in WINDOWS.items():
        if lower <= hours <= upper:
            return label
    return None


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def payload_record(*, body: bytes | None, url: str, status_code: int | None,
                   fetched_at_utc: datetime | str, error: str | None = None) -> dict[str, Any]:
    """Describe a response body without losing its exact bytes."""
    raw = bytes(body) if body is not None else None
    return {
        "body": raw,
        "metadata": {
            "url": str(url), "http_status": status_code,
            "fetch_outcome": "success" if raw is not None and error is None else "error",
            "fetched_at_utc": utc_datetime(fetched_at_utc).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
            "compression": "gzip" if raw is not None else None,
            "error": str(error) if error else None,
            "body_bytes": len(raw) if raw is not None else None,
        },
    }


def append_diagnostic(root: str | Path, *, season: str, gameweek: int | None,
                      kind: str, message: str, at_utc: datetime | str) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    record = {"at_utc": utc_datetime(at_utc).isoformat(timespec="seconds"),
              "season": str(season), "gameweek": gameweek,
              "kind": str(kind), "message": str(message)}
    path = root / "diagnostics.jsonl"
    with path.open("ab") as stream:
        stream.write(_canonical_bytes(record) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def diagnostic_exists(root: str | Path, *, season: str, gameweek: int | None,
                      kind: str, message_prefix: str = "",
                      existing_records: list[Mapping[str, Any]] | None = None) -> bool:
    path = Path(root) / "diagnostics.jsonl"
    items: list[Mapping[str, Any]] = list(existing_records or [])
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                items.append(json.loads(line))
            except ValueError:
                continue
    for item in items:
        if (item.get("season") == str(season) and item.get("gameweek") == gameweek
                and item.get("kind") == kind
                and str(item.get("message", "")).startswith(message_prefix)):
            return True
    return False


def record_deadline_move(root: str | Path, *, season: str, gameweek: int,
                         current_window: str, deadline_utc: datetime | str,
                         at_utc: datetime | str,
                         remote_manifests: Mapping[str, Mapping[str, Any]] | None = None,
                         existing_diagnostics: list[Mapping[str, Any]] | None = None) -> bool:
    """Append a diagnostic if another immutable window records a different deadline."""
    root = Path(root)
    current = utc_datetime(deadline_utc)
    for label in WINDOWS:
        manifest_path = root / str(season) / f"gw{int(gameweek):02d}" / label / "manifest.json"
        try:
            manifest = (remote_manifests or {}).get(label)
            if manifest is None and manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            previous = manifest.get("deadline_utc") if manifest else None
            previous_dt = utc_datetime(previous) if previous else None
        except (OSError, ValueError, TypeError, AttributeError):
            continue
        if previous_dt and previous_dt != current:
            if label == current_window:
                message = (f"Deadline changed after the {label} snapshot: "
                           f"{previous_dt.isoformat()} -> {current.isoformat()}.")
            else:
                message = (f"Deadline changed between {label} and {current_window} captures: "
                           f"{previous_dt.isoformat()} -> {current.isoformat()}.")
            if not diagnostic_exists(root, season=season, gameweek=gameweek,
                                     kind="deadline_moved", message_prefix=message,
                                     existing_records=existing_diagnostics):
                append_diagnostic(root, season=season, gameweek=gameweek,
                                  kind="deadline_moved", message=message, at_utc=at_utc)
            return True
    return False


def record_missed_windows(root: str | Path, *, season: str, gameweek: int,
                          hours_to_deadline: float, at_utc: datetime | str,
                          existing_windows: set[str] | None = None,
                          existing_diagnostics: list[Mapping[str, Any]] | None = None) -> list[str]:
    """Record each window once after its latest eligible capture time has passed."""
    root = Path(root)
    missed = []
    for label, (lower, _) in WINDOWS.items():
        snapshot = root / str(season) / f"gw{int(gameweek):02d}" / label
        message = f"The {label} capture window passed without a snapshot."
        if (hours_to_deadline < lower and not (snapshot / "manifest.json").exists()
                and label not in (existing_windows or set())
                and not diagnostic_exists(root, season=season, gameweek=gameweek,
                                          kind="missed_capture_window", message_prefix=message,
                                          existing_records=existing_diagnostics)):
            append_diagnostic(root, season=season, gameweek=gameweek,
                              kind="missed_capture_window", message=message, at_utc=at_utc)
            missed.append(label)
    return missed


def event_scores_final(event: Mapping[str, Any]) -> bool:
    """Use data_checked when present; otherwise retain legacy finished semantics."""
    return bool(event.get("finished")) and bool(event.get("data_checked", True))


def actual_reconciliation_due(root: str | Path, *, season: str, gameweek: int,
                              now: datetime | str, interval_hours: int = ACTUAL_RECHECK_HOURS) -> bool:
    path = Path(root) / "actuals" / str(season) / f"gw{int(gameweek):02d}.jsonl"
    index_path = Path(root) / "reconciliation-index.json"
    index_key = f"{season}:GW{int(gameweek):02d}"
    last_checked = None
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(index, dict):
                last_checked = index.get(index_key)
        except (OSError, ValueError, TypeError):
            last_checked = None
    if not last_checked and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("record_type") == "reconciliation":
                last_checked = record.get("reconciled_at_utc")
    if not last_checked:
        return True
    elapsed = utc_datetime(now) - utc_datetime(last_checked)
    return elapsed.total_seconds() >= int(interval_hours) * 3600


def append_finalized_actuals(root: str | Path, *, season: str, gameweek: int,
                             actuals: Mapping[int, tuple[float, float]],
                             data_checked: bool, reconciled_at_utc: datetime | str) -> int:
    """Append score revisions and a reconciliation checkpoint; never edits a forecast."""
    if not data_checked:
        raise ValueError("Actual scores cannot be finalized before data_checked is true")
    path = Path(root) / "actuals" / str(season) / f"gw{int(gameweek):02d}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    latest: dict[int, tuple[float, float, int]] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
                if item.get("record_type") != "actual_score":
                    continue
                player_id = int(item["player_id"])
                latest[player_id] = (float(item["actual_points"]), float(item["actual_minutes"]),
                                     int(item["revision"]))
            except (ValueError, KeyError, TypeError):
                continue
    stamp = utc_datetime(reconciled_at_utc).isoformat(timespec="seconds")
    changed_players = []
    new_records = []
    for player_id, (points, minutes) in sorted(actuals.items()):
        player_id = int(player_id)
        points, minutes = float(points), float(minutes)
        prior = latest.get(player_id)
        if prior and math.isclose(points, prior[0], abs_tol=1e-9) and math.isclose(minutes, prior[1], abs_tol=1e-9):
            continue
        revision = (prior[2] + 1) if prior else 1
        record = {"record_type": "actual_score", "season": str(season),
                  "gameweek": int(gameweek), "player_id": player_id,
                  "previous_actual_points": prior[0] if prior else None,
                  "previous_actual_minutes": prior[1] if prior else None,
                  "actual_points": points, "actual_minutes": minutes,
                  "revision": revision, "reconciled_at_utc": stamp,
                  "data_checked": True}
        new_records.append(record)
        changed_players.append(player_id)
    with path.open("ab") as stream:
        for item in new_records:
            stream.write(_canonical_bytes(item) + b"\n")
        checkpoint = {"record_type": "reconciliation", "season": str(season),
                      "gameweek": int(gameweek), "reconciled_at_utc": stamp,
                      "data_checked": True, "changed_player_ids": changed_players}
        stream.write(_canonical_bytes(checkpoint) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    index_path = path.parents[2] / "reconciliation-index.json"
    try:
        index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {}
        if not isinstance(index, dict):
            index = {}
    except (OSError, ValueError):
        index = {}
    index[f"{season}:GW{int(gameweek):02d}"] = stamp
    temp_index = index_path.with_suffix(".tmp")
    temp_index.write_bytes(_canonical_bytes(index) + b"\n")
    os.replace(temp_index, index_path)
    return len(changed_players)


def _source_summary(record: Mapping[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {"fetch_outcome": "missing", "error": "not fetched", "sha256": None}
    return dict(record.get("metadata") or {})


def _model_json_value(body: bytes | None, key: str) -> Any:
    if body is None:
        return None
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return value.get(key) if isinstance(value, dict) else None


def capture_snapshot(
    root: str | Path,
    *,
    season: str,
    gameweek: int,
    window_label: str,
    deadline_utc: datetime | str,
    captured_at_utc: datetime | str,
    sources: Mapping[str, Mapping[str, Any] | None],
    histories: Mapping[int | str, Mapping[str, Any] | None],
    model_status: str,
    model_reason: str | None = None,
    model_sha256: str | None = None,
    model_artifact_body: bytes | None = None,
    code_commit_sha: str | None = None,
    forecast_rows: list[Mapping[str, Any]] | None = None,
    provenance: Mapping[str, str] | None = None,
) -> tuple[Path, bool]:
    """Atomically persist one immutable `(season, GW, window)` snapshot.

    Returns `(snapshot_path, created)`. Existing identities are never replaced.
    Forecast rows are accepted only with a compatible model and model digest.
    """
    if window_label not in WINDOWS:
        raise ValueError(f"Unknown capture window: {window_label}")
    if int(gameweek) < 1 or int(gameweek) > 50:
        raise ValueError(f"Invalid Gameweek: {gameweek}")
    deadline = utc_datetime(deadline_utc)
    captured = utc_datetime(captured_at_utc)
    if captured >= deadline:
        raise ValueError("Cannot capture a pre-deadline snapshot at or after its captured deadline")
    if forecast_rows and (model_status != "valid" or not model_sha256 or model_artifact_body is None):
        raise ValueError("Forecast snapshots require a valid model and model artifact SHA-256")
    if model_artifact_body is not None and hashlib.sha256(model_artifact_body).hexdigest() != model_sha256:
        raise ValueError("Captured model artifact bytes do not match the supplied SHA-256")

    destination = Path(root) / str(season) / f"gw{int(gameweek):02d}" / window_label
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        return destination, False

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{window_label}-", dir=parent))
    try:
        payload_dir = staging / "payloads"
        history_dir = payload_dir / "element-summary"
        history_dir.mkdir(parents=True)
        source_metadata: dict[str, Any] = {}
        provenance_manifest = dict(provenance or {})
        for name, record in sources.items():
            record = record or {}
            body = record.get("body")
            source_metadata[name] = _source_summary(record)
            if body is not None:
                body = bytes(body)
                (payload_dir / f"{name}.json.gz").write_bytes(gzip.compress(body, mtime=0))
                provenance_manifest[f"source.{name}"] = "captured"

        history_status: dict[str, Any] = {}
        for player_id, record in sorted(histories.items(), key=lambda item: str(item[0])):
            player_key = str(player_id)
            record = record or {}
            body = record.get("body")
            history_status[player_key] = _source_summary(record)
            if body is not None:
                (history_dir / f"{player_key}.json.gz").write_bytes(gzip.compress(bytes(body), mtime=0))
                provenance_manifest[f"history.{player_key}"] = "captured"

        forecast_metadata = {"included": bool(forecast_rows), "model_status": model_status,
                             "model_sha256": model_sha256 if forecast_rows else None,
                             "skip_reason": None if forecast_rows else model_reason}
        if forecast_rows:
            forecast_bytes = _canonical_bytes(list(forecast_rows))
            (staging / "forecast.json.gz").write_bytes(gzip.compress(forecast_bytes, mtime=0))
            forecast_metadata["sha256"] = hashlib.sha256(forecast_bytes).hexdigest()
            provenance_manifest["forecast.projections"] = "derived"
        if model_artifact_body is not None:
            (staging / "model.json.gz").write_bytes(gzip.compress(bytes(model_artifact_body), mtime=0))
            provenance_manifest["model.artifact"] = "captured"
        if model_sha256:
            provenance_manifest["model.artifact_sha256"] = "captured"

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "feature_policy_version": FEATURE_POLICY_VERSION,
            "identity": {"season": str(season), "gameweek": int(gameweek), "window_label": window_label},
            "context_snapshot": {"included": True, "snapshot_id": f"{season}:GW{int(gameweek)}:{window_label}"},
            "captured_at_utc": captured.isoformat(timespec="seconds"),
            "deadline_utc": deadline.isoformat(timespec="seconds"),
            "code_commit_sha": code_commit_sha,
            "sources": source_metadata,
            "player_history": {"requested": len(histories), "available": sum(
                1 for item in history_status.values() if item.get("fetch_outcome") == "success"),
                "players": history_status},
            "missing_fields": ([f"source.{name}" for name, item in source_metadata.items()
                                if item.get("fetch_outcome") != "success"]
                               + [f"history.{player_id}" for player_id, item in history_status.items()
                                  if item.get("fetch_outcome") != "success"]
                               + (["model.artifact"] if model_sha256 and model_artifact_body is None else [])),
            "model": {"status": model_status, "sha256": model_sha256,
                      "artifact_captured": model_artifact_body is not None,
                      "artifact_model_version": _model_json_value(model_artifact_body, "model_version"),
                      "artifact_feature_policy": _model_json_value(model_artifact_body, "feature_policy"),
                      "reason": model_reason},
            "forecast": {**forecast_metadata,
                         "snapshot_id": f"{season}:GW{int(gameweek)}:{window_label}:{model_sha256}" if forecast_rows else None,
                         "primary_decision_snapshot": bool(forecast_rows and window_label == "6h"),
                         "comparison_snapshot": bool(forecast_rows and window_label == "24h")},
            "provenance": provenance_manifest,
            "bundle": {"compression": "gzip", "payload_layout": "payloads/"},
        }
        (staging / "manifest.json").write_bytes(_canonical_bytes(manifest) + b"\n")
        # Publish files atomically and publish the manifest last. A directory
        # rename is not reliable on Windows/OneDrive; readers treat the final
        # manifest as the commit marker for a complete immutable bundle.
        destination.mkdir(parents=True, exist_ok=True)
        staged_files = sorted(staging.rglob("*"), key=lambda item: (item.name == "manifest.json", str(item)))
        for staged_file in staged_files:
            if not staged_file.is_file():
                continue
            relative = staged_file.relative_to(staging)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_file, target)
        return destination, True
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def verify_snapshot(snapshot_path: str | Path) -> list[str]:
    """Verify all stored source payload hashes from the immutable manifest."""
    path = Path(snapshot_path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    source_names = manifest.get("sources", {})
    for name, metadata in source_names.items():
        digest = metadata.get("sha256")
        payload_path = path / "payloads" / f"{name}.json.gz"
        if not digest:
            continue
        try:
            actual = hashlib.sha256(gzip.decompress(payload_path.read_bytes())).hexdigest()
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            errors.append(f"{name}: unreadable gzip ({exc})")
            continue
        if actual != digest:
            errors.append(f"{name}: SHA-256 mismatch")
    for player_id, metadata in manifest.get("player_history", {}).get("players", {}).items():
        digest = metadata.get("sha256")
        if not digest:
            continue
        payload_path = path / "payloads" / "element-summary" / f"{player_id}.json.gz"
        try:
            actual = hashlib.sha256(gzip.decompress(payload_path.read_bytes())).hexdigest()
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            errors.append(f"history {player_id}: unreadable gzip ({exc})")
            continue
        if actual != digest:
            errors.append(f"history {player_id}: SHA-256 mismatch")
    model_info = manifest.get("model", {})
    model_digest = model_info.get("sha256")
    if model_digest and model_info.get("artifact_captured"):
        try:
            actual = hashlib.sha256(gzip.decompress((path / "model.json.gz").read_bytes())).hexdigest()
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            errors.append(f"model artifact: unreadable gzip ({exc})")
        else:
            if actual != model_digest:
                errors.append("model artifact: SHA-256 mismatch")
    forecast_digest = manifest.get("forecast", {}).get("sha256")
    if forecast_digest:
        try:
            actual = hashlib.sha256(gzip.decompress((path / "forecast.json.gz").read_bytes())).hexdigest()
        except (OSError, EOFError, gzip.BadGzipFile) as exc:
            errors.append(f"forecast: unreadable gzip ({exc})")
        else:
            if actual != forecast_digest:
                errors.append("forecast: SHA-256 mismatch")
    return errors
