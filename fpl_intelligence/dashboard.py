"""Read-only local-store loader for the Football Intelligence dashboard."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import gzip
import json
import os
from pathlib import Path
from typing import Any

from .analysis import AnalysisResult, analyze_observations
from .contract import EvidenceQuality, PlayerMatchObservation, validate_observation


DEFAULT_STORE_RELATIVE = Path(".local") / "football-intelligence"
MAX_MANIFESTS = 10_000
MAX_FIXTURES = 1_500
MAX_NORMALIZED_BYTES_PER_FIXTURE = 4 * 1024 * 1024


@dataclass(frozen=True)
class DashboardSnapshot:
    store_path: str
    status: str
    manifest_count: int
    fixture_count: int
    observation_count: int
    analyzed_player_count: int
    collection_status_counts: dict[str, int]
    last_data_retrieved_utc: str | None
    analysis: AnalysisResult | None
    warnings: tuple[str, ...]
    observations: tuple[PlayerMatchObservation, ...] = ()


def default_store_path() -> Path:
    configured = os.environ.get("FPL_INTELLIGENCE_STORE")
    if configured and configured.strip():
        return Path(configured.strip()).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / DEFAULT_STORE_RELATIVE).resolve()


def _contained_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError("manifest path is missing")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("manifest path escapes the configured store") from exc
    return candidate


def _retrieval_time(manifest: dict[str, Any]) -> str:
    times = [
        str(document.get("retrieved_at_utc"))
        for document in manifest.get("raw_documents", {}).values()
        if isinstance(document, dict) and document.get("retrieved_at_utc")
    ]
    return max(times, default=str(manifest.get("source_timestamp") or ""))


def _observation_from_dict(value: dict[str, Any]) -> PlayerMatchObservation:
    prepared = dict(value)
    prepared["completeness"] = EvidenceQuality(prepared["completeness"])
    prepared["missing_fields"] = tuple(prepared.get("missing_fields") or ())
    prepared["role_segments"] = tuple(prepared.get("role_segments") or ())
    prepared["metric_definitions"] = dict(prepared.get("metric_definitions") or {})
    return PlayerMatchObservation(**prepared)


def load_dashboard_snapshot(root: str | Path | None = None) -> DashboardSnapshot:
    """Read and validate the latest local revision per source fixture.

    The reader does not fetch data and does not write to the store. On a missing
    local dataset it returns an explicit empty state for hosted deployments.
    """
    store = Path(root).expanduser().resolve() if root is not None else default_store_path()
    if not store.exists():
        return DashboardSnapshot(
            str(store), "NO_LOCAL_DATASET", 0, 0, 0, 0, {}, None, None, (),
        )
    if not store.is_dir():
        return DashboardSnapshot(
            str(store), "INVALID_STORE_PATH", 0, 0, 0, 0, {}, None, None,
            ("Configured intelligence store path is not a directory.",),
        )

    manifests = sorted((store / "derived").rglob("*.manifest.json"))
    warnings: list[str] = []
    if len(manifests) > MAX_MANIFESTS:
        warnings.append(f"Store has more than {MAX_MANIFESTS} manifests; only the newest bounded subset was read.")
        manifests = manifests[-MAX_MANIFESTS:]

    latest: dict[tuple[str, str, str, str], tuple[str, Path, dict[str, Any]]] = {}
    for path in manifests:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("manifest must contain a JSON object")
            source = str(manifest["source_id"])
            competition = str(manifest["competition_id"])
            season = str(manifest["season_id"])
            fixture = str(manifest["fixture_id"])
            key = (source, competition, season, fixture)
            current = (_retrieval_time(manifest), path, manifest)
            if key not in latest or current[0] > latest[key][0]:
                latest[key] = current
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Unreadable intelligence manifest {path.name}: {type(exc).__name__}.")

    ordered_latest = sorted(latest.values(), key=lambda item: (item[0], item[1].as_posix()))
    if len(ordered_latest) > MAX_FIXTURES:
        ordered_latest = ordered_latest[-MAX_FIXTURES:]
        warnings.append(f"Only the latest {MAX_FIXTURES} fixture revisions were loaded.")

    observations: list[PlayerMatchObservation] = []
    statuses: dict[str, int] = {}
    last_success: str | None = None
    for retrieved, manifest_path, manifest in ordered_latest:
        status = str(manifest.get("collection_status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if status in {"data_successfully_collected", "data_incomplete"} and retrieved:
            last_success = max(last_success or retrieved, retrieved)
        try:
            source = str(manifest["source_id"])
            fixture = str(manifest["fixture_id"])
            documents = manifest.get("raw_documents")
            if not isinstance(documents, dict) or not documents:
                raise ValueError("manifest has no raw-document provenance")
            for raw in documents.values():
                if not isinstance(raw, dict):
                    raise ValueError("raw-document entry is malformed")
                raw_path = _contained_path(store, raw.get("store_path"))
                if not raw_path.is_file():
                    raise ValueError("a referenced immutable raw document is missing")
                raw_hash = str(raw.get("sha256") or "")
                if len(raw_hash) != 64 or any(ch not in "0123456789abcdef" for ch in raw_hash):
                    raise ValueError("raw-document hash is invalid")
            observation_path = _contained_path(store, manifest.get("observations_path"))
            if not observation_path.is_file():
                if int(manifest.get("observation_count") or 0) == 0:
                    continue
                raise ValueError("normalized observation file is missing")
            with gzip.open(observation_path, "rb") as stream:
                body = stream.read(MAX_NORMALIZED_BYTES_PER_FIXTURE + 1)
            if len(body) > MAX_NORMALIZED_BYTES_PER_FIXTURE:
                raise ValueError("normalized observation file exceeds the dashboard read limit")
            decoded = body.decode("utf-8")
            records = [_observation_from_dict(json.loads(line)) for line in decoded.splitlines() if line.strip()]
            if len(records) != int(manifest.get("observation_count") or 0):
                raise ValueError("normalized observation count differs from its manifest")
            for row in records:
                validate_observation(row)
                if row.source_id != source or row.fixture_id != fixture:
                    raise ValueError("observation identity differs from its fixture manifest")
            observations.extend(records)
        except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError, ValueError,
                KeyError, TypeError, gzip.BadGzipFile) as exc:
            warnings.append(
                f"Skipped fixture revision {manifest_path.name}: {type(exc).__name__}: {exc}"
            )

    if not observations:
        state = "NO_ANALYZABLE_OBSERVATIONS" if latest else "NO_LOCAL_DATASET"
        return DashboardSnapshot(
            str(store), state, len(latest), len(latest), 0, 0, statuses, last_success,
            None, tuple(warnings),
        )
    try:
        result = analyze_observations(observations)
    except (TypeError, ValueError) as exc:
        warnings.append(f"Analysis validation failed: {type(exc).__name__}: {exc}")
        return DashboardSnapshot(
            str(store), "ANALYSIS_FAILED", len(latest), len(latest), len(observations),
            0, statuses, last_success, None, tuple(warnings),
        )
    state = "PARTIAL" if warnings else "READY"
    return DashboardSnapshot(
        str(store), state, len(latest), result.fixture_count, result.observation_count,
        result.analyzed_player_count, statuses, last_success, result, tuple(warnings),
        tuple(observations),
    )
