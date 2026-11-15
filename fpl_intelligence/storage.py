"""Local, immutable, content-addressed storage for raw and normalized data."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping

from .contract import CONTRACT_VERSION, PlayerMatchObservation, validate_observation
from .sources.statsbomb_open_data import MatchBundle


class StoreError(RuntimeError):
    """The intelligence store could not safely persist or verify a record."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _safe_component(value: object) -> str:
    text = str(value)
    if not text or text in {".", ".."} or re.fullmatch(r"[A-Za-z0-9._-]+", text) is None:
        raise StoreError(f"Unsafe intelligence-store path component: {text!r}")
    return text


def _write_immutable(path: Path, body: bytes) -> bool:
    """Write atomically; return False when an identical immutable blob exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != body:
            raise StoreError(f"Immutable path already exists with different contents: {path}")
        return False
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # A same-directory rename is atomic and supported on Windows; hard
            # links are disabled in some sandboxed and managed environments.
            os.rename(temp_name, path)
            created = True
        except FileExistsError:
            if path.read_bytes() != body:
                raise StoreError(f"Concurrent writer created a conflicting immutable path: {path}")
            created = False
        return created
    except OSError as exc:
        raise StoreError(f"Could not persist immutable intelligence data at {path}: {exc}") from exc
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def bundle_digest(bundle: MatchBundle) -> str:
    metadata = {name: document.sha256 for name, document in sorted(bundle.documents.items())}
    if not metadata:
        raise StoreError("Cannot persist a bundle without at least one raw source response.")
    return hashlib.sha256(_canonical(metadata)).hexdigest()


def persist_bundle(root: str | Path, bundle: MatchBundle,
                   observations: Iterable[PlayerMatchObservation], *,
                   status: str, warnings: Iterable[str] = ()) -> dict[str, object]:
    """Store exact source bytes and one immutable normalized revision per input hash.

    No file in the store is written into a Git-managed directory by this module;
    callers choose an explicit local path, and the repository ignores its default
    ``.local/football-intelligence`` location.
    """
    root = Path(root).expanduser().resolve()
    source = _safe_component(bundle.source_id)
    competition = _safe_component(bundle.competition_id)
    season = _safe_component(bundle.season_id)
    fixture = _safe_component(bundle.match_id)
    rows = tuple(sorted(observations, key=lambda item: item.source_player_id))
    for row in rows:
        validate_observation(row)
        if row.source_id != source or row.fixture_id != fixture:
            raise StoreError("Observation identity does not match the raw source bundle.")

    raw_manifest = {}
    for name, document in sorted(bundle.documents.items()):
        key = _safe_component(name)
        actual_hash = hashlib.sha256(document.body).hexdigest()
        if actual_hash != document.sha256:
            raise StoreError(f"Raw {key} payload does not match its declared SHA-256.")
        raw_path = root / "raw" / source / actual_hash[:2] / f"{actual_hash}.json.gz"
        _write_immutable(raw_path, gzip.compress(document.body, mtime=0))
        raw_manifest[key] = {
            "url": document.url, "sha256": actual_hash,
            "retrieved_at_utc": document.retrieved_at_utc,
            "source_timestamp": document.source_timestamp,
            "store_path": raw_path.relative_to(root).as_posix(),
        }

    digest = bundle_digest(bundle)
    revision = hashlib.sha256(f"{CONTRACT_VERSION}:{digest}".encode("ascii")).hexdigest()
    base = root / "derived" / source / competition / season / fixture
    observation_path = base / f"{revision}.observations.jsonl.gz"
    manifest_path = base / f"{revision}.manifest.json"
    serialized = b"".join(_canonical(row.to_dict()) + b"\n" for row in rows)
    _write_immutable(observation_path, gzip.compress(serialized, mtime=0))
    manifest = {
        "schema_version": CONTRACT_VERSION,
        "source_id": source,
        "adapter_version": "1.0.0",
        "competition_id": bundle.competition_id,
        "season_id": bundle.season_id,
        "fixture_id": bundle.match_id,
        "source_timestamp": bundle.source_timestamp,
        "bundle_sha256": digest,
        "revision_id": revision,
        "collection_status": str(status),
        "raw_documents": raw_manifest,
        "observation_count": len(rows),
        "observations_path": observation_path.relative_to(root).as_posix(),
        "warnings": sorted(set(str(item) for item in warnings)),
    }
    manifest_body = _canonical(manifest) + b"\n"
    created = _write_immutable(manifest_path, manifest_body)
    return {"duplicate": not created, "manifest_path": str(manifest_path),
            "observations_path": str(observation_path), "bundle_sha256": digest,
            "revision_id": revision, "observation_count": len(rows), "manifest": manifest}


def list_fixture_revisions(root: str | Path, *, competition_id: int, season_id: int,
                           fixture_id: int, source_id: str = "statsbomb_open_data") -> list[dict[str, object]]:
    """List immutable revisions for one source fixture, newest files last by hash."""
    base = (Path(root).expanduser().resolve() / "derived" / _safe_component(source_id)
            / _safe_component(competition_id) / _safe_component(season_id) / _safe_component(fixture_id))
    revisions = []
    if not base.exists():
        return revisions
    for path in sorted(base.glob("*.manifest.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StoreError(f"An existing fixture manifest is unreadable: {path}") from exc
        if value.get("fixture_id") != fixture_id or value.get("schema_version") != CONTRACT_VERSION:
            raise StoreError(f"An existing fixture manifest has mismatched identity: {path}")
        revisions.append(value)
    return revisions
