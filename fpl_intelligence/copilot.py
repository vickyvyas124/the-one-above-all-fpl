"""Private, append-only observation notes and deterministic evidence verdicts.

This module is deliberately separate from the production FPL model. It does
not infer identity from names, fetch FPL data, or alter a forecast.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Mapping

from .contract import PlayerMatchObservation


METRICS = frozenset({
    "minutes", "shots", "shots_on_target", "xg", "xa", "key_passes",
    "chances_created", "touches_attacking_area", "touches_in_box",
    "progressive_carries", "progressive_passes", "defensive_actions",
})
OPERATORS = frozenset({"at_least", "at_most", "greater_than", "less_than", "equals"})
_VERDICTS = frozenset({"AGREE", "PARTIALLY_AGREE", "DISAGREE", "INSUFFICIENT_DATA"})


class CopilotStoreError(RuntimeError):
    """A note or identity record could not be safely read or written."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _record_id(value: object) -> str:
    return hashlib.sha256(_canonical(value) + b"\n").hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_record(root: str | Path, collection: str, record: Mapping[str, Any]) -> str:
    """Persist one content-addressed record without ever replacing an existing record."""
    if re.fullmatch(r"[a-z_]+", collection) is None:
        raise CopilotStoreError("Invalid record collection.")
    body = _canonical(dict(record)) + b"\n"
    record_id = hashlib.sha256(body).hexdigest()
    directory = Path(root).expanduser().resolve() / "copilot" / collection
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record_id}.json"
    if path.exists():
        if path.read_bytes() != body:
            raise CopilotStoreError("A content-addressed record path contains different data.")
        return record_id
    descriptor, temp_name = tempfile.mkstemp(prefix=".record-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.rename(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise CopilotStoreError("Concurrent writer created a conflicting record.")
    except OSError as exc:
        raise CopilotStoreError(f"Could not persist {collection} record: {exc}") from exc
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return record_id


def _read_records(root: str | Path, collection: str) -> list[dict[str, Any]]:
    directory = Path(root).expanduser().resolve() / "copilot" / collection
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != path.stem:
                raise ValueError("record hash mismatch")
            record = json.loads(body)
            if not isinstance(record, dict):
                raise ValueError("record is not an object")
            records.append(record)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CopilotStoreError(f"Invalid immutable {collection} record {path.name}: {exc}") from exc
    return records


def record_verified_mapping(
    root: str | Path, *, source_id: str, season: str, source_player_id: str,
    fpl_player_id: int, source_player_name: str, fpl_player_name: str,
    verified_by: str, evidence: str, confirmed: bool,
) -> str:
    """Record an explicitly human-verified, season-specific identity link.

    Names are retained for audit display only; they are never used to resolve a
    mapping. Conflicting mappings make the identity unresolved until corrected.
    """
    if confirmed is not True:
        raise ValueError("The mapping must be explicitly confirmed by a human reviewer.")
    strings = (source_id, season, source_player_id, source_player_name, fpl_player_name,
               verified_by, evidence)
    if any(not isinstance(value, str) or not value.strip() for value in strings):
        raise ValueError("Mapping source, season, names, reviewer, and verification evidence are required.")
    if type(fpl_player_id) is not int or fpl_player_id < 1:
        raise ValueError("A positive exact FPL player ID is required.")
    source_id, season, source_player_id = source_id.strip(), season.strip(), source_player_id.strip()
    source_player_name, fpl_player_name = source_player_name.strip(), fpl_player_name.strip()
    verified_by, evidence = verified_by.strip(), evidence.strip()
    key = (source_id, season, source_player_id)
    for existing in _read_records(root, "mappings"):
        if (existing.get("source_id"), existing.get("season"), existing.get("fpl_player_id")) == (
            source_id, season, fpl_player_id,
        ) and existing.get("source_player_id") != source_player_id:
            raise ValueError("FPL player already has a different source identity in this season; mapping rejected.")
        if (existing.get("source_id"), existing.get("season"), existing.get("source_player_id")) == key:
            if (existing.get("fpl_player_id") == fpl_player_id
                    and existing.get("source_player_name") == source_player_name
                    and existing.get("fpl_player_name") == fpl_player_name):
                return _record_id(existing)
            raise ValueError("Conflicting mapping or identity evidence already exists; this identity remains unresolved.")
    record = {
        "schema_version": "1.0", "source_id": source_id, "season": season,
        "source_player_id": source_player_id, "fpl_player_id": fpl_player_id,
        "source_player_name": source_player_name, "fpl_player_name": fpl_player_name,
        "verified_by": verified_by, "verification_evidence": evidence,
        "verified_at_utc": _utc_now(), "status": "VERIFIED",
    }
    return _write_record(root, "mappings", record)


def resolve_verified_mapping(root: str | Path, *, source_id: str, season: str,
                             source_player_id: str) -> int | None:
    matches = [item for item in _read_records(root, "mappings") if
               item.get("source_id") == source_id and item.get("season") == season and
               item.get("source_player_id") == str(source_player_id) and item.get("status") == "VERIFIED"]
    ids = {item.get("fpl_player_id") for item in matches}
    return next(iter(ids)) if len(ids) == 1 else None


def record_user_observation(
    root: str | Path, *, text: str, author: str, season: str,
    fpl_player_id: int | None = None, metric: str | None = None,
    operator: str | None = None, threshold: float | None = None,
) -> str:
    """Save a human observation and optional structured claim immutably."""
    if not isinstance(text, str) or not text.strip() or not isinstance(author, str) or not author.strip():
        raise ValueError("Observation text and author are required.")
    if not isinstance(season, str) or not season.strip():
        raise ValueError("Season is required; use an explicit source season label.")
    structured = any(value is not None for value in (metric, operator, threshold, fpl_player_id))
    claim = None
    if structured:
        if type(fpl_player_id) is not int or fpl_player_id < 1:
            raise ValueError("A structured claim requires an exact positive FPL player ID.")
        if metric not in METRICS or operator not in OPERATORS:
            raise ValueError("Choose a supported measured metric and comparison operator.")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or threshold < 0:
            raise ValueError("Claim threshold must be a finite nonnegative number.")
        if not math.isfinite(float(threshold)):
            raise ValueError("Claim threshold must be finite.")
        claim = {"fpl_player_id": fpl_player_id, "metric": metric,
                 "operator": operator, "threshold": float(threshold)}
    record = {
        "schema_version": "1.0", "created_at_utc": _utc_now(), "author": author.strip(),
        "text": text.strip(), "season": season.strip(), "claim": claim,
        "classification": "USER_OPINION",
    }
    return _write_record(root, "notes", record)


def _matches(value: float, operator: str, threshold: float) -> bool:
    return {
        "at_least": value >= threshold, "at_most": value <= threshold,
        "greater_than": value > threshold, "less_than": value < threshold,
        "equals": value == threshold,
    }[operator]


def evaluate_user_observation(
    root: str | Path, note_id: str, observations: Iterable[PlayerMatchObservation],
) -> dict[str, Any]:
    """Evaluate a structured user claim against exact-season verified source evidence."""
    notes = {_record_id(item): item for item in _read_records(root, "notes")}
    note = notes.get(note_id)
    if note is None:
        raise ValueError("Observation note was not found in the local immutable store.")
    claim = note.get("claim")
    verdict = "INSUFFICIENT_DATA"
    checked: list[dict[str, Any]] = []
    if isinstance(claim, dict):
        for row in observations:
            if row.season != note["season"] or row.source_id == "":
                continue
            mapped = resolve_verified_mapping(
                root, source_id=row.source_id, season=row.season,
                source_player_id=row.source_player_id,
            )
            if mapped != claim["fpl_player_id"]:
                continue
            measured = getattr(row, claim["metric"], None)
            if measured is None:
                checked.append({"fixture_id": row.fixture_id, "value": None,
                                "evidence_sha256": row.raw_bundle_sha256})
            else:
                checked.append({"fixture_id": row.fixture_id, "value": float(measured),
                                "matches": _matches(float(measured), claim["operator"], claim["threshold"]),
                                "evidence_sha256": row.raw_bundle_sha256})
        known = [item for item in checked if item["value"] is not None]
        outcomes = [item["matches"] for item in known]
        if outcomes and len(outcomes) == len(checked) and all(outcomes):
            verdict = "AGREE"
        elif outcomes and len(outcomes) == len(checked) and not any(outcomes):
            verdict = "DISAGREE"
        elif outcomes:
            verdict = "PARTIALLY_AGREE"
    result = {
        "schema_version": "1.0", "note_id": note_id, "evaluated_at_utc": _utc_now(),
        "channel": "AI_RULE_BASED", "verdict": verdict,
        "evidence": checked, "evidence_count": len(checked),
        "reason": ("Deterministic comparison of the structured claim with verified, season-matched source metrics."
                   if checked else "No source observations matched an explicitly verified season-specific player mapping."),
    }
    result_id = _write_record(root, "evaluations", result)
    return {"evaluation_id": result_id, **result}


def list_user_observations(root: str | Path) -> list[dict[str, Any]]:
    return _read_records(root, "notes")


def list_evaluations(root: str | Path) -> list[dict[str, Any]]:
    return _read_records(root, "evaluations")


def record_user_assessment(root: str | Path, *, note_id: str, verdict: str) -> str:
    """Append the manager's own evidence assessment, separate from the rule verdict."""
    if verdict not in _VERDICTS:
        raise ValueError("Unsupported USER assessment.")
    notes = {_record_id(item) for item in _read_records(root, "notes")}
    if note_id not in notes:
        raise ValueError("Observation note was not found in the local immutable store.")
    return _write_record(root, "evaluations", {
        "schema_version": "1.0", "note_id": note_id, "evaluated_at_utc": _utc_now(),
        "channel": "USER", "verdict": verdict, "evidence": [],
        "reason": "Manager-entered assessment; kept separate from source-derived verdicts.",
    })


def combine_assessments(root: str | Path, *, note_id: str, user_verdict: str,
                        ai_verdict: str, user_assessment_id: str,
                        ai_assessment_id: str) -> dict[str, Any]:
    """Append a transparent combined assessment; disagreement never hides either input."""
    if user_verdict not in _VERDICTS or ai_verdict not in _VERDICTS:
        raise ValueError("Unsupported USER or AI assessment.")
    notes = {_record_id(item) for item in _read_records(root, "notes")}
    if note_id not in notes:
        raise ValueError("Observation note was not found in the local immutable store.")
    evaluations = {_record_id(item): item for item in _read_records(root, "evaluations")}
    user_record, ai_record = evaluations.get(user_assessment_id), evaluations.get(ai_assessment_id)
    if (user_record is None or user_record.get("note_id") != note_id or user_record.get("channel") != "USER"
            or user_record.get("verdict") != user_verdict):
        raise ValueError("USER assessment reference does not match the submitted assessment.")
    if (ai_record is None or ai_record.get("note_id") != note_id or ai_record.get("channel") != "AI_RULE_BASED"
            or ai_record.get("verdict") != ai_verdict):
        raise ValueError("AI assessment reference does not match the submitted assessment.")
    if user_verdict == "INSUFFICIENT_DATA" or ai_verdict == "INSUFFICIENT_DATA":
        combined = "INSUFFICIENT_DATA"
    elif user_verdict == ai_verdict:
        combined = user_verdict
    else:
        combined = "PARTIALLY_AGREE"
    record = {
        "schema_version": "1.0", "note_id": note_id, "evaluated_at_utc": _utc_now(),
        "channel": "COMBINED", "verdict": combined,
        "user_verdict": user_verdict, "ai_verdict": ai_verdict,
        "user_assessment_id": user_assessment_id, "ai_assessment_id": ai_assessment_id,
        "reason": "Combined result preserves the independent USER and AI_RULE_BASED assessments.",
    }
    record_id = _write_record(root, "evaluations", record)
    return {"evaluation_id": record_id, **record}
