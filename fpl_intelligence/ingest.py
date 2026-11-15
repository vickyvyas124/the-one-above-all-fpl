"""Idempotent orchestration for one source fixture."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import CollectionStatus
from .sources.statsbomb_open_data import MatchBundle, StatsBombOpenDataAdapter, SourceDataError
from .storage import persist_bundle


@dataclass(frozen=True)
class IngestResult:
    status: CollectionStatus
    observations: int
    duplicate: bool
    manifest_path: str | None
    warnings: tuple[str, ...] = ()


def collect_match(adapter: StatsBombOpenDataAdapter, store_root: str | Path, *,
                  competition_id: int, season_id: int, match_id: int) -> IngestResult:
    """Fetch, validate, normalize and persist one StatsBomb match.

    Completed matches without published event or lineup files are recorded as
    unavailable/incomplete. Non-completed matches are not represented as zeros.
    """
    bundle = adapter.fetch_match_bundle(competition_id, season_id, match_id)
    match = bundle.match or {}
    match_status = str(match.get("match_status") or "").strip().lower()
    if match_status not in {"available", "completed"}:
        receipt = persist_bundle(store_root, bundle, (), status=CollectionStatus.MATCH_NOT_COMPLETED.value)
        return IngestResult(CollectionStatus.MATCH_NOT_COMPLETED, 0, bool(receipt["duplicate"]),
                            str(receipt["manifest_path"]))
    if bundle.events is None and bundle.lineups is None:
        receipt = persist_bundle(store_root, bundle, (),
                                 status=CollectionStatus.MATCH_COMPLETED_STATS_UNAVAILABLE.value,
                                 warnings=("Both event and lineup payloads are unavailable.",))
        return IngestResult(CollectionStatus.MATCH_COMPLETED_STATS_UNAVAILABLE, 0,
                            bool(receipt["duplicate"]), str(receipt["manifest_path"]),
                            ("Both event and lineup payloads are unavailable.",))
    if bundle.events is None or bundle.lineups is None:
        warning = "One of the event or lineup payloads is unavailable; no player observations were inferred."
        receipt = persist_bundle(store_root, bundle, (), status=CollectionStatus.DATA_INCOMPLETE.value,
                                 warnings=(warning,))
        return IngestResult(CollectionStatus.DATA_INCOMPLETE, 0, bool(receipt["duplicate"]),
                            str(receipt["manifest_path"]), (warning,))
    try:
        observations = adapter.normalize_match(bundle)
    except (SourceDataError, ValueError, KeyError, TypeError) as exc:
        warning = f"Normalization failed: {type(exc).__name__}: {exc}"
        receipt = persist_bundle(store_root, bundle, (), status=CollectionStatus.ANALYSIS_FAILED.value,
                                 warnings=(warning,))
        return IngestResult(CollectionStatus.ANALYSIS_FAILED, 0, bool(receipt["duplicate"]),
                            str(receipt["manifest_path"]), (warning,))
    incomplete = any(row.missing_fields for row in observations)
    status = CollectionStatus.DATA_INCOMPLETE if incomplete else CollectionStatus.DATA_COLLECTED
    warnings = ("Some optional measurements are unknown; see each observation's missing_fields.",) if incomplete else ()
    receipt = persist_bundle(store_root, bundle, observations, status=status.value, warnings=warnings)
    return IngestResult(status, len(observations), bool(receipt["duplicate"]),
                        str(receipt["manifest_path"]), warnings)
