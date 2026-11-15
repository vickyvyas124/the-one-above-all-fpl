"""Versioned records for source observations and future intelligence claims.

Unavailable measurements remain ``None``. A numeric zero is reserved for a
measured absence (for example, a complete event feed with no shots).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping


CONTRACT_VERSION = "1.0.0"


class CollectionStatus(str, Enum):
    MATCH_NOT_COMPLETED = "match_not_completed"
    MATCH_COMPLETED_STATS_UNAVAILABLE = "match_completed_stats_unavailable"
    DATA_COLLECTED = "data_successfully_collected"
    DATA_INCOMPLETE = "data_incomplete"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"


class EvidenceClass(str, Enum):
    OBSERVED = "OBSERVED"
    INFERRED = "INFERRED"
    REPORTED = "REPORTED"
    UNVERIFIED = "UNVERIFIED"


class EvidenceQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INCOMPLETE = "incomplete"


COUNT_FIELDS = (
    "shots", "shots_on_target", "key_passes", "chances_created",
    "touches_attacking_area", "touches_in_box", "progressive_carries",
    "progressive_passes", "defensive_actions",
)
NUMERIC_FIELDS = ("minutes", "xg", "xa")
REQUIRED_IDENTITY_FIELDS = (
    "source_id", "source_player_id", "fixture_id", "season", "match_date",
    "player_name", "club", "opponent",
)


@dataclass(frozen=True)
class PlayerMatchObservation:
    schema_version: str
    source_id: str
    source_player_id: str
    fpl_player_id: int | None
    season: str
    gameweek: int | None
    competition_round: str | None
    fixture_id: str
    match_date: str
    player_name: str
    club: str
    opponent: str
    position: str | None
    minutes: float | None
    started: bool | None
    substitute: bool | None
    shots: int | None
    shots_on_target: int | None
    xg: float | None
    xa: float | None
    key_passes: int | None
    chances_created: int | None
    touches_attacking_area: int | None
    touches_in_box: int | None
    progressive_carries: int | None
    progressive_passes: int | None
    defensive_actions: int | None
    role_segments: tuple[Mapping[str, Any], ...]
    source_timestamp: str | None
    retrieved_at_utc: str
    source_url: str
    raw_bundle_sha256: str
    completeness: EvidenceQuality
    missing_fields: tuple[str, ...] = ()
    metric_definitions: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["completeness"] = self.completeness.value
        result["role_segments"] = [dict(segment) for segment in self.role_segments]
        result["metric_definitions"] = dict(self.metric_definitions)
        return result


@dataclass(frozen=True)
class IntelligenceInsight:
    """Future claim contract; Phase 1 emits observations, not insights."""

    schema_version: str
    insight_id: str
    classification: EvidenceClass
    evidence_quality: EvidenceQuality
    source_id: str
    source_timestamp: str | None
    retrieved_at_utc: str
    fixture_ids: tuple[str, ...]
    source_player_ids: tuple[str, ...]
    fpl_player_id: int | None
    measurements: Mapping[str, Any]
    comparison_period: str
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["classification"] = self.classification.value
        result["evidence_quality"] = self.evidence_quality.value
        return result


def validate_observation(observation: PlayerMatchObservation) -> None:
    """Fail closed on identity, provenance, or measured-value inconsistencies."""
    if observation.schema_version != CONTRACT_VERSION:
        raise ValueError("Unsupported football observation schema version.")
    for field_name in REQUIRED_IDENTITY_FIELDS:
        value = getattr(observation, field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Observation is missing required identity field: {field_name}.")
    if observation.gameweek is not None and observation.gameweek < 1:
        raise ValueError("A known Gameweek must be a positive integer.")
    if observation.fpl_player_id is not None and observation.fpl_player_id < 1:
        raise ValueError("A known FPL player ID must be positive.")
    if len(observation.raw_bundle_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in observation.raw_bundle_sha256
    ):
        raise ValueError("Observation provenance requires a lowercase SHA-256 digest.")
    for field_name in NUMERIC_FIELDS:
        value = getattr(observation, field_name)
        if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
            raise ValueError(f"Measured field {field_name} must be unknown or a finite nonnegative value.")
    for field_name in COUNT_FIELDS:
        value = getattr(observation, field_name)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"Measured count {field_name} must be unknown or a nonnegative integer.")
    for name in observation.missing_fields:
        if name not in {"position", *NUMERIC_FIELDS, *COUNT_FIELDS, "started", "substitute"}:
            raise ValueError(f"Unknown missing-field marker: {name}.")
        if getattr(observation, name) is not None:
            raise ValueError(f"Field {name} is marked missing but has a value.")

