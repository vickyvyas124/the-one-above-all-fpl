"""Time-safe bridge between football insights and existing FPL decisions.

This adapter is diagnostic only. Until a candidate adjustment is validated on
chronological, point-in-time matched data, it always returns the existing FPL
baseline as the selected forecast.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable

from .contract import (
    EvidenceClass,
    EvidenceQuality,
    IntelligenceInsight,
)


BASELINE_FEATURE_OVERLAPS = {
    "xg_per90": ("xGI_per90",),
    "xa_per90": ("xGI_per90",),
    "minutes_per_match": ("minutes_and_expected_minutes",),
    "start_rate": ("starts_and_appearance_rate",),
}


@dataclass(frozen=True)
class ContextSignal:
    insight_id: str
    player_id: int
    source_id: str
    source_player_ids: tuple[str, ...]
    fixture_ids: tuple[str, ...]
    indicator: str
    classification: EvidenceClass
    evidence_quality: EvidenceQuality
    trend: str
    baseline_overlap: tuple[str, ...]
    measurements: dict[str, object]
    can_change_forecast: bool = False


@dataclass(frozen=True)
class ExcludedSignal:
    insight_id: str
    reason: str


@dataclass(frozen=True)
class FplDecisionContext:
    fpl_player_id: int
    target_deadline_utc: str
    baseline_points: float | None
    intelligence_adjusted_points: float | None
    selected_points: float | None
    official_ep_next: float | None
    actual_points: float | None
    status: str
    recommendation_changed: bool
    signals: tuple[ContextSignal, ...]
    excluded_signals: tuple[ExcludedSignal, ...]
    gate_reason: str


def _number_or_none(value: object, *, minimum: float | None = None) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        return None
    return result


def _utc_timestamp(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _trend(insight: IntelligenceInsight) -> str:
    delta = _number_or_none(insight.measurements.get("absolute_change"))
    if delta is None:
        return "unclassified"
    if delta == 0:
        return "unchanged"
    higher_is_better = insight.measurements.get("higher_is_better")
    if higher_is_better is not True:
        return "increased" if delta > 0 else "decreased"
    positive = delta > 0
    return "positive_trend" if positive else "negative_trend"


def build_fpl_decision_context(
    *,
    fpl_player_id: int,
    target_deadline_utc: str,
    baseline_points: object,
    insights: Iterable[IntelligenceInsight],
    official_ep_next: object = None,
    actual_points: object = None,
    actual_available_at_utc: str | None = None,
) -> FplDecisionContext:
    """Attach eligible evidence to an FPL baseline without changing its score.

    Evidence must have an exact FPL ID, valid source/retrieval timestamps no
    later than the target deadline, a supported classification, and at least
    medium sample quality. Missing or post-deadline evidence is excluded.
    Baseline-overlapping indicators remain visible but cannot be counted twice.
    """
    try:
        normalized_player_id = int(fpl_player_id)
        exact_player_id = float(fpl_player_id) == normalized_player_id
    except (TypeError, ValueError, OverflowError):
        normalized_player_id, exact_player_id = 0, False
    if isinstance(fpl_player_id, bool) or not exact_player_id or normalized_player_id < 1:
        raise ValueError("A positive, verified FPL player ID is required.")
    deadline = _utc_timestamp(target_deadline_utc)
    if deadline is None:
        raise ValueError("Target deadline must be a timezone-aware ISO-8601 timestamp.")
    baseline = _number_or_none(baseline_points, minimum=0.0)
    benchmark = _number_or_none(official_ep_next, minimum=0.0)
    actual = None
    if actual_points is not None:
        actual_value = _number_or_none(actual_points)
        actual_time = _utc_timestamp(actual_available_at_utc)
        if actual_value is None or actual_time is None or actual_time <= deadline:
            raise ValueError("Actual points require a valid score available strictly after the target deadline.")
        actual = actual_value

    signals: list[ContextSignal] = []
    excluded: list[ExcludedSignal] = []
    seen_ids: set[str] = set()
    for insight in insights:
        insight_id = str(getattr(insight, "insight_id", "") or "")
        if not insight_id:
            excluded.append(ExcludedSignal("", "Insight has no stable ID."))
            continue
        if insight_id in seen_ids:
            raise ValueError(f"Duplicate intelligence insight ID: {insight_id}.")
        seen_ids.add(insight_id)
        if insight.fpl_player_id != normalized_player_id:
            excluded.append(ExcludedSignal(insight_id, "No exact verified FPL player-ID match."))
            continue
        if insight.classification == EvidenceClass.UNVERIFIED:
            excluded.append(ExcludedSignal(insight_id, "UNVERIFIED evidence cannot enter FPL context."))
            continue
        if insight.classification not in {EvidenceClass.OBSERVED, EvidenceClass.INFERRED}:
            excluded.append(ExcludedSignal(insight_id, "Evidence classification is not eligible for player context."))
            continue
        if insight.evidence_quality not in {EvidenceQuality.HIGH, EvidenceQuality.MEDIUM}:
            excluded.append(ExcludedSignal(insight_id, "Evidence quality is too low or incomplete."))
            continue
        source_time = _utc_timestamp(insight.source_timestamp)
        retrieved_time = _utc_timestamp(insight.retrieved_at_utc)
        if source_time is None or retrieved_time is None:
            excluded.append(ExcludedSignal(insight_id, "Source or retrieval time is missing or unverified."))
            continue
        if source_time > deadline or retrieved_time > deadline:
            excluded.append(ExcludedSignal(insight_id, "Evidence was published or retrieved after the target deadline."))
            continue
        if not insight.source_player_ids or not insight.fixture_ids:
            excluded.append(ExcludedSignal(insight_id, "Insight is missing player or fixture provenance."))
            continue
        indicator = str(insight.measurements.get("indicator") or "unknown")
        signals.append(ContextSignal(
            insight_id=insight_id,
            player_id=normalized_player_id,
            source_id=insight.source_id,
            source_player_ids=tuple(insight.source_player_ids),
            fixture_ids=tuple(insight.fixture_ids),
            indicator=indicator,
            classification=insight.classification,
            evidence_quality=insight.evidence_quality,
            trend=_trend(insight),
            baseline_overlap=BASELINE_FEATURE_OVERLAPS.get(indicator, ()),
            measurements=dict(insight.measurements),
        ))

    signals.sort(key=lambda item: (item.indicator, item.insight_id))
    excluded.sort(key=lambda item: (item.insight_id, item.reason))
    if baseline is None:
        status = "BASELINE_FORECAST_UNAVAILABLE"
        gate_reason = "No valid existing FPL baseline forecast was supplied."
    elif signals:
        status = "CONTEXT_ONLY_UNCALIBRATED"
        gate_reason = (
            "No candidate intelligence adjustment has repeatable chronological out-of-sample validation "
            "on point-in-time matched Premier League data. Existing baseline features also overlap xG/xA, "
            "minutes, and starts."
        )
    else:
        status = "BASELINE_ONLY_NO_ELIGIBLE_CONTEXT"
        gate_reason = "No time-safe, exactly matched intelligence evidence passed the context gates."

    return FplDecisionContext(
        fpl_player_id=normalized_player_id,
        target_deadline_utc=deadline.isoformat(),
        baseline_points=baseline,
        intelligence_adjusted_points=None,
        selected_points=baseline,
        official_ep_next=benchmark,
        actual_points=actual,
        status=status,
        recommendation_changed=False,
        signals=tuple(signals),
        excluded_signals=tuple(excluded),
        gate_reason=gate_reason,
    )
