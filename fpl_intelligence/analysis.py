"""Exploratory, evidence-linked player trend analysis.

This module reports descriptive changes only. It does not produce FPL points,
probabilities, recommendations, or a weighted player score.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from statistics import fmean
from typing import Iterable, Mapping

from .contract import (
    CONTRACT_VERSION,
    EvidenceClass,
    EvidenceQuality,
    IntelligenceInsight,
    PlayerMatchObservation,
    validate_observation,
)


@dataclass(frozen=True)
class MetricSpec:
    field: str
    label: str
    unit: str
    per90: bool = False
    higher_is_better: bool = False
    domain: str = "involvement"


TREND_METRICS = (
    MetricSpec("minutes", "minutes_per_match", "minutes"),
    MetricSpec("started", "start_rate", "share", higher_is_better=True, domain="minutes"),
    MetricSpec("xg", "xg_per90", "per 90", per90=True, higher_is_better=True, domain="goal_threat"),
    MetricSpec("shots", "shots_per90", "per 90", per90=True, higher_is_better=True, domain="goal_threat"),
    MetricSpec("shots_on_target", "shots_on_target_per90", "per 90", per90=True,
               higher_is_better=True, domain="goal_threat"),
    MetricSpec("xa", "xa_per90", "per 90", per90=True, higher_is_better=True, domain="chance_creation"),
    MetricSpec("key_passes", "key_passes_per90", "per 90", per90=True, higher_is_better=True, domain="chance_creation"),
    MetricSpec("chances_created", "chances_created_per90", "per 90", per90=True,
               higher_is_better=True, domain="chance_creation"),
    MetricSpec("touches_attacking_area", "attacking_area_events_per90", "per 90", per90=True,
               higher_is_better=True, domain="box_involvement"),
    MetricSpec("touches_in_box", "box_events_per90", "per 90", per90=True, higher_is_better=True, domain="box_involvement"),
    MetricSpec("defensive_actions", "defensive_actions_per90", "per 90", per90=True, domain="defense"),
)
ATTACKING_METRICS = {
    "xg_per90", "shots_per90", "shots_on_target_per90", "xa_per90", "key_passes_per90",
    "chances_created_per90", "attacking_area_events_per90", "box_events_per90",
}


@dataclass(frozen=True)
class EmergingOpportunity:
    rank: int
    source_id: str
    source_player_id: str
    fpl_player_id: int | None
    player_name: str
    club: str
    season: str
    fixture_ids: tuple[str, ...]
    source_timestamp: str | None
    classification: EvidenceClass
    improving_dimensions: tuple[str, ...]
    trend_insight_ids: tuple[str, ...]
    evidence_quality: EvidenceQuality
    explanation: str


@dataclass(frozen=True)
class AnalysisResult:
    insights: tuple[IntelligenceInsight, ...]
    emerging_opportunities: tuple[EmergingOpportunity, ...]
    observation_count: int
    fixture_count: int
    analyzed_player_count: int
    skipped_players: Mapping[str, str]
    metric_coverage: Mapping[str, float]


def _metric_value(row: PlayerMatchObservation, spec: MetricSpec) -> float | None:
    value = getattr(row, spec.field)
    if value is None:
        return None
    if spec.field == "started":
        return float(bool(value))
    value = float(value)
    if not math.isfinite(value):
        return None
    if spec.per90:
        if row.minutes is None or row.minutes <= 0:
            return None
        return value * 90.0 / float(row.minutes)
    return value


def _dominant_position(rows: list[PlayerMatchObservation]) -> tuple[str | None, int, tuple[str, ...]]:
    known = [(row.fixture_id, row.position.strip()) for row in rows
             if row.position is not None and row.position.strip()]
    if not known:
        return None, 0, ()
    counts = Counter(position for _, position in known)
    dominant = sorted(counts, key=lambda name: (-counts[name], name.casefold()))[0]
    fixture_ids = tuple(fixture for fixture, position in known if position == dominant)
    return dominant, len(known), fixture_ids


def _insight_id(source_id: str, player_id: str, season: str, kind: str,
                metric: str, fixture_ids: tuple[str, ...]) -> str:
    payload = json.dumps([source_id, player_id, season, kind, metric, fixture_ids],
                         separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{source_id}:{player_id}:{season}:{kind}:{metric}:{digest}"


def _make_insight(*, source_id: str, player_id: str, fpl_player_id: int | None,
                  name: str, season: str, rows: list[PlayerMatchObservation],
                  metric: str, measurements: Mapping[str, object],
                  quality: EvidenceQuality, explanation: str,
                  fixture_ids: tuple[str, ...]) -> IntelligenceInsight:
    return IntelligenceInsight(
        schema_version=CONTRACT_VERSION,
        insight_id=_insight_id(source_id, player_id, season, "trend", metric, fixture_ids),
        classification=EvidenceClass.INFERRED,
        evidence_quality=quality,
        source_id=source_id,
        source_timestamp=max((row.source_timestamp for row in rows if row.source_timestamp), default=None),
        retrieved_at_utc=max(row.retrieved_at_utc for row in rows),
        fixture_ids=fixture_ids,
        source_player_ids=(player_id,),
        fpl_player_id=fpl_player_id,
        measurements={"player_name": name, "season": season, "indicator": metric, **dict(measurements)},
        comparison_period="latest measured matches versus preceding measured matches",
        explanation=explanation,
    )


def analyze_observations(
    observations: Iterable[PlayerMatchObservation],
    *,
    recent_matches: int = 3,
    baseline_matches: int = 3,
    minimum_matches_per_period: int = 2,
) -> AnalysisResult:
    """Create per-player trends from chronological, same-season, same-club rows.

    Emerging opportunities are exploratory candidates: at least two independently
    measured attacking domains must rise. The candidate ordering is transparent
    and unweighted (count of rising domains, then player name); it is not a
    learned score or a recommendation.
    """
    if (minimum_matches_per_period < 2 or recent_matches < minimum_matches_per_period
            or baseline_matches < minimum_matches_per_period):
        raise ValueError("Each configured comparison period must allow at least two matches.")

    rows = tuple(observations)
    for row in rows:
        validate_observation(row)
    keys = [(row.source_id, row.source_player_id, row.fixture_id) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate source/player/fixture observation; analysis stopped.")

    mapping_by_source_player: dict[tuple[str, str], set[int]] = defaultdict(set)
    mapped_player_by_source_id: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        if row.fpl_player_id is not None:
            mapping_by_source_player[(row.source_id, row.source_player_id)].add(row.fpl_player_id)
            mapped_player_by_source_id[(row.source_id, row.fpl_player_id)].add(row.source_player_id)
    if any(len(values) > 1 for values in mapping_by_source_player.values()):
        raise ValueError("A source player maps to multiple FPL IDs; analysis stopped.")
    if any(len(values) > 1 for values in mapped_player_by_source_id.values()):
        raise ValueError("Multiple source players map to one FPL ID; analysis stopped.")

    groups: dict[tuple[str, str, str, str], list[PlayerMatchObservation]] = defaultdict(list)
    for row in rows:
        groups[(row.source_id, row.source_player_id, row.season, row.club)].append(row)

    insights: list[IntelligenceInsight] = []
    emerging_rows: list[tuple[str, str, str, tuple[str, ...], tuple[str, ...],
                              EvidenceQuality, str, int | None, str, tuple[str, ...], str | None]] = []
    skipped: dict[str, str] = {}
    analyzed_players: set[tuple[str, str]] = set()

    for (source_id, player_id, season, club), player_rows in sorted(groups.items()):
        ordered = sorted(player_rows, key=lambda item: (item.match_date, item.fixture_id))
        group_key = f"{source_id}:{player_id}:{season}:{club}"
        recent_count = min(recent_matches, len(ordered) - minimum_matches_per_period)
        if recent_count < minimum_matches_per_period:
            skipped[group_key] = "Not enough same-season, same-club matches for both comparison periods."
            continue
        recent = ordered[-recent_count:]
        baseline_count = min(baseline_matches, len(ordered) - recent_count)
        baseline = ordered[-recent_count - baseline_count:-recent_count]
        if baseline_count < minimum_matches_per_period:
            skipped[group_key] = "Not enough baseline matches for the configured comparison."
            continue

        mapping = mapping_by_source_player.get((source_id, player_id), set())
        fpl_player_id = next(iter(mapping)) if mapping else None
        name = ordered[-1].player_name
        period_metrics: dict[str, dict[str, object]] = {}
        for spec in TREND_METRICS:
            baseline_values = [_metric_value(row, spec) for row in baseline]
            recent_values = [_metric_value(row, spec) for row in recent]
            baseline_valid = [value for value in baseline_values if value is not None]
            recent_valid = [value for value in recent_values if value is not None]
            if (len(baseline_valid) < minimum_matches_per_period
                    or len(recent_valid) < minimum_matches_per_period):
                continue
            baseline_mean = fmean(baseline_valid)
            recent_mean = fmean(recent_valid)
            delta = recent_mean - baseline_mean
            pct = (delta / abs(baseline_mean) * 100.0) if baseline_mean != 0 else None
            quality = (EvidenceQuality.HIGH
                       if len(baseline_valid) >= baseline_matches and len(recent_valid) >= recent_matches
                       else EvidenceQuality.MEDIUM)
            fixture_ids = tuple(row.fixture_id for row in baseline + recent
                                if _metric_value(row, spec) is not None)
            values = {
                "unit": spec.unit,
                "baseline_mean": round(baseline_mean, 6),
                "recent_mean": round(recent_mean, 6),
                "absolute_change": round(delta, 6),
                "percent_change": round(pct, 2) if pct is not None else None,
                "baseline_match_count": len(baseline_valid),
                "recent_match_count": len(recent_valid),
                "higher_is_better": spec.higher_is_better,
                "domain": spec.domain,
            }
            direction = "increased" if delta > 0 else "decreased" if delta < 0 else "was unchanged"
            explanation = (
                f"{spec.label} {direction} from {baseline_mean:.3f} to {recent_mean:.3f} {spec.unit} "
                f"across the measured comparison periods. This is an inferred trend, not a causal or "
                "future-performance claim."
            )
            insight = _make_insight(
                source_id=source_id, player_id=player_id, fpl_player_id=fpl_player_id,
                name=name, season=season, rows=ordered, metric=spec.label,
                measurements=values, quality=quality, explanation=explanation,
                fixture_ids=fixture_ids,
            )
            insights.append(insight)
            period_metrics[spec.label] = {
                "delta": delta, "higher_is_better": spec.higher_is_better,
                "domain": spec.domain, "quality": quality, "insight_id": insight.insight_id,
                "fixture_ids": fixture_ids,
            }

        baseline_minutes = [float(row.minutes) for row in baseline if row.minutes is not None]
        recent_minutes = [float(row.minutes) for row in recent if row.minutes is not None]
        if (len(baseline_minutes) >= minimum_matches_per_period
                and len(recent_minutes) >= minimum_matches_per_period):
            baseline_mean_minutes = fmean(baseline_minutes)
            recent_mean_minutes = fmean(recent_minutes)
            baseline_sd = math.sqrt(fmean((value - baseline_mean_minutes) ** 2
                                          for value in baseline_minutes))
            recent_sd = math.sqrt(fmean((value - recent_mean_minutes) ** 2
                                        for value in recent_minutes))
            quality = (EvidenceQuality.HIGH
                       if len(baseline_minutes) >= baseline_matches
                       and len(recent_minutes) >= recent_matches else EvidenceQuality.MEDIUM)
            fixture_ids = tuple(row.fixture_id for row in baseline + recent if row.minutes is not None)
            insights.append(_make_insight(
                source_id=source_id, player_id=player_id, fpl_player_id=fpl_player_id,
                name=name, season=season, rows=ordered, metric="minutes_variability",
                measurements={
                    "unit": "minutes standard deviation",
                    "baseline_standard_deviation": round(baseline_sd, 4),
                    "recent_standard_deviation": round(recent_sd, 4),
                    "change_in_standard_deviation": round(recent_sd - baseline_sd, 4),
                    "baseline_match_count": len(baseline_minutes),
                    "recent_match_count": len(recent_minutes),
                },
                quality=quality,
                explanation=(
                    "Reports the change in match-to-match minutes spread. It describes participation "
                    "variation only and does not establish injury, fatigue, or future starting probability."
                ),
                fixture_ids=fixture_ids,
            ))

        base_position, base_n, base_fixtures = _dominant_position(baseline)
        recent_position, recent_n, recent_fixtures = _dominant_position(recent)
        if (base_n >= minimum_matches_per_period and recent_n >= minimum_matches_per_period
                and base_position and recent_position and base_position != recent_position):
            fixture_ids = tuple(dict.fromkeys(base_fixtures + recent_fixtures))
            role_insight = _make_insight(
                source_id=source_id, player_id=player_id, fpl_player_id=fpl_player_id,
                name=name, season=season, rows=ordered, metric="listed_position_shift",
                measurements={
                    "baseline_dominant_position": base_position,
                    "recent_dominant_position": recent_position,
                    "baseline_known_match_count": base_n,
                    "recent_known_match_count": recent_n,
                },
                quality=EvidenceQuality.HIGH if base_n >= baseline_matches and recent_n >= recent_matches
                else EvidenceQuality.MEDIUM,
                explanation=(
                    f"The source-listed dominant position changed from {base_position} in the baseline "
                    f"to {recent_position} in the recent period. This describes lineup position labels; "
                    "it does not establish a tactical role change."
                ),
                fixture_ids=fixture_ids,
            )
            insights.append(role_insight)

        rising_metrics = tuple(sorted(
            metric for metric, value in period_metrics.items()
            if metric in ATTACKING_METRICS and float(value["delta"]) > 0
        ))
        rising_domains = tuple(sorted({str(period_metrics[metric]["domain"]) for metric in rising_metrics}))
        if len(rising_domains) >= 2:
            trend_ids = tuple(str(period_metrics[metric]["insight_id"]) for metric in rising_metrics)
            evidence_fixtures = tuple(sorted({
                fixture for metric in rising_metrics
                for fixture in period_metrics[metric]["fixture_ids"]
            }))
            source_timestamp = max(
                (row.source_timestamp for row in ordered if row.source_timestamp), default=None,
            )
            quality = (EvidenceQuality.HIGH if all(
                period_metrics[metric]["quality"] == EvidenceQuality.HIGH for metric in rising_metrics
            ) else EvidenceQuality.MEDIUM)
            emerging_rows.append((source_id, name, player_id, rising_domains, trend_ids,
                                  quality, club, fpl_player_id, season, evidence_fixtures,
                                  source_timestamp))
        analyzed_players.add((source_id, player_id))

    # Ranking is deliberately unweighted: number of rising attacking domains,
    # followed by stable name/source-ID tie breaks. No composite performance score.
    emerging_rows.sort(key=lambda item: (-len(item[3]), item[1].casefold(), item[2]))
    opportunities = tuple(
        EmergingOpportunity(
            rank=index + 1, source_id=source_id,
            source_player_id=player_id, fpl_player_id=fpl_id, player_name=name, club=club,
            season=season, fixture_ids=fixture_ids, source_timestamp=source_timestamp,
            classification=EvidenceClass.INFERRED,
            improving_dimensions=dimensions, trend_insight_ids=insight_ids,
            evidence_quality=quality,
            explanation=(
                f"{len(dimensions)} separately reported attacking domains increased: "
                f"{', '.join(dimensions)}. Exploratory shortlist only; this is not a transfer or points recommendation."
            ),
        )
        for index, (source_id, name, player_id, dimensions, insight_ids, quality, club,
                    fpl_id, season, fixture_ids, source_timestamp)
        in enumerate(emerging_rows)
    )

    coverage: dict[str, float] = {}
    if rows:
        for spec in TREND_METRICS:
            coverage[spec.label] = round(
                sum(_metric_value(row, spec) is not None for row in rows) / len(rows), 4
            )
    return AnalysisResult(
        insights=tuple(sorted(insights, key=lambda item: item.insight_id)),
        emerging_opportunities=opportunities,
        observation_count=len(rows),
        fixture_count=len({(row.source_id, row.fixture_id) for row in rows}),
        analyzed_player_count=len(analyzed_players),
        skipped_players=skipped,
        metric_coverage=coverage,
    )
