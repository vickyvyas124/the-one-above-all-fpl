"""Time-safe, walk-forward comparison for an intelligence forecast candidate.

This module screens candidate forecasts against the existing FPL baseline. It
does not fit a model, publish an artifact, change a production forecast, or
promote a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable

import pandas as pd

from fpl_validation import score_gameweek_group


@dataclass(frozen=True)
class WalkForwardForecast:
    season: str
    gameweek: int
    fpl_player_id: int
    deadline_utc: str
    feature_available_at_utc: str
    candidate_trained_through_utc: str
    baseline_points: float
    candidate_points: float
    actual_points: float
    actual_available_at_utc: str
    candidate_version: str
    candidate_artifact_sha256: str
    official_ep_next: float | None = None
    ep_next_available_at_utc: str | None = None


@dataclass(frozen=True)
class ValidationPolicy:
    minimum_seasons: int = 2
    minimum_gameweeks_per_season: int = 6
    minimum_players_per_gameweek: int = 20
    minimum_improved_gameweek_share: float = 2 / 3

    def __post_init__(self) -> None:
        if self.minimum_seasons < 2:
            raise ValueError("Walk-forward review requires at least two seasons.")
        if self.minimum_gameweeks_per_season < 1 or self.minimum_players_per_gameweek < 3:
            raise ValueError("Validation coverage thresholds are too small.")
        if not math.isfinite(self.minimum_improved_gameweek_share) or not (
            0.5 < self.minimum_improved_gameweek_share <= 1.0
        ):
            raise ValueError("Improved-Gameweek share must be greater than one half and at most one.")


@dataclass(frozen=True)
class WalkForwardReport:
    status: str
    candidate_version: str | None
    candidate_artifact_sha256: str | None
    season_count: int
    gameweek_count: int
    player_gameweek_rows: int
    baseline_metrics: dict[str, float | int | None] | None
    candidate_metrics: dict[str, float | int | None] | None
    official_ep_next_metrics: dict[str, float | int | None] | None
    per_season: tuple[dict[str, object], ...]
    per_gameweek: tuple[dict[str, object], ...]
    gate_reasons: tuple[str, ...]
    promotion_allowed: bool = False


def _timestamp(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp.")
    try:
        result = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp.") from exc
    if result.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware ISO-8601 timestamp.")
    return result.astimezone(timezone.utc)


def _score(value: object, name: str, *, optional: bool = False) -> float | None:
    if optional and value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite, nonnegative number.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite, nonnegative number.") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite, nonnegative number.")
    return number


def _validate_forecast(row: WalkForwardForecast) -> None:
    if not isinstance(row.season, str) or not row.season.strip():
        raise ValueError("Forecast season is required.")
    if type(row.gameweek) is not int or row.gameweek < 1:
        raise ValueError("Forecast Gameweek must be a positive integer.")
    if type(row.fpl_player_id) is not int or row.fpl_player_id < 1:
        raise ValueError("A positive, exact FPL player ID is required.")
    if not isinstance(row.candidate_version, str) or not row.candidate_version.strip():
        raise ValueError("Candidate model version is required.")
    digest = row.candidate_artifact_sha256
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("Candidate artifact must have a lowercase SHA-256 digest.")

    deadline = _timestamp(row.deadline_utc, "deadline_utc")
    features_at = _timestamp(row.feature_available_at_utc, "feature_available_at_utc")
    trained_through = _timestamp(row.candidate_trained_through_utc, "candidate_trained_through_utc")
    actual_at = _timestamp(row.actual_available_at_utc, "actual_available_at_utc")
    if features_at > deadline:
        raise ValueError("Candidate features became available after the target deadline.")
    if trained_through >= features_at:
        raise ValueError("Candidate training cutoff must precede the feature snapshot time.")
    if actual_at <= deadline:
        raise ValueError("Actual points must become available strictly after the target deadline.")

    for field in ("baseline_points", "candidate_points", "actual_points"):
        _score(getattr(row, field), field)
    if row.official_ep_next is not None:
        _score(row.official_ep_next, "official_ep_next")
        ep_time = _timestamp(row.ep_next_available_at_utc, "ep_next_available_at_utc")
        if ep_time > deadline:
            raise ValueError("Official ep_next became available after the target deadline.")
    elif row.ep_next_available_at_utc is not None:
        raise ValueError("An ep_next timestamp cannot be supplied without an ep_next value.")


def _as_metrics(values: dict[str, object], prefix: str) -> dict[str, float | int | None]:
    selected = {
        "Players": values.get("Players"),
        "MAE": values.get(f"{prefix} MAE"),
        "RMSE": values.get(f"{prefix} RMSE"),
        "Bias": values.get(f"{prefix} bias"),
        "Spearman": values.get(f"{prefix} Spearman"),
        "Top-10 capture": values.get(f"{prefix} top-10 capture"),
        "Top-20 capture": values.get(f"{prefix} top-20 capture"),
    }
    # The shared FPL scorer reports MAE, bias, rank and capture; add RMSE over
    # the same complete matched rows without changing its established metrics.
    rmse_key = "_candidate_errors" if prefix == "Candidate" else "_baseline_errors"
    errors = values.get(rmse_key, ())
    if errors:
        selected["RMSE"] = math.sqrt(sum(float(error) ** 2 for error in errors) / len(errors))
        if prefix == "Baseline":
            selected["Bias"] = sum(float(error) for error in errors) / len(errors)
    result: dict[str, float | int | None] = {}
    for key, value in selected.items():
        if value is None:
            result[key] = None
        elif key == "Players":
            result[key] = int(value)
        else:
            number = float(value)
            result[key] = number if math.isfinite(number) else None
    return result


def _score_group(rows: list[WalkForwardForecast]) -> dict[str, object]:
    frame = pd.DataFrame([{
        "player_id": row.fpl_player_id,
        "actual_points": row.actual_points,
        "candidate_points": row.candidate_points,
        "baseline_points": row.baseline_points,
        "official_ep_next": row.official_ep_next,
    } for row in rows])
    scored = score_gameweek_group(
        frame,
        actual_col="actual_points",
        model_col="candidate_points",
        baseline_col="baseline_points",
        benchmark_col="official_ep_next",
        include_minutes_metrics=False,
        include_captain_metrics=False,
    )
    scored["_candidate_errors"] = [row.candidate_points - row.actual_points for row in rows]
    scored["_baseline_errors"] = [row.baseline_points - row.actual_points for row in rows]
    ep_rows = [row for row in rows if row.official_ep_next is not None]
    scored["_ep_next_count"] = len(ep_rows)
    scored["_ep_next_errors"] = [float(row.official_ep_next) - row.actual_points for row in ep_rows]
    ep_scored = score_gameweek_group(
        frame[frame["official_ep_next"].notna()].copy(),
        actual_col="actual_points",
        model_col="candidate_points",
        baseline_col="baseline_points",
        benchmark_col="official_ep_next",
        include_minutes_metrics=False,
        include_captain_metrics=False,
    ) if ep_rows else {}
    scored["_ep_candidate_errors"] = [row.candidate_points - row.actual_points for row in ep_rows]
    scored["_ep_baseline_errors"] = [row.baseline_points - row.actual_points for row in ep_rows]
    for stored, metric in (
        ("_ep_candidate_spearman", "Model Spearman"),
        ("_ep_baseline_spearman", "Baseline Spearman"),
        ("_ep_candidate_top10", "Model top-10 capture"),
        ("_ep_baseline_top10", "Baseline top-10 capture"),
        ("_ep_candidate_top20", "Model top-20 capture"),
        ("_ep_baseline_top20", "Baseline top-20 capture"),
    ):
        scored[stored] = ep_scored.get(metric)
    return scored


def _aggregate_scores(
    groups: list[dict[str, object]], prefix: str, *, ep_cohort: bool = False,
) -> dict[str, float | int | None]:
    if prefix == "Candidate":
        errors_key = "_ep_candidate_errors" if ep_cohort else "_candidate_errors"
        errors = [error for group in groups for error in group.get(errors_key, ())]
        spearman_key, top10_key, top20_key = ("_ep_candidate_spearman", "_ep_candidate_top10", "_ep_candidate_top20") if ep_cohort else (
            "Model Spearman", "Model top-10 capture", "Model top-20 capture",
        )
    elif prefix == "Baseline":
        errors_key = "_ep_baseline_errors" if ep_cohort else "_baseline_errors"
        errors = [error for group in groups for error in group.get(errors_key, ())]
        spearman_key, top10_key, top20_key = ("_ep_baseline_spearman", "_ep_baseline_top10", "_ep_baseline_top20") if ep_cohort else (
            "Baseline Spearman", "Baseline top-10 capture", "Baseline top-20 capture",
        )
    else:
        errors = [error for group in groups for error in group.get("_ep_next_errors", ())]
        spearman_key, top10_key, top20_key = (
            "EP next Spearman", "EP next top-10 capture", "EP next top-20 capture",
        )
    players = len(errors)
    if not players:
        return {"Players": 0, "MAE": None, "RMSE": None, "Bias": None,
                "Spearman": None, "Top-10 capture": None, "Top-20 capture": None}
    rank_groups = [group[spearman_key] for group in groups if _finite_or_none(group.get(spearman_key)) is not None]
    top10_groups = [group[top10_key] for group in groups if _finite_or_none(group.get(top10_key)) is not None]
    top20_groups = [group[top20_key] for group in groups if _finite_or_none(group.get(top20_key)) is not None]
    return {
        "Players": players,
        "MAE": sum(abs(float(error)) for error in errors) / players,
        "RMSE": math.sqrt(sum(float(error) ** 2 for error in errors) / players),
        "Bias": sum(float(error) for error in errors) / players,
        "Spearman": (sum(float(value) for value in rank_groups) / len(rank_groups)
                     if rank_groups else None),
        "Top-10 capture": (sum(float(value) for value in top10_groups) / len(top10_groups)
                           if top10_groups else None),
        "Top-20 capture": (sum(float(value) for value in top20_groups) / len(top20_groups)
                           if top20_groups else None),
    }


def evaluate_walk_forward(
    forecasts: Iterable[WalkForwardForecast],
    *,
    policy: ValidationPolicy = ValidationPolicy(),
) -> WalkForwardReport:
    """Compare a single candidate artifact with the baseline on safe common cohorts.

    Records are point-in-time checked and grouped by season/Gameweek. A passing
    screen means human review is warranted; this function never authorizes or
    performs model promotion.
    """
    rows = tuple(forecasts)
    if not rows:
        return WalkForwardReport(
            "NO_CANDIDATE_DATA", None, None, 0, 0, 0, None, None, None, (), (),
            ("No candidate forecasts were supplied; no validation or promotion is possible.",),
        )

    for row in rows:
        _validate_forecast(row)
    versions = {row.candidate_version for row in rows}
    digests = {row.candidate_artifact_sha256 for row in rows}
    if len(versions) != 1 or len(digests) != 1:
        raise ValueError("A walk-forward evaluation must use one candidate version and artifact hash.")
    keys = [(row.season, row.gameweek, row.fpl_player_id) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate season/Gameweek/player forecast row.")

    by_season_gw: dict[tuple[str, int], list[WalkForwardForecast]] = {}
    by_season: dict[str, list[WalkForwardForecast]] = {}
    for row in rows:
        by_season_gw.setdefault((row.season, row.gameweek), []).append(row)
        by_season.setdefault(row.season, []).append(row)

    ordered_weeks = sorted(by_season_gw.items(), key=lambda item: (item[0][0], item[0][1]))
    week_results = []
    scored_weeks: list[tuple[str, int, list[WalkForwardForecast], dict[str, object]]] = []
    for (season, gameweek), group in ordered_weeks:
        score = _score_group(group)
        candidate = _as_metrics(score, "Candidate")
        baseline = _as_metrics(score, "Baseline")
        week_results.append({
            "season": season,
            "gameweek": gameweek,
            "players": len(group),
            "candidate": candidate,
            "baseline": baseline,
        })
        scored_weeks.append((season, gameweek, group, score))

    season_results = []
    for season in sorted(by_season):
        season_scores = [score for score_season, _, _, score in scored_weeks if score_season == season]
        season_results.append({
            "season": season,
            "gameweeks": len({row.gameweek for row in by_season[season]}),
            "players": len(by_season[season]),
            "candidate": _aggregate_scores(season_scores, "Candidate"),
            "baseline": _aggregate_scores(season_scores, "Baseline"),
        })

    pooled_scores = [score for _, _, _, score in scored_weeks]
    candidate_metrics = _aggregate_scores(pooled_scores, "Candidate")
    baseline_metrics = _aggregate_scores(pooled_scores, "Baseline")
    ep_groups = [score for score in pooled_scores if score.get("_ep_next_count", 0) > 0]
    ep_next_metrics = None
    if ep_groups:
        ep_next_official = _aggregate_scores(ep_groups, "EP next")
        ep_next_candidates = _aggregate_scores(ep_groups, "Candidate", ep_cohort=True)
        ep_next_baseline = _aggregate_scores(ep_groups, "Baseline", ep_cohort=True)
        ep_next_metrics = {
            "EP next MAE": ep_next_official["MAE"],
            "EP next Spearman": ep_next_official["Spearman"],
            "EP next Top-10 capture": ep_next_official["Top-10 capture"],
            "EP next Top-20 capture": ep_next_official["Top-20 capture"],
            "Candidate MAE on ep_next cohort": ep_next_candidates["MAE"],
            "Baseline MAE on ep_next cohort": ep_next_baseline["MAE"],
            "Candidate Spearman on ep_next cohort": ep_next_candidates["Spearman"],
            "Baseline Spearman on ep_next cohort": ep_next_baseline["Spearman"],
            "Candidate Top-10 capture on ep_next cohort": ep_next_candidates["Top-10 capture"],
            "Baseline Top-10 capture on ep_next cohort": ep_next_baseline["Top-10 capture"],
            "Candidate Top-20 capture on ep_next cohort": ep_next_candidates["Top-20 capture"],
            "Baseline Top-20 capture on ep_next cohort": ep_next_baseline["Top-20 capture"],
            "Players": ep_next_official["Players"],
        }

    reasons: list[str] = []
    if len(by_season) < policy.minimum_seasons:
        reasons.append(
            f"Only {len(by_season)} season(s); at least {policy.minimum_seasons} are required."
        )
    for result in season_results:
        if int(result["gameweeks"]) < policy.minimum_gameweeks_per_season:
            reasons.append(
                f"{result['season']} has {result['gameweeks']} evaluated Gameweeks; "
                f"at least {policy.minimum_gameweeks_per_season} are required."
            )
    low_coverage = [item for item in week_results if item["players"] < policy.minimum_players_per_gameweek]
    if low_coverage:
        reasons.append(
            f"{len(low_coverage)} Gameweek(s) have fewer than "
            f"{policy.minimum_players_per_gameweek} common players."
        )

    candidate_mae = candidate_metrics["MAE"]
    baseline_mae = baseline_metrics["MAE"]
    candidate_rmse = candidate_metrics["RMSE"]
    baseline_rmse = baseline_metrics["RMSE"]
    if candidate_mae is None or baseline_mae is None or candidate_mae >= baseline_mae:
        reasons.append("Candidate MAE does not improve on the baseline over the pooled cohort.")
    season_improvements = [
        result["candidate"]["MAE"] is not None
        and result["baseline"]["MAE"] is not None
        and result["candidate"]["MAE"] < result["baseline"]["MAE"]
        for result in season_results
    ]
    if not season_improvements or not all(season_improvements):
        reasons.append("Candidate MAE improvement is not present in every evaluated season.")
    improved_weeks = sum(
        item["candidate"]["MAE"] is not None
        and item["baseline"]["MAE"] is not None
        and item["candidate"]["MAE"] < item["baseline"]["MAE"]
        for item in week_results
    )
    week_share = improved_weeks / len(week_results)
    if week_share < policy.minimum_improved_gameweek_share:
        reasons.append(
            f"Candidate MAE improves in {week_share:.0%} of Gameweeks; "
            f"at least {policy.minimum_improved_gameweek_share:.0%} are required."
        )
    if candidate_rmse is None or baseline_rmse is None or candidate_rmse > baseline_rmse:
        reasons.append("Candidate RMSE is worse than the baseline or could not be measured.")
    for metric in ("Spearman", "Top-10 capture", "Top-20 capture"):
        candidate_value = candidate_metrics[metric]
        baseline_value = baseline_metrics[metric]
        if candidate_value is None or baseline_value is None or candidate_value < baseline_value:
            reasons.append(f"Candidate {metric} is worse than the baseline or could not be measured.")

    status = "ELIGIBLE_FOR_HUMAN_REVIEW" if not reasons else "INSUFFICIENT_OR_FAILED_EVIDENCE"
    if not reasons:
        reasons.append(
            "Screening criteria passed. Decision-outcome review, calibration, and independent approval "
            "are still required; automatic promotion is disabled."
        )
    return WalkForwardReport(
        status=status,
        candidate_version=next(iter(versions)),
        candidate_artifact_sha256=next(iter(digests)),
        season_count=len(by_season),
        gameweek_count=len(by_season_gw),
        player_gameweek_rows=len(rows),
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        official_ep_next_metrics=ep_next_metrics,
        per_season=tuple(season_results),
        per_gameweek=tuple(week_results),
        gate_reasons=tuple(reasons),
        promotion_allowed=False,
    )


def _finite_or_none(value: object) -> float | int | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None
