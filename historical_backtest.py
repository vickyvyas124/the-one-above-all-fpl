"""Leakage-conscious historical FPL proxy backtest for completed prior seasons.

The script uses per-Gameweek results but creates features only from earlier matches
that conservatively appear finalized before a proxy deadline. It is not an exact replay of old FPL API
snapshots. `ep_next` is intentionally excluded because archived `ep_this`/`xP` fields
can be collected after deadlines and may leak outcomes.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import fpl_learning
from fpl_validation import score_gameweek_group

SOURCES = {
    "2023-24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
    "2024-25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv",
    "2025-26": "https://raw.githubusercontent.com/imadeddine-belkat/Premier-League-Stats/main/fpl_scraper/fpl_stats/_merged/players/2025-26_all_players_gw.csv",
}
# The 2025-26 player archive is fixture-grain and has fixture_code rather than
# a round/event column. Map each fixture code to its official FPL Gameweek using
# the companion fixture table from the same data source.
FIXTURE_SOURCES = {
    "2025-26": "https://raw.githubusercontent.com/imadeddine-belkat/Premier-League-Stats/main/fpl_scraper/fpl_stats/fixtures/2025-26_all_fixtures.csv",
}
HEADERS = {"User-Agent": "Mozilla/5.0 The-One-Above-All-FPL/2.9.1 historical-validation"}
EVALUATION_TYPE = "historical_proxy_single_fixture_no_point_in_time_snapshots"


def fetch_csv(url: str) -> pd.DataFrame:
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=60, headers=HEADERS)
            response.raise_for_status()
            if len(response.content) < 256:
                raise ValueError("Downloaded CSV is unexpectedly small")
            from io import BytesIO
            return pd.read_csv(BytesIO(response.content), low_memory=False)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not download {url}: {last_error}")


def pick_column(frame: pd.DataFrame, candidates: list[str], required=True):
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    if required:
        raise ValueError(f"Missing required column; tried {candidates}. Available: {list(frame.columns)[:60]}")
    return None


def normalize_position(value) -> str:
    if pd.isna(value):
        return "UNK"
    text = str(value).strip().upper()
    mapping = {"1": "GKP", "GK": "GKP", "GKP": "GKP", "GOALKEEPER": "GKP",
               "2": "DEF", "D": "DEF", "DEF": "DEF", "DEFENDER": "DEF",
               "3": "MID", "M": "MID", "MID": "MID", "MIDFIELDER": "MID",
               "4": "FWD", "F": "FWD", "FW": "FWD", "FWD": "FWD", "FORWARD": "FWD"}
    return mapping.get(text, text)


def _fixture_key_series(values: pd.Series) -> pd.Series:
    """Normalize numeric fixture codes and string IDs so CSV dtypes do not break joins."""
    keys = values.astype("string").str.strip().replace("", pd.NA)
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_mask = numeric.notna()
    if numeric_mask.any():
        keys.loc[numeric_mask] = numeric.loc[numeric_mask].astype("Int64").astype("string")
    return keys


def season_frame(season: str, url: str) -> pd.DataFrame:
    raw = fetch_csv(url)
    id_col = pick_column(raw, ["element", "player_id", "player_code", "element_id", "playerid", "id"])
    gw_col = pick_column(raw, ["round", "event", "gameweek", "gw", "gameweek_id"], required=False)
    points_col = pick_column(raw, ["total_points", "event_points", "gw_points", "points"])
    minutes_col = pick_column(raw, ["minutes", "mins"], required=False)
    starts_col = pick_column(raw, ["starts", "start_count"], required=False)
    position_col = pick_column(raw, ["element_type", "position", "position_short", "position_name"], required=False)
    xg_col = pick_column(raw, ["expected_goals", "xg", "expected_goals_scored"], required=False)
    xa_col = pick_column(raw, ["expected_assists", "xa"], required=False)
    home_col = pick_column(raw, ["was_home", "is_home", "home"], required=False)
    kickoff_col = pick_column(raw, ["kickoff_time", "kickoff", "date", "match_date"], required=False)
    fixture_col = pick_column(raw, ["fixture", "fixture_code", "fixture_id", "fixtureid"])
    team_col = pick_column(raw, ["team_code", "team", "team_id", "team_id_x", "team_name"])
    name_col = pick_column(raw, ["web_name", "name", "second_name", "player_name"], required=False)

    if gw_col:
        gameweek_values = pd.to_numeric(raw[gw_col], errors="coerce")
    elif season in FIXTURE_SOURCES:
        fixture_data = fetch_csv(FIXTURE_SOURCES[season])
        fixture_code_col = pick_column(fixture_data, ["code", "fixture_code", "fixture", "fixture_id", "id"])
        fixture_gw_col = pick_column(fixture_data, ["event", "round", "gameweek", "gw"])
        fixture_map = pd.DataFrame({
            "_fixture_key": _fixture_key_series(fixture_data[fixture_code_col]),
            "_gameweek": pd.to_numeric(fixture_data[fixture_gw_col], errors="coerce"),
        })
        if fixture_map["_fixture_key"].isna().any() or fixture_map["_fixture_key"].eq("").any():
            raise ValueError(f"Fixture lookup for {season} contains missing fixture identifiers")
        duplicate_keys = fixture_map[fixture_map["_fixture_key"].duplicated(keep=False)]
        if not duplicate_keys.empty:
            raise ValueError(f"Fixture lookup for {season} contains duplicate fixture identifiers: {duplicate_keys['_fixture_key'].head().tolist()}")
        if fixture_map["_gameweek"].isna().any():
            raise ValueError(f"Fixture lookup for {season} contains unassigned/postponed fixtures without a Gameweek")
        code_to_gameweek = dict(zip(fixture_map["_fixture_key"], fixture_map["_gameweek"]))
        gameweek_values = _fixture_key_series(raw[fixture_col]).map(code_to_gameweek)
        if gameweek_values.isna().any():
            raise ValueError(
                f"Some player fixtures could not be mapped to a Gameweek for {season}. "
                f"Player fixture column: {fixture_col}; lookup code column: {fixture_code_col}."
            )
        gameweek_values = pd.to_numeric(gameweek_values, errors="coerce")
    else:
        raise ValueError(
            f"Missing Gameweek column and no fixture lookup is configured for {season}. "
            f"Available columns: {list(raw.columns)[:60]}"
        )

    if pd.to_numeric(raw[id_col], errors="coerce").isna().any():
        raise ValueError(f"{season} archive contains missing or invalid player identifiers")
    if gameweek_values.isna().any():
        raise ValueError(f"{season} archive contains rows with no verified Gameweek mapping")
    if not gameweek_values.between(1, 38).all():
        raise ValueError(f"{season} archive contains Gameweeks outside 1..38")
    points_values = pd.to_numeric(raw[points_col], errors="coerce")
    if points_values.isna().any():
        raise ValueError(f"{season} archive contains rows with missing/invalid target points")
    if raw[team_col].isna().any() or raw[team_col].astype("string").str.strip().eq("").any():
        raise ValueError(f"{season} archive contains missing team identifiers")
    if kickoff_col:
        kickoff_values = pd.to_datetime(raw[kickoff_col], errors="coerce", utc=True)
        kickoff_has_clock = raw[kickoff_col].astype("string").str.contains(r"\d{1,2}:\d{2}", regex=True, na=False)
    elif season in FIXTURE_SOURCES:
        fixture_data = fetch_csv(FIXTURE_SOURCES[season])
        fixture_code_col = pick_column(fixture_data, ["code", "fixture_code", "fixture", "fixture_id", "id"])
        fixture_kickoff_col = pick_column(fixture_data, ["kickoff_time", "kickoff", "date", "match_date"])
        kickoff_map = pd.DataFrame({
            "fixture": _fixture_key_series(fixture_data[fixture_code_col]),
            "kickoff": pd.to_datetime(fixture_data[fixture_kickoff_col], errors="coerce", utc=True),
            "has_clock": fixture_data[fixture_kickoff_col].astype("string").str.contains(r"\d{1,2}:\d{2}", regex=True, na=False),
        })
        if (kickoff_map["fixture"].isna().any() or kickoff_map["fixture"].duplicated().any()
                or kickoff_map["kickoff"].isna().any() or not kickoff_map["has_clock"].all()):
            raise ValueError(f"Fixture lookup for {season} has missing, duplicate or invalid kickoff records")
        kickoff_by_fixture = dict(zip(kickoff_map["fixture"], kickoff_map["kickoff"]))
        kickoff_values = _fixture_key_series(raw[fixture_col]).map(kickoff_by_fixture)
        kickoff_has_clock = _fixture_key_series(raw[fixture_col]).map(dict(zip(kickoff_map["fixture"], kickoff_map["has_clock"]))).fillna(False)
    else:
        raise ValueError(f"{season} archive has no fixture kickoff timestamps; point-in-time history cannot be reconstructed safely")
    fixture_keys = _fixture_key_series(raw[fixture_col])
    if fixture_keys.isna().any() or fixture_keys.eq("").any():
        # A fixture-less row is only valid as an explicit zero-activity player/GW
        # row. It may represent a blank; it cannot contribute points/statistics.
        point_values = pd.to_numeric(raw[points_col], errors="coerce").fillna(0)
        minute_values = pd.to_numeric(raw[minutes_col], errors="coerce").fillna(0) if minutes_col else 0
        xg_values = pd.to_numeric(raw[xg_col], errors="coerce").fillna(0) if xg_col else 0
        xa_values = pd.to_numeric(raw[xa_col], errors="coerce").fillna(0) if xa_col else 0
        invalid_missing_fixture = (fixture_keys.isna() | fixture_keys.eq("")) & (
            point_values.ne(0) | (pd.Series(minute_values, index=raw.index) if not isinstance(minute_values, pd.Series) else minute_values).ne(0)
            | (pd.Series(xg_values, index=raw.index) if not isinstance(xg_values, pd.Series) else xg_values).ne(0)
            | (pd.Series(xa_values, index=raw.index) if not isinstance(xa_values, pd.Series) else xa_values).ne(0)
        )
        if invalid_missing_fixture.any():
            raise ValueError(f"{season} archive has points or match statistics without a fixture identifier")
    if (fixture_keys.notna() & (kickoff_values.isna() | ~kickoff_has_clock)).any():
        raise ValueError(f"{season} archive has fixtures with missing/invalid kickoff timestamps")

    duplicate_player_fixtures = pd.DataFrame({
        "player": pd.to_numeric(raw[id_col], errors="coerce"),
        "fixture": fixture_keys,
        "gameweek": gameweek_values,
    }).dropna(subset=["fixture"])
    if duplicate_player_fixtures.duplicated(["player", "fixture"]).any():
        raise ValueError(f"{season} archive contains duplicate player/fixture rows")
    fixture_events = pd.DataFrame({"fixture": fixture_keys, "gameweek": gameweek_values}).dropna(subset=["fixture"]).drop_duplicates()
    if fixture_events.duplicated("fixture").any():
        raise ValueError(f"{season} archive maps a fixture identifier to more than one Gameweek")

    subset = pd.DataFrame({
        "player_id": pd.to_numeric(raw[id_col], errors="coerce"),
        "gameweek": gameweek_values,
        "actual_points": pd.to_numeric(raw[points_col], errors="coerce"),
        "minutes": pd.to_numeric(raw[minutes_col], errors="coerce") if minutes_col else 0.0,
        "starts": pd.to_numeric(raw[starts_col], errors="coerce") if starts_col else (pd.to_numeric(raw[minutes_col], errors="coerce") > 0).astype(float) if minutes_col else 0.0,
        "position": raw[position_col].map(normalize_position) if position_col else "UNK",
        "xg": pd.to_numeric(raw[xg_col], errors="coerce") if xg_col else 0.0,
        "xa": pd.to_numeric(raw[xa_col], errors="coerce") if xa_col else 0.0,
        "was_home": raw[home_col].map(_parse_bool).astype(float) if home_col else float("nan"),
        "kickoff_time": kickoff_values,
        "player": raw[name_col].astype(str) if name_col else "Player",
        "team_key": raw[team_col].astype("string") if team_col else "__unknown_team__",
    })
    if fixture_col:
        subset["_fixture_id"] = fixture_keys
    else:
        subset["_fixture_id"] = pd.Series(pd.NA, index=subset.index, dtype="string")
    # Infer the team's scheduled fixture count from the union of fixture IDs for the
    # team/Gameweek, rather than from an individual player's rows. A player's absence
    # from a fixture must not decide whether the model forecasts a single or double GW.
    schedule_source_available = True
    if schedule_source_available:
        scheduled = subset.dropna(subset=["_fixture_id"]).drop_duplicates(["team_key", "gameweek", "_fixture_id"])
        schedule_stats = scheduled.groupby(["team_key", "gameweek"], as_index=False).agg(
            scheduled_fixture_count=("_fixture_id", "nunique"),
            schedule_was_home=("was_home", "mean"),
        schedule_first_kickoff=("kickoff_time", "min"),
        schedule_last_kickoff=("kickoff_time", "max"),
        )
        subset = subset.merge(schedule_stats, on=["team_key", "gameweek"], how="left")
        # If the archive exposes fixture IDs, missing fixtures in a team's GW are a
        # blank (zero fixtures), not an assumed single fixture.
        subset["scheduled_fixture_count"] = pd.to_numeric(
            subset["scheduled_fixture_count"], errors="coerce"
        ).fillna(0.0).clip(0, 3)
        subset["schedule_was_home"] = pd.to_numeric(subset["schedule_was_home"], errors="coerce")
    subset["schedule_source_available"] = 1 if schedule_source_available else 0
    subset["schedule_was_home"] = pd.to_numeric(subset["schedule_was_home"], errors="coerce")
    subset["player_id"] = subset["player_id"].astype(int)
    subset["gameweek"] = subset["gameweek"].astype(int)
    subset = subset[subset["gameweek"].between(1, 38)]
    if subset.empty:
        raise ValueError(f"No usable rows found for {season}")
    agg = {
        "actual_points": "sum", "minutes": "sum", "starts": "sum", "xg": "sum", "xa": "sum",
        "position": "last", "was_home": "mean", "player": "last", "team_key": "last",
        "scheduled_fixture_count": "max", "schedule_was_home": "mean",
        "schedule_first_kickoff": "min", "schedule_last_kickoff": "max",
        "schedule_source_available": "max",
    }
    grouped = subset.groupby(["player_id", "gameweek"], as_index=False).agg(agg)
    fixture_values = pd.to_numeric(grouped["scheduled_fixture_count"], errors="coerce")
    grouped["fixture_count"] = fixture_values.fillna(0).clip(0, 3)
    grouped["was_home"] = pd.to_numeric(grouped["schedule_was_home"], errors="coerce").fillna(grouped["was_home"])
    grouped["season"] = season
    grouped["source_url"] = url
    return grouped


def _parse_bool(value):
    if pd.isna(value):
        return float("nan")
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "home", "h"}:
        return 1
    if text in {"false", "0", "no", "away", "a"}:
        return 0
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def add_point_in_time_features(season_data: pd.DataFrame) -> pd.DataFrame:
    max_gw = int(season_data["gameweek"].max())
    panels = []
    for pid, group in season_data.groupby("player_id", sort=False):
        first = int(group["gameweek"].min())
        index = pd.DataFrame({"gameweek": range(first, max_gw + 1)})
        merged = index.merge(group, on="gameweek", how="left", suffixes=("", "_observed"))
        merged["player_id"] = int(pid)
        merged["observed_row"] = merged["actual_points"].notna()
        for col in ("actual_points", "minutes", "starts", "xg", "xa"):
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        # Forward-fill only: back-filling would expose a future position/team at
        # an earlier simulated deadline.
        merged["position"] = merged["position"].ffill().fillna("UNK")
        merged["player"] = merged["player"].ffill().fillna("Player")
        merged["team_key"] = merged["team_key"].ffill().fillna("__unknown_team__")
        merged["was_home"] = pd.to_numeric(merged["schedule_was_home"], errors="coerce")
        panels.append(merged)
    panel = pd.concat(panels, ignore_index=True).sort_values(["player_id", "gameweek"])
    schedule_lookup = (season_data.groupby(["team_key", "gameweek"], as_index=False)
                       .agg(fixture_count=("fixture_count", "max"), schedule_was_home=("was_home", "mean"),
                            schedule_last_kickoff=("schedule_last_kickoff", "max")))
    panel = panel.drop(columns=["fixture_count", "schedule_was_home", "schedule_last_kickoff"], errors="ignore").merge(
        schedule_lookup, on=["team_key", "gameweek"], how="left"
    )
    panel["fixture_count"] = pd.to_numeric(panel["fixture_count"], errors="coerce").fillna(0).clip(0, 3)
    panel["was_home"] = pd.to_numeric(panel["schedule_was_home"], errors="coerce")
    # Conservative as-of reconstruction for this proxy: treat the earliest target
    # fixture as the deadline boundary (two hours earlier), and only expose prior
    # event statistics once every fixture in that prior team/event was finished
    # for 72 hours. Late/postponed fixtures therefore cannot enter earlier history.
    event_first_kickoff = season_data.groupby("gameweek")["schedule_first_kickoff"].min()
    panel["_target_cutoff"] = panel["gameweek"].map(event_first_kickoff) - pd.Timedelta(hours=2)
    panel["_history_available_at"] = pd.to_datetime(panel["schedule_last_kickoff"], utc=True) + pd.Timedelta(hours=72)
    panel["xgi"] = panel["xg"] + panel["xa"]
    defaults = {"GKP": 2.6, "DEF": 2.8, "MID": 3.0, "FWD": 3.0}
    features = {
        name: [] for name in ("season_mean_prev", "recent_mean_prev", "recent_minutes_per_fixture_prev",
                              "recent_appearance_rate_prev", "recent_start_rate_prev", "recent_xgi_per90_prev",
                              "season_xgi_per90_prev", "position_prev", "pos_mean_prev")
    }
    for _, player_rows in panel.groupby("player_id", sort=False):
        player_rows = player_rows.sort_values("gameweek")
        for idx, target in player_rows.iterrows():
            cutoff = target["_target_cutoff"]
            prior = player_rows[(player_rows["gameweek"] < target["gameweek"])
                                & player_rows["observed_row"]
                                & player_rows["_history_available_at"].notna()
                                & (player_rows["_history_available_at"] < cutoff)] if pd.notna(cutoff) else player_rows.iloc[0:0]
            recent = prior.tail(5)
            season_points = prior["actual_points"]
            recent_points = recent["actual_points"]
            season_minutes = float(prior["minutes"].sum())
            recent_minutes = float(recent["minutes"].sum())
            recent_fixtures = recent["fixture_count"].replace(0, float("nan"))
            recent_minutes_per_fixture = (recent["minutes"] / recent_fixtures).replace([float("inf"), -float("inf")], float("nan")).dropna()
            recent_xgi = float(recent["xgi"].sum())
            season_xgi = float(prior["xgi"].sum())
            last_position = str(prior.iloc[-1]["position"]) if not prior.empty else "UNK"
            features["season_mean_prev"].append(float(season_points.mean()) if not prior.empty else float("nan"))
            features["recent_mean_prev"].append(float(recent_points.mean()) if not recent.empty else float("nan"))
            features["recent_minutes_per_fixture_prev"].append(float(recent_minutes_per_fixture.mean()) if not recent_minutes_per_fixture.empty else float("nan"))
            features["recent_appearance_rate_prev"].append(float((recent["minutes"] > 0).mean()) if not recent.empty else float("nan"))
            features["recent_start_rate_prev"].append(float((recent["starts"] > 0).mean()) if not recent.empty else float("nan"))
            features["recent_xgi_per90_prev"].append(recent_xgi / max(90.0, recent_minutes) * 90.0)
            features["season_xgi_per90_prev"].append(season_xgi / max(90.0, season_minutes) * 90.0)
            features["position_prev"].append(last_position)
            features["pos_mean_prev"].append(defaults.get(last_position, 2.8))
    for name, values in features.items():
        panel[name] = values
    panel["xgi_per90_prev"] = panel["recent_xgi_per90_prev"]
    panel["prediction_expected_minutes"] = panel["recent_minutes_per_fixture_prev"].fillna(0).clip(0, 90)
    # Position dummies keep the model interpretable and consistent at inference.
    # The target row's position and schedule are observed only after the target
    # matches. Position is shifted; target fixture/home fields are held neutral.
    for pos in ("GKP", "DEF", "MID", "FWD"):
        panel[f"position_{pos}"] = (panel["position_prev"].astype(str).str.upper() == pos).astype(float)
    panel["home_share"] = 0.5
    panel["attack_adjustment"] = (panel["xgi_per90_prev"].fillna(0).clip(lower=0) * 0.55).clip(upper=1.0)
    # This deliberately simple proxy only consumes rows strictly before the target GW.
    panel["prediction_model"] = (
        0.45 * panel["season_mean_prev"].fillna(panel["pos_mean_prev"])
        + 0.35 * panel["recent_mean_prev"].fillna(panel["pos_mean_prev"])
        + 0.20 * panel["pos_mean_prev"]
        + panel["attack_adjustment"]
    ).clip(lower=0)
    panel["prediction_baseline"] = panel["season_mean_prev"].fillna(panel["pos_mean_prev"]).clip(lower=0)
    # Use team-level scheduled fixture counts where fixture IDs and team keys exist.
    # Otherwise keep a one-fixture assumption instead of reading player appearance counts.
    # This proxy does not feed target-GW fixture counts, venue or later
    # reassignments into a simulated deadline prediction.
    panel["fixture_count_feature"] = 1.0
    panel["fixture_multiplier"] = 1.0
    panel["prediction_model"] = panel["prediction_model"]
    panel["prediction_baseline"] = panel["prediction_baseline"]
    # Keep the old formula as a clearly labelled fallback for the first source season,
    # where no earlier season is available to train the supervised model.
    panel["prediction_model_legacy"] = panel["prediction_model"]
    return panel


def _gw_metrics(group: pd.DataFrame) -> dict:
    return score_gameweek_group(
        group.rename(columns={"prediction_model": "projection_model", "prediction_baseline": "projection_baseline", "prediction_expected_minutes": "projection_expected_minutes", "minutes": "actual_minutes"}),
        actual_col="actual_points", model_col="projection_model", baseline_col="projection_baseline",
        benchmark_col="__none__"
    )


def _eligible_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Only rows present in the archive and with a strictly prior history record."""
    return panel[panel["observed_row"] & panel["season_mean_prev"].notna()].copy()


def _single_fixture_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Proxy cohort only: final archive says exactly one fixture in the target GW."""
    data = _eligible_rows(panel)
    return data[pd.to_numeric(data["fixture_count"], errors="coerce").eq(1)].copy()


def _write_failure_manifest(out: Path, requested: list[str], panels: dict[str, pd.DataFrame], failures: list[dict], evaluated: list[str] | None = None) -> None:
    """Record an incomplete run separately; never replace prior good artifacts."""
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status": "incomplete",
        "evaluation_type": EVALUATION_TYPE,
        "requested_seasons": requested,
        "loaded_seasons": [s for s in requested if s in panels],
        "evaluated_seasons": list(evaluated or []),
        "skipped_seasons": [s for s in requested if s not in panels],
        "failed_seasons": sorted({item["season"] for item in failures if item.get("season")}),
        "failures": failures,
        "published_artifacts_updated": False,
    }
    (out / "historical_backtest_failure.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _score_panel(season: str, panel: pd.DataFrame, model: dict | None, model_type: str, url: str):
    data = _single_fixture_rows(panel)
    if data.empty:
        raise ValueError(f"No eligible point-in-time observations for {season}")
    baseline = data["season_mean_prev"].fillna(data["pos_mean_prev"]).clip(lower=0)
    fixture_count = pd.to_numeric(data["fixture_count"], errors="coerce").fillna(0).clip(0, 3)
    home = pd.to_numeric(data["home_share"], errors="coerce").fillna(0.5).clip(0, 1)
    data["prediction_baseline"] = baseline * fixture_count * (0.95 + 0.10 * home)
    data["prediction_model"] = float("nan")
    if model:
        feature_data = data[fpl_learning.FEATURES].copy()
        feature_data["fixture_count"] = 1.0
        feature_data["home_share"] = 0.5
        rows = feature_data.to_dict("records")
        predictions = [fpl_learning.predict_features(model, row) for row in rows]
        data["prediction_model"] = pd.to_numeric(pd.Series(predictions, index=data.index), errors="coerce")
    else:
        # First available season has no earlier full-season training data. Keep the
        # previous lagged heuristic as a fallback diagnostic, not as ML evidence.
        data["prediction_model"] = data["prediction_model_legacy"] if "prediction_model_legacy" in data else data["prediction_baseline"]
    gw_rows = []
    for gw, group in data.groupby("gameweek"):
        renamed = group.rename(columns={
            "prediction_model": "projection_model",
            "prediction_baseline": "projection_baseline",
            "prediction_expected_minutes": "projection_expected_minutes",
            "minutes": "actual_minutes",
        })
        metrics = score_gameweek_group(renamed, actual_col="actual_points", model_col="projection_model",
                                       baseline_col="projection_baseline", benchmark_col="__none__")
        gw_rows.append({"season": season, "gameweek": int(gw), "model_type": model_type,
                        "evaluation_type": EVALUATION_TYPE, **metrics})
    gw_df = pd.DataFrame(gw_rows)
    summary = {
        "season": season,
        "data_source": url,
        "model_type": model_type,
        "trained_model_used": bool(model),
        "evaluation_type": EVALUATION_TYPE,
        "target_schedule_cohort": "single final-mapped fixture only",
        "excluded_non_single_fixture_rows": int(len(_eligible_rows(panel)) - len(data)),
        "training_rows": int(model.get("training_rows", 0)) if model else 0,
        "gameweeks_evaluated": int(gw_df["gameweek"].nunique()) if not gw_df.empty else 0,
        "player_gameweeks_evaluated": int(len(data)),
        "model_mae_mean_by_gw": float(gw_df["Model MAE"].mean()) if not gw_df.empty else float("nan"),
        "baseline_mae_mean_by_gw": float(gw_df["Baseline MAE"].mean()) if not gw_df.empty else float("nan"),
        "model_minutes_mae_mean_by_gw": float(gw_df["Expected-minutes MAE"].mean()) if not gw_df.empty and "Expected-minutes MAE" in gw_df else float("nan"),
        "model_spearman_mean_by_gw": float(gw_df["Model Spearman"].mean()) if not gw_df.empty else float("nan"),
        "baseline_spearman_mean_by_gw": float(gw_df["Baseline Spearman"].mean()) if not gw_df.empty else float("nan"),
        "model_top10_capture_mean_by_gw": float(gw_df["Model top-10 capture"].mean()) if not gw_df.empty else float("nan"),
        "baseline_top10_capture_mean_by_gw": float(gw_df["Baseline top-10 capture"].mean()) if not gw_df.empty else float("nan"),
        "model_top20_capture_mean_by_gw": float(gw_df["Model top-20 capture"].mean()) if not gw_df.empty else float("nan"),
        "baseline_top20_capture_mean_by_gw": float(gw_df["Baseline top-20 capture"].mean()) if not gw_df.empty else float("nan"),
        "warning": "Historical proxy, not exact point-in-time replay: archived player/team snapshots and fixture schedules are not deadline snapshots. Target fixture count, venue and target-row position are excluded from features; only final-mapped single-fixture target rows are scored. No claim about blank/double-GW performance is made.",
    }
    return pd.DataFrame([summary]), gw_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="historical-output")
    parser.add_argument("--seasons", nargs="*", default=["2023-24", "2024-25", "2025-26"])
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if len(args.seasons) < 2:
        _write_failure_manifest(out, args.seasons, {}, [{"error": "At least two complete seasons are required for a forward holdout and deployable model."}])
        print("At least two complete seasons are required; no artifacts were published.", file=sys.stderr)
        return 1
    panels: dict[str, pd.DataFrame] = {}
    failures = []
    for season in args.seasons:
        if season not in SOURCES:
            failures.append({"season": season, "error": "No source configured"})
            continue
        print(f"Downloading and engineering lagged features for {season}...")
        try:
            panels[season] = add_point_in_time_features(season_frame(season, SOURCES[season]))
            print(f"{season}: {int(panels[season]['gameweek'].nunique())} Gameweeks, {int(len(_single_fixture_rows(panels[season])))} eligible single-fixture proxy player-GWs.")
        except Exception as exc:
            failures.append({"season": season, "error": str(exc)})
            print(f"FAILED {season}: {exc}", file=sys.stderr)

    if failures or len(panels) != len(args.seasons):
        _write_failure_manifest(out, args.seasons, panels, failures or [{"error": "One or more requested seasons were not evaluated."}])
        print(f"Incomplete historical run. Loaded={list(panels)}; evaluated=[]; skipped={[s for s in args.seasons if s not in panels]}; failures={failures}", file=sys.stderr)
        return 1

    ordered_seasons = [season for season in args.seasons if season in panels]
    summaries, gameweeks = [], []
    for idx, season in enumerate(ordered_seasons):
        prior_seasons = ordered_seasons[:idx]
        prior_frames = [_single_fixture_rows(panels[prior]) for prior in prior_seasons]
        trained = None
        if prior_frames:
            try:
                train_frame = pd.concat(prior_frames, ignore_index=True)
                trained = fpl_learning.fit_ridge_model(train_frame, source_seasons=prior_seasons)
                print(f"Fitted forward model for {season} using {len(train_frame)} rows from {prior_seasons}.")
            except Exception as exc:
                failures.append({"season": season, "error": f"Model training failed: {exc}"})
                print(f"MODEL TRAINING FAILED for {season}: {exc}", file=sys.stderr)
        model_type = "trained_ridge_prior_seasons" if trained else "heuristic_fallback_no_prior_seasons"
        try:
            summary, gw = _score_panel(season, panels[season], trained, model_type, SOURCES[season])
            summaries.append(summary)
            gameweeks.append(gw)
        except Exception as exc:
            failures.append({"season": season, "error": f"Evaluation failed: {exc}"})

    if failures:
        _write_failure_manifest(out, args.seasons, panels, failures, [row["season"] for row in summaries])
        print(f"Incomplete historical run. Evaluated={ordered_seasons}; failed={failures}; no model or reports were published.", file=sys.stderr)
        return 1

    # The deployable model for future Gameweeks is fitted only on completed,
    # verified single-fixture rows. Historical metrics remain out-of-season.
    final_frames = [_single_fixture_rows(panels[season]) for season in ordered_seasons]
    final_training = pd.concat(final_frames, ignore_index=True)
    try:
        deployed_model = fpl_learning.fit_ridge_model(final_training, source_seasons=ordered_seasons)
    except Exception as exc:
        failures.append({"stage": "final_training", "error": str(exc)})
        _write_failure_manifest(out, args.seasons, panels, failures, [row["season"] for row in summaries])
        print(f"Deployable model training failed; no model or reports were published: {exc}", file=sys.stderr)
        return 1
    deployed_model["evaluation_note"] = "Expanding-season holdout on the explicitly labeled historical single-fixture proxy cohort; deployed weights are refit on all complete, verified single-fixture rows."
    deployed_model["evaluation_type"] = EVALUATION_TYPE

    summary_df = pd.concat(summaries, ignore_index=True)
    gw_df = pd.concat(gameweeks, ignore_index=True) if gameweeks else pd.DataFrame()
    metadata = {
        "status": "complete",
        "evaluation_type": EVALUATION_TYPE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_note": "Historical proxy, not exact point-in-time replay. Archived player/team snapshots and final fixture schedules are not deadline snapshots. Target fixture count, venue and target-row position are excluded from features. Training and historical scores use only rows mapped to exactly one final-schedule fixture; blank/double Gameweeks are excluded. Out-of-season metrics use only preceding seasons.",
        "requested_seasons": args.seasons,
        "evaluated_seasons": summary_df["season"].tolist(),
        "skipped_seasons": [],
        "failed_seasons": [],
        "success_seasons": summary_df["season"].tolist(),
        "failures": failures,
        "deployed_model": {"model_version": deployed_model["model_version"], "model_type": deployed_model["model_type"], "training_rows": deployed_model["training_rows"], "source_seasons": deployed_model["source_seasons"], "feature_count": len(deployed_model["features"])},
    }
    # Stage every artifact first. A download, fit, metric, or serialization failure
    # must leave the last deployable model untouched. The workflow publishes these
    # files only after this command exits successfully.
    try:
        with tempfile.TemporaryDirectory(dir=out) as stage_name:
            stage = Path(stage_name)
            fpl_learning.save_model(deployed_model, stage / "trained_fpl_model.json")
            summary_df.to_csv(stage / "historical_backtest_summary.csv", index=False)
            gw_df.to_csv(stage / "historical_backtest_gameweeks.csv", index=False)
            (stage / "historical_backtest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            for name in ("historical_backtest_summary.csv", "historical_backtest_gameweeks.csv", "historical_backtest_metadata.json"):
                os.replace(stage / name, out / name)
            # Replace the deployable artifact last, after every other staged write.
            os.replace(stage / "trained_fpl_model.json", out / "trained_fpl_model.json")
    except Exception as exc:
        failures.append({"stage": "artifact_persistence", "error": str(exc)})
        _write_failure_manifest(out, args.seasons, panels, failures)
        print(f"Artifact persistence failed; previous model preserved: {exc}", file=sys.stderr)
        return 1
    print(summary_df.to_string(index=False))
    print(f"Saved future-forecast model: {out / 'trained_fpl_model.json'} ({deployed_model['training_rows']} training rows).")
    if failures:
        print("Some datasets or model stages failed; see historical_backtest_metadata.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
