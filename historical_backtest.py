"""Leakage-conscious historical FPL proxy backtest for completed prior seasons.

The script uses per-Gameweek results but creates features only from earlier Gameweeks.
It is a historical proxy for the style of model, not an exact replay of old FPL API
snapshots. `ep_next` is intentionally excluded because archived `ep_this`/`xP` fields
can be collected after deadlines and may leak outcomes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
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
    keys = values.astype("string").str.strip()
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
    fixture_col = pick_column(raw, ["fixture", "fixture_code", "fixture_id", "fixtureid"], required=False)
    team_col = pick_column(raw, ["team_code", "team", "team_id", "team_id_x", "team_name"], required=False)
    name_col = pick_column(raw, ["web_name", "name", "second_name", "player_name"], required=False)

    if gw_col:
        gameweek_values = pd.to_numeric(raw[gw_col], errors="coerce")
    elif fixture_col and season in FIXTURE_SOURCES:
        fixture_data = fetch_csv(FIXTURE_SOURCES[season])
        fixture_code_col = pick_column(fixture_data, ["code", "fixture_code", "fixture", "fixture_id", "id"])
        fixture_gw_col = pick_column(fixture_data, ["event", "round", "gameweek", "gw"])
        fixture_map = pd.DataFrame({
            "_fixture_key": _fixture_key_series(fixture_data[fixture_code_col]),
            "_gameweek": pd.to_numeric(fixture_data[fixture_gw_col], errors="coerce"),
        }).dropna(subset=["_fixture_key", "_gameweek"]).drop_duplicates("_fixture_key", keep="last")
        code_to_gameweek = dict(zip(fixture_map["_fixture_key"], fixture_map["_gameweek"]))
        gameweek_values = _fixture_key_series(raw[fixture_col]).map(code_to_gameweek)
        if gameweek_values.notna().sum() == 0:
            raise ValueError(
                f"No fixture_code values matched the season fixture lookup for {season}. "
                f"Player fixture column: {fixture_col}; lookup code column: {fixture_code_col}."
            )
        gameweek_values = pd.to_numeric(gameweek_values, errors="coerce")
    else:
        raise ValueError(
            f"Missing Gameweek column and no fixture lookup is configured for {season}. "
            f"Available columns: {list(raw.columns)[:60]}"
        )

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
        "player": raw[name_col].astype(str) if name_col else "Player",
        "team_key": raw[team_col].astype("string") if team_col else "__unknown_team__",
    })
    if fixture_col:
        subset["_fixture_id"] = raw[fixture_col].astype("string")
    else:
        subset["_fixture_id"] = pd.Series(pd.NA, index=subset.index, dtype="string")
    subset = subset.dropna(subset=["player_id", "gameweek", "actual_points"])

    # Infer the team's scheduled fixture count from the union of fixture IDs for the
    # team/Gameweek, rather than from an individual player's rows. A player's absence
    # from a fixture must not decide whether the model forecasts a single or double GW.
    schedule_source_available = bool(fixture_col and team_col)
    if schedule_source_available:
        scheduled = subset.dropna(subset=["_fixture_id"]).drop_duplicates(["team_key", "gameweek", "_fixture_id"])
        schedule_stats = scheduled.groupby(["team_key", "gameweek"], as_index=False).agg(
            scheduled_fixture_count=("_fixture_id", "nunique"),
            schedule_was_home=("was_home", "mean"),
        )
        subset = subset.merge(schedule_stats, on=["team_key", "gameweek"], how="left")
        # If the archive exposes fixture IDs, missing fixtures in a team's GW are a
        # blank (zero fixtures), not an assumed single fixture.
        subset["scheduled_fixture_count"] = pd.to_numeric(
            subset["scheduled_fixture_count"], errors="coerce"
        ).fillna(0.0).clip(0, 3)
        subset["schedule_was_home"] = pd.to_numeric(subset["schedule_was_home"], errors="coerce")
    else:
        subset["scheduled_fixture_count"] = 1.0
        subset["schedule_was_home"] = subset["was_home"]
    subset["schedule_source_available"] = 1 if schedule_source_available else 0
    subset["schedule_was_home"] = pd.to_numeric(subset["schedule_was_home"], errors="coerce")
    subset["player_id"] = subset["player_id"].astype(int)
    subset["gameweek"] = subset["gameweek"].astype(int)
    subset = subset[subset["gameweek"].between(1, 50)]
    if subset.empty:
        raise ValueError(f"No usable rows found for {season}")
    agg = {
        "actual_points": "sum", "minutes": "sum", "starts": "sum", "xg": "sum", "xa": "sum",
        "position": "last", "was_home": "mean", "player": "last", "team_key": "last",
        "scheduled_fixture_count": "max", "schedule_was_home": "mean",
        "schedule_source_available": "max",
    }
    grouped = subset.groupby(["player_id", "gameweek"], as_index=False).agg(agg)
    fixture_values = pd.to_numeric(grouped["scheduled_fixture_count"], errors="coerce")
    if schedule_source_available:
        grouped["fixture_count"] = fixture_values.fillna(0).clip(0, 3)
    else:
        grouped["fixture_count"] = fixture_values.fillna(1).clip(1, 3)
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
        merged["position"] = merged["position"].ffill().bfill().fillna("UNK")
        merged["player"] = merged["player"].ffill().bfill().fillna("Player")
        merged["team_key"] = merged["team_key"].ffill().bfill().fillna("__unknown_team__")
        merged["was_home"] = pd.to_numeric(merged["schedule_was_home"], errors="coerce")
        panels.append(merged)
    panel = pd.concat(panels, ignore_index=True).sort_values(["player_id", "gameweek"])
    schedule_lookup = (season_data.groupby(["team_key", "gameweek"], as_index=False)
                       .agg(fixture_count=("fixture_count", "max"), schedule_was_home=("was_home", "mean")))
    panel = panel.drop(columns=["fixture_count", "schedule_was_home"], errors="ignore").merge(
        schedule_lookup, on=["team_key", "gameweek"], how="left"
    )
    schedule_known = bool(pd.to_numeric(season_data.get("schedule_source_available", pd.Series([0])), errors="coerce").fillna(0).max() > 0)
    panel["fixture_count"] = pd.to_numeric(panel["fixture_count"], errors="coerce").fillna(0 if schedule_known else 1).clip(0 if schedule_known else 1, 3)
    panel["was_home"] = pd.to_numeric(panel["schedule_was_home"], errors="coerce")
    by_player = panel.groupby("player_id", sort=False)
    panel["season_mean_prev"] = by_player["actual_points"].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    panel["recent_mean_prev"] = by_player["actual_points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    # Historical GW rows may sum minutes over double fixtures. Convert prior GWs to
    # per-fixture minutes first, then scale by the target GW's team schedule.
    panel["minutes_per_fixture"] = panel["minutes"].div(panel["fixture_count"].replace(0, float("nan"))).fillna(0)
    panel["recent_minutes_per_fixture_prev"] = panel.groupby("player_id", sort=False)["minutes_per_fixture"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).mean()
    )
    panel["prediction_expected_minutes"] = (
        panel["recent_minutes_per_fixture_prev"].fillna(0).clip(lower=0, upper=90) * panel["fixture_count"]
    )
    panel["recent_appearance_rate_prev"] = by_player["minutes"].transform(
        lambda values: values.gt(0).astype(float).shift(1).rolling(5, min_periods=1).mean()
    )
    panel["recent_start_rate_prev"] = by_player["starts"].transform(
        lambda values: values.gt(0).astype(float).shift(1).rolling(5, min_periods=1).mean()
    )
    panel["xgi"] = panel["xg"] + panel["xa"]
    panel["recent_xgi_prev"] = panel.groupby("player_id", sort=False)["xgi"].transform(
        lambda values: values.shift(1).rolling(5, min_periods=1).sum()
    )
    panel["recent_minutes_sum_prev"] = by_player["minutes"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).sum())

    pos = panel[panel["observed_row"]].groupby(["position", "gameweek"], as_index=False)["actual_points"].mean()
    pos = pos.sort_values(["position", "gameweek"])
    pos["pos_mean_prev"] = pos.groupby("position")["actual_points"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    panel = panel.merge(pos[["position", "gameweek", "pos_mean_prev"]], on=["position", "gameweek"], how="left")
    defaults = {"GKP": 2.6, "DEF": 2.8, "MID": 3.0, "FWD": 3.0}
    panel["pos_mean_prev"] = panel["pos_mean_prev"].fillna(panel["position"].map(defaults)).fillna(2.8)

    panel["xgi_per90_prev"] = panel["recent_xgi_prev"] / panel["recent_minutes_sum_prev"].clip(lower=90) * 90
    panel["recent_xgi_per90_prev"] = panel["xgi_per90_prev"]
    season_xgi_prev = by_player["xgi"].transform(lambda values: values.shift(1).expanding(min_periods=1).sum())
    season_minutes_prev = by_player["minutes"].transform(lambda values: values.shift(1).expanding(min_periods=1).sum())
    panel["season_xgi_per90_prev"] = season_xgi_prev / season_minutes_prev.clip(lower=90) * 90
    # Position dummies keep the model interpretable and consistent at inference.
    for pos in ("GKP", "DEF", "MID", "FWD"):
        panel[f"position_{pos}"] = (panel["position"].astype(str).str.upper() == pos).astype(float)
    panel["home_share"] = pd.to_numeric(panel["was_home"], errors="coerce").fillna(0.5).clip(0, 1)
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
    panel["fixture_multiplier"] = pd.to_numeric(panel["fixture_count"], errors="coerce").fillna(0 if schedule_known else 1).clip(lower=0 if schedule_known else 1, upper=3)
    home = panel["was_home"].fillna(0.5).clip(lower=0, upper=1)
    panel["venue_factor"] = 0.95 + 0.10 * home
    panel["prediction_model"] = panel["prediction_model"] * panel["fixture_multiplier"] * panel["venue_factor"]
    panel["prediction_baseline"] = panel["prediction_baseline"] * panel["fixture_multiplier"] * panel["venue_factor"]
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


def _score_panel(season: str, panel: pd.DataFrame, model: dict | None, model_type: str, url: str):
    data = _eligible_rows(panel)
    if data.empty:
        raise ValueError(f"No eligible point-in-time observations for {season}")
    baseline = data["season_mean_prev"].fillna(data["pos_mean_prev"]).clip(lower=0)
    fixture_count = pd.to_numeric(data["fixture_count"], errors="coerce").fillna(0).clip(0, 3)
    home = pd.to_numeric(data["home_share"], errors="coerce").fillna(0.5).clip(0, 1)
    data["prediction_baseline"] = baseline * fixture_count * (0.95 + 0.10 * home)
    data["prediction_model"] = float("nan")
    if model:
        rows = data[fpl_learning.FEATURES].to_dict("records")
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
        gw_rows.append({"season": season, "gameweek": int(gw), "model_type": model_type, **metrics})
    gw_df = pd.DataFrame(gw_rows)
    summary = {
        "season": season,
        "data_source": url,
        "model_type": model_type,
        "trained_model_used": bool(model),
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
        "warning": "Expanding-season holdout using only lagged features. The first source season uses a non-trained fallback because no earlier season is configured. Archived player/team snapshots are not exact historical pre-deadline FPL API states.",
    }
    return pd.DataFrame([summary]), gw_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="historical-output")
    parser.add_argument("--seasons", nargs="*", default=["2023-24", "2024-25", "2025-26"])
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels: dict[str, pd.DataFrame] = {}
    failures = []
    for season in args.seasons:
        if season not in SOURCES:
            failures.append({"season": season, "error": "No source configured"})
            continue
        print(f"Downloading and engineering lagged features for {season}...")
        try:
            panels[season] = add_point_in_time_features(season_frame(season, SOURCES[season]))
            print(f"{season}: {int(panels[season]['gameweek'].nunique())} Gameweeks, {int(len(_eligible_rows(panels[season])))} eligible player-GWs.")
        except Exception as exc:
            failures.append({"season": season, "error": str(exc)})
            print(f"FAILED {season}: {exc}", file=sys.stderr)

    if not panels:
        raise RuntimeError(f"No historical season could be evaluated: {failures}")

    ordered_seasons = [season for season in args.seasons if season in panels]
    summaries, gameweeks = [], []
    for idx, season in enumerate(ordered_seasons):
        prior_seasons = ordered_seasons[:idx]
        prior_frames = [_eligible_rows(panels[prior]) for prior in prior_seasons]
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
        summary, gw = _score_panel(season, panels[season], trained, model_type, SOURCES[season])
        summaries.append(summary)
        gameweeks.append(gw)

    # The deployable model for future Gameweeks is fitted on the full available
    # historical set. Historical metrics above remain strictly out-of-season.
    final_frames = [_eligible_rows(panels[season]) for season in ordered_seasons]
    final_training = pd.concat(final_frames, ignore_index=True)
    deployed_model = fpl_learning.fit_ridge_model(final_training, source_seasons=ordered_seasons)
    deployed_model["evaluation_note"] = "Historical metrics are expanding-season holdouts; deployable weights are refit on all available historical rows for future-only predictions."
    fpl_learning.save_model(deployed_model, out / "trained_fpl_model.json")

    summary_df = pd.concat(summaries, ignore_index=True)
    gw_df = pd.concat(gameweeks, ignore_index=True) if gameweeks else pd.DataFrame()
    summary_df.to_csv(out / "historical_backtest_summary.csv", index=False)
    gw_df.to_csv(out / "historical_backtest_gameweeks.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_note": "Historical data is used to train a ridge-regression model on lagged player form/minutes/xGI and target Gameweek schedule. Out-of-season metrics use only preceding seasons. Archived data is not an exact replay of historical live API snapshots.",
        "success_seasons": summary_df["season"].tolist(),
        "failures": failures,
        "deployed_model": {"model_version": deployed_model["model_version"], "model_type": deployed_model["model_type"], "training_rows": deployed_model["training_rows"], "source_seasons": deployed_model["source_seasons"], "feature_count": len(deployed_model["features"])},
    }
    (out / "historical_backtest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    print(f"Saved future-forecast model: {out / 'trained_fpl_model.json'} ({deployed_model['training_rows']} training rows).")
    if failures:
        print("Some datasets or model stages failed; see historical_backtest_metadata.json", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
