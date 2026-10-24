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

from fpl_validation import score_gameweek_group

SOURCES = {
    "2023-24": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2023-24/gws/merged_gw.csv",
    "2024-25": "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv",
    "2025-26": "https://raw.githubusercontent.com/imadeddine-belkat/Premier-League-Stats/main/fpl_scraper/fpl_stats/_merged/players/2025-26_all_players_gw.csv",
}
HEADERS = {"User-Agent": "Mozilla/5.0 The-One-Above-All-FPL/2.9 historical-validation"}


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


def season_frame(season: str, url: str) -> pd.DataFrame:
    raw = fetch_csv(url)
    id_col = pick_column(raw, ["element", "player_id", "player_code", "element_id", "playerid", "id"])
    gw_col = pick_column(raw, ["round", "event", "gameweek", "gw", "gameweek_id"])
    points_col = pick_column(raw, ["total_points", "event_points", "gw_points", "points"])
    minutes_col = pick_column(raw, ["minutes", "mins"], required=False)
    position_col = pick_column(raw, ["element_type", "position", "position_short", "position_name"], required=False)
    xg_col = pick_column(raw, ["expected_goals", "xg", "expected_goals_scored"], required=False)
    xa_col = pick_column(raw, ["expected_assists", "xa"], required=False)
    home_col = pick_column(raw, ["was_home", "is_home", "home"], required=False)
    fixture_col = pick_column(raw, ["fixture", "fixture_code", "fixture_id", "fixtureid"], required=False)
    team_col = pick_column(raw, ["team_code", "team", "team_id", "team_id_x", "team_name"], required=False)
    name_col = pick_column(raw, ["web_name", "name", "second_name", "player_name"], required=False)

    subset = pd.DataFrame({
        "player_id": pd.to_numeric(raw[id_col], errors="coerce"),
        "gameweek": pd.to_numeric(raw[gw_col], errors="coerce"),
        "actual_points": pd.to_numeric(raw[points_col], errors="coerce"),
        "minutes": pd.to_numeric(raw[minutes_col], errors="coerce") if minutes_col else 0.0,
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
        "actual_points": "sum", "minutes": "sum", "xg": "sum", "xa": "sum",
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
        for col in ("actual_points", "minutes", "xg", "xa"):
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
    return panel


def _gw_metrics(group: pd.DataFrame) -> dict:
    return score_gameweek_group(
        group.rename(columns={"prediction_model": "projection_model", "prediction_baseline": "projection_baseline", "prediction_expected_minutes": "projection_expected_minutes", "minutes": "actual_minutes"}),
        actual_col="actual_points", model_col="projection_model", baseline_col="projection_baseline",
        benchmark_col="__none__"
    )


def backtest_season(season: str, url: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = season_frame(season, url)
    panel = add_point_in_time_features(raw)
    # Exclude a player's first available row because their existence might only be known
    # after that GW in an aggregate archive. It also needs prior history to support a forecast.
    data = panel[panel["observed_row"] & panel["season_mean_prev"].notna()].copy()
    if data.empty:
        raise ValueError(f"No eligible point-in-time observations for {season}")
    gw_rows = []
    for gw, group in data.groupby("gameweek"):
        metrics = _gw_metrics(group)
        gw_rows.append({"season": season, "gameweek": int(gw), **metrics})
    gw_df = pd.DataFrame(gw_rows)
    summary = {
        "season": season,
        "data_source": url,
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
        "warning": "Historical proxy using only lagged stats; not an exact replay of old pre-deadline FPL API snapshots. ep_next excluded due archived timing uncertainty.",
    }
    return pd.DataFrame([summary]), gw_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="historical-output")
    parser.add_argument("--seasons", nargs="*", default=["2023-24", "2024-25", "2025-26"])
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries, gameweeks, failures = [], [], []
    for season in args.seasons:
        if season not in SOURCES:
            failures.append({"season": season, "error": "No source configured"})
            continue
        print(f"Downloading and replaying {season} using lagged features only...")
        try:
            summary, gw = backtest_season(season, SOURCES[season])
            summaries.append(summary)
            gameweeks.append(gw)
            print(f"{season}: {int(summary.iloc[0]['gameweeks_evaluated'])} Gameweeks, {int(summary.iloc[0]['player_gameweeks_evaluated'])} player-GWs.")
        except Exception as exc:
            failures.append({"season": season, "error": str(exc)})
            print(f"FAILED {season}: {exc}", file=sys.stderr)
    if not summaries:
        raise RuntimeError(f"No historical season could be evaluated: {failures}")
    summary_df = pd.concat(summaries, ignore_index=True)
    gw_df = pd.concat(gameweeks, ignore_index=True) if gameweeks else pd.DataFrame()
    summary_df.to_csv(out / "historical_backtest_summary.csv", index=False)
    gw_df.to_csv(out / "historical_backtest_gameweeks.csv", index=False)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_note": "Unofficial historical datasets; results are a lagged-stat proxy, not a perfect recreation of historical live API snapshots.",
        "success_seasons": summary_df["season"].tolist(),
        "failures": failures,
    }
    (out / "historical_backtest_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False))
    if failures:
        print("Some datasets were unavailable; see historical_backtest_metadata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
