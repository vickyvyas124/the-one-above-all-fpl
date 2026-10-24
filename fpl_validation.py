"""Leakage-conscious scoring and ranking metrics for FPL forecast snapshots."""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def spearman(actual: Iterable, predicted: Iterable) -> float:
    """Spearman rank correlation, returning NaN when the sample cannot support it."""
    pair = pd.DataFrame({"actual": list(actual), "predicted": list(predicted)}).dropna()
    if len(pair) < 3 or pair["actual"].nunique() < 2 or pair["predicted"].nunique() < 2:
        return float("nan")
    # Compute Spearman as Pearson correlation of average ranks. Calling
    # Series.corr(method="spearman") delegates to scipy.stats in pandas, but
    # SciPy is intentionally not a runtime dependency of this lightweight app
    # and GitHub Actions worker. Ranking first gives the same tie-aware metric.
    actual_rank = pair["actual"].rank(method="average")
    predicted_rank = pair["predicted"].rank(method="average")
    value = actual_rank.corr(predicted_rank)  # default Pearson correlation; no SciPy required
    return float(value) if pd.notna(value) else float("nan")


def top_k_capture(actual: Iterable, predicted: Iterable, player_ids: Iterable, k: int = 10) -> float:
    """Share of actual top-k scorers recovered by predicted top-k among the same eligible set."""
    pair = pd.DataFrame({
        "actual": list(actual), "predicted": list(predicted), "player_id": list(player_ids)
    }).dropna(subset=["actual", "predicted", "player_id"])
    if pair.empty:
        return float("nan")
    n = min(int(k), len(pair))
    if n <= 0:
        return float("nan")
    pair = pair.sort_values("player_id", kind="stable")
    actual_top = set(pair.sort_values(["actual", "player_id"], ascending=[False, True], kind="stable").head(n)["player_id"])
    predicted_top = set(pair.sort_values(["predicted", "player_id"], ascending=[False, True], kind="stable").head(n)["player_id"])
    return len(actual_top & predicted_top) / n


def score_gameweek_group(
    frame: pd.DataFrame,
    actual_col: str = "actual_points",
    model_col: str = "projection_model",
    baseline_col: str = "projection_baseline",
    benchmark_col: str = "projection_ep_next",
    include_minutes_metrics: bool = True,
    include_captain_metrics: bool = True,
) -> dict:
    """Evaluate point error and decision-oriented ranking on matched player sets.

    Minutes and captain diagnostics are optional because they are one-GW quantities;
    they must not be displayed as if they were five-GW metrics.
    """
    actual = _numeric(frame, actual_col)
    model = _numeric(frame, model_col)
    baseline = _numeric(frame, baseline_col)
    ids = frame["player_id"] if "player_id" in frame.columns else pd.Series(frame.index, index=frame.index)
    valid = actual.notna() & model.notna() & baseline.notna()
    data = frame.loc[valid].copy()
    if data.empty:
        return {"Players": 0}
    y = _numeric(data, actual_col)
    m = _numeric(data, model_col)
    b = _numeric(data, baseline_col)
    pid = data["player_id"] if "player_id" in data.columns else data.index
    out = {
        "Players": int(len(data)),
        "Model MAE": float((m - y).abs().mean()),
        "Baseline MAE": float((b - y).abs().mean()),
        "Model bias": float((m - y).mean()),
        "Model Spearman": spearman(y, m),
        "Baseline Spearman": spearman(y, b),
        "Model top-10 capture": top_k_capture(y, m, pid, 10),
        "Baseline top-10 capture": top_k_capture(y, b, pid, 10),
        "Model top-20 capture": top_k_capture(y, m, pid, 20),
        "Baseline top-20 capture": top_k_capture(y, b, pid, 20),
    }
    # For five-GW outcomes, use summed minutes across that same horizon when
    # deciding whether a player appeared at all. Do not accidentally use only the
    # first GW's minutes for a multi-GW played-only metric.
    minutes_col = ("actual_minutes_5gw" if actual_col == "actual_points_5gw"
                   and "actual_minutes_5gw" in data.columns else "actual_minutes")
    minutes = _numeric(data, minutes_col)
    played_mask = minutes.fillna(0) > 0
    if played_mask.any():
        out["Model MAE · played"] = float((m[played_mask] - y[played_mask]).abs().mean())
        out["Baseline MAE · played"] = float((b[played_mask] - y[played_mask]).abs().mean())
        out["Players who played"] = int(played_mask.sum())
    else:
        out["Model MAE · played"] = float("nan")
        out["Baseline MAE · played"] = float("nan")
        out["Players who played"] = 0

    if include_minutes_metrics:
        # Minutes are scored separately because a points miss can be caused by a missed
        # start/injury rather than a poor estimate of output when on the pitch.
        expected_minutes = _numeric(data, "projection_expected_minutes")
        minutes_valid = minutes.notna() & expected_minutes.notna()
        if minutes_valid.any():
            out["Expected-minutes MAE"] = float((expected_minutes[minutes_valid] - minutes[minutes_valid]).abs().mean())
            out["Expected-minutes bias"] = float((expected_minutes[minutes_valid] - minutes[minutes_valid]).mean())
            out["Minutes scored players"] = int(minutes_valid.sum())
        else:
            out["Expected-minutes MAE"] = float("nan")
            out["Expected-minutes bias"] = float("nan")
            out["Minutes scored players"] = 0

    # Captain regret is only computed when a complete public 15-player squad was
    # available pre-deadline and the log contains the model's XI/captain decision.
    # It compares captaincy points opportunity within the model-selected XI; it does
    # not infer a lineup after the match.
    if include_captain_metrics and actual_col == "actual_points" and "xi_eligible" in data.columns and "model_captain_pick" in data.columns:
        def _truthy(value):
            if pd.isna(value):
                return False
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"true", "1", "yes", "y"}

        xi_mask = data["xi_eligible"].map(_truthy)
        captain_mask = data["model_captain_pick"].map(_truthy)
        xi_data = data.loc[xi_mask & data[actual_col].notna()].copy()
        selected_captains = data.loc[captain_mask & data[actual_col].notna()].copy()
        if len(xi_data) >= 11 and len(selected_captains) == 1:
            selected_actual = float(pd.to_numeric(selected_captains[actual_col], errors="coerce").iloc[0])
            best_actual = float(pd.to_numeric(xi_data[actual_col], errors="coerce").max())
            out["Model captain actual pts"] = selected_actual
            out["Best XI actual captain pts"] = best_actual
            out["Captain regret (points)"] = max(0.0, best_actual - selected_actual)
            out["Captain regret coverage"] = 1
        else:
            out["Model captain actual pts"] = float("nan")
            out["Best XI actual captain pts"] = float("nan")
            out["Captain regret (points)"] = float("nan")
            out["Captain regret coverage"] = 0

    if benchmark_col in data.columns:
        ep = _numeric(data, benchmark_col)
        ep_mask = ep.notna()
        out["EP next coverage %"] = float(ep_mask.mean() * 100)
        if ep_mask.sum() >= 3:
            out["Model MAE · EP cohort"] = float((m[ep_mask] - y[ep_mask]).abs().mean())
            out["Baseline MAE · EP cohort"] = float((b[ep_mask] - y[ep_mask]).abs().mean())
            out["EP next MAE"] = float((ep[ep_mask] - y[ep_mask]).abs().mean())
            out["EP next Spearman"] = spearman(y[ep_mask], ep[ep_mask])
            out["Model Spearman · EP cohort"] = spearman(y[ep_mask], m[ep_mask])
            out["EP next top-10 capture"] = top_k_capture(y[ep_mask], ep[ep_mask], pid[ep_mask], 10)
            out["EP next top-20 capture"] = top_k_capture(y[ep_mask], ep[ep_mask], pid[ep_mask], 20)
        else:
            for key in ("Model MAE · EP cohort", "Baseline MAE · EP cohort", "EP next MAE", "EP next Spearman", "Model Spearman · EP cohort", "EP next top-10 capture", "EP next top-20 capture"):
                out[key] = float("nan")
    return out


def evaluate_log(
    frame: pd.DataFrame,
    actual_col: str = "actual_points",
    model_col: str = "projection_model",
    baseline_col: str = "projection_baseline",
    benchmark_col: str = "projection_ep_next",
    group_cols: tuple[str, ...] = ("season", "gameweek"),
    include_minutes_metrics: bool = True,
    include_captain_metrics: bool = True,
) -> pd.DataFrame:
    """Return one row of metrics per forecast snapshot group."""
    needed = {actual_col, model_col, baseline_col}
    if frame.empty or not needed.issubset(frame.columns):
        return pd.DataFrame()
    work = frame.copy()
    for col in needed | ({benchmark_col} if benchmark_col in work.columns else set()):
        work[col] = _numeric(work, col)
    for col in group_cols:
        if col not in work.columns:
            return pd.DataFrame()
    rows = []
    for key, group in work.groupby(list(group_cols), dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        scored = group[group[actual_col].notna()]
        if scored.empty:
            continue
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(score_gameweek_group(
            scored, actual_col, model_col, baseline_col, benchmark_col,
            include_minutes_metrics=include_minutes_metrics,
            include_captain_metrics=include_captain_metrics,
        ))
        if "history_enriched" in scored.columns:
            hist = scored["history_enriched"].map(lambda x: str(x).lower() in {"true", "1", "yes"})
            row["History coverage %"] = float(hist.mean() * 100)
        rows.append(row)
    return pd.DataFrame(rows)
