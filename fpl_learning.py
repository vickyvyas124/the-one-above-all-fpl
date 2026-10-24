"""Lightweight supervised FPL predictor shared by historical backtest and live scoring.

Uses a ridge-regression model trained on lagged Gameweek features. It relies on
NumPy (already installed with pandas), not SciPy/scikit-learn. The model artifact
is versioned JSON so the Streamlit app and scheduled worker use identical weights.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODEL_VERSION = "3.0"
POSITION_FEATURES = ["position_GKP", "position_DEF", "position_MID", "position_FWD"]
NUMERIC_FEATURES = [
    "season_mean_prev",
    "recent_mean_prev",
    "recent_minutes_per_fixture_prev",
    "recent_appearance_rate_prev",
    "recent_start_rate_prev",
    "season_xgi_per90_prev",
    "recent_xgi_per90_prev",
    "fixture_count",
    "home_share",
]
FEATURES = POSITION_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "actual_points"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def position_features(position: Any) -> dict[str, float]:
    text = str(position or "").strip().upper()
    mapping = {"1": "GKP", "GK": "GKP", "GKP": "GKP", "2": "DEF", "D": "DEF", "DEF": "DEF",
               "3": "MID", "M": "MID", "MID": "MID", "4": "FWD", "F": "FWD", "FW": "FWD", "FWD": "FWD"}
    text = mapping.get(text, text)
    return {f"position_{pos}": 1.0 if text == pos else 0.0 for pos in ("GKP", "DEF", "MID", "FWD")}


def make_feature_row(
    *, position: Any, history_signal: Mapping[str, Any] | None,
    fixture_count: int, home_share: float,
    fallback_season_mean: float = 0.0, fallback_recent_mean: float = 0.0,
    fallback_xgi_per90: float = 0.0,
) -> dict[str, float]:
    """Build inference features matching historical_backtest.add_point_in_time_features."""
    hist = dict(history_signal or {})
    row = position_features(position)
    season_mean = hist.get("season_points_per_gw")
    recent_mean = hist.get("recent_points_per_gw")
    if season_mean is None:
        season_mean = fallback_season_mean
    if recent_mean is None:
        recent_mean = fallback_recent_mean
    row.update({
        "season_mean_prev": _safe_float(season_mean),
        "recent_mean_prev": _safe_float(recent_mean),
        "recent_minutes_per_fixture_prev": _safe_float(hist.get("recent_minutes_per_fixture")),
        "recent_appearance_rate_prev": _safe_float(hist.get("recent_appearance_rate")),
        "recent_start_rate_prev": _safe_float(hist.get("recent_start_rate")),
        "season_xgi_per90_prev": _safe_float(hist.get("season_xgi_per90"), fallback_xgi_per90),
        "recent_xgi_per90_prev": _safe_float(hist.get("recent_xgi_per90"), fallback_xgi_per90),
        "fixture_count": float(max(0, min(3, int(fixture_count)))),
        "home_share": max(0.0, min(1.0, _safe_float(home_share, 0.5))),
    })
    return row


def fit_ridge_model(frame, *, source_seasons: list[str] | None = None, alpha: float = 25.0) -> dict[str, Any]:
    """Fit a standardized ridge regression on labelled player-Gameweek rows."""
    import pandas as pd

    missing = [name for name in FEATURES + [TARGET_COLUMN] if name not in frame.columns]
    if missing:
        raise ValueError(f"Training frame is missing model columns: {missing}")
    clean = frame[FEATURES + [TARGET_COLUMN]].copy()
    for col in FEATURES + [TARGET_COLUMN]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[TARGET_COLUMN])
    clean[FEATURES] = clean[FEATURES].fillna(0.0)
    if len(clean) < 500:
        raise ValueError(f"Need at least 500 labelled player-Gameweeks to fit; got {len(clean)}")

    x = clean[FEATURES].to_numpy(dtype=float)
    y = clean[TARGET_COLUMN].to_numpy(dtype=float)
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-8] = 1.0
    z = (x - means) / scales
    y_mean = float(y.mean())
    y_centered = y - y_mean
    gram = z.T @ z
    # Do not penalize the intercept; position dummies and continuous inputs are regularized.
    beta = np.linalg.solve(gram + float(alpha) * np.eye(gram.shape[0]), z.T @ y_centered)
    return {
        "model_version": MODEL_VERSION,
        "model_type": "standardized_ridge_regression",
        "target": "total FPL points in target Gameweek",
        "features": list(FEATURES),
        "means": {k: float(v) for k, v in zip(FEATURES, means)},
        "scales": {k: float(v) for k, v in zip(FEATURES, scales)},
        "coefficients": {k: float(v) for k, v in zip(FEATURES, beta)},
        "intercept": y_mean,
        "ridge_alpha": float(alpha),
        "training_rows": int(len(clean)),
        "source_seasons": list(source_seasons or []),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_target_mean": y_mean,
    }


def predict_features(model: Mapping[str, Any] | None, row: Mapping[str, Any]) -> float | None:
    """Return non-negative expected points for one Gameweek, or None without a valid model."""
    if not model or str(model.get("model_version")) != MODEL_VERSION:
        return None
    try:
        features = list(model["features"])
        if features != FEATURES:
            return None
        total = _safe_float(model.get("intercept"))
        means = model["means"]
        scales = model["scales"]
        coefficients = model["coefficients"]
        for name in features:
            scale = max(1e-8, _safe_float(scales.get(name), 1.0))
            value = _safe_float(row.get(name))
            total += _safe_float(coefficients.get(name)) * ((value - _safe_float(means.get(name))) / scale)
        return round(max(0.0, min(35.0, total)), 3)
    except (KeyError, TypeError, ValueError):
        return None


def save_model(model: Mapping[str, Any], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")


def load_model(path: str | Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if str(payload.get("model_version")) == MODEL_VERSION and payload.get("features") == FEATURES else None
    except (OSError, ValueError, TypeError):
        return None
