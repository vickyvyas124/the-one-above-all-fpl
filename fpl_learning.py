"""Lightweight supervised FPL predictor shared by historical backtest and live scoring.

Uses a ridge-regression model trained on lagged Gameweek features. It relies on
NumPy (already installed with pandas), not SciPy/scikit-learn. The model artifact
is versioned JSON so the Streamlit app and scheduled worker use identical weights.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

MODEL_VERSION = "3.0"
FEATURE_POLICY = "point_in_time_lagged_no_target_schedule_v1"
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
    # Training fills unavailable lagged history with zero, so new players and
    # history-fetch failures must use the same value at inference. Bootstrap PPG,
    # form and cumulative xGI are not interchangeable with the lagged training data.
    season_mean = hist.get("season_points_per_gw")
    recent_mean = hist.get("recent_points_per_gw")
    # Historical archives do not record what fixture assignments were known at
    # each old deadline. Keep the public model schema stable, but make schedule
    # fields neutral in both training and inference. The engine applies the live
    # published fixture count after predicting a single-fixture expectation.
    row.update({
        "season_mean_prev": _safe_float(season_mean),
        "recent_mean_prev": _safe_float(recent_mean),
        "recent_minutes_per_fixture_prev": _safe_float(hist.get("recent_minutes_per_fixture")),
        "recent_appearance_rate_prev": _safe_float(hist.get("recent_appearance_rate")),
        "recent_start_rate_prev": _safe_float(hist.get("recent_start_rate")),
        "season_xgi_per90_prev": _safe_float(hist.get("season_xgi_per90")),
        "recent_xgi_per90_prev": _safe_float(hist.get("recent_xgi_per90")),
        "fixture_count": 1.0,
        "home_share": 0.5,
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
    clean["fixture_count"] = 1.0
    clean["home_share"] = 0.5
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
        "feature_policy": FEATURE_POLICY,
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
    if not validate_model(model):
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
    if not validate_model(model):
        raise ValueError("Refusing to save an invalid or incompatible v3.0 model artifact")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(model, indent=2, sort_keys=True), encoding="utf-8")


def load_model(path: str | Path) -> dict[str, Any] | None:
    """Return a validated model, or None for any failure.

    Use :func:`inspect_model` when the *reason* for a failure matters.
    """
    return inspect_model(path).model


@dataclass(frozen=True)
class ModelInspection:
    """Outcome of loading a model artifact, with an actionable failure reason.

    status is one of: valid, missing, unreadable, malformed, incompatible, invalid.
    ``model`` is only set when status == "valid"; a failed load never yields a model.
    """
    status: str
    reason: str
    model: dict[str, Any] | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == "valid"


REMEDIATION = {
    "missing": "Run the 'Historical FPL Backtest' workflow on main to publish trained_fpl_model.json to the fpl-data branch.",
    "unreadable": "Check that the artifact path is a readable UTF-8 file on the fpl-data branch checkout.",
    "malformed": "Re-run the 'Historical FPL Backtest' workflow on main to replace the corrupted artifact.",
    "incompatible": "The artifact was built for a different model contract; re-run the 'Historical FPL Backtest' workflow on main with the current code.",
    "invalid": "Re-run the 'Historical FPL Backtest' workflow on main to regenerate the artifact.",
}


def inspect_model(path: str | Path) -> ModelInspection:
    """Load an artifact and say exactly why it cannot be used, if it cannot."""
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ModelInspection("missing", f"no artifact file at {target}")
    except (OSError, UnicodeDecodeError) as exc:
        return ModelInspection("unreadable", f"{target} could not be read ({type(exc).__name__}: {exc})")
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return ModelInspection("malformed", f"{target.name} is not valid JSON ({exc})")
    problem = model_problem(payload)
    if problem is not None:
        return ModelInspection(problem[0], problem[1])
    return ModelInspection("valid", f"v{MODEL_VERSION} model, {int(payload['training_rows'])} training rows from {payload['source_seasons']}", payload)


def validate_model(model: Mapping[str, Any] | None) -> bool:
    """Validate the complete deployable artifact contract before serving it."""
    return model_problem(model) is None


def model_problem(model: Any) -> tuple[str, str] | None:
    """Return (category, reason) for the first contract violation, or None if valid.

    Categories: "malformed" (not a JSON object), "incompatible" (a different
    version/policy/type/target/feature set) and "invalid" (right contract, bad content).
    """
    if not isinstance(model, Mapping):
        return "malformed", f"top-level JSON value is {type(model).__name__}, expected an object"
    expected = (
        ("model_version", MODEL_VERSION, str(model.get("model_version"))),
        ("feature_policy", FEATURE_POLICY, model.get("feature_policy")),
        ("model_type", "standardized_ridge_regression", model.get("model_type")),
        ("target", "total FPL points in target Gameweek", model.get("target")),
    )
    for name, want, got in expected:
        if got != want:
            return "incompatible", f"{name} is {got!r}, this code requires {want!r}"
    if model.get("features") != FEATURES:
        return "incompatible", "feature list differs from the features this code scores with"
    try:
        for name in ("means", "scales", "coefficients"):
            values = model[name]
            if not isinstance(values, Mapping) or set(values) != set(FEATURES):
                return "invalid", f"{name} must contain exactly the model features"
            if not all(math.isfinite(float(values[key])) for key in FEATURES):
                return "invalid", f"{name} contains a non-finite value"
        if any(float(model["scales"][key]) <= 0 for key in FEATURES):
            return "invalid", "scales must all be positive"
        if not math.isfinite(float(model["intercept"])):
            return "invalid", "intercept is not finite"
        if int(model.get("training_rows", 0)) < 500:
            return "invalid", f"training_rows is {model.get('training_rows')!r}, need at least 500"
        if not isinstance(model.get("source_seasons"), list) or not model["source_seasons"]:
            return "invalid", "source_seasons must be a non-empty list"
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        return "invalid", f"artifact content is unusable ({type(exc).__name__}: {exc})"
    return None


def main(argv: list[str] | None = None) -> int:
    """``python fpl_learning.py check PATH [--strict]`` - report whether an artifact is usable.

    Exit status is 0 for a valid artifact. For an unusable one it is 0 by default
    (advisory) and 3 with --strict, so publish steps can refuse to ship it.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a trained FPL model artifact.")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="validate a trained_fpl_model.json file")
    check.add_argument("path")
    check.add_argument("--strict", action="store_true", help="exit 3 when the artifact is unusable")
    args = parser.parse_args(argv)
    result = inspect_model(args.path)
    if result.is_valid:
        print(f"MODEL-ARTIFACT: status=valid; {result.reason}")
        return 0
    print(f"MODEL-ARTIFACT: status={result.status}; {result.reason}. Action: {REMEDIATION[result.status]}")
    return 3 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
