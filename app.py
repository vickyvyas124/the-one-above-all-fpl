import streamlit as st
import pandas as pd
import requests
import html
import json
import re
import os
from pathlib import Path
import fpl_engine
import fpl_learning
from fpl_availability import availability_adjustment
from fpl_ui_labels import status_chip_labels
from fpl_lineup import score_starting_xi, select_model_lineup
from fpl_squad import load_public_squad, validate_squad_ids
from fpl_transfer import (
    gameweek_projection_series,
    is_affordable_tenths,
    official_selling_price_tenths,
    transfer_budget_tenths,
    transfer_hit_cost,
    transfer_projection_comparison,
    optimize_transfer_plans,
    transfer_plan_action_threshold,
    verified_player_price_tenths,
    exact_fpl_tenths,
    pounds_millions_to_tenths,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

st.set_page_config(page_title="The-One-Above-All | FPL Manager Hub", layout="wide")
APP_VERSION = "3.0"
FORECAST_LOG_PATH = Path(__file__).resolve().parent / "forecast_log.csv"
FORECAST_REMOTE_URL = "https://raw.githubusercontent.com/vickyvyas124/the-one-above-all-fpl/fpl-data/forecast_log.csv"
HISTORICAL_SUMMARY_URL = "https://raw.githubusercontent.com/vickyvyas124/the-one-above-all-fpl/fpl-data/historical_backtest_summary.csv"
HISTORICAL_GW_URL = "https://raw.githubusercontent.com/vickyvyas124/the-one-above-all-fpl/fpl-data/historical_backtest_gameweeks.csv"
TRAINED_MODEL_URL = "https://raw.githubusercontent.com/vickyvyas124/the-one-above-all-fpl/fpl-data/trained_fpl_model.json"
FORECAST_COLUMNS = [
    "season", "gameweek", "player_id", "player", "position", "club",
    "model_version", "forecast_created_utc", "projection_model", "projection_baseline",
    "projection_ep_next", "projection_model_5gw", "projection_baseline_5gw", "forecast_horizon_gws",
    "history_enriched", "history_sample_matches", "projection_expected_minutes", "xi_eligible", "model_captain_pick",
    "actual_points", "actual_minutes", "actual_points_5gw", "actual_minutes_5gw",
]
PLAYER_HISTORY_SIGNALS = {}
MODEL_CALIBRATION_OFFSETS = {}
CALIBRATION_METADATA = {"enabled": False, "completed_gws": 0, "samples_by_position": {}, "offsets": {}}
BASE = "https://fantasy.premierleague.com/api"

@st.cache_data(ttl=1800, show_spinner=False)
def get_json(path):
    r = requests.get(f"{BASE}/{path}", timeout=20, headers={"User-Agent":"The-One-Above-All-FPL/2.9"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60, show_spinner=False)
def get_manager_profile(team_id):
    """Short-cache public profile so rank/points refresh more frequently."""
    r = requests.get(f"{BASE}/entry/{int(team_id)}/", timeout=20,
                     headers={"User-Agent":"Mozilla/5.0 The-One-Above-All-FPL/2.9"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=120, show_spinner=False)
def get_entry_picks(team_id, gameweek):
    """Public picks for a specific Gameweek; may not expose editable picks before deadline."""
    r = requests.get(f"{BASE}/entry/{int(team_id)}/event/{int(gameweek)}/picks/",
                     timeout=20, headers={"User-Agent":"Mozilla/5.0 The-One-Above-All-FPL/2.9"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=120, show_spinner=False)
def get_entry_history(team_id):
    r = requests.get(f"{BASE}/entry/{int(team_id)}/history/",
                     timeout=20, headers={"User-Agent":"Mozilla/5.0 The-One-Above-All-FPL/2.9"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=21600, show_spinner=False)
def get_trained_model_payload():
    """Load the versioned model fitted and published by the historical backtest workflow."""
    try:
        r = requests.get(TRAINED_MODEL_URL, timeout=8, headers={"User-Agent":"The-One-Above-All-FPL/3.0"})
        r.raise_for_status()
        payload = r.json()
        if not fpl_learning.validate_model(payload):
            return None
        return payload
    except Exception:
        return None

# Dashboard and background worker consume the same model artifact.
fpl_engine.TRAINED_MODEL = get_trained_model_payload()

@st.cache_data(ttl=21600, show_spinner=False)
def get_player_summary(player_id):
    """Match-by-match public player history and remaining fixtures; cached for six hours."""
    r = requests.get(f"{BASE}/element-summary/{int(player_id)}/", timeout=15,
                     headers={"User-Agent":"Mozilla/5.0 The-One-Above-All-FPL/2.9"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=1800, show_spinner=False)
def get_live_gameweek(gameweek):
    """Public actual Gameweek points and minutes for forecast evaluation."""
    r = requests.get(f"{BASE}/event/{int(gameweek)}/live/", timeout=15,
                     headers={"User-Agent":"Mozilla/5.0 The-One-Above-All-FPL/2.9"})
    r.raise_for_status()
    return r.json()

def make_history_signal(summary, cutoff_utc=None):
    return fpl_engine.make_history_signal(summary, cutoff_utc=cutoff_utc)

def season_label(now=None):
    now = now or datetime.now(timezone.utc)
    start_year = now.year if now.month >= 7 else now.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"

@st.cache_data(ttl=300, show_spinner=False)
def get_remote_forecast_log():
    """Load persistent forecast history from the automation data branch."""
    try:
        r = requests.get(FORECAST_REMOTE_URL, timeout=5, headers={"User-Agent":"The-One-Above-All-FPL/2.9"})
        if r.status_code != 200 or not r.text.strip():
            return None
        import io
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None




@st.cache_data(ttl=1800, show_spinner=False)
def get_remote_historical_backtest():
    """Read results from the manual, point-in-time historical backtest workflow."""
    try:
        r = requests.get(HISTORICAL_SUMMARY_URL, timeout=8, headers={"User-Agent": "The-One-Above-All-FPL/2.9"})
        if r.status_code != 200 or not r.text.strip():
            return None
        import io
        return pd.read_csv(io.StringIO(r.text))
    except Exception:
        return None

def read_forecast_log():
    remote = get_remote_forecast_log()
    if isinstance(remote, pd.DataFrame):
        frame = remote.copy()
    elif FORECAST_LOG_PATH.exists():
        try:
            frame = pd.read_csv(FORECAST_LOG_PATH)
        except Exception as exc:
            frame = pd.DataFrame(columns=FORECAST_COLUMNS)
            frame.attrs["forecast_log_read_error"] = str(exc)
    else:
        frame = pd.DataFrame(columns=FORECAST_COLUMNS)
    if not {"season", "gameweek", "player_id"}.issubset(frame.columns):
        frame = pd.DataFrame(columns=FORECAST_COLUMNS)
        frame.attrs["forecast_log_read_error"] = "CSV has no recognizable forecast schema"
    for col in FORECAST_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    if "model_version" in frame.columns:
        versions = frame["model_version"].astype("string").str.strip()
        frame["model_version"] = versions.fillna("legacy-unknown").replace("", "legacy-unknown").astype(str)
    return frame[FORECAST_COLUMNS]

def write_forecast_log(frame):
    """Write a portable CSV. Streamlit-host disk can be ephemeral, so export backups are provided."""
    if frame.attrs.get("forecast_log_read_error"):
        return False
    try:
        FORECAST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp_path = FORECAST_LOG_PATH.with_suffix(".tmp")
        frame[FORECAST_COLUMNS].to_csv(temp_path, index=False)
        os.replace(temp_path, FORECAST_LOG_PATH)
        return True
    except Exception:
        return False

def append_first_forecast_snapshot(rows):
    """Take only the first pre-deadline snapshot for each season/Gameweek."""
    old = read_forecast_log()
    if old.attrs.get("forecast_log_read_error"):
        return old, False, "New forecast snapshot skipped because the existing local forecast log could not be read; its contents were preserved."
    row_version = str(rows[0].get("model_version", APP_VERSION))
    if not old.empty and ((old["season"].astype(str) == str(rows[0]["season"])) &
                          (pd.to_numeric(old["gameweek"], errors="coerce") == int(rows[0]["gameweek"])) &
                          (old["model_version"].astype(str) == row_version)).any():
        return old, False, "A forecast snapshot for this Gameweek and model version already exists; it was not overwritten."
    new = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    combined = pd.concat([old, new], ignore_index=True) if not old.empty else new
    ok = write_forecast_log(combined)
    return combined, ok, ("Pre-deadline forecast saved." if ok else "Could not write the forecast CSV on this host; use the export/import controls for a backup workflow.")

def fetch_history_signals(target_ids, cutoff_utc=None):
    results = {}
    unique_ids = sorted({int(x) for x in target_ids if x is not None})
    if not unique_ids:
        return results
    # Parallel bounded requests avoid a slow serial fetch; errors fall back to the base model.
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_id = {pool.submit(get_player_summary, pid): pid for pid in unique_ids}
        for future in as_completed(future_to_id):
            pid = future_to_id[future]
            try:
                results[pid] = make_history_signal(future.result(), cutoff_utc=cutoff_utc)
            except Exception:
                continue
    return results

@st.cache_data(ttl=300, show_spinner=False)
def get_entry_transfers(team_id):
    """Public transfer history when the endpoint is accessible; may lag or change."""
    try:
        r = requests.get(f"{BASE}/entry/{int(team_id)}/transfers/", timeout=12,
                         headers={"User-Agent":"Mozilla/5.0 The-One-Above-All-FPL/2.9"})
        r.raise_for_status()
        payload = r.json()
        return payload if isinstance(payload, list) else None
    except Exception:
        return None


def infer_free_transfers(history_payload, bootstrap_events, upcoming_gw, transfer_rows=None):
    """Reconstruct available FTs from public history, transfer costs and chip history.

    Returns (estimated_balance, confidence, note). Public FPL does not expose a
    reliably documented live FT-balance field, so this remains an estimate.
    """
    hist = history_payload.get("current", []) if isinstance(history_payload, dict) else []
    hist = [r for r in hist if isinstance(r, dict) and r.get("event") is not None
            and int(r.get("event") or 0) < int(upcoming_gw)]
    by_event = {int(r["event"]): r for r in hist}
    chip_events = {}
    for chip in (history_payload.get("chips", []) if isinstance(history_payload, dict) else []):
        try:
            ev = int(chip.get("event"))
            name = str(chip.get("name", "")).lower().replace("_", "").replace("-", "")
            chip_events.setdefault(ev, set()).add(name)
        except (TypeError, ValueError):
            continue
    event_map = {int(e["id"]): e for e in (bootstrap_events or []) if e.get("id") is not None}
    completed = [e for e in range(1, int(upcoming_gw))
                 if e in by_event and bool(event_map.get(e, {}).get("finished"))]
    if not completed:
        return 1, False, "No completed transfer history found; defaulting to one free transfer until verified."
    ft_balance = 1
    confidence = True
    # GW1/pre-season transfers don't consume the first post-GW1 free transfer.
    last_seen = 1
    for ev in range(2, max(completed) + 1):
        row = by_event.get(ev)
        if row is None:
            confidence = False
            # Preserve a conservative estimate when a historical row is missing.
            ft_balance = min(5, ft_balance)
            continue
        chips = chip_events.get(ev, set())
        is_wc_or_fh = any(x in {"wildcard", "freehit", "free_hit"} or "wildcard" in x or "freehit" in x for x in chips)
        if is_wc_or_fh:
            # Chip consumes the current week's FT while preserving banked transfers;
            # the next week's allowance replaces the used current-week FT.
            ft_balance = min(5, ft_balance)
        else:
            try:
                transfers = max(0, int(row.get("event_transfers") or 0))
            except (TypeError, ValueError):
                transfers = 0; confidence = False
            try:
                hit_count = max(0, int(round(float(row.get("event_transfers_cost") or 0) / 4.0)))
            except (TypeError, ValueError):
                hit_count = 0; confidence = False
            free_used = max(0, transfers - hit_count)
            remaining = max(0, ft_balance - min(ft_balance, free_used))
            ft_balance = min(5, remaining + 1)
        last_seen = ev
    # Account for transfers already made in the upcoming GW if the public endpoint exposes them.
    current_transfer_rows = [r for r in (transfer_rows or []) if isinstance(r, dict) and str(r.get("event", "")) == str(upcoming_gw)]
    current_chips = chip_events.get(int(upcoming_gw), set())
    if current_transfer_rows and not any(x in {"wildcard", "freehit", "free_hit"} or "wildcard" in x or "freehit" in x for x in current_chips):
        ft_balance = max(0, ft_balance - len(current_transfer_rows))
        confidence = False  # until current chip/transfer cost fields are public this is approximate
        note = f"Estimated from completed history and {len(current_transfer_rows)} transfer(s) detected for GW{upcoming_gw}; verify if a chip is active."
    else:
        note = "Reconstructed from completed transfer history, hits and chips; compare with the FPL Transfers page if you made moves this Gameweek." if confidence else "Estimated from incomplete public history; manual override is available."
    return int(max(0, min(5, ft_balance))), confidence, note


def norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def resolve_player(row, players):
    # Prefer exact full-name matches before short aliases; aliases like Palmer
    # can be ambiguous when several players share a common display name.
    full_name = norm(row[0])
    alias = norm(row[2])
    full_matches = [
        p for p in players
        if norm(p.get("first_name", "") + " " + p.get("second_name", "")) == full_name
    ]
    if len(full_matches) == 1:
        return full_matches[0]
    alias_matches = [p for p in players if norm(p.get("web_name", "")) == alias]
    if len(alias_matches) == 1:
        return alias_matches[0]
    partial = [p for p in players if alias and alias in norm(p.get("web_name", ""))]
    return partial[0] if len(partial) == 1 else None

def pos_name(p, types):
    return next((x.get("singular_name_short", "?") for x in types if x["id"] == p.get("element_type")), "?")

def estimate_points(
    p, team, fixtures, horizon=5, start_gw=None, use_history=True,
    history_signals=None, calibration_offsets=None,
):
    """Delegate to the shared engine, allowing callers to override optional inputs.

    ``None`` means use the dashboard's current shared values; an explicit empty
    dict means deliberately disable history/calibration for baseline snapshots.
    """
    if history_signals is None:
        history_signals = PLAYER_HISTORY_SIGNALS
    if calibration_offsets is None:
        calibration_offsets = MODEL_CALIBRATION_OFFSETS
    return fpl_engine.estimate_points(
        p, team, fixtures, horizon=horizon, start_gw=start_gw,
        use_history=use_history, history_signals=history_signals,
        calibration_offsets=calibration_offsets,
    )

st.markdown("""
<style>
:root {--navy:#142b3f;--navy-deep:#0e2031;--green:#7fbd43;--green-dark:#3c8d57;--surface:#fff;--surface-soft:#f3f6f8;--ink:#1f2d38;--muted:#61717e;--line:#dfe6eb;}
.block-container {padding-top:.85rem;padding-bottom:2rem;max-width:1540px;}
[data-testid="stSidebar"] {background:#f1f4f6;border-right:1px solid var(--line);}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color:var(--ink) !important;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
 color:#202238 !important;-webkit-text-fill-color:#202238 !important;background:#fff !important;opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button,
[data-testid="stSidebar"] [data-baseweb="select"] svg {color:#202238 !important;fill:#202238 !important;}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {color:var(--ink) !important;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] p {color:var(--ink) !important;}
.decision-header {display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;background:var(--navy);color:#fff;padding:.72rem .9rem;border-radius:12px;margin:1.25rem 0 .55rem;box-shadow:0 3px 10px rgba(20,43,63,.1);}
.decision-brand {font-weight:850;font-size:1.08rem;letter-spacing:-.025em;white-space:nowrap;}
.decision-deadlines {display:flex;align-items:center;justify-content:flex-end;gap:.45rem;flex-wrap:wrap;}
.deadline-chip {background:rgba(255,255,255,.12);border:1px solid rgba(255,255,255,.18);border-radius:9px;padding:.35rem .55rem;line-height:1.2;}
.deadline-chip span {display:block;color:#e8dced;font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
.deadline-chip strong {display:block;color:#fff;font-size:.82rem;font-variant-numeric:tabular-nums;}
.deadline-chip.countdown {background:#dff0ce;border-color:#c4e0a4;}
.deadline-chip.countdown span,.deadline-chip.countdown strong {color:#254b30;}
.status-strip {display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;margin:0 0 .25rem;}
.status-chip {display:inline-flex;align-items:center;gap:.28rem;border:1px solid var(--line);background:#f7f9fa;color:#40515d;padding:.3rem .55rem;border-radius:999px;font-size:.75rem;line-height:1.2;}
.status-chip strong {color:var(--navy);font-weight:800;}
.verdict-card {border:1px solid var(--line);border-left:5px solid var(--green-dark);background:var(--surface);border-radius:12px;padding:.9rem 1rem;margin:.2rem 0 .7rem;box-shadow:0 3px 12px rgba(20,43,63,.05);}
.verdict-topline {display:flex;align-items:center;gap:.55rem;flex-wrap:wrap;margin-bottom:.35rem;}
.verdict-label {font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.09em;font-weight:800;}
.verdict-value {font-size:1.35rem;line-height:1.15;color:var(--navy);font-weight:900;letter-spacing:-.035em;}
.provisional-badge {display:inline-block;background:#fff1d6;color:#704400;border:1px solid #ebcb8d;padding:.2rem .5rem;border-radius:999px;font-size:.67rem;font-weight:850;text-transform:uppercase;letter-spacing:.04em;}
.decision-card-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:.6rem;margin:.2rem 0 .8rem;}
.decision-summary-card {background:#fff;border:1px solid var(--line);border-radius:11px;padding:.7rem .78rem;min-width:0;box-shadow:0 2px 8px rgba(20,43,63,.035);}
.decision-summary-card .card-label {color:var(--muted);text-transform:uppercase;letter-spacing:.07em;font-size:.65rem;font-weight:850;margin-bottom:.25rem;}
.decision-summary-card .card-value {color:var(--navy);font-size:.98rem;line-height:1.25;font-weight:850;overflow-wrap:anywhere;}
.decision-summary-card .card-detail {color:#5f6e78;font-size:.74rem;line-height:1.35;margin-top:.25rem;}
.checklist-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.1rem .8rem;margin:-.1rem 0 .5rem;}
[data-testid="stSidebar"] [data-testid="stMetric"] {background:#fff;border:1px solid var(--line);border-radius:10px;}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {color:#53636e !important;}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {color:var(--navy) !important;}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p {color:#53636e !important;}
[data-testid="stSidebar"] [data-testid="stMetricValue"] [data-testid="stMarkdownContainer"] p {color:var(--navy) !important;}
div[data-testid="stMetric"] {padding:.72rem .88rem;border:1px solid var(--line);border-radius:11px;background:#fff;box-shadow:0 2px 8px rgba(20,43,63,.04);}
div[data-testid="stMetricLabel"] {color:#61717e;font-weight:650;}
div[data-testid="stMetricValue"] {color:var(--navy);font-weight:850;}
div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:10px;overflow:hidden;}
div[data-testid="stTabs"] button[role="tab"] {color:#53636e;font-weight:700;}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {color:var(--navy);border-bottom-color:var(--green-dark);}
div[data-testid="stExpander"] {border:1px solid var(--line);border-radius:9px;}
h2,h3 {letter-spacing:-.025em;}
.pitch-wrap {background:#0e7b45;border-radius:20px;padding:16px 12px 18px;position:relative;overflow:hidden;border:4px solid #fff;box-shadow:0 6px 20px rgba(12,90,52,.12);}
.pitch-wrap:before {content:"";position:absolute;inset:0;background:repeating-linear-gradient(0deg,rgba(255,255,255,.035) 0,rgba(255,255,255,.035) 48px,rgba(0,0,0,.025) 48px,rgba(0,0,0,.025) 96px);pointer-events:none;}
.pitch-lines {position:absolute;inset:10px;border:1.5px solid rgba(255,255,255,.28);border-radius:10px;pointer-events:none;}
.pitch-lines:before {content:"";position:absolute;left:0;right:0;top:50%;border-top:1.5px solid rgba(255,255,255,.27);}
.pitch-lines:after {content:"";position:absolute;width:90px;height:90px;left:50%;top:50%;transform:translate(-50%,-50%);border:1.5px solid rgba(255,255,255,.25);border-radius:50%;}
.pitch-title {position:relative;z-index:1;text-align:center;color:#d9ffe9;font-weight:800;text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;margin:2px 0 13px;}
.pitch-row {position:relative;z-index:1;display:flex;justify-content:center;align-items:stretch;gap:clamp(5px,1.5vw,18px);margin:15px auto;max-width:920px;}
.player-tile {width:clamp(74px,10vw,128px);min-width:0;border-radius:12px;overflow:hidden;text-align:center;background:#f8fafc;box-shadow:0 5px 10px rgba(0,0,0,.18);border:1px solid rgba(255,255,255,.8);}
.player-tile .tile-top {background:var(--navy);color:#fff;padding:7px 3px 5px;font-weight:800;font-size:clamp(.68rem,1vw,.86rem);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.player-tile .tile-avatar {height:42px;display:flex;align-items:center;justify-content:center;background:var(--surface-soft);color:var(--navy);font-size:1.35rem;font-weight:900;}
.player-tile .tile-meta {padding:5px 3px;color:#334653;font-size:clamp(.62rem,.8vw,.75rem);line-height:1.3;}
.player-tile .tile-pts {font-weight:900;color:var(--navy);font-size:.95rem;}
.pitch-caption {position:relative;z-index:1;text-align:center;color:#e4ffef;font-size:.78rem;margin-top:10px;}
.bench-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;}
.bench-card {background:var(--surface-soft);border:1px solid var(--line);border-radius:10px;padding:10px;text-align:center;}
.bench-name {font-weight:800;color:var(--navy);font-size:.84rem;}
.bench-sub {color:#5f6e78;font-size:.76rem;margin-top:3px;}
.section-intro {color:#777083;font-size:.92rem;margin-top:-.4rem;margin-bottom:.7rem;}
@media (max-width:650px) {
 .block-container {padding-left:.65rem;padding-right:.65rem;}
 .decision-header {align-items:flex-start;padding:.7rem .75rem;}
 .decision-brand {font-size:.98rem;}
 .decision-deadlines {justify-content:flex-start;width:100%;}
 .deadline-chip {padding:.3rem .42rem;}
 .deadline-chip strong {font-size:.72rem;}
 .status-chip {font-size:.68rem;padding:.28rem .45rem;}
 .verdict-card {padding:.82rem .88rem;}
 .verdict-value {font-size:1.22rem;}
 .decision-card-grid {grid-template-columns:1fr;gap:.45rem;}
 .decision-summary-card {padding:.62rem .7rem;}
 .pitch-wrap {padding:10px 5px;}
 .pitch-row {gap:4px;margin:10px auto;}
 .player-tile {width:22%;border-radius:8px;}
 .player-tile .tile-avatar {height:31px;font-size:1.05rem;}
 .player-tile .tile-meta {font-size:.59rem;padding:4px 1px;}
 .player-tile .tile-pts {font-size:.78rem;}
 .player-tile .tile-top {font-size:.59rem;padding:6px 1px 4px;}
 .bench-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
}

/* FPL-inspired manager profile + live Gameweek layout */
.manager-panel {background:#fff;border:1px solid var(--line);border-radius:12px;padding:14px;box-shadow:0 2px 9px rgba(20,43,63,.045);margin-bottom:10px;}
.manager-name {font-size:1.08rem;font-weight:850;color:var(--navy);line-height:1.25;margin:0 0 4px 0;}
.manager-label {font-size:.78rem;color:#5f6e78;margin:0 0 13px 0;}
.panel-heading {font-size:1rem;font-weight:800;color:var(--navy);border-bottom:1px solid var(--line);padding-bottom:9px;margin-bottom:10px;}
.rank-row {display:flex;justify-content:space-between;gap:10px;align-items:center;padding:6px 0;border-bottom:1px solid #f4f0f5;color:#4b4050;font-size:.86rem;}
.rank-row:last-child {border-bottom:0;}
.rank-row strong {color:var(--navy);font-variant-numeric:tabular-nums;}
.league-pill {display:flex;justify-content:space-between;gap:8px;background:var(--surface-soft);border-radius:8px;padding:8px 10px;margin:6px 0;color:var(--navy);font-size:.82rem;font-weight:650;}
.gw-panel {border:1px solid var(--line);background:#fff;border-radius:12px;padding:14px;box-shadow:0 2px 9px rgba(20,43,63,.04);}
.gw-heading {display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;}
.gw-title {font-size:1.25rem;font-weight:850;color:var(--navy);margin:0;}
.gw-sub {font-size:.78rem;color:#74687b;margin-top:2px;}
.gw-scoreband {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin:12px 0 16px;}
.gw-scorebox {background:#f8f4fa;border-radius:11px;padding:10px 7px;text-align:center;min-width:0;}
.gw-scorebox.featured {background:var(--navy);color:#fff;}
.gw-scorebox .num {font-size:clamp(1.1rem,2vw,1.8rem);font-weight:900;color:var(--navy);line-height:1.15;}
.gw-scorebox.featured .num {color:#fff;font-size:clamp(1.55rem,2.6vw,2.25rem);}
.gw-scorebox .lbl {font-size:.7rem;color:#706179;margin-top:4px;line-height:1.25;}
.gw-scorebox.featured .lbl {color:#fff;}
.player-tile {width:clamp(70px,8vw,112px);min-width:0;border-radius:8px;overflow:hidden;text-align:center;background:#fff;box-shadow:0 3px 8px rgba(0,0,0,.15);border:1px solid rgba(255,255,255,.85);}
.player-tile .tile-avatar {height:60px;display:flex;align-items:center;justify-content:center;background:linear-gradient(180deg,#f3f6f8,#fff);color:var(--navy);}
.kit-icon {width:54px;height:54px;filter:drop-shadow(0 2px 1px rgba(0,0,0,.12));}
.player-tile .tile-top {font-size:clamp(.63rem,.85vw,.8rem);padding:6px 2px 5px;}
.player-tile .tile-meta {padding:4px 2px 6px;font-size:clamp(.59rem,.75vw,.7rem);line-height:1.3;}
.player-tile .tile-pts {font-size:.88rem;}
.pitch-wrap {padding:12px 8px 13px;}
.pitch-title {margin:5px 0 10px;}
.pitch-row {gap:clamp(3px,1vw,12px);margin:12px auto;}
.manager-overview-note {background:#f3f6f8;border-left:3px solid var(--green-dark);padding:9px 10px;border-radius:8px;color:#465762;font-size:.78rem;line-height:1.45;margin-top:10px;}
@media(max-width:900px) {
 .gw-scoreband {grid-template-columns:repeat(2,minmax(0,1fr));}
 .player-tile {width:23%;max-width:98px;}
}
@media(max-width:650px) {
 .manager-panel {padding:12px;}
 .gw-panel {padding:10px;}
 .gw-title {font-size:1.15rem;}
 .gw-scorebox {padding:8px 4px;}
  .pitch-row {width:100%;gap:3px;margin:8px auto;}
  .player-tile {width:auto;max-width:none;flex:1 1 0;}
 .player-tile .tile-avatar {height:42px;}
 .kit-icon {width:40px;height:40px;}
 .pitch-title {font-size:.61rem;letter-spacing:.08em;margin:4px 0 8px;}
  .player-tile .tile-top {font-size:.56rem;padding:5px 1px 4px;}
  .player-tile .tile-meta {font-size:.55rem;padding:4px 1px;}
  .player-tile .tile-pts {font-size:.7rem;}
}
[data-testid="stSidebar"] [data-testid="stButton"] button {
 background:#deedcf !important;color:#1c3825 !important;-webkit-text-fill-color:#1c3825 !important;
 border:1px solid #7fbd43 !important;font-weight:750 !important;opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:disabled {
 background:#deedcf !important;color:#1c3825 !important;-webkit-text-fill-color:#1c3825 !important;opacity:1 !important;
}

.chip-score {font-size:1.25rem;font-weight:900;color:var(--navy);}
div[data-testid="stHorizontalBlock"] .manager-panel {height:100%;}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Optimizer settings")
    st.caption("These controls change the calculations below.")
    team_id = st.number_input("FPL Team ID", min_value=1, value=2311669, step=1)
    horizon = st.slider("Projection horizon (Gameweeks)", 1, 8, 5,
                         help="How many upcoming Gameweeks to include in player projections and transfer gains.")
    fixture_horizon = st.slider("Fixture calendar horizon (Gameweeks)", 5, 20, 15,
                                 help="How far ahead to display fixtures and identify officially assigned double Gameweeks. Player-point projections keep their own shorter horizon.")
    # Sync the last known bank, transfer history and chip history directly from public FPL endpoints.
    control_profile = {}
    control_history = {}
    control_boot = {}
    control_transfers = None
    control_transfer_history_available = False
    try:
        control_profile = get_manager_profile(int(team_id))
    except Exception:
        pass
    try:
        control_history = get_entry_history(int(team_id))
    except Exception:
        pass
    try:
        control_boot = get_json("bootstrap-static/")
    except Exception:
        pass
    control_events = control_boot.get("events", []) if isinstance(control_boot, dict) else []
    control_next = next((e for e in control_events if e.get("is_next")), None)
    control_current = next((e for e in control_events if e.get("is_current")), None)
    control_gw = int(control_next.get("id")) if control_next else int((control_current or {}).get("id", 7)) + 1
    try:
        control_transfers = get_entry_transfers(int(team_id))
        control_transfer_history_available = isinstance(control_transfers, list)
    except Exception:
        control_transfers = None
    auto_ft, auto_ft_confident, auto_ft_note = infer_free_transfers(
        control_history, control_events, control_gw,
        control_transfers if isinstance(control_transfers, list) else [],
    )
    auto_bank_raw = control_profile.get("last_deadline_bank") if isinstance(control_profile, dict) else None
    if auto_bank_raw is None:
        control_hist_rows = control_history.get("current", []) if isinstance(control_history, dict) else []
        control_hist_rows = sorted([r for r in control_hist_rows if isinstance(r, dict) and r.get("event") is not None], key=lambda r: int(r.get("event", 0)))
        auto_bank_raw = control_hist_rows[-1].get("bank") if control_hist_rows else None
    auto_bank_tenths = exact_fpl_tenths(auto_bank_raw)
    auto_bank = auto_bank_tenths / 10.0 if auto_bank_tenths is not None else None
    # Adjust the last-deadline bank by publicly visible transactions in the upcoming GW where prices are included.
    latest_bank_known = auto_bank_tenths is not None
    current_transactions = [r for r in (control_transfers if isinstance(control_transfers, list) else [])
                            if isinstance(r, dict) and str(r.get("event", "")) == str(control_gw)]
    current_tx_adjustment_tenths = 0
    for tx in current_transactions:
        try:
            out_cost = verified_player_price_tenths(tx.get("element_out_cost"))
            in_cost = verified_player_price_tenths(tx.get("element_in_cost"))
            if out_cost is None or in_cost is None:
                latest_bank_known = False
                continue
            current_tx_adjustment_tenths += out_cost - in_cost
        except (TypeError, ValueError):
            latest_bank_known = False
    if current_transactions and latest_bank_known:
        auto_bank_tenths = max(0, min(200, auto_bank_tenths + current_tx_adjustment_tenths))
        auto_bank = auto_bank_tenths / 10.0
    sync_mode = st.selectbox("Budget & transfer source", ["Auto-sync from FPL history", "Manual override"], index=0,
                             help="Bank comes from the latest public deadline record plus visible current-Gameweek transfers. Free transfers are reconstructed from historical transfers, hit costs and chips; override if the FPL page differs.")
    auto_controls = sync_mode == "Auto-sync from FPL history"
    _bank_state, _ft_state = status_chip_labels(
        auto_controls, latest_bank_known, auto_ft_confident
    )
    if auto_controls:
        st.metric("Bank available", f"£{auto_bank:.1f}m" if latest_bank_known else "Unavailable")
        st.metric("Estimated free transfers", str(auto_ft))
        bank_m = auto_bank if latest_bank_known else None
        free_transfers = auto_ft
        st.caption(f"Auto-sync · GW{control_gw} · Bank {_bank_state} · Free transfers {_ft_state}.")
        st.caption(auto_ft_note)
    else:
        bank_m = st.number_input("Bank (£m)", min_value=0.0, max_value=20.0, value=round(auto_bank, 1) if latest_bank_known else 0.0, step=0.1, key="manual_bank_override",
                                 help="Enter the current bank from your FPL Transfers page. This is the override for uncertain or unpublished current transfers.")
        free_transfers = st.selectbox("Free transfers available", [0,1,2,3,4,5], index=max(0, min(5, int(auto_ft))), key="manual_ft_override",
                                      help="Enter the exact number shown in the official FPL app.")
    bank_tenths_current = (auto_bank_tenths if auto_controls and latest_bank_known
                           else pounds_millions_to_tenths(bank_m))
    transfer_hit = st.checkbox("Allow a -4 hit if model sees a gain", value=False,
                                help="If enabled, subtracts four points from the projected gain before ranking. A hit should only be considered for a clear multi-week advantage.")
    risk = st.select_slider("Risk appetite", options=["Conservative","Moderate","Aggressive"], value="Moderate",
                             help="Conservative favors established minutes; Moderate balances projected gain and differential upside; Aggressive gives more weight to low-ownership upside.")
    if st.button("↻ Refresh rank & squad", use_container_width=True):
        get_trained_model_payload.clear()
        fpl_engine.TRAINED_MODEL = get_trained_model_payload()
        get_manager_profile.clear()
        get_entry_picks.clear()
        get_entry_history.clear()
        get_entry_transfers.clear()
        get_remote_forecast_log.clear()
        get_player_summary.clear()
        get_live_gameweek.clear()
        get_json.clear()
        st.rerun()
    st.caption("Rank refreshes from the public manager profile. Squad uses public Gameweek picks when available.")

try:
    boot = get_json("bootstrap-static/")
    fixtures = get_json("fixtures/")
    players = boot.get("elements", [])
    teams = {t["id"]: t for t in boot.get("teams", [])}
    types = boot.get("element_types", [])
    events = boot.get("events", [])
except Exception as e:
    st.error(f"Could not load public FPL data: {e}")
    st.info("Check internet access, then refresh. This app uses public read-only endpoints and requires no FPL password.")
    st.stop()

next_event = next((e for e in events if e.get("is_next")), None)
current_event = next((e for e in events if e.get("is_current")), None)
gw = next_event.get("id") if next_event else ((current_event.get("id",7)+1) if current_event else 8)
entry = {}
try:
    entry = get_manager_profile(int(team_id))
except Exception:
    pass

# Public endpoints do not require the user's FPL password. Never substitute a
# hand-entered example squad when current FPL picks cannot be verified.
elements_by_id = {int(p["id"]): p for p in players}
squad_load = load_public_squad(
    int(gw), events, lambda event_id: get_entry_picks(int(team_id), int(event_id)),
    control_transfers, control_transfer_history_available, elements_by_id,
    chip_rows=(control_history.get("chips") if isinstance(control_history, dict) else None),
)
squad_rows = []
public_picks_for_display = list(squad_load.get("picks", []))
picks_sync_ok = bool(squad_load.get("target_verified") or
                      (squad_load.get("available") and not squad_load.get("needs_confirmation")))
picks_sync_note = str(squad_load.get("reason", "Live squad unavailable. No example or saved squad is substituted."))
squad_source = str(squad_load.get("source", "unavailable"))
squad_recommendations_allowed = picks_sync_ok
manual_purchase_price_overrides = {}
if squad_load.get("available"):
    roster_ids = list(squad_load["player_ids"])
    if squad_load.get("needs_confirmation"):
        with st.sidebar:
            st.warning(picks_sync_note)
            st.caption("Prior published picks cannot prove that no current transfers, Wildcard or Free Hit changes are pending.")
            if auto_controls:
                st.caption("The auto-synced bank and free-transfer count may still reflect the prior deadline. Switch to Manual override and confirm both before using transfer-budget advice.")
            roster_options = sorted(elements_by_id)
            _player_label_by_id = {
                pid: f"{elements_by_id[pid].get('web_name', 'Player')} · {pos_name(elements_by_id[pid], types)} · {teams.get(elements_by_id[pid].get('team'), {}).get('short_name', '?')} · #{pid}"
                for pid in roster_options
            }
            _corrected_roster_ids = st.multiselect(
                "Confirm or correct current 15-player roster",
                options=roster_options, default=roster_ids,
                format_func=lambda player_id: _player_label_by_id[int(player_id)],
                key=f"confirmed_roster_{int(team_id)}_{int(gw)}",
                help="Use this roster editor to apply any transfer not reflected in public history. It must contain exactly 2 goalkeepers, 5 defenders, 5 midfielders and 3 forwards.",
            )
            _manual_roster_valid, _manual_roster_reason = validate_squad_ids(_corrected_roster_ids, elements_by_id)
            if not _manual_roster_valid:
                st.error(_manual_roster_reason)
            _roster_confirmed = st.checkbox(
                "I confirm this roster includes all transfers and chip changes",
                key=f"confirm_roster_{int(team_id)}_{int(gw)}",
            )
            _known_purchase_prices = {
                int(row["element"]): int(row["purchase_price"])
                for row in squad_load.get("picks", [])
                if row.get("purchase_price") is not None
            }
            _known_purchase_prices.update({
                int(row["in"]): int(row["purchase_price"])
                for row in squad_load.get("transactions", [])
                if row.get("purchase_price") is not None
            })
            _missing_purchase_price_ids = [
                int(pid) for pid in _corrected_roster_ids if int(pid) not in _known_purchase_prices
            ]
            _purchase_prices_confirmed = not _missing_purchase_price_ids
            if _missing_purchase_price_ids:
                st.caption("Transfer advice also needs the official purchase price for corrected players; current market price is not the same as selling price.")
                for _pid in _missing_purchase_price_ids:
                    _current_tenths = int(elements_by_id[_pid].get("now_cost") or 0)
                    manual_purchase_price_overrides[_pid] = round(st.number_input(
                        f"{elements_by_id[_pid].get('web_name', 'Player')} purchase price (£m)",
                        min_value=4.0, max_value=20.0, value=round(_current_tenths / 10, 1),
                        step=0.1, key=f"manual_purchase_price_{int(team_id)}_{int(gw)}_{_pid}",
                        help="Enter the price shown for this player in your official FPL squad/transfer view.",
                    ) * 10)
                _purchase_prices_confirmed = st.checkbox(
                    "I verified these purchase prices in FPL",
                    key=f"confirm_purchase_prices_{int(team_id)}_{int(gw)}",
                )
                if _purchase_prices_confirmed:
                    for _pid in _missing_purchase_price_ids:
                        _known_purchase_prices[_pid] = manual_purchase_price_overrides[_pid]
            if _roster_confirmed and _manual_roster_valid:
                roster_ids = [int(pid) for pid in _corrected_roster_ids]
                squad_recommendations_allowed = True
                picks_sync_note += " Manager-confirmed roster; recommendations are provisional until current GW picks are published."
            else:
                squad_recommendations_allowed = False
                picks_sync_note += " Recommendations are withheld until you confirm or correct the roster."

    for player_id in roster_ids:
        player = elements_by_id[player_id]
        full_name = (player.get("first_name", "") + " " + player.get("second_name", "")).strip()
        squad_rows.append((full_name, pos_name(player, types), player.get("web_name", full_name), player_id))
    if squad_load.get("needs_confirmation") and roster_ids != squad_load.get("player_ids"):
        _prior_pick_by_id = {int(row["element"]): dict(row) for row in public_picks_for_display}
        public_picks_for_display = [
            {**_prior_pick_by_id.get(pid, {}), "element": pid, "position": position}
            for position, pid in enumerate(roster_ids, 1)
        ]

# Pull player-level history for the current squad and a bounded pool of likely transfer targets.
# The full pool still receives a transparent baseline projection; history enrichment is intentionally cached.
def _history_priority(p):
    def _n(k):
        try: return float(p.get(k) or 0)
        except (TypeError, ValueError): return 0.0
    mins = _n("minutes")
    xgi90 = (_n("expected_goals") + _n("expected_assists")) / max(90.0, mins) * 90.0
    return 0.45 * _n("points_per_game") + 0.20 * _n("form") + 0.50 * min(1.0, mins / 900.0) + 0.20 * min(1.5, xgi90)

squad_target_ids = []
for _row in squad_rows:
    _match = resolve_player(_row, players)
    if _match:
        squad_target_ids.append(int(_match["id"]))
# Enrich the full player pool so transfer candidates, captain options and the
# saved-vs-baseline validation do not quietly use different feature coverage.
history_target_ids = [int(p["id"]) for p in players if p.get("id") is not None]
_history_event = next((e for e in events if int(e.get("id") or 0) == int(gw)), {})
try:
    _history_deadline = datetime.fromisoformat(str(_history_event.get("deadline_time")).replace("Z", "+00:00"))
    _history_cutoff = _history_deadline - timedelta(hours=2)
except (TypeError, ValueError):
    # Without a parseable deadline, use only history that was safely available
    # at the current time; this suppresses in-progress/recently finished match data.
    _history_cutoff = datetime.now(timezone.utc)
try:
    with st.spinner("Updating recent player history for the prediction engine…"):
        PLAYER_HISTORY_SIGNALS = fetch_history_signals(history_target_ids, cutoff_utc=_history_cutoff)
except Exception:
    PLAYER_HISTORY_SIGNALS = {}

# First pre-deadline visit records one frozen prediction per player/Gameweek for honest future testing.
remote_forecast_df = get_remote_forecast_log()
remote_log_active = isinstance(remote_forecast_df, pd.DataFrame)
forecast_log, forecast_snapshot_status, forecast_snapshot_note = read_forecast_log(), False, ""
# Keep forecast weights frozen during validation. A position-only offset is not allowed
# to alter rankings until our benchmark results demonstrate a repeatable benefit.
MODEL_CALIBRATION_OFFSETS = {}
CALIBRATION_METADATA = {"enabled": False, "completed_gws": 0, "samples_by_position": {}, "offsets": {}}
season_id = season_label()
_current_snapshot = pd.DataFrame()
if not forecast_log.empty:
    _snapshot_mask = ((forecast_log["season"].astype(str) == season_id)
                      & (pd.to_numeric(forecast_log["gameweek"], errors="coerce") == int(gw))
                      & (forecast_log["model_version"].astype(str) == APP_VERSION))
    _current_snapshot = forecast_log.loc[_snapshot_mask].copy()
SNAPSHOT_GW1 = {int(r["player_id"]): float(r["projection_model"]) for _, r in _current_snapshot.iterrows()
                if pd.notna(r.get("player_id")) and pd.notna(r.get("projection_model"))}
forecast_event = next((e for e in events if int(e.get("id") or 0) == int(gw)), {})
deadline_text = forecast_event.get("deadline_time")
try:
    deadline_dt = datetime.fromisoformat(str(deadline_text).replace("Z", "+00:00")) if deadline_text else None
except ValueError:
    deadline_dt = None
if deadline_dt and datetime.now(timezone.utc) < deadline_dt:
    has_snapshot = (not forecast_log.empty and
                    ((forecast_log["season"].astype(str) == season_id) &
                     (pd.to_numeric(forecast_log["gameweek"], errors="coerce") == int(gw)) &
                     (forecast_log["model_version"].astype(str) == APP_VERSION)).any())
    if has_snapshot:
        forecast_snapshot_note = "A pre-deadline forecast already exists for this Gameweek; it is preserved."
    elif not fpl_learning.validate_model(fpl_engine.TRAINED_MODEL):
        forecast_snapshot_note = "New forecast snapshot skipped: the trained v3.0 model artifact is missing, invalid, or incompatible. Existing forecast records remain available for reconciliation."
    elif remote_log_active:
        forecast_snapshot_note = "Automatic logging is enabled. The worker captures only in the final 6 hours before the deadline, with a compatible trained model and assigned fixtures; until then no current forecast record is saved."
    else:
        created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        forecast_rows = []
        for _p in players:
            _team = teams.get(_p.get("team"), {}).get("short_name", "?")
            _model, _, _ = estimate_points(_p, _team, fixtures, 1, start_gw=gw, use_history=True, calibration_offsets={})
            _base, _, _ = estimate_points(_p, _team, fixtures, 1, start_gw=gw, use_history=False, calibration_offsets={})
            _model_5, _, _ = estimate_points(_p, _team, fixtures, min(5, max(1, len(events) - int(gw) + 1)), start_gw=gw, use_history=True, calibration_offsets={})
            _base_5, _, _ = estimate_points(_p, _team, fixtures, min(5, max(1, len(events) - int(gw) + 1)), start_gw=gw, use_history=False, calibration_offsets={})
            _sig = PLAYER_HISTORY_SIGNALS.get(int(_p.get("id", -1)), {})
            try:
                _epnext = float(_p.get("ep_next")) if _p.get("ep_next") not in (None, "") else None
            except (TypeError, ValueError):
                _epnext = None
            forecast_rows.append({
                "season": season_id, "gameweek": int(gw), "player_id": int(_p["id"]),
                "player": _p.get("web_name", ""), "position": pos_name(_p, types),
                "club": _team, "projection_model": _model, "projection_baseline": _base,
                "projection_ep_next": _epnext, "projection_model_5gw": _model_5,
                "projection_baseline_5gw": _base_5, "forecast_horizon_gws": min(5, max(1, len(events) - int(gw) + 1)),
                "history_enriched": int(_p.get("id", -1)) in PLAYER_HISTORY_SIGNALS,
                "history_sample_matches": int(_sig.get("sample_matches") or 0) if _sig else None,
                "projection_expected_minutes": fpl_engine.estimate_expected_minutes(
                    _p, _sig, fpl_engine.team_fixture_count(fixtures, _p.get("team"), int(gw))
                ),
                "xi_eligible": None, "model_captain_pick": None,
                "forecast_created_utc": created_utc, "actual_points": None, "actual_minutes": None,
                "actual_points_5gw": None, "actual_minutes_5gw": None, "model_version": APP_VERSION,
            })
        forecast_log, forecast_snapshot_status, forecast_snapshot_note = append_first_forecast_snapshot(forecast_rows)
else:
    forecast_snapshot_note = "No new forecast snapshot was written because this is not currently before the selected Gameweek deadline."

try:
    history_payload = get_entry_history(int(team_id))
    season_history = history_payload.get("current", []) if isinstance(history_payload, dict) else []
    season_history = sorted(
        [row for row in season_history if isinstance(row, dict) and row.get("event") is not None],
        key=lambda row: int(row.get("event", 0))
    )
    latest_history = season_history[-1] if season_history else {}
    previous_history = season_history[-2] if len(season_history) >= 2 else {}
except Exception:
    season_history = []
    latest_history = {}
    previous_history = {}

# Use public chip history when it is available, while keeping a manual status fallback.
chip_period_key = "first_half" if int(gw) < 20 else "second_half"
chip_period_label = "First-half chips (before GW19 deadline)" if int(gw) < 20 else "Second-half chips (GW20 onwards)"
_chip_history = history_payload.get("chips", []) if isinstance(history_payload, dict) else []
_chip_used_events = {"wildcard": [], "bench_boost": [], "free_hit": [], "triple_captain": []}
_chip_aliases = {
    "wildcard": "wildcard", "wc": "wildcard",
    "bboost": "bench_boost", "benchboost": "bench_boost", "bench_boost": "bench_boost",
    "freehit": "free_hit", "free_hit": "free_hit",
    "3xc": "triple_captain", "triplecaptain": "triple_captain", "triple_captain": "triple_captain",
}
for _chip in _chip_history if isinstance(_chip_history, list) else []:
    try:
        _chip_name = "".join(ch for ch in str(_chip.get("name", "")).lower() if ch.isalnum() or ch == "_")
        _chip_type = _chip_aliases.get(_chip_name)
        _chip_event = int(_chip.get("event"))
        in_period = (_chip_event < 20) if int(gw) < 20 else (_chip_event >= 20)
        if _chip_type and in_period:
            _chip_used_events[_chip_type].append(_chip_event)
    except (TypeError, ValueError, AttributeError):
        continue
wildcard_used_event = max(_chip_used_events["wildcard"], default=None)
bench_boost_used_event = max(_chip_used_events["bench_boost"], default=None)
free_hit_used_event = max(_chip_used_events["free_hit"], default=None)
triple_captain_used_event = max(_chip_used_events["triple_captain"], default=None)
with st.sidebar:
    st.divider()
    with st.expander(f"🎟️ Chip status · {chip_period_label}", expanded=False):
        st.caption("Chip status is prefilled from public manager history where available. Verify it in the FPL app if a chip is pending or history has not updated; manual editing remains available.")
        wildcard_options = ["Available", "Played in GW6", "Played in another Gameweek", "Unknown / verify"]
        if wildcard_used_event is not None:
            wildcard_default = 1 if wildcard_used_event == 6 else 2
        else:
            wildcard_default = 1 if int(gw) < 20 else 0
        wildcard_status = st.selectbox(
            "Wildcard", wildcard_options, index=wildcard_default,
            key=f"chip_wildcard_{chip_period_key}_{wildcard_used_event or 'none'}"
        )
        chip_options = ["Available", "Played in a previous Gameweek", "Unknown / verify"]
        bench_default = 1 if bench_boost_used_event is not None else 0
        freehit_default = 1 if free_hit_used_event is not None else 0
        triple_default = 1 if triple_captain_used_event is not None else 0
        bench_boost_status = st.selectbox(
            "Bench Boost", chip_options, index=bench_default,
            key=f"chip_bb_{chip_period_key}_{bench_boost_used_event or 'none'}"
        )
        free_hit_status = st.selectbox(
            "Free Hit", chip_options, index=freehit_default,
            key=f"chip_fh_{chip_period_key}_{free_hit_used_event or 'none'}"
        )
        triple_captain_status = st.selectbox(
            "Triple Captain", chip_options, index=triple_default,
            key=f"chip_tc_{chip_period_key}_{triple_captain_used_event or 'none'}"
        )

rank_change_places = None
rank_change_label = "Not enough rank-history rows yet"
# Do not gate rank history on bootstrap's event.finished flag: FPL can show
# points/ranks in the manager UI before that flag catches up. Use the latest
# two actual history rows with numeric overall ranks, ordered by Gameweek.
rank_history_rows = sorted(
    [
        row for row in season_history
        if row.get("event") is not None
        and isinstance(row.get("overall_rank"), (int, float))
        and int(row.get("overall_rank") or 0) > 0
    ],
    key=lambda row: int(row.get("event", 0))
)
if len(rank_history_rows) >= 2:
    previous_rank_row = rank_history_rows[-2]
    latest_rank_row = rank_history_rows[-1]
    prev_rank = int(previous_rank_row["overall_rank"])
    latest_rank = int(latest_rank_row["overall_rank"])
    rank_change_places = prev_rank - latest_rank
    prev_gw = int(previous_rank_row["event"])
    latest_gw = int(latest_rank_row["event"])
    if rank_change_places > 0:
        rank_change_label = f"↑ {rank_change_places:,} places gained (GW{prev_gw}→GW{latest_gw})"
    elif rank_change_places < 0:
        rank_change_label = f"↓ {abs(rank_change_places):,} places lost (GW{prev_gw}→GW{latest_gw})"
    else:
        rank_change_label = f"No change (GW{prev_gw}→GW{latest_gw})"
else:
    rank_history_rows = rank_history_rows[-2:]

# Public pick history includes purchase prices on many FPL responses. Use them to estimate
# the manager-specific selling price rather than assuming every asset sells at today's market price.
purchase_price_by_id = {}
purchase_price_current_ids = set()
purchase_price_source_by_id = {}
for _pick in public_picks_for_display:
    try:
        if _pick.get("purchase_price") is not None:
            _price_id = int(_pick.get("element"))
            purchase_price_by_id[_price_id] = int(_pick.get("purchase_price"))
            purchase_price_current_ids.add(_price_id)
            purchase_price_source_by_id[_price_id] = (
                "Current FPL picks" if picks_sync_ok else
                f"Last published GW{squad_load.get('baseline_gw')} picks"
            )
    except (TypeError, ValueError):
        continue
if not purchase_price_by_id:
    try:
        _finished_event_ids = [int(e["id"]) for e in events if e.get("finished") and e.get("id") is not None]
        _last_finished_event = max(_finished_event_ids) if _finished_event_ids else None
        if _last_finished_event:
            _last_picks_payload = get_entry_picks(int(team_id), int(_last_finished_event))
            _last_picks = _last_picks_payload.get("picks", []) if isinstance(_last_picks_payload, dict) else []
            for _pick in _last_picks:
                if _pick.get("purchase_price") is not None and _pick.get("element") is not None:
                    purchase_price_by_id[int(_pick["element"])] = int(_pick["purchase_price"])
                    purchase_price_source_by_id[int(_pick["element"])] = "Last completed GW picks"
    except Exception:
        pass
for _transaction in squad_load.get("transactions", []):
    if _transaction.get("purchase_price") is not None:
        purchase_price_by_id[int(_transaction["in"])] = int(_transaction["purchase_price"])
        purchase_price_current_ids.add(int(_transaction["in"]))
        purchase_price_source_by_id[int(_transaction["in"])] = "Public transfer record"
if squad_load.get("needs_confirmation"):
    for _pid, _price in manual_purchase_price_overrides.items():
        if _pid in roster_ids and _purchase_prices_confirmed:
            purchase_price_by_id[_pid] = int(_price)
            purchase_price_current_ids.add(int(_pid))
            purchase_price_source_by_id[int(_pid)] = "Manager-verified FPL purchase price"

_player_projection_series_cache = {}
def _projection_series_for(player):
    player_id = int(player["id"])
    if player_id not in _player_projection_series_cache:
        club_code = teams.get(player.get("team"), {}).get("short_name", "?")
        series = gameweek_projection_series(
            player, club_code, fixtures, horizon=int(horizon), start_gw=int(gw),
            estimate_points=estimate_points,
        )
        if series and player_id in SNAPSHOT_GW1:
            series[0]["points"] = float(SNAPSHOT_GW1[player_id])
        _player_projection_series_cache[player_id] = series
    return [dict(row) for row in _player_projection_series_cache[player_id]]

# Profile, history and public squad sync loaded above; visual overview is rendered after squad matching.
resolved=[]
unmatched=[]
for row in squad_rows:
    p = (next((candidate for candidate in players if int(candidate.get("id", -1)) == int(row[3])), None)
         if len(row) > 3 else resolve_player(row, players))
    if p:
        teamname=teams.get(p.get("team"),{}).get("short_name","?")
        _projection_series = _projection_series_for(p)
        pts = round(sum(row["points"] for row in _projection_series), 1)
        _,diff,nfx=estimate_points(p, teamname, fixtures, horizon, start_gw=gw, calibration_offsets={})
        resolved.append({
            "Player": p.get("web_name"), "Position": pos_name(p,types), "Club": teamname,
            "Team ID": int(p.get("team")) if p.get("team") is not None else None,
            "Price (£m)": round((p.get("now_cost") or 0)/10,1),
            "Sell price (£m)": round(official_selling_price_tenths(
                purchase_price_by_id.get(int(p["id"]), int(p.get("now_cost") or 0)),
                int(p.get("now_cost") or 0),
            ) / 10, 1),
            "Sell price (tenths)": official_selling_price_tenths(
                purchase_price_by_id.get(int(p["id"]), int(p.get("now_cost") or 0)),
                int(p.get("now_cost") or 0),
            ),
            "Selling price source": purchase_price_source_by_id.get(int(p["id"]), "Estimated from current price"),
            "Form": float(p.get("form") or 0), "PPG": float(p.get("points_per_game") or 0),
            "Minutes": int(p.get("minutes") or 0), "Status": p.get("status","?"),
            "Chance (%)": p.get("chance_of_playing_next_round"),
            f"Proj GW{gw}–GW{gw+horizon-1}": pts, "Avg FDR": diff,
             "xG": round(float(p.get("expected_goals") or 0), 2),
            "xA": round(float(p.get("expected_assists") or 0), 2),
            "xG/90": round(float(p.get("expected_goals") or 0) / max(90, int(p.get("minutes") or 0)) * 90, 3),
            "xA/90": round(float(p.get("expected_assists") or 0) / max(90, int(p.get("minutes") or 0)) * 90, 3),
             "xG": round(float(p.get("expected_goals") or 0), 2),
        "xA": round(float(p.get("expected_assists") or 0), 2),
        "xG/90": round(float(p.get("expected_goals") or 0) / max(90, int(p.get("minutes") or 0)) * 90, 3),
        "xA/90": round(float(p.get("expected_assists") or 0) / max(90, int(p.get("minutes") or 0)) * 90, 3),
        "Selected (%)": float(p.get("selected_by_percent") or 0), "id": p["id"],
        })
    else:
        unmatched.append(row[2])

# Select one legal XI once. The pitch, bench, captain planner and chip estimate
# all consume this exact lineup and the same one-Gameweek projection map.
_players_by_id = {int(player["id"]): player for player in players if player.get("id") is not None}
lineup_projection_by_id = {}
lineup_candidates = []
for _resolved in resolved:
    _pid = int(_resolved["id"])
    _player = _players_by_id.get(_pid)
    if not _player:
        continue
    _projection = _projection_series_for(_player)[0]["points"]
    lineup_projection_by_id[_pid] = float(_projection)
    lineup_candidates.append({
        "Player": str(_resolved["Player"]), "Position": str(_resolved["Position"]),
        "Club": str(_resolved["Club"]), "Price (£m)": float(_resolved["Price (£m)"]),
        "GW projected pts": float(_projection), "id": _pid,
    })
selected_lineup = (select_model_lineup(lineup_candidates, lineup_projection_by_id)
                   if squad_recommendations_allowed else None)
model_xi_start_ids = ({int(player["id"]) for player in selected_lineup["starters"]}
                      if selected_lineup else set())
# Before confirmation, show the verified last-published roster's positional layout
# as a reference. It is not treated as a current model lineup or recommendation.
reference_lineup = None
if selected_lineup:
    reference_lineup = selected_lineup
elif squad_load.get("available") and len(resolved) == 15:
    _resolved_by_id = {int(row["id"]): row for row in resolved}
    _ordered_reference = [
        {"id": int(pick["element"]), "Player": _resolved_by_id[int(pick["element"])]["Player"],
         "Position": _resolved_by_id[int(pick["element"])]["Position"],
         "GW projected pts": float(lineup_projection_by_id.get(int(pick["element"]), 0.0))}
        for pick in public_picks_for_display if int(pick["element"]) in _resolved_by_id
    ]
    if len(_ordered_reference) == 15:
        _ref_ids = {int(pick["element"]): pick for pick in public_picks_for_display}
        _reference_captain = next((p for p in _ordered_reference if _ref_ids[p["id"]].get("is_captain")), None)
        _reference_vice = next((p for p in _ordered_reference if _ref_ids[p["id"]].get("is_vice_captain")), None)
        _role_order = {"GK": 1, "GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
        _starters = sorted(_ordered_reference[:11], key=lambda p: (_role_order.get(p["Position"].upper(), 9), p["id"]))
        _bench = _ordered_reference[11:]
        _formation_counts = {role: sum(p["Position"].upper() == role for p in _starters) for role in ("DEF", "MID", "FWD")}
        reference_lineup = {
            "starters": _starters, "bench": _bench, "captain": _reference_captain,
            "vice": _reference_vice,
            "formation": f"{_formation_counts['DEF']}-{_formation_counts['MID']}-{_formation_counts['FWD']}",
            "source_reference": True,
        }

def _deadline_text(deadline, zone_name, zone_label):
    if deadline is None:
        return "Not announced"
    try:
        localized = deadline.astimezone(ZoneInfo(zone_name))
        return f"{localized:%a %d %b · %H:%M} {zone_label}"
    except (OverflowError, ValueError):
        return "Unavailable"


if deadline_dt is None:
    deadline_countdown = "Not announced"
else:
    _countdown_delta = deadline_dt - datetime.now(timezone.utc)
    if _countdown_delta.total_seconds() <= 0:
        deadline_countdown = "Deadline passed"
    else:
        _countdown_days, _countdown_remainder = divmod(int(_countdown_delta.total_seconds()), 86400)
        _countdown_hours, _countdown_remainder = divmod(_countdown_remainder, 3600)
        deadline_countdown = f"{_countdown_days}d {_countdown_hours}h {_countdown_remainder // 60}m"

_model_status = "Trained" if fpl_learning.validate_model(fpl_engine.TRAINED_MODEL) else "Fallback"
_squad_source_status = (
    f"verified public picks · GW{gw}" if squad_source == "current" else
    f"auto-reconstructed from GW{squad_load.get('baseline_gw')} · provisional until GW{gw} picks publish"
    if squad_source == "prior" and squad_recommendations_allowed else
    f"last published GW{squad_load.get('baseline_gw')} · manual correction needed"
    if squad_source == "prior" else
    "unavailable · check public picks"
)
if squad_source == "prior" and auto_controls and not squad_load.get("transfer_history_reconstructed"):
    _bank_state, _ft_state = "prior deadline · verify", "verify in FPL"
st.markdown(
    "<div class='decision-header'>"
    f"<div class='decision-brand'>⚽ The-One-Above-All · GW{int(gw)}</div>"
    "<div class='decision-deadlines'>"
    f"<div class='deadline-chip'><span>Deadline · UK</span><strong>{html.escape(_deadline_text(deadline_dt, 'Europe/London', 'UK'))}</strong></div>"
    f"<div class='deadline-chip'><span>Deadline · IST</span><strong>{html.escape(_deadline_text(deadline_dt, 'Asia/Kolkata', 'IST'))}</strong></div>"
    f"<div class='deadline-chip countdown'><span>Time remaining</span><strong>{html.escape(deadline_countdown)}</strong></div>"
    "</div></div>"
    "<div class='status-strip'>"
    f"<span class='status-chip'><strong>Model</strong> {html.escape(_model_status)}</span>"
    f"<span class='status-chip'><strong>Squad</strong> {html.escape(_squad_source_status)}</span>"
    f"<span class='status-chip'><strong>Bank</strong> {f'£{bank_m:.1f}m' if bank_m is not None else 'Unavailable'} · {html.escape(_bank_state)}</span>"
    f"<span class='status-chip'><strong>Free transfers</strong> {int(free_transfers)} · {html.escape(_ft_state)}</span>"
    "</div>",
    unsafe_allow_html=True,
)

with st.expander("How decisions are made", expanded=False):
    st.markdown("**Decision rule:** HOLD unless a legal multi-transfer plan clears the projected starting-XI advantage threshold after hits. Check the official FPL bank and selling price before acting. This dashboard never submits transfers automatically.")
    st.write("Plans compare hold and roll with up to all free transfers plus two paid transfers. Paid plans always appear for comparison and require you to accept their hit.")
    st.markdown("""
    - Blends PPG/form with xG+xA, shrinks small samples toward a position baseline, estimates minutes from starts and appearances, and adjusts scheduled fixtures individually.
    - Uses public player status and chance-of-playing fields where available.
    - Does not access your FPL password, private `my-team` endpoint, or personal selling prices.
    - Model weights remain heuristic and are evaluated against historical and live validation metrics.
    - Public FPL data is unofficial and may change. Refresh requests use the existing cache and refresh controls.
    - Current public picks are preferred. Before deadline, the last completed picks and complete public transfer history are used to reconstruct a provisional squad. Manual correction appears only if that evidence is missing, inconsistent or ambiguous.
    """)

decide_tab, squad_tab, fixtures_tab, model_tab = st.tabs(["Decide", "Squad", "Fixtures", "Model"])

captain_pick = None
vice_pick = None

with squad_tab:
    # FPL-inspired manager profile on the left and Gameweek pitch on the right.
    profile_col, pitch_col = st.columns([0.72, 1.85], gap="large")

    with profile_col:
        live_points = entry.get("summary_overall_points", "—") if entry else "—"
        live_rank = entry.get("summary_overall_rank", "—") if entry else "—"
        live_value = entry.get("last_deadline_value") if entry else None
        value_text = f"£{live_value/10:.1f}m" if isinstance(live_value, (int,float)) else "—"
        gw_points_summary = entry.get("summary_event_points") if entry else None
        if gw_points_summary is None:
            gw_points_summary = latest_history.get("points", "—") if latest_history else "—"
        gw_rank_summary = entry.get("summary_event_rank") if entry else None
        if gw_rank_summary is None:
            gw_rank_summary = latest_history.get("rank", "—") if latest_history else "—"
        st.markdown(
            "<div class='manager-panel'>"
            "<div class='manager-name'>⚽ The-One-Above-All</div>"
            "<div class='manager-label'>Manager profile · team ID "+str(int(team_id))+"</div>"
            "<div class='panel-heading'>Points &amp; rankings</div>"
            f"<div class='rank-row'><span>Season total points</span><strong>{html.escape(str(live_points))}</strong></div>"
            f"<div class='rank-row'><span>Overall rank</span><strong>{html.escape(f'{live_rank:,}' if isinstance(live_rank,int) else str(live_rank))}</strong></div>"
            f"<div class='rank-row'><span>Points earned this Gameweek</span><strong>{html.escape(str(gw_points_summary))}</strong></div>"
            f"<div class='rank-row'><span>Gameweek rank</span><strong>{html.escape(f'{gw_rank_summary:,}' if isinstance(gw_rank_summary,int) else str(gw_rank_summary))}</strong></div>"
            f"<div class='rank-row'><span>Overall-rank movement</span><strong>{html.escape(rank_change_label)}</strong></div>"
            f"<div class='manager-overview-note'>Rank movement is calculated from the latest two Gameweek history entries with overall-rank values. If GW7 is not present in the public history yet, this will keep showing the newest available comparison until FPL publishes GW7's rank.</div>"
            f"<div class='rank-row'><span>Team value</span><strong>{html.escape(value_text)}</strong></div>"
            "</div>",
            unsafe_allow_html=True
        )
        _past_seasons = history_payload.get("past", []) if isinstance(history_payload, dict) else []
        _past_seasons = [x for x in _past_seasons if isinstance(x, dict) and x.get("season_name")]
        def _season_start_year(row):
            match = re.search(r"(20\d{2})", str(row.get("season_name", "")))
            return int(match.group(1)) if match else 0
        _latest_past = max(_past_seasons, key=_season_start_year) if _past_seasons else None
        if _latest_past:
            _current_points = entry.get("summary_overall_points") if entry else None
            _current_rank = entry.get("summary_overall_rank") if entry else None
            _completed_gw = int(latest_history.get("event") or 0) if latest_history else 0
            _past_rank_raw = str(_latest_past.get("rank", "—"))
            _past_rank_text = f"{int(_past_rank_raw):,}" if _past_rank_raw.isdigit() else _past_rank_raw
            _current_rank_raw = str(_current_rank if _current_rank is not None else "—")
            _current_rank_text = f"{int(_current_rank_raw):,}" if _current_rank_raw.isdigit() else _current_rank_raw
            st.markdown(
                "<div class='manager-panel'><div class='panel-heading'>Previous season vs this season</div>"
                f"<div class='rank-row'><span>Previous season ({html.escape(str(_latest_past.get('season_name','')))}) · final points</span><strong>{html.escape(str(_latest_past.get('total_points','—')))}</strong></div>"
                f"<div class='rank-row'><span>Previous season · final rank</span><strong>{html.escape(_past_rank_text)}</strong></div>"
                f"<div class='rank-row'><span>Current season · points to date (GW{_completed_gw})</span><strong>{html.escape(str(_current_points if _current_points is not None else '—'))}</strong></div>"
                f"<div class='rank-row'><span>Current season · rank to date</span><strong>{html.escape(_current_rank_text)}</strong></div>"
                "<div class='manager-overview-note'>Previous season figures are final; current-season figures are a partial-season checkpoint. Use the same Gameweek checkpoint for a fair progress comparison.</div></div>",
                unsafe_allow_html=True
            )
        st.markdown(
            "<div class='manager-panel'><div class='panel-heading'>Squad controls</div>"
            f"<div class='rank-row'><span>Bank ({'auto' if auto_controls else 'manual'})</span><strong>{f'£{bank_m:.1f}m' if bank_m is not None else 'Unavailable'}</strong></div>"
            f"<div class='rank-row'><span>Free transfers ({'estimated' if auto_controls else 'manual'})</span><strong>{free_transfers}</strong></div>"
            f"<div class='rank-row'><span>Points horizon</span><strong>{horizon} GWs</strong></div>"
            f"<div class='rank-row'><span>Fixture calendar</span><strong>{fixture_horizon} GWs</strong></div>"
            f"<div class='rank-row'><span>Risk mode</span><strong>{html.escape(risk)}</strong></div>"
            "</div>",
            unsafe_allow_html=True
        )
        # Saved, non-live league rank is deliberately labelled as the last screenshot checkpoint.
        st.markdown(
            "<div class='manager-panel'><div class='panel-heading'>League tracking</div>"
            "<div class='league-pill'><span>India league rank</span><strong>Not connected</strong></div>"
            "<div class='manager-overview-note'>This release does not fetch India-league rank, so it avoids displaying an outdated saved number.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with pitch_col:
        _latest_gw_label = latest_history.get("event", "—") if latest_history else "—"
        _lineup_source_label = (
            f"model-selected XI from verified GW{gw} picks" if selected_lineup and squad_source == "current" else
            "model-selected from auto-reconstructed prior squad · provisional" if selected_lineup and squad_source == "prior" else
            "model-selected from verified public picks" if selected_lineup else
            f"last published GW{squad_load.get('baseline_gw')} reference · confirm before recommendations" if reference_lineup else
            "unavailable · public squad could not be verified"
        )
        st.markdown(f"**Latest GW result · GW{_latest_gw_label}**")
        st.caption("Latest recorded points and rank from the public manager history.")
        st.markdown(f"**Next GW · GW{int(gw)}**")
        st.caption(f"Lineup source: {_lineup_source_label}. Pitch numbers are projections for the selected horizon.")
        if unmatched:
            st.warning("Names not uniquely matched to public data (review manually): " + ", ".join(unmatched))
        if resolved:
            squad = pd.DataFrame(resolved)
            def esc(value):
                import html
                return html.escape(str(value))
            squad_lookup = {norm(r["Player"]): r for _, r in squad.iterrows()}
            def get_squad_record(alias):
                return squad_lookup.get(norm(alias))
            def tile(alias):
                r = get_squad_record(alias)
                if r is None:
                    return f"<div class='player-tile'><div class='tile-top'>{esc(alias)}</div><div class='tile-avatar'>?</div><div class='tile-meta'>Data pending</div></div>"
                short = str(r["Player"])
                _role = ("C" if selected_lineup and int(r["id"]) == int(selected_lineup["captain"]["id"])
                        else "VC" if selected_lineup and int(r["id"]) == int(selected_lineup["vice"]["id"]) else "")
                pts = lineup_projection_by_id.get(int(r["id"]), 0.0)
                club = str(r["Club"])
                color = "#52715e"
                shirt = (f"<svg class='kit-icon' viewBox='0 0 64 64' aria-label='shirt'>"
                         f"<path d='M21 8 L13 12 L4 28 L14 34 L20 24 L20 56 L44 56 L44 24 L50 34 L60 28 L51 12 L43 8 C39 15 25 15 21 8Z' fill='{color}' stroke='#ffffff' stroke-width='1.7' stroke-linejoin='round'/>"
                         f"<path d='M24 8 C26 18 38 18 40 8' fill='none' stroke='#f9f6fb' stroke-width='3'/>"
                         f"<text x='32' y='39' text-anchor='middle' font-size='12' font-weight='900' fill='#ffffff'> {esc(club[:3])} </text></svg>")
                return (f"<div class='player-tile'><div class='tile-top'>{esc(short)}{(' · ' + _role) if _role else ''}</div>"
                        f"<div class='tile-avatar'>{shirt}</div><div class='tile-meta'>{esc(club)} · £{float(r['Price (£m)']):.1f}m<br>"
                         f"<span class='tile-pts'>{float(pts):.1f} proj.</span></div></div>")
            if reference_lineup:
                _role_titles = {"GK": "GOALKEEPER", "GKP": "GOALKEEPER", "DEF": "DEFENDERS", "MID": "MIDFIELDERS", "FWD": "FORWARDS"}
                _lineup_groups = {"GOALKEEPER": [], "DEFENDERS": [], "MIDFIELDERS": [], "FORWARDS": []}
                for _player in reference_lineup["starters"]:
                    _lineup_groups[_role_titles[_player["Position"].upper()]].append(_player["Player"])
                xi_rows = list(_lineup_groups.items())
                bench_names = [player["Player"] for player in reference_lineup["bench"]]
            else:
                xi_rows, bench_names = [], []
            last_gw = latest_history if latest_history else {}
            current_gw_points = entry.get("summary_event_points") if entry else None
            if current_gw_points is None:
                current_gw_points = last_gw.get("points", "—")
            current_gw_rank = entry.get("summary_event_rank") if entry else None
            if current_gw_rank is None:
                current_gw_rank = last_gw.get("rank", "—")
            scoreband = (
                "<div class='gw-scoreband'>"
                f"<div class='gw-scorebox'><div class='num'>{esc(current_gw_rank if current_gw_rank is not None else '—')}</div><div class='lbl'>GW rank</div></div>"
                f"<div class='gw-scorebox featured'><div class='num'>{esc(current_gw_points if current_gw_points is not None else '—')}</div><div class='lbl'>Latest points</div></div>"
                f"<div class='gw-scorebox'><div class='num'>{esc(entry.get('summary_overall_rank', '—') if entry else '—')}</div><div class='lbl'>Overall rank</div></div>"
                f"<div class='gw-scorebox'><div class='num'>{esc(entry.get('summary_overall_points', '—') if entry else '—')}</div><div class='lbl'>Total points</div></div>"
                "</div>"
            )
            st.markdown(scoreband, unsafe_allow_html=True)
            _pitch_title = ("Model-selected starting XI · one-GW projections" if selected_lineup
                            else f"GW{squad_load.get('baseline_gw')} published roster layout · reference only")
            pitch_html = f"<div class='pitch-wrap'><div class='pitch-lines'></div><div class='pitch-title'>{esc(_pitch_title)}</div>"
            for label, names in xi_rows:
                pitch_html += f"<div class='pitch-title'>{label}</div><div class='pitch-row'>"
                pitch_html += "".join(tile(name) for name in names)
                pitch_html += "</div>"
            _pitch_note = ("Formation, captain choices and Bench Boost estimates use this same selected XI. Projections are estimates, not guaranteed points."
                           if selected_lineup else "This is the last published positional layout, not a current lineup recommendation. Confirm or correct your current roster in the sidebar.")
            pitch_html += f"<div class='pitch-caption'>{esc(_pitch_note)}</div></div>"
            st.markdown(pitch_html, unsafe_allow_html=True)
            st.markdown("#### Bench")
            bench_parts=[]
            for name in bench_names:
                r=get_squad_record(name)
                if r is not None:
                    sub=f"{esc(r['Club'])} · £{float(r['Price (£m)']):.1f}m · {lineup_projection_by_id.get(int(r['id']), 0.0):.1f} GW pts"
                else:
                    sub="Data pending"
                bench_parts.append(f"<div class='bench-card'><div class='bench-name'>{esc(name)}</div><div class='bench-sub'>{sub}</div></div>")
            st.markdown("<div class='bench-grid'>"+"".join(bench_parts)+"</div>", unsafe_allow_html=True)
            total_current_cost = squad["Price (£m)"].sum()
            st.caption(f"Public prices sum to £{total_current_cost:.1f}m for matched players. This is not necessarily your squad's purchase/selling value.")
            with st.expander("View full squad statistics"):
                st.dataframe(squad.drop(columns=["id"]), use_container_width=True, hide_index=True)
        else:
            st.warning(picks_sync_note)


    st.subheader("Captain & vice-captain planner")
    st.markdown(
        "<div class='section-intro'>Ranks your current starting XI for the next Gameweek using the same transparent projection model, current fixture difficulty, availability and minutes signals.</div>",
        unsafe_allow_html=True
    )

    if resolved and squad_recommendations_allowed:
        player_by_id = {int(player["id"]): player for player in players if player.get("id") is not None}
        captain_pool_ids = set(model_xi_start_ids)
        captain_source = "Model-selected legal XI" if selected_lineup else "Unavailable: no verified legal XI"
        current_captain_id = None

        for pick in public_picks_for_display:
            try:
                element_id = int(pick.get("element"))
                if pick.get("is_captain"):
                    current_captain_id = element_id
            except (TypeError, ValueError):
                continue

        captain_rows = []
        for element_id in captain_pool_ids:
            player = player_by_id.get(int(element_id))
            if not player:
                continue
            club_code = teams.get(player.get("team"), {}).get("short_name", "?")
            gw_projection = lineup_projection_by_id.get(int(element_id), 0.0)
            _, avg_fdr, fixture_rows = estimate_points(player, club_code, fixtures, horizon=1, start_gw=gw)
            next_fixtures = [
                fx for fx in fixtures
                if fx.get("event") is not None and int(fx.get("event")) == int(gw)
                and not fx.get("finished")
                and (fx.get("team_h") == player.get("team") or fx.get("team_a") == player.get("team"))
            ]
            next_fixtures.sort(key=lambda fx: fx.get("kickoff_time") or "")
            opponent_labels = []
            for fx in next_fixtures:
                is_home = fx.get("team_h") == player.get("team")
                opp_id = fx.get("team_a") if is_home else fx.get("team_h")
                opponent = teams.get(opp_id, {}).get("short_name", "TBC")
                fdr_key = "team_h_difficulty" if is_home else "team_a_difficulty"
                fdr = fx.get(fdr_key)
                opponent_labels.append(f"{opponent} ({'H' if is_home else 'A'}, FDR {fdr if fdr is not None else '?'})")
            if len(next_fixtures) > 1:
                opponent_labels.append(f"DGW ×{len(next_fixtures)}")
            captain_rows.append({
                "Player": player.get("web_name", "Unknown"),
                "Club": club_code,
                "Next fixture(s)": " + ".join(opponent_labels) if opponent_labels else "Fixture TBC",
                "Fixtures this GW": len(next_fixtures),
                "Avg FDR": avg_fdr,
                "Projected GW points": gw_projection,
                "Captain 2× estimate": round(gw_projection * 2, 1),
                "Minutes": int(player.get("minutes") or 0),
                "Status": player.get("status", "?"),
                "Chance (%)": player.get("chance_of_playing_next_round"),
                "Ownership (%)": float(player.get("selected_by_percent") or 0),
                "Element ID": int(player["id"]),
            })

        if captain_rows:
            captain_df = pd.DataFrame(captain_rows).sort_values(
                ["Projected GW points", "Minutes"], ascending=False
            ).reset_index(drop=True)
            captain_pick = captain_df.loc[captain_df["Element ID"].eq(int(selected_lineup["captain"]["id"]))].iloc[0]
            vice_pick = captain_df.loc[captain_df["Element ID"].eq(int(selected_lineup["vice"]["id"]))].iloc[0]

            c1, c2 = st.columns(2, gap="medium")
            with c1:
                st.markdown(
                    "<div class='manager-panel'><div class='panel-heading'>⭐ Model captain pick</div>"
                    f"<div class='manager-name'>{html.escape(str(captain_pick['Player']))} · {html.escape(str(captain_pick['Club']))}</div>"
                    f"<div class='manager-label'>Next fixture: {html.escape(str(captain_pick['Next fixture(s)']))}</div>"
                    f"<div class='rank-row'><span>Projected GW points</span><strong>{captain_pick['Projected GW points']:.1f}</strong></div>"
                    f"<div class='rank-row'><span>Simple 2× captain estimate</span><strong>{captain_pick['Captain 2× estimate']:.1f}</strong></div>"
                    f"<div class='rank-row'><span>Minutes recorded</span><strong>{int(captain_pick['Minutes']):,}</strong></div></div>",
                    unsafe_allow_html=True
                )
            with c2:
                if vice_pick is not None:
                    st.markdown(
                        "<div class='manager-panel'><div class='panel-heading'>🛡️ Model vice-captain pick</div>"
                        f"<div class='manager-name'>{html.escape(str(vice_pick['Player']))} · {html.escape(str(vice_pick['Club']))}</div>"
                        f"<div class='manager-label'>Next fixture: {html.escape(str(vice_pick['Next fixture(s)']))}</div>"
                        f"<div class='rank-row'><span>Projected GW points</span><strong>{vice_pick['Projected GW points']:.1f}</strong></div>"
                        f"<div class='rank-row'><span>Simple 2× captain estimate</span><strong>{vice_pick['Captain 2× estimate']:.1f}</strong></div>"
                        f"<div class='rank-row'><span>Minutes recorded</span><strong>{int(vice_pick['Minutes']):,}</strong></div></div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.info("Only one eligible starter was available for the captaincy shortlist.")

            if current_captain_id is not None:
                current_captain_name = player_by_id.get(current_captain_id, {}).get("web_name", "Unknown")
                if picks_sync_ok:
                    st.caption(f"Currently recorded captain: {current_captain_name}. Candidate source: {captain_source}.")
                else:
                    st.caption(f"Last published GW{squad_load.get('baseline_gw')} captain: {current_captain_name}; this is not assumed for GW{gw}. Candidate source: {captain_source}.")
            else:
                st.caption(f"Candidate source: {captain_source}.")
            if not picks_sync_ok:
                st.warning("This captain recommendation uses your manager-confirmed prior roster and is provisional. Verify all transfers, chip changes and team news before the deadline.")
            if not any(int(row.get("Fixtures this GW", 0)) > 0 for row in captain_rows):
                st.info("Fixture assignments for the next Gameweek may not be complete yet. Captain recommendations are provisional.")

            st.markdown("#### Captaincy shortlist")
            st.dataframe(
                captain_df.drop(columns=["Element ID"]).head(7),
                use_container_width=True,
                hide_index=True
            )
            st.markdown(
                f"**Quick captain summary:** Model captain: **{captain_pick['Player']}**; vice-captain: **{vice_pick['Player'] if vice_pick is not None else 'not available'}**. "
                f"The estimates are for GW{gw} only and reflect known fixtures, minutes, form, xG/xA and availability inputs. "
                "They are heuristic estimates, not official FPL forecasts. Check press-conference news before locking the armband."
            )
        else:
            st.warning("Could not identify starting-XI players for the captaincy comparison. Refresh data or verify the squad.")
    elif resolved:
        st.info("Captain recommendations are withheld until you confirm or correct the current roster in the sidebar.")
    else:
        st.warning("Captaincy recommendations require matched squad players.")

with fixtures_tab:
    st.subheader("Fixtures & double-Gameweek planner")
    st.markdown(
        "<div class='section-intro'>Your squad's fixtures across the selected planning window, including home/away opponents and official FPL fixture difficulty (FDR). Easier fixtures generally have a lower FDR.</div>",
        unsafe_allow_html=True
    )

    # The slider's horizon counts distinct Gameweeks, not a fixed number of fixture rows.
    available_gws = sorted({
        int(ev.get("id")) for ev in events
        if ev.get("id") is not None and int(ev.get("id")) >= int(gw)
    })
    target_gws = available_gws[:int(fixture_horizon)]
    if not target_gws:
        target_gws = list(range(int(gw), int(gw) + int(fixture_horizon)))

    # Count assigned fixtures per club per Gameweek, including every row in a DGW.
    club_gw_counts = {}
    for fx in fixtures:
        eid = fx.get("event")
        if eid is None or int(eid) not in target_gws:
            continue
        for club_id in (fx.get("team_h"), fx.get("team_a")):
            if club_id is not None:
                key = (int(eid), int(club_id))
                club_gw_counts[key] = club_gw_counts.get(key, 0) + 1

    fixture_matrix_rows = []
    if resolved:
        squad_df_for_fx = pd.DataFrame(resolved)
        for _, player in squad_df_for_fx.iterrows():
            team_id_value = player.get("Team ID")
            if pd.isna(team_id_value):
                continue
            club_id = int(team_id_value)
            row = {
                "Player": player.get("Player", "—"),
                "Club": player.get("Club", "—"),
            }
            fixture_count = 0
            dgw_summary = []
            for event_id in target_gws:
                weekly = [
                    fx for fx in fixtures
                    if fx.get("event") is not None
                    and int(fx.get("event")) == int(event_id)
                    and not fx.get("finished")
                    and club_id in (fx.get("team_h"), fx.get("team_a"))
                ]
                weekly.sort(key=lambda fx: fx.get("kickoff_time") or "")
                fixture_count += len(weekly)
                cell_parts = []
                for fx in weekly:
                    is_home = int(fx.get("team_h")) == club_id
                    opp_id = fx.get("team_a") if is_home else fx.get("team_h")
                    opponent = teams.get(opp_id, {}).get("short_name", "TBC")
                    difficulty_key = "team_h_difficulty" if is_home else "team_a_difficulty"
                    fdr = fx.get(difficulty_key)
                    venue = "H" if is_home else "A"
                    fdr_label = f"FDR {fdr}" if fdr is not None else "FDR ?"
                    cell_parts.append(f"{opponent} ({venue}) · {fdr_label}")
                if not cell_parts:
                    row[f"GW{event_id}"] = "TBC / no assigned fixture"
                else:
                    is_double = len(weekly) > 1
                    row[f"GW{event_id}"] = " + ".join(cell_parts) + (" · DGW" if is_double else "")
                    if is_double:
                        dgw_summary.append(f"GW{event_id} ×{len(weekly)}")
            row["Fixture count"] = fixture_count
            row["Double GWs"] = ", ".join(dgw_summary) if dgw_summary else "—"
            fixture_matrix_rows.append(row)

    if fixture_matrix_rows:
        fixture_df = pd.DataFrame(fixture_matrix_rows)
        st.dataframe(fixture_df, use_container_width=True, hide_index=True)
        dgw_players = [f"{r['Player']} ({r['Double GWs']})" for r in fixture_matrix_rows if r["Double GWs"] != "—"]
        if dgw_players:
            st.success("Double-Gameweek watch for your squad: " + "; ".join(dgw_players))
        else:
            st.info("No double Gameweek is currently assigned to your squad within this horizon. Postponed matches become visible here once FPL assigns them to a Gameweek.")
    else:
        st.warning("Fixture matrix is unavailable until at least one squad player matches public FPL data.")

    with st.expander("View all published fixtures in these Gameweeks"):
        all_fixtures = []
        for fx in fixtures:
            eid = fx.get("event")
            if eid is None or int(eid) not in target_gws or fx.get("finished"):
                continue
            home_id, away_id = fx.get("team_h"), fx.get("team_a")
            if home_id is None or away_id is None:
                continue
            home_team = teams.get(home_id, {}).get("name", teams.get(home_id, {}).get("short_name", "Home"))
            away_team = teams.get(away_id, {}).get("name", teams.get(away_id, {}).get("short_name", "Away"))
            kickoff = fx.get("kickoff_time")
            if kickoff:
                try:
                    kickoff_label = pd.to_datetime(kickoff, utc=True).strftime("%a %d %b · %H:%M UTC")
                except Exception:
                    kickoff_label = str(kickoff)
            else:
                kickoff_label = "Kick-off TBC"
            is_dgw_fixture = (
                club_gw_counts.get((int(eid), int(home_id)), 0) > 1
                or club_gw_counts.get((int(eid), int(away_id)), 0) > 1
            )
            all_fixtures.append({
                "Gameweek": int(eid),
                "Kick-off": kickoff_label,
                "Fixture": f"{home_team} vs {away_team}",
                "Home FDR": fx.get("team_h_difficulty"),
                "Away FDR": fx.get("team_a_difficulty"),
                "Double-GW flag": "DGW fixture" if is_dgw_fixture else "",
            })
        if all_fixtures:
            all_fixtures_df = pd.DataFrame(all_fixtures).sort_values(["Gameweek", "Kick-off"])
            st.dataframe(all_fixtures_df, use_container_width=True, hide_index=True)
        else:
            st.info("No unpublished fixtures have been assigned to this horizon yet.")

    st.markdown(
        f"<div class='manager-overview-note'>Fixture calendar: GW{target_gws[0] if target_gws else gw}–GW{target_gws[-1] if target_gws else gw + fixture_horizon - 1}. "
        "Double Gameweeks are identified only when multiple fixtures have already been assigned to the same Gameweek in public FPL data. "
        "The app cannot reliably predict an unannounced DGW date in advance.</div>",
        unsafe_allow_html=True
    )

    st.subheader("Chip strategy & timing")
    st.markdown(
        "<div class='section-intro'>Use-now suitability scores are heuristic indicators built from this Gameweek's fixtures, projected points and reliability. They are not statistically calibrated probabilities.</div>",
        unsafe_allow_html=True
    )

    def chip_reliability(player):
        """Heuristic reliability weight: minutes history blended with FPL availability flags."""
        mins = float(player.get("minutes") or 0)
        minutes_rel = max(0.15, min(1.0, mins / 600.0))
        availability_rel = availability_adjustment(player)
        return 0.55 * minutes_rel + 0.45 * availability_rel

    elements_by_id_chip = {
        int(player["id"]): player for player in players if player.get("id") is not None
    }
    chip_starter_ids = [int(player["id"]) for player in selected_lineup["starters"]] if selected_lineup else []
    chip_bench_ids = [int(player["id"]) for player in selected_lineup["bench"]] if selected_lineup else []
    chip_squad_source = "model-selected XI and bench" if selected_lineup else "unavailable: no verified legal squad"
    _chip_player_ids = chip_starter_ids + chip_bench_ids
    if _chip_player_ids and all(pid in SNAPSHOT_GW1 for pid in _chip_player_ids):
        chip_projection_source = "saved pre-deadline v3.0 snapshot"
    elif any(pid in SNAPSHOT_GW1 for pid in _chip_player_ids):
        chip_projection_source = "mixed saved snapshot/live engine (partial archive coverage)"
    else:
        chip_projection_source = "live v3.0 prediction engine"

    # Next-GW projection and assigned fixtures for every player in the squad.
    chip_gw_fixture_counts = {}
    for fixture in fixtures:
        if fixture.get("event") is None or int(fixture.get("event")) != int(gw) or fixture.get("finished"):
            continue
        for club_id in (fixture.get("team_h"), fixture.get("team_a")):
            if club_id is not None:
                chip_gw_fixture_counts[int(club_id)] = chip_gw_fixture_counts.get(int(club_id), 0) + 1

    chip_player_rows = {}
    for player_id in set(chip_starter_ids + chip_bench_ids):
        player = elements_by_id_chip.get(int(player_id))
        if not player:
            continue
        club_code = teams.get(player.get("team"), {}).get("short_name", "?")
        projected = lineup_projection_by_id.get(int(player_id), 0.0)
        chip_player_rows[int(player_id)] = {
            "player": player,
            "projected": float(projected),
            "reliability": chip_reliability(player),
            "fixture_count": int(chip_gw_fixture_counts.get(int(player.get("team", -1)), 0)),
            "club": club_code,
        }

    starter_chip_rows = [chip_player_rows[pid] for pid in chip_starter_ids if pid in chip_player_rows]
    bench_chip_rows = [chip_player_rows[pid] for pid in chip_bench_ids if pid in chip_player_rows]

    top_captain_for_chip = max(starter_chip_rows, key=lambda row: row["projected"], default=None)
    captain_pts_chip = top_captain_for_chip["projected"] if top_captain_for_chip else 0.0
    captain_reliability_chip = top_captain_for_chip["reliability"] if top_captain_for_chip else 0.0
    captain_dgw_chip = bool(top_captain_for_chip and top_captain_for_chip["fixture_count"] >= 2)

    bench_points_chip = sum(row["projected"] for row in bench_chip_rows)
    bench_reliability_chip = (
        sum(row["reliability"] for row in bench_chip_rows) / len(bench_chip_rows)
        if bench_chip_rows else 0.0
    )
    bench_dgw_count_chip = sum(1 for row in bench_chip_rows if row["fixture_count"] >= 2)

    blank_starters_chip = sum(1 for row in starter_chip_rows if row["fixture_count"] == 0)
    blank_all_chip = sum(
        1 for row in (starter_chip_rows + bench_chip_rows) if row["fixture_count"] == 0
    )

    # The fixtures planner has already computed club_gw_counts and target_gws.
    squad_club_ids_chip = {
        int(row["Team ID"]) for row in resolved
        if row.get("Team ID") is not None and not pd.isna(row.get("Team ID"))
    }
    squad_dgw_gws_chip = sorted({
        int(event_id) for event_id in target_gws
        if any(club_gw_counts.get((int(event_id), club_id), 0) > 1 for club_id in squad_club_ids_chip)
    })
    league_dgw_gws_chip = sorted({
        int(event_id) for event_id in target_gws
        if any(
            club_gw_counts.get((int(event_id), int(club_id)), 0) > 1
            for club_id in teams.keys()
        )
    })

    # Compare this Gameweek with the best currently known later opportunity for chips.
    # These are fixture/projection comparisons, not guaranteed points or perfect future forecasts.
    future_chip_options = []
    all_chip_ids = sorted(set(chip_starter_ids + chip_bench_ids))
    for future_gw in sorted({int(x) for x in target_gws if int(x) > int(gw)}):
        future_points_by_id = {}
        future_fixture_counts = {}
        for player_id in all_chip_ids:
            player = elements_by_id_chip.get(int(player_id))
            if not player:
                continue
            club_code = teams.get(player.get("team"), {}).get("short_name", "?")
            future_pts, _, future_fx_count = estimate_points(player, club_code, fixtures, horizon=1, start_gw=future_gw)
            future_points_by_id[int(player_id)] = float(future_pts)
            future_fixture_counts[int(player_id)] = int(future_fx_count)
        future_bench_total = sum(future_points_by_id.get(pid, 0.0) for pid in chip_bench_ids)
        future_best_captain = max((future_points_by_id.get(pid, 0.0) for pid in chip_starter_ids), default=0.0)
        future_blanks = sum(1 for pid in chip_starter_ids if future_fixture_counts.get(pid, 0) == 0)
        future_chip_options.append({
            "gameweek": future_gw,
            "bench_points": future_bench_total,
            "captain_points": future_best_captain,
            "blank_starters": future_blanks,
        })
    best_future_bench = max(future_chip_options, key=lambda row: row["bench_points"], default=None)
    best_future_tc = max(future_chip_options, key=lambda row: row["captain_points"], default=None)
    best_future_fh = max(future_chip_options, key=lambda row: row["blank_starters"], default=None)

    # Readiness scores are intentionally transparent rule-based indexes.
    # They express how favourable a chip looks for THIS Gameweek, not probability of success.
    bench_score = round(30 * min(bench_points_chip / 18.0, 1.0)
                        + 20 * bench_reliability_chip
                        + 50 * min(bench_dgw_count_chip / 2.0, 1.0))
    tc_score = round(30 * min(captain_pts_chip / 14.0, 1.0)
                     + 20 * captain_reliability_chip
                     + (50 if captain_dgw_chip else 0))
    freehit_score = round(85 * min(blank_starters_chip / 6.0, 1.0)
                          + 15 * min(blank_all_chip / 8.0, 1.0))

    # Wildcard readiness estimates need for a rebuild, and is shown only when this half's chip is available.
    wildcard_distress = 0
    for row in starter_chip_rows:
        p = row["player"]
        status_value = str(p.get("status") or "").lower()
        if status_value in {"i", "s", "u", "d"} or row["reliability"] < 0.50 or row["fixture_count"] == 0:
            wildcard_distress += 1
    wildcard_score = min(100, wildcard_distress * 17)

    def chip_state_is_available(value):
        return value == "Available"

    def chip_card(title, status_text, score_text, action_text, detail):
        return (
            "<div class='manager-panel'>"
            f"<div class='panel-heading'>{html.escape(title)}</div>"
            f"<div class='manager-name'>{html.escape(score_text)}</div>"
            f"<div class='manager-label'>{html.escape(status_text)}</div>"
            f"<div class='league-pill'><span>Model guidance</span><strong>{html.escape(action_text)}</strong></div>"
            f"<div class='manager-overview-note'>{html.escape(detail)}</div>"
            "</div>"
        )

    chip_recommendations = []
    if wildcard_status == "Available":
        wc_action = "Consider rebuild" if wildcard_score >= 50 else "Save for a clearer need"
        wc_detail = f"{wildcard_distress} potential starting-XI issues flagged by public minutes/status/fixture signals. This is a rough rebuild-need indicator, not a full squad optimization."
        chip_recommendations.append(("Wildcard", wildcard_status, f"{wildcard_score}% readiness", wc_action, wc_detail))
    else:
        if int(gw) < 20:
            wc_detail = "First-half Wildcard is already marked as played. The second Wildcard set unlocks from GW20 under the 2026/27 rules."
        else:
            wc_detail = "Wildcard is marked as played or unverified for this half. Update status in the sidebar if needed."
        chip_recommendations.append(("Wildcard", wildcard_status, "—", "Not available now", wc_detail))

    if chip_state_is_available(bench_boost_status):
        _bench_components = " · ".join(
            f"{row['player'].get('web_name', 'Player')} {row['projected']:.1f}"
            for row in bench_chip_rows
        ) or "not available"
        _bench_basis = f"GW{gw}, one-Gameweek estimate · {chip_squad_source} · {chip_projection_source}"
        if bench_dgw_count_chip >= 2 and bench_points_chip >= 12 and bench_reliability_chip >= 0.70:
            bb_action = "Consider this GW"
            bb_detail = f"{_bench_basis}: {_bench_components} = {bench_points_chip:.1f} points; {bench_dgw_count_chip} bench player(s) have a Double Gameweek."
        elif bench_dgw_count_chip > 0:
            bb_action = "Monitor / compare bench"
            bb_detail = f"{_bench_basis}: {_bench_components} = {bench_points_chip:.1f} points; {bench_dgw_count_chip} bench player(s) have a Double Gameweek."
        else:
            bb_action = "Save for a stronger bench/DGW"
            bb_detail = f"{_bench_basis}: {_bench_components} = {bench_points_chip:.1f} points across {len(bench_chip_rows)} matched bench players. No bench Double Gameweek is currently assigned."
        if best_future_bench and best_future_bench["bench_points"] >= bench_points_chip + 2.0:
            bb_action = f"Save for GW{best_future_bench['gameweek']}"
            bb_detail = (f"{_bench_basis}: {_bench_components} = {bench_points_chip:.1f}; the same four-player bench projects "
                        f"{best_future_bench['bench_points']:.1f} in one Gameweek in GW{best_future_bench['gameweek']} using the live prediction engine. Recheck closer to that deadline.")
        elif bench_points_chip >= 12 and bench_reliability_chip >= 0.70:
            bb_action = "Potential use this GW"
            bb_detail = (f"{_bench_basis}: {_bench_components} = {bench_points_chip:.1f} with heuristic reliability {bench_reliability_chip:.0%}. "
                        + (f"Best later known bench estimate: {best_future_bench['bench_points']:.1f} in GW{best_future_bench['gameweek']}." if best_future_bench else "No later bench opportunity is currently comparable in the visible fixture window."))
        else:
            bb_action = "Hold — bench not strong enough yet"
            bb_detail = (f"Current bench estimate is {bench_points_chip:.1f}. Use only when the bench has a materially stronger projected total; "
                        + (f"the best later known estimate is {best_future_bench['bench_points']:.1f} in GW{best_future_bench['gameweek']}." if best_future_bench else "no later fully scheduled opportunity is currently available."))
        bb_score_text = f"{bench_score}% use-now suitability"
    else:
        bb_action = "Already used / verify status"
        bb_detail = "This chip is not marked available. Update the sidebar chip status if this is incorrect."
        bb_score_text = "—"

    if chip_state_is_available(triple_captain_status):
        if captain_dgw_chip and captain_pts_chip >= 10 and captain_reliability_chip >= 0.70:
            tc_action = "Consider this GW"
            tc_detail = f"Best current XI projection is {captain_pts_chip:.1f} points and that player's club has multiple fixtures in this Gameweek."
        elif captain_dgw_chip:
            tc_action = "Check minutes/news first"
            tc_detail = f"Best current XI projection is {captain_pts_chip:.1f} points, but the reliability signal is {captain_reliability_chip:.0%}."
        else:
            tc_action = "Usually save for a premium DGW"
            tc_detail = f"Best current captain projection is {captain_pts_chip:.1f} points; no Double Gameweek is assigned to that captain's club this GW."
        if best_future_tc and best_future_tc["captain_points"] >= captain_pts_chip + 1.5:
            tc_action = f"Save for GW{best_future_tc['gameweek']}"
            tc_detail = (f"Current best captain estimate: {captain_pts_chip:.1f}; best later projected squad player is "
                        f"{best_future_tc['captain_points']:.1f} in GW{best_future_tc['gameweek']}. This comparison is based on currently published fixtures and current player inputs.")
        elif captain_dgw_chip and captain_pts_chip >= 8.0 and captain_reliability_chip >= 0.70:
            tc_action = "Potential use this GW"
            tc_detail = (f"Current captain candidate projects {captain_pts_chip:.1f} across this Gameweek's assigned fixtures with reliability {captain_reliability_chip:.0%}. "
                        + (f"Best later projected captain: {best_future_tc['captain_points']:.1f} in GW{best_future_tc['gameweek']}." if best_future_tc else "No later stronger captain projection is visible."))
        else:
            tc_action = "Hold — wait for a stronger captain spot"
            tc_detail = (f"Current captain candidate projects {captain_pts_chip:.1f}. "
                        + (f"Best later candidate currently projects {best_future_tc['captain_points']:.1f} in GW{best_future_tc['gameweek']}." if best_future_tc else "No later candidate can be compared yet."))
        tc_score_text = f"{tc_score}% use-now suitability"
    else:
        tc_action = "Already used / verify status"
        tc_detail = "This chip is not marked available. Update the sidebar chip status if this is incorrect."
        tc_score_text = "—"

    if chip_state_is_available(free_hit_status):
        if blank_starters_chip >= 4:
            fh_action = "Consider this GW"
            fh_detail = f"{blank_starters_chip} of the recorded/published starting XI have no fixture. Compare a Free Hit XI against normal transfers."
        elif blank_starters_chip >= 2:
            fh_action = "Review the blank risk"
            fh_detail = f"{blank_starters_chip} starters have no fixture; compare against the next known blank before using the chip."
        else:
            fh_action = "Save for a major Blank GW"
            fh_detail = f"{blank_starters_chip} starting players currently have no fixture. A Free Hit is usually more valuable when several starters blank."
        if best_future_fh and best_future_fh["blank_starters"] >= blank_starters_chip + 1 and best_future_fh["blank_starters"] >= 3:
            fh_action = f"Save for GW{best_future_fh['gameweek']}"
            fh_detail = (f"Current XI has {blank_starters_chip} blanks; the largest later blank signal in the visible schedule is "
                        f"{best_future_fh['blank_starters']} in GW{best_future_fh['gameweek']}. This is only based on announced fixtures.")
        elif blank_starters_chip >= 4:
            fh_action = "Potential use this GW"
            fh_detail = (f"{blank_starters_chip} current starters have no fixture. Compare the projected Free Hit XI with the normal team before confirming. "
                        + (f"Best later known blank count: {best_future_fh['blank_starters']} in GW{best_future_fh['gameweek']}." if best_future_fh else "No later blank opportunity is currently assigned."))
        else:
            fh_action = "Hold for a major Blank GW"
            fh_detail = (f"Current XI has {blank_starters_chip} blanks. Usually wait until several starters lack a fixture. "
                        + (f"Largest later known blank count is {best_future_fh['blank_starters']} in GW{best_future_fh['gameweek']}." if best_future_fh else "No major later blank is visible yet."))
        fh_score_text = f"{freehit_score}% use-now suitability"
    else:
        fh_action = "Already used / verify status"
        fh_detail = "This chip is not marked available. Update the sidebar chip status if this is incorrect."
        fh_score_text = "—"

    chip_cards = [
        ("Wildcard", wildcard_status, wc_score_text if "wc_score_text" in locals() else ("—" if wildcard_status != "Available" else f"{wildcard_score}% rebuild-need score"), wc_action if "wc_action" in locals() else "Check status", wc_detail if "wc_detail" in locals() else "Update status in the sidebar."),
        ("Bench Boost", bench_boost_status, bb_score_text, bb_action, bb_detail),
        ("Triple Captain", triple_captain_status, tc_score_text, tc_action, tc_detail),
        ("Free Hit", free_hit_status, fh_score_text, fh_action, fh_detail),
    ]
    if not selected_lineup:
        chip_cards = [
            (name, status, "—", "Not available", "Confirm a current roster in the sidebar before using a chip estimate.")
            if status == "Available" else (name, status, score, action, detail)
            for name, status, score, action, detail in chip_cards
        ]
    chip_cols = st.columns(4, gap="small")
    for chip_col, card in zip(chip_cols, chip_cards):
        with chip_col:
            st.markdown(chip_card(*card), unsafe_allow_html=True)

    future_dgw_msg = (
        f"Next assigned Double Gameweek(s) for your squad in the visible calendar: {', '.join('GW'+str(x) for x in squad_dgw_gws_chip)}."
        if squad_dgw_gws_chip else
        "No Double Gameweek is currently assigned to your squad in the visible calendar. The panel will update after FPL assigns any postponed fixtures to a Gameweek."
    )
    st.markdown(
        f"<div class='manager-overview-note'><strong>Chip timing:</strong> {html.escape(future_dgw_msg)} "
        f"Current window: {html.escape(chip_period_label)}. First-half chips must be used by the GW19 deadline and do not carry over; the second set unlocks from GW20. Only one chip can be active in one Gameweek. Scores are heuristic suitability indexes, not probabilities of future points.</div>",
        unsafe_allow_html=True
    )

with model_tab:
    st.subheader("Prediction engine & model validation")
    st.caption("v3.0 is a frozen validation release: it adds FPL ep_next benchmarking, full-player history coverage tracking, one- and five-Gameweek evaluation, and ranking metrics. It does not adapt its weights while we measure performance.")

    # Show next-Gameweek candidates without presenting projections as certainties.
    next_week_rows = []
    for _p in players:
        _gw_pts, _fdr, _fixture_count = estimate_points(_p, teams.get(_p.get("team"), {}).get("short_name", "?"), fixtures, 1, start_gw=gw, use_history=True, calibration_offsets={})
        _gw_pts = SNAPSHOT_GW1.get(int(_p.get("id", -1)), _gw_pts)
        _sig = PLAYER_HISTORY_SIGNALS.get(int(_p.get("id", -1)), {})
        _next_chance = _p.get("chance_of_playing_next_round")
        next_week_rows.append({
            "Player": _p.get("web_name"), "Position": pos_name(_p, types),
            "Club": teams.get(_p.get("team"), {}).get("short_name", "?"),
            "Price (£m)": round(float(_p.get("now_cost") or 0) / 10, 1),
            f"GW{gw} projected pts": _gw_pts, "FPL ep_next": pd.to_numeric(pd.Series([_p.get("ep_next")]), errors="coerce").iloc[0], "Fixtures": _fixture_count,
            "History enriched": int(int(_p.get("id", -1)) in PLAYER_HISTORY_SIGNALS),
            "Recent pts/appearance": round(float(_sig["recent_points_per_appearance"]), 2) if _sig.get("recent_points_per_appearance") is not None else None,
            "Recent mins/fixture": round(float(_sig["recent_minutes_per_fixture"]), 1) if _sig.get("recent_minutes_per_fixture") is not None else None,
            "History matches": int(_sig.get("sample_matches") or 0),
            "Status": _p.get("status", "?"), "Chance (%)": _next_chance,
            "Selected (%)": float(_p.get("selected_by_percent") or 0), "id": int(_p["id"]),
        })
    next_week_df = pd.DataFrame(next_week_rows)
    next_week_df = next_week_df[next_week_df["Fixtures"] > 0].sort_values(f"GW{gw} projected pts", ascending=False)
    engine_c1, engine_c2, engine_c3, engine_c4 = st.columns(4)
    engine_c1.metric("History-enriched players", f"{len(PLAYER_HISTORY_SIGNALS)}")
    engine_c2.metric("Candidates with a GW fixture", f"{len(next_week_df)}")
    _forecast_record_saved = (not _current_snapshot.empty and "projection_model" in _current_snapshot
                              and pd.to_numeric(_current_snapshot["projection_model"], errors="coerce").notna().any())
    engine_c3.metric("Forecast record", "Saved" if _forecast_record_saved else "Not saved",
                     help=f"Requires season {season_id}, GW{gw}, model version {APP_VERSION}; legacy or other-version records do not count.")
    engine_c4.metric("Model version", APP_VERSION)
    st.caption("Calibration is paused while v3.0 is benchmarked, so the tested model does not change weights between Gameweeks. Persistent forecast logging is performed by the scheduled worker in the final 6 hours before deadline when the model artifact and fixtures are valid.")
    if forecast_snapshot_note:
        st.caption(forecast_snapshot_note)
    if not next_week_df.empty:
        st.markdown(f"**Top projected players for GW{gw}**")
        st.dataframe(next_week_df.head(15).drop(columns=["id"]), use_container_width=True, hide_index=True)

    # The same verified XI is rendered in Squad, used for captain advice and chips.
    if selected_lineup:
        best_xi = selected_lineup["starters"]
        bench_ordered = selected_lineup["bench"]
        captain_pick = selected_lineup["captain"]
        vice_pick = selected_lineup["vice"]
        xi_a, xi_b, xi_c = st.columns([1.1, 1.1, 1.8])
        xi_a.metric("Model-best formation", selected_lineup["formation"])
        xi_b.metric("XI + captain projected points", f"{selected_lineup['projected_points']:.1f}",
                    help=(f"Starter projections total {selected_lineup['base_xi_projected_points']:.1f}; "
                          f"the captain's extra 1× projection adds {selected_lineup['captain_extra_points']:.1f}. "
                          "Vice-captain points are conditional and are not added."))
        xi_c.metric("Captain / vice-captain", f"{captain_pick['Player']} / {vice_pick['Player']}", help="These are selected from the same legal XI displayed on the Squad tab.")
        xi_tab, bench_tab = st.columns([1.4, 1])
        with xi_tab:
            st.markdown("**Model-selected starting XI**")
            xi_frame = pd.DataFrame(best_xi)
            xi_frame["Captain"] = xi_frame["id"].eq(captain_pick["id"]).map({True: "C", False: ""})
            xi_frame["Vice"] = xi_frame["id"].eq(vice_pick["id"]).map({True: "VC", False: ""})
            st.dataframe(xi_frame[["Position", "Player", "Club", "GW projected pts", "Captain", "Vice"]].rename(columns={"GW projected pts": f"GW{gw} pts"}), use_container_width=True, hide_index=True)
        with bench_tab:
            st.markdown("**Suggested bench order**")
            bench_frame = pd.DataFrame(bench_ordered)
            bench_frame["Order"] = ["Goalkeeper bench", "Outfield 1", "Outfield 2", "Outfield 3"]
            st.dataframe(bench_frame[["Order", "Player", "Position", "GW projected pts"]].rename(columns={"GW projected pts": f"GW{gw} pts"}), use_container_width=True, hide_index=True)
        st.caption("Projection source is the current v3.0 one-Gameweek forecast when available, otherwise the live prediction engine. These estimates do not guarantee points or submit changes to FPL.")
    elif resolved and not squad_recommendations_allowed:
        st.info("A projected XI is withheld because current picks and public transfer history do not provide a complete, consistent roster. Correct the roster in the sidebar to continue.")
    elif resolved:
        st.warning("The current FPL roster does not form a valid 15-player squad and legal XI. Transfer and chip advice are withheld until roster data is verified.")
    else:
        st.info(picks_sync_note)

    # Update actual scores only in local fallback mode; the scheduled worker owns the persistent branch.
    forecast_log = read_forecast_log()
    finished_gws = {int(e["id"]) for e in events if e.get("finished") and e.get("id") is not None}
    if not remote_log_active and not forecast_log.empty:
        changed_actuals = False
        for _saved_gw in sorted(pd.to_numeric(forecast_log["gameweek"], errors="coerce").dropna().astype(int).unique()):
            _mask = (pd.to_numeric(forecast_log["gameweek"], errors="coerce") == int(_saved_gw))
            if int(_saved_gw) not in finished_gws or not forecast_log.loc[_mask, "actual_points"].isna().any():
                continue
            try:
                _live = get_live_gameweek(int(_saved_gw))
                _actual_map = {int(el["id"]): (float(el.get("stats", {}).get("total_points") or 0), float(el.get("stats", {}).get("minutes") or 0))
                               for el in _live.get("elements", []) if el.get("id") is not None}
                for _idx in forecast_log.index[_mask]:
                    _pid = int(forecast_log.at[_idx, "player_id"])
                    if _pid in _actual_map:
                        forecast_log.at[_idx, "actual_points"] = _actual_map[_pid][0]
                        forecast_log.at[_idx, "actual_minutes"] = _actual_map[_pid][1]
                        changed_actuals = True
            except Exception:
                pass
        if changed_actuals:
            write_forecast_log(forecast_log)

    forecast_log = read_forecast_log()
    if not forecast_log.empty:
        from fpl_validation import evaluate_log, score_gameweek_group
        _version_rows = forecast_log[forecast_log["model_version"].astype(str).eq(APP_VERSION)].copy()
        scored = _version_rows[pd.to_numeric(_version_rows["actual_points"], errors="coerce").notna()].copy()
        if not scored.empty:
            _gw_metrics = evaluate_log(scored, actual_col="actual_points", model_col="projection_model",
                                       baseline_col="projection_baseline", benchmark_col="projection_ep_next")
            for _metric_col in _gw_metrics.columns:
                if ("MAE" in _metric_col or "bias" in _metric_col.lower() or "Spearman" in _metric_col
                        or "capture" in _metric_col or "coverage %" in _metric_col or "regret" in _metric_col.lower()):
                    _gw_metrics[_metric_col] = pd.to_numeric(_gw_metrics[_metric_col], errors="coerce").round(3)
            st.markdown("**Current-season validation · one Gameweek**")
            st.caption("Model, baseline and FPL ep_next are compared against actual points for the same pre-deadline snapshot. ep_next metrics use only rows where the benchmark was captured. Captain regret is reported only when a complete public squad was visible before the deadline, so the model XI/captain were logged without hindsight.")
            st.dataframe(_gw_metrics.sort_values(["season", "gameweek"], ascending=[False, False]), use_container_width=True, hide_index=True)
            _usable = _gw_metrics[pd.to_numeric(_gw_metrics.get("Players", 0), errors="coerce") >= 20]
            if not _usable.empty:
                _agg = {
                    "Gameweeks evaluated": int(len(_usable)),
                    "Mean model MAE by GW": round(float(_usable["Model MAE"].mean()), 3),
                    "Mean baseline MAE by GW": round(float(_usable["Baseline MAE"].mean()), 3),
                    "Mean model Spearman": round(float(_usable["Model Spearman"].mean()), 3) if "Model Spearman" in _usable else float("nan"),
                    "Mean top-10 capture": round(float(_usable["Model top-10 capture"].mean()), 3) if "Model top-10 capture" in _usable else float("nan"),
                }
                if "Captain regret (points)" in _usable.columns:
                    _captain_scored = pd.to_numeric(_usable["Captain regret (points)"], errors="coerce").dropna()
                    _agg["Gameweeks with captain-regret data"] = int(len(_captain_scored))
                    _agg["Mean captain regret (points) — lower is better"] = (
                        round(float(_captain_scored.mean()), 3) if not _captain_scored.empty else float("nan")
                    )
                st.write(_agg)
                if len(_usable) < 4:
                    st.info("Early evidence only: fewer than four completed v3.0 Gameweeks have been scored. Do not change model weights yet.")
            _horizon_scored = _version_rows[pd.to_numeric(_version_rows.get("actual_points_5gw"), errors="coerce").notna()].copy()
            if not _horizon_scored.empty:
                _h_metrics = evaluate_log(_horizon_scored, actual_col="actual_points_5gw", model_col="projection_model_5gw",
                                          baseline_col="projection_baseline_5gw", benchmark_col="__no_ep_next__",
                                          include_minutes_metrics=False, include_captain_metrics=False)
                for _metric_col in _h_metrics.columns:
                    if "MAE" in _metric_col or "Spearman" in _metric_col or "capture" in _metric_col:
                        _h_metrics[_metric_col] = pd.to_numeric(_h_metrics[_metric_col], errors="coerce").round(3)
                st.markdown("**Five-Gameweek validation · transfer horizon**")
                st.caption("These results appear only after all Gameweeks in a forecast's horizon have finished. Historical points are summed for each player over the same horizon.")
                st.dataframe(_h_metrics.sort_values(["season", "gameweek"], ascending=[False, False]), use_container_width=True, hide_index=True)
            else:
                st.info("No complete five-Gameweek outcome is available yet. Those metrics will populate automatically when each forecast horizon finishes.")
        else:
            st.info("No v3.0 forecast has completed results yet. The previous-version GW8 seed is kept as legacy reference and is not mixed into the v3.0 validation metrics.")

        _historic = get_remote_historical_backtest()
        st.markdown("**Previous seasons · retrospective backtest**")
        if isinstance(_historic, pd.DataFrame) and not _historic.empty:
            _numeric_cols = [c for c in _historic.columns if c.endswith("_mae_mean_by_gw") or "spearman_mean_by_gw" in c or "capture_mean_by_gw" in c]
            for _col in _numeric_cols:
                _historic[_col] = pd.to_numeric(_historic[_col], errors="coerce").round(3)
            st.caption("Historical proxy replay uses only prior Gameweeks to build each prediction. It is not an exact replay of archived live API snapshots; do not compare it directly with the current-season ep_next benchmark.")
            st.dataframe(_historic, use_container_width=True, hide_index=True)
        else:
            st.info("Historical backtest not run yet. After adding the manual Historical FPL Backtest workflow, run it once from GitHub Actions to test archived 2023–24, 2024–25 and 2025–26 Gameweeks.")

        with st.expander("Forecast history: backup and audit", expanded=False):
            st.caption("Forecasts and actuals are persisted automatically by the GitHub Actions workflow. CSV downloads are optional offline backups.")
            st.download_button("Download forecast/backtest CSV", data=forecast_log.to_csv(index=False).encode("utf-8"), file_name="the_one_above_all_forecast_log.csv", mime="text/csv")
            if not remote_log_active:
                upload = st.file_uploader("Restore or merge a previously downloaded forecast CSV (local fallback only)", type=["csv"], key="forecast_restore_upload")
                if upload is not None and st.button("Merge forecast backup", key="merge_forecast_backup"):
                    try:
                        restored = pd.read_csv(upload)
                        for col in FORECAST_COLUMNS:
                            if col not in restored.columns:
                                restored[col] = None
                        restored = restored[FORECAST_COLUMNS]
                        combined = pd.concat([forecast_log, restored], ignore_index=True)
                        combined = combined.drop_duplicates(subset=["season", "gameweek", "player_id", "model_version"], keep="last")
                        if write_forecast_log(combined):
                            st.success("Local forecast backup merged. The scheduled worker's persistent branch remains authoritative once it is active.")
                            st.rerun()
                        else:
                            st.error("The hosting filesystem did not allow the CSV to be saved.")
                    except Exception as exc:
                        st.error(f"Could not restore that CSV: {exc}")
    else:
        st.info("No forecast log found. Once the GitHub Actions workflow is enabled, it will capture a pre-deadline snapshot and reconcile actual results automatically.")

    st.caption("Validation note: v3.0 weights are frozen during the test. All-player MAE, played-only MAE, expected-minutes error, EP-next benchmark error, Spearman correlation, top-10/top-20 capture, captain regret where a pre-deadline XI was logged, and five-Gameweek error are reported separately. Historical proxy results are kept separate from live pre-deadline validation.")

with decide_tab:
    try:
        _overall_rank_display = f"{int(entry.get('summary_overall_rank')):,}" if entry and entry.get("summary_overall_rank") is not None else "—"
    except (TypeError, ValueError):
        _overall_rank_display = "—"
    _summary_top = st.columns(3, gap="small")
    _summary_top[0].metric("Gameweek", f"GW{int(gw)}")
    _summary_top[1].metric("Overall points", str(entry.get("summary_overall_points", "—") if entry else "—"))
    _summary_top[2].metric("Overall rank", _overall_rank_display)
    _summary_bottom = st.columns(3, gap="small")
    _rank_delta_display = (f"{rank_change_places:+,}" if rank_change_places is not None else "—")
    _summary_bottom[0].metric("Rank change", _rank_delta_display, help=rank_change_label)
    _summary_bottom[1].metric("Free transfers", str(free_transfers), help=f"{_ft_state} · {auto_ft_note if auto_controls else 'Manual override'}")
    _summary_bottom[2].metric("Bank", f"£{bank_m:.1f}m" if bank_m is not None else "Unavailable", help=_bank_state)
    if resolved:
        df = pd.DataFrame(resolved)
        candidates = pd.DataFrame([{
            "Player": p.get("web_name"), "Position": pos_name(p,types),
            "Club": teams.get(p.get("team"),{}).get("short_name","?"),
            "Price (£m)": ((verified_player_price_tenths(p.get("now_cost")) / 10)
                            if verified_player_price_tenths(p.get("now_cost")) is not None else None),
            "Price (tenths)": verified_player_price_tenths(p.get("now_cost")),
            "Form": float(p.get("form") or 0), "PPG": float(p.get("points_per_game") or 0),
            "Minutes": int(p.get("minutes") or 0), "Status": p.get("status","?"),
            "Chance (%)": p.get("chance_of_playing_next_round"),
            f"Projected {horizon}GW pts": round(sum(row["points"] for row in _projection_series_for(p)), 1),
            "Expected minutes this GW": fpl_engine.estimate_expected_minutes(
                p, PLAYER_HISTORY_SIGNALS.get(int(p["id"]), {}),
                fpl_engine.team_fixture_count(fixtures, p.get("team"), int(gw)),
            ),
            "Availability factor": availability_adjustment(p),
             "xG": round(float(p.get("expected_goals") or 0), 2),
                "xA": round(float(p.get("expected_assists") or 0), 2),
                "xG/90": round(float(p.get("expected_goals") or 0) / max(90, int(p.get("minutes") or 0)) * 90, 3),
                "xA/90": round(float(p.get("expected_assists") or 0) / max(90, int(p.get("minutes") or 0)) * 90, 3),
                 "xG": round(float(p.get("expected_goals") or 0), 2),
            "xA": round(float(p.get("expected_assists") or 0), 2),
            "xG/90": round(float(p.get("expected_goals") or 0) / max(90, int(p.get("minutes") or 0)) * 90, 3),
            "xA/90": round(float(p.get("expected_assists") or 0) / max(90, int(p.get("minutes") or 0)) * 90, 3),
            "Selected (%)": float(p.get("selected_by_percent") or 0), "id": p["id"],
        } for p in players])
        # Rank complete legal plans by the same per-GW lineup selector used by
        # the dashboard, so bench upgrades do not masquerade as XI gains.
        _optimization_plans = {}
        if selected_lineup:
            _candidate_rows = candidates.sort_values(f"Projected {horizon}GW pts", ascending=False).head(30)
            _optimization_players = {}
            for _, _row in _candidate_rows.iterrows():
                _p = _players_by_id.get(int(_row["id"]))
                if not _p:
                    continue
                _optimization_players[int(_row["id"])] = {
                    "id": int(_row["id"]), "Player": str(_row["Player"]),
                    "position": str(_row["Position"]), "Position": str(_row["Position"]),
                    "club": str(_row["Club"]),
                    "price_tenths": (int(_row["Price (tenths)"])
                                      if pd.notna(_row["Price (tenths)"]) else None),
                    "projections": [float(r["points"]) for r in _projection_series_for(_p)],
                    "Minutes": int(_row["Minutes"]), "Status": _row["Status"],
                    "Chance (%)": _row["Chance (%)"],
                }
            _optimization_squad = []
            for _r in resolved:
                _p = _players_by_id.get(int(_r["id"]))
                if not _p:
                    continue
                _optimization_squad.append({
                    "id": int(_r["id"]), "Player": str(_r["Player"]),
                    "position": str(_r["Position"]), "Position": str(_r["Position"]),
                    "club": str(_r["Club"]), "price_tenths": int(_p.get("now_cost") or 0),
                    "sell_tenths": int(_r["Sell price (tenths)"]),
                    "projections": [float(r["points"]) for r in _projection_series_for(_p)],
                    "Minutes": int(_r["Minutes"]), "Status": _r["Status"],
                    "Chance (%)": _r["Chance (%)"],
                })

            def _lineup_points_for_roster(_roster, _gw_index):
                _projection_map = {int(p["id"]): float(p["projections"][_gw_index]) for p in _roster}
                return score_starting_xi(_roster, _projection_map)

            if len(_optimization_squad) == 15:
                _optimization_plans = optimize_transfer_plans(
                    _optimization_squad, list(_optimization_players.values()),
                    free_transfers, bank_tenths_current,
                    horizon=int(horizon),
                    start_gw=int(gw), beam_width=120, candidate_limit=30,
                    budget_verified=(bank_tenths_current is not None and all(int(r["id"]) in purchase_price_current_ids for r in resolved)),
                    score_roster=_lineup_points_for_roster,
                )
        # AUTOMATIC TRANSFER SCAN: evaluate every possible player-out / player-in pair.
        squad_names = sorted(df["Player"].dropna().unique().tolist())
        club_counts_auto = df["Club"].value_counts().to_dict()
        hit_cost_auto = transfer_hit_cost(free_transfers, transfer_hit)
        _sale_prices_verified = all(int(row["id"]) in purchase_price_current_ids for row in resolved)
        _budget_state_verified = (
            bank_tenths_current is not None and (not auto_controls or
            (latest_bank_known and auto_ft_confident
             and (squad_source != "prior" or bool(squad_load.get("transfer_history_reconstructed")))))
        )
        transfer_inputs_verified = _sale_prices_verified and _budget_state_verified
        auto_rows = []

        def reliability_for_candidate(cand):
            mins = float(cand.get("Minutes", 0) or 0)
            mins_rel = max(0.15, min(1.0, mins / 600.0))
            avail_rel = availability_adjustment({
                "status": cand.get("Status"),
                "chance_of_playing_next_round": cand.get("Chance (%)"),
            })
            return mins_rel, avail_rel

        for _, outgoing in df.iterrows():
            outgoing_price_tenths = int(outgoing.get("Sell price (tenths)", round(float(outgoing.get("Sell price (£m)", outgoing["Price (£m)"])) * 10)))
            bank_tenths = bank_tenths_current if bank_tenths_current is not None else 0
            budget_ceiling_tenths = transfer_budget_tenths(outgoing_price_tenths, bank_tenths) if bank_tenths_current is not None else -1
            budget_ceiling = budget_ceiling_tenths / 10
            same_position = candidates[candidates["Position"] == outgoing["Position"]]
            possible_buys = same_position[
                (~same_position["Player"].isin(squad_names)) &
                same_position["Price (tenths)"].map(lambda price: is_affordable_tenths(price, budget_ceiling_tenths))
            ]
            for _, incoming in possible_buys.iterrows():
                # If buying from the same club as the outgoing player, the sale frees that club slot.
                club_after = int(club_counts_auto.get(incoming["Club"], 0)) + 1 - int(incoming["Club"] == outgoing["Club"])
                if club_after > 3:
                    continue

                outgoing_proj = float(outgoing.get(f"Proj GW{gw}–GW{gw+horizon-1}", 0.0))
                incoming_proj = float(incoming.get(f"Projected {horizon}GW pts", 0.0))
                raw_gain = incoming_proj - outgoing_proj
                net_gain = raw_gain - hit_cost_auto
                mins_rel, avail_rel = reliability_for_candidate(incoming)
                ownership_pct = float(incoming.get("Selected (%)", 0) or 0)
                differential = max(0.0, min(1.0, 1.0 - min(ownership_pct, 10.0) / 10.0))
                xgi90 = max(0.0, min(0.8, float(incoming.get("xG/90", 0) or 0) + float(incoming.get("xA/90", 0) or 0)))
                attacking = xgi90 / 0.8

                if risk == "Conservative":
                    score = net_gain * mins_rel * avail_rel
                elif risk == "Aggressive":
                    score = (net_gain
                             - 0.15 * (1.0 - mins_rel) * max(0.0, net_gain)
                             - 0.25 * (1.0 - avail_rel) * max(0.0, net_gain)
                             + 1.25 * differential + 0.75 * attacking)
                else:
                    score = (net_gain * (0.45 + 0.55 * mins_rel) * avail_rel
                             + 0.45 * differential + 0.25 * attacking)

                if net_gain <= 0:
                    verdict = "Avoid on projection"
                elif mins_rel < 0.50 or avail_rel < 0.80:
                    verdict = "Potential upgrade — verify minutes/news"
                elif net_gain >= 5:
                    verdict = "Worth investigating"
                else:
                    verdict = "Marginal upgrade"

                outgoing_status = str(outgoing.get("Status", "")).lower()
                outgoing_chance = outgoing.get("Chance (%)")
                outgoing_unavailable = outgoing_status in {"i", "s", "u"} or (pd.notna(outgoing_chance) and float(outgoing_chance) <= 25)
                outgoing_minutes_rel, outgoing_avail_rel = reliability_for_candidate(outgoing)
                outgoing_projection = float(outgoing_proj)
                captaincy_exposure = False
                try:
                    captaincy_exposure = str(outgoing.get("Player", "")) in {str(captain_pick.get("Player", "")), str(vice_pick.get("Player", ""))}
                except Exception:
                    captaincy_exposure = False
                # Action recommendations require a robust net points edge; differential ownership only ranks candidates.
                min_gain_threshold = 5.0
                if outgoing_unavailable:
                    min_gain_threshold = 2.5
                if int(free_transfers) >= 4:
                    min_gain_threshold = max(3.5, min_gain_threshold - 1.5)
                if captaincy_exposure:
                    min_gain_threshold += 2.5
                if free_transfers == 0:
                    min_gain_threshold = max(8.0, min_gain_threshold + 3.0)
                if free_transfers == 0 and not transfer_hit:
                    verdict = "HOLD — hit disabled"
                elif net_gain < min_gain_threshold:
                    verdict = "HOLD — gain below threshold"
                elif mins_rel < 0.70 or avail_rel < 0.80:
                    verdict = "HOLD — reliability too low"
                else:
                    verdict = "Actionable candidate — recheck before deadline"
                auto_rows.append({
                    "Sell": outgoing["Player"], "Sell club": outgoing["Club"],
                    "Buy": incoming["Player"], "Buy club": incoming["Club"],
                    "Sell ID": int(outgoing["id"]), "Buy ID": int(incoming["id"]),
                    "Selling price source": outgoing.get("Selling price source", "Unavailable"),
                    "Position": outgoing["Position"], "Buy price (£m)": float(incoming["Price (£m)"]),
                    "Budget ceiling (£m)": round(budget_ceiling, 1),
                    "Club count after transfer": club_after,
                    "Outgoing projected pts": round(outgoing_proj, 1),
                    "Incoming projected pts": round(incoming_proj, 1),
                    "Expected minutes this GW": float(incoming["Expected minutes this GW"]),
                    "Availability factor": float(incoming["Availability factor"]),
                    "Projected gain": round(net_gain, 1),
                    "Minutes reliability (%)": round(mins_rel * 100),
                    "Availability reliability (%)": round(avail_rel * 100),
                    "Ownership (%)": ownership_pct, "Decision score": round(float(score), 3),
                    "Minimum gain to act": round(float(min_gain_threshold), 1),
                    "Outgoing unavailable": bool(outgoing_unavailable),
                    "Captaincy exposure": bool(captaincy_exposure),
                    "Actionable": bool(net_gain >= min_gain_threshold and mins_rel >= 0.70 and avail_rel >= 0.80
                                       and (free_transfers > 0 or transfer_hit) and not unmatched),
                    "Verdict": verdict,
                })

        hours_to_deadline_now = None
        try:
            if deadline_dt is not None:
                hours_to_deadline_now = (deadline_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0
        except Exception:
            hours_to_deadline_now = None

        decision_label = "VERIFY DATA"
        reason_line = "No recommendation is available until a legal, affordable candidate can be assessed."
        candidate_summary = "Not available"
        gain = None
        threshold = None
        best_each = pd.DataFrame()
        if not selected_lineup or unmatched or not transfer_inputs_verified:
            auto_rows = []
            decision_label = "VERIFY DATA"
            if not selected_lineup:
                reason_line = picks_sync_note
            elif unmatched:
                reason_line = "At least one public squad member could not be matched; transfer legality cannot be verified."
            elif not _sale_prices_verified:
                reason_line = "Current pick data is missing an official purchase price for at least one player, so the selling-price budget cannot be verified."
            elif squad_source == "prior" and auto_controls:
                reason_line = "The prior squad is manager-confirmed, but the bank and free-transfer count may still reflect GW{}; switch to Manual override and confirm both in FPL before using transfer advice.".format(squad_load.get("baseline_gw"))
            elif bank_tenths_current is None:
                reason_line = "The bank is missing or is not an exact £0.1m amount, so affordability cannot be verified. Confirm the FPL bank or enter a valid manual override."
            else:
                reason_line = "Auto-synced free-transfer history is uncertain. Confirm the value in FPL or use manual override before relying on a transfer shortlist."
        if auto_rows:
            auto_df = pd.DataFrame(auto_rows)
            # Show the highest-scoring legal swap for each player we could sell.
            best_each = (auto_df.sort_values(["Actionable", "Decision score", "Projected gain"], ascending=False)
                         .drop_duplicates(subset=["Sell"], keep="first")
                         .sort_values(["Actionable", "Decision score", "Projected gain"], ascending=False)
                         .head(10))
            top = best_each.iloc[0]
            gain = float(top["Projected gain"])
            threshold = float(top["Minimum gain to act"])
            reliable = float(top["Minutes reliability (%)"]) >= 70 and float(top["Availability reliability (%)"]) >= 80
            candidate_is_robust = bool(top.get("Actionable", False)) and gain >= threshold and reliable and not unmatched
            candidate_summary = f"{top['Sell']} → {top['Buy']} · {top['Buy club']}"
            if unmatched:
                decision_label = "VERIFY DATA"
                reason_line = "At least one squad member is unmatched, so the legal squad and club limits cannot be safely assessed."
            elif free_transfers == 0 and not transfer_hit:
                decision_label = "HOLD"
                reason_line = f"No transfer is justified without a points hit. The -4 hit is disabled; best candidate: {candidate_summary}."
            elif candidate_is_robust and (free_transfers > 0 or (transfer_hit and gain >= threshold)) and hours_to_deadline_now is not None and hours_to_deadline_now > 24 and not bool(top.get("Outgoing unavailable", False)):
                decision_label = "HOLD FOR NOW"
                reason_line = "It clears the points/reliability gate, but the deadline is still more than 24 hours away. Recheck inside 24 hours and after team news."
            elif candidate_is_robust and (free_transfers > 0 or (transfer_hit and gain >= threshold)) and hours_to_deadline_now is not None and hours_to_deadline_now > 0:
                decision_label = "TRANSFER"
                reason_line = f"Clears the {threshold:.1f}-point action threshold and candidate reliability checks. Verify team news and official selling price/bank."
            else:
                reason = []
                if gain < threshold: reason.append(f"projected edge {gain:+.1f} is below the {threshold:.1f}-point action threshold")
                if not reliable: reason.append("incoming minutes/availability are not reliable enough")
                if free_transfers == 0 and transfer_hit and gain < threshold: reason.append("the gain does not repay the hit with enough margin")
                decision_label = "HOLD"
                reason_line = "; ".join(reason) if reason else "The model cannot establish a sufficiently robust advantage."
        else:
            reason_line = "No legal, affordable one-for-one transfers were found. Check squad matching, budget, and club-count warnings."

        _preferred_plan = None
        _all_optimization_plans = list(_optimization_plans.values())
        _projection_only_plans = []
        for _plan in _all_optimization_plans:
            _projection_only_plans.extend(_plan.get("projection_only_alternatives", []))
            if _plan.get("projection_only") and _plan.get("transfer_count", 0) > 0:
                _projection_only_plans.append(_plan)
        _verified_plans = [plan for plan in _all_optimization_plans if plan.get("affordable") is True]
        _best_verified_plan = max(_verified_plans, key=lambda plan: plan["cumulative_net_gain"]) if _verified_plans else None
        _best_projection_only = (max(_projection_only_plans, key=lambda plan: plan["cumulative_net_gain"])
                                 if _projection_only_plans else None)
        _projection_threshold = (transfer_plan_action_threshold(_best_projection_only["paid_transfers"])
                                 if _best_projection_only else float("inf"))
        _projection_only_outscores = bool(
            _best_projection_only and _best_verified_plan
            and _best_projection_only["cumulative_net_gain"] > _best_verified_plan["cumulative_net_gain"]
            and _best_projection_only["cumulative_net_gain"] >= _projection_threshold
        )
        if _optimization_plans and transfer_inputs_verified and not unmatched and _projection_only_outscores:
            _preferred_plan = None
            decision_label = "VERIFY DATA"
            candidate_summary = " · ".join(f"{m['out']['Player']} → {m['in']['Player']}"
                                           for m in _best_projection_only["transfers"])
            gain = float(_best_projection_only["cumulative_net_gain"])
            threshold = transfer_plan_action_threshold(_best_projection_only["paid_transfers"])
            reason_line = _best_projection_only.get("budget_reason") or "The higher-projected plan has unverified affordability; confirm its prices before acting."
        elif _optimization_plans and transfer_inputs_verified and not unmatched:
            _preferred_plan = _best_verified_plan
            if _preferred_plan is None:
                decision_label = "VERIFY DATA"
                reason_line = "Only projection-only transfer plans are available; confirm bank and manager-specific prices before acting."
            else:
                _preferred_gain = float(_preferred_plan["cumulative_net_gain"])
                _preferred_threshold = transfer_plan_action_threshold(_preferred_plan["paid_transfers"])
                if _preferred_plan["transfer_count"] == 0 or _preferred_gain < _preferred_threshold:
                    decision_label = "HOLD / ROLL"
                    candidate_summary = "Retain free transfers"
                    gain = _preferred_gain
                    threshold = _preferred_threshold
                    reason_line = "The best verified plan does not clear the engine's projected advantage threshold over hold."
                elif hours_to_deadline_now is not None and hours_to_deadline_now > 24:
                    decision_label = "HOLD FOR NOW"
                    candidate_summary = " · ".join(f"{m['out']['Player']} → {m['in']['Player']}" for m in _preferred_plan["transfers"])
                    gain = _preferred_gain
                    threshold = _preferred_threshold
                    reason_line = "The plan clears the projected advantage threshold, but the deadline is more than 24 hours away. Recheck closer to deadline and after team news."
                elif _preferred_plan["paid_transfers"]:
                    decision_label = "TRANSFER WITH -4 · ACCEPT HIT"
                    candidate_summary = " · ".join(f"{m['out']['Player']} → {m['in']['Player']}" for m in _preferred_plan["transfers"])
                    gain = _preferred_gain
                    threshold = _preferred_threshold
                    reason_line = f"Best verified plan uses {_preferred_plan['free_transfers_used']} free transfer(s) and {_preferred_plan['paid_transfers']} paid transfer(s). Accept the {_preferred_plan['hit_cost']}-point hit before acting."
                else:
                    decision_label = "TRANSFER"
                    candidate_summary = " · ".join(f"{m['out']['Player']} → {m['in']['Player']}" for m in _preferred_plan["transfers"])
                    gain = _preferred_gain
                    threshold = _preferred_threshold
                    reason_line = f"Best verified plan uses {_preferred_plan['free_transfers_used']} free transfer(s) and improves projected starting-XI points over hold."
        elif _optimization_plans and not transfer_inputs_verified:
            decision_label = "VERIFY DATA"
            reason_line = "Projected plans are shown below, but current bank, free-transfer count, or selling-price inputs are uncertain; affordability is unverified."

        _provisional = (_model_status == "Fallback" or bool(unmatched) or squad_source == "prior"
                        or (auto_controls and (not latest_bank_known or not auto_ft_confident))
                        or not transfer_inputs_verified)
        _provisional_badge = '<span class="provisional-badge">Provisional</span>' if _provisional else ""
        _candidate_safe = html.escape(candidate_summary)
        _gain_detail = f"Projected gain {gain:+.1f} points · needed {threshold:.1f}" if gain is not None else "Projected gain · not available"
        st.markdown(
            "<div class='verdict-card'><div class='verdict-topline'>"
            f"<span class='verdict-label'>Recommended action</span><span class='verdict-value'>{html.escape(decision_label)}</span>"
            f"{_provisional_badge}</div>"
            f"<div class='verdict-candidate'>{_candidate_safe}</div>"
            f"<div class='verdict-gain'>{html.escape(_gain_detail)}</div>"
            f"<div class='verdict-reason'>{html.escape(reason_line)}</div></div>", unsafe_allow_html=True,
        )

        _captain_name = str(captain_pick.get("Player", "Not available")) if captain_pick is not None else "Not available"
        _vice_name = str(vice_pick.get("Player", "Not available")) if vice_pick is not None else "Not available"
        _captain_pts = captain_pick.get("GW projected pts") if captain_pick is not None else None
        _vice_pts = vice_pick.get("GW projected pts") if vice_pick is not None else None
        _captain_detail = f"{float(_captain_pts):.1f} projected points" if _captain_pts is not None else "Not available"
        _vice_detail = f"{float(_vice_pts):.1f} projected points" if _vice_pts is not None else "Not available"
        _chip_choices = [card for card in chip_cards if card[1] == "Available"]
        _chip_choice = next((card for card in _chip_choices if card[3] in {"Consider this GW", "Potential use this GW"}), _chip_choices[0] if _chip_choices else None)
        _chip_name, _chip_action, _chip_detail = ("Not available", "Not available", "No chip recommendation is available.") if _chip_choice is None else (_chip_choice[0], _chip_choice[3], _chip_choice[4])
        st.markdown(
            "<div class='decision-card-grid'>"
            f"<div class='decision-summary-card'><div class='card-label'>Transfer</div><div class='card-value'>{_candidate_safe}</div><div class='card-detail'>{html.escape(_gain_detail)}</div></div>"
            f"<div class='decision-summary-card'><div class='card-label'>Captain and vice-captain</div><div class='card-value'>C: {html.escape(_captain_name)} · VC: {html.escape(_vice_name)}</div><div class='card-detail'>{_captain_detail} · {_vice_detail}</div></div>"
            f"<div class='decision-summary-card'><div class='card-label'>Chip · {html.escape(_chip_name)}</div><div class='card-value'>{html.escape(_chip_action)}</div><div class='card-detail'>{html.escape(str(_chip_detail))}</div></div>"
            "</div>", unsafe_allow_html=True,
        )
        st.markdown("**Before transferring**")
        _check_left, _check_right = st.columns(2, gap="small")
        with _check_left:
            st.checkbox("Squad is correct", key="transfer_check_squad")
            st.checkbox("Official selling price checked", key="transfer_check_price")
        with _check_right:
            st.checkbox("Player availability checked", key="transfer_check_availability")
            st.checkbox("Team news checked", key="transfer_check_team_news")

        if not best_each.empty:
            st.markdown("#### Suggested transfers")
            st.caption(f"Sorted by existing {risk.lower()} decision score. Every point value is an estimate; outgoing sale prices use current FPL purchase prices and the official half-profit rule.")
            _default_transfer_table = pd.DataFrame({
                "Sell": best_each["Sell"], "Buy": best_each["Buy"],
                "Gain": best_each["Projected gain"].map(lambda value: f"{float(value):+.1f}"),
                "Needed": best_each["Minimum gain to act"].map(lambda value: f"{float(value):.1f}"),
                "Reliability": best_each.apply(lambda row: f"{int(row['Minutes reliability (%)'])}% min · {int(row['Availability reliability (%)'])}% avail", axis=1),
            })
            st.dataframe(_default_transfer_table, use_container_width=True, hide_index=True)
            with st.expander("Transfer vs hold · Gameweek by Gameweek", expanded=False):
                _transfer_labels = [f"{row['Sell']} → {row['Buy']}" for _, row in best_each.iterrows()]
                _selected_transfer_index = st.selectbox(
                    "Transfer to inspect", range(len(_transfer_labels)),
                    format_func=lambda index: _transfer_labels[index], key="transfer_projection_detail",
                )
                _detail = best_each.iloc[int(_selected_transfer_index)]
                _sell_player = _players_by_id.get(int(_detail["Sell ID"]))
                _buy_player = _players_by_id.get(int(_detail["Buy ID"]))
                _sell_series = _projection_series_for(_sell_player) if _sell_player else []
                _buy_series = _projection_series_for(_buy_player) if _buy_player else []
                _comparison_rows = transfer_projection_comparison(_sell_series, _buy_series, hit_cost_auto)
                _snapshot_coverage = sum(pid in SNAPSHOT_GW1 for pid in (int(_detail["Sell ID"]), int(_detail["Buy ID"])))
                _transfer_projection_source = (
                    "saved pre-deadline v3.0 snapshot for both players"
                    if _snapshot_coverage == 2 else
                    "mixed saved snapshot/live prediction engine"
                    if _snapshot_coverage == 1 else
                    "live v3.0 prediction engine"
                )
                if _comparison_rows:
                    _comparison_frame = pd.DataFrame(_comparison_rows)
                    _comparison_frame["Raw difference"] = _comparison_frame["Raw difference"].map(lambda value: f"{value:+.1f}")
                    _comparison_frame["Net difference"] = _comparison_frame["Net difference"].map(lambda value: f"{value:+.1f}")
                    _comparison_frame["Cumulative net"] = _comparison_frame["Cumulative net"].map(lambda value: f"{value:+.1f}")
                    _comparison_frame["Transfer hit"] = _comparison_frame["Transfer hit"].map(lambda value: f"-{value:.0f}" if value else "—")
                    st.dataframe(_comparison_frame, use_container_width=True, hide_index=True)
                    st.caption(f"Hold = keep {html.escape(str(_detail['Sell']))}. Candidate = buy {html.escape(str(_detail['Buy']))}. Projection source: {_transfer_projection_source}. Current FPL price budget: sale £{float(_detail['Budget ceiling (£m)']) - float(bank_m):.1f}m + bank £{float(bank_m):.1f}m; hit cost {hit_cost_auto} points, charged once. All values are model estimates, not guaranteed returns.")
            with st.expander("More columns", expanded=False):
                st.caption(f"Risk mode: {risk} · Horizon: {horizon} GWs · Transfer hit applied: {hit_cost_auto} points.")
                st.dataframe(best_each[["Sell","Sell club","Selling price source","Buy","Buy club","Position","Buy price (£m)","Budget ceiling (£m)","Club count after transfer","Outgoing projected pts","Incoming projected pts","Projected gain","Minimum gain to act","Expected minutes this GW","Availability factor","Minutes reliability (%)","Availability reliability (%)","Captaincy exposure","Actionable","Ownership (%)","Decision score","Verdict"]], use_container_width=True, hide_index=True)
                st.caption("PR #53 expected minutes scale the fallback projection; the trained v3.0 model does not use that estimate as a prediction feature. Current availability adjusts the nearest Gameweek only. The separate reliability score uses season minutes and availability flags; today's injury status is not assumed to persist across later weeks.")

        if _optimization_plans:
            st.markdown("#### Multi-transfer plans")
            st.caption("Bounded beam search: at most 30 incoming candidates and 120 states per transfer count; plans use the expected XI plus captain bonus each Gameweek. The preferred-plan threshold is +4 cumulative points over the selected horizon for free-transfer plans, or +6 after paid-transfer hits. Up to two paid transfers are compared and require hit acceptance. The search is approximate and does not prove a global optimum.")
            _plan_rows = []
            for _count, _plan in sorted(_optimization_plans.items()):
                _table_plans = [_plan, *_plan.get("projection_only_alternatives", [])]
                for _shown_plan in _table_plans:
                    _move_text = " · ".join(f"{m['out']['Player']} → {m['in']['Player']}" for m in _shown_plan["transfers"]) or "Hold / roll"
                    _risk_text = " · ".join(
                        f"{m['in']['Player']}: {int(m['in'].get('Minutes', 0))} min · {m['in'].get('Status', '?')} · {m['in'].get('Chance (%)') if pd.notna(m['in'].get('Chance (%)')) else 'chance n/a'}%"
                        for m in _shown_plan["transfers"]
                    ) or "No incoming-player risk"
                    _plan_rows.append({
                        "Transfers": _move_text,
                        "Free used": _shown_plan["free_transfers_used"],
                        "Paid": _shown_plan["paid_transfers"],
                        "Hit": f"-{_shown_plan['hit_cost']} pts" if _shown_plan["hit_cost"] else "None",
                        "Budget before (£m)": (f"£{_shown_plan['budget_before_tenths'] / 10:.1f}m" if _shown_plan["budget_before_tenths"] is not None else "Unverified"),
                        "Budget after (£m)": (f"£{_shown_plan['budget_after_tenths'] / 10:.1f}m" if _shown_plan["budget_after_tenths"] is not None else "Unverified"),
                        "Affordability": ("Not needed for hold" if _shown_plan["transfer_count"] == 0
                                           else "Verified" if _shown_plan["affordable"] is True else "Unverified"),
                        "Plan status": ("Hold / roll baseline" if _shown_plan["transfer_count"] == 0
                                        else "Projection only — verify budget" if _shown_plan.get("projection_only")
                                        else "Verified budget"),
                        "Minutes / availability": _risk_text,
                        "Net gain vs hold": f"{_shown_plan['cumulative_net_gain']:+.1f}",
                        "By Gameweek": " · ".join(f"GW{r['gameweek']} {r['plan_points']:.1f} vs {r['hold_points']:.1f}" for r in _shown_plan["gameweeks"]),
                    })
            st.dataframe(pd.DataFrame(_plan_rows), use_container_width=True, hide_index=True)
            if not transfer_inputs_verified:
                st.warning("These projections do not establish affordability because a required budget or transfer input is unverified.")

        with st.expander("Manual transfer comparison (optional)", expanded=False):

            # Current squad selector
            squad_names = sorted(df["Player"].unique().tolist())
            sell_name = st.selectbox("Sell player", squad_names, index=0)
            sell = df[df["Player"]==sell_name].iloc[0]
            sale_price_tenths = int(sell.get("Sell price (tenths)", round(float(sell.get("Sell price (£m)", sell["Price (£m)"])) * 10)))
            bank_tenths = bank_tenths_current if bank_tenths_current is not None else 0
            max_price_tenths = transfer_budget_tenths(sale_price_tenths, bank_tenths)
            max_price = max_price_tenths / 10
            options = candidates[(candidates["Position"]==sell["Position"]) &
                                 (~candidates["Player"].isin(squad_names)) &
                                 candidates["Price (tenths)"].map(lambda price: is_affordable_tenths(price, max_price_tenths))].copy()
            _manual_budget_verified = (bank_tenths_current is not None and (not auto_controls or
                                       (squad_source != "prior" and latest_bank_known and auto_ft_confident)))
            if (bank_tenths_current is None or int(sell["id"]) not in purchase_price_current_ids
                    or not _manual_budget_verified):
                options = options.iloc[0:0]
                if bank_tenths_current is None:
                    st.info("Current bank is unavailable. This comparison is withheld; switch to manual override and enter the FPL bank.")
                if int(sell["id"]) not in purchase_price_current_ids:
                    st.info("The selected player's official current purchase price is unavailable, so the selling-price budget cannot be verified.")
                if not _manual_budget_verified:
                    st.info("This prior-squad fallback does not verify current bank or free transfers. Switch to Manual override and confirm both before comparing transfers.")

            # Enforce the official maximum of three players from any one club.
            # If selling from the candidate's club, the outgoing player frees a slot there.
            club_counts = df["Club"].value_counts().to_dict()
            options["Current club count"] = options["Club"].map(club_counts).fillna(0).astype(int)
            same_club_as_sold = options["Club"].eq(sell["Club"])
            options["Club count after transfer"] = (
                options["Current club count"] + 1 - same_club_as_sold.astype(int)
            )
            options = options[options["Club count after transfer"] <= 3].copy()

            st.caption("Three-per-club rule is enforced in the shortlist. Current matched squad counts: " +
                       ", ".join(f"{club}: {count}" for club, count in sorted(club_counts.items())))
            if unmatched:
                st.warning("Some recorded squad names did not match public data; club counts may be incomplete until those aliases are corrected.")

            options["Raw gain"] = options[f"Projected {horizon}GW pts"] - float(sell[f"Proj GW{gw}–GW{gw+horizon-1}"])
            options["Hit cost"] = transfer_hit_cost(free_transfers, transfer_hit)
            options["Projected gain"] = options["Raw gain"] - options["Hit cost"]
            # Risk setting affects the ranking, not the raw point projection.
            ownership = options["Selected (%)"].fillna(0).astype(float).clip(0, 100)
            # Reliability is deliberately stronger than in v1.3: fewer than 300 minutes
            # should materially reduce confidence in a transfer recommendation.
            minutes_reliability = (options["Minutes"].astype(float) / 600.0).clip(0.15, 1.0)
            availability_reliability = options.apply(
                lambda row: availability_adjustment({
                    "status": row.get("Status"),
                    "chance_of_playing_next_round": row.get("Chance (%)"),
                }),
                axis=1,
            )

            # Differential upside is capped: low ownership alone must never dominate the model.
            differential_bonus = (1.0 - ownership.clip(0, 10) / 10.0).clip(0, 1)
            xgi90 = (options["xG/90"].fillna(0).astype(float) +
                     options["xA/90"].fillna(0).astype(float)).clip(0, 0.8)
            attacking_bonus = xgi90 / 0.8  # 0..1, based on current public xG/xA fields

            if risk == "Conservative":
                # Strongly discounts low-minute/uncertain players; likely to prefer established minutes.
                options["Decision score"] = options["Projected gain"] * minutes_reliability * availability_reliability
            elif risk == "Aggressive":
                # Mostly preserves upside while giving an explicit bonus to low-owned, attacking differentials.
                options["Decision score"] = (
                    options["Projected gain"]
                    - 0.15 * (1.0 - minutes_reliability) * options["Projected gain"].clip(lower=0)
                    - 0.25 * (1.0 - availability_reliability) * options["Projected gain"].clip(lower=0)
                    + 1.25 * differential_bonus
                    + 0.75 * attacking_bonus
                )
            else:
                # Moderately discounts uncertainty without discarding projected points.
                options["Decision score"] = (
                    options["Projected gain"] * (0.45 + 0.55 * minutes_reliability)
                    * availability_reliability
                    + 0.45 * differential_bonus
                    + 0.25 * attacking_bonus
                )
            options = options.sort_values(["Decision score","Projected gain","Minutes"], ascending=False).head(15)
            if options.empty:
                st.info("No affordable out-of-squad player candidates matched this position under the rough budget estimate.")
            else:
                st.caption(f"Replacement ceiling: £{max_price:.1f}m, using the official half-profit selling price + available bank.")
                st.caption(f"Ranking mode: {risk}. Horizon: {horizon} GWs. Hit penalty: {4 if (transfer_hit and free_transfers == 0) else 0} points. Decision score now explicitly weights minutes reliability, availability, differential ownership and xG+xA/90.")
                options["Minutes reliability (%)"] = (minutes_reliability * 100).round(0).astype(int)
                options["Availability reliability (%)"] = (availability_reliability * 100).round(0).astype(int)
                st.dataframe(options[["Player","Club","Price (£m)","Current club count","Club count after transfer","Form","PPG","Minutes","Minutes reliability (%)","Status","Chance (%)","Availability reliability (%)","xG","xA","xG/90","xA/90",f"Projected {horizon}GW pts","Selected (%)","Projected gain","Decision score"]], use_container_width=True, hide_index=True)
                best=options.iloc[0]
                if free_transfers == 0 and not transfer_hit:
                    st.info(f"No transfer is actionable with the -4 hit disabled. The highest-ranked comparison is {best['Player']} ({best['Club']}) at {best['Projected gain']:+.1f} estimated points over {horizon} GWs before any hit.")
                else:
                    st.success(f"Top candidate under {risk.lower()} settings: {best['Player']} ({best['Club']}) — projected gain {best['Projected gain']:+.1f} points over {horizon} GWs after the configured hit. Decision score ranks estimates; it does not guarantee returns or rank gains.")

    else:
        st.markdown(
            "<div class='verdict-card'><div class='verdict-topline'>"
            "<span class='verdict-label'>Recommended action</span><span class='verdict-value'>VERIFY DATA</span></div>"
            f"<div class='verdict-reason'>{html.escape(picks_sync_note)}</div></div>",
            unsafe_allow_html=True,
        )
        st.info("No transfer, captain or chip recommendation is generated without a verified current 15-player FPL squad.")
