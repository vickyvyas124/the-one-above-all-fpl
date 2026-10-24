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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

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

# Recorded squad from the user's GW8 screenshot. IDs are resolved by player name
# against the current public bootstrap dataset, so names can be reviewed/edited.
RECORDED_SQUAD = [
    ("David Raya", "GKP", "Raya"),
    ("Tzolakis", "GKP", "Tzolakis"),
    ("Egan", "DEF", "Egan"),
    ("Lewis Hall", "DEF", "Hall"),
    ("Riccardo Calafiori", "DEF", "Calafiori"),
    ("Maxim De Cuyper", "DEF", "De Cuyper"),
    ("Davis", "DEF", "Davis"),
    ("Cole Palmer", "MID", "Palmer"),
    ("Marcus Tavernier", "MID", "Tavernier"),
    ("Pascal Groß", "MID", "Groß"),
    ("Morgan Rogers", "MID", "Rogers"),
    ("Kevin Schade", "MID", "Schade"),
    ("Charalampos Kostoulas", "FWD", "Kostoulas"),
    ("João Pedro", "FWD", "João Pedro"),
    ("Erling Haaland", "FWD", "Haaland"),
]

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
        if (str(payload.get("model_version")) != fpl_learning.MODEL_VERSION
                or payload.get("feature_policy") != fpl_learning.FEATURE_POLICY
                or not payload.get("coefficients")):
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

def make_history_signal(summary):
    return fpl_engine.make_history_signal(summary)

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
        except Exception:
            frame = pd.DataFrame(columns=FORECAST_COLUMNS)
    else:
        frame = pd.DataFrame(columns=FORECAST_COLUMNS)
    for col in FORECAST_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    if "model_version" in frame.columns:
        versions = frame["model_version"].astype("string").str.strip()
        frame.loc[versions.isna() | versions.eq(""), "model_version"] = "legacy-unknown"
    return frame[FORECAST_COLUMNS]

def write_forecast_log(frame):
    """Write a portable CSV. Streamlit-host disk can be ephemeral, so export backups are provided."""
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
    row_version = str(rows[0].get("model_version", APP_VERSION))
    if not old.empty and ((old["season"].astype(str) == str(rows[0]["season"])) &
                          (pd.to_numeric(old["gameweek"], errors="coerce") == int(rows[0]["gameweek"])) &
                          (old["model_version"].astype(str) == row_version)).any():
        return old, False, "A forecast snapshot for this Gameweek and model version already exists; it was not overwritten."
    new = pd.DataFrame(rows, columns=FORECAST_COLUMNS)
    combined = pd.concat([old, new], ignore_index=True) if not old.empty else new
    ok = write_forecast_log(combined)
    return combined, ok, ("Pre-deadline forecast saved." if ok else "Could not write the forecast CSV on this host; use the export/import controls for a backup workflow.")

def fetch_history_signals(target_ids):
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
                results[pid] = make_history_signal(future.result())
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
        return payload if isinstance(payload, list) else []
    except Exception:
        return []


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
:root {--pl-purple:#37003c;--pl-lime:#00ff87;--pl-cyan:#04f5ff;--ink:#202238;--muted:#6b7280;}
.block-container {padding-top:1rem;padding-bottom:2.5rem;max-width:1540px;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#19051d 0%,#37003c 60%,#24142b 100%);}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color:#f6f3fa !important;}
[data-testid="stSidebar"] input,[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
 color:#202238 !important;-webkit-text-fill-color:#202238 !important;background:#fff !important;opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button,
[data-testid="stSidebar"] [data-baseweb="select"] svg {color:#202238 !important;fill:#202238 !important;}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {color:#f6f3fa !important;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] p {color:#f6f3fa !important;}
.hero {padding:1.4rem 1.6rem;border-radius:20px;background:linear-gradient(115deg,#210026 0%,#37003c 48%,#563b83 100%);color:#fff;margin-bottom:1.1rem;box-shadow:0 12px 32px rgba(55,0,60,.16);position:relative;overflow:hidden;}
.hero:after {content:"";position:absolute;width:260px;height:260px;border-radius:50%;right:-65px;top:-130px;background:rgba(0,255,135,.12);}
.hero-top {display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap;}
.hero h1 {color:#fff;margin:0;font-size:clamp(1.55rem,3vw,2.3rem);font-weight:850;letter-spacing:-.04em;}
.hero p {color:#e8dced;margin:.4rem 0 0 0;font-size:.95rem;}
.gw-chip {display:inline-block;background:#00ff87;color:#210026;padding:.42rem .8rem;border-radius:999px;font-weight:800;font-size:.86rem;}
.kicker {font-size:.75rem;text-transform:uppercase;letter-spacing:.11em;color:#d7c9df;font-weight:700;margin-bottom:.2rem;}
div[data-testid="stMetric"] {padding:.85rem 1rem;border:1px solid #ebe7ef;border-radius:15px;background:linear-gradient(180deg,#fff,#fbf9fc);box-shadow:0 4px 14px rgba(33,0,38,.05);}
div[data-testid="stMetricLabel"] {color:#686173;font-weight:650;}
div[data-testid="stMetricValue"] {color:#37003c;font-weight:850;}
div[data-testid="stDataFrame"] {border:1px solid #e9e4ee;border-radius:13px;overflow:hidden;}
h2,h3 {letter-spacing:-.025em;}
.pitch-wrap {background:#0e7b45;border-radius:20px;padding:16px 12px 18px;position:relative;overflow:hidden;border:4px solid #fff;box-shadow:0 6px 20px rgba(12,90,52,.12);}
.pitch-wrap:before {content:"";position:absolute;inset:0;background:repeating-linear-gradient(0deg,rgba(255,255,255,.035) 0,rgba(255,255,255,.035) 48px,rgba(0,0,0,.025) 48px,rgba(0,0,0,.025) 96px);pointer-events:none;}
.pitch-lines {position:absolute;inset:10px;border:1.5px solid rgba(255,255,255,.28);border-radius:10px;pointer-events:none;}
.pitch-lines:before {content:"";position:absolute;left:0;right:0;top:50%;border-top:1.5px solid rgba(255,255,255,.27);}
.pitch-lines:after {content:"";position:absolute;width:90px;height:90px;left:50%;top:50%;transform:translate(-50%,-50%);border:1.5px solid rgba(255,255,255,.25);border-radius:50%;}
.pitch-title {position:relative;z-index:1;text-align:center;color:#d9ffe9;font-weight:800;text-transform:uppercase;letter-spacing:.14em;font-size:.72rem;margin:2px 0 13px;}
.pitch-row {position:relative;z-index:1;display:flex;justify-content:center;align-items:stretch;gap:clamp(5px,1.5vw,18px);margin:15px auto;max-width:920px;}
.player-tile {width:clamp(74px,10vw,128px);min-width:0;border-radius:12px;overflow:hidden;text-align:center;background:#f8fafc;box-shadow:0 5px 10px rgba(0,0,0,.18);border:1px solid rgba(255,255,255,.8);}
.player-tile .tile-top {background:linear-gradient(135deg,#37003c,#594277);color:#fff;padding:7px 3px 5px;font-weight:800;font-size:clamp(.68rem,1vw,.86rem);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.player-tile .tile-avatar {height:42px;display:flex;align-items:center;justify-content:center;background:linear-gradient(180deg,#f4eefa,#fff);color:#37003c;font-size:1.35rem;font-weight:900;}
.player-tile .tile-meta {padding:5px 3px;color:#333044;font-size:clamp(.62rem,.8vw,.75rem);line-height:1.3;}
.player-tile .tile-pts {font-weight:900;color:#37003c;font-size:.95rem;}
.pitch-caption {position:relative;z-index:1;text-align:center;color:#e4ffef;font-size:.78rem;margin-top:10px;}
.bench-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;}
.bench-card {background:#f5f2f8;border:1px solid #e6deeb;border-radius:12px;padding:10px;text-align:center;}
.bench-name {font-weight:800;color:#37003c;font-size:.84rem;}
.bench-sub {color:#6b6473;font-size:.76rem;margin-top:3px;}
.section-intro {color:#777083;font-size:.92rem;margin-top:-.4rem;margin-bottom:.7rem;}
@media (max-width:650px) {
 .block-container {padding-left:.65rem;padding-right:.65rem;}
 .hero {padding:1.05rem;border-radius:15px;}
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
.manager-panel {background:#fff;border:1px solid #eee7f0;border-radius:16px;padding:15px 15px 13px;box-shadow:0 4px 18px rgba(55,0,60,.045);margin-bottom:12px;}
.manager-name {font-size:1.17rem;font-weight:850;color:#37003c;line-height:1.25;margin:0 0 4px 0;}
.manager-label {font-size:.78rem;color:#75647b;margin:0 0 13px 0;}
.panel-heading {font-size:1rem;font-weight:800;color:#37003c;border-bottom:1px solid #eee7f0;padding-bottom:9px;margin-bottom:10px;}
.rank-row {display:flex;justify-content:space-between;gap:10px;align-items:center;padding:6px 0;border-bottom:1px solid #f4f0f5;color:#4b4050;font-size:.86rem;}
.rank-row:last-child {border-bottom:0;}
.rank-row strong {color:#37003c;font-variant-numeric:tabular-nums;}
.league-pill {display:flex;justify-content:space-between;gap:8px;background:#faf7fb;border-radius:9px;padding:8px 10px;margin:6px 0;color:#37003c;font-size:.82rem;font-weight:650;}
.gw-panel {border:1px solid #eee7f0;background:#fff;border-radius:18px;padding:16px;box-shadow:0 4px 18px rgba(55,0,60,.045);}
.gw-heading {display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:12px;}
.gw-title {font-size:1.35rem;font-weight:850;color:#37003c;margin:0;}
.gw-sub {font-size:.78rem;color:#74687b;margin-top:2px;}
.gw-scoreband {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:7px;margin:12px 0 16px;}
.gw-scorebox {background:#f8f4fa;border-radius:11px;padding:10px 7px;text-align:center;min-width:0;}
.gw-scorebox.featured {background:linear-gradient(135deg,#04d8ef,#5f6cff);color:#fff;}
.gw-scorebox .num {font-size:clamp(1.1rem,2vw,1.8rem);font-weight:900;color:#37003c;line-height:1.15;}
.gw-scorebox.featured .num {color:#fff;font-size:clamp(1.55rem,2.6vw,2.25rem);}
.gw-scorebox .lbl {font-size:.7rem;color:#706179;margin-top:4px;line-height:1.25;}
.gw-scorebox.featured .lbl {color:#fff;}
.player-tile {width:clamp(70px,8vw,112px);min-width:0;border-radius:8px;overflow:hidden;text-align:center;background:#fff;box-shadow:0 3px 8px rgba(0,0,0,.15);border:1px solid rgba(255,255,255,.85);}
.player-tile .tile-avatar {height:60px;display:flex;align-items:center;justify-content:center;background:linear-gradient(180deg,#f4eefa,#fff);color:#37003c;}
.kit-icon {width:54px;height:54px;filter:drop-shadow(0 2px 1px rgba(0,0,0,.12));}
.player-tile .tile-top {font-size:clamp(.63rem,.85vw,.8rem);padding:6px 2px 5px;}
.player-tile .tile-meta {padding:4px 2px 6px;font-size:clamp(.59rem,.75vw,.7rem);line-height:1.3;}
.player-tile .tile-pts {font-size:.88rem;}
.pitch-wrap {padding:12px 8px 13px;}
.pitch-title {margin:5px 0 10px;}
.pitch-row {gap:clamp(3px,1vw,12px);margin:12px auto;}
.manager-overview-note {background:#faf7fb;border-left:3px solid #00d978;padding:9px 10px;border-radius:8px;color:#625369;font-size:.78rem;line-height:1.45;margin-top:10px;}
@media(max-width:900px) {
 .gw-scoreband {grid-template-columns:repeat(2,minmax(0,1fr));}
 .player-tile {width:23%;max-width:98px;}
}
@media(max-width:650px) {
 .manager-panel {padding:12px;}
 .gw-panel {padding:10px;}
 .gw-title {font-size:1.15rem;}
 .gw-scorebox {padding:8px 4px;}
 .player-tile {width:23%;max-width:76px;}
 .player-tile .tile-avatar {height:42px;}
 .kit-icon {width:40px;height:40px;}
 .pitch-row {gap:3px;margin:9px auto;}
 .pitch-title {font-size:.61rem;letter-spacing:.08em;margin:4px 0 8px;}
}
[data-testid="stSidebar"] [data-testid="stButton"] button {
 background:#00ff87 !important;color:#210026 !important;-webkit-text-fill-color:#210026 !important;
 border:1px solid #00ff87 !important;font-weight:800 !important;opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] button:disabled {
 background:#00ff87 !important;color:#210026 !important;-webkit-text-fill-color:#210026 !important;opacity:1 !important;
}

.chip-score {font-size:1.25rem;font-weight:900;color:#37003c;}
div[data-testid="stHorizontalBlock"] .manager-panel {height:100%;}
</style>
<div class="hero">
  <div class="hero-top">
    <div>
      <div class="kicker">Fantasy Premier League · Manager dashboard</div>
      <h1>⚽ The-One-Above-All</h1>
      <p>Squad hub · Transfer intelligence · Five-Gameweek planning</p>
    </div>
    <div class="gw-chip">NEXT GAMEWEEK · AUTO DATA</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.info("Decision rule: HOLD unless a transfer clears the points-gain and reliability gates. Public FPL bank/free-transfer sync is estimated; verify it against your Transfers page before acting. This dashboard never submits transfers automatically.")
if fpl_engine.TRAINED_MODEL:
    st.success(f"Prediction engine v{APP_VERSION}: historical model loaded ({fpl_engine.TRAINED_MODEL.get('training_rows', 0):,} player-Gameweek training rows; seasons {', '.join(fpl_engine.TRAINED_MODEL.get('source_seasons', []))}).")
else:
    st.warning("Prediction engine v3.0: trained historical model not published/loaded yet; projections are using the fallback formula. Run Historical FPL Backtest successfully to train and publish trained_fpl_model.json.")

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
    control_transfers = []
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
    except Exception:
        control_transfers = []
    auto_ft, auto_ft_confident, auto_ft_note = infer_free_transfers(
        control_history, control_events, control_gw, control_transfers
    )
    auto_bank_raw = control_profile.get("last_deadline_bank") if isinstance(control_profile, dict) else None
    if auto_bank_raw is None:
        control_hist_rows = control_history.get("current", []) if isinstance(control_history, dict) else []
        control_hist_rows = sorted([r for r in control_hist_rows if isinstance(r, dict) and r.get("event") is not None], key=lambda r: int(r.get("event", 0)))
        auto_bank_raw = control_hist_rows[-1].get("bank") if control_hist_rows else None
    auto_bank = float(auto_bank_raw) / 10.0 if auto_bank_raw is not None else 0.7
    # Adjust the last-deadline bank by publicly visible transactions in the upcoming GW where prices are included.
    latest_bank_known = auto_bank_raw is not None
    current_transactions = [r for r in control_transfers if isinstance(r, dict) and str(r.get("event", "")) == str(control_gw)]
    current_tx_adjustment = 0.0
    for tx in current_transactions:
        try:
            out_cost = tx.get("element_out_cost")
            in_cost = tx.get("element_in_cost")
            if out_cost is None or in_cost is None:
                latest_bank_known = False
                continue
            current_tx_adjustment += (float(out_cost) - float(in_cost)) / 10.0
        except (TypeError, ValueError):
            latest_bank_known = False
    if current_transactions and latest_bank_known:
        auto_bank = max(0.0, min(20.0, auto_bank + current_tx_adjustment))
    sync_mode = st.selectbox("Budget & transfer source", ["Auto-sync from FPL history", "Manual override"], index=0,
                             help="Bank comes from the latest public deadline record plus visible current-Gameweek transfers. Free transfers are reconstructed from historical transfers, hit costs and chips; override if the FPL page differs.")
    auto_controls = sync_mode == "Auto-sync from FPL history"
    if auto_controls:
        st.metric("Bank available", f"£{auto_bank:.1f}m")
        st.metric("Estimated free transfers", str(auto_ft))
        bank_m = auto_bank
        free_transfers = auto_ft
        st.caption(f"Auto-sync · GW{control_gw} · Bank {'synced' if latest_bank_known else 'fallback estimate'} · Free transfers {'history-derived' if auto_ft_confident else 'estimated'}.")
        st.caption(auto_ft_note)
    else:
        bank_m = st.number_input("Bank (£m)", min_value=0.0, max_value=20.0, value=round(auto_bank, 1), step=0.1, key="manual_bank_override",
                                 help="Enter the current bank from your FPL Transfers page. This is the override for uncertain or unpublished current transfers.")
        free_transfers = st.selectbox("Free transfers available", [0,1,2,3,4,5], index=max(0, min(5, int(auto_ft))), key="manual_ft_override",
                                      help="Enter the exact number shown in the official FPL app.")
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

# Public endpoints do not require the user's FPL password. Prefer the selected
# upcoming Gameweek's public picks if the full 15-player list is available.
squad_rows = RECORDED_SQUAD
public_picks_for_display = []
picks_sync_ok = False
picks_sync_note = "Public picks for the upcoming Gameweek were not available; using the saved squad baseline."
try:
    picks_payload = get_entry_picks(int(team_id), int(gw))
    picks = picks_payload.get("picks", []) if isinstance(picks_payload, dict) else []
    elements_by_id = {int(p["id"]): p for p in players}
    if len(picks) == 15 and all(int(pk.get("element", -1)) in elements_by_id for pk in picks):
        dynamic_rows = []
        for pick in sorted(picks, key=lambda x: int(x.get("position", 99))):
            player = elements_by_id[int(pick["element"])]
            full_name = (player.get("first_name", "") + " " + player.get("second_name", "")).strip()
            dynamic_rows.append((full_name, pos_name(player, types), player.get("web_name", full_name)))
        squad_rows = dynamic_rows
        public_picks_for_display = sorted(picks, key=lambda x: int(x.get("position", 99)))
        picks_sync_ok = True
        picks_sync_note = f"Squad loaded from public picks for GW{gw}; the lineup and bench tiles will use these picks."
except Exception:
    pass

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
try:
    with st.spinner("Updating recent player history for the prediction engine…"):
        PLAYER_HISTORY_SIGNALS = fetch_history_signals(history_target_ids)
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
SNAPSHOT_5GW = {int(r["player_id"]): float(r["projection_model_5gw"]) for _, r in _current_snapshot.iterrows()
               if pd.notna(r.get("player_id")) and pd.notna(r.get("projection_model_5gw"))}
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
    elif remote_log_active:
        forecast_snapshot_note = "Automatic logging is enabled. The scheduled GitHub workflow will record this Gameweek before its deadline; refresh after the next scheduled run."
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
for _pick in public_picks_for_display:
    try:
        if _pick.get("purchase_price") is not None:
            purchase_price_by_id[int(_pick.get("element"))] = int(_pick.get("purchase_price"))
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
    except Exception:
        pass

# Profile, history and public squad sync loaded above; visual overview is rendered after squad matching.\n\n# Build squad from public picks when available, otherwise saved screenshot baseline
resolved=[]
unmatched=[]
for row in squad_rows:
    p=resolve_player(row, players)
    if p:
        teamname=teams.get(p.get("team"),{}).get("short_name","?")
        pts,diff,nfx=estimate_points(p, teamname, fixtures, horizon, start_gw=gw, calibration_offsets={})
        if int(horizon) == 5 and int(p["id"]) in SNAPSHOT_5GW:
            pts = SNAPSHOT_5GW[int(p["id"])]
        resolved.append({
            "Player": p.get("web_name"), "Position": pos_name(p,types), "Club": teamname,
            "Team ID": int(p.get("team")) if p.get("team") is not None else None,
            "Price (£m)": round((p.get("now_cost") or 0)/10,1),
            "Sell price (£m)": round((purchase_price_by_id.get(int(p["id"]), int(p.get("now_cost") or 0)) + max(0, (int(p.get("now_cost") or 0) - purchase_price_by_id.get(int(p["id"]), int(p.get("now_cost") or 0))) // 2) if int(p.get("now_cost") or 0) > purchase_price_by_id.get(int(p["id"]), int(p.get("now_cost") or 0)) else int(p.get("now_cost") or 0)) / 10, 1),
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

# Compute a shared preview of the highest-projection legal XI. The captain shortlist
# and later XI display both use this same formation logic and point model.
model_xi_start_ids = set()
model_xi_preview = []
if len(resolved) == 15:
    from itertools import combinations as _decision_combinations
    _decision_player_by_id = {int(player["id"]): player for player in players if player.get("id") is not None}
    for _resolved in resolved:
        _player = _decision_player_by_id.get(int(_resolved["id"]))
        if not _player:
            continue
        _proj, _, _ = estimate_points(_player, str(_resolved["Club"]), fixtures, horizon=1, start_gw=gw)
        model_xi_preview.append({
            "Player": str(_resolved["Player"]), "Position": str(_resolved["Position"]),
            "Club": str(_resolved["Club"]), "GW projected pts": float(_proj), "id": int(_resolved["id"]),
        })
    _best_preview = None
    _best_preview_score = -1e9
    if len(model_xi_preview) == 15:
        for _combo in _decision_combinations(model_xi_preview, 11):
            _gk = sum(1 for _x in _combo if _x["Position"] in {"GKP", "GK"})
            _def = sum(1 for _x in _combo if _x["Position"] == "DEF")
            _mid = sum(1 for _x in _combo if _x["Position"] == "MID")
            _fwd = sum(1 for _x in _combo if _x["Position"] == "FWD")
            if _gk != 1 or not (3 <= _def <= 5) or not (2 <= _mid <= 5) or not (1 <= _fwd <= 3):
                continue
            _top_pts = max(float(_x["GW projected pts"]) for _x in _combo)
            _score = sum(float(_x["GW projected pts"]) for _x in _combo) + _top_pts
            if _score > _best_preview_score:
                _best_preview_score, _best_preview = _score, list(_combo)
    if _best_preview:
        model_xi_start_ids = {int(_x["id"]) for _x in _best_preview}

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
        f"<div class='rank-row'><span>Bank ({'auto' if auto_controls else 'manual'})</span><strong>£{bank_m:.1f}m</strong></div>"
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
    st.markdown("<div class='gw-heading'><div><div class='gw-title'>Gameweek "+str(latest_history.get("event", gw))+"</div><div class='gw-sub'>Latest recorded Gameweek · starting XI and substitutes</div></div><span class='gw-chip'>NEXT · GW"+str(gw)+"</span></div>", unsafe_allow_html=True)
    if unmatched:
        st.warning("Names not uniquely matched to public data (review manually): " + ", ".join(unmatched))
    if resolved:
        squad = pd.DataFrame(resolved)
        def esc(value):
            import html
            return html.escape(str(value))
        projection_col = f"Proj GW{gw}–GW{gw+horizon-1}"
        squad_lookup = {norm(r["Player"]): r for _, r in squad.iterrows()}
        aliases = {
            "Raya": ["raya"], "Egan": ["egan"], "Hall": ["hall"], "Calafiori": ["calafiori"],
            "Palmer": ["palmer"], "Tavernier": ["tavernier"], "Groß": ["groß","gross"],
            "Rogers": ["rogers"], "Kostoulas": ["kostoulas"], "João Pedro": ["joãopedro","joaopedro"],
            "Haaland": ["haaland"], "Tzolakis": ["tzolakis"], "De Cuyper": ["decuyper"],
            "Schade": ["schade"], "Davis": ["davis"]
        }
        def get_squad_record(alias):
            for key in aliases.get(alias, [norm(alias)]):
                if norm(key) in squad_lookup:
                    return squad_lookup[norm(key)]
            matches = [r for k, r in squad_lookup.items() if norm(alias) in k or k in norm(alias)]
            return matches[0] if len(matches) == 1 else None
        club_colors = {
            "ARS":"#ef0107", "CHE":"#034694", "MCI":"#6cabdd", "LIV":"#c8102e",
            "NEW":"#242424", "BHA":"#0057b8", "BOU":"#da291c", "BRE":"#e30613",
            "FUL":"#222222", "EVE":"#003399", "TOT":"#ffffff", "AVL":"#670e36",
            "CRY":"#1b458f", "MUN":"#da291c", "WHU":"#7a263a", "NFO":"#dd0000",
            "WOL":"#fdb913", "IPS":"#005a9c", "LEI":"#003090", "SOU":"#d71920"
        }
        def tile(alias):
            r = get_squad_record(alias)
            if r is None:
                return f"<div class='player-tile'><div class='tile-top'>{esc(alias)}</div><div class='tile-avatar'>?</div><div class='tile-meta'>Data pending</div></div>"
            short = str(r["Player"])
            pts = r.get(projection_col, 0)
            club = str(r["Club"])
            color = club_colors.get(club, "#594277")
            shirt = (f"<svg class='kit-icon' viewBox='0 0 64 64' aria-label='shirt'>"
                     f"<path d='M21 8 L13 12 L4 28 L14 34 L20 24 L20 56 L44 56 L44 24 L50 34 L60 28 L51 12 L43 8 C39 15 25 15 21 8Z' fill='{color}' stroke='#ffffff' stroke-width='1.7' stroke-linejoin='round'/>"
                     f"<path d='M24 8 C26 18 38 18 40 8' fill='none' stroke='#f9f6fb' stroke-width='3'/>"
                     f"<text x='32' y='39' text-anchor='middle' font-size='12' font-weight='900' fill='#ffffff'> {esc(club[:3])} </text></svg>")
            return (f"<div class='player-tile'><div class='tile-top'>{esc(short)}</div>"
                    f"<div class='tile-avatar'>{shirt}</div><div class='tile-meta'>{esc(club)} · £{float(r['Price (£m)']):.1f}m<br>"
                    f"<span class='tile-pts'>{float(pts):.1f} pts</span></div></div>")
        if picks_sync_ok and public_picks_for_display:
            # Use actual public pick positions and player roles to render the current squad.
            picks_by_id = {int(p.get("id")): p for p in players}
            starters = [pk for pk in public_picks_for_display if int(pk.get("position", 99)) <= 11]
            actual_groups = {1: [], 2: [], 3: [], 4: []}
            for pick in starters:
                player = picks_by_id.get(int(pick.get("element", -1)))
                if player:
                    role = int(player.get("element_type", 0))
                    if role in actual_groups:
                        actual_groups[role].append(player.get("web_name", "Player"))
            xi_rows = [
                ("GOALKEEPER", actual_groups[1]),
                ("DEFENDERS", actual_groups[2]),
                ("MIDFIELDERS", actual_groups[3]),
                ("FORWARDS", actual_groups[4])
            ]
            bench_names = []
            for pick in public_picks_for_display:
                if int(pick.get("position", 0)) >= 12:
                    player = picks_by_id.get(int(pick.get("element", -1)))
                    if player:
                        bench_names.append(player.get("web_name", "Player"))
        else:
            xi_rows = [
                ("GOALKEEPER", ["Raya"]),
                ("DEFENDERS", ["Egan","Hall","Calafiori"]),
                ("MIDFIELDERS", ["Palmer","Tavernier","Groß","Rogers"]),
                ("FORWARDS", ["Kostoulas","João Pedro","Haaland"])
            ]
            bench_names = ["Tzolakis","De Cuyper","Schade","Davis"]
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
        pitch_html = "<div class='pitch-wrap'><div class='pitch-lines'></div><div class='pitch-title'>Starting XI · public picks when available</div>"
        for label, names in xi_rows:
            pitch_html += f"<div class='pitch-title'>{label}</div><div class='pitch-row'>"
            pitch_html += "".join(tile(name) for name in names)
            pitch_html += "</div>"
        pitch_html += "<div class='pitch-caption'>Projected points are model estimates over your selected horizon — not Gameweek scores.</div></div>"
        st.markdown(pitch_html, unsafe_allow_html=True)
        st.markdown("#### Bench")
        bench_parts=[]
        for name in bench_names:
            r=get_squad_record(name)
            if r is not None:
                sub=f"{esc(r['Club'])} · £{float(r['Price (£m)']):.1f}m · {float(r.get(projection_col,0)):.1f} proj."
            else:
                sub="Data pending"
            bench_parts.append(f"<div class='bench-card'><div class='bench-name'>{esc(name)}</div><div class='bench-sub'>{sub}</div></div>")
        st.markdown("<div class='bench-grid'>"+"".join(bench_parts)+"</div>", unsafe_allow_html=True)
        total_current_cost = squad["Price (£m)"].sum()
        st.caption(f"Public prices sum to £{total_current_cost:.1f}m for matched players. This is not necessarily your squad's purchase/selling value.")
        with st.expander("View full squad statistics"):
            st.dataframe(squad.drop(columns=["id"]), use_container_width=True, hide_index=True)
    else:
        st.error("No squad names could be matched. Update the name aliases in RECORDED_SQUAD.")


st.subheader("Captain & vice-captain planner")
st.markdown(
    "<div class='section-intro'>Ranks your current starting XI for the next Gameweek using the same transparent projection model, current fixture difficulty, availability and minutes signals.</div>",
    unsafe_allow_html=True
)

if resolved:
    resolved_by_norm = {norm(row.get("Player", "")): row for row in resolved}
    player_by_id = {int(player["id"]): player for player in players if player.get("id") is not None}
    captain_pool_ids = set()
    captain_source = "Saved lineup baseline"
    current_captain_id = None
    current_vice_id = None

    if picks_sync_ok and public_picks_for_display:
        captain_source = f"Public picks for GW{gw}"
        for pick in public_picks_for_display:
            try:
                element_id = int(pick.get("element"))
                if int(pick.get("position", 99)) <= 11:
                    captain_pool_ids.add(element_id)
                if pick.get("is_captain"):
                    current_captain_id = element_id
                if pick.get("is_vice_captain"):
                    current_vice_id = element_id
            except (TypeError, ValueError):
                continue
    else:
        # Last lineup shared in chat: Raya; Egan/Hall/Calafiori; Palmer/Tavernier/Groß/Rogers;
        # Kostoulas/João Pedro/Haaland. Captain Haaland, vice-captain Palmer.
        saved_starting_aliases = [
            "Raya", "Egan", "Hall", "Calafiori", "Palmer", "Tavernier",
            "Groß", "Rogers", "Kostoulas", "João Pedro", "Haaland"
        ]
        saved_starting_full_names = {
            "raya": "David Raya",
            "egan": "Egan",
            "hall": "Lewis Hall",
            "calafiori": "Riccardo Calafiori",
            "palmer": "Cole Palmer",
            "tavernier": "Marcus Tavernier",
            "groß": "Pascal Groß",
            "rogers": "Morgan Rogers",
            "kostoulas": "Charalampos Kostoulas",
            "joão pedro": "João Pedro",
            "haaland": "Erling Haaland",
        }
        for alias in saved_starting_aliases:
            normalized = norm(alias)
            matched = []
            for row in resolved:
                if row.get("id") is None:
                    continue
                player_obj = player_by_id.get(int(row["id"]), {})
                full = norm((player_obj.get("first_name", "") + " " + player_obj.get("second_name", "")).strip())
                web = norm(row.get("Player", ""))
                target_full = norm(saved_starting_full_names.get(alias.lower(), alias))
                if web == normalized or full == target_full or (normalized in web and len(normalized) >= 4):
                    matched.append(row)
            if len(matched) == 1:
                captain_pool_ids.add(int(matched[0]["id"]))
        for row in resolved:
            if norm(row.get("Player", "")) == "haaland":
                current_captain_id = int(row["id"])
            elif norm(row.get("Player", "")) == "palmer":
                current_vice_id = int(row["id"])

    if model_xi_start_ids:
        captain_pool_ids = set(model_xi_start_ids)
        captain_source = "Model-optimized legal XI (shared with starting-XI optimizer)"

    captain_rows = []
    for element_id in captain_pool_ids:
        player = player_by_id.get(int(element_id))
        if not player:
            continue
        club_code = teams.get(player.get("team"), {}).get("short_name", "?")
        gw_projection, avg_fdr, fixture_rows = estimate_points(
            player, club_code, fixtures, horizon=1, start_gw=gw
        )
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
        captain_pick = captain_df.iloc[0]
        vice_pick = captain_df.iloc[1] if len(captain_df) > 1 else None

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
            st.caption(f"Currently recorded captain: {current_captain_name}. Candidate source: {captain_source}.")
        else:
            st.caption(f"Candidate source: {captain_source}.")
        if not picks_sync_ok:
            st.warning("Current public picks were unavailable. The recommendation uses the model-optimized legal XI based on your matched 15-player squad; verify your actual team before the deadline.")
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
else:
    st.warning("Captaincy recommendations require matched squad players.")

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
    status_value = str(player.get("status") or "").lower()
    chance_value = player.get("chance_of_playing_next_round")
    if status_value in {"i", "s", "u"}:
        availability_rel = 0.20
    elif status_value == "d":
        availability_rel = 0.60
    elif chance_value is not None:
        availability_rel = max(0.20, min(1.0, float(chance_value) / 100.0))
    else:
        availability_rel = 0.88
    return 0.55 * minutes_rel + 0.45 * availability_rel

elements_by_id_chip = {
    int(player["id"]): player for player in players if player.get("id") is not None
}
chip_starter_ids, chip_bench_ids = [], []
if picks_sync_ok and public_picks_for_display:
    for pick in public_picks_for_display:
        try:
            pid = int(pick.get("element"))
            position_num = int(pick.get("position", 99))
            if position_num <= 11:
                chip_starter_ids.append(pid)
            elif position_num >= 12:
                chip_bench_ids.append(pid)
        except (TypeError, ValueError):
            continue
    chip_squad_source = "public picks"
else:
    # Last recorded GW8 lineup screenshot; do not pretend this is a live/current lineup.
    start_full_names = {
        norm("David Raya"), norm("Egan"), norm("Lewis Hall"), norm("Riccardo Calafiori"),
        norm("Cole Palmer"), norm("Marcus Tavernier"), norm("Pascal Groß"), norm("Morgan Rogers"),
        norm("Charalampos Kostoulas"), norm("João Pedro"), norm("Erling Haaland")
    }
    bench_full_names = {
        norm("Tzolakis"), norm("Maxim De Cuyper"), norm("Kevin Schade"), norm("Davis")
    }
    for recorded_row in RECORDED_SQUAD:
        matched_player = resolve_player(recorded_row, players)
        if not matched_player:
            continue
        if norm(recorded_row[0]) in start_full_names:
            chip_starter_ids.append(int(matched_player["id"]))
        elif norm(recorded_row[0]) in bench_full_names:
            chip_bench_ids.append(int(matched_player["id"]))
    chip_squad_source = "saved lineup baseline"

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
    projected, _, _ = estimate_points(player, club_code, fixtures, horizon=1, start_gw=gw)
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
    future_best_captain = max(future_points_by_id.values(), default=0.0)
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
    if bench_dgw_count_chip >= 2 and bench_points_chip >= 12 and bench_reliability_chip >= 0.70:
        bb_action = "Consider this GW"
        bb_detail = f"{bench_dgw_count_chip} bench player(s) have a Double Gameweek; projected bench total {bench_points_chip:.1f} points."
    elif bench_dgw_count_chip > 0:
        bb_action = "Monitor / compare bench"
        bb_detail = f"{bench_dgw_count_chip} bench player(s) have a Double Gameweek; projected bench total {bench_points_chip:.1f} points."
    else:
        bb_action = "Save for a stronger bench/DGW"
        bb_detail = f"Projected bench total is {bench_points_chip:.1f} points across {len(bench_chip_rows)} matched bench players. No bench Double Gameweek is currently assigned."
    if best_future_bench and best_future_bench["bench_points"] >= bench_points_chip + 2.0:
        bb_action = f"Save for GW{best_future_bench['gameweek']}"
        bb_detail = (f"Current bench estimate: {bench_points_chip:.1f} points; the best later scheduled Gameweek in the planning window is "
                    f"GW{best_future_bench['gameweek']} at {best_future_bench['bench_points']:.1f}. Recheck the squad closer to that deadline.")
    elif bench_points_chip >= 12 and bench_reliability_chip >= 0.70:
        bb_action = "Potential use this GW"
        bb_detail = (f"Current bench estimate is {bench_points_chip:.1f} with reliability {bench_reliability_chip:.0%}. "
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
engine_c3.metric("Forecast record", "Saved" if not forecast_log.empty and ((forecast_log["season"].astype(str) == season_id) & (pd.to_numeric(forecast_log["gameweek"], errors="coerce") == int(gw))).any() else "Not saved")
engine_c4.metric("Model version", APP_VERSION)
st.caption("Calibration is paused while v3.0 is benchmarked, so the tested model does not change weights between Gameweeks.")
if forecast_snapshot_note:
    st.caption(forecast_snapshot_note)
if not next_week_df.empty:
    st.markdown(f"**Top projected players for GW{gw}**")
    st.dataframe(next_week_df.head(15).drop(columns=["id"]), use_container_width=True, hide_index=True)

# Optimize the current squad's legal XI, then account for the captain's additional expected points.
# This is a separate recommendation from the saved/public lineup and never submits changes to FPL.
if len(resolved) == 15:
    from itertools import combinations
    gw_projection_by_id = {int(r["id"]): float(r[f"GW{gw} projected pts"]) for _, r in next_week_df.iterrows()}
    xi_pool = []
    for _r in resolved:
        _pid = int(_r["id"])
        _position = str(_r["Position"])
        xi_pool.append({
            "Player": str(_r["Player"]), "Position": _position, "Club": str(_r["Club"]),
            "Price (£m)": float(_r["Price (£m)"]), "GW projected pts": gw_projection_by_id.get(_pid, 0.0),
            "id": _pid,
        })
    best_xi = None
    best_xi_score = -1e9
    for _combo in combinations(xi_pool, 11):
        _gk = sum(1 for _x in _combo if _x["Position"] in {"GKP", "GK"})
        _def = sum(1 for _x in _combo if _x["Position"] == "DEF")
        _mid = sum(1 for _x in _combo if _x["Position"] == "MID")
        _fwd = sum(1 for _x in _combo if _x["Position"] == "FWD")
        if _gk != 1 or not (3 <= _def <= 5) or not (2 <= _mid <= 5) or not (1 <= _fwd <= 3):
            continue
        _captain_projection = max(float(_x["GW projected pts"]) for _x in _combo)
        _score = sum(float(_x["GW projected pts"]) for _x in _combo) + _captain_projection
        if _score > best_xi_score:
            best_xi_score = _score
            best_xi = list(_combo)
    if best_xi:
        best_xi = sorted(best_xi, key=lambda _x: ( {"GKP": 1, "GK": 1, "DEF": 2, "MID": 3, "FWD": 4}.get(_x["Position"], 5), -_x["GW projected pts"] ))
        _gk = sum(1 for _x in best_xi if _x["Position"] in {"GKP", "GK"})
        _def = sum(1 for _x in best_xi if _x["Position"] == "DEF")
        _mid = sum(1 for _x in best_xi if _x["Position"] == "MID")
        _fwd = sum(1 for _x in best_xi if _x["Position"] == "FWD")
        formation = f"{_def}-{_mid}-{_fwd}"
        captain_pick = max(best_xi, key=lambda _x: _x["GW projected pts"])
        vice_options = [x for x in best_xi if x["id"] != captain_pick["id"]]
        vice_pick = max(vice_options, key=lambda _x: _x["GW projected pts"])
        starting_ids = {x["id"] for x in best_xi}
        bench = [x for x in xi_pool if x["id"] not in starting_ids]
        bench_gk = sorted([x for x in bench if x["Position"] in {"GKP", "GK"}], key=lambda x: -x["GW projected pts"])
        bench_outfield = sorted([x for x in bench if x["Position"] not in {"GKP", "GK"}], key=lambda x: -x["GW projected pts"])
        bench_ordered = bench_gk[:1] + bench_outfield[:3]
        xi_a, xi_b, xi_c = st.columns([1.1, 1.1, 1.8])
        xi_a.metric("Model-best formation", formation)
        xi_b.metric("XI projected points", f"{sum(x['GW projected pts'] for x in best_xi):.1f}")
        xi_c.metric("Captain / vice-captain", f"{captain_pick['Player']} / {vice_pick['Player']}", help="Captain gets an additional copy of their projected points; vice-captain is the next-highest projected legal XI player.")
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
            bench_frame["Order"] = ["Goalkeeper bench", "Outfield 1", "Outfield 2", "Outfield 3"][:len(bench_frame)]
            st.dataframe(bench_frame[["Order", "Player", "Position", "GW projected pts"]].rename(columns={"GW projected pts": f"GW{gw} pts"}), use_container_width=True, hide_index=True)
        st.caption("This selection maximizes the sum of one-Gameweek projections plus the captain's additional expected points across legal FPL formations. It only chooses an XI from your 15 matched squad players; it does not change transfers or submit anything to FPL.")
    else:
        st.warning("The current 15-player squad could not form a legal XI under the standard positional constraints. Check unmatched positions/player data.")
else:
    st.info(f"Best-XI optimizer needs all 15 squad members matched; currently matched: {len(resolved)}.")

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

st.subheader("Automatic transfer recommendations")
st.caption(f"Team ID: {int(team_id)} • The optimizer scans every matched squad player and searches for legal, affordable replacements. Budget source: {'public FPL sync' if auto_controls else 'manual override'}.")
st.write("This engine is allowed to recommend HOLD. It only labels a transfer actionable when the estimated multi-Gameweek points edge clears a conservative threshold, incoming minutes/availability are reliable, the squad is fully matched, and a points hit is justified. It never submits transfers automatically.")
if resolved:
    df = pd.DataFrame(resolved)
    candidates = pd.DataFrame([{
        "Player": p.get("web_name"), "Position": pos_name(p,types),
        "Club": teams.get(p.get("team"),{}).get("short_name","?"),
        "Price (£m)": round((p.get("now_cost") or 0)/10,1),
        "Form": float(p.get("form") or 0), "PPG": float(p.get("points_per_game") or 0),
        "Minutes": int(p.get("minutes") or 0), "Status": p.get("status","?"),
        "Chance (%)": p.get("chance_of_playing_next_round"),
        f"Projected {horizon}GW pts": (SNAPSHOT_5GW.get(int(p["id"]), estimate_points(p, teams.get(p.get("team"),{}).get("short_name","?"), fixtures, horizon, start_gw=gw, calibration_offsets={})[0]) if int(horizon) == 5 else estimate_points(p, teams.get(p.get("team"),{}).get("short_name","?"), fixtures, horizon, start_gw=gw, calibration_offsets={})[0]),
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
    # AUTOMATIC TRANSFER SCAN: evaluate every possible player-out / player-in pair.
    squad_names = sorted(df["Player"].dropna().unique().tolist())
    club_counts_auto = df["Club"].value_counts().to_dict()
    hit_cost_auto = 4 if free_transfers == 0 else 0
    auto_rows = []

    def reliability_for_candidate(cand):
        mins = float(cand.get("Minutes", 0) or 0)
        mins_rel = max(0.15, min(1.0, mins / 600.0))
        status_val = str(cand.get("Status", "")).lower()
        chance_val = cand.get("Chance (%)")
        if status_val in {"i", "s", "u"}:
            avail_rel = 0.20
        elif status_val == "d":
            avail_rel = 0.60
        elif pd.notna(chance_val):
            avail_rel = max(0.20, min(1.0, float(chance_val) / 100.0))
        else:
            avail_rel = 0.88
        return mins_rel, avail_rel

    for _, outgoing in df.iterrows():
        outgoing_price = float(outgoing.get("Sell price (£m)", outgoing["Price (£m)"]))
        budget_ceiling = outgoing_price + float(bank_m)
        same_position = candidates[candidates["Position"] == outgoing["Position"]]
        possible_buys = same_position[
            (~same_position["Player"].isin(squad_names)) &
            (same_position["Price (£m)"] <= budget_ceiling)
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
                "Position": outgoing["Position"], "Buy price (£m)": float(incoming["Price (£m)"]),
                "Budget ceiling (£m)": round(budget_ceiling, 1),
                "Club count after transfer": club_after,
                "Outgoing projected pts": round(outgoing_proj, 1),
                "Incoming projected pts": round(incoming_proj, 1),
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

    if unmatched:
        st.warning("Automatic recommendations are provisional because some seeded squad names did not match public FPL records: " + ", ".join(unmatched) + ". Club limits and squad context may be incomplete until every player is matched.")

    hours_to_deadline_now = None
    try:
        if deadline_dt is not None:
            hours_to_deadline_now = (deadline_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0
    except Exception:
        hours_to_deadline_now = None

    if auto_rows:
        auto_df = pd.DataFrame(auto_rows)
        # Show the highest-scoring legal swap for each player we could sell.
        best_each = (auto_df.sort_values(["Actionable", "Decision score", "Projected gain"], ascending=False)
                     .drop_duplicates(subset=["Sell"], keep="first")
                     .sort_values(["Actionable", "Decision score", "Projected gain"], ascending=False)
                     .head(10))
        st.caption(f"Risk mode: {risk} • Horizon: {horizon} GWs • Transfer hit applied: {hit_cost_auto} points. Budget uses the synced bank and estimated personal sell price; verify exact FPL values before acting.")
        st.dataframe(best_each[["Sell","Sell club","Buy","Buy club","Position","Buy price (£m)","Budget ceiling (£m)","Club count after transfer","Outgoing projected pts","Incoming projected pts","Projected gain","Minimum gain to act","Minutes reliability (%)","Availability reliability (%)","Captaincy exposure","Actionable","Ownership (%)","Decision score","Verdict"]], use_container_width=True, hide_index=True)
        top = best_each.iloc[0]
        gain = float(top["Projected gain"])
        threshold = float(top["Minimum gain to act"])
        reliable = float(top["Minutes reliability (%)"]) >= 70 and float(top["Availability reliability (%)"]) >= 80
        candidate_is_robust = bool(top.get("Actionable", False)) and gain >= threshold and reliable and not unmatched
        if unmatched:
            st.warning("Decision: HOLD / verify squad. At least one squad member is unmatched, so the optimizer cannot safely assess the legal squad or club limit.")
            st.markdown(f"**Best candidate for review:** {top['Sell']} → {top['Buy']} · estimated {gain:+.1f} points over {horizon} Gameweeks. This is not an action recommendation until all 15 players match.")
        elif free_transfers == 0 and not transfer_hit:
            st.info(f"**Decision: HOLD — no transfer justified without a points hit.** Best candidate is {top['Sell']} → {top['Buy']} at {gain:+.1f} projected points after accounting for the -4 hit. Taking that hit is disabled, so the model will not recommend taking one.")
        elif candidate_is_robust and (free_transfers > 0 or (transfer_hit and gain >= threshold)) and hours_to_deadline_now is not None and hours_to_deadline_now > 24 and not bool(top.get("Outgoing unavailable", False)):
            st.info(f"**Decision: HOLD FOR NOW — candidate to monitor:** {top['Sell']} → {top['Buy']} ({top['Buy club']}) · estimated net gain {gain:+.1f} over {horizon} Gameweeks. It clears the points/reliability gate, but the deadline is still more than 24 hours away.")
            st.markdown("**Suggested timing:** Re-check within 24 hours of the deadline, then again after the latest team news. Only act if the move still clears the threshold and the official FPL selling price/bank confirm it is affordable. A projection is not a guarantee.")
        elif candidate_is_robust and (free_transfers > 0 or (transfer_hit and gain >= threshold)) and hours_to_deadline_now is not None and hours_to_deadline_now > 0:
            st.success(f"**Decision: TRANSFER CANDIDATE — final review window** · {top['Sell']} → {top['Buy']} ({top['Buy club']}) · estimated net gain {gain:+.1f} points over {horizon} Gameweeks. It clears the {threshold:.1f}-point action threshold and the candidate's reliability checks.")
            st.markdown("**Before acting:** Check press-conference/injury news and verify the real selling price/bank in FPL. If new information drops the gain below the threshold or minutes/availability confidence, switch back to HOLD.")
        else:
            reason = []
            if gain < threshold: reason.append(f"projected edge {gain:+.1f} is below the {threshold:.1f}-point action threshold")
            if not reliable: reason.append("incoming minutes/availability are not reliable enough")
            if free_transfers == 0 and transfer_hit and gain < threshold: reason.append("the gain does not repay the hit with enough margin")
            st.info("**Decision: HOLD — no transfer clears the decision gate.** " + ("; ".join(reason) if reason else "The model cannot establish a sufficiently robust advantage."))
            st.markdown(f"Best candidate to monitor: **{top['Sell']} → {top['Buy']}**, estimated {gain:+.1f} points over {horizon} Gameweeks. Reassess after team news or if the player’s status changes; an unused free transfer can roll over up to the rules limit.")
    else:
        st.info("No legal, affordable one-for-one transfers were found. Check the matched squad, budget and club-count warnings.")
        st.markdown("**Quick summary:** No transfer recommendation can be made until the squad and affordability data are resolved.")

    st.divider()
    st.subheader("Manual transfer comparison (optional)")
    st.write("Use this section only to investigate a specific player-out choice; the optimizer above scans your squad automatically.")

    # Current squad selector
    squad_names = sorted(df["Player"].unique().tolist())
    sell_name = st.selectbox("Sell player", squad_names, index=(squad_names.index("Schade") if "Schade" in squad_names else 0))
    sell = df[df["Player"]==sell_name].iloc[0]
    max_price = float(sell.get("Sell price (£m)", sell["Price (£m)"])) + bank_m
    options = candidates[(candidates["Position"]==sell["Position"]) &
                         (~candidates["Player"].isin(squad_names)) &
                         (candidates["Price (£m)"] <= max_price)].copy()

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
    options["Hit cost"] = 4 if free_transfers == 0 else 0
    options["Projected gain"] = options["Raw gain"] - options["Hit cost"]
    # Risk setting affects the ranking, not the raw point projection.
    ownership = options["Selected (%)"].fillna(0).astype(float).clip(0, 100)
    # Reliability is deliberately stronger than in v1.3: fewer than 300 minutes
    # should materially reduce confidence in a transfer recommendation.
    minutes_reliability = (options["Minutes"].astype(float) / 600.0).clip(0.15, 1.0)
    unknown_status = options["Chance (%)"].isna()
    doubtful = options["Status"].astype(str).str.lower().eq("d")
    availability_reliability = pd.Series(1.0, index=options.index)
    availability_reliability.loc[unknown_status] = 0.88
    availability_reliability.loc[doubtful] = 0.60
    unavailable = options["Status"].astype(str).str.lower().isin(["i", "s", "u"])
    availability_reliability.loc[unavailable] = 0.20

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
        st.caption(f"Replacement ceiling: approximately £{max_price:.1f}m, using the manager-specific sale estimate + synced bank. Confirm the current official FPL selling price before confirming a transfer.")
        st.caption(f"Ranking mode: {risk}. Horizon: {horizon} GWs. Hit penalty: {4 if (transfer_hit and free_transfers == 0) else 0} points. Decision score now explicitly weights minutes reliability, availability, differential ownership and xG+xA/90.")
        options["Minutes reliability (%)"] = (minutes_reliability * 100).round(0).astype(int)
        options["Availability reliability (%)"] = (availability_reliability * 100).round(0).astype(int)
        st.dataframe(options[["Player","Club","Price (£m)","Current club count","Club count after transfer","Form","PPG","Minutes","Minutes reliability (%)","Status","Chance (%)","Availability reliability (%)","xG","xA","xG/90","xA/90",f"Projected {horizon}GW pts","Selected (%)","Projected gain","Decision score"]], use_container_width=True, hide_index=True)
        best=options.iloc[0]
        st.success(f"Top candidate under {risk.lower()} settings: {best['Player']} ({best['Club']}) — projected gain {best['Projected gain']:+.1f} points over {horizon} GWs. Decision score includes your risk preference, minutes and availability reliability, ownership and xG/xA signals; it is not a guaranteed return.")
        st.write("Decision rule: only transfer if the gain remains worthwhile after checking availability, actual selling price, expected minutes and the other 14 squad slots. A high projection alone is not a command to transfer.")

st.subheader("What the model does — and does not do")
st.markdown("""
- Blends PPG/form with a conservative xG+xA signal, shrinks small samples toward a position baseline, estimates minutes from starts and appearances, penalizes uncertain/unavailable status, and adjusts each scheduled fixture individually. Displays cumulative xG/xA and per-90 rates.
- Uses public player status and chance-of-playing fields where provided.
- Does **not** access your FPL password, private `my-team` endpoint, or your personal selling prices.
- Building on v2.8, this v3.0 validation release adds matched live benchmarks, expected-minutes diagnostics, historical replay and explicit ranking metrics alongside the existing transfer HOLD gate. The model remains heuristic until enough saved forecasts and actual scores exist; no claim of self-learning accuracy is made before backtesting shows an improvement. The transfer search still evaluates one transfer at a time and does not submit transfers to FPL.
- Public FPL API is unofficial/undocumented and may change. Player/fixture data is cached for 30 minutes; manager profile/rank for 60 seconds; public picks/history for 2 minutes.
- A sidebar **Refresh rank & squad** button clears data caches and reruns the dashboard. Finance sync is derived from public data; use Manual override if the FPL Transfers page shows a different live balance.
- Squad sync attempts to use all 15 players from `entry/{id}/event/{gw}/picks/` for the next Gameweek. If the picks endpoint doesn't provide all 15 (especially before deadline), the app clearly falls back to the saved screenshot baseline.
- The app does not access the authenticated `/my-team/{id}/` endpoint and does not ask for or store an FPL password/cookie. This means the latest editable pre-deadline team might not always be publicly available.
""")
st.caption(f"The-One-Above-All v3.0 · Last refreshed in this session: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Use the app's refresh/reload to request fresh data.")
