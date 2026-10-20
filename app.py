import streamlit as st
import pandas as pd
import requests
import html
from datetime import datetime, timezone

st.set_page_config(page_title="The-One-Above-All | FPL Manager Hub", layout="wide")
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
    r = requests.get(f"{BASE}/{path}", timeout=20, headers={"User-Agent":"FPL-Optimizer/1.0"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=60, show_spinner=False)
def get_manager_profile(team_id):
    """Short-cache public profile so rank/points refresh more frequently."""
    r = requests.get(f"{BASE}/entry/{int(team_id)}/", timeout=20,
                     headers={"User-Agent":"Mozilla/5.0 FPL-Optimizer/2.0"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=120, show_spinner=False)
def get_entry_picks(team_id, gameweek):
    """Public picks for a specific Gameweek; may not expose editable picks before deadline."""
    r = requests.get(f"{BASE}/entry/{int(team_id)}/event/{int(gameweek)}/picks/",
                     timeout=20, headers={"User-Agent":"Mozilla/5.0 FPL-Optimizer/2.0"})
    r.raise_for_status()
    return r.json()

@st.cache_data(ttl=120, show_spinner=False)
def get_entry_history(team_id):
    r = requests.get(f"{BASE}/entry/{int(team_id)}/history/",
                     timeout=20, headers={"User-Agent":"Mozilla/5.0 FPL-Optimizer/2.0"})
    r.raise_for_status()
    return r.json()

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

def estimate_points(p, team, fixtures, horizon=5, start_gw=None):
    """Transparent heuristic using form, PPG, xG/xA, minutes and availability."""
    def num(key):
        try:
            return float(p.get(key) or 0)
        except (TypeError, ValueError):
            return 0.0

    ppg, form = num("points_per_game"), num("form")
    minutes, starts = num("minutes"), int(num("starts"))
    status = str(p.get("status") or "").lower()
    chance = p.get("chance_of_playing_next_round")
    pos = p.get("element_type")
    baseline = {1: 3.2, 2: 3.5, 3: 3.8, 4: 4.0}.get(pos, 3.6)

    # FPL bootstrap cumulative expected-goal fields, converted to per-90 rates.
    xg90 = num("expected_goals") / max(minutes, 90) * 90
    xa90 = num("expected_assists") / max(minutes, 90) * 90
    goal_threat = xg90 + xa90
    # Conservative signal; do not let noisy xG/xA overwhelm historical points.
    attacking_signal = min(1.5, goal_threat * 2.0)
    raw = 0.50 * ppg + 0.25 * form + 0.25 * (baseline + attacking_signal)
    sample_weight = min(0.85, minutes / 900.0)
    per_match = sample_weight * raw + (1.0 - sample_weight) * baseline

    if minutes <= 0:
        expected_minutes = 35.0
    elif starts > 0:
        avg_start_minutes = minutes / starts
        start_rate = min(1.0, starts / max(1.0, num("starts") + num("substitute_appearances")))
        expected_minutes = min(90.0, avg_start_minutes) * (0.65 + 0.35 * start_rate)
    else:
        expected_minutes = min(60.0, minutes / max(1.0, num("appearances")))

    minutes_factor = max(0.35, min(1.0, expected_minutes / 80.0))
    if status in {"i", "s", "u"}:
        availability_factor = 0.20
    elif status == "d":
        availability_factor = 0.55
    elif chance is not None:
        availability_factor = max(0.20, min(1.0, float(chance) / 100.0))
    else:
        availability_factor = 0.90

    upcoming = [f for f in fixtures if not f.get("finished")
                and f.get("event") is not None
                and (f.get("team_h") == p.get("team") or f.get("team_a") == p.get("team"))]
    if start_gw is not None:
        # Horizon counts distinct Gameweeks. Keep every fixture within them,
        # including two fixtures assigned to the same Gameweek (DGW).
        upcoming = [f for f in upcoming
                    if int(start_gw) <= int(f["event"]) < int(start_gw) + int(horizon)]
    else:
        event_ids = sorted({int(f["event"]) for f in upcoming})
        target_ids = set(event_ids[:int(horizon)])
        upcoming = [f for f in upcoming if int(f["event"]) in target_ids]
    upcoming.sort(key=lambda f: (f.get("event") or 99, f.get("kickoff_time") or ""))
    scores, diffs = [], []
    for f in upcoming:
        home = f.get("team_h") == p.get("team")
        diff = float(f.get("team_h_difficulty" if home else "team_a_difficulty") or 3)
        diffs.append(diff)
        fixture_factor = max(0.82, min(1.16, 1.0 + (3.0 - diff) * 0.06))
        scores.append(per_match * minutes_factor * availability_factor * fixture_factor)
    if not scores:
        return 0.0, 3.0, 0
    return round(sum(scores), 1), round(sum(diffs) / len(diffs), 2), len(scores)

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
st.info("Forecasts are estimates, not official FPL projections. Confirm player availability and your personal selling prices before transferring.")

with st.sidebar:
    st.header("⚙️ Optimizer settings")
    st.caption("These controls change the calculations below.")
    team_id = st.number_input("FPL Team ID", min_value=1, value=2311669, step=1)
    horizon = st.slider("Projection horizon (Gameweeks)", 1, 8, 5,
                         help="How many upcoming Gameweeks to include in player projections and transfer gains.")
    fixture_horizon = st.slider("Fixture calendar horizon (Gameweeks)", 5, 20, 15,
                                 help="How far ahead to display fixtures and identify officially assigned double Gameweeks. Player-point projections keep their own shorter horizon.")
    bank_m = st.number_input("Bank (£m)", min_value=0.0, max_value=20.0, value=0.7, step=0.1,
                             help="Added to the selling player's current public price to estimate an affordable replacement. Actual selling price may differ.")
    free_transfers = st.selectbox("Free transfers available", [1,2,3,4,5], index=0,
                                  help="Used in the decision guidance; the current shortlist compares one transfer at a time.")
    transfer_hit = st.checkbox("Allow a -4 hit if model sees a gain", value=False,
                                help="If enabled, subtracts four points from the projected gain before ranking. A hit should only be considered for a clear multi-week advantage.")
    risk = st.select_slider("Risk appetite", options=["Conservative","Moderate","Aggressive"], value="Moderate",
                             help="Conservative favors established minutes; Moderate balances projected gain and differential upside; Aggressive gives more weight to low-ownership upside.")
    if st.button("↻ Refresh rank & squad", use_container_width=True):
        get_manager_profile.clear()
        get_entry_picks.clear()
        get_entry_history.clear()
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

# Chip status is not reliably exposed by the public manager profile, so keep it editable.
chip_period_key = "first_half" if int(gw) < 20 else "second_half"
chip_period_label = "First-half chips (before GW19 deadline)" if int(gw) < 20 else "Second-half chips (GW20 onwards)"
with st.sidebar:
    st.divider()
    with st.expander(f"🎟️ Chip status · {chip_period_label}", expanded=False):
        st.caption("Seeded from your last screenshot for the first half: Wildcard used in GW6; Bench Boost, Free Hit and Triple Captain available. Please update these manually after playing a chip.")
        wildcard_options = ["Available", "Played in GW6", "Played in another Gameweek", "Unknown / verify"]
        wildcard_default = 1 if int(gw) < 20 else 0
        wildcard_status = st.selectbox(
            "Wildcard", wildcard_options, index=wildcard_default,
            key=f"chip_wildcard_{chip_period_key}"
        )
        chip_options = ["Available", "Played in a previous Gameweek", "Unknown / verify"]
        chip_default = 0
        bench_boost_status = st.selectbox(
            "Bench Boost", chip_options, index=chip_default,
            key=f"chip_bb_{chip_period_key}"
        )
        free_hit_status = st.selectbox(
            "Free Hit", chip_options, index=chip_default,
            key=f"chip_fh_{chip_period_key}"
        )
        triple_captain_status = st.selectbox(
            "Triple Captain", chip_options, index=chip_default,
            key=f"chip_tc_{chip_period_key}"
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

# Profile, history and public squad sync loaded above; visual overview is rendered after squad matching.\n\n# Build squad from public picks when available, otherwise saved screenshot baseline
resolved=[]
unmatched=[]
for row in squad_rows:
    p=resolve_player(row, players)
    if p:
        teamname=teams.get(p.get("team"),{}).get("short_name","?")
        pts,diff,nfx=estimate_points(p, teamname, fixtures, horizon, start_gw=gw)
        resolved.append({
            "Player": p.get("web_name"), "Position": pos_name(p,types), "Club": teamname,
            "Team ID": int(p.get("team")) if p.get("team") is not None else None,
            "Price (£m)": round((p.get("now_cost") or 0)/10,1),
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
    st.markdown(
        "<div class='manager-panel'><div class='panel-heading'>Squad controls</div>"
        f"<div class='rank-row'><span>Bank entered</span><strong>£{bank_m:.1f}m</strong></div>"
        f"<div class='rank-row'><span>Free transfers entered</span><strong>{free_transfers}</strong></div>"
        f"<div class='rank-row'><span>Points horizon</span><strong>{horizon} GWs</strong></div>"
        f"<div class='rank-row'><span>Fixture calendar</span><strong>{fixture_horizon} GWs</strong></div>"
        f"<div class='rank-row'><span>Risk mode</span><strong>{html.escape(risk)}</strong></div>"
        "</div>",
        unsafe_allow_html=True
    )
    # Saved, non-live league rank is deliberately labelled as the last screenshot checkpoint.
    st.markdown(
        "<div class='manager-panel'><div class='panel-heading'>League tracking</div>"
        "<div class='league-pill'><span>India league</span><strong>14,002</strong></div>"
        "<div class='manager-overview-note'>Last manually captured from your FPL screenshot. This figure is not live-synced.</div>"
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
            st.warning("Captaincy shortlist uses your last saved lineup screenshot because current public picks were unavailable. Confirm your actual starting XI before the deadline.")
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

st.subheader("Automatic transfer recommendations")
st.caption("Team ID: 2311669 • The optimizer scans every matched squad player and searches for legal, affordable replacements.")
st.write("Recommendations are generated automatically from projected points, risk settings, estimated budget and the three-per-club rule. They are decision support, not automatic transfers.")
if resolved:
    df = pd.DataFrame(resolved)
    candidates = pd.DataFrame([{
        "Player": p.get("web_name"), "Position": pos_name(p,types),
        "Club": teams.get(p.get("team"),{}).get("short_name","?"),
        "Price (£m)": round((p.get("now_cost") or 0)/10,1),
        "Form": float(p.get("form") or 0), "PPG": float(p.get("points_per_game") or 0),
        "Minutes": int(p.get("minutes") or 0), "Status": p.get("status","?"),
        "Chance (%)": p.get("chance_of_playing_next_round"),
        f"Projected {horizon}GW pts": estimate_points(p, teams.get(p.get("team"),{}).get("short_name","?"), fixtures, horizon, start_gw=gw)[0],
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
    hit_cost_auto = 4 if (transfer_hit and free_transfers == 0) else 0
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
        outgoing_price = float(outgoing["Price (£m)"])
        budget_ceiling = outgoing_price + float(bank_m)  # approximate; personal sell price may differ
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
                "Verdict": verdict,
            })

    if unmatched:
        st.warning("Automatic recommendations are provisional because some seeded squad names did not match public FPL records: " + ", ".join(unmatched) + ". Club limits and squad context may be incomplete until every player is matched.")

    if auto_rows:
        auto_df = pd.DataFrame(auto_rows)
        # Show the highest-scoring legal swap for each player we could sell.
        best_each = (auto_df.sort_values(["Decision score", "Projected gain"], ascending=False)
                     .drop_duplicates(subset=["Sell"], keep="first")
                     .sort_values(["Decision score", "Projected gain"], ascending=False)
                     .head(10))
        st.caption(f"Risk mode: {risk} • Horizon: {horizon} GWs • Transfer hit applied: {hit_cost_auto} points. Prices use public prices plus bank as an approximation.")
        st.dataframe(best_each[["Sell","Sell club","Buy","Buy club","Position","Buy price (£m)","Budget ceiling (£m)","Club count after transfer","Outgoing projected pts","Incoming projected pts","Projected gain","Minutes reliability (%)","Availability reliability (%)","Ownership (%)","Decision score","Verdict"]], use_container_width=True, hide_index=True)
        top = best_each.iloc[0]
        if unmatched:
            st.warning("Automatic recommendations are provisional because one or more squad players are not matched to public FPL data. Verify the full squad and club counts before acting.")
            st.markdown(f"**Quick summary:** Top model-ranked move is **{top['Sell']} → {top['Buy']}** at an estimated **{top['Projected gain']:+.1f} points** over {horizon} Gameweeks, but this is not actionable until squad matching is complete.")
        elif str(top["Sell"]).lower() == "haaland":
            st.warning("Captaincy check required: Haaland was captain in your last recorded lineup. This model does not fully price the opportunity cost of replacing your captain.")
            st.markdown(f"**Quick summary:** The model ranks **Haaland → {top['Buy']}** at **{top['Projected gain']:+.1f} points** over {horizon} Gameweeks, but do not act before comparing captaincy options and verifying actual selling price and team news.")
        elif float(top["Projected gain"]) > 0:
            st.success(f"Top automatic move under {risk.lower()} settings: {top['Sell']} → {top['Buy']} ({top['Buy club']}), projected {top['Projected gain']:+.1f} points over {horizon} GWs. Verdict: {top['Verdict']}. Check team news and actual selling price before acting.")
            if float(top["Projected gain"]) >= 5 and float(top["Minutes reliability (%)"]) >= 70 and float(top["Availability reliability (%)"]) >= 80:
                summary_action = "This is worth investigating as a potential transfer."
            else:
                summary_action = "This is only a candidate; uncertainty makes it unsuitable as an automatic transfer instruction."
            st.markdown(
                f"**Quick summary:** Best model-ranked move: **{top['Sell']} → {top['Buy']}** for an estimated **{top['Projected gain']:+.1f} points** over {horizon} Gameweeks. "
                f"{summary_action} You have **{free_transfers} free transfer(s)** recorded; verify the player's role, availability and your personal selling price before deciding."
            )
        else:
            st.info("No legal transfer has a positive projected gain under the current settings. Rolling the transfer may be better.")
            st.markdown(
                f"**Quick summary:** The current model finds no legal, affordable move with positive projected gain over {horizon} Gameweeks under {risk.lower()} settings. "
                f"With **{free_transfers} free transfer(s)** available, preserving flexibility looks preferable until team news or projections improve."
            )
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
    max_price = float(sell["Price (£m)"]) + bank_m
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
    options["Hit cost"] = 4 if (transfer_hit and free_transfers == 0) else 0
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
        st.caption(f"Approximate replacement ceiling: £{max_price:.1f}m, using current price + bank. Enter actual selling value separately in your final decision.")
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
- This v2.6 is a transparent baseline, not a trained machine-learning model. It does not yet optimize the entire 15-player squad simultaneously, model price changes, or estimate rank probabilities.
- Public FPL API is unofficial/undocumented and may change. Player/fixture data is cached for 30 minutes; manager profile/rank for 60 seconds; public picks/history for 2 minutes.
- A sidebar **Refresh rank & squad** button clears data caches and reruns the dashboard.
- Squad sync attempts to use all 15 players from `entry/{id}/event/{gw}/picks/` for the next Gameweek. If the picks endpoint doesn't provide all 15 (especially before deadline), the app clearly falls back to the saved screenshot baseline.
- The app does not access the authenticated `/my-team/{id}/` endpoint and does not ask for or store an FPL password/cookie. This means the latest editable pre-deadline team might not always be publicly available.
""")
st.caption(f"Last refreshed in this session: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Use the app's refresh/reload to request fresh data.")
