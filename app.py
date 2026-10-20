import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timezone

st.set_page_config(page_title="The-One-Above-All | FPL Optimizer", layout="wide")
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

def norm(s):
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def resolve_player(row, players):
    # Match exact normalized display name first, then web_name; show unmatched instead of guessing.
    candidates = [p for p in players if norm(p.get("web_name","")) == norm(row[2])]
    if not candidates:
        candidates = [p for p in players if norm(p.get("first_name","")+" "+p.get("second_name","")) == norm(row[0])]
    if not candidates:
        # cautious partial match using web_name only if unique
        candidates = [p for p in players if norm(row[2]) in norm(p.get("web_name",""))]
    return candidates[0] if len(candidates) == 1 else None

def pos_name(p, types):
    return next((x.get("singular_name_short", "?") for x in types if x["id"] == p.get("element_type")), "?")

def estimate_points(p, team, fixtures, horizon=5):
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
                and (f.get("team_h") == p.get("team") or f.get("team_a") == p.get("team"))]
    upcoming.sort(key=lambda f: (f.get("event") or 99, f.get("kickoff_time") or ""))
    upcoming = [f for f in upcoming if f.get("event") is not None][:horizon]
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
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#101b35 0%,#182746 100%);}
[data-testid="stSidebar"] * {color: #f3f6ff !important;}
.hero {padding: 1.35rem 1.5rem; border-radius: 18px; background: linear-gradient(115deg,#101b35 0%,#244b78 58%,#2d8a78 100%); color: white; margin-bottom: 1rem;}
.hero h1 {color: white; margin: 0; font-size: 2rem;}
.hero p {color: #dce9ff; margin: .45rem 0 0 0;}
.kpi {padding: 1rem; border: 1px solid #e5eaf2; border-radius: 14px; background: #fff; box-shadow: 0 2px 8px rgba(16,27,53,.04);}
.section-label {font-size: .85rem; text-transform: uppercase; letter-spacing: .08em; color: #64748b;}
div[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
</style>
<div class="hero">
  <h1>⚽ The-One-Above-All</h1>
  <p>Personal FPL decision desk · Team 2311669 · Five-Gameweek planning</p>
</div>
""", unsafe_allow_html=True)
st.info("Forecasts are estimates, not official FPL projections. Confirm player availability and your personal selling prices before transferring.")

with st.sidebar:
    st.header("⚙️ Optimizer settings")
    st.caption("These controls change the calculations below.")
    team_id = st.number_input("FPL Team ID", min_value=1, value=2311669, step=1)
    horizon = st.slider("Projection horizon (Gameweeks)", 1, 8, 5,
                         help="How many upcoming Gameweeks to include in player projections and transfer gains.")
    bank_m = st.number_input("Bank (£m)", min_value=0.0, max_value=20.0, value=0.7, step=0.1,
                             help="Added to the selling player's current public price to estimate an affordable replacement. Actual selling price may differ.")
    free_transfers = st.selectbox("Free transfers available", [1,2,3,4,5], index=0,
                                  help="Used in the decision guidance; the current shortlist compares one transfer at a time.")
    transfer_hit = st.checkbox("Allow a -4 hit if model sees a gain", value=False,
                                help="If enabled, subtracts four points from the projected gain before ranking. A hit should only be considered for a clear multi-week advantage.")
    risk = st.select_slider("Risk appetite", options=["Conservative","Moderate","Aggressive"], value="Moderate",
                             help="Conservative favors established minutes; Moderate balances projected gain and differential upside; Aggressive gives more weight to low-ownership upside.")
    st.caption("Squad is seeded from your latest screenshot. Verify names/prices below.")

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
    entry = get_json(f"entry/{int(team_id)}/")
except Exception:
    pass

c1,c2,c3,c4 = st.columns(4)
c1.metric("Next Gameweek", f"GW{gw}")
c2.metric("Public data players", len(players))
c3.metric("Recorded bank", f"£{bank_m:.1f}m")
c4.metric("Projection horizon", f"{horizon} GWs")
if entry:
    a,b,c = st.columns(3)
    a.metric("Latest public overall rank", f"{entry.get('summary_overall_rank', '—'):,}" if isinstance(entry.get('summary_overall_rank'),int) else "—")
    b.metric("Overall points", f"{entry.get('summary_overall_points','—')}")
    c.metric("Team value (public profile)", f"£{(entry.get('last_deadline_value') or 0)/10:.1f}m")
else:
    st.info("Could not load public manager profile right now. Your saved screenshot remains the manual baseline: 491 points, rank 581,139, India league rank 12,964.")

# Build name-resolved squad from saved baseline
resolved=[]
unmatched=[]
for row in RECORDED_SQUAD:
    p=resolve_player(row, players)
    if p:
        teamname=teams.get(p.get("team"),{}).get("short_name","?")
        pts,diff,nfx=estimate_points(p, teamname, fixtures, horizon)
        resolved.append({
            "Player": p.get("web_name"), "Position": pos_name(p,types), "Club": teamname,
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

st.subheader("Your recorded squad — current public player data")
if unmatched:
    st.warning("Names not uniquely matched to public data (review manually): " + ", ".join(unmatched))
if resolved:
    squad = pd.DataFrame(resolved)
    st.dataframe(squad.drop(columns=["id"]), use_container_width=True, hide_index=True)
    total_current_cost = squad["Price (£m)"].sum()
    st.caption(f"Sum of current public prices for matched players: £{total_current_cost:.1f}m. This is not necessarily your squad's purchase/selling value.")
else:
    st.error("No squad names could be matched. Update the name aliases in RECORDED_SQUAD.")

st.subheader("Transfer comparison")
st.write("Choose a player to sell and compare with an affordable replacement at the same position. Budget uses current public price as a rough estimate; actual FPL selling prices may differ.")
if resolved:
    df = pd.DataFrame(resolved)
    candidates = pd.DataFrame([{
        "Player": p.get("web_name"), "Position": pos_name(p,types),
        "Club": teams.get(p.get("team"),{}).get("short_name","?"),
        "Price (£m)": round((p.get("now_cost") or 0)/10,1),
        "Form": float(p.get("form") or 0), "PPG": float(p.get("points_per_game") or 0),
        "Minutes": int(p.get("minutes") or 0), "Status": p.get("status","?"),
        "Chance (%)": p.get("chance_of_playing_next_round"),
        f"Projected {horizon}GW pts": estimate_points(p, teams.get(p.get("team"),{}).get("short_name","?"), fixtures, horizon)[0],
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
    # Current squad selector
    squad_names = sorted(df["Player"].unique().tolist())
    sell_name = st.selectbox("Sell player", squad_names, index=(squad_names.index("Schade") if "Schade" in squad_names else 0))
    sell = df[df["Player"]==sell_name].iloc[0]
    max_price = float(sell["Price (£m)"]) + bank_m
    options = candidates[(candidates["Position"]==sell["Position"]) &
                         (~candidates["Player"].isin(squad_names)) &
                         (candidates["Price (£m)"] <= max_price)].copy()
    options["Raw gain"] = options[f"Projected {horizon}GW pts"] - float(sell[f"Proj GW{gw}–GW{gw+horizon-1}"])
    options["Hit cost"] = 4 if transfer_hit else 0
    options["Projected gain"] = options["Raw gain"] - options["Hit cost"]
    # Risk setting affects the ranking, not the raw point projection.
    ownership = options["Selected (%)"].fillna(0).astype(float)
    minutes_reliability = (options["Minutes"].astype(float) / 900.0).clip(0, 1)
    differential_bonus = (1.0 / (1.0 + ownership)) * (1.0 - ownership.clip(0,100) / 100.0)
    if risk == "Conservative":
        options["Decision score"] = options["Projected gain"] * (0.65 + 0.35 * minutes_reliability)
    elif risk == "Aggressive":
        options["Decision score"] = options["Projected gain"] + 1.5 * differential_bonus
    else:
        options["Decision score"] = options["Projected gain"] + 0.5 * differential_bonus
    options = options.sort_values(["Decision score","Projected gain","Minutes"], ascending=False).head(15)
    if options.empty:
        st.info("No affordable out-of-squad player candidates matched this position under the rough budget estimate.")
    else:
        st.caption(f"Approximate replacement ceiling: £{max_price:.1f}m, using current price + bank. Enter actual selling value separately in your final decision.")
        st.caption(f"Ranking mode: {risk}. Horizon: {horizon} GWs. Hit penalty: {4 if transfer_hit else 0} points.")
        st.dataframe(options[["Player","Club","Price (£m)","Form","PPG","Minutes","Status","Chance (%)","xG","xA","xG/90","xA/90",f"Projected {horizon}GW pts","Selected (%)","Projected gain","Decision score"]], use_container_width=True, hide_index=True)
        best=options.iloc[0]
        st.success(f"Top candidate under {risk.lower()} settings: {best['Player']} ({best['Club']}) — projected gain {best['Projected gain']:+.1f} points over {horizon} GWs. Decision score includes your risk preference; it is not a guaranteed return.")
        st.write("Decision rule: only transfer if the gain remains worthwhile after checking availability, actual selling price, expected minutes and the other 14 squad slots. A high projection alone is not a command to transfer.")

st.subheader("What the model does — and does not do")
st.markdown("""
- Blends PPG/form with a conservative xG+xA signal, shrinks small samples toward a position baseline, estimates minutes from starts and appearances, penalizes uncertain/unavailable status, and adjusts each scheduled fixture individually. Displays cumulative xG/xA and per-90 rates.
- Uses public player status and chance-of-playing fields where provided.
- Does **not** access your FPL password, private `my-team` endpoint, or your personal selling prices.
- This v1.3 is a transparent baseline, not a trained machine-learning model. It does not yet optimize the entire 15-player squad simultaneously, model price changes, or estimate rank probabilities.
- Public FPL API is unofficial/undocumented and may change. Data is cached for 30 minutes to reduce requests.
""")
st.caption(f"Last refreshed in this session: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Use the app's refresh/reload to request fresh data.")
