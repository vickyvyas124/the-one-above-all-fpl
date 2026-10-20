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
    # Transparent baseline model, not an official prediction.
    # Points-per-game and form are blended, then lightly adjusted for fixture difficulty.
    ppg = float(p.get("points_per_game") or 0)
    form = float(p.get("form") or 0)
    minutes = float(p.get("minutes") or 0)
    mins_factor = min(1.0, minutes / max(1, (int(p.get("starts",0) or 0) * 80))) if p.get("starts") else min(1.0, minutes/450)
    if minutes == 0:
        mins_factor = 0.55
    base = 0.6 * ppg + 0.4 * form
    if base <= 0:
        base = max(1.0, float(p.get("total_points") or 0) / max(1, int(p.get("starts",0) or 1)))
    # Expected upcoming fixture difficulty average. FPL difficulty runs 1-5; neutral is 3.
    upcoming = [f for f in fixtures if not f.get("finished") and (f.get("team_h") == p.get("team") or f.get("team_a") == p.get("team"))]
    upcoming.sort(key=lambda f: f.get("event") or 99)
    upcoming = [f for f in upcoming if f.get("event") is not None][:horizon]
    if upcoming:
        diffs=[]
        for f in upcoming:
            home = f.get("team_h") == p.get("team")
            diffs.append(float(f.get("team_h_difficulty" if home else "team_a_difficulty") or 3))
        avg_diff=sum(diffs)/len(diffs)
        fixture_factor=max(0.78, min(1.22, 1.0 + (3.0-avg_diff)*0.07))
    else:
        avg_diff=3.0
        fixture_factor=1.0
    # Forecast per match, scaled by likely minutes and horizon; intentionally conservative.
    per_match=max(0.0, base) * max(0.45, mins_factor) * fixture_factor
    return round(per_match * len(upcoming or range(horizon)), 1), round(avg_diff, 2), len(upcoming)

st.title("⚽ The-One-Above-All — FPL Optimizer")
st.caption("Free prototype • Team ID 2311669 • GW8 baseline recorded from your screenshots")
st.warning("Forecasts are transparent estimates, not official FPL projections. Public endpoints may change; private squad data is not accessed. Confirm injuries and your actual selling prices before making a transfer.")

with st.sidebar:
    st.header("Optimizer settings")
    team_id = st.number_input("FPL Team ID", min_value=1, value=2311669, step=1)
    horizon = st.slider("Projection horizon (Gameweeks)", 1, 8, 5)
    bank_m = st.number_input("Bank (£m)", min_value=0.0, max_value=20.0, value=0.7, step=0.1)
    free_transfers = st.selectbox("Free transfers available", [1,2,3,4,5], index=0)
    transfer_hit = st.checkbox("Allow a -4 hit if model sees a gain", value=False)
    risk = st.select_slider("Risk appetite", options=["Conservative","Moderate","Aggressive"], value="Moderate")
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
    options["Projected gain"] = options[f"Projected {horizon}GW pts"] - float(sell[f"Proj GW{gw}–GW{gw+horizon-1}"])
    if not transfer_hit:
        # Show all legal one-transfer options; transfer count is accounted for in recommendation text.
        pass
    options = options.sort_values(["Projected gain","Minutes"], ascending=False).head(15)
    if options.empty:
        st.info("No affordable out-of-squad player candidates matched this position under the rough budget estimate.")
    else:
        st.caption(f"Approximate replacement ceiling: £{max_price:.1f}m, using current price + bank. Enter actual selling value separately in your final decision.")
        st.dataframe(options[["Player","Club","Price (£m)","Form","PPG","Minutes","Status","Chance (%)",f"Projected {horizon}GW pts","Selected (%)","Projected gain"]], use_container_width=True, hide_index=True)
        best=options.iloc[0]
        st.success(f"Model's top rough candidate for {sell_name}: {best['Player']} ({best['Club']}) — projected gain {best['Projected gain']:+.1f} points over {horizon} GWs before uncertainty/transfer-hit adjustments.")
        st.write("Decision rule: only transfer if the gain remains worthwhile after checking availability, actual selling price, expected minutes and the other 14 squad slots. A high projection alone is not a command to transfer.")

st.subheader("What the model does — and does not do")
st.markdown("""
- Blends season points-per-game and current form (60/40), adjusts for expected minutes and average FPL fixture difficulty, and projects over the selected horizon.
- Uses public player status and chance-of-playing fields where provided.
- Does **not** access your FPL password, private `my-team` endpoint, or your personal selling prices.
- This v1 is a transparent baseline, not a trained machine-learning model. It does not yet optimize the entire 15-player squad simultaneously, model price changes, or estimate rank probabilities.
- Public FPL API is unofficial/undocumented and may change. Data is cached for 30 minutes to reduce requests.
""")
st.caption(f"Last refreshed in this session: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Use the app's refresh/reload to request fresh data.")
