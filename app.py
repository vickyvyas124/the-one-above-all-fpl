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
    # Try full name before a possibly ambiguous web_name alias. This prevents
    # a common surname/alias match from blocking a unique full-name match.
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

    # Final conservative fallback: only use a unique partial web-name match.
    partial = [p for p in players if alias and alias in norm(p.get("web_name", ""))]
    if len(partial) == 1:
        return partial[0]
    return None

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
.block-container {padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1600px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#101b35 0%,#182746 100%);}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color:#f3f6ff !important;}

/* Critical readability fix: keep editable values dark on white controls. */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] div,
[data-testid="stSidebar"] [data-baseweb="input"] input,
[data-testid="stSidebar"] [data-testid="stNumberInput"] input {
  color:#172033 !important;
  -webkit-text-fill-color:#172033 !important;
  background-color:#ffffff !important;
  opacity:1 !important;
}
[data-testid="stSidebar"] [data-testid="stNumberInput"] button,
[data-testid="stSidebar"] [data-baseweb="select"] svg {color:#172033 !important; fill:#172033 !important;}
[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {color:#f3f6ff !important;}
[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
[data-testid="stSidebar"] [data-testid="stCheckbox"] p {color:#f3f6ff !important;}
.hero {padding:1.35rem 1.5rem; border-radius:18px; background:linear-gradient(115deg,#101b35 0%,#244b78 58%,#2d8a78 100%); color:white; margin-bottom:1rem;}
.hero h1 {color:white; margin:0; font-size:2rem;}
.hero p {color:#dce9ff; margin:.45rem 0 0 0;}
div[data-testid="stMetric"] {padding:1rem; border:1px solid #e5eaf2; border-radius:14px; background:#fff; box-shadow:0 2px 8px rgba(16,27,53,.04);}
div[data-testid="stDataFrame"] {border:1px solid #e5eaf2; border-radius:12px; overflow:hidden;}
h2, h3 {letter-spacing:-.02em;}
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
            st.warning(
                "The move list is only provisional: one or more squad players are not matched to current public FPL records. "
                "Do not act on the top move until all 15 players are matched, because club-limit checks may be incomplete."
            )
            st.markdown(
                f"**Quick summary:** The model currently ranks **{top['Sell']} → {top['Buy']}** at an estimated **{top['Projected gain']:+.1f} points** over {horizon} Gameweeks, "
                f"but this is **not an actionable recommendation** until squad matching is complete. You have **{free_transfers} free transfer(s)** recorded."
            )
        elif top["Sell"] == "Haaland":
            st.warning(
                "Captaincy check required: Haaland is the captain in your last recorded lineup. "
                "This version does not fully model the captaincy opportunity cost of selling a premium captain, so do not execute this swap based on the projection alone."
            )
            if float(top["Projected gain"]) > 0:
                st.markdown(
                    f"**Quick summary:** The model ranks **{top['Sell']} → {top['Buy']}** at **{top['Projected gain']:+.1f} points** over {horizon} Gameweeks, "
                    f"but the captaincy trade-off is not priced into this recommendation. Verify captain options, current form, availability and your actual selling price before deciding."
                )
            else:
                st.markdown(
                    f"**Quick summary:** No positive-gain move is recommended on the current estimates. **Keep the captaincy implications in mind** before considering a premium forward sale."
                )
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
- This v1.8 is a transparent baseline, not a trained machine-learning model. It does not yet optimize the entire 15-player squad simultaneously, model price changes, or estimate rank probabilities.
- Public FPL API is unofficial/undocumented and may change. Data is cached for 30 minutes to reduce requests.
""")
st.caption(f"Last refreshed in this session: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Use the app's refresh/reload to request fresh data.")
