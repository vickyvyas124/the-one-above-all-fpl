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
