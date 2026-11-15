# Independent football intelligence: Phase 2

## Scope

Phase 2 adds offline indicators over versioned player-match observations. It does not feed FPL points, transfers, captaincy, the dashboard, workflows, or training. Those are Phase 3/4 work and require separate point-in-time validation.

## Indicators

The analyzer compares the latest up to three matches with the preceding up to three matches for the same source player, season, and club. It requires at least two measurable matches in each period. The default comparison therefore needs at least four total rows; with five rows it uses three recent and two baseline rows.

It reports these measured trends when both periods have sufficient data:

- minutes per match and starting share;
- change in match-to-match minutes standard deviation (participation variation);
- xG, shots, and shots on target per 90;
- xA, key passes, and chances created per 90;
- event-location proxies for attacking-area and box involvement per 90;
- defensive actions per 90;
- a change in the source's dominant listed position, when labels are available in both periods.

Per-90 values divide the source match total by measured minutes and multiply by 90. A zero-minute row cannot produce a per-90 value. Unknown metrics are omitted from the comparison and remain visible in the result's coverage ratios; the engine never fills missing values with zero.

Each trend is classified as INFERRED and links to the source player and all fixtures used. The explanation states the comparison and cautions that it is not a causal or future-performance claim. Position-label changes are reported as lineup labels and do not establish a tactical-role change. Evidence quality is a sample-coverage tier: HIGH means both configured periods have all configured matches measured; MEDIUM means the minimum sample is met but the windows are incomplete. It is not a statistical confidence probability.

## Emerging opportunities

The exploratory shortlist requires increases in at least two separate attacking domains: goal threat, chance creation, and box involvement. Correlated metrics inside the same domain (for example xG and shots) cannot by themselves satisfy the two-domain rule. Candidates are ordered by number of improving domains, then player name and stable source ID. No arbitrary weighted score is produced.

This is a review queue, not an FPL recommendation. The phase does not infer a player's sustainable points, compare event data with FPL returns, estimate start probabilities, detect fatigue from minutes variation, or assert manager intent. A candidate must not influence an actionable transfer until Phase 3 adds the verified squad/budget source, time-safe FPL comparison, and baseline-versus-adjusted validation.

## Data and limitations

The analyzer consumes the Phase 1 observation contract and its source provenance. It does not fetch data itself. Current StatsBomb Open Data coverage is selected and historical, not a complete current Premier League feed; FPL Gameweek remains unknown unless separately verified. Identity mapping conflicts and duplicate player-fixture rows stop analysis. Club changes are separated into different comparison groups.

Offline tests cover trend calculations, evidence classification and references, role-label handling, participation-variation reporting, missingness and coverage, minimum sample gates, duplicate/mapping failures, and the multi-domain shortlist rule. No model weights, prediction changes, automated match discovery, dashboard widgets, or external network tests are included.
