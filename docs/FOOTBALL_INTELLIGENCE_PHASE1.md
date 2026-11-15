# Independent football intelligence: Phase 1

## Scope and audit basis

This phase adds a separate evidence-ingestion foundation. It does not change FPL points projections, training, transfer rankings, lineup selection, the dashboard, or scheduled workflows. No FPL recommendation consumes these observations yet. The existing forecast model and its point-in-time snapshot path remain the source for FPL projections.

The code audit was performed against main after PR #71 (merge commit 3dcd0c4). Existing relevant components:

| Area | Existing behavior and boundary |
|---|---|
| Forecasting | fpl_engine.py uses the frozen v3.0 trained artifact when present and a heuristic fallback otherwise. Inputs are FPL history, current FPL status, and fixtures. |
| Training | fpl_learning.py fits a ridge model from lagged FPL features. It does not ingest event-level football intelligence or have candidate-versus-production promotion gates. |
| Point-in-time storage | fpl_snapshot_store.py, fpl_background_worker.py, and fpl_snapshot_publisher.py capture forecast snapshots, reconcile actuals, and publish guarded FPL artifacts. They do not maintain a match-event source ledger. |
| Squad and decisions | fpl_squad.py, fpl_account.py, fpl_transfer.py, and fpl_lineup.py provide the current verified squad, budget checks, transfer search, and formation/XI logic. Phase 1 does not duplicate or replace them. |
| Availability and validation | fpl_availability.py is a shared heuristic, not a calibrated start-probability model. historical_backtest.py is a historical proxy, not an exact point-in-time replay. |
| Current historical inputs | The existing backtest references the Vaastav Fantasy-Premier-League archive and a separate Premier-League-Stats archive. Their repo metadata is respectively NOASSERTION/other and no declared license. This phase does not broaden, copy, or train on those datasets; obtain rights review before changing their use. |

## Source review

| Source | Access, coverage, and limits | Phase 1 decision |
|---|---|---|
| [StatsBomb Open Data](https://github.com/hudl/open-data) | GitHub-hosted JSON for selected competitions and seasons, including lineups and events. It is not a current-season Premier League feed and has no FPL player-ID crosswalk. Its README permits research/genuine-interest use and asks publishers to credit StatsBomb and use its logo. It does not grant a general commercial license. | Implement a bounded, explicit adapter for selected completed matches. Require users to select competition/season/match; do not crawl the full repository. Persist local raw bytes with provenance and checksums. Any published analysis must follow source attribution/logo requirements and be rights-reviewed for its use. |
| [Premier League terms](https://www.premierleague.com/en/terms-and-conditions) and existing FPL endpoints | Website/app content is protected; the terms restrict reuse and database building without written permission. The FPL endpoint used by the current app is not a documented public data license. | Do not expand ingestion or redistribute FPL data in this phase. Keep the existing FPL account/snapshot path unchanged and request rights review before combining or republishing source data. |
| [football-data.co.uk usage policy](https://football-data.co.uk/contact.php) | Its free CSV policy is for private/personal/academic use, limits download frequency, and excludes commercial use and heavy automated model-training use. It primarily offers match results/statistics/odds rather than player-event metrics. | Excluded from ingestion and model training. |
| [football-data.org API](https://www.football-data.org/documentation/api), [pricing](https://www.football-data.org/pricing), [coverage](https://www.football-data.org/coverage) | Free-plan Premier League fixtures/results/table data, API key required, rate-limited, and no event/player-level advanced metrics. | Not integrated: duplicates much of the current fixture feed and does not meet the player-level intelligence need. Re-evaluate only if a later feature needs its coverage and registered-key terms are accepted. |
| [OpenFootball England](https://github.com/openfootball/england) | Public-domain/CC0 fixture and result data with broad season coverage, but no player-level event statistics. | Not integrated: insufficient for the intended intelligence signals and overlaps existing fixture data. |
| Unofficial Understat/FBref scrapers and similar sites | No cleared API or permission for automated collection/model-training use was established in this review. | Do not scrape or use without explicit licensing/access approval. |

No paid API, registration key, recurring job, or external account is introduced. The adapter uses a bounded request timeout and response-size limit; it requests only the selected match's match list, event file, and lineup file. It does not fetch source data during tests.

## Contract and data quality

fpl_intelligence.contract defines versioned player-match observations and a future insight shape. A source observation keeps:

- stable source player and fixture IDs, source URL, source timestamp, retrieval timestamp, and SHA-256 of the raw bundle;
- optional FPL ID only when supplied through an explicit mapping; names are never used to guess identity;
- FPL Gameweek as unknown unless a verified mapping is later supplied; competition match week is stored separately;
- measured absence as numeric zero, while unavailable metrics remain null and are named in missing_fields;
- metric definitions, role segments, and evidence quality.

The StatsBomb adapter fails closed on malformed payloads and duplicate match, event, or lineup-player IDs. It derives event-location proxies only with explicit definitions. Progressive carry/pass values remain unknown until attack-direction normalization is independently validated. The adapter does not claim complete tracking/touch coverage or cross-provider metric equivalence.

Collection state distinguishes not completed, completed but unavailable, incomplete, successful collection, and failed analysis. A missing event or lineup document never becomes a row of zero-valued player stats.

## Storage and operation

fpl_intelligence.storage writes gzipped source bytes by SHA-256, normalized JSONL revisions, and manifests with source timestamps, retrieval times, URLs, counts, and warnings. Writes are atomic and immutable; the source body hash is checked before persistence. The default local location is .local/football-intelligence/, ignored by Git. The adapter and store are not called by the dashboard, worker, or workflows in this phase.

To try the adapter in a local research script, import StatsBombOpenDataAdapter, choose a competition/season/match listed in the source catalogue, then pass the returned bundle through collect_match. Do not commit the local store or publish raw payloads.

## Validation and next gates

Offline tests cover normalization, explicit ID mapping, unknown Gameweek, incomplete inputs, duplicate IDs, and immutable storage. There is no live-source test. Before any model integration:

1. verify rights, attribution, permitted storage, and publication rules for the intended deployment;
2. curate an explicit source/player crosswalk with provenance and no name-only joins;
3. validate metric definitions, event coverage, and source completeness against sampled matches;
4. evaluate point-in-time availability and leakage risk before model training;
5. run a separately versioned candidate backtest against the frozen production forecast, with promotion and rollback gates;
6. only then decide whether an insight should affect FPL projections or recommendations.

Phase 1 intentionally implements none of those later model, dashboard, training, or recommendation changes.
