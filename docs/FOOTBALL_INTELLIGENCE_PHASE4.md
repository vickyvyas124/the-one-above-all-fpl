# Independent football intelligence: Phase 4

## Dashboard surface

The Football Intelligence tab is a read-only view over the optional local store at `.local/football-intelligence` (or the path explicitly set through `FPL_INTELLIGENCE_STORE`). It shows collection status, analyzed fixture/player counts, measured-data coverage, evidence-linked trends, exploratory shortlist entries, and data-quality warnings. The view is cached briefly for dashboard responsiveness. It does not contact a data source, write to the store, expose raw files, or change FPL projections, transfers, captaincy, or HOLD decisions.

Hosted deployments without a local store show an explicit empty state. They do not silently populate sample data. Invalid stores and records are reported and excluded from analysis. The loader bounds the number and size of manifests and normalized fixture files it reads, validates normalized observations and their fixture identity, and rejects paths that escape the configured store. Raw files are checked for presence and a well-formed recorded digest; their contents are not re-downloaded or re-hashed while rendering the dashboard.

Collection status and successful source retrieval time come from local collection manifests. “Analysis generated” is the dashboard render time, not a source update time. StatsBomb Open Data coverage is selected historical competition coverage; it is not a complete, current Premier League feed. The exploratory shortlist is evidence context only and is not a transfer recommendation.

## Scheduling decision

No scheduled collector is added in this phase. The implemented source does not provide complete current Premier League coverage or a deadline-aligned historical feed, and scheduling it would create an appearance of current coverage without supporting evidence. The app's FPL background worker and recommendation paths remain unchanged.

Before adding a scheduled source, select a source with documented rights for the intended collection and display, confirm its competition/season coverage, freshness and stable player/fixture identifiers, define an update cadence and failure policy, and add validation for deadline-safe timestamps and provenance. Any future collector must preserve immutable source data and make freshness, partial coverage and failures visible. Review the source's attribution and usage terms before public display or redistribution.

## Attribution and sharing

The dashboard names StatsBomb Open Data and links to its public repository. This is attribution context, not a grant of rights for every use. Before publicly sharing derived output, review the current source terms and attribution/branding requirements. Raw source files remain local and are not displayed or published by this dashboard.

