# Pivot-feedback certified predecision author measurement

Status:
`PASS_INDEPENDENT_EXACT_FOUR_AUTHOR_MEMBERSHIP_CONSOLIDATED_TEMPORAL_LEAKAGE_AUDIT`

The registered measurement population contains all 3,256,271 independently
verified author-episode rows. Membership is the exact verified
author-opportunity base: its upstream ABI already requires at least one
eligible focal-field work in `t-3..t-1`. No additional minimum-history
threshold or threshold-specific membership flag was selected.

The operator-local consolidated Parquet has 74 columns:

| Availability class | Columns |
|---|---:|
| Private operator-local identifiers | 5 |
| Predecision, through `t-1` | 51 |
| Registered outcome-blind membership | 1 |
| Current-snapshot ORCID sensitivity | 5 |
| Current-snapshot work-time affiliation proxy | 12 |

The independent verifier used PyArrow sorting and exact column-array
comparison, not the producer's DuckDB join. It recovered all four component
views with zero key or value differences. All 74 columns were classified,
forbidden outcome columns were absent, the first target-topic year never
exceeded `t-1`, and all ten non-ranked cap monotonicity families had zero
violations.

ORCID remains a current-snapshot sensitivity only. Same-affiliation remains a
secondary institution-ID proxy on predecision shared works with explicit
missingness. The full artifact and raw numeric IDs remain operator-local.

This pass permits a successor gate to decide whether the registered
predecision measurement surface is complete. It does not itself authorize
outcome opening, pivot classification, final eligibility, model fitting, G4,
or a scientific-effect claim.
