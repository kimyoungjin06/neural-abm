# Certified raw-authorship identity and affiliation replay

Status: `PASS_INDEPENDENT_EXACT_FOUR_RAW_AUTHORSHIP_ORCID_SENSITIVITY_AND_WORK_TIME_INSTITUTION_PROXY`

The exact-four raw-authorship component is complete and independently
verified. The full 1,361,551,038-row authorship surface was replayed from
1,363 source files with exact source stats unchanged before and after.

Among candidate author tokens, 713,195 have exactly one checksum-valid ORCID;
none have multiple checksum-valid ORCIDs. This remains a current-snapshot
OpenAlex-token sensitivity flag, not natural-person identity.

The receipt-bound institution relation contains 635,155 distinct relevant
author–work–institution rows. Of 563,486 prior target-active peer edges,
309,956 have a shared prior work where both authors have institution data and
211,192 share at least one identical institution ID on such a work.

| Cap | Authors with comparable peer | Authors with same-affiliation bridge |
|---:|---:|---:|
| 10 | 133,664 | 106,551 |
| 50 | 143,628 | 112,889 |
| 100 | 144,967 | 113,234 |

The independent verifier used pair-first author–ORCID aggregation and a
peer-work-first normalized replay. Identity, institution relation, affiliation
edges, and all three author cap projections had zero bidirectional row
differences and zero affiliation monotonicity violations.

No cap is selected. Same-affiliation remains a secondary current-snapshot
work-time proxy with missing institution coverage explicit. Visible success,
pivot classification, outcomes, final eligibility, model fitting, and G4
remain unopened.
