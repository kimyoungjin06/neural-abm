# Certified D0 leave-out and prior-peer replay

Status: `PASS_INDEPENDENT_EXACT_FOUR_OUTCOME_BLIND_INCUMBENT_FOCAL_AND_PRIOR_TARGET_PEER_LEAVEOUT`

The exact-four normalized component is complete and independently verified.
It contains 30,765 target works, 563,486 distinct candidate–prior-target-peer
edges, and 3,256,271 author-episode measurement rows.

| Cap | Authors with a prior target-active peer | Network any-incumbent exclusions |
|---:|---:|---:|
| 10 | 209,783 | 191,961 |
| 50 | 225,987 | 269,392 |
| 100 | 228,607 | 284,667 |

The independent DuckDB replay reconstructed target incumbency from the full
normalized author-event source, rebuilt edges by finding target-peer prior
works before reverse-joining candidates, and recalculated every author
leave-out column. Target, edge, and all three cap projections had zero
bidirectional row differences, with zero leave-out monotonicity violations.

No cap is selected. Raw authorship, ORCID stability, same-affiliation,
visible-success, pivot classification, entry or persistence outcomes, final
eligibility, model fitting, and G4 remain unopened. The only eligible successor
is a separate receipt-bound raw-authorship identity and work-time institution
proxy component.
