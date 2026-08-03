# S0 threshold-free core inventory

Status:
`PASS_S0_THRESHOLD_FREE_CORE_FIELD_YEAR_INVENTORY_WITH_ORCID_AFFILIATION_AND_GRAPH_TOPOLOGY_PENDING`

The full normalized scan rehashed 8,192 selected input partitions
(9,127,313,828 bytes), then aggregated 492,293,206 canonical work rows and
779,630,258 distinct author-work event rows into 18,544 field-year cells over
all 26 fields. The independently verified attention surface contains 18,518
non-null-year cells at annual, trailing-three-year, and trailing-five-year
resolutions.

The classified-field surface contains 354,145,248 works, of which 329,740,704
have usable canonical dates. It contains 675,475,922 distinct author-work
pairs, of which 673,318,978 are history-usable. The source still contains
138,147,958 works without a primary field and 27,565,532 works without a
publication year; the observed non-null year range is 1000–2050 and therefore
includes known temporal anomalies rather than silently filtering them.

This is a core-component pass, not an S0 gate pass. ORCID/token stability,
author-work affiliation, and distinct coauthor graph topology remain
`NOT_RUN`. D0 and model fitting are not authorized, and no empirical or
scientific effect is established.
