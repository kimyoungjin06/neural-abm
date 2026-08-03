# S0 exact non-ranked coauthor graph profiles

Status:
`PASS_S0_EXACT_NONRANKED_COAUTHOR_GRAPH_PROFILES_WITH_CURRENT_SNAPSHOT_AUTHOR_TOKEN_LIMITATIONS`

The full CPU/RAM/NVMe run rehashed all 4,096 author-event artifacts and scanned
779,630,258 distinct author-work rows. For each of 18,544 primary-field by
publication-year cells it materialized all three predeclared, non-ranked
declared-team caps 10, 50, and 100, producing 55,632 compact profile rows.
Unknown publication year uses internal sentinel zero and is restored to null
in 78 output rows; all 78 have nonzero node support in the full run.

The cap-10/50/100 profiles contain 776,875,376 / 1,207,070,104 /
1,289,015,377 distinct undirected edges after repeated coauthorships are
deduplicated within field-year cells. Corresponding field-year-summed node
counts are 352,175,135 / 378,768,092 / 380,010,841. These are sums over cells,
not global unique people or one global graph.

The builder required materialized node and edge counts to equal the compact
sums at every cap, removed every raw author and edge scratch artifact before
completion, and wrote the receipt last. The independent verifier then
replayed degree-sum equality, nested cap margins, connected-component bounds,
largest-component shares, and all current hashes.

The unrestricted 47,627,187,584 declared pair-incidence profile remains a
diagnostic and was not edge-materialized. The finite caps are all reported;
none is selected or preferred. This component alone does not identify people,
historical collaboration, candidate eligibility, S0 completion, or an
empirical effect. The separate full ORCID/affiliation receipt and the
three-component S0 completion receipt are required before D0 can open.
