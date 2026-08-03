# Pivot-feedback cutoff-primitives production summary

Status: `PASS_NORMALIZED_CURRENT_SNAPSHOT_CUTOFF_PRIMITIVES_WITH_RETROSPECTIVE_TOPIC_AND_CITATION_PROXY_LIMITATIONS`

The full CPU/I/O production run projected all 493 main-work files, reproduced
492,293,207 raw rows, quarantined the frozen single `W-1` row, and closed all
492,293,206 retained work IDs to the receipt-bound canonical time spine. The
result is a 4,096-partition, 4.12 GB normalized work-metadata surface with tree
SHA-256 `a37448b4df39c00e97b540b3ac7d06de6f2e9ce12186838120d59c820aa89d77`.

The independent verifier rehashed all 4,101 output artifacts, rescanned every
work row, replayed every canonical join, rehashed the adjudicated topic
relation, verified the citation partition tree, and confirmed that the output
tree was unchanged. The production receipt SHA-256 is
`91e3f689e574d524eabbe7daa58f5e678da37f851ba9b7af0561dab57b70846e`;
the independent verification report SHA-256 is
`703d98eeddb5266cef507ef00400c5cb3700958bf3a1744fc16b6ade2121d485`.

This is a normalized primitives pass, not the final analytic panel. Topic
membership and citation event time remain retrospective current-snapshot
proxies. Candidate eligibility, work-type inclusion, team caps, same-day
treatment, fixed outcome horizon, thresholds, sampling, and weights remain for
a versioned G0 successor. The G-1 registry may add this as completed foundation
evidence, but `cutoff_panel` remains a blocker until that successor is frozen
and verified.
