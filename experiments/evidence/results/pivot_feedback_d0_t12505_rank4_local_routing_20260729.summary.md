# T12505 rank 4 local routing

Status: `PASS_INDEPENDENT_T12505_RANK4_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN`

- The exact normalized replay selected 32 titles in each of three windows.
- All 96 selected IDs had titles in the frozen current snapshot.
- The recent 64 titles route to 12 accept, 3 reject, 11 ambiguous, and 38
  opaque quarantine rows.
- The opaque fraction is `0.59375`, so the production-rule-v2
  majority-opaque branch fires first.
- Median topic score falls by `0.95375`, while dominant-source share rises by
  `0.46875`.
- The dry-run state is `WOULD_CONTENT_FAIL_INDEXING_ANOMALY`.
- No human review, API request, production disposition, certificate, outcome,
  or model was opened.

This is an indexing-quality dry run and not semantic ground truth or
out-of-sample validation of candidate rule v2.
