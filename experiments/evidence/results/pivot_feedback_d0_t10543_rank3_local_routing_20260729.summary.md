# T10543 Medicine rank-3 local routing

Status: `PASS_INDEPENDENT_T10543_RANK3_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN`

- The exact 96-title sample and all selected IDs were independently replayed.
- The six-year recent author-linked pool is 4,645 works.
- Recent routes are 21 accept, 0 reject, 16 ambiguous, and 27 opaque
  quarantine.
- The majority-opaque v2 branch does not fire because 27/64 is 42.1875%.
- The retained joint anomaly branch does fire: topic-score median drops by
  0.9323 and dominant-source share rises by 0.53125.
- The rule-v2 dry-run state is
  `WOULD_CONTENT_FAIL_INDEXING_ANOMALY`.
- No API request, production disposition, selection certificate, outcome, or
  model was opened.

The result is an indexing-quality screen on the frozen current snapshot, not
semantic ground truth or a scientific claim.
