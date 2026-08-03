# Candidate rule v2 retrospective replay

Status: `PASS_INDEPENDENT_CANDIDATE_RULE_V2_FIVE_CANDIDATE_REPLAY`

The disclosed post-diagnostic amendment adds one first-precedence branch:
if at least half of the 64 recent sampled titles are quarantined as opaque
archival identifiers, the candidate dry-run state is
`WOULD_CONTENT_FAIL_INDEXING_ANOMALY`.

The branch uses neither source name, primary-topic score, API labels, nor human
review. Across the five already sampled candidates:

- 4 candidate states are unchanged.
- T10543 changes from `WOULD_CONTENT_UNRESOLVED` to
  `WOULD_CONTENT_FAIL_INDEXING_ANOMALY`.
- The v2 replay totals are 3 pass, 2 indexing fail, and 0 unresolved.

The amendment was designed after observing the T10543 rule-v1 gap, so this is
not out-of-sample validation. No new title, API request, production freeze,
production disposition, selection certificate, outcome, or model was opened.
