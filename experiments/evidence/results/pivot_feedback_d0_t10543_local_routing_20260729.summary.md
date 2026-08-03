# T10543 local routing and candidate-v1 dry-run

Status: `PASS_INDEPENDENT_T10543_LOCAL_ROUTING_AND_CANDIDATE_V1_DRY_RUN`

- Exact current-snapshot sample: 96 titles, 32 per window.
- Recent local routes: 8 accept, 1 reject, 3 informative ambiguous, and
  52 opaque quarantined.
- The 52 opaque rows are 81.25% of the recent sample. All come from
  `Digital Kenyon (Kenyon College)` and have a median current-snapshot primary
  topic score of 0.03645.
- The unchanged candidate rule v1 returns `WOULD_CONTENT_UNRESOLVED`.
- No opaque identifier was sent to an API. No API request, production
  disposition, selection certificate, reselection, outcome read, or model fit
  occurred.

The unresolved result is a rule-coverage finding, not ordinary semantic
ambiguity. Candidate rule v1 only treats a joint baseline-to-signal score drop
and source-share increase as a decisive indexing anomaly. Here the baseline is
already contaminated, so its median topic score is already low and the
required score drop does not occur.

This is current-snapshot indexing-quality evidence, not semantic ground truth,
a historical topic claim, a pivot classification, or a scientific effect.
