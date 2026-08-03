# T10151 rank 5 local routing

Status: `PASS_INDEPENDENT_T10151_RANK5_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN`

- The topic card was frozen before the T10151 title sample was read.
- The producer and verifier replayed 492,293,206 work rows and 779,630,258
  author-event rows and selected the same 96 IDs.
- All 96 selected titles were found.
- Among 64 recent titles, 27 were accepted, 12 rejected, 25 ambiguous, and 0
  opaque.
- The accept and reject lower bounds are both below `0.50`.
- Indexing diagnostics do not trigger either anomaly branch.
- The dry-run state is `WOULD_CONTENT_UNRESOLVED`.
- No API request, human label, production disposition, certificate, outcome,
  or model was opened.

`CONTENT_UNRESOLVED` is neither pass nor fail. The local card is a conservative
current-snapshot routing heuristic, not semantic ground truth.
