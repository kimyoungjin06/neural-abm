# Pivot-feedback D0 rank-4 production disposition

Medicine rank 4, T12505, entry 1993 is frozen as `CONTENT_FAIL` after the
independently replayed rank-4 local-routing result. The reason is
`RECENT_SAMPLE_MAJORITY_OPAQUE_INDEXING_ARTIFACTS`: 38 of the 64 recent titles
were opaque archival identifiers.

The cumulative production ledger therefore contains seven rows: three pass
and four fail. Frontier v5 retains the Computer Science field certificate but
does not issue a Medicine or global certificate. Medicine rank 5, T10151,
entry 1993 becomes the next required candidate; only a pass there could pair
with the already-frozen rank-6 pass.

No new title, API request, human label, deterministic reselection, outcome, or
model was opened by this production closeout. The indexing decision is not
semantic ground truth or an out-of-sample validation of candidate rule v2.
