# Pivot-feedback D0 production candidate rule and frontier v2

Status:
`PASS_INDEPENDENT_PRODUCTION_CANDIDATE_RULE_AND_FRONTIER_V2`

The simple candidate rule was accepted without changing its predeclared
thresholds. It froze three `CONTENT_PASS` dispositions and one
`CONTENT_FAIL`: T12505 fails specifically for the joint score/source indexing
anomaly.

The 51-row adaptive frontier now contains three passes, one fail, one
unresolved row, and 46 structurally unevaluated rows. Computer Science ranks 1
and 45 receive a field-level certificate. Medicine rank 1 fails, so rank 2
T10543/1993 becomes the only next required candidate; rank 6 T12421 is already
a later compatible pass under a rank-2-pass branch.

No global certificate or deterministic reselection was issued. The component
read no new titles or APIs. Its only successor is an exact local 96-title
T10543 sample, with 32 deterministic titles in each of the earlier, baseline,
and signal windows.
