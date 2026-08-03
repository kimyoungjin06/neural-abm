# T10543 Medicine rank-3 local routing

Medicine rank 3 reuses topic T10543 but changes the candidate entry year to
1986, so it requires a distinct entry-relative sample. The normalized replay
found 657 baseline works in 1980–1982 and 3,988 signal works in 1983–1985.

The sample is substantially interpretable in baseline but is dominated by
Digital Kenyon archival identifiers in signal. The baseline primary-topic
score median is 0.97865 and the signal median is 0.04635. The dominant-source
share rises from 0.3125 to 0.84375.

Because fewer than half of the recent titles are opaque, the new v2
majority-opaque branch does not fire. The retained v1 joint score/source
anomaly branch does fire by a wide margin, producing
`WOULD_CONTENT_FAIL_INDEXING_ANOMALY` without an API call.

A successor gate is still required before converting the dry-run state into a
production disposition and regenerating the adaptive frontier.
