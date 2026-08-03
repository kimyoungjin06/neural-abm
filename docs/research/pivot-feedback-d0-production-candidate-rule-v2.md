# Production candidate rule v2

Production rule v2 preserves every v1 disposition and adds one disclosed
post-diagnostic branch: a candidate fails the content screen as an indexing
anomaly when at least half of its 64 recent sampled titles are opaque archival
identifiers.

The resulting five dispositions are:

- Computer Science rank 1: pass.
- Computer Science rank 45: pass.
- Medicine rank 1: fail under the retained joint score/source anomaly branch.
- Medicine rank 2 T10543: fail under the majority-opaque branch.
- Medicine rank 6: pass.

Frontier v3 does not skip ahead to a certificate. Because Medicine ranks 1 and
2 fail, rank 3 becomes the next required first candidate. Rank 3 shares T10543
but has entry year 1986, so it needs its own entry-relative 96-title sample.
If rank 3 passes, the already-frozen rank 6 pass is the first later compatible
candidate and the Medicine pair would be `(3, 6)`. If rank 3 fails or remains
unresolved, the frontier must expand again.

No global certificate or deterministic episode reselection is valid until the
lazy state machine closes.
