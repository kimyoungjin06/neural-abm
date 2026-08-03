# T10543 local routing v1

The exact 96-title sample for Medicine rank 2, topic T10543, entry year 1993
was routed locally before any API decision. The independently replayed recent
sample contains 8 conservative positive matches, 1 explicit negative match,
3 informative ambiguous titles, and 52 opaque archival identifiers.

The opaque rows are not useful API inputs. They are identifiers such as
`PVN-LOT-*`, `PVN-RDO-*`, or `PSB-LOT-*`, not semantic titles. All 52 recent
opaque rows come from Digital Kenyon and their median current-snapshot primary
topic score is 0.03645. Sending them to Claude or OpenAI would add cost without
adding title evidence.

The frozen candidate rule v1 remains unresolved because quarantined rows count
as neither accept nor reject. Its existing indexing-anomaly branch also
requires both a baseline-to-signal topic-score drop of at least 0.15 and a
dominant-source-share increase of at least 0.25. Source concentration rises,
but the score drop is only 0.0095 because contamination is already present in
the baseline window.

This exposes a simple rule gap: a candidate sample can be unusable because
opaque indexing contamination is already persistent at baseline. Any successor
rule must be versioned as a post-diagnostic amendment, replayed across every
currently sampled candidate, and kept separate from semantic ground-truth or
scientific claims.
