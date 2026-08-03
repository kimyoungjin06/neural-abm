# T12505 rank 4 local routing

Medicine rank 4, T12505, entry 1993 was evaluated with an exact
entry-relative 96-title sample. The normalized producer and independent
verifier each replayed all 492,293,206 work rows and 779,630,258 author-event
rows, then verified the same 32-per-window selected IDs. All 96 titles were
found.

Among the 64 recent titles, 38 are opaque archival identifiers, 12 match the
frozen T12505 positive card, 3 match only negative patterns, and 11 remain
informative but ambiguous. The opaque fraction `0.59375` crosses candidate
rule v2's majority threshold. The retained joint anomaly is also present:
median primary-topic score falls from `0.99330` to `0.03955`, while the
dominant-source share rises from `0.37500` to `0.84375`.

The dry-run result is therefore `WOULD_CONTENT_FAIL_INDEXING_ANOMALY`, with
the majority-opaque branch taking precedence. No human or API label was used,
and no production disposition, certificate, outcome, fit, or scientific
effect is created by this step.
