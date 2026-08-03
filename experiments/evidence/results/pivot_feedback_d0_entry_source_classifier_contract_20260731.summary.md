# Pivot-feedback D0 entry-source and classifier contract v1

Status:
`PASS_EXACT_FOUR_ENTRY_SOURCE_CLASSIFIER_CONTRACT_RECEIPTS_AND_FIXTURES`

The three threshold-free pivot inputs now have exact receipt-bound sources.
The pre-entry count comes from the verified predecision author measurement,
the trailing count uses distinct eligible target-topic works in `t-3..t-1`,
and the focal count uses the same work and author-linkage ABI in entry year
`t`.

`content-confirmed` is fixed at the episode-topic level. Each of the four
selected topics already has a frozen `CONTENT_PASS` global certificate, so no
new OpenAI, Claude, OpenAlex, human, or other individual-work review is
required. The count still requires an exact current-snapshot primary-topic and
field match. Neither provider agreement nor that current-snapshot annotation
is treated as work-level semantic ground truth or historical observability.

The independent verifier validated 10 repository bindings and five
operator-receipt hashes and statuses. It opened zero operator Parquet files,
read zero production rows, created zero API requests, and opened no outcome.

A 13-work, 13-event fixture reconstructed five registered author rows and all
four states. Duplicate events, all specified exclusion categories, a
zero-valid-author-token work, and retained null retraction were accounted for
with zero row or accounting mismatch.

This result completes the source-and-classifier contract, not the production
source table or production pivot labels. A successor gate is required before
reading production rows.
