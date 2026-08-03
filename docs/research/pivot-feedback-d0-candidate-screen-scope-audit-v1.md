# Pivot-feedback D0 candidate-screen scope audit v1

## Why the scope audit was necessary

The v12 ledger authorized a zero-cost complete candidate-work manifest design.
Before copying work-level rows, this audit checked what the frozen 70,277-row
candidate table actually contains and whether the four-card title router can
operate on it.

The candidate rows are not works. Each is a field-topic-entry-year combination
with positive baseline attention and a larger recent attention window. The
four topic cards correspond only to the four provisionally selected episodes.

## Exact expansion

Each candidate has six pre-entry years: three baseline and three recent years.
The independent replay established:

| Quantity | Rows |
|---|---:|
| Candidate field-topic-entry-year rows | 70,277 |
| Distinct candidate topics | 994 |
| Candidate × six-year rows | 421,662 |
| Unique field-topic-publication-year cells | 99,433 |
| Distinct linked works represented by unique cells | 22,437,851 |
| Work memberships with overlapping candidates retained | 102,900,039 |

The 421,662-row candidate-window map and 994-row topic profile are private,
create-only Parquet artifacts. They contain counts, not titles or work IDs.

## Router coverage

The four cards cover 4 topics, 236 candidate rows, and 195,708 distinct linked
works. This is 0.4024% of candidate topics and 0.3358% of candidate rows.

Consequently, “apply the four-card router to the complete universe” is not an
executable instruction. Expanding all 22.4 million work rows first would not
repair the missing 990 topic semantics.

## Corrected execution strategy

A simpler successor should use deterministic lazy evaluation:

1. preserve the frozen candidate rank and compatibility rules;
2. inspect candidates in rank order;
3. obtain a frozen content disposition only for candidates that can still
   alter the selected pair;
4. skip candidates only with an explicit structural-incompatibility reason;
5. stop only when a selection-equivalence certificate proves that no
   unevaluated candidate can alter the pair; and
6. invalidate and resume the certificate if an earlier disposition changes.

The content layer should use deterministic metadata and local rules first.
Only ambiguous sampled titles enter the cross-provider API queue. This produces
the same selected pair as eager full-universe evaluation only if the
equivalence certificate is implemented and independently verified.

## Claim boundary

This audit is a workload and router-coverage result. It did not read the raw
OpenAlex main table, titles, work IDs, credentials, APIs, or outcomes. It does
not establish content validity, rule accuracy, a complete candidate screen,
reselection, a valid author panel, researcher pivots, model results, or a
scientific effect.
