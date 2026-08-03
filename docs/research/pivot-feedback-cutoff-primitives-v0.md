# Pivot-feedback normalized cutoff primitives v0

## Outcome

This successor turns the completed canonical-time, author-history,
work-topic, and citation-chronology components into one deterministic
relational interface for annual cutoff views. It does **not** freeze the final
analytic population or any scientific threshold.

The central design choice is normalization. A literal author-work-topic table
would repeat the same work metadata once per author-topic combination and
materialize billions of redundant rows. The v0 component instead binds five
lossless surfaces:

```text
                         entry year t
                              |
                    cutoff = Dec 31(t-1)
                              |
      +-----------------------+------------------------+
      |                       |                        |
 work_metadata         author/events          citation-safe edges
 work mod 4096         author mod 4096         cited-work mod 4096
      |                       |                        |
      +----------- work_id --+--- work_id ------------+
                              |
               adjudicated work-topic relation
                    sorted (work, topic)
                              |
             deterministic normalized cutoff view
                              |
       G0 successor chooses eligibility, horizons,
       thresholds, sampling, weights, and estimators
```

The normal form is information-equivalent to the corresponding joined table
while keeping each parent receipt and limitation visible.

## New production surface

`work_metadata/part-0000.parquet` through `part-4095.parquet` contain one row
per grammar-valid canonical work, partitioned by `work_id_numeric mod 4096`.
The columns are:

- receipt-bound canonical publication day, year, and temporal disposition;
- a frozen, lossless work-type code;
- `is_paratext`, nullable `is_retracted`, and declared author count;
- nullable current primary topic, field, and domain IDs.

The extractor projects no current citation count, FWCI, citation percentile,
or normalized citation outcome field. It reproduces the full 492,293,207-row
source accounting, quarantines only the frozen `W-1` row, and must close
exactly to all 492,293,206 canonical work IDs.

## Cutoff ABI

For integer entry year `t`, the public cutoff is December 31 of `t-1`.
The following predicates are frozen:

\[
\operatorname{precutoff\_work}(w,t)
= \mathbf{1}\{d_w\in\{2,3\}\}\,
  \mathbf{1}\{y_w\le t-1\},
\]

where `d_w` is the canonical temporal disposition and `y_w` is its
receipt-bound publication year. Author events use the same work-side predicate.
A collaboration edge at `t` may use only a shared author-event work satisfying
that predicate. The team-size cap is intentionally deferred to G0.

A topic edge is included only after joining to a pre-cutoff work. This prevents
future works from entering the view but does not make current topic annotations
historically observed. A citation edge is included only when its
`available_at <= December 31(t-1)`. In the current parent, `available_at` is the
citing work's publication date: a bibliographic event-time convention, not the
date OpenAlex or a researcher first observed the edge.

## Why this is not the final panel

G0 still needs a versioned successor to choose:

- candidate-author eligibility and minimum history;
- included work types, team caps, and same-day citation treatment;
- pivot, attention-surge, and public-success thresholds;
- a fixed citation maturation horizon and right-censoring rule;
- risk-set sampling, weights, support thresholds, and estimators.

Choosing those values inside this extractor would use a data-readiness step to
silently perform the measurement freeze. Keeping primitives lossless makes
alternative G0 choices reproducible and prevents the extractor from encoding a
preferred result.

## Claim boundary

A full production pass may establish exact projected work-metadata row
conservation, strict modulo partitioning, row-level closure to the canonical
time spine, forbidden-outcome-column exclusion, receipt-bound parent surfaces,
and deterministic cutoff predicates. It cannot establish historical topic or
citation observability, author-person validity, a final analytic risk set, G-1,
G0, empirical G1, an NABM effect, or a scientific effect.

The one-file pilot is permanently nonpromotable. Only a full 493-file run plus
the independent full artifact and row-level verifier may promote this narrow
component.

## Production result, 2026-07-23

The full run passed the narrow component contract:

- 493 source files and 492,293,207 raw rows;
- one frozen `W-1` quarantine and 492,293,206 retained rows;
- exact source-to-canonical ID closure in all 4,096 partitions;
- 4,101 receipt artifacts and 4,122,556,142 output bytes including receipt;
- work-metadata tree SHA-256
  `a37448b4df39c00e97b540b3ac7d06de6f2e9ce12186838120d59c820aa89d77`;
- output receipt SHA-256
  `91e3f689e574d524eabbe7daa58f5e678da37f851ba9b7af0561dab57b70846e`.

The independent verifier rehashed all 4,101 artifacts, scanned every retained
row, replayed every canonical join, rehashed the adjudicated relation, checked
the citation manifest tree, and observed an unchanged output tree. Its report
SHA-256 is
`703d98eeddb5266cef507ef00400c5cb3700958bf3a1744fc16b6ade2121d485`.

An additive producer-independent source-replay verifier subsequently closed
the source-projection boundary. It reopened all 493 main-work files, validated
their content checkpoints and unchanged rich identities, independently parsed
the eight-column projection, and compared every value and null mask for all
492,293,206 retained works. The source and output projection digest is
`9a8f9b13250ab8b11aecd19b0d67b2a1dba9d4395e1d6e63da79f50f8989abf3`;
the external report SHA-256 is
`889f45167345187b520100a90fd5434feca1cda2b1137e44061c3d13cfedaefc`.
This run did not repeat the optional full-byte hash of all 150 GB of source
files, so that narrower limitation remains explicit.

The fail-closed G-1 registry records this as a tenth completed repository
foundation. It keeps three blockers because the final cutoff panel still needs
the G0-resolved choices listed above; outcome-blind field structure and the
representative discovery pilot also remain missing.
