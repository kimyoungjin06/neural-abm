# Pivot-feedback D0 persistence outcome contract v1

Date: 2026-07-31

Status:
`PASS_INDEPENDENT_D0_PERSISTENCE_OUTCOME_CONTRACT_AND_FIXTURE_REPLAY_NO_PRODUCTION_ROW_READ`

## Decision

The D0 persistence outcome is defined only for the 4,301 independently
verified `NEW_TOPIC_ENTRY` rows. The other 3,251,970 classified rows do not
receive a persistence zero and are not materialized in the persistence output.

For a matured primary pivot, persistence is true when the same current
OpenAlex author token has at least one distinct eligible work in `t+1`,
`t+2`, or `t+3` whose current-snapshot primary topic and field exactly match
the certified episode.

This is a current-snapshot-annotated persistence measure, not proof of a
researcher's true topic persistence.

## Denominator and output states

The production persistence source must retain every `NEW_TOPIC_ENTRY` row and
only those rows. It has three mutually exclusive states:

- `MATURED_PERSISTENCE`: complete three-year follow-up and at least one
  qualifying author-work pair;
- `MATURED_NO_PERSISTENCE`: complete three-year follow-up and zero qualifying
  pairs; and
- `RIGHT_CENSORED`: the `t+1..t+3` window is not complete.

Partial follow-up cannot be promoted to persistence even if a positive work is
already visible. Counts and the persistence boolean remain null for a
right-censored row.

## Maturation

The bound source snapshot is dated 2026-03-30, so 2025 is the last complete
calendar year. A row is matured only when `entry_year + 3 <= 2025`.

All four D0 episodes are matured:

| Episode | Entry year | Last required outcome year |
|---:|---:|---:|
| 1 | 2003 | 2006 |
| 2 | 2012 | 2015 |
| 3 | 1993 | 1996 |
| 4 | 1953 | 1956 |

The right-censoring state remains mandatory for future reuse and was exercised
with a synthetic sentinel.

## Frozen ABI

The work and author-linkage rules are inherited without outcome-driven
modification:

- work types: 1, 2, 3, 4, 7, 16, 18, and 21;
- no team-size cap;
- exact current-snapshot primary-topic and field match;
- paratext and retracted works excluded;
- null `is_retracted` retained and reported;
- null publication year and missing/nonmatching topic or field excluded and
  reported;
- exact author-work pairs deduplicated; and
- author identity remains an OpenAlex author token, not a verified person.

Zero becomes `MATURED_NO_PERSISTENCE` only after the complete window and these
eligibility rules have been applied.

## Independent fixture replay

The validator independently checked 11 repository bindings and three operator
receipts without opening any Parquet data page. It reconstructed two synthetic
fixtures with zero row or accounting mismatch:

- a complete window containing one persistent and four nonpersistent primary
  pivots, work-type/topic/retraction/temporal/missing-year exclusions, a
  duplicate author-work event, an unregistered author, and a zero-token work;
- an incomplete window containing a positive `t+1` work that still remains
  `RIGHT_CENSORED`.

The fixtures cover `MATURED_PERSISTENCE`, `MATURED_NO_PERSISTENCE`, and
`RIGHT_CENSORED`.

## Boundary and successor

This validation read zero production classification rows and zero production
post-entry rows. It created no API request and did not authorize outcome
analysis, G0, model fitting, or G4.

A successor gate may authorize only the receipt-bound production persistence
source build for the exact 4,301-row denominator. That build must rehash all
source partitions, write private operator-local artifacts, and be independently
replayed before persistence results may be interpreted.
