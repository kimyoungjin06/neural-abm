# Pivot empirical outcome-selection trend: S1 structural build v1

Date: 2026-10-07

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_empirical_outcome_selection_trend_s1_build_v1.yaml`](../../experiments/configs/pivot_empirical_outcome_selection_trend_s1_build_v1.yaml)

Implements stage S1 of
[N10 v2](pivot-empirical-outcome-selection-trend-v2.md) (`58c87f4`).

## What S1 does

S1 is the first stage to read empirical rows. It reads only the Works columns
it needs and computes no persistence label.

1. **Checks before any row.** Every Works file must match its size and stat
   binding from the completed transfer. Every footer must contain the
   registered leaves with their registered types; a missing nested leaf is
   otherwise read as empty. The G4 guard file must match its hash and row
   count. `/ssd` must hold the 200 GiB budget plus a 100 GiB reserve.
2. **Accounting over every row** of the snapshot, as in the lineage's work
   spine. Work identifiers must parse, with `W-1` quarantined, and be unique
   across files. Publication year and date must agree. Any failure stops the
   build.
3. **The extract.** Eligible works whose primary field is 16, 32, or 11, dated
   1900–2017, of the eight registered types. Paratext and retracted works are
   excluded. The `is_xpac` works are kept under the inherited rules and their
   share is reported. Authorships are kept as distinct work–author pairs.
4. **Episodes.** Incumbent-only attention for decision years 2000–2012, then
   the hash-bound N10 surge, spacing, and partition functions. The structural
   indexing screen drops a surge when the median topic score falls by 0.15 and
   the dominant source's share rises by 0.25 together.
5. **Entrants.** For each episode: authors whose first topic work falls in
   years t to t+2, with a work in the episode field in the three preceding
   years. The entry date and quarter come from the hash-bound N10 v2 clock,
   and the SQL clock is checked against it. The G4 author guard is applied by
   exact numeric id first. Then confirmatory entrants who also entered a
   discovery episode are removed.
6. **Floors and disposition.** Floors are 30 precise entrants in two response
   years. `S1_PASS` needs at least four confirmatory episodes meeting the
   floors in every field. Otherwise S1 stops, with no expansion.

## Outputs

| File | Content |
|---|---|
| `s1_summary.json` | Counts only: source accounting, failures, episodes, partition, entrants, disposition |
| `s1_receipt.json` | Hashes of the identity artifacts, the contract, builder, and authorization |
| `s1_timing.json` | Run times; not an identity artifact |
| `private/` | Extract, episodes, entrants, part digests. Mode 0700, never committed |

A second full run must reproduce every identity artifact byte for byte.

## Authorization

The footer-only preflight reads no row value and runs now. A run needs a
committed authorization file. That file names this contract and builder by
hash and quotes the user. S1 does not authorize persistence labels, S2, or
S3.

## Boundary

S1 computes no persistence label and reads no G4 outcome. It uses the G4
guard's author-id column only, for the anti-join. It reads no G1 or
model-selection row and fits nothing.
