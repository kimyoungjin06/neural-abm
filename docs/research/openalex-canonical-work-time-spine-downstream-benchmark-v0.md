# Canonical work-time-spine bounded downstream benchmark v0

Status: `PARTIAL_FIXED_OVERHEAD_SINGLE_RUN_AND_BOUNDED_H9185_EVIDENCE_OPERATOR_LOCAL_NONPROMOTABLE`

Empirical result: `COMPLETED_EXIT_0_INDEPENDENT_VERIFICATION_PASS`

Independent benchmark verification:
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_BENCHMARK_VERIFICATION_NONPRODUCTION`

Downstream capacity gate: `NOT_PASSED`

Full authorization: `NOT_GRANTED`

Production status: `PRODUCTION_NOT_RUN`

This note defines and records the first bounded, operator-local downstream
benchmark for the canonical work-time-spine implementation. It is derived only
from the completed one-file pilot staging artifact and is permanently
nonpromotable. On 2026-07-21 the run completed with exit status 0 from 11:14:58
to 11:21:50 KST. It executed three repeats of `D4096` demultiplex, `R4096`
pilot-occupancy single-run sort/Parquet, and `H9185_BOUNDED_BYTES`; each repeat
performed three H9185 sweeps.

The result passed its producer-side bounded integrity checks and a separately
implemented read-only verifier. That independent pass is limited to the bounded
nonproduction benchmark artifact and does not promote the capacity gate.

The benchmark is not a production execution, production receipt, capacity-gate
decision, or full authorization. It must not mutate the pilot output, any parent
receipt, the frozen production configuration, or a production receipt lineage.
Only a later, separately reviewed decision artifact with explicit operator
acceptance may change the downstream capacity gate.

## Exact operator-local boundary

The completed operator-local paths are:

- verified pilot input root:
  `/ssd/openalex-g1-canonical-work-time-spine-pilot-20260721`;
- fresh benchmark scratch root:
  `/ssd/openalex-g1-canonical-work-time-spine-downstream-work-20260721`;
- fresh benchmark result root:
  `/ssd/openalex-g1-canonical-work-time-spine-downstream-benchmark-20260721`;
- immutable result artifact:
  `/ssd/openalex-g1-canonical-work-time-spine-downstream-benchmark-20260721/openalex_g1_canonical_work_time_spine_downstream_benchmark.json`.

The completed result artifact has SHA-256
`6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109`
and result fingerprint
`5e12129cafbda0bc5918eedeb43c3ec6352357f3e78e67b0d6d6494f7a71e287`.
The benchmark tool SHA-256 was
`c7809ac74425a837e2b5f58c021f46c297ba0cc7cedd4d60ecf0ad969c6332ff`.

The pilot input root is read-only evidence. Its receipt SHA-256 is
`eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099`,
and its independently verified scope is the one-file
`PRODUCTION_SOURCE_PREFIX_PILOT_UNPROMOTABLE` scope with 1,000,000 retained
records. At 16 bytes per staging record, the selected staging payload is exactly
16,000,000 bytes.

The benchmark must not open the OpenAlex source HDD, accept a source-root
argument, rescan source Parquet, or read the content or temporal parent payloads.
In particular, it must not access the production source under `/hdd`. It may
read only the receipt-bound pilot control and staging files needed to validate
and benchmark the selected staging payload. The input tree identity must be
captured before and after the run and remain unchanged. The completed result
records `source_hdd_opened=false`, an unchanged pilot tree, and the same pilot
receipt SHA-256
`eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099`.

Scratch and result roots must be new, private operator-local directories outside
the repository, release artifacts, pilot output root, and production receipt
lineage. A benchmark result is planning evidence only and must use a distinct
nonpromotable schema and status. It is never a programmatic parent of a future
full receipt.

## First-run scope

### D4096: fixed-overhead demultiplex

`D4096` uses the exact production-width 16-byte staging record, the exact
4,096-way `work_id_numeric mod 4096` partition rule, 4,096 spool handles, 8,192
records of buffering per handle, and the frozen 1,048,576-row demultiplex chunk.
Because the pilot has 1,000,000 retained rows, this is a single demultiplex
chunk. It can measure:

- wall, user, and system time;
- aggregate records/s and bytes/s;
- peak RSS and peak open-file-descriptor count;
- logical input and spool bytes;
- the 4,096-handle/file setup and close/fsync fixed cost;
- fsync latency distribution and tail;
- per-partition occupancy and byte counts; and
- input/output row, byte, and record-digest closure.

Across the three completed repeats, `D4096` recorded 46,507.987305 to
47,093.936769 rows/s. Peak sampled FD count was 4,103, peak sampled RSS across
the run was 229,204 KiB, and the largest repeat-level demultiplex fsync p99 was
0.005614316 seconds. Each repeat passed its bounded 4,096-way demultiplex
integrity checks.

This is evidence about the bounded invocation's fixed overhead and one-chunk
demultiplex path. It is not a measurement of sustained full-corpus demultiplex
throughput, long-run memory stability, or a future live FD preflight. The
observed minimum is not an adopted capacity floor.

### R4096: actual pilot occupancy, single-run sort and Parquet

`R4096` retains the pilot's observed modulo-4096 occupancy and processes every
spool partition through the production-form stable sort, Parquet publication,
decoded validation, and relevant per-output hashing path. The contract records:

- the exact nonempty-partition count and occupancy min/quantiles/max;
- stable-sort, Parquet encode, Parquet decode/validation, and SHA-256 time and
  rates separately;
- per-partition run count, rows, encoded bytes, and elapsed time;
- peak RSS, peak descriptors, logical I/O, and publish/fsync latency;
- exact schema and modulo-membership checks;
- strict increasing numeric IDs and duplicate rejection; and
- equality of pre-Parquet and decoded-Parquet row counts and record digests.

No `R4096` partition can exceed the 1,048,576-row sort chunk because the entire
pilot contains only 1,000,000 rows. Consequently, this run measures only the
single-run path. A successful result does not establish Python k-way heap-merge
throughput or multi-run behavior.

The three completed repeats recorded 8,984.610449 to 9,312.868615 aggregate
R4096 rows/s. Each repeat wrote 22,242,648 Parquet bytes, had maximum run count
one and zero multi-run partitions, and passed its bounded single-run integrity
checks. Decoded-Parquet validation ranged from 431,318.587526 to
472,448.583553 rows/s; Parquet SHA-256 ranged from 50,230,029.434946 to
51,843,284.976371 bytes/s. These rates are bounded pilot-occupancy observations,
not full-like or lower-confidence-bound rates.

### H9185: exact file count with bounded bytes

Each repeat created an exact 9,185-entry benchmark topology and performed three
hash sweeps, for nine sweeps in total. Every sweep hashed 49,237,710 bytes.
Observed artifact inspection rates ranged from 14,640.669980 to
16,115.674928 artifacts/s.

This establishes the fixed file-count traversal for the bounded benchmark
surface only. The result explicitly records
`cache_condition=UNCONTROLLED_WARM_OR_CACHE_ASSISTED` and
`full_byte_finalization_surface_measured=false`. It does not measure the full
26,314,428,448-byte single-sweep surface or the 133,719,625,888-byte
receipt-surface and reconstruction logical-read envelope.

## Explicitly unmeasured after the first run

Even a clean `D4096 + R4096` result leaves all of the following open:

- a full-like modulo occupancy profile, whose mean is approximately 120,200
  retained rows per partition rather than the pilot's approximately 244;
- any partition requiring two or more 1,048,576-row external runs;
- high-fan-in or adversarial k-way merge behavior, including the contract's
  theoretical 470-run one-partition bound;
- the full 26.3-GB 9,185-artifact sweep byte surface, despite measuring its exact
  entry count with only 49,237,710 bytes per sweep;
- the 133.7-GB repeated reconstruction/rehash logical-read envelope;
- five or more repeats across independent quiet-window sessions;
- a lower-confidence-bound throughput floor or upper-confidence-bound latency;
- controlled or separately characterized cold-cache and warm-cache conditions;
- long-duration allocator, filesystem, SSD thermal, writeback, and contention
  behavior; and
- a live full-run disk, RAM, nofile, device-separation, atomic-fsync, source
  health, or sustained quiet-window preflight.

Logical I/O, GNU time filesystem counters, and physical-device I/O must remain
separate metrics. A cache-assisted result must not be relabeled as a physical
SSD throughput floor.

## Completed bounded invocation

The completed run used the bounded, nonpromotable execution flag, three repeats,
and three artifact sweeps per repeat. The command contract was:

```bash
umask 077
uv run python \
  scripts/benchmark_openalex_canonical_work_time_spine_downstream.py \
  --pilot-output-root /ssd/openalex-g1-canonical-work-time-spine-pilot-20260721 \
  --scratch-root /ssd/openalex-g1-canonical-work-time-spine-downstream-work-20260721 \
  --output /ssd/openalex-g1-canonical-work-time-spine-downstream-benchmark-20260721/openalex_g1_canonical_work_time_spine_downstream_benchmark.json \
  --repeats 3 \
  --artifact-sweeps 3 \
  --execute-bounded-nonpromotable
```

The explicit flag authorized only this bounded operator-local benchmark. It did
not grant source-HDD access, a production run, a capacity-gate decision, or full
authorization.

## Independent read-only verification

The separately implemented verifier is
`scripts/verify_openalex_canonical_work_time_spine_downstream_benchmark_v1.py`,
with SHA-256
`d1f9468f528034e64c20dca5b8b05f72ea5913f1c0571ab972de7c8503b6e0d3`.
Its actual stdout status was
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_BENCHMARK_VERIFICATION_NONPRODUCTION`.

The verifier reconstructed result SHA-256
`6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109`,
result fingerprint
`5e12129cafbda0bc5918eedeb43c3ec6352357f3e78e67b0d6d6494f7a71e287`,
three repeats, 9,185 artifacts per repeat and 27,555 in total, 1,000,000 rows
per repeat, and staging SHA-256
`ccbe4ffe751d148076c1711838aa15b976dfb305da4a0b28ef4937743b2a1165`.
It observed unchanged pilot, scratch, and result trees across verification.

The reconstructed boundary retained `production_component_pass=false`,
`pass_g1=false`, `downstream_capacity_gate_passed=false`,
`full_authorization_sufficient=false`, and
`programmatic_parent_of_full_receipt=false`. This is artifact-integrity evidence,
not operator acceptance, a rate-floor decision, or a production pass.

## Result and decision rules

The result's exact status is
`PARTIAL_FIXED_OVERHEAD_SINGLE_RUN_AND_BOUNDED_H9185_EVIDENCE_OPERATOR_LOCAL_NONPROMOTABLE`.
It completed three producer-validated repeats with an unchanged pilot input and
a private nonpromotable result. The separate read-only verifier passed the exact
bounded nonproduction result.

The strongest permitted interpretation is partial fixed-overhead, single-run,
and bounded-byte H9185 evidence. It may inform a later
representative/adversarial benchmark or a conservative analytic route. It is
not enough to set the downstream capacity gate to `PASS`.

Any row, byte, digest, schema, partition-membership, strict-order, path,
permission, input-tree-identity, or stage-completion mismatch is a failed
benchmark result. Missing multi-run, full-byte H9185/finalization,
repeated-session, or cache-control coverage is not a bounded-run integrity
failure, but it must remain explicitly unmeasured and prevents promotion beyond
partial evidence.

Regardless of the first-run outcome:

- `production_component_pass` and `pass_g1` remain false;
- production remains `PRODUCTION_NOT_RUN`;
- the downstream capacity gate remains `NOT_PASSED` until a separate decision;
- full authorization remains `NOT_GRANTED`;
- no production or pilot receipt may be rewritten; and
- no scientific, G-1, G0, empirical G1, or NABM-effect claim follows.
