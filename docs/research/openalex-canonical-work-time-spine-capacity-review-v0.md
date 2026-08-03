# Canonical work-time spine downstream capacity review v0

Status: `PARTIAL_BOUNDED_AND_SYNTHETIC_STRESS_EVIDENCE_OPERATOR_LOCAL_NONPROMOTABLE`

Downstream capacity gate: `NOT_PASSED`

Operator acceptance: `NOT_RECORDED`

Independent downstream-benchmark verification:
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_BENCHMARK_VERIFICATION_NONPRODUCTION`

External stress-result verification:
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_STRESS_VERIFICATION_NONPRODUCTION`

Production execution: `PRODUCTION_NOT_RUN`

This document is a conservative analytic planning artifact for the production
canonical work-time spine. The v0 calculator does not ingest or validate a
benchmark, pass a gate, grant full authorization, or create a production
receipt. The frozen production runner and configuration are inputs to this
review and are not modified by it.

Post-review update, 2026-07-21: the operator-local one-file scan-and-staging
pilot measured 1,000,000 rows in 11.69 seconds and produced a simple
5,754.907590-second full-row scan-and-staging projection. A separately
authorized, staging-derived downstream benchmark subsequently completed from
11:14:58 to 11:21:50 KST with exit status 0. It recorded three repeats of
`D4096`, pilot-occupancy single-run `R4096`, and bounded-byte H9185 evidence
without opening the source HDD.

This is the completed bounded nonpromotable staging-derived benchmark that the
earlier review recommended. It is an operational evidence overlay, not an
input mutation or retrospective status change to the frozen calculator.

That downstream result is partial and nonpromotable. Its separate read-only
verifier passed the bounded artifact, but operator acceptance is
`NOT_RECORDED`, the downstream gate remains `NOT_PASSED`, and full execution
remains unauthorized and `PRODUCTION_NOT_RUN`. The frozen calculator input
intentionally retains its historical
`ANALYTIC_ESTIMATE_ONLY_EMPIRICAL_BENCHMARK_NOT_RUN` status and is not silently
rewritten by this dated operational overlay.

A second bounded successor completed from 12:15:32 to 12:16:21 KST. It used
synthetic IDs derived from the pilot temporal payload to exercise one full-like
mean partition and balanced two- and eight-run frozen sort/merge kernels. It
opened no source HDD. This strengthens the partial kernel evidence but does not
establish an actual full 4,096-partition occupancy profile, an accepted rate
floor, or a capacity pass. A separate read-only verifier subsequently passed
the exact bounded nonproduction stress artifact.

The repository-only calculator is
`scripts/review_openalex_canonical_work_time_spine_capacity.py`; its input
template is
`experiments/configs/openalex_g1_canonical_work_time_spine_capacity_review_v0.yaml`.
The calculator reads the frozen YAML and runner bytes only to verify their
SHA-256 bindings. It never imports or executes the runner, accepts no source
path, opens no OpenAlex payload, and writes no file. It emits JSON to stdout:

```bash
uv run python scripts/review_openalex_canonical_work_time_spine_capacity.py
```

## Frozen inputs and claim boundary

| Input | SHA-256 |
| --- | --- |
| `experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml` | `6e8b74ddd12cbef46c8d15d3560f0d54f470b5017685fc1bce2389250cde0484` |
| `scripts/audit_openalex_canonical_work_time_spine_v3.py` | `bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280` |

The following distinctions are mandatory:

- Exact byte counts come from the frozen row count and fixed-width ABI.
- Upper bounds come from the frozen Parquet, control-artifact, and atomic-file
  envelopes. They need not be reached by an actual run.
- Logical I/O counts user-space payload reads and writes. Cache, writeback,
  filesystem metadata, and device amplification can make physical device I/O
  different.
- Conditional time values use hypothetical rate floors. They are neither
  measurements nor unconditional wall-time bounds.
- A dated hardware observation is not a live execution preflight.
- Analytic review and operator acceptance are separate acts. Neither is
  inferred from the other.

## 2026-07-21 bounded downstream evidence

The current operator-local result is documented in
`docs/research/openalex-canonical-work-time-spine-downstream-benchmark-v0.md`.
Its artifact is
`/ssd/openalex-g1-canonical-work-time-spine-downstream-benchmark-20260721/openalex_g1_canonical_work_time_spine_downstream_benchmark.json`, with:

- SHA-256
  `6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109`;
- result fingerprint
  `5e12129cafbda0bc5918eedeb43c3ec6352357f3e78e67b0d6d6494f7a71e287`;
- benchmark-tool SHA-256
  `c7809ac74425a837e2b5f58c021f46c297ba0cc7cedd4d60ecf0ad969c6332ff`;
  and
- unchanged pilot receipt SHA-256
  `eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099`.

The independent verifier
`scripts/verify_openalex_canonical_work_time_spine_downstream_benchmark_v1.py`
has SHA-256
`d1f9468f528034e64c20dca5b8b05f72ea5913f1c0571ab972de7c8503b6e0d3`.
It returned
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_BENCHMARK_VERIFICATION_NONPRODUCTION`
and reconstructed the same result SHA-256 and fingerprint, three repeats,
9,185 artifacts per repeat and 27,555 total, 1,000,000 rows per repeat, and
staging SHA-256
`ccbe4ffe751d148076c1711838aa15b976dfb305da4a0b28ef4937743b2a1165`.
Pilot, scratch, and result trees remained unchanged. All production, capacity-
gate, full-authorization, promotion, and G1 pass fields remained false.

| Bounded observation | Completed result |
| --- | --- |
| Repeat scope | Three repeats in one run; zero independently established quiet-window sessions |
| D4096 aggregate rate | 46,507.987305 to 47,093.936769 rows/s |
| D4096 resource sample | peak FD 4,103; maximum sampled RSS 229,204 KiB |
| D4096 fsync tail | largest repeat-level p99 0.005614316 seconds |
| R4096 aggregate rate | 8,984.610449 to 9,312.868615 rows/s |
| R4096 output shape | 22,242,648 Parquet bytes per repeat; maximum run count 1; zero multi-run partitions |
| Decoded validation | 431,318.587526 to 472,448.583553 rows/s |
| Parquet SHA-256 | 50,230,029.434946 to 51,843,284.976371 bytes/s |
| H9185 | exact 9,185 entries and three sweeps per repeat; 49,237,710 bytes per sweep |
| H9185 artifact rate | 14,640.669980 to 16,115.674928 artifacts/s |
| Cache boundary | `UNCONTROLLED_WARM_OR_CACHE_ASSISTED` |

The benchmark verifies a bounded 4,096-handle fixed-overhead path, the actual
pilot modulo occupancy's single-run sort/Parquet path, and exact 9,185-entry
traversal with bounded bytes. It does not measure full-like occupancy,
multi-run/adversarial Python heap merge, a full 26,314,428,448-byte sweep, the
133,719,625,888-byte reconstruction logical-read envelope, five or more repeats
across independent sessions, a one-sided lower confidence bound, controlled
cache conditions, or a live full-run preflight. Its minimum rates are not
adopted capacity floors. Independent verification passed only this bounded
nonproduction artifact and does not close any of those measurement gaps.

## 2026-07-21 synthetic stress evidence

The completed successor is documented in
`docs/research/openalex-canonical-work-time-spine-downstream-stress-v1.md`. Its
operator-local result is
`/ssd/openalex-g1-canonical-work-time-spine-downstream-stress-v1-20260721/openalex_g1_canonical_work_time_spine_downstream_stress_v1.json`, with:

- SHA-256
  `f93960649ff2742cc1b4f7efca82fd88b86c85c1ef5066f066d2507f9a766e46`;
- result fingerprint
  `7f6ef339b7a7c64702a9b8dcb31d09d75439d90bc45c576e6cca43603c69960e`;
  and
- stress-tool SHA-256
  `19bae4881bce030876d884a96500936d3b5a96ef97ccfffe28d7a3d605154b44`.

The run completed exactly three repeats of each synthetic scenario:

| Scenario | Stable-sort rows/s, min / median / max | Merge, Parquet, and validation rows/s, min / median / max | Output bytes/repeat |
| --- | --- | --- | ---: |
| `RMEAN`, 120,189 rows, one run | 12,372,760.963558 / 12,606,356.198867 / 12,931,891.542931 | 828,444.009429 / 6,204,904.491482 / 6,356,851.959592 | 2,405,052 |
| `A2_BALANCED`, 2,097,152 rows, two runs | 18,301,352.648573 / 19,219,472.854577 / 19,608,897.698903 | 754,522.670378 / 762,531.406236 / 765,221.960315 | 41,927,598 |
| `A8_BALANCED`, 8,388,608 rows, eight runs | 21,813,862.292422 / 22,807,153.771282 / 22,925,957.911998 | 695,693.770302 / 730,265.246055 / 742,416.120272 | 167,709,850 |

The retained surface has 60 regular mode-`0600` files totaling 1,654,322,493
bytes, no symlink, 58 mode-`0700` directories, and nine internal
`.work_time_spine_tmp` directories at mode `0775`. Each wider internal directory
is protected below a mode-`0700` scenario ancestor. The planned shell-level
`umask 077` was not inherited by the separately launched Python invocation;
the document therefore relies only on the observed ancestor and file modes,
not on an untrue process-umask claim.

The result records `UNCONTROLLED_WARM_OR_CACHE_ASSISTED`; immediate validation
and uncontrolled concurrent host load prevent a cache-neutral interpretation.
All three repeats came from one invocation, with zero independent quiet-window
sessions and no one-sided lower confidence bound. The measurements are not
adopted capacity floors. They validate synthetic RMEAN and balanced fan-in two
and eight producer paths only; the actual full occupancy distribution, fan-in
above eight, full-byte finalization, and integrated end-to-end rate model remain
open.

The external verifier
`scripts/verify_openalex_canonical_work_time_spine_downstream_stress_v1.py`,
frozen at SHA-256
`4ade14fd53f80a964ff8668b7f4e2391ea220402327fcc0fca9c0dd88e19373f`,
returned
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_STRESS_VERIFICATION_NONPRODUCTION`.
It pins the exact result SHA-256 and fingerprint, enforces nested schema and
numeric invariants, and verifies the exact Parquet row-group layout with no
dictionary pages.
It reconstructed result SHA-256
`f93960649ff2742cc1b4f7efca82fd88b86c85c1ef5066f066d2507f9a766e46`,
fingerprint
`7f6ef339b7a7c64702a9b8dcb31d09d75439d90bc45c576e6cca43603c69960e`,
and scratch forensic tree SHA-256
`41abadafd837b1c5a6d8a029a60102eb412625f31ecf2a2175cf658c0a2002f2`.
It independently checked 60 mode-`0600` files totaling 1,654,322,493 bytes,
58 mode-`0700` directories, nine contained mode-`0775` internal directories,
and no symlink. Result, scratch, pilot, and first-benchmark tree identity and
content SHA-256 observations remained unchanged before and after verification.

The producer artifact's immutable historical field remains
`independent_verification_status=PENDING`; the external PASS is separate current
evidence and does not rewrite producer bytes or pass the capacity gate.

## Contract anchors

The full contract fixes 492,293,206 retained records, a 16-byte record ABI,
4,096 modulo partitions, 1,048,576-record sort chunks, and exactly 9,185
receipt-bound artifacts. The principal payload bounds are:

| Quantity | Formula | Bytes | GiB |
| --- | ---: | ---: | ---: |
| staging | `492293206 * 16` | 7,876,691,296 | 7.336 |
| spool | same as staging | 7,876,691,296 | 7.336 |
| external runs | same as staging | 7,876,691,296 | 7.336 |
| Parquet | `492293206 * 32 + 4096 * 131072` | 16,290,253,504 | 15.172 |
| control artifacts | frozen upper bound | 2,147,483,648 | 2.000 |
| next atomic artifact | `492293206 * 32 + 131072` | 15,753,513,664 | 14.672 |
| worst remaining disk workspace | sum of the six rows above | 57,821,324,704 | 53.850 |

The existing full-run thresholds remain 153,143,246,352 bytes for a fresh
invocation, 95,321,921,648 bytes for a validated nonterminal resume, and a
preferred fresh floor of 171,798,691,840 bytes. This review does not alter
those values.

## Phase estimates

### 1. 4,096-handle spool and NumPy demultiplex

Verified staging is read once and the same 16-byte payload is written once:

```text
logical read  =  7,876,691,296 bytes
logical write =  7,876,691,296 bytes
logical total = 15,753,382,592 bytes
```

The incremental on-disk spool workspace is exactly 7,876,691,296 bytes. With
the already present on-disk staging payload, the combined payload is
15,753,382,592 bytes; adding the entire frozen control-artifact bound gives a
conservative 17,900,866,240-byte phase surface. RAM planning remains separate:
536,870,912 bytes of aggregate per-handle buffers plus a 134,217,728-byte
demultiplex engineering estimate. At most 4,096 spool data files are open and
at most 4,096 data-file close/fsync operations occur.

The hash-work model counts raw staging SHA-256, framed 24-byte-per-record
record-digest input, each 45-byte domain/row-count digest header, and the inline
spool digest. It is an engineering CPU-work model rather than additional disk
traffic.

Under the unvalidated planning assumptions below, the additive conditional
estimate with the 2x safety multiplier is 5,398 seconds, or 1.499 hours. It is
not a measured duration.

### 2. External sort and direct Parquet publication

For a conservative distribution-independent envelope, each binary payload is
read or written according to the frozen call path:

| Operation | Bound |
| --- | ---: |
| read spool and runs, then validate and hash Parquet | 48,333,889,600 bytes |
| write runs and Parquet | 24,166,944,800 bytes |
| total logical I/O | 72,500,834,400 bytes |

For nonempty partition row counts `r_i`, the run count is
`sum(ceil(r_i / 1048576))`. Therefore a global distribution-independent bound
is:

```text
ceil(492293206 / 1048576) + 4096 - 1 = 4,565 run files
```

Together with 4,096 Parquet candidates, this gives at most 8,661 data-file
fsync operations. It does not count directory fsyncs or every metadata syscall.
The worst-case conditional CPU model treats all retained records as eligible
for the multi-run Python heap merge. Actual modulo occupancy may make most or
all partitions single-run and faster, but that cannot be assumed before it is
observed.

The phase uses the frozen 57,821,324,704-byte peak workspace envelope, which
already includes staging, spool, runs, all Parquet bounds, controls, and the
largest next atomic artifact. Under the unvalidated planning rates, the
additive 2x conditional estimate is 26,072 seconds, or 7.242 hours. This number
is a sensitivity calculation, not a runtime commitment.

### 3. One 9,185-artifact final rehash sweep

A single receipt-artifact sweep has exactly 9,185 entries. Using the complete
staging, Parquet, and control bounds gives:

```text
7,876,691,296 + 16,290,253,504 + 2,147,483,648
= 26,314,428,448 bytes
```

The sweep is read-only and needs zero incremental payload workspace; the
published payload-and-control surface is bounded by 26,314,428,448 bytes. The
hypothetical 2x conditional estimate is 1,756 seconds, or 0.488 hours.

One sweep is not the whole same-invocation finalization cost. The frozen call
path constructs the receipt artifact contracts, validates those contracts,
reconstructs the receipt, rehashes staging and Parquet during reusable-artifact
validation, decodes Parquet again, and rereads control artifacts. A
conservative logical-read envelope for that receipt-surface and output-
reconstruction portion is:

```text
5 * staging + 5 * parquet_upper_bound + 6 * control_upper_bound
= 133,719,625,888 bytes
```

This 133,719,625,888-byte value is deliberately exposed so that the phrase
"9,185-artifact rehash" is not mistaken for a single-pass finalization model.
It excludes parent-control revalidation, own frozen-input revalidation, and
filesystem metadata/directory-walk bytes, so it is not an upper bound for the
entire finalization invocation. Caching may reduce physical reads, while
filesystem behavior may add other traffic.

## Dated hardware observation

At `2026-07-20T17:11:18+09:00`, control-plane-only commands observed:

| Observation | Value |
| --- | ---: |
| local output device | Samsung SSD 990 PRO 4 TB, ext4, nonrotational |
| output filesystem available | 3,657,099,960,320 bytes |
| available RAM | 1,037,850,583,040 bytes |
| soft/hard nofile | 1,048,576 / 1,048,576 |
| source/output `st_dev` | 2050 / 66309, distinct |

Arithmetic comparison gives 23.880x the fresh disk threshold, 60.410x the
16-GiB RAM threshold, and 128x the 8,192 nofile threshold at that instant.
These are snapshot comparisons only. No source payload was opened, and no
atomic rename/fsync probe, storage-throughput benchmark, source-drive health
review, or quiet-window review was performed for this observation. The values
may be stale at execution time and do not pass any preflight or capacity gate.
A later operational or drive-health finding always supersedes this dated
comparison for execution decisions.

## Hypothetical rate-floor sensitivity

The template intentionally uses low, round planning values and labels every
one as unmeasured:

| Component | Hypothetical floor |
| --- | ---: |
| sequential read | 64 MiB/s |
| sequential write | 64 MiB/s |
| SHA-256 input | 64 MiB/s |
| demultiplex | 250,000 records/s |
| stable sort | 250,000 records/s |
| worst-case heap merge | 100,000 records/s |
| Parquet encode/decode | 250,000 records/s each |
| data-file fsync | 50 operations/s |
| artifact-contract inspection | 100 artifacts/s |
| additive time multiplier | 2.0x |

The calculator rounds every component upward, adds the components rather than
assuming overlap, and applies the 2x multiplier. This is conservative relative
to those assumed floors, but it is not evidence that the floors hold on the
target runtime. Native allocator behavior, Python loop cost, directory fsyncs,
thermal throttling, other processes, device error recovery, and filesystem
cache effects remain unmeasured.

The bounded result does not validate these hypothetical floors as a set. Its
D4096 aggregate wall rate is below the hypothetical 250,000-record/s demultiplex
floor, while its R4096 aggregate wall rate includes 4,096-way fixed file and
fsync overhead and is not the same quantity as an isolated sort or Parquet
floor. Decoded validation exceeded 250,000 rows/s, but the Parquet SHA-256 rate
was below 64 MiB/s and the H9185 sweeps were cache assisted and only
49,237,710 bytes each. With three repeats, no independent session, and no lower
confidence bound, none of these observations becomes an adopted full-run rate
floor.

The synthetic stress successor separately observed high isolated stable-sort
and balanced fan-in-two/fan-in-eight merge rates through the frozen kernels.
Those observations are still not adopted floors: RMEAN is one synthetic mean
partition rather than the actual 4,096-partition distribution, fan-in stops at
eight, validation was immediate and cache assisted, concurrent host load was
uncontrolled, and the three repeats supply neither independent sessions nor a
lower confidence bound. External verification passed the bounded artifact but
does not change those measurement limitations.

## Why the capacity gate remains not passed

The frozen production policy permits either a bounded nonpromotable benchmark
or a conservative analytic estimate with explicit operator acceptance. The
completed first benchmark and synthetic stress successor contribute real fixed-
overhead, single-run, full-like-mean, balanced fan-in-two/eight, and bounded-
byte evidence, but do not complete either review route. Both remain outside the
production receipt lineage and leave the following minimum work:

1. Resolve the residual occupancy and merge boundary: the actual full
   4,096-partition distribution remains unmeasured, and balanced fan-in stops at
   eight rather than exercising higher or adversarial fan-in.
2. Measure representative large-byte H9185/finalization surfaces rather than
   only the exact entry count with 49,237,710 bytes per sweep.
3. Repeat under at least five observations across independent intended
   quiet-window sessions, characterize cache state, and derive reviewed lower
   confidence bounds before adopting any rate floor.
4. Integrate scan, fixed-overhead, occupancy, merge, finalization, and residual
   assumptions into a reviewed rate model, then record explicit operator
   acceptance in a separate capacity decision artifact.
5. Repeat source-drive health and live disk/RAM/nofile/atomic-fsync and quiet-
   window preflights at the eventual full-execution time.

A one-file pilot measures scan and staging only; extrapolating its scan rate is
not a downstream benchmark. A future analytic-route decision would instead
need an explicit evidence-backed justification for every adopted floor and
unmodeled allowance. Under either route, independent review and explicit
**operator acceptance** are required. Only a separate decision artifact may
change the downstream-capacity gate. This document remains `NOT_PASSED`, and it
does not grant full authorization.
