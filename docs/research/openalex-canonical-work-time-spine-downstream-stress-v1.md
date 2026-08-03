# Canonical work-time-spine downstream stress successor v1

Status: `PARTIAL_FULL_LIKE_PARTITION_AND_BALANCED_MULTI_RUN_STRESS_OPERATOR_LOCAL_NONPROMOTABLE`

Execution status: `COMPLETED_RESULT_PUBLISHED`

External independent verification:
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_STRESS_VERIFICATION_NONPRODUCTION`

Downstream capacity gate: `NOT_PASSED`

Operator acceptance: `NOT_RECORDED`

Full authorization: `NOT_GRANTED`

Production status: `PRODUCTION_NOT_RUN`

This document defines and records the bounded downstream stress successor. The
implementation surface is
`scripts/benchmark_openalex_canonical_work_time_spine_downstream_stress_v1.py`.
On 2026-07-21 it completed from 12:15:32 to 12:16:21 KST and published exactly
three repeats of each contracted scenario. Producer-side integrity closure
passed. A separately implemented read-only verifier subsequently passed the
bounded nonproduction stress result.

The result is operator-local, synthetic, operational-only, and
permanently nonpromotable. It cannot pass the downstream capacity gate,
authorize a full run, create or mutate a production receipt, or establish a
production or scientific result.

## Exact operator-local inputs and paths

The only operator-local evidence inputs are the independently verified pilot
output and the independently verified first bounded downstream result:

- pilot output root:
  `/ssd/openalex-g1-canonical-work-time-spine-pilot-20260721`;
- first downstream benchmark result:
  `/ssd/openalex-g1-canonical-work-time-spine-downstream-benchmark-20260721/openalex_g1_canonical_work_time_spine_downstream_benchmark.json`.

Their required bindings are:

- pilot receipt SHA-256:
  `eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099`;
- pilot staging SHA-256:
  `ccbe4ffe751d148076c1711838aa15b976dfb305da4a0b28ef4937743b2a1165`;
- first benchmark SHA-256:
  `6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109`;
  and
- first benchmark fingerprint:
  `5e12129cafbda0bc5918eedeb43c3ec6352357f3e78e67b0d6d6494f7a71e287`.

The exact stress paths are:

- retained scratch root:
  `/ssd/openalex-g1-canonical-work-time-spine-downstream-stress-v1-work-20260721`;
- private result root:
  `/ssd/openalex-g1-canonical-work-time-spine-downstream-stress-v1-20260721`;
- result artifact:
  `/ssd/openalex-g1-canonical-work-time-spine-downstream-stress-v1-20260721/openalex_g1_canonical_work_time_spine_downstream_stress_v1.json`.

The immutable result artifact has SHA-256
`f93960649ff2742cc1b4f7efca82fd88b86c85c1ef5066f066d2507f9a766e46`
and result fingerprint
`7f6ef339b7a7c64702a9b8dcb31d09d75439d90bc45c576e6cca43603c69960e`.
The stress tool SHA-256 was
`19bae4881bce030876d884a96500936d3b5a96ef97ccfffe28d7a3d605154b44`.
Pilot, first-benchmark, scratch, and result paths remained mutually disjoint.

The harness accepts no source-root argument and must not open source Parquet or
the OpenAlex source HDD. It reads the verified pilot staging payload and the
first benchmark control result, plus the pinned repository implementations
needed to run the kernels. Pilot and first-benchmark trees must remain unchanged
before and after execution; the result records both as unchanged.

## Executed synthetic scenario contract

The frozen sort chunk is 1,048,576 records. Each scenario targets only synthetic
partition 4,095 and was repeated exactly three times in one invocation.

| Scenario | Exact rows | Expected runs | Bounded role |
| --- | ---: | ---: | --- |
| `RMEAN` | 120,189 | 1 | One full-like mean partition at `ceil(492293206 / 4096)` rows |
| `A2_BALANCED` (`A2`) | 2,097,152 | 2 | Two balanced, production-size runs through the actual heap-merge branch |
| `A8_BALANCED` (`A8`) | 8,388,608 | 8 | Eight balanced, production-size runs through the actual heap-merge branch |

`RMEAN` represents the full contract's mean rows for one partition. It is not
an observed full 4,096-partition occupancy distribution. `A2` and `A8` exercise
balanced fan-in two and eight; they do not establish the theoretical 470-run
one-partition bound or an adversarially skewed full snapshot.

The harness creates deterministic numeric IDs in a disjoint synthetic namespace
starting at `2^63`, with every ID mapped to partition 4,095. Publication-day,
publication-year, temporal-disposition, and reserved-byte payloads are cycled
from the verified pilot staging records. These are synthetic benchmark records,
not additional OpenAlex observations, and they must never enter production or
release evidence.

### Completed aggregate observations

The result records the following wall-rate distributions over exactly three
repeats. Values are minimum / median / maximum; they are observations, not
adopted capacity floors.

| Scenario | Stable-sort rows/s | Merge, Parquet, and validation rows/s | Output bytes/repeat |
| --- | --- | --- | ---: |
| `RMEAN` | 12,372,760.963558 / 12,606,356.198867 / 12,931,891.542931 | 828,444.009429 / 6,204,904.491482 / 6,356,851.959592 | 2,405,052 |
| `A2_BALANCED` | 18,301,352.648573 / 19,219,472.854577 / 19,608,897.698903 | 754,522.670378 / 762,531.406236 / 765,221.960315 | 41,927,598 |
| `A8_BALANCED` | 21,813,862.292422 / 22,807,153.771282 / 22,925,957.911998 | 695,693.770302 / 730,265.246055 / 742,416.120272 | 167,709,850 |

The retained scratch surface contains 60 regular files totaling 1,654,322,493
bytes. All regular files are mode `0600` and no symlink is present. Fifty-eight
directories are mode `0700`. Nine internal `.work_time_spine_tmp` directories
are mode `0775`; each is contained below its own mode-`0700` scenario ancestor,
so the wider internal mode does not expose the tree through its protected path.

## Frozen-kernel and claim boundary

The successor binds the frozen production runner
`scripts/audit_openalex_canonical_work_time_spine_v3.py` at SHA-256
`bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280`.
It calls the runner's actual fixed-width stable-sort and direct
k-way-merge-to-Parquet kernels. This code-path reuse is intended to measure the
real kernel behavior; it does not turn synthetic input into a production run.

The harness must not call the production partition publisher, finalizer, output
receipt builder, or full receipt validation path. Per scenario it must validate
the exact expected run count, synthetic-ID namespace and modulo, strict ordering,
row count, record digest, Parquet decoding, and output SHA-256 before reporting
bounded kernel closure.

The recorded cache label is `UNCONTROLLED_WARM_OR_CACHE_ASSISTED` because each
scenario was validated immediately after it ran. Concurrent host load was not
controlled. The three repeats are not independent quiet-window sessions, and
no one-sided lower confidence bound was computed. Observed minimum rates are
not adopted capacity floors.

## Executed invocation and shell-boundary note

The reviewed protocol contained the shell setup and Python invocation below.
The Python command was launched as a separate invocation, so the preceding
shell-level `umask 077` was not inherited by that process.

```bash
umask 077
uv run python \
  scripts/benchmark_openalex_canonical_work_time_spine_downstream_stress_v1.py \
  --pilot-output-root /ssd/openalex-g1-canonical-work-time-spine-pilot-20260721 \
  --first-benchmark-result /ssd/openalex-g1-canonical-work-time-spine-downstream-benchmark-20260721/openalex_g1_canonical_work_time_spine_downstream_benchmark.json \
  --scratch-root /ssd/openalex-g1-canonical-work-time-spine-downstream-stress-v1-work-20260721 \
  --output /ssd/openalex-g1-canonical-work-time-spine-downstream-stress-v1-20260721/openalex_g1_canonical_work_time_spine_downstream_stress_v1.json \
  --repeats 3 \
  --execute-synthetic-nonpromotable
```

The explicit flag authorizes only these bounded synthetic stress kernels. It is
not source-HDD access, full execution authorization, operator acceptance, or a
capacity-gate decision. Effective containment remained in place through the
mode-`0700` result, scratch, repeat, and scenario ancestors and mode-`0600`
regular files. The nine mode-`0775` internal temporary directories remained
unreachable through their mode-`0700` scenario ancestors. This records the
actual permission boundary rather than claiming that the planned shell umask
applied.

## Result state and remaining limitations

The completed result uses schema
`openalex_g1_canonical_work_time_spine_downstream_stress_benchmark.v1` and exact
status
`PARTIAL_FULL_LIKE_PARTITION_AND_BALANCED_MULTI_RUN_STRESS_OPERATOR_LOCAL_NONPROMOTABLE`.
Its strongest permitted interpretation is
`PARTIAL_SYNTHETIC_FULL_LIKE_PARTITION_AND_BALANCED_FAN_IN_KERNEL_EVIDENCE_ONLY`.

External independent verification returned
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_STRESS_VERIFICATION_NONPRODUCTION`.
The verifier is
`scripts/verify_openalex_canonical_work_time_spine_downstream_stress_v1.py`,
frozen at SHA-256
`4ade14fd53f80a964ff8668b7f4e2391ea220402327fcc0fca9c0dd88e19373f`.
It pins the exact result SHA-256 and fingerprint, enforces nested schema and
numeric invariants, and verifies the exact Parquet row-group layout with no
dictionary pages.

It reconstructed result SHA-256
`f93960649ff2742cc1b4f7efca82fd88b86c85c1ef5066f066d2507f9a766e46`,
fingerprint
`7f6ef339b7a7c64702a9b8dcb31d09d75439d90bc45c576e6cca43603c69960e`,
and scratch forensic tree SHA-256
`41abadafd837b1c5a6d8a029a60102eb412625f31ecf2a2175cf658c0a2002f2`.
It independently checked 60 files totaling 1,654,322,493 bytes, all mode
`0600`, with no symlink; 58 mode-`0700` directories; and nine contained
mode-`0775` internal directories below mode-`0700` scenario ancestors. Result,
scratch, pilot, and first-benchmark tree identities and content SHA-256 values
were unchanged before and after verification.

The immutable producer result still contains
`independent_verification_status=PENDING`. That field records the historical
producer state at publication and is not rewritten. The external verifier's
separate live PASS is the current independent-verification evidence; it does
not alter any producer bytes or promote capacity or production.

Despite completed RMEAN/A2/A8 execution, the following remain unmeasured:

- the actual full 4,096-partition occupancy profile;
- fan-in above eight, including the theoretical 470-run bound;
- the full 26.3-GB H9185 sweep and 133.7-GB reconstruction envelope;
- controlled cold/warm cache strata and independent quiet-window sessions;
- five or more independent observations and a reviewed lower confidence bound;
- an accepted full-runtime model and explicit operator capacity decision; and
- a future live source-health, disk, RAM, nofile, atomic-fsync, device, parent,
  and sustained-quiet-window preflight.

The completed stress result records:

- `actual_full_4096_partition_occupancy_profile_measured=false`;
- `full_byte_finalization_surface_measured=false`;
- `production_execution_status=PRODUCTION_NOT_RUN`;
- `production_component_pass=false` and `pass_g1=false`;
- `full_authorization_sufficient=false`;
- `downstream_capacity_gate_passed=false`;
- `programmatic_parent_of_full_receipt=false`; and
- `operator_acceptance_status=NOT_RECORDED`.

No stress artifact may mutate the pilot, first benchmark, capacity review,
production configuration, or any receipt. No production, G-1, G0, empirical
G1, NABM-effect, or scientific claim follows from this protocol.
