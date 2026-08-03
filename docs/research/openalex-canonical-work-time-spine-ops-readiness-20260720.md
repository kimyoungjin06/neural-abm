# Canonical work-time-spine operational readiness — 2026-07-20 to 2026-07-21

Historical close status on 2026-07-21: `POST_TEST_MEDIA_SCREEN_PASS / ONE_FILE_PILOT_COMPLETED_NONPROMOTABLE / BOUNDED_STRESS_RESULT_INDEPENDENTLY_VERIFIED / FULL_BLOCKED`

Current component status after the separately authorized 2026-07-22 run: `PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_TEMPORAL_V3_PARENT`

This began as a read-only operator snapshot for the production-capable
canonical work-time-spine successor. After explicit maintenance authorization,
SMART tooling was installed and non-captive drive self-tests were completed.
On 2026-07-21, after a new live preflight and explicit pilot authorization, one
source-file prefix pilot was executed and independently verified. This record
is not a production parent, a drive repair record, or a full authorization.
Production full mode had not been executed at the close of that dated
readiness review. The later full run below supersedes only that operational
`FULL_BLOCKED / PRODUCTION_NOT_RUN` state; it does not promote the pilot or
synthetic benchmark artifacts.

## 2026-07-22 full production execution and verification

After a fresh live preflight and explicit full authorization, the production
runner completed all 493 source files. It read 492,293,207 rows, quarantined
the frozen `W-1` row, retained 492,293,206 distinct numeric work IDs, and found
zero duplicate-ID groups or excess rows. The receipt-bound output contains
4,096 partitions and 9,185 artifacts. Its operator-local root is
`/ssd/openalex-g1-canonical-work-time-spine-full-20260722`.

The authoritative receipt is `work_time_spine_audit.outputs.json`, SHA-256
`2760b3aa5c5960d9356cc30339338418ff9000ab22e9c4b94caf82f6216f9aa1`.
The task fingerprint is
`94cf717b7d986a3402b6e91a6c20ad5223f652d724899b1210bd94208bd703b0`,
and the result fingerprint is
`58492b9182d9b213c0616ce5d29d69bd82714167f57619885dc013dd0219144d`.

The independent verifier then ran in its default `full` Parquet mode and
returned `PASS_READ_ONLY_INDEPENDENT_OUTPUT_VERIFICATION`. It reconstructed all
9,185 artifact contracts, streamed all 4,096 published partitions, confirmed
the production component PASS, and observed the output tree unchanged. The
compact repository evidence is
`experiments/evidence/results/openalex_g1_canonical_full_work_time_spine_20260722.summary.{json,md}`.
This closes only `canonical_full`; G-1 remains pending on five other required
components.

## Historical 2026-07-21 decision

The short and extended SMART self-tests both completed without error. Critical
media-error attributes remained zero, SMART error logs remained empty, and no
new matching kernel event appeared before, during, or after the pilot. This
passes the current media screen, not an unconditional drive-safety guarantee.
The operator reports that the 2026-07-17 event coincided with an earlier
RAID/SATA contact fault; that causal attribution was not independently
reconstructed from SMART or kernel evidence.

The one-file pilot completed with a valid nonpromotable receipt, passed the
implementation-level independent verifier, and produced an operator-local
benchmark artifact. A later staging-derived downstream benchmark completed
three bounded repeats of the 4,096-way demultiplex and pilot-occupancy
single-run Parquet paths, plus bounded-byte hash sweeps over an exact
9,185-entry topology. A separate read-only verifier passed that exact bounded
nonproduction artifact. A synthetic stress successor then completed one
full-like mean partition and balanced two- and eight-run frozen sort/merge
kernels. Its producer result and a separate read-only external verification are
complete. These remain partial downstream evidence, not a capacity-gate
decision. The actual full occupancy distribution, fan-in above eight, full-byte
finalization, independent cache-controlled sessions, operator acceptance, a
future full-run quiet window and preflight, and explicit full authorization
were open, so the dated review concluded that full execution must not start at
that time.

## Historical 2026-07-21 gate snapshot

| Gate | Current decision | Operational evidence |
| --- | --- | --- |
| Exact parent controls | `PASS_PILOT_INVOCATION` | Both receipts and the frozen config/tool anchors matched the production contract immediately before and during the pilot. |
| Source/output device separation | `PASS_PILOT_INVOCATION` | Runner evidence records source `st_dev=2050` and output `st_dev=66309` with `PASS_DISTINCT_DEVICES`. |
| `/ssd` available space | `PASS_PILOT_INVOCATION` | 3,589,265,149,952 bytes observed against the selected 36,366,736,928-byte fresh-pilot requirement. |
| Available RAM | `PASS_PILOT_INVOCATION` | 1,026,828,238,848 bytes observed against the 9,685,696,512-byte requirement. |
| `RLIMIT_NOFILE` | `PASS_PILOT_INVOCATION` | Soft limit 1,048,576 against the 8,192 requirement. |
| Source-I/O observations | `PASS_PILOT_WINDOW_ONLY` | The final 30-second pre-pilot window had zero `sda` reads, writes, queue, and utilization, with no user-space source holder. |
| Sustained exclusive quiet window | `FULL_PENDING` | The pilot window was adequate for the bounded 11.69-second run; it does not establish a future multi-hour full window. |
| Source-drive health | `PASS_POST_TEST_MEDIA_SCREEN_CURRENT_SNAPSHOT` | Both self-tests completed without error, key media attributes are zero, error logs are empty, and no new kernel storage event was found. Residual connection-path causality is not independently proven. |
| Exact fresh output directory and atomic probe | `PASS_PILOT_INVOCATION` | Fresh private output was selected on `/ssd`; the runner recorded atomic rename/fsync/readback `PASS`. |
| Runtime CLI path binding | `PASS_PILOT_INVOCATION` | Exact source and parent paths were passed explicitly; frozen runner/config and parent hashes matched. |
| One-file pilot | `PASS_NONPROMOTABLE_VALIDATION_ONLY` | One lexical-prefix file, 1,000,000 rows and 312,250,942 source bytes; production and G1 pass fields remain false. |
| Downstream capacity | `PARTIAL_BOUNDED_AND_STRESS_EVIDENCE_VERIFIED / NOT_PASSED` | The first bounded D4096/R4096/H9185 result and later synthetic RMEAN/A2/A8 stress result both passed separate read-only verification. Actual full occupancy, fan-in above eight, full-byte finalization, independent-session/cache controls, and an accepted rate-floor decision remain absent. |
| Full authorization | `NOT_GRANTED` | No full-run acknowledgement or authorization was issued. |

The pilot FD result proves only that invocation's configured soft-limit gate,
not exact future full-process headroom after libraries and unrelated
descriptors are open. The runner's 4,096 spool handles still fit by a wide
static margin, but the dated review required the live measurement to be
repeated immediately before any later full execution.

## Frozen parent paths and hashes

- Source root:
  `/hdd/raid_backup_20260624/data/OpenAlex/parquet_exports/openalex_works_20260330_repairreplay_20260410_190630`
- Content receipt:
  `/ssd/openalex-g1-content-freeze-20260719/content_freeze_audit.outputs.json`
  - SHA-256:
    `082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12`
- Temporal receipt:
  `/ssd/openalex-g1-main-temporal-field-validity-v3-full-20260719/main_temporal_validity_audit.outputs.json`
  - SHA-256:
    `2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe`
- Temporal config SHA-256:
  `672940908090fedd0b562dd938ad4583f5e8238830fc1ff5b613be9618f033f7`
- Temporal tool SHA-256:
  `76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d`

The exact no-Parquet control-plane validation performed after the runner freeze
separately passed all 493 tasks and 507 parent-control snapshots. This
operational snapshot does not replace that contract validation.

## Source-drive evidence and limitation

The current boot journal records the following sequence at approximately
2026-07-17 14:45:48 local time:

- `READ FPDMA QUEUED` failure on `/dev/sda`;
- a SATA link hard reset; and
- a kernel read I/O error around sector 12,540,392,480.

No later matching kernel event was found through the extended-test start. The
initial UDisks SMART snapshot, updated around 2026-07-20 17:05, reported:

- overall failing: false;
- reallocated, pending, offline-uncorrectable, UDMA-CRC, and
  reported-uncorrectable raw counts: zero;
- its summary self-test-status property: `success`; and
- temperature: approximately 32 °C.

Those reassuring attributes did not close the earlier event. The same snapshot
reported `runtime-bad-block-total` raw 4 and a vendor-encoded command-timeout
value. The UDisks `success` property does not establish that a prior test entry
exists: the subsequent smartctl self-test log contained no pre-maintenance
entries. UDisks' summary is not being promoted to a fresh full-drive health
pass.

## 2026-07-20 SMART maintenance follow-up

`smartmontools` 7.4 was installed during the approved maintenance window. The
package also enabled and started `smartd`; that service state is an explicit
host mutation from this follow-up. The tested device was bound by the stable
path
`/dev/disk/by-id/ata-ST8000DM004-2U9188_ZR153GQW`, which resolved to the direct
SATA device `/dev/sda`.

The fresh core report (`smartctl -a -d ata`) exited 0 and reported overall
health `PASSED`. An explicit-ATA extended query (`smartctl -x -d ata`) exited 4
because some log commands failed; it is not recorded as an unqualified command
success. A separate earlier auto-detected `smartctl -x /dev/sda` invocation
exited 0 and supplied the extended error, device-statistics, and SATA-PHY
sections. These command paths are retained as distinct provenance lanes. Their
combined readable evidence showed:

- zero reallocated, pending, offline-uncorrectable,
  reported-uncorrectable, and UDMA-CRC sectors/errors;
- no ATA extended errors and no pending defects;
- no previously logged self-test;
- `Runtime_Bad_Block` raw 4 and `Command_Timeout` raw `6 126 126`;
- from the auto-detected report, 126 device-statistics resets between command
  acceptance and completion, 524 hardware resets, and four SATA PHY COMRESET
  FIS events; and
- temperature 33 °C at review time, with a recorded lifetime maximum of 44 °C.

These counters do not establish whether the 2026-07-17 incident was media,
link, controller, or power-path related. They are retained as diagnostic
context, not converted into a causal claim.

The non-captive short self-test started at 2026-07-20 18:06:33 KST and completed
without error. Immediately afterward, the key media attributes and error log
remained unchanged and no new matching kernel event was present. After a final
24-second zero-I/O check, the non-captive extended self-test started at
2026-07-20 18:08:37 KST. The device reported a 961-minute duration and an
expected completion time of 2026-07-21 10:09:38 KST. At 10:12 KST its final
log reported `Extended offline / Completed without error`, with no first-error
LBA. The core final SMART query exited 0; overall health remained `PASSED`, the
short test also remained complete without error, key media attributes and the
error log were unchanged, and no new matching kernel event was found.

Operator-provided maintenance history reports that the 2026-07-17 event
coincided with a prior RAID/SATA contact fault. This attribution is recorded as
`USER_PROVIDED_NOT_INDEPENDENTLY_VERIFIED`. The completed tests and quiet kernel
record are consistent with, but do not prove, a transient connection-path
cause rather than media failure.

## 2026-07-21 one-file pilot evidence

The explicitly authorized pilot used `--execute --max-files 1` without either
full-authorization flag. It ran from 2026-07-21 10:18:37 to 10:18:48 KST and
returned exit status 0. Its private output root is
`/ssd/openalex-g1-canonical-work-time-spine-pilot-20260721`; the separate
benchmark root is
`/ssd/openalex-g1-canonical-work-time-spine-pilot-benchmark-20260721`.

The pilot report records:

- status `PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE`;
- exactly one source file, 1,000,000 raw and retained rows, and 312,250,942
  source bytes;
- zero partitions, `production_status=PRODUCTION_NOT_RUN`,
  `production_component_pass=false`, and `pass_g1=false`; and
- eight receipt-bound artifacts with receipt SHA-256
  `eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099`.

The read-only independent verifier returned
`PASS_READ_ONLY_INDEPENDENT_OUTPUT_VERIFICATION`, reconstructed the same eight
artifacts and one-file scope, and observed an unchanged output tree. The
external capture and benchmark record:

- wall time 11.69 seconds and maximum RSS 152,388 KiB;
- 85,543.199316 rows/s and 26,710,944.568007 source bytes/s;
- a simple scan-and-staging full-row projection of 5,754.907590 seconds
  (approximately 95.9 minutes); and
- benchmark SHA-256
  `8179b531ae8f70f28d707e0a043ac4aea13c3b5a831744fb683d6ec4962f5666`.

The timing format does not encode the command exit status or cryptographically
pair itself with the report. The operator observed both in one shell execution,
but the benchmark correctly retains
`external_command_exit_status=NOT_CAPTURED_BY_V1_FORMAT` and
`pilot_report_time_pairing=OPERATOR_ASSERTED_NOT_PROGRAMMATICALLY_VERIFIED`.
GNU time's filesystem counters are not physical-device byte accounting. No new
matching kernel storage event was found after the pilot. A 10:28 KST post-pilot
core SMART query also exited 0: overall health remained `PASSED`, the zero
critical-media-error attributes and empty error log were unchanged, and the
pre-existing `Runtime_Bad_Block` and `Command_Timeout` raw values did not
increase.

## 2026-07-21 bounded downstream evidence

The reviewed harness accepted no source path and read only the independently
verified 16,000,000-byte pilot staging payload from `/ssd`. It copied that
payload into a fresh private benchmark scratch tree and did not reopen the
OpenAlex source HDD. From 11:14:58 to 11:21:50 KST it completed three repeats
with exit status 0. Each repeat exercised the frozen runner's downstream
private kernels and retained exactly 9,185 benchmark artifacts; no production
finalizer or receipt builder was called.

Across the three repeats:

- D4096 processed 1,000,000 records at 46,507.987305 to 47,093.936769
  records/s, sampled a 4,103-descriptor peak, and observed at most 229,204 KiB
  sampled RSS. Its 4,096 close/fsync calls dominated the phase, with the
  largest repeat-level p99 at 0.005614316 seconds.
- R4096 published and decoded all 4,096 pilot-occupancy partitions at
  8,984.610449 to 9,312.868615 records/s. Every partition was nonempty, the
  largest contained 301 rows, no partition used more than one sort run, and
  the aggregate Parquet surface was 22,242,648 bytes per repeat.
- Each H9185 sweep hashed exactly 9,185 entries and 49,237,710 bounded bytes.
  Nine warm-or-cache-assisted sweeps observed 14,640.669980 to
  16,115.674928 artifact contracts/s.

The immutable operator-local result is
`/ssd/openalex-g1-canonical-work-time-spine-downstream-benchmark-20260721/openalex_g1_canonical_work_time_spine_downstream_benchmark.json`, with SHA-256
`6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109`.
The pilot receipt remained
`eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099`,
and the benchmark's before/after pilot tree-identity digests matched. The
benchmark itself records `downstream_capacity_gate_passed=false`,
`operator_acceptance_status=NOT_RECORDED`, and
`production_execution_status=PRODUCTION_NOT_RUN`.

The independent verifier
`scripts/verify_openalex_canonical_work_time_spine_downstream_benchmark_v1.py`
has SHA-256
`d1f9468f528034e64c20dca5b8b05f72ea5913f1c0571ab972de7c8503b6e0d3`.
It returned
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_BENCHMARK_VERIFICATION_NONPRODUCTION`
and reconstructed result SHA-256
`6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109`,
fingerprint
`5e12129cafbda0bc5918eedeb43c3ec6352357f3e78e67b0d6d6494f7a71e287`,
three repeats, 9,185 artifacts per repeat and 27,555 total, 1,000,000 rows per
repeat, and staging SHA-256
`ccbe4ffe751d148076c1711838aa15b976dfb305da4a0b28ef4937743b2a1165`.
It observed unchanged pilot, scratch, and result trees. Production, capacity-
gate, full-authorization, promotion, and G1 pass fields remained false.

These measurements are intentionally not converted into full-run row-rate
projections. The sample has about 244 rows per partition rather than the full
mean near 120,200; it cannot enter the multi-run heap path; its 49.2 MB H9185
surface is not the 26.3 GB one-sweep or 133.7 GB reconstruction envelope; and
the repeats were not independent quiet-window sessions with controlled cache
state or a one-sided confidence floor.

## 2026-07-21 synthetic downstream stress evidence

The stress successor accepted only the verified pilot and first downstream
benchmark as operator-local evidence inputs and did not open the source HDD. It
completed from 12:15:32 to 12:16:21 KST with exactly three repeats of each
scenario. The operator-local result is
`/ssd/openalex-g1-canonical-work-time-spine-downstream-stress-v1-20260721/openalex_g1_canonical_work_time_spine_downstream_stress_v1.json`, with SHA-256
`f93960649ff2742cc1b4f7efca82fd88b86c85c1ef5066f066d2507f9a766e46`,
fingerprint
`7f6ef339b7a7c64702a9b8dcb31d09d75439d90bc45c576e6cca43603c69960e`,
and tool SHA-256
`19bae4881bce030876d884a96500936d3b5a96ef97ccfffe28d7a3d605154b44`.

Observed minimum / median / maximum rates were:

- RMEAN, 120,189 rows and one run: stable sort
  12,372,760.963558 / 12,606,356.198867 / 12,931,891.542931 rows/s;
  merge, Parquet, and validation
  828,444.009429 / 6,204,904.491482 / 6,356,851.959592 rows/s; and
  2,405,052 output bytes per repeat.
- A2, 2,097,152 rows and two runs: stable sort
  18,301,352.648573 / 19,219,472.854577 / 19,608,897.698903 rows/s;
  merge, Parquet, and validation
  754,522.670378 / 762,531.406236 / 765,221.960315 rows/s; and
  41,927,598 output bytes per repeat.
- A8, 8,388,608 rows and eight runs: stable sort
  21,813,862.292422 / 22,807,153.771282 / 22,925,957.911998 rows/s;
  merge, Parquet, and validation
  695,693.770302 / 730,265.246055 / 742,416.120272 rows/s; and
  167,709,850 output bytes per repeat.

The retained surface has 60 mode-`0600` regular files totaling 1,654,322,493
bytes, no symlink, 58 mode-`0700` directories, and nine internal
`.work_time_spine_tmp` directories at mode `0775`. Each wider internal directory
is protected by a mode-`0700` scenario ancestor. The planned shell-level
`umask 077` was not inherited by the separately launched Python invocation;
effective containment is therefore attributed to the observed protected
ancestors and file modes, not to the planned umask.

Immediate run validation made the result
`UNCONTROLLED_WARM_OR_CACHE_ASSISTED`, and concurrent host load was uncontrolled.
The three repeats are not independent quiet-window sessions; no one-sided lower
confidence bound was computed and no observed rate is an adopted capacity
floor.

The external verifier
`scripts/verify_openalex_canonical_work_time_spine_downstream_stress_v1.py`,
frozen at SHA-256
`4ade14fd53f80a964ff8668b7f4e2391ea220402327fcc0fca9c0dd88e19373f`,
returned
`PASS_READ_ONLY_INDEPENDENT_BOUNDED_DOWNSTREAM_STRESS_VERIFICATION_NONPRODUCTION`.
It pins the exact result SHA-256 and fingerprint, enforces nested schema and
numeric invariants, and verifies the exact Parquet row-group layout with no
dictionary pages.
It reconstructed the same result SHA-256 and fingerprint plus scratch forensic
tree SHA-256
`41abadafd837b1c5a6d8a029a60102eb412625f31ecf2a2175cf658c0a2002f2`,
independently checked the 60 files and 1,654,322,493 bytes, and observed result,
scratch, pilot, and first-benchmark tree identities and content SHA-256 values
unchanged before and after verification.

The immutable producer result still records
`independent_verification_status=PENDING`; that is its historical publication
state, while the separate external verifier now supplies the live PASS. Neither
field changes the result's production, capacity-gate, full-authorization,
promotion, operator-acceptance, or G1 states, which remain false, not passed,
not granted, or not recorded.

## Historical required actions before the now-completed full run

These were the fail-closed conditions recorded on 2026-07-21. They are retained
for provenance; the 2026-07-22 execution used a fresh live preflight and
explicit authorization, and its receipt plus independent verification are the
current evidence.

1. Close or explicitly bound the remaining downstream evidence: actual full
   4,096-partition occupancy, fan-in above eight or adversarial merge, a
   representative full-byte finalization surface, and enough independent
   cache-characterized sessions to adopt conservative rate floors.
2. Obtain explicit operator acceptance for a separately documented downstream-
   capacity decision. The bounded result is independently verified, but the
   current capacity gate remains `NOT_PASSED`.
3. At the future execution time, repeat SMART/kernel, sustained exclusive
   source-I/O, disk, RAM, FD, device-separation, parent, and fresh atomic-output
   preflights.
4. Require a separate explicit full authorization and all three full CLI
   acknowledgements. The pilot does not grant any of them.

At the time of the readiness review, Science Atlas had an embedding file
memory-mapped on `/ssd`; its short observed I/O counters were unchanged. That
observation was not permission for concurrent heavy work. The later full run
used a fresh source-I/O check and excluded competing heavy source scans.
