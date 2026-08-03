# OpenAlex Canonical Work-Time Spine v2 Production Design

Status: `IMPLEMENTED_PRODUCTION_CAPABLE_PRODUCTION_NOT_RUN`

Production status: `PRODUCTION_NOT_RUN`

This note defines the production-capable, external-memory successor to the
closed synthetic canonical work-time spine. It is paired with
`experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml` and the
new runner path `scripts/audit_openalex_canonical_work_time_spine_v3.py`.
There is no production result or receipt yet. The executable contract and a
future execution result are separate evidence states.

The v0 and v1 configurations, runners, tests, and design notes remain
byte-exact historical surfaces. The new configuration loads the v1 scientific
contract by its exact SHA-256 and overlays only the production execution,
resource, resume, and completion rules. It also pins the v2 synthetic runner
at SHA-256
`6a2d1d87dd4a3e7d3c97056fabbe951a1528cfd62d8448acc74ae337f344b7b1`.
Synthetic success remains synthetic evidence and cannot be relabeled as a
production pass.

The production runner and configuration are frozen without a circular hash
dependency. The exact runner SHA-256 is
`bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280`.
The semantic configuration SHA-256 is
`3e21393832cc7c2624177fd7f73f0c1ddb80ece21dfd2b4d84e173a7f32f741a`;
it is calculated from canonical compact UTF-8 JSON with sorted keys after only
`runner_contract.required_tool_sha256` is replaced by 64 zeroes. That semantic
digest is deliberately not stored inside the configuration. Production and
pilot modes require this repository-default configuration path and verify both
the runner hash and the independently calculated semantic configuration hash
before execution.

## Why a separate production successor is necessary

The synthetic runner validates the 16-byte record ABI, disposition logic,
4,096-way product, checkpoints, and receipt-last completion on a small closed
fixture. Its execution strategy also retains task records, the global numeric
ID collection, and all partition record lists in memory. Global duplicate
validation uses repeated list counting. Those operations are valid under the
10,000-row synthetic cap but are not a production algorithm for 492,293,206
retained rows.

Changing only a CLI guard would therefore create an unbounded memory path and
an avoidable quadratic duplicate check. The v2 production contract replaces
those operational parts with one-pass streaming staging, bounded partition
spooling, bounded chunk sorting, and a direct k-way merge. It does not modify
the already-audited temporal-v3 parent or the v1 scientific classification
contract.

## Production pipeline schematic

```text
content receipt ----\
                     +--> exact control validation + live stat cross-binding
temporal-v3 receipt -/                         |
                                               v
493 main Parquet files -- own three-column source scan, exactly one pass
                                               |
                                               v
                          493 verified 16-byte staging shards
                                               |
                                  one staging pass, modulo 4096
                                               v
                            4096 bounded raw spool files
                                               |
                         <=1,048,576 rows per sorted binary run
                                               v
                   <=470 runs for even the worst possible partition
                                               |
                               one-pass k-way merge to Parquet
                                               v
                 4096 partition checkpoints + 4096 Parquet files
                                               |
                         streaming final validation of 9,185 artifacts
                                               v
                              receipt written last, or no PASS
```

The workload is CPU and storage I/O. It uses no GPU.

## Explicit execution authority

The repository configuration alone never authorizes a full source scan. Full
production mode requires all three CLI acknowledgements together:

```text
--execute --authorize-full --acknowledge-exclusive-source-scan
```

Full mode refuses `--max-files` and any test-fixture flag. The third
acknowledgement records that the operator checked for a competing content
freeze, temporal audit, or other heavy scan of the same HDD-backed source.
Production full and pilot modes fix the source batch at 262,144 rows and the
sort chunk at 1,048,576 rows; CLI overrides are rejected because the RAM
envelope and byte-level plan ABI are calculated from those exact values. A
closed fixture may choose smaller positive values but may not exceed either
bound.

Disk, available RAM, file-descriptor capacity, distinct source/output devices,
parent receipts, paths, and live source identities are checked before any
source payload is opened. The distinct-device gate applies to full and pilot
production-source modes; a closed fixture is exempt.

A lexical-prefix production pilot instead requires `--execute --max-files 1`;
exactly one source file is selected. Any other `--max-files` value is rejected.
It is always
`PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE`; it cannot
accept the full authorization flag, use the production schema family, mint
the full receipt basename, or claim global accounting or uniqueness.

Closed test fixtures are Python-API-only and use the distinct literal status
`PASS_TEST_FIXTURE_CANONICAL_WORK_TIME_SPINE_EXTERNAL_MEMORY_CONTRACT_ONLY`.
Their report, state, and output schemas are disjoint from both pilot and
production schemas. A test-fixture result has
`production_component_pass=false` and forbids the production root ID.

### Pre-full operational gates remain open

Production-capable code is not evidence of acceptable real-HDD throughput. At
contract freeze, full-execution readiness was
`BLOCKED_PENDING_OPERATOR_GATES`, and the contract update itself ran neither
the pilot nor the full job.

The frozen contract required exactly one one-file nonpromotable production
pilot before any new full authorization. The post-freeze overlay below records
the later completion of that pilot requirement only; it does not pass the
other full gates. Because the runner report does not itself establish timing,
wall time must be captured externally, for example with:

```text
/usr/bin/time -f 'elapsed_seconds=%e\nmaximum_rss_kib=%M\nfilesystem_inputs=%I\nfilesystem_outputs=%O' \
  -o <operator-local-benchmark-dir>/openalex_g1_canonical_work_time_spine_pilot_time.txt \
  uv run python scripts/audit_openalex_canonical_work_time_spine_v3.py \
  --execute --max-files 1 --output-dir <operator-local-pilot-output>
```

A separate transformation was still unimplemented when this frozen contract
was recorded. Its reviewed successor must combine that capture with
`scope.raw_rows` and `scope.source_bytes` from the pilot report. It writes the
operator-local
`openalex_g1_canonical_work_time_spine_pilot_benchmark.json` using schema
`openalex_g1_canonical_work_time_spine_pilot_benchmark.v1`, outside the
repository, release package, pilot output root, and runner receipt. It records
the status `PILOT_BENCHMARK_CAPTURED_OPERATOR_LOCAL_NONPROMOTABLE` and
pilot wall seconds, rows/s, source bytes/s, and the simple full-scan
extrapolation `492293207 / rows_per_second`. That extrapolation excludes
partition sorting, final validation, and storage contention; it is a planning
diagnostic, not a full-runtime result or a programmatic parent of the eventual
receipt. The frozen YAML field therefore remains
`PENDING_IMPLEMENTATION_AND_REVIEW`, and that historical contract snapshot
retains pilot status `NOT_RUN`. The post-freeze overlay below records the later
operator-local pilot without silently rewriting the frozen YAML.

The one-file pilot exercises scan and staging only. It does not measure the
4,096-handle spool and NumPy demultiplex, external sort and Parquet
publication, or 9,185-artifact final rehash. A separate downstream-pipeline
capacity-review field in the frozen YAML therefore remains `NOT_RUN`. Before
full authorization, the
operator must accept either a bounded, nonpromotable staging-derived downstream
benchmark or a conservative analytic time/I/O estimate. Pilot extrapolation
alone is insufficient and this review is not a programmatic receipt parent.

Four other adjacent gates also remain explicit:

- a fresh read-only SMART or equivalent source-drive health review was
  `NOT_RUN` at contract freeze, required because of the prior `/dev/sda`
  read/reset event;
- a source process/I/O quiet-window check was `NOT_RUN` at contract freeze, and
  concurrent heavy scans are not allowed;
- a separately implemented and reviewed read-only verifier was
  `PENDING_IMPLEMENTATION_AND_REVIEW` at contract freeze; the runner's
  self-validation is not independent; and
- a new explicit full authorization is `NOT_GRANTED`. The authorization for
  implementation work in this turn does not authorize the full source run.

These are operator gates and recommendations, not additional scientific
parents and not current PASS evidence.

### Post-freeze operational update — 2026-07-20 through 2026-07-21

The frozen YAML above is unchanged and intentionally retains the original
`PENDING_IMPLEMENTATION_AND_REVIEW` and `NOT_RUN` fields. The following newer
repository-only surfaces are an operational overlay, not a silent contract
refreeze or a gate promotion:

- the pilot benchmark converter is implemented, synthetic-tested, and reviewed.
  A separately authorized one-file pilot completed on 2026-07-21 with runner
  exit status 0, a valid nonpromotable receipt, and an implementation-level
  independent verifier pass. Its 11.69-second capture implies a 5,754.907590-
  second simple full-row scan-and-staging projection, but the v1 GNU `time`
  format cannot programmatically prove exit status, command identity, fresh
  execution, or report/capture pairing, so the artifact remains a
  manual-review-only, nonpromotable diagnostic;
- the conservative analytic downstream-capacity review is implemented, but its
  downstream empirical benchmark is `NOT_RUN`, its gate is `NOT_PASSED`, and
  operator acceptance is `NOT_RECORDED`. The one-file scan-and-staging pilot
  does not satisfy this downstream requirement;
- the implementation-level independent verifier and its closed-fixture tests
  now exist and passed the pilot output, but no production output or receipt
  exists to verify; and
- the host snapshot was followed by an authorized SMART maintenance window. A
  fresh short and extended self-test both completed without error, key media
  attributes stayed zero, error logs stayed empty, and no new matching kernel
  event was found. This is `PASS_POST_TEST_MEDIA_SCREEN_CURRENT_SNAPSHOT`, not
  proof of the operator-reported prior RAID/SATA contact-fault attribution. The
  pilot's quiet window and fresh output preflight passed only for that bounded
  invocation and do not establish a future full-run window.

Consequently the one-file pilot is complete but nonpromotable. Full execution
remains `NOT_RUN`, the production receipt remains unavailable, downstream
capacity remains `NOT_PASSED`, and explicit full authorization remains
`NOT_GRANTED`.

## Exact production parents

Parent and owned-control JSON use family-specific exact byte ABIs; semantic JSON
equivalence is insufficient and a serializer from another family is rejected.

| JSON family | Required publication ABI |
| --- | --- |
| temporal v3 parent | `CANONICAL_JSON_SORTED_KEYS_COMPACT_UTF8_PLUS_LF` (`allow_nan=false`) |
| content-freeze parent | `JSON_SORTED_KEYS_INDENT_2_UTF8_PLUS_LF` |
| canonical-spine owned controls | `JSON_SORTED_KEYS_INDENT_2_UTF8_PLUS_LF` |

Every parent control file is hashed and parsed from the same stable regular-file
snapshot, then compared with its own producer-family serialization. These
format checks do not open any source Parquet payload.

The content baseline is the current full receipt with these anchors:

| Commitment | Exact value |
| --- | --- |
| outputs receipt SHA-256 | `082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12` |
| status | `PASS_CURRENT_CONTENT_BASELINE_COMPONENT` |
| state | `COMPLETE` |
| task fingerprint | `f915cade4ce6581832c46957a2ad0a37ceabe2296099f0faeaac56ffa035693e` |
| result fingerprint | `12ebd47d88973f37f4ed5b6f8cdd5901eb5c0b7005a41a6ab3b96423a40bbdf1` |
| checkpoint tree | `e509867989cec9dfd398221ee5d7abd69dc73fa7330acf9302413184d786bfef` |
| content tree | `73bb8b0409c782ce5964d480ac6c6cecb04bab8c09df31999e581000b79ccff6` |
| full scope | 21,019 files; 988,404,537,807 bytes |

The temporal parent is only the exact full v3 pass:

| Commitment | Exact value |
| --- | --- |
| outputs receipt SHA-256 | `2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe` |
| status | `PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE` |
| config SHA-256 | `672940908090fedd0b562dd938ad4583f5e8238830fc1ff5b613be9618f033f7` |
| tool SHA-256 | `76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d` |
| task fingerprint | `5d86a797481a14e95d08ecbe333bb207ff476bf7dc49886aec1db7b5b366001d` |
| result fingerprint | `a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5` |
| checkpoint tree | `725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392` |
| source tree | `da3fa570a5fb49a36668dffdee008c2fea4d4b6fd5a3f7f17873df78b603b2a0` |
| full scope | 493 files; 492,293,207 raw rows; 150,062,449,219 bytes |

Temporal v1 and v2 have no fallback path. The temporal state is correctly
accepted as `COMPLETE_PENDING_RECEIPT` only when its receipt exists and
satisfies `RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED`.
Parent code is loaded only from this repository root. A temporal-repository
root override is not an execution authority and any nondefault value is
rejected before parent control access. The imported temporal producer module
is treated as immutable: the spine runner may not rewrite any imported global.
Its `REPO_ROOT`, the current repository root, and the requested temporal root
must resolve identically. Its `PRODUCTION_CONFIG_SHA256` and
`FROZEN_CONFIG_SHA256` constants, the observed temporal configuration digest,
and the anchored parent configuration digest must also all be exactly equal.

## Parent validation does not rescan source payload

Parent validation opens the parent control artifacts and live stat metadata.
It recomputes receipt, artifact, task, result, checkpoint-tree, source-tree,
and cross-parent commitments from those control surfaces. It does not open
source payload columns. Each parent control JSON is opened as a regular,
no-follow file snapshot once; its SHA-256 and parsed JSON are derived from the
same bytes so a concurrent replacement cannot split the checked and parsed
versions.

In particular, the old synthetic validator's five-column payload rescan of
`id`, `publication_date`, `publication_year`, `created_date`, and
`updated_date` is forbidden. The production component performs its own
three-column source scan of only `id`, `publication_date`, and
`publication_year` exactly once in a successful fresh full run. Each file's
nine-field identity is checked before and after that scan. Finalization checks
the live stat again but does not reopen payload.

A valid resumed scan checkpoint is accepted only after its checkpoint,
staging shard, parent bindings, and current nine-field stat all match. Reuse
adds zero source-payload passes. A corrupt reusable scan artifact fails closed
and requires a fresh output root; the runner does not silently rescan and
overwrite it.

## Streaming scan and staging

Source tasks are processed in lexical relative-path order, one source file and
one projected record batch at a time. The batch size is 262,144 rows. The
implementation may not construct an in-memory collection of all records, all
numeric IDs, or task/partition row payloads. It may retain the
bounded control metadata for 493 scan checkpoints and 4,096 partition
checkpoints, at most 4,589 small dictionaries in those two checkpoint
collections; those are not row payloads. The 4,589 ceiling is not asserted for
other, separately bounded control objects such as manifest or receipt-artifact
metadata.

The scan enforces separate retained-buffer caps of 256 MiB for the Arrow
batch, 256 MiB for its serialized IPC projection, and 4 MiB for the encoded
16-byte records. Their simultaneous retained-payload bound is therefore
541,065,216 bytes. Every individual cap and the combined cap must pass before
another buffer grows, and the prior encoded buffer is explicitly released
before the next batch is serialized. This bound describes retained payload
buffers, not total RSS or unobserved native-allocator scratch.

Each valid row is encoded immediately into the inherited fixed 16-byte
little-endian ABI. A task first writes an uncommitted file under
`.work_time_spine_tmp/staging`, fsyncs and validates it, then atomically
publishes one shard under `staging/main`. The scan checkpoint follows the
verified shard. The global staging manifest appears only after all 493 task
checkpoints validate and all inherited identifier, publication, disposition,
and parent-metric invariants close.

The exact staging payload is:

```text
492,293,206 * 16 = 7,876,691,296 bytes
```

Local staging shards are resumable operator data, not a released product. No
manifest, partition stage, report, or receipt can promote a run containing a
rejected row.

## Bounded modulo spool and external sort

Verified staging is read once. Every record is assigned by exact numeric
modulo to one of 4,096 raw files under `.work_time_spine_tmp/spool`:

```text
partition_id = work_id_numeric mod 4096
```

The spool writer uses stable NumPy batch demultiplexing in chunks of at most
1,048,576 records. It keeps up to all 4,096 partition handles open, with 8,192
records or 131,072 bytes buffered per handle and a 536,870,912-byte aggregate
buffer bound. Handle eviction and reopen are forbidden; each touched partition
is closed and fsynced at most once when spooling finishes. This avoids the
near-per-record fsync churn that a small LRU would cause under uniform modulo
traffic. The sum of raw spool payload is bounded at the same 7,876,691,296
bytes as staging. Spool and run files are never reused across invocations,
even when apparently complete. Every nonterminal invocation removes the
private temporary tree and rebuilds it from verified staging.

One partition is sorted at a time in chunks of at most 1,048,576 records, or
16,777,216 encoded bytes. Sorting uses a 16-byte NumPy structured dtype with
the inherited field offsets and stable `argsort` on `work_id_numeric`; a
Python `Record` object or tuple per row is forbidden. The 128 MiB working
value is a declared engineering preflight estimate covering input, keys,
permutation, output, and native sort scratch; it is not a directly measured
runtime upper bound.
Parquet row batches likewise use fixed-width NumPy columns without Python row
objects. Runs remain in the 16-byte ABI under
`.work_time_spine_tmp/runs`. Even if every retained ID fell in one partition,
there would be 470 runs, below the 512-open-run limit. A one-pass k-way merge
writes directly to the atomic Parquet candidate under
`.work_time_spine_tmp/sorted`; intermediate merged binary runs are forbidden.
The aggregate external-run payload is therefore bounded at 7,876,691,296
bytes.

IDs must be strictly increasing during merge. Equal numeric IDs necessarily
share the same modulo partition, so a strict-order failure detects a duplicate
without a global in-memory ID set. Pre-Parquet and decoded-Parquet canonical
record counts and digests must match. All 4,096 output partitions and all
4,096 partition checkpoints are required, including valid empty Parquet files.

The decoded-Parquet validator also binds each partition's non-null
publication-year extrema and W0 disposition counts into its checkpoint. Reuse
revalidates those fields. For a full run, their bounded global aggregation
must recover publication-year minimum 1000, maximum 2050, and exactly one W0
row. The full report includes the extrema and the observed W0 disposition
cross-tab. Its category is deliberately not precommitted, and
`w_zero_entity_validity` remains `UNRESOLVED`.

The complete `.work_time_spine_tmp` tree is absent before final validation and
is forbidden from both the completion surface and receipt.

## Exact resource preflight

The disk budget includes every simultaneously relevant class instead of only
the durable product:

| Class | Formula | Bytes |
| --- | ---: | ---: |
| staging | `492293206 * 16` | 7,876,691,296 |
| partition spool | `492293206 * 16` | 7,876,691,296 |
| external runs | `492293206 * 16` | 7,876,691,296 |
| all Parquet outputs | `492293206 * 32 + 4096 * 131072` | 16,290,253,504 |
| control artifacts | fixed upper bound | 2,147,483,648 |
| largest next atomic artifact | `492293206 * 32 + 131072` | 15,753,513,664 |

The unmultiplied worst case is 57,821,324,704 bytes. Applying the exact 1.5
safety multiplier and then adding an 8 GiB reserve gives the base operational
floor:

```text
ceil(1500 * 57,821,324,704 / 1000) + 8,589,934,592
= 95,321,921,648 bytes
= 88.775 GiB
```

A fresh full invocation must remain restartable even after the worst resident
artifact set has consumed space. It therefore adds one unmultiplied worst-case
resident set to the base floor:

```text
95,321,921,648 + 57,821,324,704
= 153,143,246,352 bytes
= 142.626 GiB
```

That is the required fresh-full threshold; 160 GiB (171,798,691,840 bytes) is
the preferred fresh-full floor. A nonterminal full resume requires the base
95,321,921,648-byte threshold. The plan binds both thresholds, and an
invocation selects one without changing the plan. No completed-artifact
subtraction is permitted. This deliberately conservative rule avoids making
restartability depend on complex artifact-subtraction accounting.

The `--resume` flag alone cannot select the lower threshold. The invocation is
`NONTERMINAL_RESUME` only after an existing nonterminal plan and its binding
surface have validated, including the existence of every checkpoint/artifact
pair in the state's declared completed scan and partition prefixes. A missing
pair fails before the lower threshold can be selected. A missing output, an
empty output directory, or an output containing only the newly acquired lock
remains `FRESH` and must pass the 153,143,246,352-byte threshold.

The RAM preflight sizing estimate uses the executable 541,065,216-byte
simultaneous scan-payload cap, declared 128 MiB native-sort and 128 MiB NumPy
spool-demultiplex estimates, 512 MiB spool buffers, a declared 16 MiB
Parquet-writer estimate, a declared 1 MiB merge-heap estimate, and an 8 GiB
interpreter/allocator reserve. The phase maximum is taken across scan, sort,
and demultiplex work, so the resulting sizing value remains 9,685,696,512
bytes. It is not a measured total-process RSS or an exact native-allocator
scratch bound. The plan preserves that calculated value separately from the
more conservative 16 GiB required-available-RAM gate; 32 GiB remains
preferred. The soft file-descriptor limit must be at least 8,192.
The runtime fingerprint additionally binds the NumPy version, alongside
PyArrow, Parquet C++, Python, and machine fields, into the plan, task
fingerprint, receipt, and resume equality gate.

Disk-side quantities are executable checks: staging, spool, and run bytes;
per-partition and aggregate Parquet bytes; aggregate control bytes; and the
largest atomic candidate must stay within their plan-bound formulas. The RAM
engineering estimates remain plan- and runtime-fingerprint-bound even though
they are not presented as direct memory measurements.
The per-partition Parquet formula is rechecked for fresh, resumed, and final
artifacts. The control aggregate includes `audit.lock` and the canonical
receipt; receipt bytes are computed and checked before receipt publication.

The live checks are recorded in `preflight_evidence.json` using a distinct
production, pilot, or fixture schema. It binds the task fingerprint; the
`FRESH` or `NONTERMINAL_RESUME` invocation kind; observed disk and the
`selected_required_available_disk_bytes` threshold; observed and required RAM
and nofile values; the
atomic rename/fsync/readback PASS; source and output device IDs; the
distinct-device PASS or closed-fixture exemption; and path redaction. The file
is receipt-bound but intentionally excluded from plan, task, and result
fingerprints because its live values are invocation-specific. An initial or
nonterminal resumed invocation refreshes it after a new successful preflight
and before source payload access. Terminal resume validates its receipt-bound
bytes and SHA-256 without rewriting it.

## Resume, corruption, and output security

Resume requires exact plan bytes. The plan binds the new config and tool, the
v1 foundation config, the v2 synthetic runner dependency, runtime fingerprint,
both exact parent receipts, task identities, resource bounds, and execution
mode.

A scan or partition checkpoint without its artifact is not reusable, and an
artifact without its checkpoint is not reusable. A mismatch follows
`FAIL_CLOSED_NO_AUTOMATIC_OVERWRITE`. Private spool and run work is never a
reusable cross-invocation artifact: incomplete or complete temporary work is
removed, then rebuilt from verified staging on every nonterminal invocation.
The completed counts in `run_state.json` are lower bounds on pairs that must
exist: deleting both members of any declared-complete pair is corruption, not
permission to regenerate it. Complete pairs ahead of state may arise in a
checkpoint-before-state-update crash window, but every such pair still enters
the normal exact reuse-validation path.
Only a fresh invocation writes the initial `RUNNING_SCAN` state with zero
progress. A nonterminal resume preserves its already validated monotonic state
until normal reuse or new publication advances it, so a second crash cannot
erase the durable evidence that justified the lower disk threshold.

A valid completed receipt is terminal. Terminal resume validates every
receipt-bound artifact before opening any source payload and never rewrites a
valid completion. It validates the canonical on-disk plan, report, receipt,
and exact receipt-declared artifact set before comparing them with the current
invocation. If that self-consistent terminal surface is merely incompatible
with the current expected plan, runtime, config, or tool identity, the runner
raises `CURRENT_INVOCATION_INCOMPATIBLE_NO_WRITE` and preserves every completed
byte. A current source-stat change or parent-control drift detected after the
disk surface self-validates follows the same no-write compatibility policy.
`KeyboardInterrupt` or `SystemExit` during terminal validation likewise
preserves every byte and is re-raised; operator interruption is not evidence
of intrinsic corruption. In fact, terminal resume is entirely read-only:
canonical serialization, hash, tree, artifact-set, or reconstruction mismatch
raises `TERMINAL_VALIDATION_FAILED_NO_WRITE` while preserving the observed
bytes. Automatic invalidation or repair is forbidden; destructive quarantine
or repair requires an explicit future operator action. Receipt-first
invalidation remains only within the same invocation when its own incomplete
finalization fails: remove the receipt, write
`FINALIZATION_FAILED_NO_VALID_RECEIPT` without a result fingerprint, then
remove the JSON and Markdown reports.

The output must be outside the repository, source, and both parent output
roots. Source, output, and parent path ancestors may not be symlinks.
The overlap, ancestry, and symlink policy is a read-only gate that runs before
the atomic-filesystem probe, directory creation, lock creation, or any other
write. Terminal resume repeats the same gate; a receipt does not bypass it.
Only a path that passes this policy may host the atomic-rename/fsync probe.
Final exact-set validation derives the allowed file set from the plan and uses
one frozen seven-directory allowlist for every scope; an undeclared empty
directory is a failure, not an ignorable entry.
Receipt presence and invocation classification observed before acquiring the
exclusive lock are not authoritative. Immediately after lock acquisition the
runner rechecks them, before any preflight-evidence, state, or temporary-tree
mutation. If another writer published the receipt in that interval, the
invocation routes directly to terminal validation and cannot mutate or
invalidate the newly completed output merely because its pre-lock view was
stale.
For a receipt already present at invocation start, the lock itself is also a
read-only terminal artifact: it must preexist as an operator-owned regular
file with mode `0600`, and is opened without create or chmod. A missing lock
fails without creating one. A valid terminal resume preserves the nine-field
stat identity of every file, including `audit.lock`.
Source payload opens use `O_NOFOLLOW`; private temporary-record creation adds
`O_EXCL`; and atomic control-JSON writes use directory descriptors and
no-follow opens. Phase boundaries revalidate source identities, while
published artifacts are revalidated as regular non-symlink files. The
contract does not claim that every data operation is descriptor-relative.
The atomic-rename/fsync/readback probe additionally requires its target to
have a hardlink count of one, but this version does not claim a hardlink-count
check for every published artifact. An exclusive non-implicit lock, `0700`
directories, `0600` files, exact entry sets, and path-redacted control
artifacts remain required. Individual work IDs and all staging or spine data
remain operator-local and are forbidden from repository and release packages.
The output root itself is rechecked as operator-owned mode `0700` at phase,
final, and terminal-resume boundaries; validating only descendant modes is not
sufficient.

## Exact 9,185-artifact completion surface

The full receipt binds seven fixed artifacts:

- `audit_plan.json`;
- `run_state.json`;
- `preflight_evidence.json`;
- `staging_manifest.json`;
- `work_time_spine_manifest.json`;
- `work_time_spine_audit.json`; and
- `work_time_spine_audit.md`.

It additionally binds 493 scan checkpoints, 493 staging shards, 4,096
partition checkpoints, and 4,096 Parquet partitions:

```text
7 + 2*493 + 2*4096 = 9,185 regular artifacts
```

`audit.lock` and `work_time_spine_audit.outputs.json` are the two declared
receipt exclusions. Final validation streams across all 9,185 artifacts and
may not reconstruct all records or IDs in memory. The receipt is written last
only after the exact file set, bytes, hashes, schemas, row counts, trees,
digests, parents, accounting, gate states, and claim boundary all validate.
For production and pilot runs, the immediately pre-receipt gate also repeats
stable-snapshot validation of the repository-default configuration hash and
nine-field identity, the executing v3 tool hash and identity, the foundation
configuration SHA-256, and the pinned v2 predecessor tool SHA-256. The plan's
v3 tool hash and identity come from one same-file-descriptor snapshot and that
hash must already equal the configured runner pin. Thus a repository edit
during a long run cannot produce a receipt that silently cites a changed
repository file. This is frozen-file provenance, not byte attestation of the
Python process's already-loaded code objects; neither loaded-code identity nor
process-image identity is claimed. Closed fixtures are explicitly exempt from
these repository-frozen-input gates.
All substantive reconstruction and exact-set validation occurs before receipt
publication. After its atomic write, operations are read-only and limited to
canonical receipt readback and byte equality, receipt size/headroom checking,
and an exact file/directory surface and mode recheck. Terminal resume also
requires the receipt bytes to equal the canonical published serialization,
rather than merely parsing to the same JSON value. An operator
`KeyboardInterrupt` or `SystemExit` after the atomic receipt publication
preserves the published bytes and is re-raised. A failed pre-receipt
validation leaves no authoritative receipt.

## Claim and downstream boundary

Only the exact full production schema family may emit:

```text
PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_TEMPORAL_V3_PARENT
```

That future status is not a current result. If eventually earned, it may
establish exact current-byte retained work-ID accounting, deterministic
publication-time dispositions, strict modulo partitions, and a checksummed
row-level spine commitment. It still cannot establish historical content
identity, validate `W0` as an entity, or claim row-level identity to a
temporal-v3 artifact that does not exist.

It also does not establish citation chronology or observation time, author or
candidate validity, cutoff-aware G-1 readiness, G0 maturation, empirical G1,
an NABM effect, or any scientific effect. Those gates retain their explicit
`UNVERIFIED`, `BLOCKED`, `UNRESOLVED`, `PENDING`, and `NOT_RUN` states.

Finally, author-v1 and citation-v1 are byte-bound to the synthetic canonical
spine contract. A future production spine receipt does not silently upgrade
them. Each consumer needs a new versioned config, runner, tests, and exact
production-receipt binding before it can use this product or change status.
