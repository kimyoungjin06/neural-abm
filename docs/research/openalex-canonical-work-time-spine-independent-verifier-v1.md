# Canonical Work-Time Spine Independent Verifier v1

## Status and purpose

- Implementation: `IMPLEMENTED_AND_SYNTHETIC_TESTED`
- Pilot output verified: `YES — 2026-07-21 OPERATOR_LOCAL_NONPROMOTABLE`
- Production output verified: `NO`
- Production status: `PRODUCTION_NOT_RUN`

This document defines the repository-only, read-only verifier implemented by
`scripts/verify_openalex_canonical_work_time_spine_v1.py`.

The verifier is a successor-side check of a completed canonical work-time-spine
output. It does not import or call the production v3 runner, the synthetic v2
predecessor, or the frozen production YAML. Its ABI constants and validation
logic are implemented separately so that a defect in the production runner's
terminal-resume validation is not automatically repeated by importing that
validation code.

The verifier independently copies the reviewed current-contract constants into
its own versioned implementation. These include the exact 493-task production
record commitment, the one-task pilot-prefix commitment, source totals, resource
envelope, frozen repository input paths and hashes, content/temporal parent
anchors, and the 2026-03-30 classifier cutoff. It does not read the YAML, import
the runner, or reopen parent artifacts at runtime. Any intentional change to
those constants requires a new verifier version and review; merely producing a
self-consistent output with a `FULL` scope label is insufficient.

The verifier never starts a source scan, pilot, or production execution. It
accepts one already completed output root and emits its result only to stdout.
It does not create a verifier receipt inside that root.

## Dated operator-local verification

On 2026-07-21 the default full-validation mode verified the completed
operator-local pilot root
`/ssd/openalex-g1-canonical-work-time-spine-pilot-20260721`. It returned
`PASS_READ_ONLY_INDEPENDENT_OUTPUT_VERIFICATION`, reconstructed the exact
one-file nonpromotable scope and eight receipt-bound artifacts, and observed an
unchanged output tree. The verified receipt SHA-256 was
`eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099`;
the result fingerprint was
`8464cbec9ff5d611d0f2779364b7b3ee836b49b31cead32e39aae81d90937a97`.

This is a pilot-output verification, not a production-output verification. It
does not change `production_component_pass=false`, `pass_g1=false`, or
`PRODUCTION_NOT_RUN`.

## Invocation

The strong/default mode streams every published Parquet partition:

```bash
uv run python scripts/verify_openalex_canonical_work_time_spine_v1.py \
  --output-root /path/to/completed-output
```

An explicitly weaker control-and-hash-only mode exists for operational triage:

```bash
uv run python scripts/verify_openalex_canonical_work_time_spine_v1.py \
  --output-root /path/to/completed-output \
  --parquet-validation hash-only
```

`hash-only` has a distinct success status and must not be reported as the full
independent verification pass.

## Read-only and integrity policy

The verifier opens output files only with `O_RDONLY | O_NOFOLLOW`. It does not
take or modify the producer's lock, repair a receipt, quarantine a corrupt
artifact, rewrite canonical JSON, or delete undeclared entries.

It requires:

- a private `0700` output root and directories owned by the verifier user;
- regular, single-link `0600` files owned by the verifier user;
- no symlink or special-file entry in the output tree or its path components;
- the exact seven terminal directories and no temporary tree;
- the exact artifact file set derived from the plan, plus only the receipt and
  empty `audit.lock`;
- stable descriptor and path identities while each file is read; and
- an unchanged whole-tree identity snapshot after successful verification.

A mismatch is reported without changing the evidence. A concurrent mutation is
an error, not something the verifier attempts to repair.

## Independently reconstructed checks

### Terminal control plane

The verifier independently checks:

- strict JSON decoding with duplicate-key and non-finite-number rejection;
- exact pretty, sorted UTF-8 JSON plus one trailing LF;
- the versioned plan, preflight, checkpoint, manifest, report, state, and
  receipt schema families for full, pilot, or closed-fixture scope;
- the exact plan, scan-checkpoint, partition-checkpoint, report, preflight,
  state, receipt, and manifest ABI key sets;
- the exact full/pilot task count, lexical task order, parent-derived task-record
  digest, source totals, frozen input hashes, parent anchors, and resource gates;
- the compact-canonical plan task fingerprint;
- the compact-canonical report result fingerprint;
- the terminal state's exact plan/result/receipt binding;
- receipt status, scope, production-pass boundary, runtime fingerprint, tree
  commitments, completion ABI, artifact count, and exact artifact contracts;
- every receipt artifact's byte count and SHA-256; and
- relative-path normalization and absence of absolute operator-path strings in
  the terminal JSON surface.

The recorded preflight evidence is checked against the plan's selected disk,
RAM, and nofile requirements. The current output device is checked against the
recorded output device. The verifier does not reopen the source root merely to
refresh the source-device observation.

### Scan and staging bridge

For each task, the verifier derives the scan checkpoint and staging shard names
from the SHA-256 of the relative source path. It checks checkpoint self-hashes,
task and temporal-parent bindings, metric hashes, recorded source identities,
staging byte and SHA-256 contracts, and scan accounting.

It streams every 16-byte staging record independently, verifies reserved and
null-sentinel constraints, independently derives calendar year from the encoded
day, and recomputes the exact six-way disposition under the frozen 2026-03-30
classifier. It reconstructs disposition and publication-year counts and the
framed row-record SHA-256. The staging manifest and scan/staging tree digests
must then reconstruct exactly. A closed fixture created with the producer API's
nondefault snapshot argument is outside this verifier v1 ABI.

### Partition and Parquet bridge

In the default `full` verifier mode, each Parquet partition is streamed with
bounded batches. The verifier checks:

- the exact Arrow schema, footer row count, uncompressed columns, and absence
  of column statistics;
- partition modulo membership and globally strict increasing work IDs;
- day/year nullability, temporal-disposition, and calendar-year consistency;
- decoded minimum and maximum IDs, publication-year extrema, nullability,
  disposition counts, and the W0 cross-tab; and
- equality of the independently reconstructed decoded record digest with both
  the pre-Parquet and post-Parquet checkpoint commitments.

It then reconstructs all partition checkpoint self-hashes, the partition
checkpoint tree, the spine manifest, and the spine data tree. Report identifier
and publication accounting must agree with the independently aggregated scan
and partition evidence.

The verifier does not externally sort the staging shards again. Therefore it
does not independently prove row-for-row staging-to-partition derivation beyond
the producer's receipt-bound partition checkpoint bridge, plus independently
checked row totals, disposition counts, year extrema, W0 accounting, partition
membership, strict ID uniqueness, and both staging and Parquet record digests.
Doing that exact bridge again would require a second bounded external-sort
execution and is outside this read-only verifier.

## Independence and claim boundary

The independence is implementation-level only. It means that the verifier does
not reuse the production runner's Python module or call its validation helpers.
It does **not** establish any of the following:

- trusted-launcher or process-isolation evidence;
- attestation of the exact code loaded in the producer or verifier process;
- independence of the shared Python, NumPy, PyArrow, operating-system, or
  hardware stack;
- replay of the original three-column source projection;
- current source-file identity or source semantic validity;
- current content-baseline or temporal-parent artifact availability;
- historical content identity, W0 entity validity, citation chronology,
  author/candidate validity, G-1, G0, empirical G1, an NABM effect, or a
  scientific effect.

Accordingly, a full verifier pass strengthens the terminal artifact and
row-commitment evidence. It does not by itself promote a closed fixture or pilot,
turn `PRODUCTION_NOT_RUN` into a production pass, or unblock downstream scientific
claims.

The filesystem checks use `O_NOFOLLOW`, component `lstat` checks, stable
descriptor/path identities per read, receipt readback, and whole-tree identity
snapshots before and after verification. They detect ordinary concurrent drift,
but they are not a hostile-process isolation boundary: without a producer lock
or Linux `openat2`-style descriptor-relative resolution, a privileged adversary
that swaps an ancestor transiently and restores all identities remains outside
the guarantee.
