# Pivot-feedback cutoff-primitives source replay v2

Status:
`PASS_INDEPENDENT_FULL_SOURCE_PROJECTION_REPLAY_AND_CANONICAL_TIME_VERIFICATION_WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS`

Production execution: `COMPLETED_2026-07-23`

## Purpose

The first cutoff-primitives verifier established full output integrity,
canonical-time closure, manifest reconstruction, and parent bindings. It did
not reopen the original main-work files and therefore did not independently
prove that each retained work ID received the correct source metadata.

The additive verifier
`scripts/verify_pivot_feedback_cutoff_primitives_source_replay_v2.py` closes
that boundary without importing the producer or changing the production
output. It independently parses:

- work ID;
- work type;
- `is_paratext`;
- nullable `is_retracted`;
- declared author count; and
- nullable primary topic, domain, and field IDs.

The verifier writes 19-byte fixed-width records to 4,096 private scratch
partitions, sorts each partition by work ID, and compares every value and null
mask with the published work-metadata shard. It separately replays the
canonical publication day, year, null masks, and temporal disposition.

## Adversarial acceptance boundary

The regression fixture includes a metadata permutation in which:

- the same valid work IDs remain present;
- canonical time remains unchanged;
- every metadata field retains the same global marginal distribution; but
- the metadata rows are attached to different work IDs.

The v1 aggregate and canonical checks need not detect that construction. The v2
row-level replay rejects it at the first mismatched work ID and field.

Additional tests reject retraction and topic-hierarchy null-mask changes,
canonical-time changes, malformed topic hierarchy masks, and modified
content-checkpoint payloads.

## Full result

| Quantity | Value |
| --- | ---: |
| Source files | 493 |
| Source bytes | 150,062,449,219 |
| Raw / retained rows | 492,293,207 / 492,293,206 |
| Frozen `W-1` quarantine | 1 |
| Source-output row comparisons | 492,293,206 |
| Compared source fields including work ID | 8 |
| Work partitions | 4,096 |
| Cutoff artifacts rehashed | 4,101 |
| Canonical spine artifacts rehashed | 4,096 |
| Runtime | 850.73 seconds |
| GPU used | no |

The source and output projection SHA-256 values are identical:

`9a8f9b13250ab8b11aecd19b0d67b2a1dba9d4395e1d6e63da79f50f8989abf3`.

The partition-comparison tree SHA-256 is:

`23dab2cd3d39aeb381e977a58ac181a8ba16e88a1af5f52f38fbeb93508b9f92`.

The production cutoff receipt remained
`91e3f689e574d524eabbe7daa58f5e678da37f851ba9b7af0561dab57b70846e`,
and the verifier observed an unchanged output tree.

The operator-local full report is 4,974 bytes with SHA-256:

`889f45167345187b520100a90fd5434feca1cda2b1137e44061c3d13cfedaefc`.

The tracked compact evidence is
`experiments/evidence/results/pivot_feedback_cutoff_primitives_source_replay_full_20260723.summary.json`.

## Content-binding boundary

For each of the 493 files, the verifier recomputed and validated the
content-checkpoint self-digest and exact content SHA declaration, then required
the current source file to match the checkpoint's device, inode, mode, size,
mtime, and ctime before and after the data-page scan.

The full run did not rehash all 150 GB of source bytes again. The stronger
`--rehash-source-content` option exists, but it was not used in this result.
Accordingly, this pass establishes exact row-level replay against the currently
stat-identical, checkpoint-bound files; it does not claim a second independent
full-byte source hash.

## Reproduction shape

```bash
uv run python \
  scripts/verify_pivot_feedback_cutoff_primitives_source_replay_v2.py \
  --source-root /operator/local/openalex-derived-root \
  --content-root /operator/local/content-freeze-output \
  --canonical-root /operator/local/canonical-spine-output \
  --output-root /operator/local/cutoff-primitives-output \
  --scratch-root /operator/local/fresh-ssd-scratch \
  --report /operator/local/source-replay-verification.json \
  --authorize-full
```

Adding `--rehash-source-content` performs the optional second full-byte source
hash. Prefix runs use `--max-source-files N` without `--authorize-full` and are
permanently nonpromotable.

## Scientific boundary

This result closes a data-engineering projection gap. It does not validate
OpenAlex author IDs as people, establish historical topic or citation
observability, resolve candidate eligibility, choose work types or maturation
horizons, materialize the final analytic panel, pass G-1 or G0, run empirical
G1, fit an NABM, or establish a scientific effect.
