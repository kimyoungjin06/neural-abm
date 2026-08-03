# OpenAlex work-topic relation one-file-per-table pilot — 2026-07-21

Receipt status: `PILOT_WORK_TOPIC_RELATION_ONLY`

Production component: `NOT_ESTABLISHED_PILOT`

G-1 data readiness: `PENDING`

Current-status note: the subsequent
[full 2026-07-22 audit](../../experiments/evidence/results/openalex_g1_work_topic_relation_full_20260722.summary.md)
completed and supersedes this pilot's then-current `NOT_RUN` state. The full
component is `BLOCKED_WORK_TOPIC_RELATION_COMPONENT` because two duplicate
edge groups carry different scores.

## Scope and decision

The bounded production-source pilot completed from 16:26:42 to 16:26:46 KST
with exit status 0. It selected exactly one lexical-prefix main Parquet file and
one lexical-prefix topics Parquet file, processed 1,000,000 rows from each, and
published the frozen pilot status `PILOT_WORK_TOPIC_RELATION_ONLY`.

This run validates the live source, taxonomy, current-content parent, projected
parser, checkpoint, external sort, aggregation, receipt-last, and terminal-tree
mechanics for that two-file subset. It does **not** establish the production
work-topic component, duplicate-edge prevalence, exact-score consistency for
the full corpus, primary-topic child inclusion, W0 sensitivity, G-1, G0, or a
scientific effect. A max-files pilot is permanently nonpromotable by contract.

The pilot used the root NVMe through a fresh private operator-local output at
`/var/tmp/openalex-g1-work-topic-relation-pilot-20260721`. The project SSD was
intentionally avoided because another repository was sustaining material
writes there. The OpenAlex source HDD and root NVMe were idle in the final two
one-second pre-run samples. This short observation is not a multi-hour quiet
window or authorization for a future full run.

## Frozen inputs

| Input | Bound value |
| --- | --- |
| Config | `experiments/configs/openalex_g1_work_topic_relation_v0.yaml` |
| Config SHA-256 | `7e7d50f212e9738dfce5b437174cfecbfa76679e532a15e23d84ca88c5e0b7fc` |
| Runner | `scripts/audit_openalex_work_topic_relation.py` |
| Runner SHA-256 | `5503ed01c80282fa118ae47f86901e6e5588f52e04c19a1c3646e86274622786` |
| Content receipt SHA-256 | `082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12` |
| Taxonomy canonical SHA-256 | `8b9f4e2f492eb6939e1dbb6a8187f06d4ea8fdebb9c1bafce017cd39e54401b4` |
| Selected source files | 2 |
| Selected source bytes | 330,884,816 |
| Selected footer rows | 2,000,000 |

All four taxonomy topic-file hashes matched the frozen config. The selected
main and topics source identities matched the current-content plan's rich-stat
records before and after access.

## Execution and retained evidence

The exact command shape was:

```bash
uv run python scripts/audit_openalex_work_topic_relation.py \
  --source-root <exact-derived-root> \
  --taxonomy-root <exact-taxonomy-root> \
  --content-freeze-receipt <exact-content-receipt> \
  --output-dir /var/tmp/openalex-g1-work-topic-relation-pilot-20260721 \
  --max-files-per-table 1 \
  --execute
```

The full-only `--allow-full-scope` flag was deliberately absent.

| Evidence | Value |
| --- | --- |
| Receipt schema | `openalex_g1_work_topic_relation_outputs.v1` |
| Receipt status | `PILOT_WORK_TOPIC_RELATION_ONLY` |
| Receipt SHA-256 | `1d5aa95243019492f6f5fb0f976dd5beca6567c6b111645cd054ce56d9539bd6` |
| JSON report SHA-256 | `c6f98d529e082218b6bf2baa901bc38093c7db609a03251ad6c034956724f183` |
| Result fingerprint | `10027757e31fb4c0243d96c8bfe3584eb5c214b720814a7d0b5d79e89fc09e74` |
| Task fingerprint | `309fe1a94532cb63b88b69c3dc6a0c96e963a9ce11e29fc3ff722f3e23491b0b` |
| Checkpoint tree | `f51789d0d6b5a02928f1c216125a11181c9dd7a88ceb73d925e771f69b0f908d` |
| Receipt-bound artifacts | 10 files / 112,027,326 bytes |
| Output footprint | approximately 107 MiB |
| State | `COMPLETE_PENDING_RECEIPT` plus valid final receipt |

The output root and directories are mode `0700`; retained regular files are
mode `0600`; no symlink or operator path occurs in the published report,
Markdown, or receipt. A fresh Python process called the runner's terminal
receipt validator and returned `True`, rehashing the complete retained artifact
set and reconstructing its report, state, checkpoint, sort, and receipt
cross-bindings. This is a separate invocation of the same implementation, not
an independent verifier with separately frozen code.

External `/usr/bin/time` observed 5.30 seconds elapsed and 637,036 KiB maximum
RSS. Those timing bytes were not captured into a command-bound artifact, so
they are an operator observation rather than a promotable benchmark.

## Why the visible subset counts are not findings

The pilot selected one file independently from each table. It was designed as
a parser and lifecycle smoke test, not as a matched or representative sample.
The report therefore correctly leaves `all_local_checks_pass=false`, W0
structural sensitivity `NOT_ESTABLISHED_PILOT`, and the relation component
`NOT_ESTABLISHED_PILOT`. In particular, the reported 457,569 non-null primary
rows without a selected child edge occur under the intentionally incomplete
topics subset and cannot be interpreted as production primary-child failures.

## Remaining full-run boundary

The full run remains `NOT_RUN`. It would read the exact 493-main plus 930-topic
task set: 1,423 files, 1,420,855,753 footer rows, and 166,699,471,247 declared
source bytes. Its preflight requires at least 114,560,014,340 free output bytes
and 55,445,774,404 available memory bytes. A fresh private output path, a new
source/output quiet-window check, and separate explicit authorization for the
full 1,423-file CPU/I/O scan and `--allow-full-scope` remain required.
