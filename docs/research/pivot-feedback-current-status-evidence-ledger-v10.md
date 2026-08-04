# Pivot-feedback current-status and evidence ledger v10

Date: 2026-08-05

Status:
`CURRENT_HUMAN_FACING_INDEX_COMPLETE_PRIVATE_G4_SOURCE_SET_VERIFIED_ROOT_AND_LIST_BINDING_NEXT_NO_F_PASS`

## Immutable predecessors

The machine-readable authority remains
[`pivot_feedback_current_state_v1.yaml`](../../experiments/configs/pivot_feedback_current_state_v1.yaml),
SHA-256
`52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd`.
Ledgers v1-v9 and their production predecessors remain immutable. This v10
ledger indexes the separately authorized private feature/source successor; it
does not create a new gate, run an explicit-list preflight, issue a byte seal,
derive labels, pass F or G0, or open G4.

## Complete private G4 source set

The
[`G4 private feature and source materialization`](../../experiments/evidence/results/pivot_feedback_f_g4_private_feature_source_v1_20260805.summary.md)
adds all source rows frozen for the 37 field-27 G4 episodes to the already
verified private membership and author-cold guard. The materialization contains
217,839,267 predecision author-episode feature rows and 267,491,648
episode-focal-peer contributing relations.

The private public-success source has 30,198,415 field-27 reference works, of
which 3,493,298 satisfy the registered field-year top-decile age-3 citation
proxy with inclusive cutoff ties. That proxy uses current-snapshot
bibliographic event time. It does not establish what OpenAlex historically
displayed or what a researcher actually observed.

The postdecision source contains 1,607,759 distinct focal-author target-topic
work events at offsets t through t+3 and 3,392,785 full work-byline rows. Those
are private source rows, not entry or persistence labels and not analytic
outcomes.

## Independent full replay

The independent verifier does not import the producer. It rebinds the frozen
membership parent, work metadata, author-history events, citation chronology,
and topic taxonomy, then replays every materialized feature and source column.

All output-row checks close: feature and relation keys are unique; the feature
projection equals the membership risk rows; relation algebra and per-feature
peer counts agree; inclusive-tie success cutoffs agree; postdecision keys and
offsets agree; every byline has an event and every event contains its focal
author. All 14 recorded output mismatch counts are zero.

Full raw-source replay also returns zero mismatches for features, contributing
relations, public-success works, postdecision events, and bylines. This is a
byte- and row-backed source-package result, not evidence for an association,
prediction, transport statement, causal mechanism, or scientific effect.

## Privacy and package boundary

The receipt binds 153 artifacts totaling 3,067,046,872 bytes. The private root
and nested directories are mode `0700`; all files are mode `0600`. Raw author
and work tokens remain repository-forbidden. No entry or persistence label,
descriptive rate, outcome state, model score, treatment effect, or predicted
probability appears in the package. Confirmatory or holdout analytic rows read
remains exactly zero.

The complete private G4 artifact set now exists, but that phrase refers only to
the registered source set. Earlier-partition author-cold guard application is
still pending because those analytic rows do not yet exist. No preflight or
seal has occurred.

## Next bounded action

Exactly one next artifact is authorized: bind the exact private source root and
its complete explicit 153-artifact list. This is a narrow chain-of-custody
successor, not another scientific gate. It may not read labels, scores, or
analytic outcomes.

Explicit-list preflight remains unauthorized until that root-and-list binding
passes. Byte sealing follows a successful preflight successor; analytic opening
remains a separate later successor. `pass_f=false`, `pass_g0=false`, and
`g4_open=false`.
