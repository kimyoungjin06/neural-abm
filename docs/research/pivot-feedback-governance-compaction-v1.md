# Pivot-feedback governance compaction v1

Date: 2026-08-03

Status: `SHADOW_ONLY_INDEPENDENTLY_VALIDATED_NO_AUTHORITY_CHANGE`

## Outcome

The current scientific boundaries can be preserved with six live phases and
one optional causal lane instead of routing every small artifact through a new
gate-state version. The machine-readable shadow contract is
[`pivot_feedback_governance_compaction_v1.yaml`](../../experiments/configs/pivot_feedback_governance_compaction_v1.yaml),
SHA-256
`4c2562be842ec30b3fafee5320e88f5949b28fcd89e6091b34e5e43dcdd4c4ee`.

This contract is deliberately non-authorizing. The effective machine authority
remains gate v61, and G0 v4 remains unchanged. The validated G1 replication
tolerance receipt is recorded as post-v61 evidence but remains unbound into
G0. No successor, active-routing change, data access, section completion,
model fit, holdout operation, or claim promotion is authorized here.

## Why compaction is warranted

The live repository contains ten legacy logical gate nodes but sixty
`pivot_feedback_gate_state_v2` through `v61` files, fifty-nine gate-state test
files, thirty-seven gate-state validators, and fifty-nine explicit parent
references. The scientific protections are substantive; the per-microstep
authorization implementation is the source of the excess.

The compaction keeps every scientific transition receipt and irreversible
boundary while reducing the active routing vocabulary:

| Shadow phase | Legacy nodes | Current shadow state | Claim ceiling |
| --- | --- | --- | --- |
| `D` data and threshold-free structure | G-1A, S0 | Complete in the bounded source/structure role | No natural-person identity or effect claim |
| `C` discovery and calibration | D0 | Complete for discovery measurement only | No association, causality, or transport |
| `F` freeze, panel conformance, and seal | G0, G0.5 | Pending and unsealed | No empirical signature or effect |
| `E` empirical signature | G1 | Not run | At most a bounded observational signature |
| `M` model added value and conditional simulation | G2, G3 | Not run | At most predictive or generative adequacy |
| `H` sealed cold holdout | G4 | Logically reserved, not byte-sealed or opened | At most bounded noncausal transport |
| optional `CAUSAL` lane | G5 | Not run | Design-specific causality only if separately identified |

All ten legacy nodes are mapped exactly once. No legacy file is deleted,
rewritten, relabeled as passed, or removed from active routing by this step.

## Just-in-time decision staging

The legacy G0 surface contains nineteen quantitative choices. The shadow
contract preserves all nineteen but moves six later-stage choices out of the
pre-G1 blocker:

| Entry boundary | Choices required | Count |
| --- | --- | ---: |
| Before confirmatory G1 access | Existing twelve G0 v4 choices plus `g1_replication_tolerance` | 13 |
| Before G2 model-score access | score margin, maximum calibration degradation, macro-trajectory tolerance | 3 |
| Before G3 simulation execution | draw/MCSE rule, parameter budget, tuning budget | 3 |

The G2 values may not depend on G1 effect magnitude and must be frozen before
G2 model scores are read. The G3 values must be frozen before simulation
results are inspected. This is staged preregistration, not post-result tuning.

The same rule applies to the fifteen required sections. Thirteen remain before
G1; `g2_predictive_gate` moves to the G2 entry boundary and
`g3_simulation_gate` moves to the G3 entry boundary. Every section remains
mapped exactly once.

This staging is a governance proposal only. Under the still-effective v61/G0
v4 authority, all seven choices and all fifteen sections remain pending until a
separately authorized successor changes that state.

## Preserved hard boundaries

Four irreversible operations remain explicitly fail-closed:

1. first confirmatory read requires the freeze phase, panel conformance, and
   holdout byte seal;
2. first model-selection score read requires a passed empirical phase and a
   locked pre-G2 registry;
3. first holdout open requires a resolved model phase and byte-locked model and
   scoring contract; and
4. claim promotion requires claim-specific receipts and a claim-ceiling check.

The equivalence contract also preserves sixteen current denials, including G0
successor binding, G1 or production application, G2/G3 choice resolution,
required-section completion, confirmatory or holdout access, G4 selection and
materialization, holdout preflight and seal, G0 passage, model fitting, G4
opening, outcome-based selection, causality, and scientific-effect promotion.

## Hash-scope correction

Scientific lineage roots remain hash-bound for data identity, measurement and
estimand contracts, partitions, precommitted decisions, holdout manifests and
receipts, and claim-bearing results. Human navigation documents, lint
configuration, general test-runner configuration, and nonsemantic formatting
are not scientific lineage roots by default.

Executable code remains hash-bound when it defines a claim-bearing
transformation, opens or scores a sealed partition, or generates a
claim-bearing result. This narrows accidental hash cascades without weakening
scientific reproducibility.

## Independent validation

The independent verifier is
[`verify_pivot_feedback_governance_compaction_v1.py`](../../scripts/verify_pivot_feedback_governance_compaction_v1.py),
SHA-256
`5088a21941afaddce8a3dc02aec3709f6739161755db41bb419a6fa529600263`.
It reads repository metadata only and checks:

- exact hashes and statuses for gate v61, the legacy gate sequence, G0 v4, the
  human ledger, and the G1 tolerance contract and validation receipt;
- live counts for all sixty gate-state files, fifty-nine tests, thirty-seven
  validators, and fifty-nine parent references;
- exact one-to-one legacy-node mapping and acyclic live-phase dependencies;
- all current denials and claim ceilings;
- complete, duplicate-free coverage of nineteen choices and fifteen sections;
- all four irreversible fail-closed boundaries; and
- a five-step migration in which only shadow schema and repository-only
  equivalence validation are currently permitted.

The result is
[`pivot_feedback_governance_compaction_v1_20260803.summary.json`](../../experiments/evidence/results/pivot_feedback_governance_compaction_v1_20260803.summary.json),
SHA-256
`e03c07949195c556ba566e759a2a88f4d8a4d8373af4b515505a975f141c2296`,
with status
`PASS_SHADOW_SIX_PHASE_SEMANTIC_EQUIVALENCE_NO_AUTHORITY_CHANGE`.

## Migration boundary

Only the first two administrative steps are represented:

1. create the shadow schema; and
2. independently validate semantic equivalence.

Parallel-read routing, an authority successor, and removal of v2-v61 from
active routing remain `NOT_AUTHORIZED`. If later approved, the authority
successor must be a milestone transition that binds v61 and this validation
receipt. It must not rewrite the legacy chain or infer that compaction passes
G0 or any scientific gate.
