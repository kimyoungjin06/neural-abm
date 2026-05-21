# Toy 1-10 Capability Matrix

This matrix records the current ABM coverage after the Toy1-10 paper-candidate
validation and the Toy6-10 sensitivity additions. It is meant to answer two
questions:

- Which ABM shapes are already represented by the toy suite?
- What parts are common framework surface versus toy-specific domain logic?

Source of truth for machine-readable support is
`src/neural_abm/capabilities.py`.

## Matrix

| Toy | Domain Shape | State Kind | Action/Output Space | Update/Runner | Coordination Support | Backends | Current Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Toy1 | Neural HK classification | supervised probe/model state | class probability distribution | classification learner | `none/none`, `output_average/output_similarity`, `latent_average/latent_similarity`, `parameter_average/state_similarity`, `parameter_aligned_average/state_similarity`, `parameter_aligned_average/aligned_state_similarity` | `loop` | Neural social-learning baseline with accuracy and consensus ground truth. |
| Toy2 | Spatial prisoner's dilemma | binary spatial state | binary action/probability | game-policy runner | `none/none`, `output_average/none`, `output_average/output_similarity` | `loop`, `batched`, `tensor_batched`, `auto` | Binary game dynamics, reputation, mobility, tensor-state backend. |
| Toy3 | Opinion rewiring | continuous opinions plus graph | continuous scalar opinion | HK/Deffuant/neural opinion runner | `none/none`, `output_average/bounded_confidence`, `output_average/output_similarity` | `loop` | Continuous opinions, bounded confidence, endogenous rewiring. |
| Toy4 | Public goods and commons | binary public-goods state plus resource | binary contribution/probability | binary spatial/public-goods runner | `none/none`, `output_average/none`, `output_average/output_similarity` | `loop`, `batched`, `tensor_batched`, `auto` | Group externalities, resource collapse, reputation, tensor-state backend. |
| Toy5 | Contagion and adoption | binary adoption/threshold state | binary adoption/probability | binary contagion runner | `none/none`, `output_average/none`, `output_average/output_similarity` | `loop`, `batched`, `tensor_batched`, `auto` | Threshold cascades, absorbing adoption, reputation, tensor-state backend. |
| Toy6 | Categorical spatial game | categorical policy logits and grid state | K-way categorical distribution | categorical spatial-game runner | `none/none`, `output_average/none`, `output_average/output_similarity` | `loop` | Multi-action categorical coverage and payoff-profile sensitivity. |
| Toy7 | Resource intensity | continuous scalar propensities plus resource | continuous extraction intensity | adaptive resource runner | `none/none`, `output_average/none`, `output_average/output_similarity` | `loop` | Continuous scalar/resource coverage; environment-dominated benchmark. |
| Toy8 | Async event ABM | event states plus scheduled hazard queue | activation/failure/recovery events | asynchronous event runner | `none/none`, `output_average/none`, `output_average/output_similarity` | `loop` | Event scheduler, stale-event invalidation, hazard recomputation. |
| Toy9 | Heterogeneous agent rules | group-specific binary state | binary action/rule output | heterogeneous-rule runner | `none/none`, `output_average/none`, `output_average/output_similarity` | `loop` | Mixed local rules, group composition, coordination gating. |
| Toy10 | Market/ecology network | multi-channel continuous market/ecology state | harvest intensity plus price/conservation channels | market/ecology runner | `none/none`, `output_average/none`, `output_average/output_similarity` | `loop` | Multi-channel social messages, dynamic network churn, ecology feedback. |

## NABM Suitability

The NABM status vocabulary is intentionally small:

- `full`: neural local or social update is the core experiment path for the toy.
- `compatible`: the toy follows the common NABM config, result, social, and
  sweep contracts, but the neural path is limited, optional, or not yet the main
  mechanism.
- `reference`: the model or policy is kept as a comparison baseline rather than
  as the center of the NABM claim.

Toy-level status is:

| Toy | NABM Status | Neural Role | Social Channels | Reference Policies |
| --- | --- | --- | --- | --- |
| Toy1 | `full` | Local classifier training plus social output, latent, and parameter mixing are the primary experiment path. | `output_distribution`, `latent_state`, `parameters` | none |
| Toy2 | `full` | Neural policy local learning and social policy distillation are the primary path. | `action_probability`, `policy_distribution`, `reputation` | `rd_well_mixed`, `fermi_imitation`, `reputation_imitation` |
| Toy3 | `full` | Neural opinion updates and social output mixing drive the opinion/rewiring experiment path. | `opinion_output`, `bounded_confidence`, `peer_graph` | none |
| Toy4 | `full` | Neural contribution policy learning and social distillation are the primary path. | `action_probability`, `policy_distribution`, `reputation`, `resource_state` | `imitation`, `reputation_imitation` |
| Toy5 | `full` | Neural adoption policy learning and social distillation are the primary path. | `adoption_probability`, `policy_distribution`, `reputation`, `exposure_state` | `simple_contagion`, `complex_threshold`, `reputation_imitation` |
| Toy6 | `compatible` | Shared config, result, social, and sweep contracts; neural path is limited relative to full NABM toys. | `categorical_policy`, `output_distribution` | none |
| Toy7 | `compatible` | Shared config, result, social, and sweep contracts; resource dynamics remain the dominant toy-specific mechanism. | `continuous_action`, `resource_state` | none |
| Toy8 | `compatible` | Shared config, result, social, and sweep contracts; event scheduling remains toy-specific. | `event_hazard`, `event_state` | none |
| Toy9 | `compatible` | Shared config, result, social, and sweep contracts; heterogeneous local rules remain toy-specific. | `binary_action_probability`, `group_state` | none |
| Toy10 | `compatible` | Shared config, result, social, and sweep contracts; market/ecology feedback remains toy-specific. | `harvest_intensity`, `price_signal`, `conservation_signal` | none |

Toy2, Toy4, and Toy5 reference policies are important for comparison and
validation, but they are not the center of the NABM mechanism claim. Toy6-10
remain `compatible` until their neural local or social path becomes the main
validated experiment path.

## Completeness Snapshot

This table separates implementation reuse from claim strength. Reusing a
runner, config shape, or artifact contract does not upgrade the NABM status.

| Toy | Current Claim | Runner Reuse | Artifact Contract | Evidence Readiness |
| --- | --- | --- | --- | --- |
| Toy1 | `full` NABM supervised/social-learning path. | Classification runner local to Toy1. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Included in the default evidence matrix as no-social versus output-average accuracy. |
| Toy2 | `full` NABM binary spatial game path with reference policies. | `BinarySpatialRunner`. | Common binary result, metadata, summary, aggregate, and micro artifacts. | Included in the default evidence matrix as neural policy versus RD, Fermi, and reputation references. |
| Toy3 | `full` NABM opinion/social-output path. | Toy3-specific opinion runner. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Included in the default evidence matrix as neural output averaging versus HK/Deffuant polarization. |
| Toy4 | `full` NABM public-goods binary spatial path. | `BinarySpatialRunner`. | Common binary result, metadata, summary, aggregate, and micro artifacts. | Included in the default evidence matrix as neural policy versus imitation and reputation-imitation payoff. |
| Toy5 | `full` NABM contagion/adoption binary spatial path. | `BinarySpatialRunner`. | Common binary result, metadata, summary, aggregate, and micro artifacts. | Included in the default evidence matrix as neural policy versus threshold, contagion, and reputation references. |
| Toy6 | `compatible`; multi-action categorical coverage. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Compatible but not evidence-default. |
| Toy7 | `compatible`; continuous scalar/resource coverage. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Compatible but not evidence-default. |
| Toy8 | `compatible`; asynchronous event coverage. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Compatible but not evidence-default. |
| Toy9 | `compatible`; heterogeneous local-rule coverage. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Compatible but not evidence-default. |
| Toy10 | `compatible`; market/ecology network coverage. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Compatible but not evidence-default. |

Run artifacts record the same classification fields in `metadata.json`,
`summary.json`, and sweep summary outputs:

```text
nabm_status
neural_role
social_channels
reference_policies
```

## Common Surface

All Toy1-10 configs use the same public top-level shape:

```text
run
simulation
model
domain
logging
```

All runners return the same public result envelope:

```text
run_dir
toy
final_fragmentation_components where meaningful
domain_metrics
```

CSV outputs preserve toy-specific fields under `domain_*` names. This keeps the
public contract stable while allowing each toy to own its domain equations.

Toy1-10 sweep scripts share common output specs, result-row field mapping,
summary writers, grouped-summary construction, optional grouped Markdown hooks,
spec-bound compatibility helpers for script-local summary functions, and
selected row/config/result extraction helpers in `src/neural_abm/sweep.py`,
including final aggregate-metric CSV readers.
Toy1 now uses shared explicit-case orchestration for base YAML loading,
generated-config routing, run execution, row assembly, and summary/grouped
output writing while keeping its case matrix and Markdown readout local. Toy3-10
share common coordination/case iteration, prepared-case config writing with
nested updates and optional Toy-specific mutation hooks, run execution, row
assembly, and main orchestration helpers.
Toy2 and Toy3-10 also share the common CLI argument block, with toy-specific
legacy defaults preserved through helper options. Toy2 still owns its
payoff/regime matrix semantics, conditional grid pruning, and RD reference
insertion, but its point-row orchestration, non-overwriting output path
resolution, and summary/grouped/Markdown output writing run through the shared
sweep helpers. The individual scripts still own their domain parameter
definitions and grouped-summary aggregation specs, while domain run-name and
config-update builders are isolated as small toy-specific functions.

## Generalization Coverage

The current suite covers these ABM families:

- Neural supervised social learning: Toy1.
- Binary spatial games and cascades: Toy2, Toy4, Toy5.
- Continuous opinion dynamics and graph rewiring: Toy3.
- Multi-action categorical games: Toy6.
- Continuous scalar resource extraction: Toy7.
- Asynchronous event scheduling: Toy8.
- Heterogeneous local rules and group gates: Toy9.
- Multi-channel market/ecology feedback: Toy10.

The suite is now broad enough to test most new ABM additions against an
existing family. A new model should first be mapped to one of these dimensions:

- state representation: binary, categorical, continuous scalar, continuous
  vector, event queue, multi-channel, or heterogeneous group state;
- output representation: binary probability, categorical distribution,
  continuous scalar/vector, event hazard, or multi-channel message;
- topology behavior: static graph, dynamic rewiring, event-time graph snapshot,
  or mobility;
- coordination path: no social, static output averaging, dynamic
  output-similarity peers, bounded confidence, latent averaging, or parameter
  averaging;
- runner contract: binary result or domain result.
- sweep contract: shared coordination expansion, explicit-case orchestration,
  point-row orchestration, point execution adapters, parameter-grid iteration,
  prepared-case config writing, nested config updates, run execution, main
  orchestration, output path resolution, output specs, and stable CSV/Markdown
  field writing, with domain-specific parameter definitions and mutation hooks.

## Current Limits

The common structure is not yet a fully generic ABM engine. These areas remain
toy-specific:

- domain equations and payoff/resource/event laws;
- non-binary tensor-state acceleration outside Toy2/Toy4/Toy5;
- event-time simulation beyond Toy8;
- dynamic graph rewiring policies outside Toy3 and Toy10;
- general multi-channel message schemas beyond Toy10;
- generalized heterogeneous group composition beyond Toy9.

This is intentional for now. The validation baseline is stable, and further
abstraction should be extracted only where at least two toy families need the
same mechanism.

## Validation Anchors

Recent validation outputs:

- `experiments/results/toy_validation_representative_toy1_10_social_calibrated_seeds01_03_report.md`
- `experiments/results/toy_validation_paper_candidate_toy1_10_social_calibrated_seeds01_05_report.md`

Recent sensitivity outputs:

- `experiments/results/toy6_toy7_sensitivity_findings.md`
- `experiments/results/toy8_async_sensitivity_seeds01_05_grouped_summary.csv`
- `experiments/results/toy9_toy10_sensitivity_findings.md`
- `experiments/results/toy10_social_calibration_findings.md`
