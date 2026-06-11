# Toy Feature Taxonomy and Capability Matrix

This document records the current ABM coverage by capability family. Stable IDs
remain available for configs, artifacts, tests, and paper traceability, but
user-facing organization starts from feature taxonomy rather than numeric order.

It is meant to answer three questions:

- Which ABM shapes are already represented by the toy suite?
- Which feature family should a new model be compared against first?
- What parts are common framework surface versus toy-specific domain logic?

Source of truth for machine-readable support is
`src/neural_abm/capabilities.py`.

## Stable IDs vs Feature Names

Stable IDs are compatibility identifiers, not the main conceptual taxonomy.
Product copy, selection UIs, and planning documents should prefer the feature
names below, while preserving `toy1`-style IDs in artifact paths, config names,
test names, and reproducibility references.

| Stable ID | Feature Name | Domain Family | State Family | Output Family | Topology Family | Unit Surface | Evidence Role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `toy1` | Neural HK Classification | supervised social learning | supervised probe | categorical distribution | static population | torch-backed distribution/latent/parameter | default evidence |
| `toy2` | Spatial Prisoner's Dilemma | binary spatial game | binary spatial | binary probability | spatial grid with mobility | binary policy tensor-backed | default evidence |
| `toy3` | Opinion Rewiring | continuous opinion dynamics | continuous graph | continuous scalar | dynamic rewiring | continuous output, toy-specific | default evidence |
| `toy4` | Public Goods Commons | binary public-goods commons | binary resource | binary probability | spatial group/resource | binary policy tensor-backed | default evidence |
| `toy5` | Contagion Adoption | binary contagion cascade | binary threshold | binary probability | spatial exposure | binary policy tensor-backed | default evidence |
| `toy6` | Categorical Spatial Game | categorical spatial game | categorical grid | categorical distribution | spatial grid | probability distribution | parity coverage |
| `toy7` | Resource Intensity | continuous resource extraction | continuous resource | bounded scalar | spatial resource | bounded scalar | parity coverage |
| `toy8` | Async Event ABM | asynchronous event dynamics | event queue | event hazard | event-time snapshot | scalar probability | parity coverage |
| `toy9` | Heterogeneous Agent Rules | heterogeneous rule dynamics | heterogeneous group state | binary probability | static group network | scalar probability | parity coverage |
| `toy10` | Market Ecology Network | market/ecology feedback | multi-channel continuous | multi-channel bounded scalar | dynamic network churn | bounded scalar per channel | parity coverage |

## Feature Groups

Output representation:

- Binary probability: `toy2`, `toy4`, `toy5`, `toy9`.
- Categorical distribution: `toy1`, `toy6`.
- Continuous scalar: `toy3`.
- Bounded scalar: `toy7`.
- Event hazard: `toy8`.
- Multi-channel bounded scalar: `toy10`.

Topology behavior:

- Static population or grid: `toy1`, `toy6`, `toy7`, `toy9`.
- Spatial grid with binary interaction or exposure: `toy2`, `toy4`, `toy5`.
- Dynamic rewiring or churn: `toy3`, `toy10`.
- Event-time snapshot scheduling: `toy8`.

Claim/evidence role:

- Default evidence families: `toy1`, `toy2`, `toy3`, `toy4`, `toy5`.
- Parity coverage families: `toy6`, `toy7`, `toy8`, `toy9`, `toy10`.

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
| Toy6 | `compatible` | Unit-backed `strategy_distribution` social mixing uses the probability-distribution channel; cyclic payoff, local logits, and action sampling remain toy-specific. | `categorical_policy`, `output_distribution` | none |
| Toy7 | `compatible` | Unit-backed extraction-intensity social mixing uses the bounded-scalar channel; resource dynamics and continuous intensity semantics remain toy-specific. | `continuous_action`, `resource_state` | none |
| Toy8 | `compatible` | Unit-backed activation-propensity social mixing uses the scalar social path; event scheduling remains toy-specific. | `event_hazard`, `event_state` | none |
| Toy9 | `compatible` | Unit-backed heterogeneous action-probability social mixing uses the scalar social path; heterogeneous local rules remain toy-specific. | `binary_action_probability`, `group_state` | none |
| Toy10 | `compatible` | Unit-backed price/conservation social mixing applies the bounded-scalar channel per field; market/ecology feedback remains toy-specific. | `harvest_intensity`, `price_signal`, `conservation_signal` | none |

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
| Toy6 | `compatible`; multi-action categorical coverage with unit-backed distribution social-mixing parity. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Parity slice complete; compatible but not evidence-default. |
| Toy7 | `compatible`; continuous scalar/resource coverage with unit-backed bounded-scalar social-mixing parity. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Parity slice complete; compatible but not evidence-default. |
| Toy8 | `compatible`; asynchronous event coverage with unit-backed activation-propensity social-mixing parity. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Parity slice complete; compatible but not evidence-default. |
| Toy9 | `compatible`; heterogeneous local-rule coverage with unit-backed action-probability social-mixing parity. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Parity slice complete; compatible but not evidence-default. |
| Toy10 | `compatible`; market/ecology network coverage with unit-backed per-channel bounded-scalar social-mixing parity. | `DomainToyRunner` through `DomainRunSettings`. | Common domain result, metadata, summary, aggregate, and micro artifacts. | Parity slice complete; compatible but not evidence-default. |

Run artifacts record the same classification fields in `metadata.json` and
`summary.json`:

```text
toy_display_name
domain_family
state_family
output_family
topology_family
coordination_family
unit_surface
evidence_role
```

Run artifacts and sweep summary outputs also keep the existing NABM fields:

```text
nabm_status
neural_role
social_channels
reference_policies
```

## Common Surface

All checked-in model-family configs use the same public top-level shape:

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

Model-family sweep scripts share common output specs, result-row field mapping,
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
Toy2 and Toy3-10 also share the common CLI argument block, with domain-specific
default choices preserved through helper options. Toy2 still owns its
payoff/regime matrix semantics, conditional grid pruning, and RD reference
insertion, but its point-row orchestration, non-overwriting output path
resolution, and summary/grouped/Markdown output writing run through the shared
sweep helpers. The individual scripts still own their domain parameter
definitions and grouped-summary aggregation specs, while domain run-name and
config-update builders are isolated as small toy-specific functions.

## Unit-Backed Migration Parity

The migration-parity pass moved one social-exchange slice from each compatible
Toy6-10 runner through the NABM Unit typed-channel surface. This is an
engineering reuse claim, not a performance or status-upgrade claim.

| Toy | Typed unit surface | Migrated social slice | Domain semantics that stay toy-owned |
| --- | --- | --- | --- |
| Toy6 | `PROBABILITY_DISTRIBUTION_CHANNEL` | `strategy_distribution` output averaging. | Strategy identity, cyclic payoffs, local logit update, action sampling, payoff EMA, strategy entropy, and evidence criteria. |
| Toy7 | `BOUNDED_SCALAR_CHANNEL` | `extraction_intensity` output averaging. | Resource dynamics, payoff calculation, noisy intensity sampling, propensity updates, and continuous-action interpretation. |
| Toy8 | scalar social path | `activation_propensity` hazard mixing. | Event queue, stale-event invalidation, event scheduling, event application, hazard semantics, counters, and event-time RNG. |
| Toy9 | scalar social path | `heterogeneous_action_probability` output averaging. | Group assignment, group-specific local rules, coordination gates, action sampling, payoff computation, propensity learning, and payoff EMA. |
| Toy10 | `BOUNDED_SCALAR_CHANNEL`, applied per channel | `price_expectation` and `conservation_norm` output averaging. | Composite market/ecology similarity, harvest construction, market price, resource transition, payoff updates, dynamic rewiring, channel aggregation, and evidence criteria. |

These migrations keep Toy6-10 in the `compatible` category. They show that the
existing toy suite exercises probability-distribution, scalar, and bounded
scalar social exchange without requiring the unit to own toy-specific domain
equations.

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
- general multi-channel message schemas beyond Toy10's per-channel bounded
  scalar parity slice;
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
