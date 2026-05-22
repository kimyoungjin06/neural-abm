# NABM Unit v1 Manuscript Table Candidates

Date: 2026-05-21

These are manuscript-facing table candidates derived from
`paper/claim-matrix.md`. They are intentionally compact: each table carries the
claim boundary needed to avoid turning bounded evidence into a general
superiority claim.

## Table 1: Unit Boundary

| Surface | Unit-owned | Domain-owned | Paper role |
| --- | --- | --- | --- |
| Generic lifecycle | Local/social update sequencing, typed exchange, adapter dispatch, stable diagnostics | Domain objective terms, environment transitions, evidence gates | Architecture claim |
| Binary policy lifecycle | `BinaryPolicyLearningUnit` readout, probability construction, sampling callback, commit callback, refresh, post-readout ordering | Toy2 payoff advantages; Toy4 welfare/resource advantages; Toy5 adoption/readiness meaning | Reuse claim |
| Binary revision lifecycle | Optional stay/switch sequencing through `BinaryRevisionLearningUnit` | Revision pressure meaning and sampled action consequences | Prototype mechanism boundary |
| Readiness propagation | Peer-readiness aggregation once readiness values are defined | Threshold, direction, confidence, and commitment meaning | Toy5 holdout claim |
| Backend commits | Loop, batched, and tensor-runtime commit dispatch behind adapters | Accelerator cache ownership and domain state mutation semantics | Engineering claim |
| Diagnostics | Stable local/social/revision/readiness aggregate and micro fields | Case-specific ceiling metrics, tolerances, claim groups, and interpretation | Auditability claim |

Caption candidate:

> The NABM Unit v1 contract owns lifecycle order, typed exchange, backend
> dispatch, and stable diagnostics. Domain adapters retain reward, threshold,
> teacher, basin, readiness-meaning, and evidence-gate semantics.

Limitation:

> This table supports a reusable architecture claim, not a claim that neural
> policies generally outperform classical ABM baselines.

## Table 2: Toy5 Safety and Spread Grid

| Case | Output-average baseline | Negative control | Threshold-aware main | Main TtC | Primary reading |
| --- | ---: | ---: | ---: | ---: | --- |
| No-seed heterogeneous safety | 5/5 | 0/5 | 5/5 | 0.0 | Safety preserved; non-directional control self-excites. |
| Lattice `k=4`, threshold `0.85` | 0/5 | 5/5 | 5/5 | 36.2 | Main recovers spread where output averaging stalls. |
| Lattice `k=4`, threshold `0.95` | 0/5 | 5/5 | 5/5 | 37.0 | Main recovers spread under higher threshold. |
| Lattice `k=6`, threshold `0.85` | 0/5 | 5/5 | 5/5 | 25.0 | More local connectivity speeds spread. |
| Lattice `k=6`, threshold `0.95` | 0/5 | 5/5 | 5/5 | 25.0 | Main remains stable at higher threshold. |
| Rewired `k=6`, `p=0.10`, threshold `0.85` | 0/5 | 5/5 | 5/5 | 9.6 | Shortcut topology accelerates spread. |
| Rewired `k=6`, `p=0.10`, threshold `0.95` | 0/5 | 5/5 | 5/5 | 10.0 | Shortcut topology remains robust at higher threshold. |

Caption candidate:

> In the Toy5 grid, the threshold-aware readiness adapter preserves no-seed
> safety and reaches full cascades in all tested sparse-seed spread cases where
> output averaging stalls.

Limitation:

> The exposure-anchor negative control also reaches `5/5` in seeded spread
> cases, so this table supports robustness and safety separation, not
> uniqueness of threshold-aware direction.

Source artifacts:

- `experiments/evidence/toy5_neural_threshold_target_threshold_aware_grid_quick.yaml`
- `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_findings.md`

## Table 3: Toy2/Toy4 Failure-Mode Triage

| Artifact | Toy | Best main variant | Final hits | Mean TtC | Classification | Paper reading |
| --- | --- | --- | ---: | ---: | --- | --- |
| `revision_operator_quick` | Toy2 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 2/3 | 19.33 | Stochastic gate brittleness; baseline-favored | Ceiling is reachable, but final-epoch hazard remains. |
| `revision_operator_quick` | Toy4 | `revision_operator_mixed_objective_basin_w0p5_0p5_h1` | 1/3 | 19.00 | Stochastic gate brittleness; baseline-favored | Failure is not a clean mechanism impossibility claim. |
| `basin_credit_objective_blend_quick` | Toy2 | `mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1` | 3/3 | 12.00 | Slow TtC gate lag; baseline-favored | Final convergence succeeds, speed gate fails. |
| `basin_credit_objective_blend_quick` | Toy4 | `mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1` | 3/3 | 11.00 | Success; baseline-favored | Stable but slower than the clean hand-coded baseline. |
| `revision_operator_precommitment_controls_quick` | Toy2 | `revision_operator_precommitment_peer_evidence_w1p0` | 3/3 | 9.67 | Success; baseline-favored | Precommitment plus peer evidence removes late hazard. |
| `revision_operator_precommitment_controls_quick` | Toy4 | `revision_operator_precommitment_peer_evidence_w1p0` | 3/3 | 9.33 | Success; baseline-favored | Precommitment plus peer evidence removes late hazard. |
| `precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick` | Toy2 | `revision_precommitment_peer_evidence_open_sparse_p0p1` | 5/5 | 9.40 | Success | Targeted stress succeeds without a baseline-favored tag. |
| `precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick` | Toy4 | `revision_precommitment_peer_evidence_open_sparse_p0p1` | 5/5 | 9.00 | Success | Targeted stress succeeds without a baseline-favored tag. |

Caption candidate:

> Gate 3 separates Toy2/Toy4 outcomes into final-epoch hazard, slow
> time-to-ceiling, baseline-favored success, and targeted stress success before
> adding another loss or sampler path.

Limitation:

> This is a diagnostic taxonomy. It should not be cited as evidence that the
> current Toy2/Toy4 neural path is generally faster than reputation imitation.

Source artifacts:

- `experiments/results/nabm_effect_matrix/toy24_gate3_evidence_triage_findings.md`
- `experiments/results/nabm_effect_matrix/evidence_profile_index_gate3.md`

## Table 4: Toy2/Toy4 Reputation-Fragility Stress

| Toy | Variant | Role | Final hits | Ever hits | Mean TtC | Terminal ceiling rate | Metric mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `reputation_imitation_open_sparse_noisy_p0p1_s1p0` | Baseline | 0/5 | 0/5 | n/a | 0.00 | 2.4973 |
| Toy2 | `revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0` | Main | 5/5 | 5/5 | 9.4 | 1.00 | 3.0000 |
| Toy4 | `reputation_imitation_open_sparse_noisy_p0p1_s1p0` | Baseline | 0/5 | 0/5 | n/a | 0.00 | 0.4083 |
| Toy4 | `revision_precommitment_peer_evidence_open_sparse_noisy_p0p1_s1p0` | Main | 5/5 | 5/5 | 9.0 | 1.00 | 0.6000 |

Caption candidate:

> With sparse initial action seeds, open boundaries, and noisy ranking, the
> reputation-imitation baseline reaches no ceiling hits in Toy2 or Toy4, while
> the precommitment plus peer-evidence candidate reaches `5/5`.

Limitation:

> This is targeted baseline-fragility evidence. The stress weakens the
> reputation baseline's own information channel, so it is not a general
> classical-baseline dominance result.

Source artifacts:

- `experiments/evidence/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick.yaml`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_findings.md`

## Table 5: Toy4 Heterogeneous Local-Resource Stress

| Variant | Role | Final hits | Ever hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rep_clean_hetero` | Clean baseline | 5/5 | 5/5 | 15.0 | 0.600000 | 1.000 |
| `rep_noisy_s2p0_hetero` | Noisy baseline diagnostic | 3/5 | 3/5 | 47.333 | 0.494039 | 0.600 |
| `rev_pop_global_obs_noisy_s2p0_hetero` | Negative control | 0/5 | 0/5 | n/a | -0.302000 | 0.000 |
| `rev_local_global_obs_noisy_s2p0_hetero` | Local threshold diagnostic | 5/5 | 5/5 | 33.0 | 0.600000 | 1.000 |
| `rev_local_hidden_obs_noisy_s2p0_hetero` | Local threshold diagnostic | 5/5 | 5/5 | 32.2 | 0.600000 | 1.000 |
| `rev_local_sustain_obs_noisy_s2p0_hetero` | Main | 5/5 | 5/5 | 31.8 | 0.600000 | 1.000 |

Caption candidate:

> In Toy4 with checkerboard extraction heterogeneity and noisy reputation, local
> resource-threshold variants remain stable across five seeds while the noisy
> reputation diagnostic and population-threshold control fail.

Limitation:

> Clean reputation imitation remains faster and stable in clean ranking
> conditions. Local-sustain is only slightly faster than hidden or global
> resource observation, so this table supports robustness, not necessity.

Source artifacts:

- `experiments/evidence/toy4_resource_threshold_heterogeneous_local_observation_stress_quick.yaml`
- `experiments/results/nabm_effect_matrix/toy4_hetero_local_obs_stress_quick_findings.md`

## Table 6: Adapter-Only Congestion Holdout

| Case | Variant | Group | Capacity hits | Max error | Max overcrowding | Mean welfare | Mean TtC |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `symmetric_capacity` | `imitation_baseline` | Baseline | 0/3 | 10 | 10 | 0.500 | n/a |
| `symmetric_capacity` | `global_pressure_negative_control` | Negative control | 0/3 | 10 | 10 | 0.500 | n/a |
| `symmetric_capacity` | `adapter_capacity_policy_main` | Main | 3/3 | 0 | 0 | 1.000 | 1.000 |
| `asymmetric_capacity` | `imitation_baseline` | Baseline | 0/3 | 14 | 14 | 0.300 | n/a |
| `asymmetric_capacity` | `global_pressure_negative_control` | Negative control | 0/3 | 14 | 14 | 0.300 | n/a |
| `asymmetric_capacity` | `adapter_capacity_policy_main` | Main | 3/3 | 0 | 0 | 1.000 | 1.000 |
| `noisy_preference_capacity` | `imitation_baseline` | Baseline | 0/3 | 12 | 12 | 0.400 | n/a |
| `noisy_preference_capacity` | `global_pressure_negative_control` | Negative control | 0/3 | 12 | 12 | 0.400 | n/a |
| `noisy_preference_capacity` | `adapter_capacity_policy_main` | Main | 3/3 | 0 | 0 | 1.000 | 1.000 |

Caption candidate:

> A source-free adapter-only congestion holdout uses the NABM Unit v1 binary
> policy lifecycle for capacity-matched allocation rather than full adoption.
> The adapter capacity policy reaches zero capacity error while imitation and
> global-pressure controls overcrowd.

Limitation:

> This is non-cascade extensibility evidence, but it is still a tiny scripted
> binary domain. It should not be presented as a full general-purpose ABM
> framework demonstration.

Source artifacts:

- `experiments/evidence/adapter_only_congestion_holdout_quick.yaml`
- `experiments/results/nabm_effect_matrix/adapter_only_congestion_holdout_quick_findings.md`

## Table 7: Adapter-Only Stochastic Commons Holdout

| Case | Variant | Group | Min resource | Max collapse epochs | Mean welfare | Mean harvest | Recovery hits | Max recovery |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `steady_regen_commons` | `greedy_harvest_baseline` | Baseline | 0.040 | 17 | -0.136 | 1.000 | 0 | n/a |
| `steady_regen_commons` | `global_pressure_negative_control` | Negative control | 0.316 | 6 | 0.076 | 0.385 | 0 | n/a |
| `steady_regen_commons` | `adapter_local_resource_main` | Main | 0.458 | 0 | 0.067 | 0.300 | 0 | n/a |
| `localized_resource_shock` | `greedy_harvest_baseline` | Baseline | 0.027 | 19 | -0.143 | 1.000 | 0 | n/a |
| `localized_resource_shock` | `global_pressure_negative_control` | Negative control | 0.316 | 5 | 0.071 | 0.353 | 3 | 2 |
| `localized_resource_shock` | `adapter_local_resource_main` | Main | 0.439 | 0 | 0.069 | 0.312 | 3 | 2 |
| `heterogeneous_need_commons` | `greedy_harvest_baseline` | Baseline | 0.028 | 19 | -0.141 | 1.000 | 0 | n/a |
| `heterogeneous_need_commons` | `global_pressure_negative_control` | Negative control | 0.282 | 7 | 0.076 | 0.395 | 3 | 5 |
| `heterogeneous_need_commons` | `adapter_local_resource_main` | Main | 0.485 | 0 | 0.066 | 0.288 | 3 | 1 |

Caption candidate:

> A source-free adapter-only stochastic commons holdout uses the NABM Unit v1
> binary policy lifecycle in a closed-loop setting: harvest decisions deplete
> future resources, conservation regenerates resources, and shocks perturb local
> stocks. The local-resource adapter avoids collapse across all seeds while
> greedy harvest and global-pressure controls dip below the resource floor.

Limitation:

> This is stronger closed-loop extensibility evidence than the fixed threshold
> and capacity holdouts, but it is still a compact scripted binary commons. It
> should not be presented as a full general-purpose ABM framework proof.

Source artifacts:

- `experiments/evidence/adapter_only_stochastic_commons_quick.yaml`
- `experiments/results/nabm_effect_matrix/adapter_only_stochastic_commons_quick_findings.md`
