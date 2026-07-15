# Manuscript Claim Matrix

Date: 2026-05-21

## Purpose

This matrix is the paper-facing boundary for current NABM Unit v1 evidence. It
links each usable manuscript claim to the code path, manifest, result artifact,
candidate figure or table, and limitation that must travel with the claim.

Claims not listed here should stay out of the primary manuscript narrative until
their evidence is upgraded.

## Primary Claim Path

| Claim | Code path | Evidence artifact | Result summary | Paper asset | Required limitation |
| --- | --- | --- | --- | --- | --- |
| A reusable NABM Unit v1 contract can own binary policy lifecycle plumbing without owning domain rewards, thresholds, teacher signals, or evidence criteria. | `src/neural_abm/unit.py`, `src/neural_abm/spatial_binary.py`, `src/neural_abm/binary_revision.py`, `src/neural_abm/readiness.py`, `src/neural_abm/domain_learning_diagnostics.py` | `docs/decisions/0010-nabm-unit-v1-contract.md`, `docs/nabm-unit-v1-boundary-audit.md`, `docs/nabm-unit-v1-completeness-checklist.md`, `tests/test_nabm_unit_docs.py` | Toy2, Toy4, and Toy5 route key binary policy/readiness/diagnostic lifecycle surfaces through shared unit infrastructure while preserving domain-specific semantics in adapters. | `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 1. | This is an engineering architecture claim, not a claim that NABM policies generally outperform classical ABM rules. |
| Threshold-aware readiness can preserve no-seed safety and recover full cascades on a bounded Toy5 topology/threshold grid where output averaging stalls. | `BinaryReadinessPropagationUnit` plus Toy5 threshold-aware adapter semantics. | `experiments/evidence/toy5_neural_threshold_target_threshold_aware_grid_quick.yaml`, `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_findings.md` | Main threshold-aware path reaches `5/5` final hits in no-seed safety and all six spread cases; output-average baseline reaches `0/5` in all spread cases. | `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 2. | Exposure-anchor negative controls also spread in seeded cases, so the evidence supports robustness and safety separation, not uniqueness of threshold-aware direction. |
| Toy2/Toy4 failures are diagnosable as stochastic gate brittleness, slow TtC, baseline-favored environments, or true mechanism failure candidates without adding another policy loss. | Evidence profile adapter and gate trajectory-status fields. | `experiments/results/nabm_effect_matrix/toy24_gate3_evidence_triage_findings.md`, `experiments/results/nabm_effect_matrix/evidence_profile_index_gate3.md` | Revision quick failures are final-epoch hazard cases; Toy2 objective+basin miss is slow TtC; selected precommitment/peer-evidence paths recover stable final hits. | `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 3. | This is a diagnostic claim. It should not be phrased as algorithmic superiority over reputation imitation. |
| Under sparse seeds, open boundaries, and noisy peer ranking, Toy2/Toy4 precommitment plus peer evidence survives a condition that breaks reputation imitation. | Toy2/Toy4 objective+basin, optional revision, evidence precommitment, and peer-evidence path. | `experiments/evidence/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick.yaml`, `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_findings.md` | Reputation imitation reaches `0/5` final and ever hits in both Toy2 and Toy4; peer-evidence main reaches `5/5` with mean TtC `9.4` and `9.0`. | `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 4. | The stress directly weakens the reputation baseline's information channel. It is targeted baseline-fragility evidence, not a general proof that NABM beats hand-coded rules. |
| In Toy4, local resource-threshold variants remain stable when resource extraction is spatially heterogeneous and reputation ranking is noisy. | Toy4 resource-threshold adapter with objective+basin, precommitment, peer evidence, and local/global/hidden/local-sustain observation variants. | `experiments/evidence/toy4_resource_threshold_heterogeneous_local_observation_stress_quick.yaml`, `experiments/results/nabm_effect_matrix/toy4_hetero_local_obs_stress_quick_findings.md` | Local-threshold variants reach `5/5`; noisy reputation diagnostic reaches `3/5`; population-threshold negative control reaches `0/5`; local-sustain main has mean TtC `31.8`. | `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 5. | Clean reputation imitation remains `5/5` and faster at mean TtC `15.0`; local-sustain is only slightly faster than hidden/global observation. |
| A source-free adapter can use NABM Unit v1 for a non-cascade binary capacity-allocation holdout. | Public binary policy lifecycle callbacks only; the holdout domain lives in `scripts/run_adapter_congestion_holdout_evidence.py`, not `src/neural_abm`. | `experiments/evidence/adapter_only_congestion_holdout_quick.yaml`, `experiments/results/nabm_effect_matrix/adapter_only_congestion_holdout_quick_findings.md` | In symmetric, asymmetric, and noisy-preference capacity cases, the adapter capacity main reaches zero capacity error; imitation and global-pressure controls overcrowd. | `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 6. | This strengthens extensibility beyond a threshold-cascade holdout, but remains a tiny scripted binary domain rather than a general-purpose framework proof. |
| A source-free adapter can use NABM Unit v1 in a closed-loop stochastic commons where actions change future state. | Public binary policy lifecycle callbacks only; the holdout domain lives in `scripts/run_adapter_stochastic_commons_holdout_evidence.py`, not `src/neural_abm`. | `experiments/evidence/adapter_only_stochastic_commons_quick.yaml`, `experiments/results/nabm_effect_matrix/adapter_only_stochastic_commons_quick_findings.md` | In steady-regeneration, localized-shock, and heterogeneous-need cases, the local-resource main avoids collapse while greedy harvest and global-pressure controls dip below the resource floor. | `paper/tables/nabm-unit-v1-manuscript-tables.md` Table 7. | This strengthens closed-loop extensibility, but remains a compact scripted binary commons rather than a full general-purpose ABM framework proof. |
| The torch-free surface carries a researcher-authored replicated counterfactual scenario study end to end, with seed-paired deltas, percentile intervals, and a passing zero-intensity placebo control. | `src/neural_abm/scenario_lite.py`, `src/neural_abm/workflow_lite.py`, `examples/research_pivot_study.py` | `docs/case-studies/researcher-pivot/README.md`, `docs/case-studies/researcher-pivot/data/study_results.json`, `tests/test_scenario_lite.py`, `tests/test_public_api_examples.py` | Seed grants raise the productive pivot rate +0.190 [0.112, 0.263] over 100 paired replicates; hype matches the gain (+0.193) at 2.4x the pivots and ~10x the failures; the zero-intensity placebo collapses to +0.004 [-0.017, 0.017]. | `paper/sections/07-research-application.md`, `paper/figures/pivot_composition.png`. | The domain model is stylized and theory-parameterized, not calibrated to bibliometric data; effect sizes are model-internal quantities, not field estimates. |
| The NABM lifecycle supports a controlled fixed-rule-versus-learning-rule comparison in which every divergence is attributable to the learning rule, and the resulting dynamics (imitative cascades, targeted cautionary hype immunity) are expressible only with endogenous rule change. | `src/neural_abm/unit.py` lifecycle via `examples/research_pivot_learning_study.py` (shared population/network with Study 1). | `docs/case-studies/researcher-pivot/README.md` Study 2, `docs/case-studies/researcher-pivot/data/learning_study_results.json`, `tests/test_public_api_examples.py` | Imitative learning raises failed pivots +0.30 [0.07, 0.44] at baseline (information cascade); cautionary learning cuts hype failures -0.137 [-0.200, -0.087] with baseline exactly unchanged and the learned attention weight falling 1.00 -> 0.81 only under hype. | `paper/sections/07-research-application.md`, `paper/figures/pivot_learning_failed_trajectories.png`, `paper/figures/pivot_learning_attention_weights.png`. | This is an expressiveness claim about endogenous rule change, not a performance claim: the frozen arm is a control, learning modes are postulated rather than estimated, and cautionary learning loses productive pivots under seed grants (-0.073). |

## Secondary or Deferred Claims

| Claim candidate | Current status | What is missing |
| --- | --- | --- |
| Revision operators are a solved structural mechanism. | Deferred. Revision paths are useful as opt-in lifecycle infrastructure, but quick evidence is gate-sensitive. | A stress where the revision operator itself is necessary, not only compatible with precommitment/peer evidence. |
| Basin credit is a finalized learned critic. | Deferred. Current basin-credit and basin-phase critic work is prototype-level. | Holdout evidence showing learned credit transfers beyond Toy2/Toy4 quick gates and improves beyond prototype/fallback heuristics. |
| NABM policies generally outperform Fermi, RD, reputation imitation, or threshold baselines. | Unsupported. Several clean environments remain baseline-favored. | A set of stresses where the classical baseline fails for mechanistic reasons not injected directly into its own signal channel. |
| Toy6-Toy10 are full NABM evidence cases. | Unsupported. Decision 0005 marks them as compatible or limited rather than primary full NABM evidence. | Migration through the same unit contract plus manifest-backed evidence. |

## Manuscript Framing Rules

- Lead with the reusable unit contract and domain-boundary discipline.
- Present Toy5 as the cleanest bounded robustness case for readiness
  propagation and safety/spread separation.
- Present Toy2/Toy4 as diagnostic and targeted robustness evidence, not as a
  speed race against reputation imitation.
- Keep clean baseline-favored environments visible; they are part of the claim,
  not an embarrassment to hide.
- Report final ceiling hits, ever ceiling hits, mean TtC, and the relevant
  failure-mode tag together whenever a Toy2/Toy4 result is discussed.

## Immediate Paper Work

1. Add a compact unit-boundary table from Decision 0010 and the boundary audit.
2. Turn the Toy5 grid into a safety/spread table.
3. Turn Gate 3 into a failure-mode taxonomy table.
4. Add the two targeted Toy2/Toy4 and Toy4 stress tables only after stating the
   baseline-fit caveat.
5. Add the adapter-only congestion table only as extensibility evidence, not as
   a new ABM superiority result.
6. Add the stochastic commons table only as closed-loop adapter evidence, not as
   a full framework proof.
7. Integrate the Section 7 research application: present Study 1 as
   replication/placebo discipline and Study 2 as an expressiveness claim
   about endogenous rule change, with the frozen arm described as a control
   and the cautionary seed-grant cost (-0.073) kept visible.
8. Carry the capacity-ladder justification (Section 7.5) wherever a learned
   policy is introduced, so policy-class size is stated as a per-question
   modeling assumption rather than an implementation default.
