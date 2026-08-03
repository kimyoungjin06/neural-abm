# Manuscript Claim Matrix

Date: 2026-07-16

Operational status refresh: 2026-08-03

## Purpose

This matrix is the paper-facing boundary for current NABM Unit v1 evidence. It
links each usable manuscript claim to the code path, manifest, result artifact,
candidate figure or table, and limitation that must travel with the claim.

Claims not listed here should stay out of the primary manuscript narrative until
their evidence is upgraded.

## Route Boundary

The document date above records the original manuscript-matrix checkpoint.
Current pivot-feedback status is consolidated in the
[`current-status and evidence ledger v2`](../docs/research/pivot-feedback-current-status-evidence-ledger-v2.md),
with machine-readable authority in
[`pivot_feedback_current_state_v1.yaml`](../experiments/configs/pivot_feedback_current_state_v1.yaml).
The protocol v0.4 byte is retained because multiple immutable contracts bind
its hash; the ledger is the additive operational update rather than an in-place
protocol mutation.

This matrix governs the current method-first manuscript. The prospective
science-first pivot-feedback study is governed by
[`docs/case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md`](../docs/case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md).
Its proposed empirical and adaptive-model claims are not current evidence and
must not be promoted into the primary claim path before their gates pass.
The current operational boundary is:

- bounded G-1A source and primitive fidelity is satisfied, while immutable
  legacy G-1 remains unpassed;
- the discovery-only D0 entry and persistence measurement chain is complete
  for four certified episodes in two fields;
- all 3,256,271 registered author-episode rows have an independently replayed
  observable-history classification, including 4,301 `NEW_TOPIC_ENTRY` rows;
- the 4,301-row three-year persistence source and aggregate descriptive
  readout are independently replayed, but are not adjusted comparisons;
- the active F phase has a complete thirteen-choice pre-G1 registry and a
  separately replayed thirteen-section outcome-blind freeze component, but no
  atomic F promotion, final-panel-conformance receipt, or holdout receipt; six
  G2/G3 choices and two G2/G3 sections are deferred to their registered later
  entry boundaries; and
- empirical G1 is `NOT_RUN`, no baseline or NABM model has run, and G4 is
  logically reserved but neither byte-sealed nor opened.

These advances establish source, measurement, and precommitment infrastructure
only. Author IDs remain current-snapshot tokens rather than verified persons;
historical topic and citation visibility remains unidentifiable from the
snapshot; final confirmatory field selection remains pending; and no
association, timing gap, adaptive-model advantage, causal mechanism, transport,
or scientific effect is established.

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
| The torch-free surface carries a researcher-authored replicated counterfactual scenario study end to end, with paired mean-effect CIs separated from empirical replicate intervals and a true-null invariant. | `src/neural_abm/scenario_lite.py`, `src/neural_abm/workflow_lite.py`, `examples/research_pivot_study.py` | `docs/case-studies/researcher-pivot/README.md`, `docs/case-studies/researcher-pivot/data/study_results.json`, `tests/test_scenario_lite.py`, `tests/test_research_pivot_evidence.py` | Seed grants raise the productive pivot rate +0.191 [95% mean CI 0.181, 0.202]; hype raises it +0.187 [0.180, 0.194] at 2.3x the pivots and 8.6x the failures. The direct hype-minus-grant contrast is -0.004 [-0.012, 0.004], without an equivalence claim. Zeroing every grant component yields exactly 0.000 [0.000, 0.000]. | `paper/sections/07-research-application.md`, `paper/figures/pivot_composition.png`. | The domain model is hand-set and uncalibrated; the true-null result validates the pipeline invariant, not the scenario's external validity. Hype bundles multiple mechanisms, so susceptibility is not separately identified. |
| The NABM lifecycle supports a controlled frozen-versus-adaptive comparison with component-keyed paired shocks and complete named-parameter traces. | `src/neural_abm/unit.py` lifecycle via `examples/research_pivot_learning_study.py` (shared population and base-network generators with Study 1). | `docs/case-studies/researcher-pivot/README.md` Study 2, `docs/case-studies/researcher-pivot/data/learning_study_results.json`, `tests/test_research_pivot_evidence.py` | Imitative learning raises failed pivots +0.281 [0.235, 0.322] at baseline and +0.289 [0.268, 0.308] under grants, but not under hype (-0.009 [-0.024, 0.007]). Failure-only learning reduces hype failures -0.144 [-0.159, -0.129] and productive pivots -0.050 [-0.056, -0.044]; the attention coefficient falls 1.00 -> 0.81 while other weights and bias also move. | `paper/sections/07-research-application.md`, `paper/figures/pivot_learning_failed_trajectories.png`, `paper/figures/pivot_learning_attention_weights.png`. | This is a lifecycle expressiveness and audit claim, not performance, causal-mechanism, canonical-cascade, or ABM-framework exclusivity evidence. The frozen policy is a sigmoid reparameterization, learning rules are postulated, and failure-only learning also loses productive pivots under grants (-0.069). |

## Prospective Pivot-Feedback Claim Gates

These rows are planned tests, not supported claims.

Here, `exposure` means a time-eligible bibliographic and network-availability
proxy. It does not establish receipt, attention, belief revision, or social
influence. Chaperoned and independent entry must be separated before the study
distinguishes recruitment or capacity transfer from a pattern consistent with
network-local outcome learning.

The bounded G-1A receipt, completed D0 measurement chain, and pending G0 v4
successor are fail-closed source, measurement, and design evidence. Their
validation does not change any row below from planned or untested to supported.

| Candidate claim | Current status | Promotion evidence | Mandatory stop condition |
| --- | --- | --- | --- |
| The field-balanced H3 timing-gap primary is positive: topic entry peaks at least one cohort-year after adjusted three-year persistence peaks. | Planned and untested. The compact cross-field result is profile-feasibility only; final field selection is `PENDING` and empirical G1 is `NOT_RUN`. | Authoritative taxonomy/reference binding, outcome-blind maximin field selection, at least three independent episodes in each of three G1 fields, past-only cohort clocks, equal field and within-field episode weights, the registered positive interval, leave-one-field-out direction, and one frozen alternative pivot construction. | No reproducible timing gap; a lead-attention or matched non-surge equivalence gate fails; an atomic author/topic/shock split is violated; or one field or shock alone supplies the result. |
| The pre-surge episode-level ratio `R = bibliographic outcome-availability lag / prior-collaborator exposure-opportunity timescale` predicts the magnitude or ordering of the later H3 gap across fields. | Planned, untested secondary transport endpoint. It is not a co-primary, does not measure actual information receipt, and no model run exists. | Pre-surge time-eligible estimators frozen at G0, episode-level uncertainty, no confirmatory-outcome field selection, and successful prediction in the sealed domain-and-author-cold G4 field after the H3 primary passes. | H3 fails; `R` only fits inspected episodes; its ordering fails in domain-and-author-cold G4; or it is interpreted as a causal field coefficient or measured cognitive diffusion rate. |
| Time-eligible target-topic outcomes among prior collaborators predict entry beyond field-wide attention, matched non-neighbor exposure, and pre-entry selection covariates. | Untested and observational. | Incumbent-only leave-out attention, peer-active and visible-success increments, past-only ties and outcomes, a pre-specified split between chaperoned and independent entry, stable pretrend-adjusted direction, and future-tie, future-success, unrelated-topic-success, matched-non-neighbor, and shuffled-network controls. | The network-local association disappears under the registered controls. If it is confined to chaperoned entry, report recruitment or capacity transfer rather than social learning. |
| A pre-entry target-topic capacity-bridge proxy is associated with later-cohort persistence after conditioning on the prior-collaborator outcome-availability proxy. | Untested and observational. | Four-state signal/capacity-proxy coding, pre-entry network definition, multiple outcomes, matched risk sets, and final holdout transport; a post-entry tie is not the primary comparator. | Sparse cells, collinearity, or selection adjustment prevents separation from the signal proxy. |
| Adaptive NABM adds scientific value beyond simpler models. | Untested. | Frozen comparison against fixed/logistic hazard, fixed-rule ABM, and explicit adaptive baseline using log score, Brier score, calibration, and macro timing. | A simpler model matches holdout prediction or intervention ordering. |
| Within the registered simulation, disclosure delay, modeled information-propagation speed, or structural support changes a robust phase boundary. | Untested and model-based. | Shared latent outcomes, orthogonalized channels, explicit delay-by-diffusion-speed manipulation, paired shocks, uncertainty analysis, and stable held-out intervention ordering. | Policy ranking changes across plausible calibration, measurement, seed, or architecture choices. |

H3 is the single confirmatory primary at family-wise alpha 0.05. Only after H3
and all registered equivalence controls pass may H1, H2, and H4 enter their
three-hypothesis Holm family. The episode-level `R` result is secondary and
cannot rescue H3. The G4 transport test is domain-and-author-cold; an
author-warm or in-domain result cannot replace it. Any placebo, multiplicity,
atomic-split, or cold-holdout failure triggers the stop boundary in the table
rather than a post hoc field, endpoint, or model substitution.

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
   replication/true-null discipline and Study 2 as a bounded lifecycle claim
   about endogenous rule change, with the frozen arm described as a control,
   multivariate parameter movement visible, and the failure-only grant cost
   (-0.069 productive pivots) retained.
8. Carry the capacity-ladder justification (Section 7.5) wherever a learned
   policy is introduced, so policy-class size is stated as a per-question
   modeling assumption rather than an implementation default.
