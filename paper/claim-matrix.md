# Manuscript Claim Matrix

Date: 2026-07-16

Operational status refresh: 2026-08-03

Paper-1 direction refresh: 2026-10-13

## Purpose

This matrix is the paper-facing boundary for current NABM Unit v1 evidence. It
links each usable manuscript claim to the code path, manifest, result artifact,
candidate figure or table, and limitation that must travel with the claim.

Claims not listed here should stay out of the primary manuscript narrative until
their evidence is upgraded.

## Paper-1 Direction (2026-10-13)

The user set this direction after the empirical model-ladder stage closed. It
is recorded in
[consolidation v5](../docs/research/pivot-policy-ladder-consolidation-v5.md).

Paper-1 is a method and framework paper. Its central claim is:

> NABM is a reusable framework that separates policy learning from social
> interaction, so fixed policies, adaptive policies, and external priors can be
> compared and traced within one simulation lifecycle.

Performance comparisons are conditional cases that illustrate the claim. None
of them is the central claim.

### What the central claim needs

| Component | Package evidence | Repository-only evidence | Status | Gap |
| --- | --- | --- | --- | --- |
| Policy learning is separated from social interaction | The unit-contract and two adapter-holdout rows of the Primary Claim Path | None needed | Supported for binary lifecycles | None |
| Fixed and adaptive policies are compared in one lifecycle | The Study 2 row: a frozen-versus-adaptive comparison with paired shocks and named-parameter traces | The Pivot policy ladder reproduces its fixed rule exactly before any learned arm is read (P1). The information-spread ladder adds tables written before the run (P7) | Supported | None. The outcome record is in the package with its ordering rule (Decision 0017, O27 answered by N15) |
| External priors enter the same lifecycle | None | A once-elicited LLM prior in two domains (P3, P9), with an exploration floor | Supported. `neural_abm.policy_ladder` holds the prior rule and exploration floor on the `main` / next-alpha candidate (Decision 0017) | Not in the released `v0.1.0a5` tag |
| Policies are traced and scored out of sample | `neural_abm.policy_scoring` (Decision 0016) | Pivot probe-cohort scoring | Supported | None |
| The structure is reusable across domains | The unit contract spans Toy2, Toy4, Toy5, and two adapters; `neural_abm.policy_ladder` and `neural_abm.policy_learners` (Decision 0017) | A second domain runs the comparison through a shared core, and both recorded Pivot runs reproduce byte for byte through the package (C47, C48, N15 parity) | Supported on the `main` / next-alpha candidate | Not in the released `v0.1.0a5` tag |

Two consumers share the policy-ladder semantics byte for byte, and Decision
0017 put them in the `main` / next-alpha package. Cite them as package API
there, and note that the released `v0.1.0a5` tag predates them.

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

## Policy-Ladder Conditional Cases

Rows P1 to P6 come from the repository-only researcher-pivot policy ladder,
and P7 to P11 from its second domain, information spread. They are
synthetic, stylized worlds, and their code is in prototype scripts, not in
`src/neural_abm`. Each statement is hash-audited in the consolidation ledgers
[v1](../docs/research/pivot-policy-ladder-consolidation-v1.md) to
[v7](../docs/research/pivot-policy-ladder-consolidation-v7.md). Ledger ids are
in parentheses. Use them as conditional cases for the central claim, never as
general performance claims.

| Id | Claim | Code path | Evidence artifact | Result summary | Required limitation |
| --- | --- | --- | --- | --- | --- |
| P1 | A fixed rule, pooled learners, and an external prior can be compared in one ABM design, with the rule ABM reproduced exactly before any learned arm is read. | `scripts/prototype_pivot_nonlinear_policy_abm_v2.py`, `scripts/prototype_pivot_llm_prior_pilot_v3.py`, `scripts/prototype_pivot_llm_persona_agent_v1.py` | `experiments/evidence/results/pivot_llm_persona_agent_v1_20261001.summary.md` (nesting checks); consolidation v1 to v4 | Each ladder step exactly reproduces its rule-based predecessor before a learned arm is read. In N07 every nesting check passed exactly, including one showing that the decision machinery without the persona reproduces the fixed-rule arm in every linear-world replicate. | Repository-only scripts. The pooled outcome record, prior slot, and exploration floor are not package API. |
| P2 | Policy class is a per-question modeling assumption that the lifecycle lets one test. | `scripts/prototype_pivot_matched_marginal_discriminator_v3.py`, `scripts/prototype_pivot_nonlinear_policy_abm_v2.py` | `pivot_matched_marginal_discriminator_v3_20260928`, `pivot_nonlinear_policy_abm_v2_20260928` summaries | Feature-conditional learning is needed when structure reverses (C07). Nonlinear policies beat linear ones on interactions (C08). The MLP wins smooth boundaries and the tree axis-aligned ones (C09). Every learner beats the prior with field-wide evidence and loses to it with neighbor evidence (C11). | Synthetic worlds built for this ladder. A neural policy is not shown to be necessary (R01), since linear and tree learners are in the comparison. |
| P3 | An external prior elicited once from a local LLM can enter the lifecycle, provided it is calibrated and paired with an exploration floor. | `scripts/prototype_pivot_llm_prior_pilot_v1.py` to `_v4.py` | `pivot_llm_prior_pilot_v1_20260929` to `v4_20260930` summaries | Used raw, the prior self-seals (C17). Shifted to a stated base rate, it stops sealing (C19). With a 20% floor it beat the hand rule on entrant precision while keeping pooled learning within the registered margin in 7 of 9 cells (C22), and 6 of 9 on fresh seeds (C29). | One Qwen3 release, 4B the smallest that worked (C26, C27). "Within the margin" is a registered tolerance, not equivalence. |
| P4 | Pooled learning adds modest cost over the rule ABM; per-agent learning and every-step LLM queries do not. | `scripts/prototype_pivot_relative_delay_cost_v1.py`, `scripts/prototype_pivot_llm_agent_reference_v1.py` | `pivot_relative_delay_cost_v1_20260929`, `pivot_llm_agent_reference_v1_20260930` summaries | Pooled linear, tree, and MLP policies cost 1.9 to 6.7 times the rule ABM per agent-step, and per-agent learners 57 to 65 times (C16). Elicitation took under 10 GPU minutes (C25). The every-step LLM agent needed about 110 GPU-seconds per run, against about 5 CPU-seconds for an NABM learner run (C35). | One machine and one design. Costs are for this ladder, not a general benchmark. |
| P5 | In the specified synthetic worlds and comparison conditions, the NABM hybrid met the registered performance criteria against two local LLM implementations. | `scripts/prototype_pivot_llm_agent_reference_v1.py`, `scripts/prototype_pivot_llm_persona_agent_v1.py` | `pivot_llm_agent_reference_v1_20260930`, `pivot_llm_persona_agent_v1_20261001` summaries | The hybrid matched or beat an every-step forecaster agent (C32) and a persona agent (C36) in 9 of 9 cells. The forecaster over-entered once outcomes were public (C34). The persona said Yes in 0.15% of questions and self-sealed (C37). | Both agents used the same Qwen3-4B-Instruct-2507. "Matches" means each mean metric within a 0.02 margin, not statistical equivalence. Calibration and decision rules differ between arms and drive much of the result. |
| P6 | The registration discipline stopped an empirical test that could not answer its question before confirmatory outcomes were read. | `scripts/build_pivot_empirical_ladder_risk_sets_v1.py`, `scripts/prototype_pivot_empirical_ladder_recovery_v3.py`, `scripts/explore_pivot_empirical_ladder_power_levers_v1.py` | `pivot_empirical_ladder_recovery_v3_20261012`, `pivot_empirical_ladder_power_levers_v1_20261013` summaries; consolidation v5 | At the measured data shape the registered analysis recovered learning in 1 of 8 (W1) and 3 of 8 (W2) replicates, against a required 7 of 8. No false recovery was observed (0 of 8 per world). The explored alternatives did not achieve power and false-recovery control together. N11c and N11d were not run. | A limitation and a protocol case, not evidence about learning. It does not imply the absence of learning. 0 of 8 bounds a world's false-recovery rate only below about 31%. v3 does not reproduce the empirical structure completely. |
| P7 | A policy table written before the run suffices in stationary worlds; the closed loop matters once run-specific information is no longer rare. | `scripts/run_info_spread_policy_ladder_study_v2.py`, `scripts/power_info_spread_policy_ladder_v4.py`, `scripts/run_info_spread_policy_ladder_transfer_v1.py`, `scripts/run_info_spread_policy_ladder_flip_rate_v1.py` | `info_spread_policy_ladder_study_v2_d4_20261019`, `info_spread_policy_ladder_d2d_controls_20261018`, `info_spread_policy_ladder_transfer_v1_20261021`, `info_spread_policy_ladder_flip_rate_v1_20261021` summaries; consolidation v6 (C49, C51) and v7 (C56, C57) | Every learner beat the hand rule. The closed loop at matched evidence was material in 1 of 12 groups (TABLE_SUFFICES_EVEN_IN_B), and the four-donor table beat the learner in all 12, also when its donors ran in other worlds of the same generator. With each run's relation reversed at rate p, the learner beat the four-donor table from p = 0.25 in all 4 families. | One generator and veracity relation; transfer across relations or domains is not tested (O30). One kind of run-specific information and one table size (O29). |
| P8 | NABM learners beat AgentTorch-style LLM archetypes given the same public information. | `scripts/run_info_spread_policy_ladder_llm_arms_v1.py` | `info_spread_policy_ladder_llm_arms_d5_20261019` summary; consolidation v6 (C52) and v7 (C60) | MOSTLY_POSITIVE in both regimes, 12 of 12 groups, by 0.16-0.23, and again under the agents' utility (C60). The archetypes also fell below the hand rule. | One local model (Qwen3-4B-Instruct-2507), one prompt, K = 2 threshold bins and 4 item classes, fixed in advance. Archetypes over-shared (1.4-1.7 times the hand rule). |
| P9 | On share precision, an LLM prior adds nothing detectable inside the learner, even when the prior alone is informative. | `scripts/run_info_spread_policy_ladder_llm_arms_v1.py`, `scripts/run_info_spread_policy_ladder_unclipped_prior_v1.py` | `info_spread_policy_ladder_llm_arms_d5_20261019`, `info_spread_policy_ladder_unclipped_prior_v1_20261021` summaries; consolidation v6 (C53) and v7 (C58, C59) | L1 - L0 was MIXED_OR_NONE in both regimes. Read without the inherited clip, the prior carried the relation's direction and beat the hand rule in all 12 groups; started from it, the learner was ahead in 5 of 12 groups, by 0.012-0.020, and behind in none. | On precision only: under the agents' utility the prior lowers it (P11). One model, one prompt, and the readout matters: the inherited clip saturated Regime A. |
| P10 | A registered null check caught a leak before any study group was read. | `scripts/run_info_spread_policy_ladder_study_v1.py`, `scripts/diagnose_info_spread_null_leak_v1.py` | `info_spread_policy_ladder_study_v1_d4_20261016`, `info_spread_policy_ladder_null_leak_diagnosis_20261018` summaries; consolidation v6 (C50, R16) | MLP learners recognized items already fact-checked and beat the hand rule in a world with nothing to learn. Study v1's groups were not read, and the measure moved to unverified items. | A protocol case, not a performance result. The diagnosis was exploratory. |
| P11 | Scored by the agents' utility, the framework comparisons hold and the LLM-prior readings change. | `scripts/run_info_spread_policy_ladder_utility_v1.py`, `scripts/info_spread_utility_measure_v1.py` | `info_spread_policy_ladder_utility_v1_20261022` summary; consolidation v7 (C60, C61, R19) | 15 of 20 summaries agree with precision: the best table, the closed loop, the hand rule, and the archetypes. All 5 changes involve an LLM prior: L1 - L0 lowers utility in 6 of 6 Regime A and 4 of 6 Regime B groups. | The utility counts only the sharer's own gain and cost, not the reach of false items (O31). Why the prior lowers utility is not tested (O28). |

## Gaps the Central Claim Exposes

This section proposes resolutions for next steps N13 (comparison design) and
N14 (framework) in consolidation v5. It is a proposal awaiting the user's
decision. Nothing here is authorized to run.

1. **A second domain, through a common API (N13 and N14 together).**
   - Run the fixed / adaptive / external-prior comparison in a second model
     family. This tests reusability, which is what the central claim asserts.
     It is not a performance test.
   - The same consumer satisfies the bridge rule, so only the semantics it
     confirms are extracted.
   - Candidate API surface, to be confirmed by that consumer:
     - a policy slot that holds a fixed rule, a learned policy, or a
       prior-initialized policy;
     - a pooled outcome record;
     - an exploration floor;
     - an exact nesting check that the learned machinery with learning off
       reproduces the fixed rule;
     - scoring through `policy_scoring`.
   - The second domain's prior can be a hand rule or a rule from the
     literature, so the step needs no GPU. An LLM prior there is optional.
2. **A fairer LLM comparison (N13), only if P5 stays in the paper.**
   - The candidate is an LLM baseline given the same information and
     recalibrated on public outcomes.
   - Today the hybrid is calibrated to a base rate and learns from outcomes,
     while neither LLM agent is recalibrated.
   - If this is not run, P5 must carry that asymmetry as a limitation.
3. **A larger local model (optional).** It tests sensitivity to model scale
   only and does not remove the comparison asymmetry. It would run on the GPU
   server.
4. **Inputs from the LLM-ABM survey.** The
   [related-work contrast](related-work/llm-abm-frameworks-v1.md) adds two
   candidates:
   - **For gap 2: an AgentTorch-style archetype baseline,** an LLM queried
     per archetype at each step without outcome learning. It is the closest
     published alternative to a once-elicited prior.
   - **For gap 1: an information- or opinion-spread domain,** where most LLM-ABM
     applications sit and where the repository's DeGroot and Granovetter
     reductions already hold.
5. **Chosen on 2026-10-13.** The user chose the information-spread domain
   and the archetype baseline. The draft design is
   [info-spread-policy-ladder-design-v1.md](../docs/research/info-spread-policy-ladder-design-v1.md).
   - It adds a precomputed-table arm, which tests whether NABM differs from a
     rule table computed before the run.
   - It contrasts a regime where a table should suffice with one where
     computation during the run should matter.
6. **Resolved on 2026-10-19 (consolidation v6).**
   - **Gap 1, a second domain.** Answered: P7 to P10, C47.
   - **N14.** Answered by the byte-for-byte Pivot parity, C48, and Decision
     0017.
   - **Gap 2, a fairer LLM comparison.** Partly answered by the archetype
     baseline (P8). The every-step agents of P5 were not rerun with
     recalibration, so P5 keeps its asymmetry as a limitation.
7. **Resolved on 2026-10-22 (consolidation v7).**
   - **N15.** The package reproduces both recorded Pivot studies (C55).
   - **O23 to O26.** Answered: P7, P9, and P11.
   - **New gaps.** O28 to O31, and a release that ships the package surface
     (N22). Each awaits the user's choice.

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
| A neural policy is necessary for the policy-ladder results. | Unsupported (R01). Linear and tree learners are in every comparison, and the gain comes from learned policies rather than neural ones as such. | A registered test where every non-neural learner fails for a stated reason. |
| NABM is statistically equivalent to, or generally better than, LLM-agent simulation. | Unsupported. P5 is a registered-tolerance result against two implementations of one 4B model. | An equivalence design, a recalibrated LLM baseline, and more than one model family. |
| Real researchers' entry choices learn from visible outcomes. | Not established. The current N11 design closed before confirmatory scoring (consolidation v5). This does not imply the absence of learning. | A design that shows power at its own measured shape (O22). Outside Paper-1. |

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
- Treat the unit contract as the first half of the central claim of
  2026-10-13. The second half is a reusable structure for comparing and
  tracing fixed, adaptive, and prior-informed policies. Present P1 to P6 as
  conditional cases of that structure.
- Report 0 of n replicates with its one-sided 95% upper bound, about 31% at
  n = 8.
- Write "matched" only with its registered margin, and never as statistical
  equivalence.
- State the LLM scope wherever P3 or P5 appears: one local Qwen3 model
  family, two agent designs, and calibration and decision rules that differ
  between arms.
- Present the empirical stage (P6) as a limitation and a protocol case, with
  its closure statement.

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
9. Restructure the outline around the central claim, with a policy-ladder
   section built from P1 to P6.
10. Section 8 is drafted from P1 to P11 with the consolidation v7 reporting
    rules (`sections/08-policy-ladder.md`, N16, updated in N19). Its numbers
    are checked against the recorded results by
    `tests/test_paper_section_08_policy_ladder.py`. Report every comparison
    that involves an LLM prior on both precision and utility (P11).
11. Cite the prior rule, outcome record, exploration floor, and prior-anchored
    learners as `main` / next-alpha package API (Decision 0017), not as part
    of the released `v0.1.0a5` tag.
