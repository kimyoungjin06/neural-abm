# Paper Outline

Working title (2026-10-13 candidate):

> Neural ABM: Comparing and Tracing Fixed, Adaptive, and Prior-Informed Agent
> Policies in One Simulation Lifecycle

Previous working title: Neural Social Nodes for Agent-Based Modeling.

Central claim (set by the user on 2026-10-13; see
[claim-matrix.md](claim-matrix.md) and
[consolidation v5](../docs/research/pivot-policy-ladder-consolidation-v5.md)):
NABM is a reusable framework that separates policy learning from social
interaction, so fixed policies, adaptive policies, and external priors can be
compared and traced within one simulation lifecycle. Performance comparisons
are conditional cases, not the central claim.

Route status: this is the method-first outline. The prospective science-first
pivot-feedback route is specified in the
[empirical/NABM study protocol](../docs/case-studies/researcher-pivot/pivot-feedback-empirical-nabm-protocol.md).
Do not enlarge Section 7 into a substantive science-of-science claim. If the
protocol's empirical signature, baseline-model, and final-holdout gates pass,
open a separate `paper/pivot-feedback/` companion track; keep Section 7 as
stylized lifecycle evidence for this method-first manuscript.

The repository-only prospective cross-field tranche is routed through
`experiments/configs/openalex_g1_cross_field_feasibility_v0.yaml`,
`experiments/evidence/results/openalex_g1_cross_field_feasibility_20260718.summary.json`,
`experiments/evidence/results/openalex_g1_cross_field_feasibility_20260718.summary.md`,
`experiments/configs/openalex_g1_field_taxonomy_reference_v0.yaml`,
`experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.json`,
`experiments/evidence/results/openalex_g1_field_taxonomy_reference_20260718.summary.md`,
`experiments/configs/openalex_g1_w0_positive_id_sensitivity_v0.yaml`,
`experiments/evidence/results/openalex_g1_w0_positive_id_sensitivity_20260718.summary.json`,
`experiments/evidence/results/openalex_g1_w0_positive_id_sensitivity_20260718.summary.md`,
`experiments/configs/openalex_g1_work_topic_relation_v0.yaml`,
`scripts/audit_openalex_work_topic_relation.py`,
`experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml`, and
`scripts/plan_research_pivot_field_matrix.py`. Its compact result is
profile-feasibility only, and its planner is fail-closed engineering support.
The compact feasibility artifact records `field_entity_validity=UNVERIFIED`,
`candidate_eligibility=NOT_ESTABLISHED`, and
`final_field_selection=PENDING_G0`. The later full CPU/I/O component has status
`PASS_CURRENT_SNAPSHOT_TAXONOMY_REFERENCE_COMPONENT`. Across 1,423 files and
1,420,855,753 projected rows, it validates 1,282,707,794 complete non-null
hierarchy references, permits 138,147,958 all-null main-work rows, quarantines
one frozen `W-1` row, and finds zero partial or invalid hierarchy rows. The pass
covers non-null current-snapshot projected references and exact 26-field plus
null margins only. Raw source validity remains `BLOCKED`, the retained `W0`
remains `UNRESOLVED`, and parent content identity remains `UNVERIFIED`. A
separate bounded sensitivity pass shows that excluding W0 changes only the
null margin by one; all 26 non-null field margins are unchanged. The full
work-topic relation audit finds 714 duplicate edge groups and zero missing
primary-child edges, but two duplicate groups have different scores. It
therefore blocks without materializing the exact-score dedup view. These
components do not rewrite the compact labels or establish broader field or
relation validity.
Final empirical field selection remains `PENDING`, empirical G1 is `NOT_RUN`
with `pass_g1=false`, and neither a statistical model nor an NABM has run.
These artifacts do not expand the method-first primary claim path. The separate
planner keeps scientific selection `PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT`
and `execution_authorized=false`.

## 1. Introduction

- Motivation: rule-based ABM agents are interpretable but limited in adaptive
  learning behavior; fully neural and LLM-agent simulations adapt but resist
  attribution, audit, and seeded replication.
- Positioning: the auditable middle — the maximum agent adaptivity that
  still supports paired counterfactual control, named-parameter audit, and
  deterministic replication. Not a cheaper substitute for LLM-agent
  simulation and not a replacement for classical ABM; a different point with
  different guarantees.
- Goal: define a reusable Neural ABM Node with explicit social mixing.
- Claim: the central claim above. Policy learning is separated from social
  interaction, so fixed, adaptive, and prior-informed policies can be compared
  and traced in one lifecycle. The framework is a temporal heterogeneous graph
  simulation, not a direct Transformer replacement.
- Contribution preview: unit contract (Sec. 3), toy evidence (Secs. 4-5),
  calibration discipline (Sec. 6), an end-to-end research application
  with a fixed-rule-versus-learning controlled comparison (Sec. 7), and a
  policy ladder that adds external priors and conditional cost and LLM-agent
  comparisons (Sec. 8).

## 2. Background

- Agent-based models and bounded confidence.
- Verified classical instances and labeled near variants: exact DeGroot and
  Granovetter special cases, FJ-like pre-mix anchoring, and self-excluding HK
  bounded confidence as settings of the unit lifecycle:
  `docs/classical-reductions.md`, `examples/classical_reductions.py`.
- Neural agent-based models.
- LLM-agent social simulation and its audit/reproducibility critiques, as
  the opposite pole motivating the auditable-middle positioning. Contrast
  axes and sources are in
  [related-work/llm-abm-frameworks-v1.md](related-work/llm-abm-frameworks-v1.md).
  The axes are:
  - what decides an action;
  - how behavior changes;
  - learning across agents;
  - rule agents in the same environment;
  - reproducibility controls;
  - scale;
  - validation against real data.
- Social learning via state-based and output-based updates; imitative versus
  postulated failure-only outcome updates as the Sec. 7 learning-rule axis,
  with the canonical informational-cascade definition kept separate.
- Dynamic graph learning, federated learning, and distillation lineages.
- Model capacity as a modeling assumption: cue-weighting/improper-linear
  views of judgment and the interaction-first ABM tradition (grounds for
  Sec. 7.5).

## 3. Neural ABM Node

Draft: [sections/03-neural-abm-node.md](sections/03-neural-abm-node.md)

- Fast, medium, and slow state.
- Common node interface.
- NABM Unit v1 contract: what the reusable unit owns, and which domain
  semantics stay outside it.
- Social mixing taxonomy.
- Communication budget and synchronization modes.
- Logging requirements.

## 4. Toy 1: Neural HK Classification

- Biased private data shards.
- State-similarity versus output-similarity peer selection.
- Mixing ablations.
- Accuracy, consensus, polarization, and fragmentation.

## 5. Toy 2: Neural Spatial Prisoner's Dilemma

- Game dynamics and social mixing.
- Cooperation, payoff, exploitation, and cluster formation.
- Transfer of the same mixer taxonomy from Toy 1.

## 6. Calibration and Analysis

Draft:
[sections/06-calibration-and-analysis.md](sections/06-calibration-and-analysis.md)

- Manuscript claim matrix: every claim must map to code path, manifest, result
  artifact, figure/table candidate, and limitation.
- Manuscript table candidates: unit-boundary, Toy5 safety/spread, Gate 3
  failure modes, Toy2/Toy4 reputation fragility, and Toy4 local resource stress.
- Toy5 threshold-aware readiness grid: no-seed safety and sparse-seed spread.
- Toy2/Toy4 failure-mode taxonomy: stochastic final-epoch hazard, slow TtC,
  baseline-favored environment, and mechanism failure candidate.
- Targeted Toy2/Toy4 robustness: sparse seeds, open boundaries, noisy ranking,
  and Toy4 heterogeneous resource extraction.
- Micro-state logs.
- Sensitivity analysis.
- Time-to-ceiling as a secondary diagnostic: distinguish collapse recovery,
  slow monotone climb, ambiguous probability dwell, threshold oscillation,
  polarization, and policy-action lag before treating speed as an objective.
- Possible simulation-based inference or temporal-GNN posterior estimation.

## 7. Research Application: Replicated Scenario Studies

Draft:
[sections/07-research-application.md](sections/07-research-application.md)

- A researcher-authored science-of-science question (field-pivot
  productivity) carried end to end on the reusable surfaces.
- Study 1: baseline/counterfactual environments, seed-paired replication,
  placebo control, sensitivity sweeps (torch-free surface).
- Study 2: frozen versus imitative and failure-only learning agents on the
  NABM lifecycle; environment-dependent feedback loops, trade-offs, and full
  named-parameter traces as bounded endogenous rule-change results.
- Expressiveness and discipline claim only; the fixed rule is a control, not
  a defeated baseline.
- Keep the prospective H3 field-balanced timing-gap primary, the episode-level
  bibliographic outcome-availability-lag / prior-collaborator
  exposure-opportunity-timescale transport secondary, equivalence placebos,
  Holm hierarchy, and domain-and-author-cold G4 plan in the separate
  pivot-feedback route. The current cross-field profile and matrix planner are
  not Section 7 empirical results.
- Treat prior-collaborator activity and outcome availability as proxies; test
  them against field-wide attention, homophily, chaperoned entry or
  recruitment, and capacity bridges before invoking social learning.

## 8. Policy Ladder: Fixed, Adaptive, and Prior-Informed Policies

Source: claim matrix rows P1 to P10. Every result here is synthetic and
repository-only, and each is a conditional case of the central claim.
Draft: [sections/08-policy-ladder.md](sections/08-policy-ladder.md).

- Design (P1): one ABM design holds a fixed rule, pooled learners, and an
  external prior. Each step reproduces its rule-based predecessor exactly
  before a learned arm is read. Seeds are paired, and every claim is bound to
  a hash-audited ledger.
- Policy class as a modeling assumption (P2): feature-conditional and
  nonlinear learning are needed only in some worlds; the best family depends
  on boundary geometry; and evidence scale decides whether learning beats the
  prior. A neural policy is not shown to be necessary.
- External priors (P3): a local LLM consulted once, calibrated to a stated
  base rate, and paired with an exploration floor. Report the self-sealing
  failure of the raw prior alongside the calibrated result.
- Cost (P4): pooled learners against the rule ABM, per-agent learners, and
  every-step LLM queries, measured on one machine.
- Conditional LLM-agent comparison (P5): "In the specified synthetic worlds
  and comparison conditions, the NABM hybrid met the registered performance
  criteria against two local LLM implementations." State the 0.02 margin, the
  single 4B model, and the calibration asymmetry. Keep this subsection only if
  the gap analysis in the claim matrix keeps P5.
- The empirical stage as a limitation (P6): the protocol stopped a test that
  lacked power at the measured data shape before confirmatory outcomes were
  read. Quote the closure statement, including that it does not imply the
  absence of learning.
- A second domain, information spread (P7 to P10, consolidation v6):
  - **Reuse.** The same comparison runs through a shared core. The Pivot
    ladder, re-expressed through that core, reproduces its record byte for
    byte (C47, C48).
  - **The precomputed-table critique (P7).**
    - In stationary worlds, a table written before the run from same-world
      donor runs is not detectably different from the closed loop in 11 of
      12 groups, and the four-donor table beats it in all 12.
    - The closed loop wins only when the run carries information the donors
      lack.
    - State the same-world boundary.
  - **LLM comparators (P8, P9).** NABM learners beat AgentTorch-style
    archetypes in all 12 groups. An LLM prior adds nothing detectable inside
    the learner. State the saturated Regime A readout.
  - **Protocol (P10).** A registered null check caught a leak before any study
    group was read, and the measure moved to unverified items.
  - **Package status.** Decision 0017 put the shared semantics in
    `neural_abm.policy_ladder` and `neural_abm.policy_learners` on the
    `main` / next-alpha candidate. Both recorded Pivot runs reproduce byte
    for byte through them, including under delays (N15).

## 9. Discussion

- What the toy models show.
- What the research application shows: the contract carries a full study,
  and learning direction is a first-class experimental axis.
- What the policy ladder shows: fixed, adaptive, and prior-informed policies
  can be compared in one lifecycle, and when each is worth its cost.
- When NABM is more than a precomputed table:
  - In stationary, same-world designs, it is not.
  - It is when a run carries information that no table written before the
    run can hold.
  - Where between the two the closed loop begins to matter is open (O26).
- Limits of the evidence:
  - the worlds are synthetic and were designed by the authors;
  - the empirical test of learning closed without an answer;
  - the LLM comparisons cover one model family, and one of its readouts
    saturated;
  - the table comparisons use donors from the same world;
  - neural necessity is not established.
- Limits of the Transformer analogy.
- Limits of parameter averaging.
- Path to richer ABM environments.

## 10. Conclusion

- Neural Social Nodes are a practical unit for NABM prototyping.
- The lifecycle makes the choice between a fixed rule, a learned policy, and
  an external prior an explicit, testable modeling decision.
- The primitive claim should be earned through ablations, not assumed.
