# Paper Outline

Working title:

> Neural Social Nodes for Agent-Based Modeling

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
- Claim: the right positioning is a temporal heterogeneous graph simulation
  framework, not a direct Transformer replacement.
- Contribution preview: unit contract (Sec. 3), toy evidence (Secs. 4-5),
  calibration discipline (Sec. 6), and an end-to-end research application
  with a fixed-rule-versus-learning controlled comparison (Sec. 7).

## 2. Background

- Agent-based models and bounded confidence.
- Verified classical instances and labeled near variants: exact DeGroot and
  Granovetter special cases, FJ-like pre-mix anchoring, and self-excluding HK
  bounded confidence as settings of the unit lifecycle:
  `docs/classical-reductions.md`, `examples/classical_reductions.py`.
- Neural agent-based models.
- LLM-agent social simulation and its audit/reproducibility critiques, as
  the opposite pole motivating the auditable-middle positioning.
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

## 8. Discussion

- What the toy models show.
- What the research application shows: the contract carries a full study,
  and learning direction is a first-class experimental axis.
- Limits of the Transformer analogy.
- Limits of parameter averaging.
- Path to richer ABM environments.

## 9. Conclusion

- Neural Social Nodes are a practical unit for NABM prototyping.
- The primitive claim should be earned through ablations, not assumed.
