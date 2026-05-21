# Decision 0007: Basin-Centric Relational NABM Roadmap

## Status

Proposed.

## Date

2026-05-13

## Context

The recent Toy2/Toy4 work added a basin-credit implementation slice on top of
the existing neural loop backend and fixed social mixer. That slice is useful
because it makes basin-targeted credit assignable, testable, and visible in
aggregate and micro logs. It should not yet be treated as the final research
mechanism.

The intended long-term architecture is no longer policy-centric. The project
direction is transition-centric and basin-centric:

```text
ARE Tokens
  -> Relational Transition Encoder
  -> Post-Social State Representation
  -> Contrastive Basin Critic
  -> Counterfactual Basin Credit
  -> Agent Policy Update
```

The core unit is an ARE token set:

- `Agent`: action, policy probability, payoff/EMA, local observation,
  reputation, and memory latent.
- `Relation`: graph edge, neighbor role, influence or reputation exposure, and
  local coupling features.
- `Environment`: global resource, payoff regime, macro action rate,
  fragmentation, and domain state.
- `Event`: an extension token for shocks, interventions, rule changes, and
  external perturbations.

The central learning question is not whether an agent action is individually
good. The central question is whether an agent-level intervention moves the
post-social collective trajectory toward the target basin.

## Current Implementation Boundary

The current implementation is a prototype scaffold, not the final basin-centric
NABM mechanism.

Implemented:

- `BasinCreditConfig` exposes the v1 config axis under Toy2 and Toy4 policy
  domain config.
- The executable v1 critic mode is named `prototype_phase`; `contrastive_phase`
  is reserved until a learned contrastive critic exists.
- Toy2 and Toy4 reject basin credit unless the run uses `neural_policy`, the
  `loop` neural backend, and a state-continuation-compatible objective path.
- Toy2 and Toy4 compute one-step ablation credit for revised agents through a
  post-social runner hook.
- Aggregate and micro diagnostics expose basin score, score delta, credit
  positivity, phase confidence, and applied per-agent credit.
- `experiments/evidence/toy24_basin_credit_quick.yaml` records the intended
  comparison variants and Toy2/Toy4 success criteria.
- `scripts/run_basin_credit_evidence_gate.py` evaluates existing evidence
  matrix run rows and writes an audited JSON/Markdown pass/fail report for the
  basin-credit scaffold.
- `scripts/run_basin_credit_evidence_workflow.py` can run the Toy2/Toy4
  basin-credit evidence matrix and immediately evaluate the hardened gate.

Not implemented yet:

- The critic is not a true contrastive learner. It is a fixed-shape,
  hand-engineered phase embedding scored against online prototypes.
- The critic has no positive/negative phase-window training data, InfoNCE loss,
  frozen evaluation mode, or replayable train/eval separation.
- The evidence gate now rejects malformed or mismatched run rows and separates
  missing evidence from failed criteria, but it does not by itself prove the
  contrastive critic or relational encoder claims.

This boundary matters for claim discipline. The current code can be described
as `prototype_phase_basin_credit` or `basin-credit v1 scaffold`. It should not
yet be described as a learned contrastive basin critic or as evidence for the
full transition-centric relational NABM architecture.

## Decision

Keep the current Toy2/Toy4 implementation as a bounded v1 scaffold, but change
the roadmap and claim language around it.

The target architecture remains basin-centric and relational. The next work
must close the gap in this order:

1. Keep the current scaffold honest and auditable.
2. Enforce evidence success criteria through an executable report gate.
3. Replace prototype scoring with a real contrastive phase critic.
4. Only then decide whether to learn the relation operator.

Teacher, bootstrap, replay, and reputation-imitation paths remain baselines and
diagnostics. They are not the main mechanism for the new NABM claim.

## Improvement Plan

### Phase 0: Claim Cleanup

Goal: prevent the prototype from overstating the mechanism.

Work:

- Rename or document the current critic as a prototype phase critic unless the
  contrastive learner is implemented in the same change.
- Add a short implementation note near `BasinCreditConfig` explaining that v1
  supports only one-step ablation and prototype scoring.
- Split diagnostics into mechanism diagnostics and scaffold diagnostics where
  needed. Prototype-specific values should not be reported as contrastive loss
  or learned phase quality.

Exit criteria:

- Docs and public labels do not imply InfoNCE or learned contrastive training
  when the run is using prototype scoring.
- Tests still prove that disabling all basin-credit weights preserves the
  existing objective path.

### Phase 1: Post-Social Basin State

Goal: make basin credit answer the intended question: whether an intervention
changes the post-social collective trajectory.

Work:

- Add a shared post-social basin-state builder for Toy2/Toy4 after local update,
  social coordination, action commit, and domain payoff/resource update.
- Include ARE-compatible fields: final actions, post-social policy
  probabilities or logits, payoffs, resource state, macro action rate,
  reputation exposure, and fragmentation.
- Change one-step ablation to preserve the rest of the realized path while
  changing only the selected intervention variable.
- Add invariant tests that prove a counterfactual changes only the selected
  agent action/intervention and recomputes only the expected dependent domain
  state.

Exit criteria:

- Toy2 and Toy4 basin diagnostics are explicitly post-social.
- Micro diagnostics can distinguish observed post-social score from selected
  counterfactual post-social score.
- Pre-social basin-credit code paths are removed or labeled diagnostic-only.

### Phase 2: Evidence Gate

Goal: make success or failure machine-auditable before adding a larger critic.

Work:

- Extend the basin-credit evidence report to encode success criteria:
  Toy2 final ceiling `3/3` and mean time-to-ceiling `< 10`; Toy4 final ceiling
  `>= 2/3` and mean time-to-ceiling `< 12`.
- Mark teacher/bootstrap/replay variants as baselines or diagnostics, not main
  success variants.
- Add a report field that says whether main success was achieved without
  teacher/bootstrap/replay.
- Reject stale or malformed run rows whose label, case, variant, seed, toy, or
  group does not match the manifest.
- Preserve generated configs, run directories, aggregate CSVs, micro CSVs, and
  summary rows for audit.

Exit criteria:

- A single evidence command produces an explicit pass/fail basin-credit report.
- The report separates effect size, variance across seeds, and claim status.
- Missing seeds produce `inconclusive`; malformed observed rows produce an
  input error or an explicit inconclusive result rather than a silent pass/fail.
- Failure produces a clear next diagnostic rather than encouraging another
  unconstrained parameter sweep.

### 2026-05-19 Time-to-Ceiling Interpretation Boundary

The quick gate keeps time-to-ceiling thresholds as an operational stress test,
but faster ceiling arrival is not by itself the scientific target. In ABM
terms, slower commitment can be a valid outcome when neural agents are
integrating social information, avoiding premature cascades, or preserving
ambivalence under weak evidence.

Therefore TtC should be interpreted after the following are separated:

- final ceiling recovery versus collapse;
- slow monotone climb versus threshold-band oscillation;
- population-wide slow commitment versus polarized partial adoption;
- policy confidence versus realized action revision lag;
- local policy movement versus post-social damping.

The next diagnostic slice records post-local and post-social threshold rates,
0.4-0.6 dwell-band rates, p10/p50/p90 policy-probability quantiles, temporal
threshold-crossing counts, and action-flip rates before adding commitment
losses or social-propagation changes. Commitment-margin losses remain
diagnostic-only until this readout shows that slow TtC is a harmful commitment
failure rather than a defensible information-integration delay.

### Phase 3: True Contrastive Phase Critic

Goal: replace hand-scored prototypes with a learned phase representation.

Work:

- Build phase trajectory windows from aggregate and micro state histories.
- Define positives as windows from the same future phase family and negatives
  as windows from different terminal or macro trajectory families.
- Implement a small critic module that returns `basin_embedding`,
  `target_basin_score`, `non_target_basin_score`, and `phase_confidence`.
- Train with an InfoNCE-style phase loss and report the loss separately from
  policy loss.
- Add frozen critic evaluation mode so policy credit is not computed from a
  simultaneously drifting scorer.

Exit criteria:

- Unit tests cover Toy2/Toy4 critic batch shapes, positive/negative pair
  construction, and finite contrastive loss.
- Diagnostics include critic loss, phase-family counts, and frozen/eval score
  summaries.
- The code can still run without teacher, bootstrap, or replay paths enabled.

### Phase 4: Relational Encoder Extension

Goal: move from fixed social path plus basin credit to relational transition
learning.

Work:

- Introduce explicit `Agent`, `Relation`, and `Environment` token builders.
- Add a relational transition encoder that consumes ARE tokens and emits the
  post-social state representation used by the critic.
- Keep the fixed social mixer as a reference path while testing whether the
  learned relation encoder improves basin transitions.
- Only add `Event` tokens after shocks or interventions become part of the
  evaluated task.

Exit criteria:

- Learned relation encoding changes transition behavior, not only reporting.
- Ablations show whether basin improvement comes from the critic, the
  counterfactual credit path, or the learned relation encoder.
- Claim language can move from scaffold to mechanism only if this phase changes
  outcomes under audited evidence.

## Open Questions

- Should contrastive critic training be offline from stored trajectory windows
  first, or online with a frozen target/eval copy?
- What is the minimum ARE representation that keeps Toy2/Toy4 comparable while
  still being general enough for Toy5 or later event-token work?
- Should the first evidence gate compare against reputation imitation only, or
  include the strongest teacher/bootstrap diagnostics as non-main baselines?

## Consequences

- The project keeps the structural research direction without pretending the
  first implementation already proves it.
- Future work is blocked on representation and evidence quality, not another
  sequence of weight sweeps.
- The current basin-credit code remains useful as an integration and diagnostic
  scaffold.
- Stronger NABM claims require post-social credit, a learned contrastive critic,
  and an audited evidence gate.
