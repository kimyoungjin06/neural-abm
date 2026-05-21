# Decision 0008: Basin Critic Borrowed Concepts

## Status

Proposed.

## Date

2026-05-14

## Context

The Toy2/Toy4 basin-credit evidence now separates three claims:

- Objective-basin blending fixes the earlier `material + basin` collapse.
- Fixed replay strength is still a training protocol axis: `p2` is stable and
  slower, while `p3` is faster and less stable in Toy4.
- `credit_signal_escalation` shows that early noisy basin-credit amplification
  is the concrete failure mode to address.

The next step should not be another threshold sweep. The structural target is
to replace the hand-scored `prototype_phase` critic and hand-set replay gate
with a learned counterfactual basin critic and uncertainty-aware replay control.

This document records which concepts are borrowed from prior work, how they are
adapted, and which claims remain specific to this project. It is a claim-boundary
document, not an implementation result.

## Borrowed Concept Ledger

| Borrowed concept | Source role | NABM adaptation | Claim boundary |
| --- | --- | --- | --- |
| Counterfactual multi-agent credit assignment | COMA uses a centralized critic and a counterfactual baseline that changes one agent action while holding the rest of the joint context fixed. Difference rewards motivate the same agent-specific contribution framing. | Basin credit should compare the observed post-social basin transition with a selected one-agent counterfactual transition. The credit target is a basin-level transition, not immediate individual payoff. | We should not claim COMA itself. The adaptation is a basin-transition critic for ABM toys, not a StarCraft-style centralized actor-critic. |
| Difference-reward / wonderful-life utility intuition | Difference rewards compare global utility with and without an agent's contribution. | One-step ablation currently approximates this by replacing one agent action and measuring the change in basin score. | Current prototype scoring is a scaffold. It is not a full difference-reward guarantee unless the learned critic and counterfactual environment model are validated. |
| Contrastive predictive representation learning | CPC learns representations by predicting future latent states against negatives. | A contrastive basin critic can learn phase embeddings that rank future ceiling-basin windows above non-ceiling windows. | The project should not use `contrastive_phase` labels unless positive/negative basin windows, loss, train/eval split, and diagnostics exist. |
| Successor-feature style dynamics/reward separation | Successor features decouple environment dynamics from reward weights so representations transfer across reward changes. | A basin phase representation should encode transition tendencies separately from the local material/social objective weights. | This is conceptual guidance. We are not implementing successor-feature Bellman backups unless that is explicitly added. |
| Ensemble or bootstrap uncertainty | Bootstrapped DQN uses randomized value-function heads to represent uncertainty and improve temporally extended decisions. | A learned basin critic can expose ensemble disagreement or margin uncertainty. Replay escalation should use that uncertainty to abstain or stay at a conservative pass count. | A fixed positive-rate threshold is still hand control. An uncertainty gate becomes structural only when uncertainty is produced by the critic. |
| Conservative offline value estimation | CQL penalizes overestimated out-of-distribution values in offline RL. Counterfactual conservative Q-learning extends this concern to offline MARL with per-agent counterfactual regularization. | Basin transition samples are offline artifacts. Counterfactual action samples may be out of the logged distribution, so the critic should support abstention or conservative penalties. | This is a safety constraint on the critic. It is not an offline-RL algorithm claim unless we implement and test the conservative objective. |
| Latent interaction representation | Work on latent representations for multi-agent interaction models high-level policy/strategy effects instead of every low-level action. | The future relational NABM critic should operate on post-social phase/ARE-token representations, not raw action lists alone. | This supports the representation direction. It does not prove the ARE token encoder until that encoder changes behavior in evidence runs. |

## Proposed Structural Design

The next implementation should introduce a learned critic path behind the
reserved `critic="contrastive_phase"` name.

Target data flow:

```text
post-social trajectory logs
  -> basin transition sample table
  -> contrastive basin critic training
  -> frozen critic scoring
  -> counterfactual basin advantage
  -> uncertainty-aware replay gate
```

### Basin Transition Samples

Add a replayable dataset artifact, initially Parquet:

```text
basin_transition_samples.parquet
```

Minimum fields:

- `toy`, `run_id`, `seed`, `epoch`, `agent_id`
- observed action and counterfactual action
- phase embedding fields used by the current prototype critic
- action rate, policy rate, payoff alignment, consensus, payoff stability
- observed basin score, counterfactual basin score, selected credit
- final ceiling reached, time-to-ceiling, final metric
- variant, critic mode, replay schedule, training scope, effective pass count

Train/eval separation is mandatory. The critic must not be trained on the same
seeds used to evaluate basin-credit claims.

### Learned Critic V1

The first learned critic should be intentionally small:

- Input: phase embedding, context features, and candidate action.
- Output: `target_basin_score`, `non_target_basin_score`, `phase_confidence`,
  and optionally time-to-ceiling prediction.
- Loss: binary terminal-basin classification plus pairwise/contrastive ranking
  between ceiling and non-ceiling trajectory windows.
- Diagnostics: AUC or pairwise ranking accuracy, calibration bins, finite loss,
  positive/negative counts, train/eval seed split.

The critic should be frozen during policy-credit evidence runs unless an
explicit online-critic experiment is being tested.

### Counterfactual Basin Advantage

Replace prototype score deltas with learned action-value deltas:

```text
basin_action1_advantage =
  Q_basin(phase, action=1, context)
  - Q_basin(phase, action=0, context)
```

This keeps the counterfactual action-comparison semantics while moving the
score source from a hand-engineered cosine prototype to a trained basin critic.

### Uncertainty Replay Gate

The current gate:

```text
positive_rate >= 0.6 and credit_delta >= 0.0
```

should become:

```text
if critic_margin > 0 and critic_uncertainty is low:
    use high replay pass count
elif critic_uncertainty is high or counterfactual sample is OOD:
    use conservative replay pass count or abstain
else:
    use conservative replay pass count
```

The initial uncertainty source can be a small bootstrapped ensemble of critic
heads. The gate should log margin, uncertainty, abstention rate, and chosen pass
count.

## Long-Term Goal

The long-term research target is not another replay threshold, floor, or fixed
soft-weight policy. The final goal is a generalizable basin-aware NABM
mechanism where future basin motion supervises agent-time credit attention.

Target progression:

```text
hand basin credit
  -> learned basin critic
  -> learned replay-weight scorer
  -> learned credit-attention controller
  -> generalizable basin-aware NABM algorithm
```

The desired final architecture is:

```text
trajectory / phase tokens
  -> basin dynamics critic
  -> counterfactual action scoring
  -> agent-time credit attention
  -> replay/update weighting
  -> policy update
  -> future basin-motion supervision
```

The core claim should become:

> A neural ABM can learn which agent-time updates causally support desirable
> collective basin transitions, and can allocate policy-update credit through a
> learned basin-motion attention mechanism rather than through hand-set replay
> schedules.

This requires changing the replay-weight target. The current learned scorer is
only an intermediate step because it imitates transition-signal magnitude. The
long-term target should instead predict held-out basin motion or policy
improvement:

```text
this replay update predicts future basin_score_delta
this replay update predicts faster ceiling arrival
this replay update moves policy probability toward successful basin action
```

### Long-Term Success Criteria

The final mechanism should not be considered solved until it satisfies all of
the following:

- It matches or improves ceiling arrival on Toy2/Toy4 without relying on
  hand-set replay floors, fixed soft-min constants, or all-replay fallback as
  the main mechanism.
- It achieves the same or faster basin arrival with lower average replay weight
  or lower replay budget than prototype/all-replay controls.
- It generalizes across held-out seeds, changed payoff regimes, resource-enabled
  Toy4, at least one additional binary social-dilemma toy, and changed agent
  counts.
- It logs interpretable agent-time attention diagnostics: replay weight,
  critic margin, uncertainty, future basin-motion target, policy probability
  shift, and ablation impact.
- It has explicit train/eval separation for critic and attention targets, with
  no leakage from the evidence seeds used for the main policy claim.
- Its failure modes are visible: starvation, near-all-replay collapse, OOD
  counterfactual samples, critic overconfidence, and weak policy-probability
  movement must be separately diagnosed.

### Claim Boundary

Until those criteria hold, the implementation should be described as a sequence
of structural prototypes, not as a complete algorithm. The current evidence
supports this narrower statement:

> A frozen learned replay-weight module can preserve the Toy2/Toy4 quick gate,
> but the current target is still conservative and can stay close to all-replay,
> especially in Toy4.

The next durable step is therefore future basin-motion supervision, not replay
coefficient tuning.

## Evidence Plan

Add evidence in three steps instead of a large all-in-one run.

## Implementation Notes

### 2026-05-14 Dataset Contract Slice

Implemented the first structural slice:

- Toy2 and Toy4 basin-credit runs now write
  `basin_transition_samples.parquet` when basin diagnostics are available.
- Rows are per agent, per epoch, and only include agents inside the active basin
  training mask.
- The artifact records the prototype phase features, observed and
  counterfactual basin scores, selected credit, objective effective advantage,
  basin action-1 advantage, blended training effective advantage, and terminal
  run labels.
- `toy24_basin_transition_dataset_quick` passed with the existing
  `mixed_objective_basin_escalate_credit_p3_min2_h1` producer variant:
  Toy2 reached final ceiling `3/3` with mean time-to-ceiling `9.33333`, and
  Toy4 reached final ceiling `3/3` with mean time-to-ceiling `11.6667`.
- The full quick run produced three Toy2 artifacts and three Toy4 artifacts;
  each artifact contains 5,000 rows for 50 epochs x 100 agents.
- This does not implement `contrastive_phase`; it only creates the replayable
  offline dataset needed to train and evaluate a learned critic later.

### 2026-05-14 Offline Learned Phase-Critic Quality Slice

Implemented the second structural slice:

- Added an offline learned basin phase critic that reads
  `basin_transition_samples.parquet`, trains on configured train seeds, and
  evaluates on held-out seeds.
- The initial critic is a small linear phase scorer trained with binary
  near-term ceiling labels and a sampled pairwise ranking loss. It is a learned
  critic quality harness, not yet the runtime `contrastive_phase` critic.
- Added candidate-action scoring for the learned critic: each row is scored with
  `candidate_action=0` and `candidate_action=1`, producing a learned basin
  action-1 advantage.
- Added bootstrap-head uncertainty diagnostics and an abstention flag based on
  low action margin or high ensemble uncertainty.
- The default quick manifest trains on seeds `1,2` and evaluates on seed `3`
  from `toy24_basin_transition_dataset_quick`.
- `toy24_basin_phase_critic_quality_quick` passed:
  Toy2 eval AUC and pairwise rank accuracy were `0.900936`; Toy4 eval AUC and
  pairwise rank accuracy were `0.976190`.
- With `ensemble_size=5`, `abstention_margin_threshold=0.005`, and
  `uncertainty_threshold=0.05`, Toy2 abstained on `0.1732` of eval samples and
  Toy4 abstained on `0.2006`.
- Prototype observed-score AUC was `0.925595` for Toy2 and `0.976190` for Toy4,
  so this result shows a viable learned signal, not a claim that the learned
  critic already dominates the prototype scaffold.

### 2026-05-14 Runtime Read-Only Learned Diagnostic Slice

Implemented a runtime read-only diagnostic path:

- `BasinCreditConfig` now accepts `learned_diagnostic_enabled`,
  `learned_diagnostic_model_path`,
  `learned_diagnostic_abstention_margin_threshold`, and
  `learned_diagnostic_uncertainty_threshold`.
- Toy2 and Toy4 can load the offline critic bundle and score each live
  post-social phase with candidate actions `0` and `1`.
- Aggregate logs now include learned score means, learned action-1 advantage,
  ensemble uncertainty, abstention rate, and correlation with the prototype
  action-1 advantage when finite.
- Micro logs now include per-agent learned action scores, learned action-1
  advantage, uncertainty, and abstention.
- This path is deliberately read-only: `basin_credit_training_components` still
  comes from the prototype basin-credit path and the existing blend semantics.
  No policy update uses the learned critic yet.

### 2026-05-14 Learned Diagnostic Evidence Summary Slice

Added the first evidence artifact that summarizes read-only learned diagnostics
over actual Toy2/Toy4 trajectories:

- `toy24_basin_learned_diagnostic_quick` runs the existing
  `mixed_objective_basin_escalate_credit_p3_min2_h1` path while loading the
  learned critic bundle only for diagnostics.
- The gate passed with the same ceiling outcomes as the prototype escalation
  run: Toy2 final ceiling `3/3` with mean time-to-ceiling `9.33333`, and Toy4
  final ceiling `3/3` with mean time-to-ceiling `11.6667`.
- The learned diagnostic summary showed:
  - Toy2 prototype-vs-learned sign agreement `0.976333`, non-abstain agreement
    `0.992505`, abstention `0.1808`.
  - Toy4 prototype-vs-learned sign agreement `0.977667`, non-abstain agreement
    `0.999667`, abstention `0.2462`.
- This supports trying learned critic replay only behind an abstention/uncertainty
  gate. It still does not prove that learned replay improves the policy, because
  the learned critic has not yet been used as a training signal.

### Step 1: Dataset Contract

Manifest:

```text
experiments/evidence/toy24_basin_transition_dataset_quick.yaml
```

Exit criteria:

- Sample artifact is written for Toy2 and Toy4.
- Required fields are present.
- Train/eval seed split is explicit.
- Counts by terminal basin are non-empty or the run is marked inconclusive.

### Step 2: Critic Quality

Manifest:

```text
experiments/evidence/toy24_contrastive_basin_critic_quick.yaml
```

Exit criteria:

- Critic training finishes with finite loss.
- Eval AUC or pairwise ranking accuracy is above chance.
- Calibration and positive/negative count diagnostics are written.
- Prototype critic and learned critic score distributions are compared on the
  same held-out samples.

### Step 3: Policy Evidence

Manifest:

```text
experiments/evidence/toy24_uncertainty_replay_gate_quick.yaml
```

Required variants:

- fixed `p2`
- fixed `p3`
- prototype `credit_signal_escalation`
- learned critic with fixed `p2`
- learned critic with fixed `p3`
- learned critic with uncertainty replay gate

Success criteria should preserve the current quick-gate targets:

- Toy2: final ceiling `3/3`, mean time-to-ceiling `< 10`.
- Toy4: final ceiling at least `2/3`, mean time-to-ceiling `< 12`.

A stronger secondary target is Toy4 `3/3` final ceiling with mean
time-to-ceiling near fixed `p3`.

## Claim Discipline

Given the current evidence:

- Use `credit_signal_escalation` as the prototype control.
- Treat learned-gated credit replay as the experimental mechanism, not as a
  proven replacement.
- Do not call the current implementation a contrastive basin critic.
- Do not claim novelty from `training_passes` or hand thresholds.
- Treat the current quick gate as evidence that a variant does not collapse, not
  as proof of generalization.

After the learned critic shows advantage over the prototype control, the
defensible claim becomes narrower and stronger:

> A counterfactual basin-credit mechanism can use a learned phase critic and
> uncertainty-aware replay control to improve collective basin transitions in
> neural ABM social-dilemma toys.

## 2026-05-14 Learned Credit Replay Slice

Implemented the first policy-facing learned-credit source behind the existing
abstention and uncertainty gate.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_replay_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_replay_quick_findings.md
```

Results:

- Gate status: pass.
- Toy2 learned-gated prototype fallback: final ceiling `3/3`, mean TtC `9.33`.
- Toy4 learned-gated prototype fallback: final ceiling `3/3`, mean TtC `11.67`.
- Mean learned-credit usage over aggregate rows:
  - Toy2: `0.8192`, abstention `0.1808`.
  - Toy4: `0.7538`, abstention `0.2462`.
- Zero-fallback learned-gated diagnostic matched the prototype-fallback variant
  on both Toy2 and Toy4 quick evidence.

Claim boundary:

- This slice proves the learned critic is no longer read-only and can drive the
  replay training signal without collapsing Toy2/Toy4 quick evidence.
- It does not prove learned credit is better than prototype credit. The learned
  variants matched the prototype escalation diagnostic on final hits and
  time-to-ceiling.
- Further work should target critic distinctiveness, replay candidate
  selection, or a learned phase representation rather than another threshold
  sweep.

## 2026-05-14 Candidate-Context Critic Slice

Implemented candidate-conditioned phase features for the learned critic:

- `candidate_action_delta`
- `candidate_policy_delta`
- `candidate_phase_action_rate`
- `candidate_phase_policy_rate`
- `candidate_phase_consensus`

The critic-quality manifest is:

```text
experiments/evidence/toy24_basin_phase_critic_candidate_context_quick.yaml
```

The replay manifest is:

```text
experiments/evidence/toy24_basin_learned_credit_candidate_context_replay_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_candidate_context_replay_quick_findings.md
```

Results:

- Critic quality passed:
  - Toy2 AUC `0.9078`, pairwise rank `0.9078`.
  - Toy4 AUC `0.9759`, pairwise rank `0.9759`.
- Replay gate passed:
  - Toy2 candidate-context learned replay: final ceiling `3/3`, mean TtC `9.33`.
  - Toy4 candidate-context learned replay: final ceiling `3/3`, mean TtC `11.67`.
- Candidate-context learned source rate:
  - Toy2: `0.8013`, abstention `0.1987`.
  - Toy4: `0.8665`, abstention `0.1335`.

Claim boundary:

- This is a genuine representation change: action-0/action-1 scoring now
  recomputes candidate phase rates instead of only flipping `candidate_action`.
- It produced more distinct learned/prototype behavior on Toy2 without policy
  collapse.
- It still did not outperform prototype escalation on quick evidence.
- The next mechanism should therefore target replay selection or critic target
  definition, not more threshold adjustment.

## 2026-05-14 Learned Replay Selection Slice

Implemented learned/prototype relation as a replay candidate selector. The
learned credit path now supports:

- `all`
- `confident`
- `confident_agreement`
- `confident_disagreement`

The selection policy changes which active basin-training agents are eligible
for replay. This is a replay-policy mechanism change, not just a scalar replay
threshold.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_replay_selection_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_replay_selection_quick_findings.md
```

Result:

- Gate status: fail.
- All-replay candidate-context learned credit still matched prototype
  escalation:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`.
  - Toy4 final ceiling `3/3`, mean TtC `11.67`.
- Main `confident_agreement` replay failed:
  - Toy2 final ceiling `0/3`, mean final payoff `2.328`.
  - Toy4 final ceiling `0/3`, mean final payoff `0.298`.
- Replay selected rate shows the failure mode:
  - Toy2 `confident_agreement`: `0.0000`.
  - Toy4 `confident_agreement`: `0.0039`.
  - Toy4 `confident_disagreement`: `0.0000`.

Claim boundary:

- The implementation is useful because it exposes replay starvation as the
  concrete failure mode of naive learned/prototype relation selection.
- It is not a successful replacement for all-replay learned credit.
- The next replay-selection mechanism needs a curriculum, a minimum replay
  floor, or a different critic target before agreement/disagreement selection
  can be treated as a main policy.

## 2026-05-14 Learned Replay Floor Slice

Implemented a starvation-safe replay selector:

```text
learned_credit_replay_min_selected_rate
learned_credit_replay_floor_source
```

The tested floor source was `prototype_abs`: if the learned/prototype agreement
selector chooses too few active basin-training agents, fill the replay set with
the highest absolute prototype basin-advantage candidates.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_replay_floor_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_replay_floor_quick_findings.md
```

Result:

- Gate status: fail.
- `confident_agreement` without a floor still failed:
  - Toy2 final ceiling `0/3`, replay selected `0.0000`.
  - Toy4 final ceiling `0/3`, replay selected `0.0039`.
- `floor50` restored final ceiling outcomes but missed speed criteria:
  - Toy2 final ceiling `3/3`, mean TtC `16.00`, replay selected `0.5000`.
  - Toy4 final ceiling `3/3`, mean TtC `14.67`, replay selected `0.8912`.
- Prototype/all-replay controls remained faster:
  - Toy2 mean TtC `9.33`.
  - Toy4 mean TtC `11.67`.

Claim boundary:

- The replay floor validates the starvation diagnosis and removes final-outcome
  collapse.
- A static selected-rate floor is not enough to replace all-replay learned
  credit because it delays time-to-ceiling.
- The next structural selector should be temporal: start broad, then narrow
  once basin motion or critic confidence is established. A static floor should
  remain diagnostic, not the main mechanism.

## 2026-05-14 Learned Replay Curriculum Slice

Implemented an epoch-based replay floor curriculum:

```text
learned_credit_replay_floor_schedule: linear_decay
learned_credit_replay_floor_start_rate
learned_credit_replay_floor_decay_epochs
```

This keeps learned/prototype confident agreement as the replay selector, but it
starts with a broad replay floor and decays toward the configured minimum floor.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_replay_curriculum_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_replay_curriculum_quick_findings.md
```

Result:

- Gate status: pass.
- Main `floor50_d30` curriculum:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`.
  - Toy4 final ceiling `3/3`, mean TtC `11.67`.
- Static `floor50` remained slower:
  - Toy2 mean TtC `16.00`.
  - Toy4 mean TtC `14.67`.
- No-floor `confident_agreement` still collapsed:
  - Toy2 final ceiling `0/3`.
  - Toy4 final ceiling `0/3`.

Claim boundary:

- This is the first replay-selection variant that passes the Toy2/Toy4 quick
  gate after introducing learned/prototype relation selection.
- It matches prototype/all-replay quick performance; it does not outperform it.
- Toy4 selected almost all active candidates on average, so the result should
  be interpreted as successful broad-start curriculum control, not as proof
  that sparse learned replay is sufficient.
- The next selector should become state-dependent: narrow replay after basin
  motion, agreement coverage, or critic confidence crosses a logged threshold.

## 2026-05-14 Learned Soft Replay Slice

Implemented soft replay weighting:

```text
learned_credit_replay_mode: soft_attention
learned_credit_replay_soft_min_weight
learned_credit_replay_soft_disagreement_weight
```

This keeps the learned/prototype relation signal, but changes replay control
from a hard candidate gate to a per-agent loss weight. Candidates remain
trainable when `soft_min_weight > 0`, while margin, uncertainty, and
agreement/disagreement scale the replay strength.

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_soft_replay_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_soft_replay_quick_findings.md
```

Result:

- Gate status: pass.
- Main `soft_min50` replay:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`, mean replay weight `0.5416`.
  - Toy4 final ceiling `3/3`, mean TtC `11.67`, mean replay weight `0.8006`.
- No-floor hard `confident_agreement` still collapsed:
  - Toy2 final ceiling `0/3`, replay selected `0.0000`.
  - Toy4 final ceiling `0/3`, replay selected `0.0039`.
- `soft_min25` stayed viable in Toy2 but was too slow for Toy4:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`.
  - Toy4 final ceiling `3/3`, mean TtC `12.33`.

Claim boundary:

- Soft replay supports the structural diagnosis that the failure was hard-gate
  starvation, not the learned credit sign itself.
- This is closer to credit-attention replay than the previous fixed floor,
  because the update strength is continuous and per agent.
- It is not a Transformer-style learned attention module; the weighting formula
  still uses fixed coefficients around critic margin, uncertainty, and
  prototype/learned agreement.
- The next structural step is to learn or calibrate the replay weight scorer
  over richer basin-phase state instead of manually setting the coefficient
  policy.

## 2026-05-14 Learned Replay Weight Scorer Slice

Implemented the first frozen learned replay-weight scorer:

```text
learned_credit_replay_mode: learned_weight
learned_credit_replay_weight_model_path
```

The scorer is trained offline from basin transition samples and a
candidate-context learned basin critic. At runtime it predicts a continuous
per-agent replay loss weight from critic margin, uncertainty,
prototype/learned agreement, candidate scores, and phase-state features.

Training artifact:

```text
experiments/results/basin_critic/toy24_basin_replay_weight_scorer_q99_quick_summary.md
```

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_weight_scorer_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_weight_scorer_quick_findings.md
```

Result:

- Gate status: pass.
- Main learned replay-weight scorer:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`, mean replay weight `0.7636`.
  - Toy4 final ceiling `3/3`, mean TtC `11.67`, mean replay weight `0.9745`.
- Offline scorer quality:
  - Toy2 eval MSE `0.000584`, eval correlation `0.8436`.
  - Toy4 eval MSE `0.009975`, eval correlation `0.9395`.

Claim boundary:

- This is a structural change beyond fixed soft replay because the runtime
  replay weight now comes from a frozen learned module.
- It is still not an end-to-end attention mechanism: the scorer target is
  derived from offline transition-signal magnitude, not from direct policy
  improvement gradients.
- Toy4 remains close to all-replay, so this result should be read as
  "learned scorer can preserve the quick gate", not as proof that selective
  low-strength replay is sufficient.
- The next target should predict held-out basin motion or counterfactual policy
  improvement rather than imitate transition-signal magnitude.

## 2026-05-14 Future-Motion Replay Supervision Slice

Implemented a second scorer supervision target:

```text
target_mode: future_basin_motion
future_horizon: 5
target_column: future_basin_score_delta
```

The transition sample artifact now carries reusable forward labels:

```text
future_basin_horizon
future_mean_payoff
future_basin_score_delta
future_ceiling_reached
future_epochs_to_ceiling
future_basin_motion_positive
```

Training artifact:

```text
experiments/results/basin_critic/toy24_basin_replay_weight_scorer_future_motion_h5_q90_quick_summary.md
```

Manifest:

```text
experiments/evidence/toy24_basin_learned_credit_future_motion_weight_scorer_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_learned_credit_future_motion_weight_scorer_quick_findings.md
```

Result:

- Gate status: fail.
- The offline target is learnable:
  - Toy2 eval MSE `0.000377`, eval correlation `0.9901`.
  - Toy4 eval MSE `0.001043`, eval correlation `0.9716`.
- Runtime effect:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`, mean replay weight `0.514`.
  - Toy4 final ceiling `3/3`, mean TtC `12.00`, mean replay weight `0.514`.

Claim boundary:

- This is a structural supervision change because replay weight is trained from
  observed forward basin motion rather than current transition-signal
  magnitude.
- The h5/q90 target is too conservative as a runtime replacement for the
  magnitude-supervised scorer: Toy4 misses the strict quick gate by landing at
  mean TtC `12.00`.
- Keep the future labels as infrastructure, but do not promote this scorer as
  the main mechanism yet.
- The next structural target should add a pairwise or directional constraint:
  prefer replay candidates whose credit direction predicts faster basin entry,
  not just larger forward payoff movement.

## 2026-05-15 Direction-Pressure Split Slice

Implemented a pressure target that separates the roles of the learned modules:

```text
direction = candidate-context learned basin critic
pressure = learned replay-weight scorer
```

The new scorer supervision mode is:

```text
target_mode: intervention_pressure
target = max(
  scaled_abs(training_effective_advantage),
  scaled_positive(future_basin_score_delta)
)
```

Training artifact:

```text
experiments/results/basin_critic/toy24_basin_replay_pressure_scorer_h5_q99_quick_summary.md
```

Manifest:

```text
experiments/evidence/toy24_basin_direction_pressure_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_direction_pressure_quick_findings.md
```

Result:

- Gate status: pass.
- Main direction-pressure scorer:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`, mean replay weight `0.749`.
  - Toy4 final ceiling `3/3`, mean TtC `11.67`, mean replay weight `0.983`.
- Offline pressure fit:
  - Toy2 eval MSE `0.000462`, eval correlation `0.8851`.
  - Toy4 eval MSE `0.010451`, eval correlation `-0.1595`.

Claim boundary:

- This slice supports the direction/pressure decomposition as a runtime
  framing: the future-motion scorer failed Toy4 by under-pressuring, while the
  pressure scorer preserves enough Toy4 intervention strength.
- It is not yet proof of a fine-grained Toy4 pressure curriculum. Toy4's target
  is nearly saturated, so the pressure scorer mostly learns to stay close to
  all-replay.
- The next structural step is to make direction supervision explicitly
  pairwise/counterfactual: train the direction critic so that candidate action
  differences predict faster basin entry, then keep pressure as a separate
  domain-sensitivity module.

## 2026-05-15 Pairwise Direction-Pressure Slice

Implemented the next direction slice:

```text
direction = pairwise candidate-action basin critic
pressure = learned intervention-pressure replay scorer
```

The new critic label mode is `prototype_direction`. It duplicates each eligible
transition sample into candidate action `0` and candidate action `1`, then marks
the action aligned with the prototype basin action-1 advantage as the positive
candidate. Future basin-motion labels are used only as a relevance gate. This
is therefore a pseudo-pairwise label distilled from the prototype critic, not a
true rollout counterfactual label.

Training artifact:

```text
experiments/results/basin_critic/toy24_basin_phase_critic_pairwise_direction_quick_summary.md
```

Manifest:

```text
experiments/evidence/toy24_basin_pairwise_direction_pressure_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_pairwise_direction_pressure_quick_findings.md
```

Result:

- Gate status: pass.
- Offline direction critic:
  - Toy2 eval AUC and pairwise rank accuracy were `0.9908`.
  - Toy4 eval AUC and pairwise rank accuracy were `0.9576`.
- Main pairwise direction-pressure scorer:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`, mean replay weight `0.500`.
  - Toy4 final ceiling `3/3`, mean TtC `11.33`, mean replay weight `0.999`.
- Compared with the previous candidate-context direction-pressure diagnostic:
  - Toy2 remained tied at mean TtC `9.33`.
  - Toy4 improved slightly from mean TtC `11.67` to `11.33`.

Claim boundary:

- This slice validates explicit candidate-action direction as a viable runtime
  interface for the learned basin module.
- It does not yet validate a true counterfactual basin critic, because the
  positive action labels are still prototype-distilled.
- It does not yet validate replay-budget efficiency in Toy4, because the
  pressure module remains near all-replay there.
- The next structural step is to produce held-out counterfactual rollout or
  ablation labels for direction, then ask whether pressure can be reduced
  without losing ceiling arrival.

## 2026-05-18 Future-Outcome Direction-Pressure Slice

Implemented a direction label that no longer uses the prototype basin-advantage
sign as the direct pairwise target.

```text
direction = future-outcome candidate-action basin critic
pressure = learned intervention-pressure replay scorer
```

The new critic label mode is `future_outcome_direction`:

- If the observed trajectory improves future basin payoff or reaches/maintains
  the ceiling within the horizon, the observed action is the positive
  candidate.
- If the observed trajectory worsens, the counterfactual action is the positive
  candidate.

Training artifact:

```text
experiments/results/basin_critic/toy24_basin_phase_critic_future_outcome_direction_quick_summary.md
```

Manifest:

```text
experiments/evidence/toy24_basin_future_outcome_direction_pressure_quick.yaml
```

Findings:

```text
experiments/results/nabm_effect_matrix/toy24_basin_future_outcome_direction_pressure_quick_findings.md
```

Result:

- Gate status: pass.
- Offline direction critic:
  - Toy2 eval AUC and pairwise rank accuracy were `0.9586`.
  - Toy4 eval AUC and pairwise rank accuracy were `0.9584`.
- Main future-outcome direction-pressure scorer:
  - Toy2 final ceiling `3/3`, mean TtC `9.33`, mean replay weight `0.500`.
  - Toy4 final ceiling `3/3`, mean TtC `11.33`, mean replay weight `0.999`.
- It matched the prototype-pairwise direction-pressure diagnostic on mean TtC
  in both Toy2 and Toy4.

Claim boundary:

- This slice shows that the runtime direction-pressure interface does not need
  to train directly on the prototype action-1 advantage sign.
- It is still not a true counterfactual rollout critic. The label is based on
  observed future basin motion, so it cannot prove what the alternative action
  would have caused.
- The next structural step remains held-out rollout or ablation labels for
  candidate actions. That is the step that can turn the direction critic from
  observed-outcome supervision into a causal counterfactual module.

## References

- Foerster et al., "Counterfactual Multi-Agent Policy Gradients" (COMA):
  <https://arxiv.org/abs/1705.08926>
- Tumer and Wolpert, "Collective Intelligence and Braess' Paradox":
  <https://archive.aaai.org/Papers/AAAI/2002/AAAI02-051.pdf>
- van den Oord et al., "Representation Learning with Contrastive Predictive
  Coding": <https://arxiv.org/abs/1807.03748>
- Barreto et al., "Successor Features for Transfer in Reinforcement Learning":
  <https://arxiv.org/abs/1606.05312>
- Osband et al., "Deep Exploration via Bootstrapped DQN":
  <https://arxiv.org/abs/1602.04621>
- Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning":
  <https://arxiv.org/abs/2006.04779>
- Shao et al., "Counterfactual Conservative Q Learning for Offline Multi-agent
  Reinforcement Learning": <https://arxiv.org/abs/2309.12696>
- Xie et al., "Learning Latent Representations to Influence Multi-Agent
  Interaction": <https://arxiv.org/abs/2011.06619>
