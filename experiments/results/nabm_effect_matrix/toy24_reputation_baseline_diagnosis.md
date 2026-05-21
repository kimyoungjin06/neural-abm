# Toy2/Toy4 Reputation Baseline Diagnosis

This diagnosis follows `nabm_effect_matrix_quick_findings.md`, where Toy2 and
Toy4 neural output averaging lost to reputation imitation.

Artifacts:

- Quick matrix pairwise effects:
  `nabm_effect_matrix_quick_pairwise_effects.csv`.
- Long-horizon checkpoint diagnostic:
  `toy24_epoch50_diagnostic_checkpoints.csv`.
- Neural reputation-observation diagnostic:
  `toy24_reputation_observation_diagnostic.csv`.

The long-horizon and reputation-observation diagnostics were run into `/tmp`
run directories and summarized here to preserve the decision-relevant data.

## Finding

The Toy2/Toy4 gap is not primarily a short-horizon artifact and not fixed by
giving the neural policy reputation observations.

Reputation imitation is acting as a strong deterministic norm/convention
baseline:

- `reputation.noise = 0.0`.
- `revision_rate = 1.0`.
- revised agents copy the action of the highest-reputation spatial peer.
- initial reputation is tied to binary action history, so early cooperators or
  contributors become copied and then reinforce reputation through future
  actions.

Neural output averaging is optimizing the material local policy path. In Toy2
harsh prisoner's dilemma and Toy4 public goods, the local material incentive
pushes away from cooperation/contribution. Output averaging distills peer
outputs, but it does not add a norm reward or reputation-preserving objective.

## Evidence

### Quick Matrix

Toy2:

- Neural beats Fermi imitation: `+0.3883`.
- Neural beats RD well-mixed: `+0.3551`.
- Neural loses to reputation imitation: `-0.9833`.

Toy4:

- Neural beats imitation: `+0.2280`.
- Neural loses to reputation imitation: `-0.3720`.

### 50 Epoch Diagnostic

Toy2 reputation imitation reaches full cooperation by epoch 3 to 5 and remains
there:

- seed 1: epoch 50 payoff `3.0`, action rate `1.0`.
- seed 2: epoch 50 payoff `3.0`, action rate `1.0`.
- seed 3: epoch 50 payoff `3.0`, action rate `1.0`.

Toy2 neural output averaging collapses toward defection:

- seed 1: epoch 50 payoff `1.0`, action rate `0.0`.
- seed 2: epoch 50 payoff `1.03`, action rate `0.01`.
- seed 3: epoch 50 payoff `1.15`, action rate `0.05`.

Toy4 reputation imitation reaches full contribution by epoch 3 to 5 and remains
there:

- seed 1: epoch 50 payoff `0.6`, action rate `1.0`.
- seed 2: epoch 50 payoff `0.6`, action rate `1.0`.
- seed 3: epoch 50 payoff `0.6`, action rate `1.0`.

Toy4 neural output averaging collapses toward non-contribution:

- seed 1: epoch 50 payoff `0.0`, action rate `0.0`.
- seed 2: epoch 50 payoff `0.0`, action rate `0.0`.
- seed 3: epoch 50 payoff `0.024`, action rate `0.04`.

### Reputation Observation Diagnostic

Adding `self_neighbor_mean` reputation observations to the neural policy did
not change the conclusion:

- Toy2 reputation-observation neural mean final payoff: about `2.015`.
- Toy4 reputation-observation neural mean final payoff: `0.228`.

This is close to the quick neural output-average behavior and still far below
the reputation-imitation baseline.

## Interpretation

This is an objective mismatch, not just a missing input feature.

For Toy2, cooperation is socially good in the all-cooperate state, but
defection is individually attractive under the harsh PD payoff table. The neural
policy's local material update drifts toward defection over time.

For Toy4, contribution is individually costly under the current public-goods
setting. The group benefit is not enough, from a single agent's immediate
material perspective, to make contribution the learned dominant action.

Reputation imitation does not solve these games by learning the same objective.
It imposes a convention-following heuristic: copy high-reputation cooperative
or contributing peers. Under zero noise and full revision, that heuristic is
strong enough to lock in full cooperation/contribution.

## Claim Impact

Do not claim that current Toy2/Toy4 neural output averaging dominates all
reference policies.

A defensible claim is narrower:

- Toy2 neural output averaging beats RD and Fermi references in this quick
  matrix, but not reputation imitation.
- Toy4 neural output averaging beats imitation, but not reputation imitation.
- Reputation imitation should be treated as a norm-enforcing reference, not a
  neutral weak baseline.

## Recommended Next Mechanism Test

The next Toy2/Toy4 work should be a structural mechanism test, not an
alpha/threshold sweep.

Candidate mechanism:

- add an explicit reputation/norm-aware neural objective shared by Toy2 and
  Toy4;
- keep the same matrix comparison;
- measure whether the neural policy can preserve cooperation/contribution
  against reputation imitation without hand-picking alpha or thresholds.

Minimum instrumentation before implementation:

- log the action-1 material advantage or payoff-gradient proxy by epoch;
- log social distillation delta, not just social loss;
- log reputation-conditioned action probability buckets;
- compare material-payoff objective versus reputation/norm-shaped objective
  under the same seeds.
