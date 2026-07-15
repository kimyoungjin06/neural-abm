# Section 7 Draft: Research Application — Replicated Scenario Studies

Status: draft prose candidate.

Source case study: `docs/case-studies/researcher-pivot/README.md`.
Source code: `examples/research_pivot_study.py`,
`examples/research_pivot_learning_study.py`,
`src/neural_abm/scenario_lite.py`.
Result artifacts: `docs/case-studies/researcher-pivot/data/study_results.json`,
`docs/case-studies/researcher-pivot/data/learning_study_results.json`.

## 7.1 Purpose

Sections 4-6 validate the unit contract on capability-first toy families. This
section demonstrates the complementary claim: that the same contract carries a
researcher-authored question end to end — baseline and counterfactual
scenarios, seed-paired replication, distributional outcomes, audit rows, and a
controlled comparison between a fixed decision rule and learning agents. The
domain is a stylized science-of-science question: under which scientific
environments do researcher field pivots become productive rather than merely
frequent?

The evidence in this section is an *expressiveness and discipline* claim, in
line with the manuscript framing rules. It is not a claim that neural agents
outperform classical rules in this domain; the fixed-rule arm is the control,
not a defeated baseline.

## 7.2 Study 1: Environment Comparison Under a Fixed Rule

Study 1 runs entirely on the torch-free `api_lite` surface. A population of
120 researchers with stage-conditioned structural attributes decides whether
to pivot fields. The desire to pivot (pressure) and the capacity to pivot
productively (fit) are deliberately separated, and hype susceptibility is
heterogeneous: attention signals land hardest on resource-insecure
researchers. Attention itself does not enter fit, but the bundled hype
scenario also changes field opportunity, resources, and reputation risk.

Four environments are compared over 100 paired replicates. Component-keyed
streams hold the sampled population, base network, and agent-step shocks
common while keeping the topology intervention separate. Bracketed intervals
are 95% normal-approximation CIs for the paired mean effect; the artifact also
stores empirical replicate intervals.

- Interdisciplinary seed grants raise the productive pivot rate by +0.191
  [0.181, 0.202]. A true-null arm that zeros every grant component, including
  bridge topology, returns exactly 0.000 [0.000, 0.000].
- Hot-field hype produces a similar mean gain in this run (+0.187 [0.180,
  0.194]) at 2.3x the pivots and 8.6x the failures. The direct paired
  hype-minus-grant contrast is -0.004 [-0.012, 0.004]; no equivalence margin
  was specified, so this is not an equivalence claim.
- A social-mixing sweep shows conformity pressure suppressing both
  spontaneous pivots and the grant effect while consolidating hype-driven
  mass pivoting within the specified graph/filter/threshold design.

The methodological point for the manuscript is the replication machinery:
`run_replicated_bounded_scalar_scenarios` reports paired mean effects and keeps
mean-effect CIs distinct from replicate-level empirical intervals. The Study 1
artifact ships every primary replicate outcome, aggregate summaries, full
configuration and provenance, plus a labeled three-agent micro sample from the
first replicate of each scenario. It does not claim to contain every agent row.

## 7.3 Study 2: Fixed Rule Versus Learning Agents

Study 2 places a learnable policy on the torch-backed `NABMUnit` lifecycle.
Within Study 2, its three arms hold the population sampler, network builder,
typed bounded-scalar channel, commit adapter, seeds, and audit logging fixed.
They share one sigmoid policy over nine observable features and differ only in
the learning rule:

- `frozen`: no updates; a monotone sigmoid reparameterization initialized
  from the Study 1 coefficients;
- `imitative`: vicarious gradient updates from every observed neighbor pivot
  outcome, excluding the focal agent's own outcome;
- `cautionary`: the same neighbor-only update restricted to observed failures,
  treated here as a postulated failure-only rule.

Pivots are absorbing events with publicly visible outcomes, so outcome
learning and readiness contagion compete on the same network. Bracketed Study
2 intervals are deterministic 10,000-resample percentile-bootstrap CIs for the
paired mean delta. Over 30 seed-paired replicates:

- Imitative learning creates a self-reinforcing outcome-learning loop at
  baseline and under grants: failed pivots rise by +0.281 [0.235, 0.322] and
  +0.289 [0.268, 0.308], respectively. Under already-high-pivot hype, the
  failed-rate difference is -0.009 [-0.024, 0.007]. This is not the canonical
  informational-cascade decision model.
- Failure-only learning reduces hype failures by -0.144 [-0.159, -0.129],
  while productive pivots also fall by -0.050 [-0.056, -0.044]. The attention
  coefficient falls from 1.00 to 0.81, but every nonzero supported feature and
  the bias can update; the trace is not single-coefficient causal evidence.
- The same rule reduces grant failures by -0.061 [-0.073, -0.052] and grant
  productive pivots by -0.069 [-0.081, -0.059]. Baseline also changes slightly
  (productive-pivot delta -0.004 [-0.007, -0.001]).

The learning artifact preserves raw replicate runs, the complete config, and
mean/dispersion trajectories for all nine weights and the bias. This supports
inspection of multivariate updates, not identification of one coefficient as
the causal mechanism.

## 7.4 What This Section Claims

Three statements are supported, each with its boundary:

1. The reusable surfaces carry a replicated counterfactual study without
   torch (Study 1), including paired mean-effect intervals, separate empirical
   replicate intervals, and an exact true-null control. Limitation: the domain
   model is hand-set and not calibrated to bibliometric data.
2. The lifecycle supports a controlled fixed-versus-learning comparison in
   which every observed divergence is attributable to the learning rule
   within Study 2. Limitation: the frozen policy is a sigmoid
   reparameterization of Study 1 coefficients, and the learning modes are
   postulated rather than estimated from data.
3. Study 2 demonstrates endogenous coefficient change and preserves its full
   named-parameter traces. A static-rule configuration of this model cannot
   express those updates without adding an adaptive rule. This is a repository
   lifecycle claim, not a claim of performance, causal identification, or
   exclusivity relative to other adaptive ABM frameworks.

## 7.5 Why the Policy Class Is Deliberately Small

The Study 2 policy is a sigmoid over nine named features — ten parameters
per agent. This is a methodological choice, not a technical ceiling, and it
rests on four grounds:

1. **Information budget.** Each agent receives a sparse stream of observed
   neighbor outcomes. This motivates a small policy, but does not establish
   that all ten parameters are statistically identified.
2. **Audit requirement.** Every parameter has a name, and the artifact stores
   all weight and bias trajectories. The resulting traces reveal multivariate
   movement; they do not make any one coefficient causal evidence.
3. **Attribution discipline.** In a fixed-versus-learning comparison, model
   capacity is a confound: with a large policy class, divergence between
   arms could reflect capacity interacting with the dynamics rather than the
   learning rule. Minimal capacity keeps the controlled comparison clean.
4. **Interaction-level complexity.** The system-level nonlinearity lives in
   threshold transitions, absorbing events, and network exchange. The simple
   policy keeps that interaction structure visible without implying that a
   larger policy would necessarily be inferior.

The lifecycle accepts arbitrary torch modules, and state-dict and
distillation commit adapters already exist, so capacity is an escalation
ladder rather than a cap: increase it only when the hypothesis itself
concerns representation learning, the observational stream is rich enough to
identify the added parameters, and an audit plan exists for them.

## 7.6 Manuscript Insertion Notes

Place this section after the calibration/analysis section and before the
discussion. Figure candidates, in order of value:

1. `paper/figures/pivot_learning_failed_trajectories.png` (Study 2 cumulative
   failure curves; the three-arm comparison is the section's core evidence).
2. `paper/figures/pivot_composition.png` (Study 1 hype paradox composition).
3. `paper/figures/pivot_learning_attention_weights.png` (full parameter
   audit, including the attention trajectory; pairs with the failure curves).

Wording guardrails, mirroring the claim matrix:

- Do not describe Study 2 as the neural policy "beating" the frozen rule;
  the frozen arm is a control, and failure-only learning loses productive
  pivots under seed grants.
- Keep the exact true-null result visible as a pipeline invariant, without
  treating it as validation of the hand-set domain model.
- Do not use "statistically indistinguishable," "textbook information
  cascade," or "targeted immunity" without additional equivalence, canonical
  cascade, and coefficient-ablation evidence.
- State that outcomes are immediately and perfectly observable in Study 2;
  delayed or noisy disclosure is the natural robustness extension.

## 7.7 Conceptual Reference Boundary

The push-pull terminology is a conceptual borrowing from Lee's
["A Theory of Migration"](https://doi.org/10.2307/2060063), not a calibration
of researcher mobility. The canonical informational-cascade definition in
Bikhchandani, Hirshleifer, and Welch's
[1992 model](https://doi.org/10.1086/261849) is narrower than the gradient
feedback implemented here; the manuscript therefore uses
"self-reinforcing outcome-learning loop" for the observed dynamics.
