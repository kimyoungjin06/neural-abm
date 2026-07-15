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
researchers while never entering fit.

Four environments are compared over 100 seed-paired replicates (common random
numbers, percentile intervals over paired deltas):

- Interdisciplinary seed grants raise the productive pivot rate by +0.190
  [95% CI 0.112, 0.263]; a zero-intensity placebo collapses to +0.004
  [-0.017, 0.017].
- Hot-field hype delivers a statistically indistinguishable productive gain
  (+0.193 [0.125, 0.263]) at 2.4x the pivots and roughly ten times the failed
  pivots (36% versus 4% of the population); hype changes who pivots, not just
  how many.
- A social-mixing sweep shows conformity pressure suppressing both
  spontaneous pivots and the grant effect while consolidating hype-driven
  mass pivoting — an emergent pattern, not a designed-in one.

The methodological point for the manuscript is the replication machinery:
`run_replicated_bounded_scalar_scenarios` pairs replicate seeds across
scenarios, so scenario deltas are paired and every reported effect carries a
distributional interval. The full audit trail (aggregate and per-agent micro
rows) ships in the result artifact.

## 7.3 Study 2: Fixed Rule Versus Learning Agents

Study 2 replaces the fixed decision rule with a learnable policy on the
torch-backed `NABMUnit` lifecycle, holding everything else fixed: the same
population sampler, network builder, typed bounded-scalar channel, commit
adapter, seeds, and audit logging. Three arms share one sigmoid policy over
nine observable features and differ only in the learning rule:

- `frozen`: no updates (the Study 1 rule in policy form) — the classical
  control arm;
- `imitative`: vicarious gradient updates from every observed neighbor pivot
  outcome;
- `cautionary`: the same update restricted to observed failures
  (negativity-biased social learning).

Pivots are absorbing events with publicly visible outcomes, so outcome
learning and readiness contagion compete on the same network. Over 30
seed-paired replicates:

- Imitative learning triggers an information cascade. Because only pivoters
  reveal outcomes and early pivoters are the best-positioned, one-sided
  success evidence raises pivot propensity population-wide: failed pivot
  rates rise by +0.30 [0.07, 0.44] even at baseline and +0.29 [0.17, 0.38]
  under seed grants, and supportive environments are hit hardest because
  support manufactures the early success stories that feed the cascade.
- Cautionary learning yields targeted hype immunity: failed pivots under
  hype fall by -0.137 [-0.200, -0.087] relative to the frozen control while
  the baseline environment is left exactly unchanged (no failures to learn
  from). The mechanism is auditable in the learned parameters: the mean
  attention weight falls from 1.00 to 0.81 under hype and moves nowhere
  else.
- Caution has a price: under seed grants, cautionary learners also forgo
  productive pivots (-0.073 [-0.127, -0.031]).

## 7.4 What This Section Claims

Three statements are supported, each with its boundary:

1. The reusable surfaces carry a complete replicated counterfactual study
   without torch (Study 1), including paired uncertainty intervals and a
   passing placebo control. Limitation: the domain model is stylized and
   theory-parameterized, not calibrated to bibliometric data.
2. The lifecycle supports a controlled fixed-versus-learning comparison in
   which every observed divergence is attributable to the learning rule
   (Study 2). Limitation: the learning modes are postulated update rules
   drawn from the social-learning literature, not estimated from data.
3. The findings unique to Study 2 — endogenous cascades from imitation,
   targeted immunity from cautionary updates, and the learned-weight
   trajectory as audit evidence — are statements about endogenous rule
   change, which a fixed-rule configuration of the same model cannot
   express. This is an expressiveness claim about the modeling language,
   not a performance claim.

## 7.5 Why the Policy Class Is Deliberately Small

The Study 2 policy is a sigmoid over nine named features — ten parameters
per agent. This is a methodological choice, not a technical ceiling, and it
rests on four grounds:

1. **Information budget.** Each agent learns from the observed pivot
   outcomes of four to five network neighbors — single-digit samples per
   agent over the whole run. Ten parameters already sit at the edge of what
   that observational stream identifies; deeper networks in this regime fit
   noise, not mechanism.
2. **Audit requirement.** The learned-attention-weight trajectory can serve
   as primary evidence only because every parameter has a name. Added
   capacity that the audit cannot follow would spend the framework's central
   guarantee to buy expressiveness the question does not need.
3. **Attribution discipline.** In a fixed-versus-learning comparison, model
   capacity is a confound: with a large policy class, divergence between
   arms could reflect capacity interacting with the dynamics rather than the
   learning rule. Minimal capacity keeps the controlled comparison clean.
4. **Interaction-first complexity.** The system-level nonlinearity lives in
   threshold transitions, absorbing events, and network contagion. Both
   headline results (cascades, targeted immunity) emerge from linear
   policies coupled through nonlinear interaction structure — consistent
   with the ABM tradition of locating complexity in interaction rather than
   in agent internals, and with the cue-weighting view of human judgment.

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
3. `paper/figures/pivot_learning_attention_weights.png` (mechanism
   auditability; pairs naturally with the cascade figure).

Wording guardrails, mirroring the claim matrix:

- Do not describe Study 2 as the neural policy "beating" the frozen rule;
  the frozen arm is a control, and cautionary learning loses productive
  pivots under seed grants.
- Keep the placebo result visible; it is the validity check that licenses
  the scenario-comparison machinery.
- State that outcomes are immediately and perfectly observable in Study 2;
  delayed or noisy disclosure is the natural robustness extension.
