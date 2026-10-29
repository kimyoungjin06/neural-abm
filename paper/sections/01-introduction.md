# Section 1 Draft: Introduction

Status: draft prose candidate.

**Sources.**
- The central claim and the Primary Claim Path of
  [claim-matrix.md](../claim-matrix.md), with policy-ladder rows P1 to P11.
- The related-work contrast in
  [related-work/llm-abm-frameworks-v1.md](../related-work/llm-abm-frameworks-v1.md).
- The hash-audited consolidation ledgers v1 to v9:
  `docs/research/pivot-policy-ladder-consolidation-v9.md` extends the others.

## 1.1 The Modeling Choice

Agent-based models earn trust through explicit rules. Every state variable,
transition, and intervention point can be named, and a seeded run can be
replayed. The same explicitness is a limit. A hand-written rule does not
change with what agents experience during a run unless the modeler writes
that change in.

A recent alternative puts a large language model behind each agent. In the
LLM-based frameworks reviewed in Section 2:
- an LLM call decides the action, usually once per agent per step;
- behavior changes through memory and prompt context, not through learned
  policies;
- controlled comparison with rule agents in the same environment is rare;
- seeds are seldom documented, and validation is mostly qualitative.

Between these poles sits a choice that modelers make implicitly. Should an
agent follow a fixed rule, adapt from the outcomes it observes, or start from
external knowledge such as a language model's judgment? This paper is about
making that choice explicit and testable.

## 1.2 This Paper

We present Neural ABM (NABM). Its central claim is structural:

> NABM is a reusable framework that separates policy learning from social
> interaction, so fixed policies, adaptive policies, and external priors can
> be compared and traced within one simulation lifecycle.

Three design commitments make the claim testable.

- **A fixed rule is an exact special case.** Each learned or prior-informed
  arm is read only after the design reproduces its rule-based predecessor
  exactly. The rule ABM is therefore a setting of the same lifecycle, not a
  separate model.
- **A policy is a table, indexed by who writes it and when.** A hand rule, a
  policy replayed from donor runs, an LLM judgment elicited once, and a
  policy that keeps learning from the run's own outcomes all fill the same
  policy slot. They differ only in who writes the table and when. The
  critique that a neural ABM is a rule table computed in advance becomes a
  comparison inside one design rather than a debate.
- **Comparisons are registered and replayable.** Seeds are paired. Designs,
  measures, and decision rules are frozen before any result is read, and an
  expectation that fails is recorded as rejected. Reruns are byte identical
  on one host, and each claim cites a hash-bound result.

NABM is neither a cheaper substitute for LLM-agent simulation nor a
replacement for classical ABM. Its agents are small learned policies, with a
fixed rule or a once-elicited LLM judgment as their prior. Their decisions in
the studies below are binary, and they produce no language. The intended
position is an auditable middle: as much agent adaptivity as still supports
paired counterfactual control, named-parameter audit, and deterministic
replication.

## 1.3 Contributions

1. **A reusable unit contract (Section 3).** The NABM Unit owns lifecycle
   order, typed social exchange, backend dispatch, and diagnostics. Domain
   adapters keep rewards, thresholds, readiness meaning, and evidence gates.
2. **Toy evidence under an explicit evidence discipline (Sections 4 to 6).**
   Capability-first toy families exercise the unit and its social-mixing
   taxonomy. A claim matrix ties every usable claim to its code path,
   artifact, and limitation.
3. **A research application (Section 7).** A researcher-authored
   science-of-science question runs end to end: replicated counterfactual
   scenarios on the torch-free surface, and a controlled comparison of a
   fixed rule with learning agents on the NABM lifecycle.
4. **A policy ladder in two domains (Section 8).** Fixed rules, tables
   written before the run, adaptive learners, and external LLM priors are
   compared in one lifecycle, first for researcher field pivots and then for
   information spread with delayed fact-checks. Each result is a conditional
   case of the central claim:
   - In stationary designs, a table written before the run suffices. The
     adaptive policy wins once about a quarter of runs carry information
     that the donor runs lacked.
   - Pooled learners cost 1.9 to 6.7 times the rule ABM per agent-step.
   - NABM learners beat AgentTorch-style LLM archetypes given the same
     public information.
   - Inside the learner, a once-elicited LLM prior adds nothing detectable
     on share precision and lowers the agents' own utility. The lifecycle
     traced that loss to the prior's parts, and the attribution changed when
     the prior's calibration was matched in the world rather than on a design
     cohort.
5. **A released package surface.** The shared policy-ladder semantics are
   package modules, `neural_abm.policy_ladder` and
   `neural_abm.policy_learners`, released in `v0.1.0a6`. Both recorded
   researcher-pivot studies reproduce byte for byte through them.

## 1.4 What This Paper Does Not Claim

- **No performance claim is central.** The comparisons illustrate the
  framework claim; none of them is the claim.
- **The worlds are synthetic** and were designed by the authors. An
  empirical test of learning against bibliometric data was closed before
  confirmatory scoring because it lacked power at the measured data shape.
  This does not imply the absence of learning.
- **Neural necessity is not established.** Linear, tree, and MLP policies in
  the same slot tie or win depending on the geometry of the world.
- **The LLM comparisons** use one local 4B model family and fixed prompts.
- **The scale is modest.** The policy-ladder designs use 256 to 1,000 agents
  per run, far below the largest LLM-agent simulations.

## 1.5 Organization

Section 2 places NABM between classical ABMs, neural ABMs, and LLM-agent
simulation. Section 3 defines the unit contract. Sections 4 and 5 present two
toy families, and Section 6 the calibration and analysis discipline. Section 7
carries a research question end to end. Section 8 builds the policy ladder in
two domains. Section 9 discusses what the evidence shows and where it stops,
and Section 10 concludes.
