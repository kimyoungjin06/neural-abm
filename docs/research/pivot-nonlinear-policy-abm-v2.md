# Pivot nonlinear policy ABM v2

Date: 2026-09-28

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_nonlinear_policy_abm_v2.yaml`](../../experiments/configs/pivot_nonlinear_policy_abm_v2.yaml)

## Why v2 exists

This successor was designed after the
[v1 result](../../experiments/evidence/results/pivot_nonlinear_policy_abm_v1_20260928.summary.md)
was opened. v1 found that field-wide nonlinear policies change who enters when
failure is visible immediately, and that a four-step failure delay saturates
entry. It also exposed four defects in its own design:

1. The policy Brier scored agents whose own outcomes were in the learner's
   buffer, so it measured memorization. Trees memorize best.
2. The local rule used the linear learner as its reference. That reference
   degrades differently across worlds when evidence is thin.
3. The entry rule required both visibility regimes to pass, but delayed failure
   saturated entry.
4. The field record held each agent's own outcome, so high-capacity learners
   could purge failed agents by memorizing identities.

v2 fixes all four. Seeds, fit worlds, learners, and the ABM itself are otherwise
the v1 design.

## Changes

**Out-of-sample probe cohort.** Each replicate draws a separate probe cohort
of 256 agents from the same population generator with a fresh seed. The
probe agents are never simulated. Their structural labels follow the same
fit-world rule, applied within the probe cohort. Every arm's final policy is
scored on these probes. For field arms that is each fold learner, for local
arms each agent's learner, for the online logistic arm each agent's final
weights, and for the fixed rule the prior. Probe Brier replaces the in-situ
policy Brier as the learning metric. The in-situ value is still reported.

**Cross-fitted field record.** Agents are split into four folds by identifier.
Each fold has its own learner, trained on public events from the other three
folds. Every agent acts on its own fold's learner, so its policy never reflects
its own outcome. An isolation check stops the run if any fold learner trained on
its own fold.

**Fixed prior as the local reference.** Useful learning asks whether a linear,
tree, or MLP learner beats the fixed prior on probe Brier, under each evidence
regime, with no DiD.

**Per-visibility decisions, with saturation.** Every decision is made
separately for each visibility regime. A regime is `ENTRY_SATURATED` when the
linear, tree, and MLP field arms each reach full entry in at least 12
replicates of every world. A two-step failure delay is added, to locate where
saturation begins.

**Exit channel.** Without identity memorization, a drop in active failed stock
can be read as interaction learning acting through the exit score. It becomes a
registered decision.

## Registered decisions

Each decision is made per visibility regime and uses the v1 paired rule: the
mean must meet the threshold, and the improvement must be strictly positive in
at least 12 of 16 replicates. DiD subtracts the same gap in `linear_fit`.

| Decision | Metric | Comparison | Statistic | Threshold |
|---|---|---|---|---:|
| Field nonlinear learning | probe Brier | tree or MLP over linear | DiD | 0.01 |
| Field geometry crossover | probe Brier | MLP over tree on smooth; tree over MLP on axis | raw | 0.005 |
| Useful learning | probe Brier | linear, tree, or MLP over fixed prior, each regime | raw | 0.01 |
| Entry behavior | productive share of entrants | tree or MLP over linear | DiD | 0.02 |
| Exit channel | active failed rate, lower is better | tree or MLP over linear | DiD | 0.01 |

Each visibility regime then reports:

- **Learning:** `OUT_OF_SAMPLE_NONLINEAR_LEARNING` or `NO_NONLINEAR_LEARNING`.
- **Crossover:** `CROSSOVER_TRANSFERS_TO_ABM`, `MLP_DOMINATES_IN_ABM`,
  `TREE_DOMINATES_IN_ABM`, or `TIED_OR_MIXED`.
- **Evidence scale:** `EVIDENCE_SCALE_GATES_USEFUL_LEARNING`,
  `USEFUL_LEARNING_SURVIVES_LOCAL_EVIDENCE`, or
  `NO_USEFUL_LEARNING_UNDER_FIELD_EVIDENCE`.
- **Entry:** `ENTRY_SATURATED`, `ENTRY_CHANNEL_ACTIVE`, or
  `ENTRY_CHANNEL_ABSENT`.
- **Exit:** `EXIT_CHANNEL_ACTIVE` or `EXIT_CHANNEL_ABSENT`.

## Design

The development profile uses 256 agents, 18 steps, 2 burn-in steps, and 16
paired replicates. It crosses three visibility regimes, three fit worlds, and
10 arms, for 1,440 runs. Fresh seed namespaces are used for the population, the
probe cohorts, and the family transfer check. The fixed and online-logistic
arms must exactly nest the persistence predecessor in the linear world.

## Boundary

The executable must be committed before the development profile runs, and a
deterministic rerun must reproduce every artifact byte for byte. No post-result
change to learners, worlds, thresholds, folds, or cohorts is authorized.

This is a synthetic generative comparison. It reads no empirical rows, selects
no model, and changes no generic NABM API. The fit worlds are stylized
alternatives. Which geometry real Pivot success follows is an empirical
question this study cannot answer.
