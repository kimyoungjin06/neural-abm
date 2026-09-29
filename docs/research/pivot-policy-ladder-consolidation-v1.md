# Pivot policy-layer ladder consolidation v1

Date: 2026-09-29

Status: `CONSOLIDATION_OF_RECORDED_RESULTS_NO_NEW_EXECUTION`

Machine-readable ledger:
[`pivot_policy_ladder_consolidation_v1.yaml`](../../experiments/configs/pivot_policy_ladder_consolidation_v1.yaml)

## Purpose

The Pivot policy-layer ladder ran from the delayed-feedback prototype to
nonlinear policy ABM v2. Along the way, three registered designs needed
successors. One stopped at a registered check before any result existed. Two
were superseded after their results were opened. Registered dispositions and
their correct reading therefore sit in different records. This document puts the
whole ladder in one place:

- which claims are confirmed, rejected, or still open;
- which capabilities stay in the Pivot domain and which may enter the generic
  NABM API; and
- what the empirical stage should measure first.

Every claim in the ledger cites a hash-bound result file and a field that must
hold. A test checks all of them, so the ledger cannot drift from the evidence.
Nothing new is executed.

## The ladder at a glance

| Step | Question | Freeze / record | Outcome |
|---|---|---|---|
| Delayed feedback | Does failure delay change learning dynamics? | `3fc30cd` / `aef43a6` | Delay amplifies imitative entry |
| Persistence and exit | Can late correction change an observable state? | `4347d53` / `b8f1e73` | Overshoot, purge, and missingness residue |
| Adaptive ladder | Is neural learning needed for that surface? | `bae8d48` / `afa56e4` | No. A scalar rule reproduces it |
| Matched-marginal v1 | Scalar, linear, or nonlinear? | `b960fe0` / `563df21` | Stopped at its invariance check |
| Matched-marginal v2 | Same, corrected | `4c5920c` / `322ffed` | Passed, but confounded |
| Matched-marginal v3 | Same, confounds removed | `14f37af` / `d6b1d6e` | Nonlinear needed; family depends on geometry |
| Nonlinear ABM v1 | Do nonlinear policies matter in the ABM? | `4ffc5dd` / `300254b` | Two of three dispositions were artifacts |
| Nonlinear ABM v2 | Same, out of sample | `c56d741` / `48ab3a3` | Learning transfers; evidence and delay gate behavior |

## What is established

All results are synthetic. The ledger ids are in parentheses.

**The mechanism surface.** Delaying failure visibility raises imitative entry
and failed pivots (C01). Failure-only learning cuts failed pivots at a cost in
productive exploration (C02). Complete delay produces overshoot and then a
purge. Permanently missing failures leave a persistent failed stock (C03).
Similar final stocks can hide different paths, so a single final stock is not
an adequate observable (C04).

**Adaptation.** A fixed rule cannot produce the visibility response (C05).
When productive fit is linear, a one-state scalar rule reproduces the neural
phase directions (C06).

**Representation.** When the success rate is fixed but the feature-outcome
association reverses, a scalar learner fails and a feature-conditional learner
succeeds (C07). When success depends on feature interactions, a nonlinear
policy is needed in both boundary geometries (C08). The better nonlinear
family depends on the geometry. The MLP wins smooth boundaries and the tree
wins axis-aligned ones. This holds in the policy layer and, out of sample, in
the ABM (C09).

**Consequences inside the ABM.** With field-wide evidence, nonlinear policies
learn interaction structure that generalizes to agents the ABM never simulated
(C10). Evidence scale decides whether learning helps at all. Every feature
learner beats the prior with field-wide evidence and loses to it with neighbor
evidence (C11). Learning changes who enters only when failure is visible at
once. From a two-step delay onward, every field arm saturates (C12). The tree
lowers failed stock on axis boundaries at every delay (C13). Post hoc, raw
entrant precision follows the geometry crossover (C14).

## What was overturned

| Claim | How it was registered | Why it does not hold | Overturned by |
|---|---|---|---|
| Neural policy is necessary (R01) | Never supported | Scalar rules reproduce the linear-fit surface; neural wins only smooth boundaries | Adaptive ladder, v3, ABM v2 |
| v2 neural-specific advantage (R02) | Registered pass | Larger on linear worlds; lost to a tuned tree on axis boundaries | v3 |
| v2 Brier margin measures nonlinearity (R03) | Registered pass | Mostly a baseline fit and calibration gap | v2 post hoc, v3 |
| ABM v1 tree dominance (R04) | Registered | Scored agents whose own outcomes trained the learner | ABM v2 |
| ABM v1 learning survives local evidence (R05) | Registered | Reference learner degraded; all lose to the prior | ABM v2 |
| ABM v1 behavior unchanged (R06) | Registered | Rule pooled regimes; entry changes under immediate visibility | ABM v2 |

Each overturned registered claim cites both the original result and the one
that overturned it.

## What is open

| Rank | Question | Kind |
|---|---|---|
| 1 | How long after entry do failures become visible in real pivots, relative to the entry wave? (O01) | Empirical |
| 1 | Does saturation depend on delay relative to entry-wave width, not on absolute delay? (O04) | Synthetic |
| 2 | Do researchers learn from field-wide outcomes or from collaborators? (O02) | Empirical |
| 3 | Is real pivot success a smooth or an axis-aligned function of attributes? (O03) | Empirical |
| 3 | Do calibration differences change ABM behavior? (O05) | Synthetic |
| 3 | Does missing failure evidence interact with nonlinear policies? (O06) | Synthetic |
| 3 | Do failure-only or mixed learning modes change the nonlinear results? (O07) | Synthetic |

O04 matters because of how the ABM enters. In every ABM v2 run, at least half
of final entry happens in the first open step, before any outcome is public.
The linear, tree, and MLP field arms reach 90% of final entry by step 4 on
average. The first public failure arrives at step 4 under immediate
visibility, step 6 under a two-step delay, and step 8 under a four-step delay.
"Saturation from a two-step delay" therefore means that failure evidence
arrives after the residual entry wave has finished. An entry wave that is this
short is a stylized feature of the model, so the finding should be restated in
relative terms before it is compared with data.

## Framework decisions

The bridge rule is to generalize a Pivot capability only when a second model
family needs the same semantics, and otherwise to record an explicit
domain-owned decision.

| Capability | Decision | Second consumer |
|---|---|---|
| Delayed, selective outcome queue | Domain-owned | None |
| Persistence and exit state machine | Domain-owned | None |
| Out-of-sample agent policy scoring | **Extraction candidate**; design first | `toy_classification` scores agents on a held-out test split |
| Cross-fitted shared evidence record | Domain-owned | None |
| Rank-matched fit worlds and matched-marginal generators | Case-owned research instruments | None |
| Batch policy families | Case-owned | None |
| Deterministic parallel task runner | Engineering pattern, not extracted | The generic sweep has no parallel need yet |

Only out-of-sample policy scoring meets the second-consumer rule. Both the
toy classifier and Pivot ABM v2 score every agent's policy on data the agent
never trained on. The next framework step is one shared scoring contract for
both consumers, designed before any API change.

The ladder also produced method guidance for fair model comparisons in NABM.
The step in parentheses is where the practice was missing and had to be added.
The last practice was used throughout:

- match the optimizer and initialization across compared families (v2);
- subtract the same gap in a world where the claimed structure is absent (v2);
- tune every family on training data only (v2);
- score policies out of sample (ABM v1);
- decide per regime, and report saturation as an outcome (ABM v1); and
- exactly nest the predecessor before interpreting a new arm (all steps).

## Empirical target specification

### T01: relative failure-visibility delay (priority 1)

**Why this first.** In the ABM, the visibility delay decides whether any
policy can change who enters, before the policy family matters. It is also the
quantity the lean H3 question already targets: whether entry peaks after the
persistence of later cohorts has begun to decline.

**What to compare.**

- *Model side:* time from entry to the first public failure, relative to the
  width of the residual entry wave.
- *Empirical side:* the lean H3 episode gap between the entry peak and the
  onset of later-cohort persistence decline, divided by the entry-wave width.

Dividing both sides by the entry-wave width makes them dimensionless, so no
mapping from model steps to calendar years is needed. The discriminating
question is whether the empirical set for this ratio lies entirely in the band
the ABM associates with saturated entry, or entirely in the band it associates
with an active entry channel.

**Required before any comparison:**

1. A simulation-side H3 adapter. The ABM does not yet log cohort persistence by
   entry step, so its H3 gap cannot be computed from existing rows.
2. ABM bands for the ratio by visibility regime, fixed in a precommit.
3. O04 resolved, so the bands do not depend on the stylized entry wave.
4. An H3 set estimator calibrated on a new disjoint stress grid.
5. An overdispersion model or a prospective support-domain rule.
6. A complete, immutable source intake.

**Current blockers, each checked against recorded results:**

- field-vector interval containment is 58.1% (111 of 191);
- containment is zero for multimodal peak sets; and
- under the beta-binomial stress, 0 of 16 estimator rows were valid.

On 2026-09-29 the OpenAlex full-core transfer had moved 689.8 of 725.0 GB. Its
state file said `RUNNING`, but no process was running. Resuming it needs your
authorization.

**Claim ceiling.** A matching band would be consistent with a delay regime. It
would not identify social learning, information receipt, or NABM necessity.

### T02: evidence regime (priority 2)

The ABM shows that neighbor-scale evidence makes feature learning worse than
the prior. The empirical question is whether pivot entry responds to
topic-level outcomes or to collaborators' outcomes. This is not designed yet.
It needs author-level data, which was deferred from the lean H3 core.

### T03: boundary geometry (priority 3)

This would be an out-of-sample tree-versus-MLP prediction of pivot success from
attribute proxies, using the v3 comparison discipline. It is ranked last
because the validity of proxies for resource security, network support, skill
distance, and openness is the binding constraint. The geometry question is
only as good as those proxies.

## Flagship outputs

| Bridge output | Supplied by |
|---|---|
| Observed and simulated entry/persistence curves | Nothing yet; requires T01 |
| Visibility-delay by social-learning phase map | Delayed feedback, persistence and exit, ABM v2 |
| Imitative, cautionary, and mixed trajectories | Persistence and exit, adaptive ladder |
| Fair baseline-ladder scorecard | Adaptive ladder, matched-marginal v3, ABM v2 |
| Capability-ablation and framework-change ledger | This consolidation |
| Support-by-attention map | Predecessor study 1 |

Five of the six outputs are supplied. The missing one is the empirical
comparison, which is why T01 comes next.

## Next steps

| Id | Kind | Action | Authorization |
|---|---|---|---|
| N01 | Synthetic | Simulation-side H3 adapter plus entry-wave width variation, to resolve O04 and fix the T01 bands | Separate precommit |
| N02 | Operational | Resume the OpenAlex transfer and close source intake | Your decision |
| N03 | Empirical method | H3 set-estimator successor on a new disjoint stress grid | Separate precommit |
| N04 | Framework | Shared out-of-sample scoring contract for F03 | Optional |

N01 and N02 are independent and can run in parallel. N03 needs neither of them
to start, but T01 cannot be compared until all three are done.

## Boundary

This document consolidates recorded results. It executes nothing, reads no
empirical rows, changes no generic NABM API, selects no model, and establishes
no neural necessity or scientific effect.
