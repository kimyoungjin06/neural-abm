# Information-spread policy ladder: study v1 (D3a)

Date: 2026-10-16

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_study_v1.yaml`](../../experiments/configs/info_spread_policy_ladder_study_v1.yaml)

Design brief: [info-spread-policy-ladder-design-v1.md](info-spread-policy-ladder-design-v1.md).
The controls that tested this design:
[D2c](../../experiments/evidence/results/info_spread_policy_ladder_d2c_controls_20261016.summary.md)
(adequate), after D2, its diagnoses, and D2b.

This document freezes the study cells for the arms that need no LLM. Their run
is stage D4. The LLM arms are left to a separate amendment, D3b, frozen before
stage D5:

- P0, the elicited prior;
- X0, the archetype baseline;
- L1, the hybrid.

## Question

When does an adaptive NABM policy differ from a policy table written before
the run, and does it differ from the best such table? This addresses the
critique that a learned policy is a precomputed rule table under another name.

Two comparators answer two questions.

- **L0 − T0-1, the closed loop at matched evidence.** T0-1 replays one donor
  run's policy trajectory. It carries the same amount of evidence as L0, but
  from another realization of the item stream.
- **L0 − T0-4, against the best table.** T0-4 averages four donors' policies.
  It has four runs of evidence and an ensemble's variance reduction.

## World

The world is the delayed-veracity spread of the brief:

- 1,000 agents on a Watts–Strogatz graph (k = 6, p = 0.1);
- 60 steps, with 20 items per step and 3 seed sharers per item;
- fact-checks revealed 4 steps after an item enters, for half of the items;
- an exploration floor ε = 0.02.

Fact-checks are public. A learner trains on the outcome of every exposure it
decided on, with one row per agent and item: the share, or else the last
decline.

**Regimes.**
- **A:** one pooled policy, with the cost c = 1 for every agent.
- **B:** agent costs drawn from {0.5, 1, 2}, a sharer-trust feature, and
  partial pooling (a pooled learner plus a per-agent linear residual).

**Geometries.**
- Study cells use the stationary veracity relations linear, smooth, and axis.
- A null geometry, in which features carry no information, is a design check.

## Arms

| Arm | Policy |
|---|---|
| F0 | Granovetter threshold rule |
| F1 | Hand rule (credibility and social proof), the prior of every learner |
| T0-1 | Trajectory table of one donor learner run (donor seed 1000) |
| T0-4 | Trajectory table averaging four donor runs (donor seeds 1000–1003) |
| L0 | Adaptive learner, refitted every 4 steps on the run's own visible outcomes |

**Learner families.**
- Regime A: `linear_exact` and `mlp`.
- Regime B: `partial_linear_exact` and `partial_mlp`.

**Fitting.**
- Linear residuals are fitted to convergence by Newton steps, with l2 = 1e-4
  pooled and 1e-2 per agent.
- MLPs use the Pivot settings (300 epochs, learning rate 0.02, weight decay
  1e-4, hidden width 16), trained on a GPU with deterministic algorithms.
- Donors are learner runs of the same family in the same world.

## Design

- **Groups.** Regime × family × geometry: 2 × 2 × 4 = 16 groups.
- **Cells.** 20 cells per group, each with its own world seed (6000 + cell
  index): a network, thresholds, and costs.
- **Replicates.** 16 paired replicates per cell, each with its own item
  stream. All arms share a replicate's keyed draws.

## Measures

- **Primary.** Share precision: the share of decided shares that are true.
- **Secondary.** False reach (mean cascade size of false items).
- **Also reported.** CPU and GPU time.

## Decision rule

The group rule, as in D2b and D2c, applies to each comparison, with Δ the
paired difference in share precision.

**Material.** The cell-resampled 95% bootstrap interval of the group's mean
cell Δ excludes zero, and that mean is at least 0.01 in magnitude.

**Closed-loop disposition.**

For each regime, count its study groups (2 families × 3 geometries = 6)
where L0 − T0-1 is material and positive.

| Disposition | Condition |
|---|---|
| `AS_EXPECTED` | Regime B in at least 4 of 6, and Regime A in at most 1 of 6 |
| `TABLE_SUFFICES_EVEN_IN_B` | Regime B in at most 1 of 6 |
| `CLOSED_LOOP_MATTERS_EVEN_IN_A` | Regime A in at least 4 of 6 |
| `MIXED` | Otherwise |

**Best-table comparison.**

For each study group, L0 − T0-4 is recorded as one of:

- `ADAPTIVE_BEATS_BEST_TABLE` (material and positive);
- `BEST_TABLE_BEATS_ADAPTIVE` (material and negative);
- `INDISTINGUISHABLE`.

Counts are reported per regime.

**Design check.**

In the null geometry, L0 − F1 must not be material and positive in any
group. A violation is reported as `DESIGN_DEFECT_IN_NULL_WORLD` and blocks
any reading of the study groups.

## Runs and record

- **Two runs.** A development run and a reproduction run, on the same host
  (the GPU server: CPUs for linear fits, GPU 2 for MLPs).
- **Reproduction check.** `cells.jsonl` and `summary.json` must be byte
  identical between the two runs. Timing goes to `progress.jsonl` only.
- **Pilot.** A full-load pilot (2 cells per group) sets the time estimate
  before the development run. Its outputs are not read for results.
- **Record.** A summary and a result test.

## Boundary

Synthetic worlds only. No empirical data and no LLM. The study makes no claim
about real information spread or real people.
