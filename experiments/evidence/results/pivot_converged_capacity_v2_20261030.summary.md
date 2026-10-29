# Pivot converged capacity comparison v2 (O36): result

Date: 2026-10-30

Status: `O36_HOLD_UNDER_BOTH`

Design: [pivot-converged-capacity-v2.md](../../../docs/research/pivot-converged-capacity-v2.md),
frozen at `1cfd1abb4ea9f7261209d61bbe48a59d5d6d9995`.
Machine-readable record:
[pivot_converged_capacity_v2_20261030.summary.json](pivot_converged_capacity_v2_20261030.summary.json).

## v1

v1, frozen at `4062c8c`, stopped at its convergence check before writing any
output. Its registered phase had reproduced both records. One inner fit of a
tuned field MLP then missed the gradient tolerance. v2 changed only how such a
fit is handled. No converged or tuned outcome was computed under v1.

## Execution

- **Host:** the workstation, on CPU with 40 workers.
- **Runs:** 2,448 ABM runs and 1,440 policy-layer rows, taking 4,610 s for
  the development run and 4,627 s for the reproduction run. Each run took
  about 77 minutes, under the quoted 2 to 5 hours.
- **Reproduction:** all five artifacts are byte identical.
- **Registered regime:** it reproduced both records row for row (1,008 ABM
  rows and 480 policy-layer rows) and both recorded dispositions. Every
  capacity claim holds under it.
- **Convergence:**
  - All 165,615 acting fits of the converged ABM regime reached the
    tolerance, and so did all 1,644 acting L-BFGS fits of the tuned ABM
    regime.
  - Of 63,777 inner L-BFGS fits under tuning, 1 missed it, which made one
    candidate unavailable once.
  - No policy-layer fit missed it.
- **Neighbor learners:** under tuning, every neighbor-learner fit kept the
  registered setting, and those rows equal the registered ones.

## The capacity claims hold under both regimes

| Claim | Registered | Converged | Tuned |
|---|---|---|---|
| C07 feature-conditional need | holds | holds | holds |
| C08 nonlinear need, both geometries | holds | holds | holds |
| C09 family depends on geometry | holds | holds | holds |
| C10 out-of-sample nonlinear learning | holds | holds | holds |
| C11 evidence scale gates learning | holds | holds | holds |
| C12 entry responds only when failure is visible at once | holds | holds | holds |
| C13 tree lowers failed stock on axis boundaries | holds | holds | holds |
| C14 raw entrant precision follows the crossover | holds | holds | holds |
| R01 stays rejected | holds | holds | holds |

## What the training changes

**In the linear world, the MLP's edge was the linear learner's training.**
Field evidence, immediate visibility, probe Brier:

| Regime | Linear | MLP | Tree | Linear minus MLP |
|---|---:|---:|---:|---|
| registered | 0.0594 | 0.0130 | 0.0727 | +0.046 [+0.042, +0.050] |
| converged | 0.0250 | 0.0172 | 0.0727 | +0.008 [+0.004, +0.012] |
| tuned | 0.0134 | 0.0142 | 0.0709 | −0.001 [−0.002, +0.001] |

- **The policy layer agrees.** Its linear-world gap, linear minus MLP, is
  +0.028 and +0.038 registered, −0.002 and −0.002 converged, and −0.001 and
  +0.000 tuned.
- **Converging the MLP hurts it.** At the registered penalty, its
  linear-world Brier rises by 0.004 [0.002, 0.006]. The 300-step budget had
  been regularizing it.
- **The tuned fits chose different trainings.**
  - In the linear world, the linear learner chose a converged fit in every
    selection, at a weight decay of 1e-6 in 66% of them.
  - In the same world, the MLP chose 300 Adam steps in 62% of its selections.
  - In the interaction worlds, the MLP most often chose a converged fit at
    1e-3.
  - Refits with fewer than four events of either label kept the registered
    setting unselected.

**In the interaction worlds, the comparisons barely move.** Probe Brier, as
above:

| World | Regime | Linear | MLP | Tree |
|---|---|---:|---:|---:|
| smooth | registered | 0.1765 | 0.0845 | 0.1243 |
| smooth | converged | 0.1760 | 0.0874 | 0.1243 |
| smooth | tuned | 0.1761 | 0.0831 | 0.1210 |
| axis | registered | 0.1874 | 0.1142 | 0.0604 |
| axis | converged | 0.1879 | 0.1274 | 0.0604 |
| axis | tuned | 0.1877 | 0.1124 | 0.0605 |

The difference-in-differences margins therefore grow, because they subtract a
smaller linear-world gap:
- **ABM, MLP field learning, smooth world, immediate visibility:** 0.046
  registered, 0.081 converged, 0.094 tuned.
- **Policy layer, nonlinear need:** 0.134 and 0.120 registered, against 0.161
  and 0.158 tuned, in the smooth world; 0.094 and 0.085, against 0.125 and
  0.120, in the axis world.
- **The crossover margins hold.** The MLP beats the tree in the smooth world
  by 0.040, 0.037, and 0.038, and the tree beats the MLP in the axis world by
  0.054, 0.067, and 0.052. Each passes in 16 of 16 replicates.

**The MLP's behavior gains are interaction-specific too.** With immediate
visibility:
- **Registered:** only the tree passed the entry and exit DiDs.
- **Converged and tuned:** the MLP also passes the entry DiD in both
  interaction worlds (tuned: 0.039 smooth, 0.043 axis, 16 of 16).
- **Exit:** under tuning, the MLP passes the exit DiD in the smooth world
  (0.030, 15 of 16). In the axis world it gains 0.011, in only 11 of 16.
- **The cause is the linear world, where the DiD subtracts:**
  - The tuned linear learner's entrant precision is 0.433, against the MLP's
    0.434, and its active failed stock is 0.010, against 0.013.
  - Registered, the linear learner's figures were 0.408 and 0.044, against the
    MLP's 0.436 and 0.011.
- **The tree gains a little.** Under tuning, it also passes the smooth-world
  exit DiD with a two- and a four-step delay (0.026, 15 of 16).

## Expectations

- **E1 held.** Under tuning every capacity claim holds.
- **E2 held.** The tuned linear-world MLP edge is −0.001.
- **E3 held.** Converging the MLP worsens its linear-world forecast by 0.004
  [0.002, 0.006].
- **E4 held.** Under tuning the MLP passes the entry DiD in both interaction
  worlds and the exit DiD in the smooth world.

## What this means

- **The capacity conclusions do not depend on the 300-step Adam budget.**
  They hold with every residual objective converged, and with each fit's
  training chosen on its own evidence.
- **The DiD rules were conservative.** v2 and v3 subtracted a linear-world
  gap that was mostly the linear learner's training. With it removed, the
  nonlinear margins grow by 0.03 to 0.05.
- **The MLP's behavior DiD failures were a training artifact.** In v2 the
  MLP failed them because of its linear-world edge. Once that edge is gone,
  the MLP's entry gains are interaction-specific in both geometries, and its
  exit gain in the smooth one.

## Boundary

The worlds are synthetic, and the hand rule is the only prior. The regimes
are three ways to train these learners, not an exhaustive search. Neighbor
learners keep the registered setting under tuning. The study selects no model
and does not establish neural necessity.
