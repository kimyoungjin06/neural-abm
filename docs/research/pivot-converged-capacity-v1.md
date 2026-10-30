# Pivot converged capacity comparison v1 (O36)

Date: 2026-10-29

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`pivot_converged_capacity_v1.yaml`](../../experiments/configs/pivot_converged_capacity_v1.yaml)

## Why this study exists

The ladder's capacity conclusions, C07 to C14 in
[consolidation v1](pivot-policy-ladder-consolidation-v1.md), come from two
studies:
- the matched-marginal discriminator v3, which compares policy families
  offline; and
- the nonlinear policy ABM v2, which runs them inside the simulation.

In both, every residual learner was trained by 300 Adam steps from the frozen
prior. [Consolidation v11](pivot-policy-ladder-consolidation-v11.md) found
that this budget leaves the linear learner far from the optimum of its own
objective where its class is right (C80). It opened O36: do the capacity
conclusions hold with every learner converged, or with the penalty set per
parameterization? The user confirmed the study on 2026-10-29.

**A pilot changed the design.** On O35's recorded evidence, the MLP converged
at the registered penalty forecast the linear world worse than its 300-step
fit, not better. So the budget handicaps the linear learner but regularizes
the MLP. Converging everything would swap one handicap for another. The study
therefore adds a regime in which each fit chooses its own training, from one
menu shared by both families.

## Design

Both studies are rerun with their own seeds, profiles, worlds, and decision
code unchanged. Only how each learner is trained changes:

| Regime | Residual learners (linear, MLP) | Trees |
|---|---|---|
| `registered` | The recorded 300 Adam steps | The recorded training |
| `converged` | The registered objective minimized to convergence | The recorded training |
| `tuned` | Chosen for each fit, on that fit's own evidence, from the residual menu | ABM: chosen the same way from the tree menu. Policy layer: v3's own tuning |

**Same seeds, on purpose.** Reusing v3's and v2's seeds lets the registered
regime reproduce both records row for row. Every regime is then paired with
the record on the same worlds, and no new world is drawn.

**Convergence.** L-BFGS with a strong Wolfe line search minimizes mean
cross-entropy plus half the weight decay times the squared parameter norm.
That is the objective Adam with coupled weight decay descends. It starts from
the registered initialization and seed, in float64. Every converged fit must
reach a maximum absolute gradient of 1e-6. Within its 20,000 iterations it may
restart from its current point up to five times.
- **ABM:** the registered weight decay, 0.0001.
- **Policy layer:** each of v3's three weight-decay candidates is converged,
  and v3's own inner folds choose among them.

**The residual menu** crosses two trainings with five weight decays:

| Training | Weight decays |
|---|---|
| 300 Adam steps, as registered | 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 |
| L-BFGS to convergence | 1e-6, 1e-5, 1e-4, 1e-3, 1e-2 |

The menu nests both the registered setting and every converged v3 candidate.
The ABM tree menu is v3's 12 tree candidates, which include the registered
tree, with the registered leaf penalty.

**Selection in the ABM** follows v3's rule:
- four inner folds within each label, by agent id;
- mean validation Brier, ties to the first candidate; then
- a refit on the full buffer with the registered refit seed.

A buffer with fewer than four events of either label keeps the registered
setting unselected. A neighbor learner sees at most four events, so under
tuning it always keeps the registered setting. C11's neighbor half is
therefore retested only under convergence.

**Rows each decision reads.** The fixed rule always comes from the registered
regime.
- `converged`: its own residual arms; the trees come from the registered
  regime.
- `tuned`: its own residual arms and trees.

The ABM runs the fixed rule and the linear, tree, and MLP arms under both
evidence regimes. The scalar and online-logistic arms are not run, since no
ABM decision reads them. That gives 2,448 ABM runs and 48 policy-layer
replicates.

## Checks

The run stops as an invalid implementation if any of these fail:
- **Parity.** The registered regime reproduces its reference row for row in
  both studies. In development the reference is the recorded rows, verified
  by SHA-256. In smoke it is the unmodified runners.
- **Records restated (development).** The registered regime's dispositions
  equal both recorded dispositions, and every capacity claim holds under it.
- **Slots.** The learner slot and the candidate slot are restored after every
  task.
- **Convergence.** Every converged fit reaches the gradient tolerance.

The registered regime runs first, so a parity failure stops the study before
the long regimes start.

## Registered decisions

v3's and v2's decision functions and thresholds judge every regime, unchanged.
Each capacity claim is then checked against the regime's dispositions, by the
same pointers the ledger uses:

| Claim | Check |
|---|---|
| C07 | Policy layer: feature-conditional need supported |
| C08 | Policy layer: nonlinear representation need in both geometries |
| C09 | Policy layer: no family dominates. ABM: the crossover transfers at all three visibilities |
| C10 | ABM: out-of-sample nonlinear learning at all three visibilities |
| C11 | ABM: evidence scale gates useful learning at all three visibilities |
| C12 | ABM: entry channel active with immediate visibility, saturated at lags 2 and 4 |
| C13 | ABM: the tree passes the exit DiD in the axis world at all three visibilities |
| C14 | ABM, immediate: the MLP's raw entrant precision exceeds the tree's in the smooth world and trails it in the axis world |
| R01 | Policy layer: no neural-specific advantage, so R01 stays rejected |

A regime is `CLAIMS_HOLD` if every check holds. Otherwise it is
`CLAIMS_CHANGE`, with the changed claims listed. The study status combines the
two retrained regimes:

| Status | Converged | Tuned |
|---|---|---|
| `O36_HOLD_UNDER_BOTH` | hold | hold |
| `O36_HOLD_WHEN_TUNED_ONLY` | change | hold |
| `O36_HOLD_WHEN_CONVERGED_ONLY` | hold | change |
| `O36_CHANGE_UNDER_BOTH` | change | change |

**Registered quantities.** Each uses a paired 95% t interval over 16
replicates, with field evidence, immediate visibility, and the linear world.
- **Linear-world MLP edge, per regime:** the linear arm's probe Brier minus
  the MLP arm's.
- **MLP budget effect:** the converged MLP's probe Brier minus the registered
  MLP's.
- **Behavior lists:** which family-geometries pass the entry and exit DiDs,
  per regime and visibility.

**Reported:**
- cell means per regime;
- the policy layer's linear-world gap;
- tuned selection frequencies;
- fit counts, fallbacks, and restarts; and
- whether the tuned neighbor rows equal the registered ones.

## Expectations before the run

- **E1.** Under tuning, every capacity claim holds.
- **E2.** Under tuning, the mean linear-world MLP edge is below 0.01.
- **E3.** The MLP budget effect's lower bound is above zero: converging the
  MLP worsens its linear-world forecast.
- **E4.** Under tuning, with immediate visibility, the MLP passes the entry or
  exit DiD in at least one interaction world. v2's MLP failed these DiDs
  because of its linear-world edge, which O35 traced to the linear learner's
  training.

Each will be recorded as held or failed. A failed expectation is recorded as a
rejected claim.

## Runtime

Representative development tasks were timed on scratch seeds, and their
outcomes were not read. Estimates:
- 2,496 tasks and 40 to 75 CPU hours;
- 1.5 to 4 hours per run at 40 workers; and
- the longest task, a tuned MLP field run, at 9 to 20 minutes.

## Boundary

The executable must be committed before the development profile runs. A
deterministic rerun on the same host must reproduce every artifact byte for
byte. No post-result change to the regimes, the menus, the thresholds, or the
decisions is authorized.

The worlds are synthetic, and the hand rule is the only prior. The study makes
no LLM or paid API calls, reads no empirical rows, changes no generic NABM API,
and selects no model. Its regimes are three ways to train these learners, not
an exhaustive search. Whatever it records describes these families under
these menus. It does not establish neural necessity.
