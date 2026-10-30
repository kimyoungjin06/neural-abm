# Information-spread crossover prediction v1 (experiment 3, stage 1)

Date: 2026-10-30

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_crossover_prediction_v1.yaml`](../../experiments/configs/info_spread_crossover_prediction_v1.yaml)

This is recommended experiment 3, which the user directed on 2026-10-30. On
the same day the user chose to run it in two stages: this synthetic stage
first, then OpenAlex discovery episodes as a separate precommit.

## Question

[The flip-rate study (O26)](info-spread-policy-ladder-flip-rate-v1.md) asked
when the in-run learner L0 overtakes a table written before the run. Each
run's credibility relation was reversed with probability p, the flip rate.
- The one-donor table T0-1 fell behind already at p = 0.1.
- The four-donor table T0-4 fell behind between p = 0.1 and p = 0.25.

That is one evidence level and one table size. **Can the crossing be
predicted, before the comparison runs, from two measurable quantities:
how different runs are, and how much evidence a run gives its learner?**

## The rule

**The idea.** A table averages the policies that k donor runs learned. That
average has two kinds of error:
- **bias from heterogeneity,** because the donors' runs differ from this
  one;
- **noise,** reduced k-fold by averaging.

The in-run learner has no heterogeneity bias but carries its run's full
estimation noise. With no other bias, the learner wins when

    p(1 − p) · Dd / V  >  (k − 1)/(k + 1)

- Dd is how far apart learners end up under the two signs;
- V is how much one learner's policy varies across runs of the same world;
- p(1 − p) · Dd is the between-run heterogeneity at flip rate p.

So a larger table moves the crossing later, and more evidence moves it
earlier.

**The registered form** keeps every bias term.
- **Quantity.** For each step, it computes the expected squared error,
  against the generator's true veracity probability, of a k-donor trajectory
  table and of the in-run learner.
- **Donor signs.** Donor signs are binomial(k, p). Donor and run errors are
  independent given their signs.
- **Prediction.** The predicted gap is the table's error minus the
  learner's, averaged over steps. The learner is predicted to win where it
  is positive.
- **Crossing.** The predicted crossing is where the gap rises above zero to
  stay there, on a 0.001 grid of p.
- **Parameters.** Nothing is fitted.

**Measurement.** Every component comes from runs that the comparison does
not use.
- **Worlds.** Eight measurement worlds (seeds 19000+), at each visibility.
- **Runs.** 16 replicates of L0 with the sign fixed at +, and 16 with it
  fixed at −.
- **Probe.** Each learner's policy at every step is evaluated on a keyed
  probe of 2,000 items at the prototype's probe exposure.
- **Components.** The mean policies, their gaps to each sign's truth, and
  the across-replicate variances give seven per-step components. The rule
  needs nothing else.

**Development.** The rule was written with O26's onsets in view. At
visibility 0.5 its k = 1 and k = 4 implications match them qualitatively:
- the one-donor table ties at p = 0 and loses at any p > 0;
- the four-donor table wins at p = 0.

No measurement or comparison of this study ran before the freeze. A
smoke-profile run (60 agents, 20 steps) went through the scoring code to
exercise it, and nothing in the design changed after it.

## The comparison

The comparison uses O26's world, stream, learner, arms, measure, and group
rule. It changes the following:

| | O26 | This study |
|---|---|---|
| Visibility (within-run evidence) | 0.5 | 0.2, 0.5, 1.0 |
| Flip rates | 0, 0.1, 0.25, 0.5 | 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5 |
| Tables | T0-1, T0-4 | T0-1, T0-2, T0-4, T0-8 (donors 1000–1007) |
| Regime and learner | A and B, linear and MLP | A, exact linear |
| World seeds | 11000+ | 18000+ |

**Pairing.** Within a cell, every flip rate and visibility shares the world,
the items, the veracities, and the learner seeds.
- Flips nest across rates.
- The revealed items nest across visibilities.

**Scope.** 27 groups (visibility × flip rate) of 20 cells with 16
replicates, and five arms.
- **Why only Regime A and the exact linear learner.** The rule needs one
  policy function per run, which the pooled learner gives. O26's onsets did
  not depend on the regime or the family.
- **Why visibility.** It sets how many outcomes a learner sees.

## Scoring

**The observed crossing** for each visibility and table size is where the
mean cell difference L0 − T0-k rises above zero to stay there, across the
nine flip rates, linearly interpolated. It is none if the difference is at
or below zero at p = 0.5.

**The score.**
- **Which series.** The nine series with k = 2, 4, 8 are scored. The rule
  puts the one-donor crossing at zero by construction, so k = 1 enters only
  as an expectation.
- **Error.** The error is the absolute difference between the predicted
  and observed crossing, with "none" counted as 0.55.

**Two baselines** are scored the same way:
- **constant:** 0.175, the midpoint of O26's four-donor bracket, for every
  series;
- **table size only:** the symmetric threshold with p(1 − p) = c(k − 1)/(k + 1),
  where c is set so that k = 4 crosses at 0.175. It ignores the measured
  evidence.

| Status | Condition |
|---|---|
| `E3_THE_RULE_PREDICTS_THE_CROSSOVER` | The rule's mean error is at most 0.05 and below both baselines' |
| `E3_THE_RULE_BEATS_THE_BASELINES_BUT_MISSES_BY_MORE_THAN_THE_BOUND` | Below both baselines, above 0.05 |
| `E3_THE_RULE_DOES_NOT_BEAT_THE_BASELINES` | Otherwise |

**Also reported:**
- every group-rule label and mean cell difference;
- how many series each predictor gets within 0.05;
- the measured components and the simple index.

## Expectations, stated before the run

These do not change the status.

| Id | Statement |
|---|---|
| X1 | At p = 0, every table with k ≥ 2 matches or beats L0 at every visibility |
| X2 | The observed crossing does not fall as k grows from 2 to 4 to 8, at every visibility |
| X3 | The observed crossing does not rise as visibility grows from 0.2 to 0.5 to 1.0, at every k |
| X4 | The rule predicts the crossover |
| X5 | At p = 0, L0 − T0-1 is indistinguishable at every visibility |

## Execution

- **Host.** GPU-Server's CPU, with 36 workers, for both runs. The code is
  shipped as a git bundle and checked out at the precommit SHA. A local
  pilot ran several times slower under other users' load.
- **Threads.** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and
  `MKL_NUM_THREADS` are set to 1 and checked.
- **Identity.** `measurement.jsonl`, `cells.jsonl`, and `summary.json` must
  be byte identical between the development and reproduction runs.
- **Runtime.** 40 to 120 minutes per run. A full-load pilot on the server,
  with scratch seeds and two cells per group, printed timings only: 60 tasks
  took 246 s at 36 workers, and a comparison cell 48 to 230 s.

## Boundary

- **Synthetic.** No empirical row is read, and no LLM is called.
- **One regime and one learner family.**
- **One kind of run-specific information:** the sign of one relation, drawn
  once per run. Shifts within a run, the other half of O29, are not studied.
- **Stage 2.** It applies the rule to OpenAlex discovery episodes. It needs
  its own precommit and authorization.
