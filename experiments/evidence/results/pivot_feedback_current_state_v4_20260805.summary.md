# Pivot-feedback active current state v4

Date: 2026-08-05

Independent verification:
`PASS_ACTIVE_E_TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_CONFIRMATORY_MATERIALIZATION_NO_ESTIMATION`

## Active result

The exact guard-only exception resolved the prior authorization contradiction.
The resumed E/G1 run then applied the guard before labels and materialized
72,715,221 confirmatory risk/label rows across all 84 frozen response cohorts.

Two of 28 registered field-episode units failed the frozen minimum of 100 entry
events: episode 11 had 58 and episode 25 had 28. The direct frozen rule excludes
failed units, fails the registered coverage requirement, and forbids episode
pooling or threshold relaxation. The current E/G1 status is therefore
`BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION`.

No feature/control source, sampling, weighting, estimator, bootstrap, model, or
hypothesis decision ran. `pass_e=false`, `pass_g1=false`, M remains `NOT_RUN`, H
remains `BYTE_SEALED_UNOPENED`, and no scientific effect is established.

## Boundary

The active state records the confirmatory materialization rather than resetting
it to zero or `NOT_RUN`. The current cohort cannot be retried by changing scope
or threshold. Any future G1 successor requires a disjoint, outcome-blind,
preauthorized cohort and a fresh freeze. No holdout outcome was read, the seal
remained unopened, and `g4_open=false`.
