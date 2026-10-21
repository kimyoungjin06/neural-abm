# Information-spread policy ladder: flip-rate study v1 (O26)

Date: 2026-10-21

Status: `O26_ONSET_AT_P_0_25_IN_EVERY_FAMILY`

Design: [flip-rate study v1](../../../docs/research/info-spread-policy-ladder-flip-rate-v1.md),
frozen at `86d3e80` before the run.

## What ran

Each run's credibility relation was reversed with probability p, the flip
rate. Donors drew their signs the same way. The design:

- the linear world;
- p in {0, 0.1, 0.25, 0.5};
- 2 regimes × 2 learner families;
- 20 cells per group, with 16 replicates per cell;
- the arms F1, T0-1, T0-4, and L0, under study v2's learners, measure, and
  rule.

**Runs.**
- GPU-Server, on GPU 2 with 24 workers.
- **Times:** 115 minutes for the development run and 114 for the
  reproduction, against a quoted estimate of 100–150.
- `cells.jsonl` and `summary.json` match byte for byte between the two runs.

## Registered result

The primary measure is share precision on unverified items.

**The design check passes.** At p = 0.5, L0 − T0-1 is material and positive
in all four families.

**The onset is p = 0.25 in all four families.** That is the smallest flip
rate at which the adaptive learner beats the four-donor table.

The table gives the range over the four families of the mean cell Δ, and the
material labels in family order: A linear, A MLP, B linear, B MLP. In the
labels, + and − are material and 0 is indistinguishable.

| p | Runs reversed | L0 − T0-1 | L0 − T0-4 | L0 − F1 |
|---|---:|---|---|---|
| 0 | 0% | −0.005 to +0.001 (0 0 0 0) | −0.048 to −0.017 (− − − −) | +0.041 to +0.081 (+ + + +) |
| 0.1 | 7.8% | +0.034 to +0.036 (+ + + +) | −0.014 to +0.010 (0 − 0 0) | +0.063 to +0.099 (+ + + +) |
| 0.25 | 25% | +0.086 to +0.099 (+ + + +) | +0.050 to +0.069 (+ + + +) | +0.113 to +0.143 (+ + + +) |
| 0.5 | 52.5% | +0.121 to +0.142 (+ + + +) | +0.084 to +0.104 (+ + + +) | +0.189 to +0.219 (+ + + +) |

## Reading

Within this design:

- **At p = 0 the result is study v2's.** The four-donor table beats the
  learner, and the single-donor table matches it.
- **The single-donor table falls behind early.** With about 8% of runs
  reversed (p = 0.1), it already loses to the adaptive learner in all four
  families, by 0.034 to 0.036. Against the four-donor table at that rate,
  the comparison is mixed.
- **By p = 0.25, the best table falls behind too.** The adaptive learner
  beats the four-donor table in every family.
- **The pattern does not depend on the regime or the learner family.**

So the precomputed-table critique holds only when run-specific information is
rare. Here the boundary lies between a tenth and a quarter of runs carrying
it, for a table built from four donors.

**Boundary.**
- **One kind of run-specific information:** the sign of one relation, drawn
  once per run. Shifts within a run are not studied.
- **One table size:** four donors. A table from more donors might move the
  onset.
- **The worlds are synthetic.**
