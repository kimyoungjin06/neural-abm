# Information-spread policy ladder: study v2, stage D4 v2

Date: 2026-10-19

Status: `D4V2_NULL_PASSES_TABLE_SUFFICES_EVEN_IN_B`

Design: [study v2](../../../docs/research/info-spread-policy-ladder-study-v2.md),
frozen at `dcc6e74` before the run, together with the LLM arms (D3b).

## What ran

The frozen study ran the five non-LLM arms (F0, F1, T0-1, T0-4, and L0) on
the GPU server, with GPU 2 for the MLPs and 24 workers:

- 16 groups (2 regimes × 2 families × 4 geometries);
- 20 cells per group and 16 replicates per cell;
- world seeds from 7000.

The primary measure is share precision on unverified items: items whose
fact-check is not public at the decision.

Run times:

| Run | Wall time |
|---|---:|
| Development | 130 min |
| Reproduction | 129 min |

`cells.jsonl` and `summary.json` match byte for byte between the two runs.
The quoted estimate was 140–200 minutes, so the runs came in below it.

## Registered result

**The design check passes.** In the four null groups, L0 − F1 is between
−0.002 and −0.001, and no group is material.

**Closed loop at matched evidence (L0 − T0-1).** The disposition is
`TABLE_SUFFICES_EVEN_IN_B`.

| Regime | Material and positive | Mean cell Δ across the 6 study groups |
|---|---|---|
| A | 1 of 6 (A/mlp/smooth, +0.011 [+0.001, +0.022]) | +0.003 to +0.011 |
| B | 0 of 6 | +0.000 to +0.010 |

In the other 11 groups, the interval covers zero.

**Best table (L0 − T0-4).** `BEST_TABLE_BEATS_ADAPTIVE` in all 12 study
groups.

| Family | L0 − T0-4 |
|---|---|
| Exact linear (A and B) | −0.010 to −0.029 |
| MLP (A and B) | −0.039 to −0.049 |

**Against the hand rule (L0 − F1).** Material and positive in all 12 study
groups, at +0.040 to +0.129.

**Mean precision on unverified items, study groups.**
- F0: 0.40, the base rate. The threshold rule ignores veracity.
- F1: 0.53–0.64.
- L0: 0.60–0.71.
- T0-4: 0.65–0.75.

## Reading

Within this design, in stationary worlds:

- **Learning from outcomes matters.** Every learner beats the hand rule.
- **The closed loop adds no detectable gain at matched evidence.** A
  single donor run's trajectory, replayed as a table, matches the adaptive
  learner in 11 of 12 groups.
- **A table written before the run beats the adaptive learner.** Averaging
  four donor runs wins in every group, more so for the MLPs.
- **The prediction that Regime B needs the closed loop is not supported.**
  The design expected per-agent costs and partial pooling to make it matter;
  here they did not.

**Boundaries.**

- **The donors are learner runs in the same world:** the same agents,
  network, and costs.
  - In Regime B, their per-agent residuals therefore carry over to the same
    agents.
  - The study does not test tables that must transfer to another world.
- **The relations are stationary.** The D2c and D2d controls showed the
  case this study leaves out. When a run carries information that donors
  lack (POS-sign), L0 beats both tables by 0.08–0.15.
- **Synthetic worlds only.** No claim is made about real information
  spread.

## Next

- **D5.** The LLM arms frozen with this study (P0, X0, and L1) run in these
  cells, paired with these arms. They run in a checkout of `dcc6e74`, with
  this record passed in.
