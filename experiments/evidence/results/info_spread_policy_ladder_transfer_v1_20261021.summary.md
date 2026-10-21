# Information-spread policy ladder: transfer study v1 (O23)

Date: 2026-10-21

Status: `O23_TRANSFERRED_TABLE_SUFFICES_IN_BOTH_REGIMES`

Design: [transfer study v1](../../../docs/research/info-spread-policy-ladder-transfer-v1.md),
frozen at `fded050` before the run.

## What ran

In study v2's cells, two tables were built from donors that ran in other
worlds from the same generator. Those worlds share the regime and the
veracity relation, but each has another network, other thresholds, and other
costs.

- T0x-1 is built from one such donor.
- T0x-4 averages four, each from a different world.

**Read from study v2.** F1, T0-1, T0-4, and L0 were read from study v2's
development cells. L0, recomputed at replicate 0 of all 320 cells, equalled
study v2's value in every one.

**Runs.**
- GPU-Server, on GPU 2 with 24 workers.
- **Times:** 36 minutes for the development run and 38 for the
  reproduction, against a quoted estimate of 30–50.
- `cells.jsonl` and `summary.json` match byte for byte between the two runs.

## Registered result

The primary measure is share precision on unverified items.

**The design check passes.** No comparison is material in any null group.

**L0 − T0x-4, the primary comparison.** The disposition is
`TRANSFERRED_TABLE_SUFFICES` in both regimes. The table built from other
worlds beats the adaptive learner in all 12 study groups, by 0.012 to 0.054.
None of the 12 has the adaptive learner ahead.

| Comparison | Study groups (mean cell Δ) | Material |
|---|---|---|
| L0 − T0x-4 | −0.054 to −0.012 | 12 of 12, all negative |
| L0 − T0x-1 | −0.007 to +0.003 | 0 of 12 |
| T0-4 − T0x-4, what same-world donors were worth | −0.005 to +0.001 | 0 of 12 |
| T0-1 − T0x-1 | −0.013 to −0.001 | 1 of 12 (B/partial linear/axis, negative) |

**The expectation stated before the run fails in Regime B.** It expected per-agent
residuals that land on other agents to keep the table from transferring.
They did not.

## Reading

Within this design:

- **The table result does not depend on same-world donors.** A table written
  before the run from donors in other worlds suffices as well as one from
  the same world. Consolidation v6's `SAME_WORLD_DONORS` boundary did not
  matter here.
- **One explanation, not tested here.** The learned policy is a belief that
  an item is true. Agents' costs enter only the decision cutoff, which is
  computed from the evaluation world's own costs. So Regime B's
  heterogeneity acts on the decision rule rather than on what must be
  learned, and a pooled belief transfers.

**Boundary.**
- **"Other worlds"** are draws from one generator, sharing the veracity
  relation. Transfer across relations or domains is not tested.
- **The worlds are synthetic.**
