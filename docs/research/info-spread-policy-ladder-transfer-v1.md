# Information-spread policy ladder: transfer study v1 (O23)

Date: 2026-10-21

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_transfer_v1.yaml`](../../experiments/configs/info_spread_policy_ladder_transfer_v1.yaml)

This is the first of the four follow-ups (O23, O26, O24, O25) that the user
chose on 2026-10-21 to run in order (N17).

## Question

[Study v2](info-spread-policy-ladder-study-v2.md) found that, in stationary
worlds, a table written before the run suffices: a single donor's trajectory
matched the adaptive learner, and the four-donor average beat it. But the
donors ran in the same world as the evaluation, with the same agents,
network, and costs. Consolidation v6 records this as a boundary
(`SAME_WORLD_DONORS`) and as open item O23.

Does a table written before the run still suffice when its donors ran in
**other worlds from the same generator**? These worlds share the regime and
the veracity relation, but each has another network, other thresholds, and
other costs.

## Design

**The cells are study v2's.**
- 2 regimes × 2 learner families × 4 geometries, with 20 cells per group and
  16 replicates per cell.
- The same primary measure: share precision on unverified items.

**Two new arms.**

| Arm | What it is |
|---|---|
| T0x-1 | The policy trajectory of one donor run in another world, replayed as a table |
| T0x-4 | The average of four donors' trajectories, each from a different other world |

**Donors.**
- **Donor worlds.** For cell c, donor k (0 to 3) runs in a world with seed
  9000 + 4c + k. That range is disjoint from the seeds of every earlier study.
  The donor world has the cell's regime, geometry, and sizes.
- **Donor runs.** Each donor runs the cell's learner family on stream
  replicate 1000, with learner seeds keyed by cell 1000 + 4c + k.
- **Per-agent parts.** A donor's per-agent residuals (Regime B) apply by
  agent index, so in another world they fall on other agents. That is part of
  what transfer means here.

**Study v2's arms are read, not rerun.**
- F0, F1, T0-1, T0-4, and L0 come from study v2's development `cells.jsonl`,
  bound by its record.
- **Parity.** At replicate 0 of every cell, L0 is recomputed. It must equal
  study v2's value exactly, and any failure blocks reading.

## Comparisons and rule

The group rule of study v2 applies, with its own bootstrap keys (tag 10):

- the cell-resampled 95% interval of the mean cell difference excludes zero;
- and that mean is at least 0.01 in magnitude.

| Comparison | Pair | Role |
|---|---|---|
| `transfer_best_table` | L0 − T0x-4 | **Primary** |
| `transfer_closed_loop` | L0 − T0x-1 | Secondary |
| `transfer_cost`, `transfer_cost_single` | T0-4 − T0x-4, T0-1 − T0x-1 | Descriptive: what same-world donors were worth |

**Disposition per regime**, counting the 6 study groups where L0 − T0x-4 is
material and positive:

| Disposition | Condition |
|---|---|
| `ADAPTIVE_BEATS_TRANSFERRED_TABLE` | In at least 4 of 6 |
| `TRANSFERRED_TABLE_SUFFICES` | In at most 1 of 6 |
| `MIXED` | Otherwise |

**Expected before the run.** This is stated so that it can be rejected, not
required.
- **Regime A:** `TRANSFERRED_TABLE_SUFFICES`. One pooled policy over item
  features should transfer.
- **Regime B:** `ADAPTIVE_BEATS_TRANSFERRED_TABLE`. Per-agent residuals land
  on other agents.

**Design check.** In the null geometry, no comparison may be material in
either direction. A violation is `DESIGN_DEFECT_IN_NULL_WORLD` and blocks
reading the study groups.

## Runs and record

1. **Pilot.** A full-load pilot (2 cells per group) for the time estimate,
   which is quoted as a range.
2. **Development and reproduction runs.**
   - On GPU-Server, on GPU 2 with 24 workers, where study v2 ran.
   - `cells.jsonl` and `summary.json` must match byte for byte.
3. **Record.** A summary and a result test.

**Cost.** Each cell runs 4 donor learners and 1 parity learner, against
study v2's 4 donors and 16 adaptive runs. So the run should take roughly a
quarter of study v2's time, which the pilot will check.

## Boundary

Synthetic worlds only, with no empirical data and no LLM. "Other worlds" are
draws from the same generator, not different domains or populations.
