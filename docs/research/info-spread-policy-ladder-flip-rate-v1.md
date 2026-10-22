# Information-spread policy ladder: flip-rate study v1 (O26)

Date: 2026-10-21

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_flip_rate_v1.yaml`](../../experiments/configs/info_spread_policy_ladder_flip_rate_v1.yaml)

This is the second of the four follow-ups the user chose on 2026-10-21 to run
in order (N17).

## Question

Two results bracket the question.

- **[Study v2](info-spread-policy-ladder-study-v2.md).** In stationary worlds,
  a table written before the run suffices.
- **The POS-sign control
  ([D2d](../../experiments/evidence/results/info_spread_policy_ladder_d2d_controls_20261018.summary.md)).**
  When each run draws its own sign for the credibility relation, the
  adaptive learner beats both tables.

Consolidation v6 records the region between the two as open item O26. Where
does the closed loop begin to matter?

## Design

**The dose: a flip rate p.**
- Each run's credibility relation keeps its sign (+3), or is reversed (−3)
  with probability p. Each donor draws its own sign the same way.
- At p = 0 every run is stationary and alike, as in study v2's linear world.
- At p = 0.5 each run's sign is a coin flip, as in the POS-sign control.
- **The flip rates:** 0, 0.1, 0.25, and 0.5.

**The stream.** The study's stream builder uses the prototype's own keyed
sign uniform u and reverses the sign when u ≥ 1 − p. Three things follow:

- **It reproduces the prototype at both ends.**
  - At p = 0 it gives the prototype's stationary stream.
  - At p = 0.5 it gives the prototype's POS-sign stream, byte for byte.
- **Flips nest across rates.** A run reversed at one rate is reversed at
  every higher rate.
- **Rates are paired.** A cell's world, item features, and learner seeds are
  the same at every flip rate.

Tests check all three.

**Cells.**
- The linear geometry.
- 2 regimes × 2 learner families × 4 flip rates give 16 groups.
- 20 cells per group (world seeds 11000 + cell index), with 16 replicates
  per cell.
- **What is study v2's:** the world spec, learners and settings,
  observability, donor replicates (1000–1003), the primary measure (share
  precision on unverified items), and the group rule.

**Arms.** F1, T0-1, T0-4, and L0, as in study v2. The donors run at the same
flip rate.

## Comparisons and rule

The group rule of study v2 applies, with its own bootstrap keys (tag 11), to:

- L0 − T0-1, the closed loop at matched evidence;
- L0 − T0-4, the best table;
- L0 − F1, the hand rule.

**The onset.** For each regime and family, the onset is the smallest flip
rate at which L0 − T0-4 is material and positive. If there is none, it is
`NONE_UP_TO_0.5`. Every comparison's label at every rate is reported with
it.

**Design check.**
- At p = 0.5, L0 − T0-1 must be material and positive in all four families:
  this is the POS-sign control inside the study.
- Otherwise the result is `POSITIVE_CONTROL_FAILS_AT_0.5`, and the onsets
  are not read.

**Expected before the run.**
- At p = 0, the four-donor table matches or beats L0, as in study v2.
- At p = 0.5, L0 beats it, as in D2d.
- The onset is not predicted.

## Runs and record

1. **Pilot.** A full-load pilot (2 cells per group) for the time estimate,
   which is quoted as a range.
2. **Development and reproduction runs.**
   - On GPU-Server, on GPU 2 with 24 workers.
   - `cells.jsonl` and `summary.json` must match byte for byte.
3. **Record.** A summary and a result test.

## Boundary

- **Synthetic worlds only,** with no empirical data and no LLM.
- **One kind of run-specific information:** the sign of one relation, drawn
  once per run. Shifts within a run are a different kind and are not
  studied here.
