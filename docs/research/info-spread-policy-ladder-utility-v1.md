# Information-spread policy ladder: utility study v1 (O25)

Date: 2026-10-21

Status: `PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION`

Machine-readable contract:
[`info_spread_policy_ladder_utility_v1.yaml`](../../experiments/configs/info_spread_policy_ladder_utility_v1.yaml)

This is the last of the four follow-ups the user chose on 2026-10-21 to run in
order (N17).

## Question

Every registered comparison so far was scored on share precision on
unverified items. Precision rewards selectivity, and it is not what the
agents optimize. D5 showed why that matters:

- the prior P0 in Regime B shared 2–15% fewer items than the learner;
- the archetypes X0 shared 1.4–1.7 times as many items as the hand rule.

Consolidation v6 records this as open item O25.

**The utility.** It is the agents' own objective:

- a share of a true item gains 1;
- a share of a false item costs the sharer c;
- a decline is worth 0.

The decision rule, share when the probability of truth reaches c/(1 + c),
maximizes it for a calibrated belief.

**The question.** Does each registered disposition still hold when every arm
is scored by this utility, in the same cells and runs?

## Measure

**Utility per agent on unverified items.**
- **Definition.** True shares minus the summed costs of false shares,
  divided by the number of agents.
- **What counts.** Decided shares, both the policy's and the exploration
  floor's, of items whose fact-check is not public at the decision.

**Regime A** (c = 1 for every agent).
- Utility is (true − false) / agents. It is computed from the counts already
  recorded by study v2, D5, and O24, so no run is needed.

**Regime B** (costs of 0.5, 1, or 2).
- **The reruns.** Every arm is rerun with
  `scripts/info_spread_utility_measure_v1.py`, using study v2's seeds,
  donors, and streams.
- **What the module does.** It adds the sharer's cost of each false share to
  the unverified measure's reconstruction. That reconstruction is checked
  against the simulator in every run.
- **Parity.** Every rerun row must reproduce its recorded precisions, false
  reach, decisions, and counts by state exactly. Any failure blocks reading.

## Arms and comparisons

**Arms.**
- F1, T0-1, T0-4, and L0, from study v2;
- P0, X0, and L1, from D5;
- P0u and L1u, from O24.

**Comparisons.** These are the ten registered comparisons:

- L0 − T0-1, L0 − T0-4, and L0 − F1;
- L1 − L0, L0 − X0, L1 − P0, and P0 − F1;
- P0u − F1, L1u − L0, and L1u − P0u.

**Rule.** This is the group rule, with bootstrap tag 13:
- the cell-resampled 95% interval of the mean cell difference excludes
  zero;
- and its magnitude is at least the regime's minimum effect.

**The minimum effect.** Utility is not on precision's scale, so the 0.01
minimum does not carry over. The minimum effect is 1% of F1's mean absolute
utility over the regime's study groups, one scale per regime. This formula is
fixed before any utility is computed.

**Dispositions.**
- **Per comparison and regime,** under utility and, from the records, under
  precision:
  - `MOSTLY_POSITIVE` if positive in at least 4 of 6 study groups;
  - `MOSTLY_NEGATIVE` if negative in at least 4 of 6;
  - `MIXED_OR_NONE` otherwise.
- **Agreement.** Whether the two summaries agree, for each comparison and
  regime.
- **Overall.** `CONCLUSIONS_HOLD_UNDER_UTILITY` if every summary agrees,
  otherwise `SOME_CONCLUSIONS_CHANGE_UNDER_UTILITY`, with the list of
  changes.

**Null groups** are reported, not judged.
- **Why there is no null check.** In the null world every share has negative
  expected utility, because the base rate, 0.4, is below the cutoff. So a
  learner that rightly learns the base rate gains utility by sharing less.
  That is not a leak.
- **The validity check** is parity instead.

**Expected before the run.** Not predicted. The comparisons most likely to
change are those that turn on how much an arm shares, P0 and X0 in
particular.

## Runs and record

1. **Pilot.** A full-load pilot (2 cells per group) for the time estimate.
2. **Development and reproduction runs.**
   - On GPU-Server, on GPU 2 with 24 workers.
   - `cells.jsonl` and `summary.json` must match byte for byte.
3. **Record.** A summary and a result test.

## Boundary

- **The utility.** One utility, the one the agents' decision rule assumes.
  Welfare effects on others, such as the reach of false items, are not
  scored.
- **The worlds.** Synthetic only, with no empirical data and no LLM call.
