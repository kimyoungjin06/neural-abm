# Information-spread policy ladder: stage D2d controls

Date: 2026-10-18

Status: `D2D_ADEQUATE_UNVERIFIED_MEASURE_READY_FOR_STUDY_V2`

Specification: [D2d](../../../docs/research/info-spread-policy-ladder-d2d-v1.md),
fixed before the run. Code: `09c0a18`. No study cell was scored.

## What ran

D2d tested the group rule on the measure the user chose on 2026-10-18:
share precision on unverified items, whose fact-check is not public at the
decision.

- **Controls:** NEG, POS-sign, and NULL.
- **Groups:** 4 families per control.
- **Size:** 20 cells per group, with 16 replicates.
- **Settings:** D3a's world, learners, and observability, with world seeds
  from 5000.

The run took 70 minutes. A full-load pilot, scaled by the factors seen in
D4, had put the estimate at 50–90 minutes.

## Result

The table shows the mean cell Δ with its cell-resampled 95% interval.

| Control / pair | Unverified measure | Old measure (all items) |
|---|---|---|
| NEG, 8 pairs | −0.004 to +0.005; all intervals cover 0 | −0.005 to +0.005; all cover 0 |
| POS-sign, L0 − T0-1, 4 groups | +0.126 to +0.152; all material | +0.135 to +0.155; all material |
| POS-sign, L0 − T0-4, 4 groups | +0.083 to +0.100; all material | +0.091 to +0.102; all material |
| NULL, L0 − F1, A/linear_exact | −0.009 [−0.014, −0.004]; not material | −0.007 [−0.012, −0.002] |
| NULL, L0 − F1, A/mlp | −0.003 [−0.007, +0.000] | +0.008 [+0.004, +0.011]; below 0.01 |
| NULL, L0 − F1, B/partial_linear_exact | −0.002 [−0.005, +0.000] | −0.000 [−0.003, +0.002] |
| NULL, L0 − F1, B/partial_mlp | −0.002 [−0.005, +0.002] | **+0.025 [+0.023, +0.028]; material** |

## Reading against the specification

**Adequate.** All three conditions hold on the unverified measure:

1. no NEG pair is material;
2. L0 − T0-1 in POS-sign is material and positive in all 4 groups;
3. L0 − F1 in NULL is not material and positive in any group.

**The NULL control detects D4's defect.** On the old measure, NULL shows
the same material gain in B/partial_mlp as D4 did (+0.025 here, +0.027 in
D4). On the unverified measure that gain is gone.

**Two further readings, still on controls only.**

- **The closed-loop effect survives the change of measure.** When a run
  carries information that donors lack, the learner still beats both tables
  on unverified items.
- **For the MLPs, the old measure was somewhat inflated.** Their POS-sign
  effects are 0.009–0.018 larger on the old measure, which is consistent
  with item recognition adding to them. For the linear learners the two
  measures differ by 0.003 or less.

## Next

As the user's choice entails:

- freeze study v2 and the LLM arms (D3b) together, on the unverified
  measure;
- then run D4 v2.
