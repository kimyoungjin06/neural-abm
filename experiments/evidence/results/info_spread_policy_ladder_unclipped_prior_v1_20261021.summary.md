# Information-spread policy ladder: unclipped-prior study v1 (O24)

Date: 2026-10-21

Status: `O24_UNCLIPPED_PRIOR_BEATS_THE_HAND_RULE_BUT_ADDS_LITTLE_INSIDE_THE_LEARNER`

Design: [unclipped-prior study v1](../../../docs/research/info-spread-policy-ladder-unclipped-prior-v1.md),
frozen at `eb49057` before the run.

## What ran

The once-elicited prior was rebuilt from D5's first registered elicitation
without the inherited probability clip. No LLM was called.

- **P0u:** the raw log-odds, distilled and calibrated as in D3b.
- **L1u:** the study learner started from P0u.

Both ran in study v2's cells. F1, L0, P0, and L1 were read from the study v2
and D5 records. L0, recomputed at replicate 0 of all 320 learner cells,
equalled study v2's value in every one.

**Runs.**
- GPU-Server, on GPU 2 with 24 workers.
- **Times:** 85 minutes for the development run and 85 for the
  reproduction, against a quoted estimate of 80–130.
- `cells.jsonl` and `summary.json` match byte for byte between the two runs.

**The unclipped prior.** Its weights were first read here.

| | Regime A | Regime B |
|---|---|---|
| Distillation R² (D5's clipped prior) | 0.81 (0.026) | 0.76 (0.48) |
| Weights: credibility, novelty, intensity, fit | +8.1, −1.6, −2.0, −0.7 | +10.9, −5.6, −3.4, −1.6 |

The world's linear relation is credibility +3, novelty −1, intensity −2, and
fit 0. So the signs of the four item weights match it.

## Registered result

The primary measure is share precision on unverified items.

**The design check passes.** L1u − P0u is not material and positive in any
null group.

The table gives the range of the mean cell Δ over each regime's 6 study
groups, and the material labels in group order: linear (axis, linear,
smooth), then MLP (axis, linear, smooth).

| Comparison | Regime A | Regime B | Disposition |
|---|---|---|---|
| P0u − F1 | +0.057 to +0.087 (+ + + + + +) | +0.057 to +0.086 (+ + + + + +) | `MOSTLY_POSITIVE` in both |
| L1u − L0 | +0.005 to +0.020 (+ + + 0 0 0) | −0.000 to +0.013 (0 0 0 0 + +) | `MIXED_OR_NONE` in both |
| L1u − P0u | −0.042 to +0.030 (+ 0 + + − 0) | −0.015 to +0.075 (+ 0 0 + − +) | `MIXED_OR_NONE` in both |
| P0u − P0, descriptive | +0.222 to +0.295 (all +) | −0.023 to −0.018 (all −) | |
| L1u − L1, descriptive | −0.002 to +0.007 (none) | −0.004 to +0.004 (none) | |

## Reading

Within this design:

- **D5's flat Regime A prior was the readout, not the model.**
  - Without the clip, the model's answers carry the relation's direction.
  - The prior alone beats the hand rule in all 12 study groups, and in
    Regime A it rises from the base rate to 0.62–0.70.
- **An informative LLM prior adds little inside the learner.**
  - L1u beats L0 in 5 of 12 study groups, by 0.012 to 0.020, and loses in
    none.
  - Learning from the run's outcomes still makes up most of the difference
    in starting point.
- **In Regime B, removing the clip lowered the prior's precision slightly.**
  This may be the selectivity of the steeper clipped prior, which O25
  scores by utility.

**Boundary.**
- **The model and readout.** One small local model, one prompt wording, and
  one readout choice.
- **The worlds are synthetic.** The model's knowledge of real information
  spread is not established.
