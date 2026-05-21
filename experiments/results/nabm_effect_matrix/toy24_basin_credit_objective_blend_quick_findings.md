# Toy2/Toy4 Basin-Credit Objective Blend Findings

Manifest:
`experiments/evidence/toy24_basin_credit_objective_blend_quick.yaml`

Generated artifacts:
- Runs: `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_runs.csv`
- Effects: `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_effects.csv`
- Pairwise effects: `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_pairwise_effects.csv`
- Gate summary: `experiments/evidence/results/toy24_basin_credit_objective_blend_quick.summary.json`
- TtC diagnostics: `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_ttc_bottlenecks_summary.md`
- Revision-pressure diagnostics: `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_revision_pressure_summary.md`

## Gate Result

Overall status after adding confidence-weighted, direction-aware, static-floor,
tail-floor, commitment/hysteresis, precommitment evidence-accumulation,
precommitment decision-feedback, and precommitment social-readiness feedback
variants: fail.

The failure is narrower than the previous commitment-only result. Hard
commitment closed the strict final-ceiling gap, and the new precommitment layer
now starts forcing action-1 before hard commitment is entered. This improves
time-to-ceiling in both toys, and Toy4 now passes the quick gate. Toy2 still
misses the quick-gate TtC threshold:

- Toy2 best main: confidence+precommitment+commitment objective+basin, 3/3
  final ceiling hits, mean time-to-ceiling 12.00, threshold < 10.
- Toy4 best main: confidence+precommitment+commitment objective+basin with
  social-readiness feedback, 3/3 final ceiling hits, mean time-to-ceiling
  11.00, threshold < 12.
- Toy2 improves from 18.33 under confidence-social to 16.00 under hard
  commitment and 12.00 under precommitment.
- Toy4 improves from 18.00 under confidence-social to 13.67 under hard
  commitment and 11.33 under precommitment.
- Feeding prior precommitment readiness into the next epoch's local decision
  probability produces positive feedback deltas, but it does not change seed
  TtC or action-rate trajectories relative to precommitment-only.
- Feeding prior precommitment readiness into social propagation is also active:
  first-10-epoch peer readiness averages 0.265 in Toy2 and 0.316 in Toy4, and
  effective social alpha increases. It still leaves Toy2 unchanged at mean TtC
  12.00, while Toy4 improves only slightly from 11.33 to 11.00.
- The remaining failure is Toy2's early ramp speed. The failure is no longer
  material+basin collapse, pure basin tiny-signal, Toy4 final leakage, or hard
  commitment absence.

## Variant Comparison

| Toy | Variant | Final hits | Mean TtC | Mean payoff | Action rate | Training effective advantage | Action1 basin advantage | Final effective alpha | Commitment rate | Max precommit rate | First precommit epoch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `basin_credit_w1p0_h1_prototype` | 0/3 |  | 2.5567 | 0.6600 | 0.0031 | 0.0031 | 0.0000 | 0.0000 | 0.0000 |  |
| Toy2 | `mixed_individual_basin_w0p5_0p5_h1` | 0/3 |  | 1.0600 | 0.0200 | -0.0969 | 0.0102 | 0.0000 | 0.0000 | 0.0000 |  |
| Toy2 | `mixed_objective_basin_w0p5_0p5_h1` | 3/3 | 22.67 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.0000 | 0.0000 | 0.0000 |  |
| Toy2 | `mixed_objective_basin_confidence_social_w0p5_0p5_h1` | 3/3 | 18.33 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.2484 | 0.0000 | 0.0000 |  |
| Toy2 | `mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1` | 3/3 | 21.33 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.2489 | 0.0000 | 0.0000 |  |
| Toy2 | `mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1` | 3/3 | 18.33 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.2489 | 0.0000 | 0.0000 |  |
| Toy2 | `mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1` | 3/3 | 16.00 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.2483 | 1.0000 | 0.0000 |  |
| Toy2 | `mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1` | 3/3 | 12.00 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.2483 | 1.0000 | 0.7667 | 4.00 |
| Toy2 | `mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1` | 3/3 | 12.00 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.2483 | 1.0000 | 0.7667 | 4.00 |
| Toy2 | `mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1` | 3/3 | 12.00 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.2500 | 1.0000 | 0.7667 | 4.00 |
| Toy2 | `mixed_objective_basin_directional_social_w0p5_0p5_h1` | 3/3 | 18.33 | 3.0000 | 1.0000 | 0.2252 | 0.0004 | 0.2484 | 0.0000 | 0.0000 |  |
| Toy4 | `basin_credit_w1p0_h1_prototype` | 0/3 |  | 0.3140 | 0.5233 | 0.0021 | 0.0021 | 0.0000 | 0.0000 | 0.0000 |  |
| Toy4 | `mixed_individual_basin_w0p5_0p5_h1` | 0/3 |  | 0.0040 | 0.0067 | -0.2076 | 0.0098 | 0.0000 | 0.0000 | 0.0000 |  |
| Toy4 | `mixed_objective_basin_w0p5_0p5_h1` | 2/3 | 16.33 | 0.5980 | 0.9967 | 0.1879 | 0.0008 | 0.0000 | 0.0000 | 0.0000 |  |
| Toy4 | `mixed_objective_basin_confidence_social_w0p5_0p5_h1` | 1/3 | 18.00 | 0.5960 | 0.9933 | 0.1879 | 0.0007 | 0.2481 | 0.0000 | 0.0000 |  |
| Toy4 | `mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1` | 2/3 | 18.00 | 0.5980 | 0.9967 | 0.1879 | 0.0008 | 0.2490 | 0.0000 | 0.0000 |  |
| Toy4 | `mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1` | 1/3 | 18.00 | 0.5960 | 0.9933 | 0.1879 | 0.0007 | 0.2487 | 0.0000 | 0.0000 |  |
| Toy4 | `mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1` | 3/3 | 13.67 | 0.6000 | 1.0000 | 0.1880 | 0.0009 | 0.2477 | 0.9967 | 0.0000 |  |
| Toy4 | `mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1` | 3/3 | 11.33 | 0.6000 | 1.0000 | 0.1880 | 0.0009 | 0.2484 | 1.0000 | 0.7933 | 3.67 |
| Toy4 | `mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1` | 3/3 | 11.33 | 0.6000 | 1.0000 | 0.1880 | 0.0009 | 0.2484 | 1.0000 | 0.7933 | 3.67 |
| Toy4 | `mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1` | 3/3 | 11.00 | 0.6000 | 1.0000 | 0.1880 | 0.0009 | 0.2500 | 1.0000 | 0.8000 | 3.67 |
| Toy4 | `mixed_objective_basin_directional_social_w0p5_0p5_h1` | 1/3 | 18.00 | 0.5960 | 0.9933 | 0.1879 | 0.0007 | 0.2481 | 0.0000 | 0.0000 |  |

## Interpretation

The old material+basin blend still collapses: final action rates are 0.02
in Toy2 and 0.0067 in Toy4, with negative training effective advantage despite
positive basin action1 advantage. This supports keeping `individual_weight` as
the material component only and demoting this variant to diagnostic status.

Pure basin credit remains a tiny-signal scaffold: its final training effective
advantage is close to zero in both toys, and it does not reach final ceiling in
either Toy2 or Toy4.

The new objective+basin blend avoids the material+basin collapse. It preserves
positive training effective advantage and high action rates in both toys, with
Toy2 reaching 3/3 final ceiling hits and Toy4 reaching 2/3. The result is a
directional recovery, not a gate pass, because convergence remains slower than
the current quick-gate thresholds.

Confidence-weighted social propagation is a structural lever, not a direct
margin/commitment loss. It scales peer influence by distance from indifference
and exposes `social_confidence_weighting`,
`mean_social_peer_confidence`, `mean_social_effective_alpha`, and
`max_social_effective_alpha` in aggregate logs. In this quick slice it reduced
Toy2 mean TtC from 22.67 to 18.33 and reduced threshold-crossing bottleneck
counts, but it degraded Toy4 from 2/3 to 1/3 final hits.

Direction-aware social propagation extends the same mechanism by multiplying
peer confidence by agreement between the peer policy and the peer's signed
objective direction. It did not change the quick evidence outcome relative to
confidence-only. That means the Toy4 failure is not explained by strongly
misaligned peers being amplified. The immediate hypothesis after this result
was tail completion: confidence gating may have reduced effective social
pressure and left one or two contributors short of the strict Toy4 ceiling. The
tail-floor variant below tests that hypothesis directly.

The floor test partially confirms the social-pressure diagnosis. Setting
`confidence_weight_floor=0.5` recovers Toy4 final hits from 1/3 to 2/3. It does
not solve the shared gate: Toy4 remains slower than the original
objective+basin variant, and Toy2 slows from 18.33 to 21.33. This argues
against a static floor as the next default.

The tail-floor test makes the next boundary sharper. With
`confidence_tail_floor=0.5` and 0.95 policy/action thresholds, the floor turns on
near the ceiling rather than throughout the whole trajectory. It activates from
epoch 13 in Toy2 for all seeds and from epoch 12/13 in Toy4, for 38-39 epochs of
each 50-epoch run. That preserves Toy2's faster confidence-social TtC
(18.33), but it does not recover Toy4: final hits remain 1/3, matching
confidence-only and direction-aware propagation. This means the missing Toy4
piece is not simply late-stage social pressure after the population is already
near 0.95. The structural issue is earlier: the shared unit needs a mechanism
that changes how high-probability but still unstable near-ceiling states are
stabilized or committed, rather than only changing the peer confidence floor.

Commitment/hysteresis is the first variant in this branch that improves shared
success probability. The mechanism is enabled only after high policy
probability, positive objective direction, and a repeated action-1 streak. It
then forces committed agents to action 1 until policy probability drops below
the exit threshold or the objective direction turns negative. In this quick
slice it enters around epoch 5-8, reaches final commitment rates of 1.00 in
Toy2 and 0.99-1.00 in Toy4, and forces 30-38 actions across each run. This
turns Toy4's strict final ceiling from 1/3 under confidence-social to 3/3 and
reduces mean TtC from 18.00 to 13.67. Toy2 also improves from 18.33 to 16.00.
The remaining failure is therefore not final stochastic action leakage; it is
the earlier ramp from initial mixed state to committed near-ceiling state.

Precommitment evidence accumulation targets that remaining ramp without adding
a supervised margin loss. It keeps a per-agent evidence counter, increases it
when post-social policy probability is high enough and objective direction is
nonnegative, decays it otherwise, and temporarily forces action 1 once the
counter crosses the readiness threshold. Hard commitment remains separate and
still requires the higher policy threshold plus action streak.

This improves the quick result materially. Toy2 seed TtCs become 13, 10, and
13, reducing mean TtC from 16.00 under commitment-only to 12.00. Toy4 seed TtCs
become 12, 12, and 10, reducing mean TtC from 13.67 to 11.33 and passing the
Toy4 quick gate. Precommitment activates before hard commitment: mean first
precommitment epoch is 4.00 in Toy2 and 3.67 in Toy4, while mean first hard
commitment epoch stays unchanged at 7.33 in Toy2 and 6.33 in Toy4. This means
the speedup is not from lowering the hard commitment threshold; it comes from
allowing accumulated evidence to affect actions before the policy has fully
entered the hard committed state.

The remaining Toy2 gap is now small but still real: mean TtC 12.00 misses the
strict quick threshold of 10. The next structural question is therefore not
"how to force commitment harder", but whether the evidence/readiness signal
should feed back into social propagation or local candidate selection earlier
than the current post-social action gate.

The decision-feedback test answers the local-candidate part conservatively. It
uses the previous epoch's precommitment/commitment readiness to blend
decision-time action probabilities toward action 1 before local action sampling.
The feedback path is active: mean first-10-epoch decision probability delta is
0.034 in Toy2 and 0.038 in Toy4, with feedback active for roughly 29% and 34%
of agents respectively. However, it produces exactly the same seed TtCs as
precommitment-only: Toy2 stays at 13, 10, 13 and Toy4 stays at 12, 12, 10. The
diagnosis is that the remaining Toy2 bottleneck is not the next-epoch local
decision readout once evidence exists. The limiting step is earlier evidence
formation/propagation before epoch 7-9, or the strict all-action ceiling
criterion itself.

The social-readiness feedback test answers the peer-propagation part in the
same conservative direction. It uses prior precommitment/commitment readiness as
an additional peer confidence term during social distillation. The path is
active: mean first-10-epoch peer readiness is 0.265 in Toy2 and 0.316 in Toy4,
with effective social alpha increasing from 0.075 to 0.097 in Toy2 and from
0.083 to 0.108 in Toy4. However, Toy2 seed TtCs remain 13, 10, and 13. Toy4
improves only one seed by one epoch, moving mean TtC from 11.33 to 11.00. This
does not justify another confidence-weight sweep. The remaining Toy2 gap is
more likely in earlier evidence formation, the objective/readiness criterion, or
the strict quick-gate ceiling definition than in late peer weighting.

The revision-pressure diagnostic reframes the same runs without adding a new
model. It asks whether objective/revision pressure appears before policy
readiness or action response. The answer is yes for the current objective+basin
family. In both Toy2 and Toy4, revision pressure appears at epoch 1, while
policy readiness appears around epoch 5. Without precommitment, aggregate action
response appears much later: Toy2 action response averages epoch 10.67 under
confidence-social and Toy4 averages epoch 11.00. Precommitment shortens the
pressure-to-action lag, but does not change pressure-to-policy lag: Toy2 moves
to action response epoch 8.33, Toy4 to epoch 7.67, while policy readiness stays
at epoch 5 in both toys.

This supports the next structural hypothesis: the bottleneck is not missing
objective pressure, and it is not solved by feeding existing readiness into
decision or social hooks. The architecture is translating an already-positive
revision signal through a policy-probability-first path. A neural revision
operator is therefore a better next prototype than another precommitment or
confidence-weight variant. The caveat is important: the diagnostic uses existing
domain advantage fields as an offline proxy, so it justifies a prototype but
does not yet prove that a learned revision operator will outperform the current
policy learner.
