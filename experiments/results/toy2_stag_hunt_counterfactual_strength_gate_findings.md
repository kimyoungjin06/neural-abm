# Toy 2 Stag-Hunt Counterfactual Strength Gate Findings

Date: 2026-04-30

This gate tests whether the `counterfactual_advantage` local neural update can be sharpened into a more RD-like stag-hunt basin response by changing update strength and action revision scope.

## Setup

- Label: `toy2_stag_hunt_counterfactual_strength_gate_seed01_03`
- Regime: `stag_hunt`
- Payoff: `R=4`, `T=3`, `P=2`, `S=0`
- Classical well-mixed unstable threshold: `p*=2/3`
- Grid: `10 x 10` toroidal von Neumann neighborhood
- Update rule: `neural_policy` with `local_update_rule=counterfactual_advantage`
- Policy prior: `match_p0`
- Mixer: `none`
- Initial cooperation probabilities: `0.55`, `0.60`, `0.65`, `0.70`, `0.75`
- Revision rates: `0.1`, `0.25`, `1.0`
- Selection strengths: `1.0`, `2.0`, `4.0`
- References: Fermi spatial imitation and RD well-mixed with matching `p0`, revision rate, and selection strength
- Seeds: `1`, `2`, `3` for spatial runs; RD uses the first seed because it is deterministic for these settings
- Epochs: `50`

## Final Cooperation

### Neural Counterfactual Advantage

| Revision | Selection | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 1 | 0.217 | 0.273 | 0.467 | 0.630 | 0.810 |
| 0.1 | 2 | 0.217 | 0.277 | 0.460 | 0.630 | 0.797 |
| 0.1 | 4 | 0.213 | 0.273 | 0.460 | 0.630 | 0.797 |
| 0.25 | 1 | 0.100 | 0.157 | 0.350 | 0.603 | 0.797 |
| 0.25 | 2 | 0.103 | 0.163 | 0.353 | 0.613 | 0.803 |
| 0.25 | 4 | 0.107 | 0.163 | 0.363 | 0.620 | 0.800 |
| 1 | 1 | 0.017 | 0.037 | 0.077 | 0.310 | 0.647 |
| 1 | 2 | 0.017 | 0.037 | 0.080 | 0.333 | 0.680 |
| 1 | 4 | 0.017 | 0.037 | 0.083 | 0.343 | 0.697 |

### Fermi Imitation Reference

| Revision | Selection | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 1 | 0.407 | 0.470 | 0.530 | 0.637 | 0.787 |
| 0.1 | 2 | 0.357 | 0.433 | 0.503 | 0.610 | 0.730 |
| 0.1 | 4 | 0.273 | 0.327 | 0.433 | 0.587 | 0.730 |
| 0.25 | 1 | 0.383 | 0.420 | 0.450 | 0.663 | 0.737 |
| 0.25 | 2 | 0.160 | 0.167 | 0.263 | 0.403 | 0.617 |
| 0.25 | 4 | 0.073 | 0.083 | 0.100 | 0.213 | 0.510 |
| 1 | 1 | 0.000 | 0.000 | 0.043 | 0.030 | 0.030 |
| 1 | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1 | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### RD Well-Mixed Reference

| Revision | Selection | p0=0.55 | p0=0.60 | p0=0.65 | p0=0.70 | p0=0.75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 1 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 0.1 | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 0.1 | 4 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 0.25 | 1 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 0.25 | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 0.25 | 4 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 1 | 1 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 1 | 2 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |
| 1 | 4 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

## Acceptance Readout

Target criteria for this gate were: `p0=0.60 < 0.25` and `p0=0.70 > 0.75`.

| Revision | Selection | p0=0.60 | p0=0.70 | p0=0.75 | p0=0.60 passes | p0=0.70 passes | p0=0.75 passes |
| ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0.1 | 1 | 0.273 | 0.630 | 0.810 | False | False | True |
| 0.1 | 2 | 0.277 | 0.630 | 0.797 | False | False | True |
| 0.1 | 4 | 0.273 | 0.630 | 0.797 | False | False | True |
| 0.25 | 1 | 0.157 | 0.603 | 0.797 | True | False | True |
| 0.25 | 2 | 0.163 | 0.613 | 0.803 | True | False | True |
| 0.25 | 4 | 0.163 | 0.620 | 0.800 | True | False | True |
| 1 | 1 | 0.037 | 0.310 | 0.647 | True | False | False |
| 1 | 2 | 0.037 | 0.333 | 0.680 | True | False | False |
| 1 | 4 | 0.037 | 0.343 | 0.697 | True | False | False |

No tested setting passed both target criteria. The closest setting by the simple miss score was `revision_rate=0.25`, `selection_strength=4` with `p0=0.60 -> 0.163` and `p0=0.70 -> 0.620`.

## Final Slopes

| Revision | Selection | Final Coop Slope | Final Policy Slope |
| ---: | ---: | ---: | ---: |
| 0.1 | 1 | 3.087 | 3.343 |
| 0.1 | 2 | 3.027 | 3.338 |
| 0.1 | 4 | 3.047 | 3.344 |
| 0.25 | 1 | 3.680 | 3.558 |
| 0.25 | 2 | 3.700 | 3.610 |
| 0.25 | 4 | 3.687 | 3.631 |
| 1 | 1 | 3.067 | 3.144 |
| 1 | 2 | 3.247 | 3.270 |
| 1 | 4 | 3.333 | 3.339 |

## Readout

The tuning gate separates two effects clearly:

- Increasing `revision_rate` sharpens the lower side of the threshold. At `revision_rate=0.25`, all tested selection strengths push `p0=0.60` below `0.25`; at `revision_rate=1.0`, `p0=0.55`, `0.60`, and `0.65` are all nearly extinguished.
- Increasing `selection_strength` from `1.0` to `4.0` has only a small effect on the neural counterfactual update. For example, at `revision_rate=0.25`, final `p0=0.70` moves only from `0.603` to `0.620`.
- The upper side remains too soft. `p0=0.70` never exceeds `0.75`; the best neural values are around `0.63` at `revision_rate=0.1` and around `0.62` at `revision_rate=0.25`.
- `revision_rate=1.0` overcorrects. It creates a sharp lower-side collapse, but also drags `p0=0.70` down to roughly `0.31` to `0.34`; even `p0=0.75` stays below the target in this fast-resampling regime.
- `p0=0.75` is the first robust above-threshold point for the neural update under moderate revision. At `revision_rate=0.1` or `0.25`, it lands around `0.80` across selection strengths.

The result is not a pure learner-strength problem. The counterfactual update now expresses a strong basin signal, but the spatial local update remains seed-sensitive near `p0=0.70` and does not reproduce the RD hard threshold at `2/3`.

## Interpretation

The previous gate showed that random policy initialization erased `p0`, and that the sampled policy-gradient update preserved the prior instead of expressing the stag-hunt threshold. This gate adds a third finding: once `p0` is visible and the local update is counterfactual, tuning strength/scope alone still cannot make the neural spatial system RD-equivalent.

The likely remaining cause is the mismatch between well-mixed RD threshold logic and local spatial basin/nucleation dynamics. The neural learner is responding to local neighborhood counterfactual payoffs, so a global `p0=0.70` can still contain local regions below the cooperation basin, especially on a `10 x 10` grid with only three seeds.

## Recommended Next Step

Do not keep sweeping `selection_strength`; it is not the active lever here. The next diagnostic should separate local spatial nucleation from learner behavior:

- Add a matched `well_mixed_neighborhood` or shuffled-neighbor diagnostic for the neural counterfactual update while keeping `policy_prior=match_p0`.
- Or run the same counterfactual update on larger grids and more seeds to measure whether the apparent threshold location moves toward `2/3` or remains spatially shifted.
- Keep `revision_rate=0.25` as the moderate diagnostic setting: it fixes the lower-side failure at `p0=0.60` without the aggressive overcollapse seen at `revision_rate=1.0`.

## Artifacts

- `experiments/results/toy2_stag_hunt_counterfactual_strength_gate_seed01_03_summary.csv`
- `experiments/results/toy2_stag_hunt_counterfactual_strength_gate_seed01_03_grouped_summary.csv`
- `experiments/results/toy2_stag_hunt_counterfactual_strength_gate_seed01_03_grouped_summary.md`
- `experiments/results/toy2_stag_hunt_counterfactual_strength_gate_seed01_03_epoch_means.csv`
- `experiments/results/toy2_stag_hunt_counterfactual_strength_gate_seed01_03_epoch_slopes.csv`

Raw run directories were kept under `/tmp/toy2_stag_hunt_counterfactual_strength_gate_seed01_03/runs`.
