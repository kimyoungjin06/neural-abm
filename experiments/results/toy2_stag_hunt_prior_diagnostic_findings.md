# Toy 2 Stag-Hunt Prior Diagnostic Findings

Date: 2026-04-30

This diagnostic decomposes why the Toy 2 neural policy did not reproduce the
classical stag-hunt basin threshold. The goal is not to redesign the learner,
but to identify whether the `p0` signal is erased by the initial neural policy
prior, the sampling/revision channel, or the local policy-gradient update.

## Setup

- Label: `toy2_stag_hunt_prior_diagnostic_seed01_03`
- Regime: `stag_hunt`
- Update rule: `neural_policy`
- Mixer: `none`
- Initial cooperation probabilities: `0.5`, `0.6`, `0.7`
- Revision rate: `0.1`
- Policy priors: `default`, `match_p0`, fixed `0.5`
- Learning: `false`, `true`
- Seeds: `1`, `2`, `3`
- Epochs: `50`
- Selection strength: `1.0`
- Policy temperature: `1.0`

## Main Readout

The main culprit is the initial neural policy prior. With the default random
network initialization, the policy channel has almost no `p0` slope, so partial
revision slowly replaces the sampled initial condition with near-`0.5` policy
actions. When the policy prior is explicitly matched to `p0`, both policy
probability and sampled cooperation preserve the stag-hunt ordering.

Final mean cooperation by prior:

| Prior | Learning | p0=0.5 | p0=0.6 | p0=0.7 | Final Coop Slope |
| --- | --- | ---: | ---: | ---: | ---: |
| `default` | false | 0.543 | 0.543 | 0.547 | 0.017 |
| `default` | true | 0.527 | 0.527 | 0.533 | 0.033 |
| fixed `0.5` | false | 0.553 | 0.553 | 0.557 | 0.017 |
| fixed `0.5` | true | 0.540 | 0.540 | 0.550 | 0.050 |
| `match_p0` | false | 0.553 | 0.640 | 0.730 | 0.883 |
| `match_p0` | true | 0.540 | 0.637 | 0.740 | 1.000 |

Final mean policy cooperation by prior:

| Prior | Learning | p0=0.5 | p0=0.6 | p0=0.7 | Final Policy Slope |
| --- | --- | ---: | ---: | ---: | ---: |
| `default` | false | 0.502 | 0.502 | 0.502 | 0.000 |
| `default` | true | 0.466 | 0.469 | 0.477 | 0.054 |
| fixed `0.5` | false | 0.500 | 0.500 | 0.500 | 0.000 |
| fixed `0.5` | true | 0.475 | 0.477 | 0.483 | 0.039 |
| `match_p0` | false | 0.500 | 0.600 | 0.700 | 1.000 |
| `match_p0` | true | 0.475 | 0.588 | 0.711 | 1.180 |

## Channel Decomposition

| Candidate Channel | Evidence | Readout |
| --- | --- | --- |
| Initial policy prior | `default` and fixed `0.5` have final policy slope near `0`; `match_p0` has final policy slope near `1.0` without learning. | Main culprit. The neural policy starts nearly independent of `p0`. |
| Sampling/revision channel | Under `match_p0`, sampled cooperation remains ordered through epoch `50`: `0.553`, `0.640`, `0.730` without learning. | Not the primary culprit. Revision noise attenuates but does not erase `p0` when policy carries it. |
| Local policy-gradient update | Under `match_p0`, learning preserves or strengthens the slope: final sampled cooperation slope `1.000`, final policy slope `1.180`. | Not the culprit in this gate. Learning does not collapse the basin signal. |
| Social mixer | Mixer is `none`; `mean_social_loss` is `0.0`. | Out of scope for this gate. |

## Interpretation

The earlier basin failure is best explained as an architecture/initialization
artifact: the random neural policy prior maps the observation stream to a
near-`0.5` cooperation probability regardless of the sampled initial condition.
Because revision rate is `0.1`, the initially sampled `p0` action field is not
erased immediately, but it is gradually overwritten by the policy prior.

The diagnostic also shows that the action sampling channel can carry the basin
signal when the policy prior carries it. This rules out sampling noise as the
dominant explanation for the current failure.

Local policy-gradient learning is not causing slope collapse in this specific
gate. With `match_p0`, learning keeps the ordering and slightly steepens the
policy slope by epoch `50`.

## Next Gate

The next useful intervention is to keep `policy_prior_cooperation_probability`
available as a diagnostic switch and run a broader stag-hunt basin comparison
against Fermi/RD references with:

- `revision_rate`: `0.1`, optionally `0.25`, `1.0`
- `policy_prior`: `default`, `match_p0`
- `learning`: `false`, `true`
- `p0`: a wider basin grid, for example `0.1`, `0.3`, `0.5`, `0.6`, `0.7`, `0.9`

This will tell whether the matched prior merely preserves the input basin signal
or whether the local policy update can reproduce the classical threshold under a
less sparse initial-condition grid.

## Artifacts

- `experiments/results/toy2_stag_hunt_prior_diagnostic_seed01_03_summary.csv`
- `experiments/results/toy2_stag_hunt_prior_diagnostic_seed01_03_grouped_summary.csv`
- `experiments/results/toy2_stag_hunt_prior_diagnostic_seed01_03_grouped_summary.md`
- `experiments/results/toy2_stag_hunt_prior_diagnostic_seed01_03_epoch_means.csv`
- `experiments/results/toy2_stag_hunt_prior_diagnostic_seed01_03_epoch_slopes.csv`

Raw run directories were kept under `/tmp/toy2_stag_hunt_prior_diagnostic_seed01_03_runs`.
