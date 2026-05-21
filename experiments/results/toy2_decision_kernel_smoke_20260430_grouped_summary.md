# Toy 2 Game-Regime Sweep: toy2_decision_kernel_smoke_20260430

| Regime | Update Rule | Mixer | Learning | Local Update | Neural Peer | Interaction | Decision | Action Temp | Init Coop | Policy Prior | Revision | Alpha | Selection | Policy Temp | Explore | Seeds | Coop Mean | Payoff Mean | Policy Coop Mean | Cluster Fraction Mean | Active Peer Components Mean |
| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `stag_hunt` | `neural_policy` | `none` | `True` | `counterfactual_advantage` | `spatial` | `spatial` | `sampled` | 0.5 | 0.6 | `match_p0` | 0.25 | 0 | 1 | 1 | 0 | 3 | 0.496667 | 2.493333 | 0.467918 | 0.406667 | 100.00 |
| `stag_hunt` | `neural_policy` | `none` | `True` | `counterfactual_advantage` | `spatial` | `spatial` | `sampled` | 1 | 0.6 | `match_p0` | 0.25 | 0 | 1 | 1 | 0 | 3 | 0.370000 | 2.130000 | 0.336109 | 0.186667 | 100.00 |
| `stag_hunt` | `neural_policy` | `none` | `True` | `counterfactual_advantage` | `spatial` | `spatial` | `sampled` | 0.5 | 0.7 | `match_p0` | 0.25 | 0 | 1 | 1 | 0 | 3 | 0.956667 | 3.813333 | 0.923933 | 0.956667 | 100.00 |
| `stag_hunt` | `neural_policy` | `none` | `True` | `counterfactual_advantage` | `spatial` | `spatial` | `sampled` | 1 | 0.7 | `match_p0` | 0.25 | 0 | 1 | 1 | 0 | 3 | 0.670000 | 2.810000 | 0.656387 | 0.670000 | 100.00 |
| `stag_hunt` | `rd_well_mixed` | `none` | `True` | `sampled_policy_gradient` | `spatial` | `spatial` | `sampled` | 1 | 0.6 | `default` | 0.25 | 0 | 1 | 1 | 0 | 1 | 0.000002 | 1.999996 | 0.000002 | 0.000000 | 0.00 |
| `stag_hunt` | `rd_well_mixed` | `none` | `True` | `sampled_policy_gradient` | `spatial` | `spatial` | `sampled` | 1 | 0.7 | `default` | 0.25 | 0 | 1 | 1 | 0 | 1 | 0.996583 | 3.977417 | 0.996583 | 0.000000 | 0.00 |

## Readout

This table is the Toy 2 validation gate summary across payoff regimes, neural policy dynamics, Fermi spatial imitation, and one RD well-mixed reference per regime.