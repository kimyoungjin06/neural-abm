# Toy 2 Game-Dynamics Gate Findings

Date: 2026-04-30

This note promotes Toy 2 from a single neural spatial PD runner check to a
small validation gate against known game-dynamics references.

## Setup

- Grid: `10 x 10` toroidal von Neumann neighborhood.
- Epochs: `50`.
- Seeds: `1, 2, 3, 4, 5` for neural and Fermi ABM runs.
- Agent model: policy MLP `6 -> 16 -> 2`.
- Neural init mode: `independent_init`.
- Social mixers: `none`, `output_average`.
- Baselines: `fermi_imitation` spatial imitation and `rd_well_mixed` aggregate
  reference.
- Regimes: `harsh_pd`, `mild_pd`, `soft_pd`, `snowdrift`, `stag_hunt`.

## Acceptance Check

| Gate | Status | Evidence |
| --- | --- | --- |
| All preset regimes run on seeds `1-5` | Pass | `toy2_regime_sweep_seeds01_05_summary.csv` has 105 rows. |
| `none` and `output_average` are recorded | Pass | Grouped summaries include both mixers for neural and Fermi runs. |
| Outcomes differ by regime/update/mixer | Pass | Cooperation mean ranges differ by `0.142-0.318` across grouped rows per regime. |
| Neural, Fermi, and RD are comparable | Pass | `toy2_neural_vs_fermi_vs_rd.png` plots trajectories on common axes. |
| Logs reconstruct cooperation/payoff/clusters/peers | Pass | Run `aggregate_metrics.csv` and `micro_state.csv` include cooperation, payoff, policy cooperation, cooperation clusters, peer fragmentation, peer IDs, and per-agent payoffs. |

## Main Readout

Final cooperation means from the regime sweep:

| Regime | Neural none | Neural output alpha=0.25 | Fermi none | RD reference |
| --- | ---: | ---: | ---: | ---: |
| `harsh_pd` | 0.142 | 0.080 | 0.000 | 0.000 |
| `mild_pd` | 0.172 | 0.104 | 0.000 | 0.000 |
| `soft_pd` | 0.318 | 0.280 | 0.000 | 0.000 |
| `snowdrift` | 0.436 | 0.454 | 0.162 | 0.429 |
| `stag_hunt` | 0.196 | 0.104 | 0.000 | 0.000 |

Interpretation:

- Toy 2 now separates payoff-regime behavior: neural cooperation rises from
  harsh/mild PD to soft PD and snowdrift.
- Fermi and RD collapse to defection in the PD and stag-hunt settings at the
  current initial cooperation probability and update scale.
- Snowdrift is the useful positive control: RD holds a mixed equilibrium near
  `0.429`, neural policy stays near `0.44-0.45`, and Fermi remains lower and
  seed-sensitive.
- `output_average` does not generally widen the cooperation basin in the PD
  settings. It suppresses cooperation in harsh/mild/soft PD and stag hunt, but
  slightly increases snowdrift cooperation and substantially increases
  snowdrift largest cooperation-cluster fraction.

## Alpha Sweep Readout

The follow-up alpha sweep uses neural policy only:

- Label: `toy2_neural_alpha_sweep_seeds01_05`
- Mixers: `none` once per seed, `output_average` at alpha `0.0, 0.1, 0.25, 0.5`
- Rows: 125 summary rows, 25 grouped rows

Mean final cooperation:

| Regime | None | Output a=0.0 | Output a=0.1 | Output a=0.25 | Output a=0.5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `harsh_pd` | 0.142 | 0.098 | 0.078 | 0.080 | 0.084 |
| `mild_pd` | 0.172 | 0.126 | 0.110 | 0.104 | 0.110 |
| `soft_pd` | 0.318 | 0.282 | 0.258 | 0.280 | 0.290 |
| `snowdrift` | 0.436 | 0.450 | 0.462 | 0.454 | 0.460 |
| `stag_hunt` | 0.196 | 0.144 | 0.102 | 0.104 | 0.096 |

Alpha slope from output alpha `0.0` to `0.5`:

| Regime | Cooperation Delta | Cluster Fraction Delta |
| --- | ---: | ---: |
| `harsh_pd` | -0.014 | 0.002 |
| `mild_pd` | -0.016 | -0.006 |
| `soft_pd` | 0.008 | -0.020 |
| `snowdrift` | 0.010 | 0.108 |
| `stag_hunt` | -0.048 | -0.032 |

Interpretation:

- The current output mixer is not a generic cooperation amplifier.
- In PD-like regimes it acts more like policy homogenization toward lower
  cooperation.
- In snowdrift it preserves mixed cooperation and increases spatial clustering,
  which is the strongest sign that the neural social pipeline is producing a
  regime-dependent nonlinear effect rather than a uniform smoothing artifact.

## Artifacts

- `experiments/results/toy2_regime_sweep_seeds01_05_summary.csv`
- `experiments/results/toy2_regime_sweep_seeds01_05_grouped_summary.csv`
- `experiments/results/toy2_regime_sweep_seeds01_05_grouped_summary.md`
- `experiments/results/toy2_neural_alpha_sweep_seeds01_05_summary.csv`
- `experiments/results/toy2_neural_alpha_sweep_seeds01_05_grouped_summary.csv`
- `experiments/results/toy2_neural_alpha_sweep_seeds01_05_grouped_summary.md`
- `paper/figures/toy2_regime_cooperation_payoff.png`
- `paper/figures/toy2_neural_vs_fermi_vs_rd.png`
- `paper/figures/toy2_alpha_sensitivity.png`

## Basin Sweep Readout

The initial-condition sweep tests whether the current patterns survive when
the starting cooperation probability changes.

- Label: `toy2_basin_sweep_seeds01_05`
- Regimes: `soft_pd`, `snowdrift`, `stag_hunt`
- Initial cooperation probabilities: `0.1, 0.3, 0.5, 0.7, 0.9`
- Neural/Fermi seeds: `1-5`
- RD reference: one run per regime and initial probability
- Rows: 315 summary rows, 75 grouped rows

Mean final cooperation:

| Regime | Condition | p0=0.1 | p0=0.3 | p0=0.5 | p0=0.7 | p0=0.9 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `soft_pd` | Neural none | 0.328 | 0.324 | 0.318 | 0.328 | 0.322 |
| `soft_pd` | Neural output | 0.284 | 0.280 | 0.280 | 0.284 | 0.278 |
| `soft_pd` | Fermi none | 0.000 | 0.000 | 0.000 | 0.068 | 0.518 |
| `soft_pd` | RD | 0.000 | 0.000 | 0.000 | 0.003 | 0.146 |
| `snowdrift` | Neural none | 0.430 | 0.438 | 0.436 | 0.444 | 0.448 |
| `snowdrift` | Neural output | 0.454 | 0.450 | 0.454 | 0.464 | 0.452 |
| `snowdrift` | Fermi none | 0.094 | 0.158 | 0.162 | 0.300 | 0.486 |
| `snowdrift` | RD | 0.429 | 0.429 | 0.429 | 0.429 | 0.429 |
| `stag_hunt` | Neural none | 0.196 | 0.198 | 0.196 | 0.206 | 0.212 |
| `stag_hunt` | Neural output | 0.108 | 0.104 | 0.104 | 0.110 | 0.106 |
| `stag_hunt` | Fermi none | 0.000 | 0.000 | 0.000 | 0.038 | 0.546 |
| `stag_hunt` | RD | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 |

Interpretation:

- `snowdrift` is the cleanest validation point. RD converges to the same mixed
  equilibrium from every p0, and neural policy tracks that basin closely while
  output mixing increases spatial cluster fraction.
- `stag_hunt` exposes a sharp classical basin threshold: RD flips from
  defection to cooperation between p0 `0.5` and `0.7`, while Fermi shows
  high-variance cooperation only at p0 `0.9`.
- Neural policy does not reproduce the stag-hunt all-or-nothing basin. It stays
  near `0.20` without social mixing and near `0.10` with output mixing across
  all tested p0 values. Treat this as a neural-policy bias or learner-lag
  finding, not as evidence of classical stag-hunt recovery.
- `soft_pd` shows a similar split: Fermi and RD require very high p0 to retain
  cooperation, while neural policy maintains a stable partial-cooperation band
  across the whole p0 sweep.

Additional artifacts:

- `experiments/results/toy2_basin_sweep_seeds01_05_summary.csv`
- `experiments/results/toy2_basin_sweep_seeds01_05_grouped_summary.csv`
- `experiments/results/toy2_basin_sweep_seeds01_05_grouped_summary.md`
- `paper/figures/toy2_basin_sensitivity.png`

## Next Experiment

The next focused test should diagnose why neural policy smooths or misses
classical basin thresholds:

1. Sweep `selection_strength` and `policy_temperature` for `stag_hunt`.
2. Keep p0 values near the RD threshold: `0.5, 0.6, 0.7`.
3. Compare realized cooperation to policy cooperation to see whether sampling,
   policy logits, or local policy updates are damping the transition.
4. Add a no-learning neural policy control to separate architecture bias from
   policy-gradient adaptation.
