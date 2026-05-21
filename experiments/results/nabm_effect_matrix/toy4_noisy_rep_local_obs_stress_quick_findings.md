# Toy4 Noisy Reputation Local Observation Stress Findings

Date: 2026-05-21

## Question

Does Toy4 local resource-threshold recovery remain stable when the
reputation-imitation baseline is stressed with noisy reputation, and does
`local_sustain` observation add value over global or hidden resource
observation?

This combines two previous stress axes:

- resource-coupled Toy4 with initial contribution probability `0.35`
- reputation noise `2.0` for the noisy baseline and neural comparison variants
- local-threshold neural variants with `global`, `hidden`, and `local_sustain`
  resource observation modes

The manifest uses short variant labels to avoid filesystem path-length issues
in generated run directories. The meaning of each variant is fixed by the
manifest updates, not by the shortened label.

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_threshold_noisy_reputation_local_observation_stress_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_noisy_rep_local_obs_stress_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_noisy_rep_local_obs_stress_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_noisy_rep_local_obs_stress_quick_profile.json`

## Result

The gate passed. Clean reputation imitation remains faster, but the noisy
reputation diagnostic degrades to `3/5` final ceiling hits. All three
local-threshold neural observation variants reach and hold the ceiling across
all five seeds.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `rep_clean` | baseline | 5/5 | 5/5 | 15.0 | 0.600000 | 1.000 |
| `rep_noisy_s2p0` | diagnostic | 3/5 | 3/5 | 47.333 | 0.493425 | 0.600 |
| `rev_pop_global_obs_noisy_s2p0` | diagnostic | 0/5 | 0/5 |  | -0.298000 | 0.000 |
| `rev_local_global_obs_noisy_s2p0` | diagnostic | 5/5 | 5/5 | 30.8 | 0.600000 | 1.000 |
| `rev_local_hidden_obs_noisy_s2p0` | diagnostic | 5/5 | 5/5 | 30.4 | 0.600000 | 1.000 |
| `rev_local_sustain_obs_noisy_s2p0` | main | 5/5 | 5/5 | 30.0 | 0.600000 | 1.000 |

Seed-level noisy baseline failures are not late-flip failures after reaching
the ceiling:

| Variant | Seed | Final ceiling | Ever ceiling | TtC | Final payoff | Terminal ceiling rate |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| `rep_noisy_s2p0` | 1 | true | true | 52 | 0.600000 | 1.000 |
| `rep_noisy_s2p0` | 2 | false | false |  | 0.504000 | 0.000 |
| `rep_noisy_s2p0` | 3 | true | true | 53 | 0.600000 | 1.000 |
| `rep_noisy_s2p0` | 4 | false | false |  | 0.163123 | 0.000 |
| `rep_noisy_s2p0` | 5 | true | true | 37 | 0.600000 | 1.000 |

## Interpretation

This is positive evidence for the local-threshold structural direction, but it
should not be read as a clean observation-mode win.

The main contrast is robust:

1. Clean reputation imitation is still best matched to the clean Toy4 setting
   and reaches the ceiling faster.
2. When reputation ordering is noisy, reputation imitation drops to `3/5`.
3. Local-threshold neural recovery keeps `5/5` ceiling stability under the same
   noisy-reputation stress flag.
4. Population thresholding remains a negative control at `0/5`.

The observation-mode result is narrower. `local_sustain` is the fastest of the
three local-threshold observation variants in this five-seed slice
(`30.0` vs `30.4` vs `30.8` mean TtC), but `global` and `hidden` also pass at
`5/5`. That means direct global resource visibility is not required, and local
sustain information is aligned with the mechanism, but the decisive mechanism
is still local thresholding plus precommitment/peer evidence rather than the
resource observation feature alone.

## Decision

Keep `local_sustain` as the main local-observation candidate for future Toy4
resource stresses because it is domain-native and slightly improves TtC here.
Do not claim it is necessary yet.

The stronger claim supported by this run is:

- hand-coded reputation imitation is fast when its ranking signal is clean;
- the local-threshold neural path is slower but more stable when that ranking
  signal is noisy;
- the population-threshold negative control confirms this is not generic
  environment weighting.

The next useful stress is therefore not more observation-mode tuning. It should
combine this local-threshold path with harder structural resource heterogeneity
or holdout-style perturbations where local sustain thresholds differ across
space.

## Verification

- `uv run ruff check src/neural_abm/config.py src/neural_abm/toy_public_goods.py tests/test_toy4_runner.py tests/test_evidence_gate.py`
- `uv run pytest tests/test_toy4_runner.py tests/test_evidence_gate.py -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_threshold_noisy_reputation_local_observation_stress_quick.yaml`
