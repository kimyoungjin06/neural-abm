# Toy4 Resource Threshold Noisy Reputation Stress Findings

Date: 2026-05-21

## Question

Does the local resource-threshold neural path become more valuable when the
hand-coded reputation-imitation baseline is made less reliable?

This stress preserves the resource-coupled Toy4 setting from the hardening run:

- initial contribution probability: `0.35`
- resource initial/capacity: `60/100`
- recovery/extraction: `0.03/0.05`

The clean reputation-imitation run is kept as the baseline. Noisy
reputation-imitation variants are diagnostic because they intentionally damage
the hand-coded reputation signal. The local threshold neural path is the main
claim group.

## Artifacts

- Manifest:
  `experiments/evidence/toy4_resource_threshold_noisy_reputation_stress_quick.yaml`
- Run rows:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_noisy_reputation_stress_quick_runs.csv`
- Gate summary:
  `experiments/evidence/results/toy4_resource_threshold_noisy_reputation_stress_quick.summary.json`
- Profile:
  `experiments/results/nabm_effect_matrix/toy4_resource_threshold_noisy_reputation_stress_quick_profile.json`

## Result

The gate passed. Clean reputation imitation is still the fastest path, but it
becomes fragile under reputation noise. The local resource-threshold neural path
keeps five-seed ceiling stability under the same noisy-reputation stress flag.

| Variant | Group | Final ceiling hits | Ever ceiling hits | Mean TtC | Mean final payoff | Terminal ceiling rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `reputation_imitation_resource_p0p35_clean` | baseline | 5/5 | 5/5 | 15.0 | 0.600000 | 1.000 |
| `reputation_imitation_resource_p0p35_noisy_s1p0` | diagnostic | 4/5 | 4/5 | 43.75 | 0.595200 | 0.720 |
| `reputation_imitation_resource_p0p35_noisy_s2p0` | diagnostic | 3/5 | 3/5 | 47.333 | 0.493425 | 0.600 |
| `revision_precommitment_peer_evidence_resource_threshold_population_envw2p0` | diagnostic | 0/5 | 0/5 |  | -0.298000 | 0.000 |
| `revision_precommitment_peer_evidence_resource_threshold_local_envw2p0_reputation_noise_s2p0` | main | 5/5 | 5/5 | 30.8 | 0.600000 | 1.000 |

## Mechanism

The noisy reputation baseline does not fail by late flips after reaching the
ceiling. It often climbs too slowly or never reaches the ceiling inside the
60-epoch window. At `noise=2.0`, seed 4 illustrates the failure: action rate
eventually reaches `0.96`, but resource recovery lags and final payoff remains
`0.122`.

Seed-4 contrast:

| Variant | Epoch | Action rate | Resource fraction | Mean reputation | Policy/action prob | High-policy rate | Ready component |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| noisy reputation s2.0 | 9 | 0.230 | 0.303 | 0.231 | 0.223 | 0.000 | 0.000 |
| noisy reputation s2.0 | 24 | 0.720 | 0.165 | 0.511 | 0.725 | 0.000 | 0.000 |
| noisy reputation s2.0 | 39 | 0.800 | 0.344 | 0.719 | 0.749 | 0.000 | 0.000 |
| noisy reputation s2.0 | 54 | 0.880 | 0.588 | 0.819 | 0.862 | 0.000 | 0.000 |
| noisy reputation s2.0 | 59 | 0.960 | 0.704 | 0.860 | 0.933 | 0.000 | 0.000 |
| local threshold neural | 9 | 0.660 | 0.579 | 0.472 | 0.685 | 0.190 | 0.010 |
| local threshold neural | 19 | 0.820 | 0.675 | 0.655 | 0.655 | 0.120 | 0.550 |
| local threshold neural | 24 | 1.000 | 0.807 | 0.780 | 0.568 | 0.000 | 1.000 |
| local threshold neural | 34 | 1.000 | 1.000 | 0.923 | 0.744 | 0.440 | 1.000 |
| local threshold neural | 59 | 1.000 | 1.000 | 0.994 | 0.990 | 1.000 | 1.000 |

This is the useful distinction: reputation imitation remains excellent when its
reputation ordering is clean, but noisy reputation delays the coordinated
resource recovery. The local threshold neural path is slower than the clean
baseline, yet more stable than noisy reputation imitation because its recovery
signal is tied to local sustain-threshold continuation rather than peer
reputation ranking.

## Decision

This is positive evidence for the current structural direction, but not a
dominance claim:

1. Clean reputation imitation is still the best matched rule for the clean Toy4
   environment.
2. The local threshold neural path becomes competitive when the baseline's
   reputation signal is noisy.
3. Population thresholding remains a negative control, so the improvement is
   still local-threshold specific rather than generic environment weighting.

The next stress should move beyond noisy reputation and test whether the same
local-threshold mechanism handles heterogeneous sustain thresholds or sparse
resource observations. Those are stronger tests of whether the structural hook
generalizes beyond damaging the hand-coded baseline's input signal.

## Verification

- `uv run ruff check tests/test_evidence_gate.py`
- `uv run pytest tests/test_evidence_gate.py -q`
- `uv run python scripts/run_basin_credit_evidence_workflow.py --manifest experiments/evidence/toy4_resource_threshold_noisy_reputation_stress_quick.yaml`
