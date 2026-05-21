# Toy2/Toy4 Basin-Credit Replay Findings

Manifest:
`experiments/evidence/toy24_basin_credit_replay_quick.yaml`

Generated artifacts:
- Runs: `experiments/results/nabm_effect_matrix/toy24_basin_credit_replay_quick_runs.csv`
- Effects: `experiments/results/nabm_effect_matrix/toy24_basin_credit_replay_quick_effects.csv`
- Pairwise effects: `experiments/results/nabm_effect_matrix/toy24_basin_credit_replay_quick_pairwise_effects.csv`
- Gate summary: `experiments/evidence/results/toy24_basin_credit_replay_quick.summary.json`

## Gate Result

Overall status: pass.

The replay variants convert the previous objective+basin recovery into a
time-to-ceiling pass:

- Toy2: `mixed_objective_basin_replay_all_p3_h1` reached 3/3 final ceiling
  hits with mean time-to-ceiling 9.33, below the <10 threshold.
- Toy4: `mixed_objective_basin_replay_all_p3_h1` reached 2/3 final ceiling
  hits with mean time-to-ceiling 11.67, below the <12 threshold.
- Toy4 p2 reached 3/3 final hits and mean payoff 0.6000, but its mean
  time-to-ceiling was 13.67, so p3 is the passing speed candidate.

## Variant Comparison

| Toy | Variant | Final hits | Mean TTC | Mean payoff | Action rate | Training passes | Training effective advantage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 | `basin_credit_w1p0_h1_prototype` | 0/3 | - | 2.5567 | 0.6600 | 1 | 0.0031 |
| Toy2 | `mixed_individual_basin_w0p5_0p5_h1` | 0/3 | - | 1.0600 | 0.0200 | 1 | -0.0969 |
| Toy2 | `mixed_objective_basin_w0p5_0p5_h1` | 3/3 | 22.67 | 3.0000 | 1.0000 | 1 | 0.2252 |
| Toy2 | `mixed_objective_basin_replay_all_p2_h1` | 3/3 | 10.33 | 3.0000 | 1.0000 | 2 | 0.2252 |
| Toy2 | `mixed_objective_basin_replay_all_p3_h1` | 3/3 | 9.33 | 3.0000 | 1.0000 | 3 | 0.2252 |
| Toy4 | `basin_credit_w1p0_h1_prototype` | 0/3 | - | 0.3140 | 0.5233 | 1 | 0.0021 |
| Toy4 | `mixed_individual_basin_w0p5_0p5_h1` | 0/3 | - | 0.0040 | 0.0067 | 1 | -0.2076 |
| Toy4 | `mixed_objective_basin_w0p5_0p5_h1` | 2/3 | 16.33 | 0.5980 | 0.9967 | 1 | 0.1879 |
| Toy4 | `mixed_objective_basin_replay_all_p2_h1` | 3/3 | 13.67 | 0.6000 | 1.0000 | 2 | 0.1880 |
| Toy4 | `mixed_objective_basin_replay_all_p3_h1` | 2/3 | 11.67 | 0.5980 | 0.9967 | 3 | 0.1879 |

## Interpretation

The replay mechanism addresses the specific failure from
`toy24_basin_credit_objective_blend_quick`: objective+basin already had the
right direction, but reached the ceiling too late. Multi-pass replay reduces
Toy2 mean time-to-ceiling from 22.67 to 9.33 and Toy4 from 16.33 to 11.67.

This is not just a material-weight rescue. The material+basin diagnostic still
collapses in both toys, with near-zero final action rates and negative training
effective advantage. The passing replay candidates preserve the objective
blend's high-action basin behavior while applying more basin-driven policy
updates per epoch.

The evidence supports `training_passes=3` as the current quick-gate candidate.
`training_scope=all` is active in the replay variants, but the quick baseline
configs use `revision_rate=1.0`, so the observed speed gain in this manifest is
primarily attributable to repeated basin replay passes rather than a wider
candidate mask.
