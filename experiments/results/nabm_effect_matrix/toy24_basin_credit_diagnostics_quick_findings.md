# Toy2/4 Basin-Credit Diagnostic Findings

This report summarizes `toy24_basin_credit_diagnostics_quick`, a rerun of the
Toy2/Toy4 basin-credit matrix with additional training-signal diagnostics.

Run scope:

- Manifest: `experiments/evidence/toy24_basin_credit_diagnostics_quick.yaml`.
- Seeds: 1, 2, 3.
- Epochs: 50.
- Run rows: `toy24_basin_credit_diagnostics_quick_runs.csv`.
- Gate summary: `experiments/evidence/results/toy24_basin_credit_diagnostics_quick.summary.json`.

## Gate Verdict

Overall gate status: **fail**.

Input validation remained clean and the case-level results matched the previous
hardened evidence run: both eligible main variants had zero final ceiling hits
in both Toy2 and Toy4.

## Diagnostic Signal

| Case | Variant | Final Ceiling Hits | Final Payoff Mean | Final Action Rate | Basin Action1 Advantage | Training Effective Advantage | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `toy2_basin_credit` | `basin_credit_w1p0_h1_prototype` | 0/3 | 2.5567 | 0.6600 | 0.0031 | 0.0031 | Direction is mostly positive, but the training signal is tiny. |
| `toy2_basin_credit` | `mixed_individual_basin_w0p5_0p5_h1` | 0/3 | 1.0600 | 0.0200 | 0.0102 | -0.0969 | Basin signal is positive, but the mixed training signal is negative. |
| `toy4_basin_credit` | `basin_credit_w1p0_h1_prototype` | 0/3 | 0.3140 | 0.5233 | 0.0021 | 0.0021 | Direction is weak and sometimes changes sign during the run. |
| `toy4_basin_credit` | `mixed_individual_basin_w0p5_0p5_h1` | 0/3 | 0.0040 | 0.0067 | 0.0098 | -0.2076 | Basin signal is positive, but material incentive dominates negatively. |

`Basin Action1 Advantage` is the action-1-vs-action-0 signed basin-credit
signal derived from the one-step ablation. `Training Effective Advantage` is
the actual signal passed into the post-social policy update after basin-credit
component blending.

## Trajectory Pattern

Pure basin-credit prototype:

- Toy2 action rate moved from `0.53` at epoch 1 to `0.66` at epoch 50, but the
  final training effective advantage was only about `0.0031`.
- Toy4 action rate stayed near the middle, ending at `0.5233`; the final
  training effective advantage was about `0.0021`.
- This supports the weak-signal diagnosis: the prototype scorer is directionally
  aligned late in the run, but the magnitude is too small to reproduce the
  linear welfare convergence path.

Mixed individual+basin variant:

- Toy2 action rate collapsed from `0.53` at epoch 1 to `0.02` at epoch 50.
- Toy4 action rate collapsed from `0.4967` at epoch 1 to `0.0067` at epoch 50.
- The basin action1 advantage was positive at the end in both toys, but the
  actual blended training effective advantage stayed negative: about `-0.0969`
  for Toy2 and `-0.2076` for Toy4.
- This supports the blend-design diagnosis: the configured `individual_weight`
  mixes in `components.material`, not the welfare-heavy objective effective
  signal, so the individual material incentive overwhelms the basin term.

## Research Implication

The diagnostic run does not justify moving directly to a learned contrastive
critic. Two lower-level issues should be resolved first:

- The prototype basin scorer provides only very small action-1 advantages in
  the pure basin-credit setting.
- The mixed variant is not a mixed welfare-objective-plus-basin variant; it is
  material-plus-basin, and that explains the observed collapse.

The next implementation step should be a targeted diagnostic or design patch
for component blending semantics, not another threshold/source sweep.
