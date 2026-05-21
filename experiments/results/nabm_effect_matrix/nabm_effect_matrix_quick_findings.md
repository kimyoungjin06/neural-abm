# NABM Effect Matrix Quick Findings

This report summarizes `nabm_effect_matrix_quick` from the default manifest at
`experiments/evidence/nabm_effect_matrix_quick.yaml`.

Run scope:

- Seeds: 1, 2, 3.
- Epochs: 5.
- Run rows: `nabm_effect_matrix_quick_runs.csv`.
- Grouped effects: `nabm_effect_matrix_quick_effects.csv`.
- Pairwise effects: `nabm_effect_matrix_quick_pairwise_effects.csv`.
- Markdown table: `nabm_effect_matrix_quick_effects.md`.

Positive effect values favor the NABM group. This is a quick evidence check,
not a paper-candidate result.

## Toy-Level Verdicts

| Toy | Verdict | Main Evidence | Claim Implication |
| --- | --- | --- | --- |
| Toy1 | Green | Output averaging improves final mean global accuracy by `+0.0521` over no-social, 95% CI `0.0333`. | The supervised/social-learning claim is directionally supported in the quick matrix. |
| Toy2 | Red for broad reference claim | Grouped effect is `-0.0800`. Neural output averaging beats Fermi (`+0.3883`) and RD well-mixed (`+0.3551`), but loses to reputation imitation (`-0.9833`). | Do not claim neural policy dominance over all Toy2 reference policies. The reputation baseline is a stronger mechanism in this quick setting. |
| Toy3 | Green | Neural output averaging reduces polarization versus Deffuant (`+0.1590`) and HK (`+0.1597`) with tight CIs. | The opinion/social-output mechanism is strongly supported in this quick matrix. |
| Toy4 | Red for broad reference claim | Grouped effect is `-0.0720`. Neural output averaging beats imitation (`+0.2280`) but loses to reputation imitation (`-0.3720`). | Do not claim broad Toy4 reference dominance. The reputation-imitation path needs mechanism diagnosis or a narrower claim. |
| Toy5 | Green within planned reference scope | Neural output averaging beats complex threshold (`+93.3333`) and simple contagion (`+92.3333`) on cascade size. | The threshold/contagion comparison supports the Toy5 NABM effect in this quick matrix. |

## Overall Assessment

The structural completion work is in good shape: shared runner boundaries,
stable artifact contracts, and repeatable matrix outputs are now in place.

The evidence completion is mixed:

- Toy1, Toy3, and Toy5 are ready for a larger paper-candidate matrix.
- Toy2 and Toy4 should not be promoted on a broad "beats reference policies"
  claim because reputation-imitation references outperform the neural path in
  this quick run.
- The pairwise effect table is necessary for interpretation. Grouped baseline
  means hide which reference policy is actually carrying or defeating the
  comparison.

Toy6-10 should remain `compatible` and not evidence-default. Their runner reuse
improves toolkit completeness but does not upgrade their NABM claim status.

## Recommended Next Step

Run an expanded matrix only for the green cases and diagnose Toy2/Toy4
separately:

- Expand Toy1, Toy3, and Toy5 to more seeds and longer epochs.
- For Toy2 and Toy4, compare neural policy against reputation-imitation with
  mechanism traces before tuning alpha or thresholds.
- Keep the claim language conservative until Toy2/Toy4 have a defensible
  reference-specific result.
