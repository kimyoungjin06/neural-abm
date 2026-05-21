# Time-to-Ceiling Bottleneck Diagnostics

Input runs: `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_runs.csv`
Early window: epochs 1-10 before ceiling
Min delta: `0.0001`
Decision gap threshold: `0.05`

## Group Summary

| Case | Variant | Group | Runs | Final Hits | Mean TtC | Top Bottlenecks | Local Delta | Social Delta | Decision Gap | PostSocial >0.7 | Dwell 0.4-0.6 | Cross 0.5 | Flip Rate | Training Adv | Training Positive Rate |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| toy2_basin_credit | basin_credit_w1p0_h1_prototype | nabm | 3 | 0 |  | no_final_ceiling_hit:3, ambivalence_dwell:3, threshold_crossing_activity:3 | 0.00158965 | 0.00216796 | 0.0074838 | 0.001 | 0.973333 | 0.177 | 0.508 | 0.0021044 | 0.65 |
| toy2_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | 3 | 3 | 22.6667 | threshold_crossing_activity:3, slow_after_positive_signal:3 | 0.0202121 | 0.0166888 | -0.0626715 | 0.552 | 0.211667 | 0.0503333 | 0.344 |  |  |
| toy2_basin_credit | linear_welfare_heavy | baseline | 3 | 3 | 23.6667 | threshold_crossing_activity:3, slow_after_positive_signal:3 | 0.0191261 | 0.0158766 | 0.0284787 | 0.527 | 0.230333 | 0.051 | 0.4 |  |  |
| toy2_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | diagnostic | 3 | 0 |  | no_final_ceiling_hit:3, weak_or_missing_credit_signal:3 | -0.0116582 | -0.0206035 | -0.0283514 | 0.000333333 | 0.261 | 0.0616667 | 0.453333 | -0.134834 | 0 |
| toy2_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | nabm | 3 | 3 | 16 | slow_after_positive_signal:3, threshold_crossing_activity:1 | 0.0123801 | 0.028002 | 0.0355915 | 0.552333 | 0.238 | 0.0556667 | 0.385 | 0.319588 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | nabm | 3 | 3 | 21.3333 | slow_after_positive_signal:3, threshold_crossing_activity:2 | 0.0123832 | 0.0277806 | 0.034598 | 0.555333 | 0.238 | 0.0563333 | 0.389667 | 0.319485 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | nabm | 3 | 3 | 12 | slow_after_positive_signal:2, threshold_crossing_activity:2 | 0.0124404 | 0.0289033 | 0.000886686 | 0.536963 | 0.245704 | 0.0573704 | 0.344556 | 0.310283 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | nabm | 3 | 3 | 12 | slow_after_positive_signal:2, threshold_crossing_activity:2 | 0.0124756 | 0.0290058 | 0.00138771 | 0.537704 | 0.245333 | 0.0573704 | 0.343889 | 0.310298 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | nabm | 3 | 3 | 12 | slow_after_positive_signal:2, threshold_crossing_activity:2 | 0.0124381 | 0.0288999 | 0.00117978 | 0.536963 | 0.245704 | 0.0573704 | 0.344556 | 0.310399 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | nabm | 3 | 3 | 18.3333 | slow_after_positive_signal:3, threshold_crossing_activity:1 | 0.0123801 | 0.0278795 | 0.0371357 | 0.552333 | 0.238 | 0.0556667 | 0.386667 | 0.320092 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | nabm | 3 | 3 | 18.3333 | slow_after_positive_signal:3, threshold_crossing_activity:1 | 0.0123801 | 0.0278795 | 0.0371357 | 0.552333 | 0.238 | 0.0556667 | 0.386667 | 0.320092 | 1 |
| toy2_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | nabm | 3 | 3 | 18.3333 | slow_after_positive_signal:3, threshold_crossing_activity:1 | 0.0123801 | 0.0278795 | 0.0371357 | 0.552333 | 0.238 | 0.0556667 | 0.386667 | 0.320092 | 1 |
| toy2_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | nabm | 3 | 3 | 22.6667 | threshold_crossing_activity:3, slow_after_positive_signal:3 | 0.0122859 | 0.0270648 | 0.0347967 | 0.541333 | 0.244 | 0.057 | 0.396667 | 0.321624 | 1 |
| toy2_basin_credit | reputation_imitation | baseline | 3 | 3 | 2.66667 | weak_or_missing_credit_signal:3, threshold_crossing_activity:3 | 0.326667 | 0 | 0.00166667 | 0.956667 | 0 | 0.37 | 0.365 |  |  |
| toy4_basin_credit | basin_credit_w1p0_h1_prototype | nabm | 3 | 0 |  | no_final_ceiling_hit:3, ambivalence_dwell:3, weak_or_missing_credit_signal:2 | 0.000392629 | 0.000751639 | -8.54291e-05 | 0.000333333 | 0.903667 | 0.168333 | 0.510333 | 0.000552463 | 0.533333 |
| toy4_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | diagnostic | 3 | 1 | 19.6667 | threshold_crossing_activity:3, slow_after_positive_signal:3, no_final_ceiling_hit:2 | 0.0214552 | 0.0174418 | -0.0593327 | 0.619333 | 0.172333 | 0.0486667 | 0.336667 |  |  |
| toy4_basin_credit | linear_welfare_heavy | baseline | 3 | 1 | 21 | slow_after_positive_signal:3, no_final_ceiling_hit:2, threshold_crossing_activity:2 | 0.0208016 | 0.0170373 | 0.0373822 | 0.599 | 0.184 | 0.0493333 | 0.400667 |  |  |
| toy4_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | diagnostic | 3 | 0 |  | no_final_ceiling_hit:3, weak_or_missing_credit_signal:3 | -0.0131859 | -0.0241005 | -0.042358 | 0 | 0.22 | 0.0633333 | 0.437667 | -0.21114 | 0 |
| toy4_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | nabm | 3 | 3 | 13.6667 | slow_after_positive_signal:3, threshold_crossing_activity:2 | 0.0132834 | 0.0291772 | 0.041767 | 0.598 | 0.214667 | 0.0586667 | 0.391667 | 0.188179 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | nabm | 3 | 2 | 18 | slow_after_positive_signal:3, no_final_ceiling_hit:1, decision_action_lag:1 | 0.0132899 | 0.0289744 | 0.0458344 | 0.599333 | 0.208667 | 0.0593333 | 0.401333 | 0.188198 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | nabm | 3 | 3 | 11.3333 | threshold_crossing_activity:2, slow_after_positive_signal:2 | 0.0133158 | 0.030006 | -0.00331194 | 0.584889 | 0.222963 | 0.0611481 | 0.33837 | 0.188044 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | nabm | 3 | 3 | 11 | threshold_crossing_activity:3, slow_after_positive_signal:2 | 0.0133485 | 0.0300992 | -0.00476994 | 0.587222 | 0.222593 | 0.0611481 | 0.33437 | 0.18804 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | nabm | 3 | 3 | 11.3333 | threshold_crossing_activity:2, slow_after_positive_signal:2 | 0.0133158 | 0.030006 | -0.00331194 | 0.584889 | 0.222963 | 0.0611481 | 0.33837 | 0.188044 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | nabm | 3 | 1 | 18 | slow_after_positive_signal:3, no_final_ceiling_hit:2, threshold_crossing_activity:2 | 0.0132903 | 0.0291189 | 0.0443205 | 0.597667 | 0.214667 | 0.0586667 | 0.395667 | 0.188185 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | nabm | 3 | 1 | 18 | slow_after_positive_signal:3, no_final_ceiling_hit:2, threshold_crossing_activity:2 | 0.0132903 | 0.0291189 | 0.0443205 | 0.597667 | 0.214667 | 0.0586667 | 0.395667 | 0.188185 | 1 |
| toy4_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | nabm | 3 | 1 | 18 | slow_after_positive_signal:3, no_final_ceiling_hit:2, threshold_crossing_activity:2 | 0.0132903 | 0.0291189 | 0.0443205 | 0.597667 | 0.214667 | 0.0586667 | 0.395667 | 0.188185 | 1 |
| toy4_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | nabm | 3 | 2 | 16.3333 | slow_after_positive_signal:3, threshold_crossing_activity:2, no_final_ceiling_hit:1 | 0.0131308 | 0.0281207 | 0.0447572 | 0.580333 | 0.215333 | 0.06 | 0.409667 | 0.188281 | 1 |
| toy4_basin_credit | reputation_imitation | baseline | 3 | 3 | 2.66667 | weak_or_missing_credit_signal:3, threshold_crossing_activity:3 | 0.326667 | 0 | 0.00166667 | 0.956667 | 0 | 0.37 | 0.365 |  |  |

## Run Details

| Case | Variant | Seed | TtC | Final Hit | Bottlenecks | First Credit | First Local | First Action |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | ---: |
| toy2_basin_credit | basin_credit_w1p0_h1_prototype | 1 |  | False | no_final_ceiling_hit;ambivalence_dwell;threshold_crossing_activity;slow_or_partial_commitment | 1 | 2 | 1 |
| toy2_basin_credit | basin_credit_w1p0_h1_prototype | 2 |  | False | no_final_ceiling_hit;ambivalence_dwell;threshold_crossing_activity;slow_or_partial_commitment | 1 | 2 | 1 |
| toy2_basin_credit | basin_credit_w1p0_h1_prototype | 3 |  | False | no_final_ceiling_hit;ambivalence_dwell;threshold_crossing_activity;slow_or_partial_commitment | 1 | 2 | 1 |
| toy2_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | 1 | 20 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy2_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | 2 | 26 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy2_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | 3 | 22 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy2_basin_credit | linear_welfare_heavy | 1 | 23 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy2_basin_credit | linear_welfare_heavy | 2 | 26 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy2_basin_credit | linear_welfare_heavy | 3 | 22 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy2_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | 1 |  | False | no_final_ceiling_hit;weak_or_missing_credit_signal | 1 |  | 1 |
| toy2_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | 2 |  | False | no_final_ceiling_hit;weak_or_missing_credit_signal | 1 |  | 1 |
| toy2_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | 3 |  | False | no_final_ceiling_hit;weak_or_missing_credit_signal | 1 |  | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | 1 | 17 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | 2 | 16 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | 3 | 15 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | 1 | 20 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | 2 | 26 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | 3 | 18 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | 1 | 13 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | 2 | 10 | True | threshold_crossing_activity | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | 3 | 13 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 1 | 13 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 2 | 10 | True | threshold_crossing_activity | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 3 | 13 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 1 | 13 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 2 | 10 | True | threshold_crossing_activity | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 3 | 13 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | 1 | 20 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | 2 | 17 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | 3 | 18 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | 1 | 20 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | 2 | 17 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | 3 | 18 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | 1 | 20 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | 2 | 17 | True | slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | 3 | 18 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | 1 | 20 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | 2 | 26 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | 3 | 22 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy2_basin_credit | reputation_imitation | 1 | 3 | True | weak_or_missing_credit_signal;threshold_crossing_activity |  | 1 | 1 |
| toy2_basin_credit | reputation_imitation | 2 | 3 | True | weak_or_missing_credit_signal;threshold_crossing_activity |  | 1 | 1 |
| toy2_basin_credit | reputation_imitation | 3 | 2 | True | weak_or_missing_credit_signal;threshold_crossing_activity |  | 1 | 1 |
| toy4_basin_credit | basin_credit_w1p0_h1_prototype | 1 |  | False | no_final_ceiling_hit;ambivalence_dwell;slow_or_partial_commitment | 1 | 2 | 1 |
| toy4_basin_credit | basin_credit_w1p0_h1_prototype | 2 |  | False | no_final_ceiling_hit;weak_or_missing_credit_signal;ambivalence_dwell;threshold_crossing_activity | 1 | 3 | 2 |
| toy4_basin_credit | basin_credit_w1p0_h1_prototype | 3 |  | False | no_final_ceiling_hit;weak_or_missing_credit_signal;ambivalence_dwell;threshold_crossing_activity | 1 | 4 | 1 |
| toy4_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | 1 | 19 | False | no_final_ceiling_hit;threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy4_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | 2 | 21 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy4_basin_credit | decision_bootstrap_w1p0_e5_linear_welfare | 3 | 19 | False | no_final_ceiling_hit;threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy4_basin_credit | linear_welfare_heavy | 1 | 19 | False | no_final_ceiling_hit;threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy4_basin_credit | linear_welfare_heavy | 2 | 21 | True | slow_after_positive_signal | 1 | 1 | 2 |
| toy4_basin_credit | linear_welfare_heavy | 3 | 23 | False | no_final_ceiling_hit;threshold_crossing_activity;slow_after_positive_signal | 1 | 1 | 1 |
| toy4_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | 1 |  | False | no_final_ceiling_hit;weak_or_missing_credit_signal | 1 |  | 1 |
| toy4_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | 2 |  | False | no_final_ceiling_hit;weak_or_missing_credit_signal | 1 |  | 2 |
| toy4_basin_credit | mixed_individual_basin_w0p5_0p5_h1 | 3 |  | False | no_final_ceiling_hit;weak_or_missing_credit_signal | 1 |  | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | 1 | 13 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | 2 | 14 | True | slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_commitment_social_w0p5_0p5_h1 | 3 | 14 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | 1 | 19 | False | no_final_ceiling_hit;decision_action_lag;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | 2 | 16 | True | slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_floor0p5_social_w0p5_0p5_h1 | 3 | 19 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | 1 | 12 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | 2 | 12 | True | slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_feedback_social_w0p5_0p5_h1 | 3 | 10 | True | threshold_crossing_activity | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 1 | 11 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 2 | 12 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 3 | 10 | True | threshold_crossing_activity | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 1 | 12 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 2 | 12 | True | slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 3 | 10 | True | threshold_crossing_activity | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | 1 | 19 | False | no_final_ceiling_hit;decision_action_lag;threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | 2 | 16 | True | slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_social_w0p5_0p5_h1 | 3 | 19 | False | no_final_ceiling_hit;threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | 1 | 19 | False | no_final_ceiling_hit;decision_action_lag;threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | 2 | 16 | True | slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_confidence_tail_floor0p5_social_w0p5_0p5_h1 | 3 | 19 | False | no_final_ceiling_hit;threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | 1 | 19 | False | no_final_ceiling_hit;decision_action_lag;threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | 2 | 16 | True | slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_directional_social_w0p5_0p5_h1 | 3 | 19 | False | no_final_ceiling_hit;threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | 1 | 19 | False | no_final_ceiling_hit;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | 2 | 16 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 2 |
| toy4_basin_credit | mixed_objective_basin_w0p5_0p5_h1 | 3 | 14 | True | threshold_crossing_activity;slow_after_positive_signal | 1 | 2 | 1 |
| toy4_basin_credit | reputation_imitation | 1 | 3 | True | weak_or_missing_credit_signal;threshold_crossing_activity |  | 1 | 1 |
| toy4_basin_credit | reputation_imitation | 2 | 3 | True | weak_or_missing_credit_signal;threshold_crossing_activity |  | 1 | 1 |
| toy4_basin_credit | reputation_imitation | 3 | 2 | True | weak_or_missing_credit_signal;threshold_crossing_activity |  | 1 | 1 |
