# NABM Effect Matrix

## Grouped Effects

| Case | Toy | Metric | Direction | Baseline Mean | NABM Mean | Effect | 95% CI |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| toy1_social_accuracy | toy1 | domain_final_mean_global_accuracy | maximize | 0.67734 | 0.729404 | 0.052064 | 0.0333006 |
| toy2_reference_payoff | toy2 | final_mean_payoff | maximize | 2.09662 | 2.01667 | -0.0799501 | 0.288334 |
| toy3_opinion_polarization | toy3 | domain_final_polarization_index | minimize | 0.166169 | 0.00683244 | 0.159337 | 0.00539748 |
| toy4_public_goods_payoff | toy4 | final_mean_payoff | maximize | 0.3 | 0.228 | -0.072 | 0.0244804 |
| toy5_cascade_size | toy5 | domain_cascade_size | maximize | 3.5 | 96.3333 | 92.8333 | 3.45712 |

## Pairwise Baseline Effects

| Case | Toy | Baseline Variant | NABM Variant | Effect | 95% CI |
| --- | --- | --- | --- | ---: | ---: |
| toy1_social_accuracy | toy1 | no_social | output_average | 0.052064 | 0.0333006 |
| toy2_reference_payoff | toy2 | fermi_imitation | neural_output_average | 0.388333 | 0.416023 |
| toy2_reference_payoff | toy2 | rd_well_mixed | neural_output_average | 0.35515 | 0.226769 |
| toy2_reference_payoff | toy2 | reputation_imitation | neural_output_average | -0.983333 | 0.226769 |
| toy3_opinion_polarization | toy3 | deffuant | neural_output_average | 0.159014 | 0.00519493 |
| toy3_opinion_polarization | toy3 | hk | neural_output_average | 0.15966 | 0.005619 |
| toy4_public_goods_payoff | toy4 | imitation | neural_output_average | 0.228 | 0.0244804 |
| toy4_public_goods_payoff | toy4 | reputation_imitation | neural_output_average | -0.372 | 0.0244804 |
| toy5_cascade_size | toy5 | complex_threshold | neural_output_average | 93.3333 | 4.28419 |
| toy5_cascade_size | toy5 | simple_contagion | neural_output_average | 92.3333 | 2.84781 |

Positive effect values favor the NABM group.
