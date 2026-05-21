# Toy Validation Report: toy_validation_representative_toy1_10_social_calibrated_seeds01_03

## Pass/Fail

| Toy | Scenario | Seeds | Status | Failed checks |
| --- | --- | ---: | --- | --- |
| toy10 | toy10_market_output_average | 3 | pass |  |
| toy1 | toy1_no_social | 3 | pass |  |
| toy1 | toy1_output_average | 3 | pass |  |
| toy2 | toy2_harsh_pd_fermi_none | 3 | pass |  |
| toy2 | toy2_harsh_pd_neural_none | 3 | pass |  |
| toy2 | toy2_harsh_pd_neural_output_average | 3 | pass |  |
| toy2 | toy2_harsh_pd_neural_reputation_observation_output_average | 3 | pass |  |
| toy2 | toy2_harsh_pd_reputation_output_average | 3 | pass |  |
| toy3 | toy3_hk_no_rewire | 3 | pass |  |
| toy3 | toy3_hk_rewire | 3 | pass |  |
| toy3 | toy3_neural_output_average | 3 | pass |  |
| toy4 | toy4_commons_collapse | 3 | pass |  |
| toy4 | toy4_static_imitation_none | 3 | pass |  |
| toy4 | toy4_static_neural_output_average | 3 | pass |  |
| toy4 | toy4_static_neural_reputation_observation_output_average | 3 | pass |  |
| toy4 | toy4_static_reputation_output_average | 3 | pass |  |
| toy5 | toy5_heterogeneous_partial | 3 | pass |  |
| toy5 | toy5_high_threshold_block | 3 | pass |  |
| toy5 | toy5_low_threshold_cascade | 3 | pass |  |
| toy5 | toy5_neural_output_average | 3 | pass |  |
| toy5 | toy5_neural_reputation_observation_output_average | 3 | pass |  |
| toy5 | toy5_reputation_output_average | 3 | pass |  |
| toy6 | toy6_categorical_output_average | 3 | pass |  |
| toy7 | toy7_resource_output_average | 3 | pass |  |
| toy8 | toy8_async_output_average | 3 | pass |  |
| toy9 | toy9_heterogeneous_output_average | 3 | pass |  |

## Key Metric Means

- toy10_market_output_average: domain_final_resource_fraction=0.4475, domain_final_market_price=0.693, domain_final_mean_harvest_intensity=0.4028, domain_cumulative_rewired_edge_count=271
- toy1_no_social: domain_final_mean_global_accuracy=0.8898, domain_final_mean_consensus=0.9548, final_fragmentation_components=50
- toy1_output_average: domain_final_mean_global_accuracy=0.8923, domain_final_mean_consensus=0.9606, final_fragmentation_components=1
- toy2_harsh_pd_fermi_none: final_action_rate=0, final_mean_policy_action_probability=0, final_mean_payoff=1, final_mean_reputation=0.006604
- toy2_harsh_pd_neural_none: final_action_rate=0.02, final_mean_policy_action_probability=0.009703, final_mean_payoff=1.06, final_mean_reputation=0.03296
- toy2_harsh_pd_neural_output_average: final_action_rate=0.02, final_mean_policy_action_probability=0.006949, final_mean_payoff=1.06, final_mean_reputation=0.02126
- toy2_harsh_pd_neural_reputation_observation_output_average: final_action_rate=0.01667, final_mean_policy_action_probability=0.00651, final_mean_payoff=1.05, final_mean_reputation=0.0205
- toy2_harsh_pd_reputation_output_average: final_action_rate=1, final_mean_policy_action_probability=1, final_mean_payoff=3, final_mean_reputation=0.9973
- toy3_hk_no_rewire: domain_final_polarization_index=0.1663, domain_final_opinion_cluster_count=2, domain_final_mean_edge_disagreement=0.4236, domain_cumulative_rewired_edge_count=0
- toy3_hk_rewire: domain_final_polarization_index=0.1664, domain_final_opinion_cluster_count=2, domain_final_mean_edge_disagreement=3.749e-05, domain_cumulative_rewired_edge_count=156.3
- toy3_neural_output_average: domain_final_polarization_index=7.748e-06, domain_final_opinion_cluster_count=1, domain_final_mean_edge_disagreement=0.0006395, domain_cumulative_rewired_edge_count=0
- toy4_commons_collapse: final_action_rate=0, domain_payoff_gini=0, domain_resource_level=0, final_mean_reputation=0, domain_collapse_time=1
- toy4_static_imitation_none: final_action_rate=0, domain_payoff_gini=0, domain_resource_level=100, final_mean_reputation=0.00261, domain_collapse_time=n/a
- toy4_static_neural_output_average: final_action_rate=0.01333, domain_payoff_gini=0.02864, domain_resource_level=100, final_mean_reputation=0.0354, domain_collapse_time=n/a
- toy4_static_neural_reputation_observation_output_average: final_action_rate=0.01333, domain_payoff_gini=0.02841, domain_resource_level=100, final_mean_reputation=0.03735, domain_collapse_time=n/a
- toy4_static_reputation_output_average: final_action_rate=1, domain_payoff_gini=0, domain_resource_level=100, final_mean_reputation=0.9973, domain_collapse_time=n/a
- toy5_heterogeneous_partial: final_action_rate=0.8067, domain_cascade_size=80.67, final_mean_reputation=0.8011, domain_low_threshold_action_rate=0.94, domain_high_threshold_action_rate=0.6733
- toy5_high_threshold_block: final_action_rate=0.05, domain_cascade_size=5, final_mean_reputation=0.05, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy5_low_threshold_cascade: final_action_rate=1, domain_cascade_size=100, final_mean_reputation=0.8299, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy5_neural_output_average: final_action_rate=1, domain_cascade_size=100, final_mean_reputation=0.9943, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy5_neural_reputation_observation_output_average: final_action_rate=1, domain_cascade_size=100, final_mean_reputation=0.9942, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy5_reputation_output_average: final_action_rate=1, domain_cascade_size=100, final_mean_reputation=0.8746, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy6_categorical_output_average: domain_final_mean_payoff=0, domain_final_strategy_entropy=0.987, domain_final_dominant_strategy_fraction=0.4, final_fragmentation_components=1
- toy7_resource_output_average: domain_final_resource_fraction=0.3052, domain_final_mean_intensity=0.4356, domain_final_mean_payoff=0.06646, final_fragmentation_components=1
- toy8_async_output_average: domain_final_active_fraction=0.3933, domain_final_failed_fraction=0.1033, domain_total_events=50, domain_final_time=6.346
- toy9_heterogeneous_output_average: domain_final_action_rate=0.3967, domain_final_group_action_rate_gap=0.2467, domain_final_mean_payoff=0.3624, final_fragmentation_components=1.667

## Directional Comparisons

- Toy1 output_average vs no_social: consensus delta 0.9606 - 0.9548; accuracy delta 0.8923 - 0.8898.
- Toy2 output_average diagnostic: action 0.02 vs no-social 0.02.
- Toy3 rewiring: edge disagreement 3.749e-05 vs 0.4236; rewired edges mean 156.3.
- Toy4 commons: static imitation action 0; domain_collapse_time mean 1.
- Toy5 thresholds: low action 1, high action 0.05, neural/social 1.
- Toy6 categorical: entropy 0.987, dominant strategy fraction 0.4.
- Toy7 resource: resource fraction 0.3052, intensity 0.4356.
- Toy8 async events: active fraction 0.3933, failed 0.1033, events 50.
- Toy9 heterogeneous agents: action fraction 0.3967, group gap 0.2467.
- Toy10 market/ecology: resource fraction 0.4475, price 0.693, rewired edges 271.
