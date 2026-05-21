# Toy Validation Report: toy_validation_paper_candidate_toy1_10_social_calibrated_seeds01_05

## Pass/Fail

| Toy | Scenario | Seeds | Status | Failed checks |
| --- | --- | ---: | --- | --- |
| toy10 | toy10_market_output_average | 5 | pass |  |
| toy1 | toy1_no_social | 5 | pass |  |
| toy1 | toy1_output_average | 5 | pass |  |
| toy2 | toy2_harsh_pd_fermi_none | 5 | pass |  |
| toy2 | toy2_harsh_pd_neural_none | 5 | pass |  |
| toy2 | toy2_harsh_pd_neural_output_average | 5 | pass |  |
| toy2 | toy2_harsh_pd_neural_reputation_observation_output_average | 5 | pass |  |
| toy2 | toy2_harsh_pd_reputation_output_average | 5 | pass |  |
| toy3 | toy3_hk_no_rewire | 5 | pass |  |
| toy3 | toy3_hk_rewire | 5 | pass |  |
| toy3 | toy3_neural_output_average | 5 | pass |  |
| toy4 | toy4_commons_collapse | 5 | pass |  |
| toy4 | toy4_static_imitation_none | 5 | pass |  |
| toy4 | toy4_static_neural_output_average | 5 | pass |  |
| toy4 | toy4_static_neural_reputation_observation_output_average | 5 | pass |  |
| toy4 | toy4_static_reputation_output_average | 5 | pass |  |
| toy5 | toy5_heterogeneous_partial | 5 | pass |  |
| toy5 | toy5_high_threshold_block | 5 | pass |  |
| toy5 | toy5_low_threshold_cascade | 5 | pass |  |
| toy5 | toy5_neural_output_average | 5 | pass |  |
| toy5 | toy5_neural_reputation_observation_output_average | 5 | pass |  |
| toy5 | toy5_reputation_output_average | 5 | pass |  |
| toy6 | toy6_categorical_output_average | 5 | pass |  |
| toy7 | toy7_resource_output_average | 5 | pass |  |
| toy8 | toy8_async_output_average | 5 | pass |  |
| toy9 | toy9_heterogeneous_output_average | 5 | pass |  |

## Key Metric Means

- toy10_market_output_average: domain_final_resource_fraction=0.4366, domain_final_market_price=0.6985, domain_final_mean_harvest_intensity=0.4026, domain_cumulative_rewired_edge_count=518.4
- toy1_no_social: domain_final_mean_global_accuracy=0.8889, domain_final_mean_consensus=0.9448, final_fragmentation_components=50
- toy1_output_average: domain_final_mean_global_accuracy=0.8915, domain_final_mean_consensus=0.9597, final_fragmentation_components=1
- toy2_harsh_pd_fermi_none: final_action_rate=0, final_mean_policy_action_probability=0, final_mean_payoff=1, final_mean_reputation=4.817e-05
- toy2_harsh_pd_neural_none: final_action_rate=0, final_mean_policy_action_probability=0.002757, final_mean_payoff=1, final_mean_reputation=0.002866
- toy2_harsh_pd_neural_output_average: final_action_rate=0, final_mean_policy_action_probability=0.002065, final_mean_payoff=1, final_mean_reputation=0.002509
- toy2_harsh_pd_neural_reputation_observation_output_average: final_action_rate=0, final_mean_policy_action_probability=0.0017, final_mean_payoff=1, final_mean_reputation=0.001986
- toy2_harsh_pd_reputation_output_average: final_action_rate=1, final_mean_policy_action_probability=1, final_mean_payoff=3, final_mean_reputation=1
- toy3_hk_no_rewire: domain_final_polarization_index=0.161, domain_final_opinion_cluster_count=2, domain_final_mean_edge_disagreement=0.4226, domain_cumulative_rewired_edge_count=0
- toy3_hk_rewire: domain_final_polarization_index=0.1611, domain_final_opinion_cluster_count=2, domain_final_mean_edge_disagreement=0.003018, domain_cumulative_rewired_edge_count=157.4
- toy3_neural_output_average: domain_final_polarization_index=6.571e-08, domain_final_opinion_cluster_count=1, domain_final_mean_edge_disagreement=4.111e-05, domain_cumulative_rewired_edge_count=0
- toy4_commons_collapse: final_action_rate=0, domain_payoff_gini=0, domain_resource_level=0, final_mean_reputation=0, domain_collapse_time=1
- toy4_static_imitation_none: final_action_rate=0, domain_payoff_gini=0, domain_resource_level=100, final_mean_reputation=1.35e-05, domain_collapse_time=n/a
- toy4_static_neural_output_average: final_action_rate=0.008, domain_payoff_gini=0.02072, domain_resource_level=100, final_mean_reputation=0.008762, domain_collapse_time=n/a
- toy4_static_neural_reputation_observation_output_average: final_action_rate=0.006, domain_payoff_gini=0.01542, domain_resource_level=100, final_mean_reputation=0.009194, domain_collapse_time=n/a
- toy4_static_reputation_output_average: final_action_rate=1, domain_payoff_gini=0, domain_resource_level=100, final_mean_reputation=1, domain_collapse_time=n/a
- toy5_heterogeneous_partial: final_action_rate=0.792, domain_cascade_size=79.2, final_mean_reputation=0.792, domain_low_threshold_action_rate=0.932, domain_high_threshold_action_rate=0.652
- toy5_high_threshold_block: final_action_rate=0.05, domain_cascade_size=5, final_mean_reputation=0.05, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy5_low_threshold_cascade: final_action_rate=1, domain_cascade_size=100, final_mean_reputation=0.9991, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy5_neural_output_average: final_action_rate=1, domain_cascade_size=100, final_mean_reputation=1, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy5_neural_reputation_observation_output_average: final_action_rate=1, domain_cascade_size=100, final_mean_reputation=1, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy5_reputation_output_average: final_action_rate=1, domain_cascade_size=100, final_mean_reputation=0.9992, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy6_categorical_output_average: domain_final_mean_payoff=0, domain_final_strategy_entropy=0.9952, domain_final_dominant_strategy_fraction=0.366, final_fragmentation_components=1
- toy7_resource_output_average: domain_final_resource_fraction=0.3044, domain_final_mean_intensity=0.4337, domain_final_mean_payoff=0.06599, final_fragmentation_components=1
- toy8_async_output_average: domain_final_active_fraction=0.488, domain_final_failed_fraction=0.27, domain_total_events=100, domain_final_time=12.65
- toy9_heterogeneous_output_average: domain_final_action_rate=0.378, domain_final_group_action_rate_gap=0.228, domain_final_mean_payoff=0.405, final_fragmentation_components=2

## Directional Comparisons

- Toy1 output_average vs no_social: consensus delta 0.9597 - 0.9448; accuracy delta 0.8915 - 0.8889.
- Toy2 output_average diagnostic: action 0 vs no-social 0.
- Toy3 rewiring: edge disagreement 0.003018 vs 0.4226; rewired edges mean 157.4.
- Toy4 commons: static imitation action 0; domain_collapse_time mean 1.
- Toy5 thresholds: low action 1, high action 0.05, neural/social 1.
- Toy6 categorical: entropy 0.9952, dominant strategy fraction 0.366.
- Toy7 resource: resource fraction 0.3044, intensity 0.4337.
- Toy8 async events: active fraction 0.488, failed 0.27, events 100.
- Toy9 heterogeneous agents: action fraction 0.378, group gap 0.228.
- Toy10 market/ecology: resource fraction 0.4366, price 0.6985, rewired edges 518.4.
