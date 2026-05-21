# Toy Validation Report: toy1_7_generalization_quick_smoke

## Pass/Fail

| Toy | Scenario | Seeds | Status | Failed checks |
| --- | --- | ---: | --- | --- |
| toy1 | toy1_no_social | 1 | pass |  |
| toy2 | toy2_harsh_pd_neural_none | 1 | pass |  |
| toy3 | toy3_hk_no_rewire | 1 | pass |  |
| toy4 | toy4_static_imitation_none | 1 | pass |  |
| toy5 | toy5_low_threshold_cascade | 1 | pass |  |
| toy6 | toy6_categorical_output_average | 1 | pass |  |
| toy7 | toy7_resource_output_average | 1 | pass |  |

## Key Metric Means

- toy1_no_social: domain_final_mean_global_accuracy=0.55, domain_final_mean_consensus=0.922, final_fragmentation_components=50
- toy2_harsh_pd_neural_none: final_action_rate=0.44, final_mean_policy_action_probability=0.4337, final_mean_payoff=2.135, final_mean_reputation=0.4629
- toy3_hk_no_rewire: domain_final_polarization_index=0.1681, domain_final_opinion_cluster_count=2, domain_final_mean_edge_disagreement=0.4186, domain_cumulative_rewired_edge_count=0
- toy4_static_imitation_none: final_action_rate=0.04, domain_payoff_gini=0.08502, domain_resource_level=100, final_mean_reputation=0.3643, domain_collapse_time=n/a
- toy5_low_threshold_cascade: final_action_rate=0.07, domain_cascade_size=7, final_mean_reputation=0.02122, domain_low_threshold_action_rate=n/a, domain_high_threshold_action_rate=n/a
- toy6_categorical_output_average: domain_final_mean_payoff=0, domain_final_strategy_entropy=0.9937, domain_final_dominant_strategy_fraction=0.37, final_fragmentation_components=1
- toy7_resource_output_average: domain_final_resource_fraction=0.721, domain_final_mean_intensity=0.5804, domain_final_mean_payoff=0.3202, final_fragmentation_components=1

## Directional Comparisons

- Toy1 output_average vs no_social: consensus delta n/a - 0.922; accuracy delta n/a - 0.55.
- Toy2 output_average diagnostic: action n/a vs no-social 0.44.
- Toy3 rewiring: edge disagreement n/a vs 0.4186; rewired edges mean n/a.
- Toy4 commons: static imitation action 0.04; domain_collapse_time mean n/a.
- Toy5 thresholds: low action 0.07, high action n/a, neural/social n/a.
- Toy6 categorical: entropy 0.9937, dominant strategy fraction 0.37.
- Toy7 resource: resource fraction 0.721, intensity 0.5804.
