# Toy2/Toy4 Domain-Uncertainty Calibration Findings

Manifest:
`experiments/evidence/toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick.yaml`

Purpose:

- Move beyond clean Toy2/Toy4 and topology-only stress.
- Keep the same readiness-propagation candidate family, but change the domain
  problem rather than directly corrupting reputation observations.
- Use this as calibration evidence, not as a final claim gate.

Stress settings:

- Toy2: Stag-Hunt payoff regime with low initial action-1 mass.
  - `domain.environment.initial_action_probability: 0.35`
  - `domain.game.family: stag_hunt`
  - payoff `R=4, T=3, P=2, S=0`
  - ceiling metric `mean_payoff`, ceiling value `4.0`
- Toy4: resource-coupled public goods.
  - `domain.environment.initial_action_probability: 0.35`
  - `domain.environment.resource_enabled: true`
  - `resource_initial: 60.0`
  - `resource_recovery_rate: 0.03`
  - `resource_extraction_per_defector: 0.05`
  - ceiling metric `mean_payoff`, ceiling value `0.6`

Run artifacts:

- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick_runs.csv`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick_effects.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick_profile.md`
- `experiments/evidence/results/toy24_precommitment_peer_evidence_domain_uncertainty_calibration_quick.summary.md`

Gate result: **fail** overall. This is expected to be informative because the
criteria were calibration criteria, not a mature claim threshold.

| Case | Variant | Role | Final hits | Ever hits | Mean TtC | Terminal ceiling rate | Late flip rate | Metric mean |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Toy2 Stag Hunt | `reputation_imitation_stag_hunt_p0p35` | baseline | 3/3 | 3/3 | 4.0 | 1.00 | 0.000000 | 4.0000 |
| Toy2 Stag Hunt | `objective_basin_stag_hunt_p0p35` | diagnostic | 2/3 | 3/3 | 16.0 | 0.87 | 0.003961 | 3.9833 |
| Toy2 Stag Hunt | `revision_objective_basin_stag_hunt_p0p35` | diagnostic | 3/3 | 3/3 | 16.7 | 0.87 | 0.003066 | 4.0000 |
| Toy2 Stag Hunt | `revision_precommitment_evidence_stag_hunt_p0p35` | diagnostic | 3/3 | 3/3 | 11.0 | 1.00 | 0.000544 | 4.0000 |
| Toy2 Stag Hunt | `revision_precommitment_peer_evidence_stag_hunt_p0p35` | main | 3/3 | 3/3 | 10.0 | 1.00 | 0.000000 | 4.0000 |
| Toy4 Resource | `reputation_imitation_resource_p0p35` | baseline | 3/3 | 3/3 | 15.0 | 1.00 | 0.000000 | 0.6000 |
| Toy4 Resource | `objective_basin_resource_p0p35` | diagnostic | 0/3 | 0/3 |  | 0.00 | 0.000000 | 0.0000 |
| Toy4 Resource | `revision_objective_basin_resource_p0p35` | diagnostic | 0/3 | 0/3 |  | 0.00 | 0.000000 | -0.0033 |
| Toy4 Resource | `revision_precommitment_evidence_resource_p0p35` | diagnostic | 0/3 | 0/3 |  | 0.00 | 0.000000 | -0.0033 |
| Toy4 Resource | `revision_precommitment_peer_evidence_resource_p0p35` | main | 0/3 | 0/3 |  | 0.00 | 0.000000 | -0.0033 |

Interpretation:

- Toy2 Stag Hunt is not a useful baseline-fragility stress at this setting.
  Reputation imitation reaches the cooperative ceiling in all seeds with mean
  TtC 4.0. The readiness candidate also succeeds, but is slower with mean TtC
  10.0.
- Toy4 resource coupling exposes a real failure boundary for the current
  readiness candidate. Reputation imitation reaches the ceiling in all seeds
  with mean TtC 15.0, while every neural objective/revision/precommitment
  variant fails to reach the ceiling even once.
- The Toy4 failure is not the previous late-flip hazard. It is an early
  resource-collapse trajectory. In seed 1, reputation imitation moves from
  action rate 0.34 at epoch 0 to 1.0 by epoch 4, then recovers resource from
  60.0 to 100.0 by epoch 15. The peer-evidence candidate moves only to 0.49 at
  epoch 1, then falls to 0.19 by epoch 10 and 0.02 by epoch 20; resource reaches
  0.0 by epoch 20.
- Precommitment does not rescue this case because the policy never enters the
  high-confidence cooperative region. This is a direction/credit failure under
  coupled resource dynamics, not a commitment-stability failure after reaching
  the ceiling.
- A follow-up diagnostic hook now records Toy4 resource-maintenance pressure in
  the existing state-continuation `environment` component. This does not affect
  default behavior because current profiles keep `environment_weight=0.0`, but
  it makes the missing signal visible. In the failing Toy4 peer-evidence seed 1,
  `domain_environment_continuation_advantage_mean` rises from 0.273 at epoch 1
  to 1.107 at epoch 10 and 1.968 at epoch 20, while
  `domain_effective_advantage_mean` remains negative because the environment
  component is not yet weighted.

Conclusion:

- The current readiness-propagation unit is robust in clean, noisy-reputation,
  sparse-seed, open-boundary, and Toy2 Stag-Hunt slices, but it is not resource
  aware enough for Toy4 coupled-resource dynamics.
- The next step should not be another peer-evidence weight sweep. The failure
  is upstream of peer evidence: the local objective/revision signal does not
  recognize that early contribution is needed to preserve the future resource
  state.
- A structural next step is justified: add resource-collapse/readiness
  diagnostics to the evidence profile and prototype a Toy4 resource-aware
  objective signal by activating the new environment component with a bounded
  `environment_weight` sweep.
