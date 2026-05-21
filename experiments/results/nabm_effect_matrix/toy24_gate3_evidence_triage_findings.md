# Toy2/Toy4 Gate 3 Evidence Triage Findings

Date: 2026-05-21

## Scope

Gate 3 separates Toy2/Toy4 outcomes into mechanism-level categories before any
new loss, sampler, or policy path is added.

Inputs:

- `experiments/evidence/toy24_revision_operator_quick.yaml`
- `experiments/evidence/toy24_basin_credit_objective_blend_quick.yaml`
- `experiments/evidence/toy24_revision_operator_precommitment_controls_quick.yaml`
- `experiments/evidence/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick.yaml`

Generated triage index:

- `experiments/results/nabm_effect_matrix/evidence_profile_index_gate3.md`
- `experiments/results/nabm_effect_matrix/evidence_profile_index_gate3.csv`
- `experiments/results/nabm_effect_matrix/evidence_profile_index_gate3.json`

## Classification Rules

The Toy2/Toy4 evidence profile adapter now marks these case-level categories:

- `toy24_triage_success`: best main variant satisfies the gate and reaches
  final ceiling in all expected seeds.
- `toy24_triage_stochastic_gate_brittleness`: best main variant reaches ceiling
  in all expected seeds at least once, but final-epoch hits miss because of
  late flips or low terminal-window ceiling rate.
- `toy24_triage_baseline_favored_environment`: a hand-coded baseline reaches
  final ceiling in all expected seeds and is at least one epoch faster than the
  best main path.
- `toy24_trajectory_success_slow_ttc`: best main reaches final ceiling in all
  expected seeds, but the case fails because mean time-to-ceiling is too slow.
- `toy24_triage_true_mechanism_failure_candidate`: best main does not even
  reach ceiling in all expected seeds.

## Triage Summary

| Artifact | Toy | Best main | Hits | Mean TtC | Classification |
| --- | --- | --- | ---: | ---: | --- |
| revision_operator_quick | toy2 | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | 2/3 | 19.33 | stochastic gate brittleness, baseline-favored |
| revision_operator_quick | toy4 | revision_operator_mixed_objective_basin_w0p5_0p5_h1 | 1/3 | 19.00 | stochastic gate brittleness, baseline-favored |
| basin_credit_objective_blend_quick | toy2 | mixed_objective_basin_confidence_precommitment_social_w0p5_0p5_h1 | 3/3 | 12.00 | slow TtC gate lag, baseline-favored |
| basin_credit_objective_blend_quick | toy4 | mixed_objective_basin_confidence_precommitment_social_feedback_w0p5_0p5_h1 | 3/3 | 11.00 | success, baseline-favored |
| revision_operator_precommitment_controls_quick | toy2 | revision_operator_precommitment_peer_evidence_w1p0 | 3/3 | 9.67 | success, baseline-favored |
| revision_operator_precommitment_controls_quick | toy4 | revision_operator_precommitment_peer_evidence_w1p0 | 3/3 | 9.33 | success, baseline-favored |
| precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick | toy2 | revision_precommitment_peer_evidence_open_sparse_p0p1 | 5/5 | 9.40 | success |
| precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick | toy4 | revision_precommitment_peer_evidence_open_sparse_p0p1 | 5/5 | 9.00 | success |

## Interpretation

Supported:

- The failed `toy24_revision_operator_quick` cases are not clean evidence of a
  structural inability to reach the desired basin. Both Toy2 and Toy4 are
  classified as stochastic gate brittleness: ceiling is reached, but final
  epoch hits are lost by late hazard.
- The objective+basin Toy2 failure is a speed/gate problem rather than a final
  adoption problem: the best main path reaches `3/3` final hits but misses the
  `mean_time_to_ceiling_lt: 10` threshold at `12.0`.
- Precommitment plus peer evidence recovers stable `3/3` or `5/5` final hits
  in the checked control/stress artifacts.

Important caveat:

- Reputation-imitation remains a strong baseline in these environments. Several
  passing main paths are still classified as baseline-favored because the
  hand-coded baseline is much faster.
- Therefore the current Toy2/Toy4 claim should be about mechanism recovery and
  failure-mode separation, not about general speed superiority over classical
  baselines.
- No best-main case in this Gate 3 slice is classified as a true mechanism
  failure candidate. Individual diagnostic variants can still fail badly, but
  the selected best-main failures are currently slow convergence or stochastic
  final-epoch hazard.

Bounded claim:

> Current Toy2/Toy4 evidence is triageable without adding another policy loss:
> quick revision failures are mostly stochastic final-epoch brittleness, the
> objective+basin Toy2 miss is slow TtC gate lag, and precommitment/peer-evidence
> variants recover stable final hits while remaining slower than reputation
> imitation in baseline-favored environments.

## Next Step

Do not tune another loss only to beat reputation-imitation speed. The next useful
Toy2/Toy4 slice is to design a stress case where reputation-imitation is
expected to be fragile, or to adjust gate reporting so final-epoch brittleness is
reported separately from trajectory-level convergence.
