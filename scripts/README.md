# Scripts

Use this directory for thin command-line entry points.

Expected future scripts:

- `run_toy1.py`: run Neural HK Classification from a YAML config.
- `run_toy2.py`: run Neural Spatial Prisoner's Dilemma from a YAML config.
- `run_toy3.py`: run Neural Opinion Rewiring from a YAML config.
- `run_toy4.py`: run Neural Public Goods and Commons from a YAML config.
- `run_toy5.py`: run Neural Contagion and Adoption from a YAML config.
- `run_toy6.py`: run the multi-action categorical spatial game from a YAML
  config.
- `run_toy7.py`: run the continuous extraction-intensity resource ABM from a
  YAML config.
- `run_toy8.py`: run the asynchronous event-driven adoption/failure ABM from
  a YAML config.
- `run_toy9.py`: run the heterogeneous-agent binary adoption ABM from a YAML
  config.
- `run_toy10.py`: run the dynamic-network market/ecology ABM from a YAML
  config.
- `run_toy2_sweep.py`: run Toy 2 payoff-regime, alpha, and initial-condition
  basin sweeps across neural, Fermi, reputation-imitation, and RD reference
  policy rules, including reputation/mobility diagnostics and optional neural
  reputation observation features. The Toy 2 sweep defaults to the legacy
  `output_average/none` social path, and can opt into
  `output_average/output_similarity` with `--peer-rules output_similarity` plus
  `--coordination-thresholds`.
- `run_toy3_sweep.py`: run Toy 3 update-rule, confidence-threshold,
  rewiring-rate, mixer, peer-rule, threshold, alpha, and seed sweeps with
  summary and grouped-summary CSV outputs. The default social peer rule remains
  `bounded_confidence`.
- `run_toy4_sweep.py`: run Toy 4 public-goods sweeps across imitation,
  neural-policy, reputation-imitation, coordination mixing, peer-rule,
  threshold, alpha, and fixed-cell mobility variants, with optional neural
  reputation observation features.
- `run_toy5_sweep.py`: run Toy 5 contagion and threshold-adoption sweeps
  across policy rules, threshold regimes, repeated-exposure settings,
  coordination mixing, threshold, alpha, and seeds with summary and
  grouped-summary CSV outputs.
- `run_toy6_sweep.py`: run Toy 6 strategy-count, initial-distribution,
  payoff-profile, mixer, threshold, alpha, and seed sweeps with summary and
  grouped-summary CSV outputs.
- `run_toy7_sweep.py`: run Toy 7 recovery-rate, extraction-cost,
  initial-intensity, exploration, mixer, threshold, alpha, and seed sweeps with
  summary and grouped-summary CSV outputs.
- `run_toy8_sweep.py`: run Toy 8 initial-state, activation, failure,
  recovery, mixer, threshold, alpha, and seed sweeps with summary and
  grouped-summary CSV outputs.
- `run_toy9_sweep.py`: run Toy 9 group-composition, coordination-gate,
  threshold, payoff, mixer, threshold, alpha, and seed sweeps with summary and
  grouped-summary CSV outputs.
- `run_toy10_sweep.py`: run Toy 10 recovery, extraction-cost, network-churn,
  initial-market-state, mixer, threshold, alpha, and seed sweeps with summary
  and grouped-summary CSV outputs.
- `run_toy_validation.py`: run the representative Toy 1-10 validation suite
  and write run-level CSV, metric CSV, and Markdown report outputs.
  Presets:
  `--preset quick` runs one short scenario per toy for fast smoke validation;
  `--preset representative` runs the default diagnostic suite;
  `--preset paper-candidate` runs the full suite with more seeds and epochs.
- `run_nabm_effect_matrix.py`: run the small Toy1-5 NABM effect matrix from
  `experiments/evidence/nabm_effect_matrix_quick.yaml`. The manifest declares
  cases, baseline/NABM variants, seeds, primary metrics, and direction
  (`maximize` or `minimize`). The default outputs are preserved under
  `experiments/results/nabm_effect_matrix/` as `<label>_runs.csv`,
  `<label>_effects.csv`, `<label>_pairwise_effects.csv`, and
  `<label>_effects.md`; generated configs are written under
  `experiments/configs/generated/evidence_matrix/<label>/`, and run
  directories are kept in the manifest `runs_dir`.
- `run_basin_credit_evidence_gate.py`: evaluate the Toy2/Toy4 basin-credit
  success gate from `experiments/evidence/toy24_basin_credit_quick.yaml` after
  `run_nabm_effect_matrix.py` has produced `<label>_runs.csv`. The gate writes
  audited JSON and Markdown summaries under `experiments/evidence/results/` and
  only treats non-teacher, non-bootstrap, non-replay NABM variants as main
  success candidates. It rejects run CSVs whose label/case/variant/seed rows do
  not match the manifest.
- `run_adapter_holdout_evidence.py`: run the adapter-only threshold adoption
  holdout from `experiments/evidence/adapter_only_threshold_holdout_quick.yaml`.
  The domain lives in the runner rather than `src/neural_abm` and uses public
  binary policy/readiness unit APIs with baseline, negative-control, and main
  variants.
- `run_basin_credit_evidence_workflow.py`: run the basin-credit evidence matrix
  and immediately evaluate the hardened gate. Use `--skip-matrix --runs-path`
  to audit an existing `<label>_runs.csv`, or `--require-pass` in automation to
  fail when the Toy2/Toy4 main-claim criteria are not met. The workflow also
  writes read-only evidence profiles by default as `<label>_profile.json`,
  `<label>_profile.md`, and `<label>_profile_cases.csv`; use
  `--profile-output-dir` to redirect them or `--skip-profile` for gate-only
  audits. Toy-specific profile adapters must interpret generated artifacts only;
  they must not own core toy semantics or change the gate.
- `python -m neural_abm.diagnostics.evidence_profile`: profile an existing
  manifest/run CSV pair without rerunning simulations. This is the preferred
  way to inspect new toy evidence artifacts before adding a toy-specific
  adapter. Generic profiles summarize case/variant hits, time-to-ceiling,
  terminal ceiling rate, and issue codes; adapters can add diagnostic notes and
  structured JSON details for toy-specific mechanisms.
- `profile_evidence_artifacts.py` or
  `python -m neural_abm.diagnostics.profile_index`: build a registry across
  many existing evidence profiles. By default it scans `experiments/evidence/*.yaml`,
  profiles manifests with matching `<label>_runs.csv`, skips missing run CSVs,
  and writes `evidence_profile_index.csv`, `evidence_profile_index.md`, and
  `evidence_profile_index.json`. The index is case-level so repeated issue
  codes such as final-epoch hazard, material+basin collapse, or missing run
  artifacts can be filtered without opening each profile.
- `run_basin_phase_critic_training.py`: train and evaluate the offline learned
  basin phase critic from `basin_transition_samples.parquet` artifacts. The
  default manifest reads the dataset-producing Toy2/Toy4 run rows, trains on
  seed 1-2 samples, evaluates on seed 3, scores candidate actions 0/1 with a
  bootstrap-head uncertainty diagnostic, and writes per-case model, prediction,
  JSON, CSV, and Markdown quality artifacts under
  `experiments/results/basin_critic/`. The candidate-context manifest
  `toy24_basin_phase_critic_candidate_context_quick.yaml` adds features that
  recompute phase action rate, policy rate, and consensus under each candidate
  action before scoring. The pairwise-direction manifest
  `toy24_basin_phase_critic_pairwise_direction_quick.yaml` uses
  `label_mode: prototype_direction` to train an explicit candidate-action
  direction critic from prototype basin-advantage labels. The
  future-outcome-direction manifest
  `toy24_basin_phase_critic_future_outcome_direction_quick.yaml` uses
  `label_mode: future_outcome_direction` to train direction labels from
  observed future basin motion instead of the prototype sign.
- `summarize_basin_learned_diagnostics.py`: summarize read-only learned basin
  runtime diagnostics from a basin-credit evidence run. It reads each run's
  `micro_state.csv` and `aggregate_metrics.csv`, compares prototype and learned
  action-1 advantages, and writes run-level, grouped CSV, and Markdown
  diagnostic summaries.
- `diagnose_time_to_ceiling_bottlenecks.py`: derive early-window
  time-to-ceiling bottleneck diagnostics from an evidence matrix run CSV. It
  reads each run's `aggregate_metrics.csv` and writes epoch-level deltas,
  run-level bottleneck flags, and a grouped Markdown report for credit, local
  policy, social mixing, decision/action, and revision-rate attribution.
- `train_basin_replay_weight_scorer.py`: train frozen per-agent basin replay
  weight scorers from `basin_transition_samples.parquet` and the
  candidate-context learned basin critic. The scorer predicts continuous
  replay loss weights for `learned_credit_replay_mode: learned_weight`.
  Supported supervision modes are the default current-signal `magnitude`
  target, `future_basin_motion`, which derives labels from forward payoff
  movement over `--future-horizon`, and `intervention_pressure`, which combines
  current signal magnitude with future basin motion as a pressure target.
- `experiments/evidence/toy24_basin_learned_credit_replay_quick.yaml`: evidence
  manifest for the first learned-credit replay slice. It keeps the prototype
  escalation replay variant as diagnostic context, then tests a gated learned
  phase-critic credit source with prototype fallback as the main NABM candidate
  and a zero-fallback variant as an isolation diagnostic.
- `experiments/evidence/toy24_basin_learned_credit_candidate_context_replay_quick.yaml`:
  evidence manifest for replay with the candidate-context learned critic model.
  It uses the same prototype escalation diagnostic but points learned credit at
  the candidate-conditioned critic artifacts.
- `experiments/evidence/toy24_basin_learned_credit_replay_selection_quick.yaml`:
  evidence manifest for learned/prototype replay selection. It compares
  candidate-context learned replay over all candidates with confident-agreement
  and confident-disagreement replay subsets.
- `experiments/evidence/toy24_basin_learned_credit_replay_floor_quick.yaml`:
  evidence manifest for starvation-safe learned/prototype replay selection. It
  keeps confident-agreement replay as the selector but fills a minimum replay
  rate from high-magnitude prototype candidates.
- `experiments/evidence/toy24_basin_learned_credit_replay_curriculum_quick.yaml`:
  evidence manifest for temporal learned/prototype replay selection. It starts
  with a broad replay floor and linearly narrows toward the configured floor as
  training progresses.
- `experiments/evidence/toy24_basin_learned_credit_soft_replay_quick.yaml`:
  evidence manifest for soft learned-credit replay weighting. It keeps the
  confident-agreement learned signal but scales each replay loss with a
  per-agent `replay_weight` instead of using only a hard include/drop gate.
- `experiments/evidence/toy24_basin_learned_credit_weight_scorer_quick.yaml`:
  evidence manifest for the first learned replay-weight scorer. It loads a
  frozen scorer artifact to predict `replay_weight` at runtime instead of using
  the fixed soft replay formula as the main mechanism.
- `experiments/evidence/toy24_basin_learned_credit_future_motion_weight_scorer_quick.yaml`:
  evidence manifest for the future-motion replay-weight scorer. It keeps the
  magnitude-supervised scorer as a diagnostic variant and tests a scorer trained
  from `future_basin_score_delta` as the main candidate.
- `experiments/evidence/toy24_basin_direction_pressure_quick.yaml`: evidence
  manifest for the direction/pressure split slice. It keeps the learned
  candidate-context basin critic as the direction module and tests an
  `intervention_pressure` replay scorer as the main pressure module.
- `experiments/evidence/toy24_basin_pairwise_direction_pressure_quick.yaml`:
  evidence manifest for the pairwise direction-pressure slice. It keeps the
  `intervention_pressure` scorer as the pressure module and swaps the direction
  module to the pairwise candidate-action basin critic trained with
  `toy24_basin_phase_critic_pairwise_direction_quick.yaml`.
- `experiments/evidence/toy24_basin_future_outcome_direction_pressure_quick.yaml`:
  evidence manifest for the future-outcome direction-pressure slice. It keeps
  the `intervention_pressure` scorer as the pressure module and swaps the
  direction module to the candidate-action critic trained with
  `toy24_basin_phase_critic_future_outcome_direction_quick.yaml`.
- `run_toy1_ablation.py`: run the first Toy 1 ablation matrix from a base
  config and write CSV/Markdown summaries.
- `run_toy1_sweep.py`: run Toy 1 alpha/threshold sweeps and grouped summaries.
- `benchmark_accelerator_policy_core.py`: benchmark per-agent neural policy
  inference against the batched accelerator MLP core on CPU/GPU devices.
- `benchmark_toy_gpu_core.py`: benchmark Toy2/4/5 neural binary runners
  end-to-end across CPU/GPU devices, agent counts, and binary coordination
  mixers, writing the common GPU-core benchmark CSV fields plus optional
  detailed stage timing CSVs for bottleneck attribution. Toy4/Toy5 also accept
  `--training-backends loop batched auto` to compare the default per-agent
  optimizer loop with the batched gradient backend, and `--warmup-runs`/
  `--repeats` to report mean/std benchmark timings.
- `plot_toy1_output_alpha.py`: plot the output-average alpha sweep as a
  paper-ready figure candidate.
- `plot_toy1_mixer_comparison.py`: plot the 5-seed Toy 1 mixer ablation as a
  paper-ready comparison figure.
- `plot_toy1_parameter_phase.py`: plot parameter-path alpha/threshold phase
  behavior.
- `plot_toy1_parameter_alignment.py`: plot raw and aligned parameter averaging
  diagnostics for independent-init agents.
- `plot_toy1_cluster_dynamics.py`: plot per-agent prediction cluster dynamics
  from probe prediction snapshots.
- `plot_toy1_cluster_comparison.py`: plot several probe-prediction snapshot
  runs side by side to compare connected, partial, and fragmented regimes.
- `plot_toy2_initial_comparison.py`: plot initial Toy 2 no-social versus
  policy-output mixing trajectories.
- `plot_toy2_regime_sweep.py`: plot Toy 2 regime, dynamics-reference,
  alpha-response, basin-sensitivity, reputation, and mobility validation
  figures.
- `plot_toy4_sweep.py`: plot Toy 4 contribution, reputation, mobility rate,
  and mobility gain diagnostics from `run_toy4_sweep.py`.
- `plot_toy_validation.py`: plot validation suite summaries from quick,
  representative, or paper-candidate result CSVs.
- Export plots from experiment runs.
- Build paper figures.
- Create archive snapshots.

Scripts should call reusable code from `src/neural_abm/` instead of containing
core simulation logic.

Toy1-10 sweep scripts use `src/neural_abm/sweep.py` for shared output specs,
result-row field mapping, stable summary CSV writing, grouped-summary
construction, spec-bound compatibility helpers for script-local summary
functions, optional grouped Markdown hooks, selected config/result extraction
for resolved row fields, and final aggregate-metric CSV readers. Toy1 now uses
the shared explicit-case orchestration for base YAML loading, generated config
routing, run execution, row assembly, and summary/grouped output writing while
keeping its case matrix and Markdown readout local. Toy3-10 use shared
coordination/case expansion,
domain parameter-grid iteration, prepared-case config writing with nested
updates and optional Toy-specific mutation hooks, run execution, row assembly,
and main orchestration helpers.
Toy2 and Toy3-10 also share common CLI argument registration with toy-specific
legacy defaults preserved through helper options. Toy2 still owns its
payoff/regime matrix semantics, conditional grid pruning, and RD reference
insertion, but its point-row orchestration, non-overwriting output path
resolution, and summary/grouped/Markdown output writing now run through the
shared sweep helpers.
Toy-specific scripts keep their own domain parameter definitions and aggregation
specs; domain run-name and config-update builders are kept as small
toy-specific functions.
