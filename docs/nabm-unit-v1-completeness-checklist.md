# NABM Unit v1 Completeness Checklist

Date: 2026-05-21

## Purpose

This checklist turns Decision 0010 and the boundary audit into an operational
completion map. It separates engineering completion from research evidence and
paper readiness so that future work does not confuse more implementation with a
stronger NABM claim.

Status terms:

- `Implemented`: code path exists and is used by at least one toy or backend.
- `Guarded`: focused tests prevent silent drift away from the unit contract.
- `Evidenced`: a quick manifest, holdout, or diagnostic artifact supports the
  claim under a named condition.
- `Paper-ready`: the claim is bounded enough to appear in a manuscript without
  overstating generality.

## V1 Completion Scorecard

| Surface | Implemented | Guarded | Evidenced | Paper-ready | Current judgment |
| --- | --- | --- | --- | --- | --- |
| Generic unit lifecycle | yes | yes | partial | no | `NABMUnit`, `NABMStep`, and `NABMLocalStep` exist; Gate 8A records that `DomainToyRunner` already owns the compatible-toy outer lifecycle while inner `step(...)` order remains toy-owned. |
| Binary policy lifecycle | yes | yes | yes | partial | Toy2, Toy4, and Toy5 route non-revision neural local policy steps through shared policy plumbing. |
| Binary revision lifecycle | yes | partial | partial | no | Optional stay/switch unit exists for Toy2/Toy4, but evidence remains prototype-level and gate-sensitive. |
| Readiness propagation | yes | yes | yes | partial | Toy5 hard holdout supports threshold-aware readiness under named stress cases. |
| Social distillation | yes | yes | partial | partial | Output-distribution mix and commit diagnostics are unit-backed across loop, batched, and tensor paths. |
| Backend local commits | yes | yes | partial | no | `NABMLocalStep` wraps batched/tensor policy-gradient commits, but backend claims are engineering claims, not NABM novelty claims. |
| Domain diagnostics plumbing | yes | yes | partial | partial | Toy2/Toy4 shared diagnostic field plumbing reduces schema drift without moving semantics into the unit. |
| Holdout migration | yes | yes | yes | partial | Toy5 now has a small threshold-aware topology/threshold grid, but negative-control separation is strongest on safety rather than spread. |
| Evidence gate integration | yes | partial | yes | partial | Manifests and profile index exist, but some gate criteria remain brittle for stochastic final-epoch failures. |
| Manuscript narrative | partial | yes | partial | partial | The paper claim matrix, table candidates, and draft prose now link bounded claims to artifacts and limitations; publication figures remain open. |
| Adapter-only extensibility | yes | yes | quick | partial | Source-free threshold, congestion, and stochastic commons holdout manifests now run binary domains with baseline, negative-control, and main variants through public unit APIs. |
| Existing-toy migration | partial | yes | parity | partial | Gate 7B/7C route Toy8 async hazard and Toy9 heterogeneous probability mixing through the unit-backed scalar path; Gate 7E/7F route Toy7 intensity and Toy10 market/ecology channels through bounded-scalar paths; Gate 7G routes Toy6 categorical distributions through the distribution path while preserving domain semantics. |

## What Is Complete Enough

The following pieces are complete enough to treat as v1 infrastructure:

- `src/neural_abm/unit.py::NABMUnit`, `NABMStep`, and `NABMLocalStep` as generic
  lifecycle primitives.
- `src/neural_abm/spatial_binary.py::BinaryPolicyLearningUnit` as the binary
  neural policy lifecycle owner.
- `src/neural_abm/spatial_binary.py::run_binary_policy_learning_step` as
  semantic-free callback plumbing for policy learning.
- `src/neural_abm/binary_revision.py::BinaryRevisionLearningUnit` as an opt-in
  revision lifecycle primitive.
- `src/neural_abm/readiness.py::BinaryReadinessPropagationUnit` as peer
  readiness aggregation after a domain defines readiness.
- `src/neural_abm/domain_learning_diagnostics.py` as schema and diagnostic
  plumbing for domain-learning extras.

These surfaces should now be protected. New toys should adapt to them first,
and generic unit changes should be treated as contract changes rather than
ordinary toy implementation details.

## What Is Not Complete

The project should not yet claim that the full NABM architecture is finished.
The remaining gaps are:

- Runner ownership is still split for binary spatial runners. Gate 8A records
  that `DomainToyRunner` already owns the Toy6-10-compatible outer lifecycle,
  but toy adapters still own substantial environment transition, phase order,
  and row-mapping logic.
- Toy2/Toy4 evidence is not clean enough to claim general algorithmic
  superiority over hand-coded baselines.
- Revision-operator evidence is structurally useful but not final enough to
  serve as a primary mechanism claim.
- Toy5 hard holdout now supports the unit lifecycle under a small
  topology/threshold grid, but exposure-anchor controls also spread in seeded
  cases, so threshold-aware uniqueness is not established.
- Paper artifacts do not yet express the current boundary: unit lifecycle
  reuse, domain-owned semantics, and bounded robustness evidence.

## Claim Boundary

Current supported claim:

> The project has a reusable neural ABM unit contract that can run binary
> policy learning, social propagation, readiness propagation, backend commits,
> and diagnostics across Toy2, Toy4, and Toy5 without moving payoff, resource,
> threshold, teacher, or basin semantics into the generic layer.

Current unsupported claims:

- Neural ABMs generally outperform Fermi, RD, reputation imitation, or
  threshold baselines.
- Basin credit is a finalized learned critic.
- Revision operators are a solved structural mechanism.
- Toy6-10 are full NABM evidence cases.
- The codebase is ready to be presented as a general-purpose ABM framework.

## Next Completion Gates

### Gate 1: Unit Contract Freeze

Goal: prevent silent expansion of generic unit semantics.

Status: first pass complete.

Artifacts:

- `src/neural_abm/README.md`
- `docs/decisions/0010-nabm-unit-v1-contract.md`
- `docs/nabm-unit-v1-boundary-audit.md`
- `docs/nabm-unit-v1-completeness-checklist.md`
- `tests/test_nabm_unit_adapter_holdout.py`
- `tests/test_nabm_unit_docs.py`

Completed work:

- Add a short contract note to `src/neural_abm/README.md` pointing to this
  checklist and the boundary audit.
- Keep Toy2/Toy4/Toy5 policy-unit guard tests active.
- Require a docs update whenever generic unit APIs gain new responsibilities.

Result:

- New unit changes can be classified as lifecycle, typed exchange, backend
  dispatch, diagnostics, or explicit contract-gap remediation.
- The README now states that generic unit code must not construct rewards,
  payoffs, thresholds, teacher signals, basin credit, readiness meaning,
  revision pressure meaning, or evidence criteria.
- The active guard surface is explicit: `tests/test_spatial_binary_runner.py`
  for unit-level binary lifecycle tests, `tests/test_toy2_runner.py`,
  `tests/test_toy4_runner.py`, and `tests/test_toy5_runner.py` for
  policy-unit adoption, `tests/test_nabm_unit_adapter_holdout.py` for
  adapter-only extensibility smoke, `tests/test_readiness.py` for readiness
  propagation, and `tests/test_nabm_unit_docs.py` for documentation
  boundaries.

Completion condition update:

- Gate 1 is complete enough for v1 infrastructure work.
- Any future generic unit API expansion should update Decision 0010, the
  boundary audit, and this checklist in the same patch.

### Gate 2: Hard Holdout Expansion

Goal: turn Toy5 from a single hard holdout into bounded robustness evidence.

Status: first pass complete.

Artifacts:

- `experiments/evidence/toy5_neural_threshold_target_threshold_aware_grid_quick.yaml`
- `experiments/evidence/results/toy5_neural_threshold_target_threshold_aware_grid_quick.summary.md`
- `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_grid_quick_findings.md`

Completed work:

- Add a small topology/threshold grid for the threshold-aware Toy5 path.
- Preserve no-seed safety cases.
- Report baseline, main, and negative-control results separately.

Result:

- Gate status: `pass`.
- Main threshold-aware path: `5/5` final ceiling hits on no-seed safety and all
  six spread cases.
- Baseline output-average path: safe in no-seed, `0/5` final ceiling hits in
  all six spread cases.
- Negative controls: non-directional no-seed control fails safety, but
  exposure-anchor controls also achieve `5/5` final hits in all seeded spread
  cases.

Completion condition update:

- The bounded robustness claim is now supported for lattice `k=4`, lattice
  `k=6`, and rewired `k=6, p=0.10` at high thresholds `0.85` and `0.95`.
- A stronger uniqueness claim for threshold-aware direction remains open.

### Gate 3: Toy2/Toy4 Evidence Triage

Goal: separate algorithmic failure from gate brittleness and baseline-fit
effects.

Status: first pass complete.

Artifacts:

- `experiments/results/nabm_effect_matrix/evidence_profile_index_gate3.md`
- `experiments/results/nabm_effect_matrix/evidence_profile_index_calibration.md`
- `experiments/results/nabm_effect_matrix/toy24_revision_operator_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_basin_credit_objective_blend_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_revision_operator_precommitment_controls_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_open_boundary_sparse_seed_stress_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy4_hetero_local_obs_stress_quick_profile.md`
- `experiments/results/nabm_effect_matrix/toy24_gate3_evidence_triage_findings.md`
- `experiments/results/nabm_effect_matrix/toy24_precommitment_peer_evidence_reputation_fragility_stress_quick_findings.md`
- `experiments/results/nabm_effect_matrix/toy4_hetero_local_obs_stress_quick_findings.md`
- refreshed gate summaries with `trajectory_status` and `failure_mode` fields
  for the same Toy2/Toy4 manifest family.

Completed work:

- Keep final-epoch ceiling failures distinct from trajectory-level convergence.
- Report pure stochastic final flips separately from mechanism failures.
- Avoid adding more policy losses until the failure mode requires one.

Result:

- Toy2/Toy4 results can be classified as success, stochastic gate brittleness,
  baseline-favored environment, or true mechanism failure.
- `toy24_revision_operator_quick` is classified as stochastic gate brittleness
  plus baseline-favored environment for both Toy2 and Toy4.
- `toy24_basin_credit_objective_blend_quick` separates Toy2 slow TtC gate lag
  from Toy4 success.
- Precommitment/peer-evidence variants recover stable final hits, but several
  cases remain baseline-favored because reputation imitation reaches the same
  ceiling faster.
- The combined reputation-fragility stress now gives a positive targeted
  contrast: noisy reputation, sparse initial action seeds, and open boundaries
  drop reputation imitation to `0/5` final and ever ceiling hits in both Toy2
  and Toy4, while the peer-evidence candidate stays at `5/5` with mean TtC
  `9.4` and `9.0`.
- The Toy4 heterogeneous local-observation stress extends the resource-local
  line: with checkerboard extraction heterogeneity and noisy reputation, the
  local-sustain resource-threshold candidate stays at `5/5` with mean TtC
  `31.8`, while noisy reputation imitation is `3/5` and the population
  threshold negative control is `0/5`.
- Gate JSON/Markdown now reports trajectory outcome separately from pass/fail,
  so final-epoch brittleness is visible without changing the gate threshold.

Completion condition update:

- Gate 3 is complete enough for first-pass evidence triage.
- The next Toy2/Toy4 evidence step should stop adding observation-mode variants
  and either promote bounded claims into the manuscript claim matrix or design a
  non-reputation baseline-fragility stress where the clean hand-coded baseline
  fails because the environment objective changes rather than because its
  ranking signal is externally noised.

### Gate 4: Manuscript Claim Matrix

Goal: turn the codebase into a paper-ready evidence package.

Status: first pass complete.

Artifacts:

- `paper/claim-matrix.md`
- `paper/tables/nabm-unit-v1-manuscript-tables.md`
- `paper/sections/03-neural-abm-node.md`
- `paper/sections/06-calibration-and-analysis.md`
- `paper/outline.md`

Completed work:

- Create a claim-to-artifact table linking each claim to code path, manifest,
  result, figure, and limitation.
- Promote only bounded claims to the paper outline.
- Keep prototype mechanisms out of the primary claim path unless their evidence
  is upgraded.

Result:

- The primary paper path is now limited to unit-contract reuse, Toy5
  threshold-aware readiness robustness, Toy2/Toy4 failure-mode triage, targeted
  Toy2/Toy4 reputation-fragility stress, and Toy4 local resource-threshold
  robustness.
- Draft manuscript tables now exist for each primary claim path and carry their
  required limitations with the table content.
- Draft prose now exists for the architecture section and calibration/analysis
  section, with explicit insertion notes that keep architecture, diagnostic,
  and robustness claims separate.
- Deferred claims are explicit: solved revision operators, finalized basin
  critics, general superiority over classical baselines, and Toy6-Toy10 as full
  NABM evidence cases.

Completion condition:

- First pass complete. A reviewer can trace each listed paper claim to a
  reproducible artifact, a stated limitation, a draft manuscript table, and
  draft prose. The remaining Gate 4 work is to convert the draft prose into the
  final paper style and add publication figures.

### Gate 5: Adapter-Only Holdout Evidence

Goal: test the next generality claim before adding another full toy: can a new
domain adapter use the shared binary lifecycle without changing `src/neural_abm`?

Status: quick evidence complete.

Artifacts:

- `tests/test_nabm_unit_adapter_holdout.py`
- `experiments/evidence/adapter_only_threshold_holdout_quick.yaml`
- `scripts/run_adapter_holdout_evidence.py`
- `experiments/results/nabm_effect_matrix/adapter_only_threshold_holdout_quick_runs.csv`
- `experiments/evidence/results/adapter_only_threshold_holdout_quick.summary.md`
- `experiments/results/nabm_effect_matrix/adapter_only_threshold_holdout_quick_findings.md`
- `experiments/evidence/adapter_only_congestion_holdout_quick.yaml`
- `scripts/run_adapter_congestion_holdout_evidence.py`
- `experiments/results/nabm_effect_matrix/adapter_only_congestion_holdout_quick_runs.csv`
- `experiments/evidence/results/adapter_only_congestion_holdout_quick.summary.md`
- `experiments/results/nabm_effect_matrix/adapter_only_congestion_holdout_quick_findings.md`
- `src/neural_abm/README.md`

Completed work:

- Added an in-memory threshold-like holdout domain in tests rather than a new
  source-level toy.
- Routed that domain through `run_binary_policy_learning_step(...)` using only
  domain-supplied callbacks for observations, decision probabilities, action
  sampling, local update, and cache refresh.
- Routed the same holdout through `BinaryReadinessPropagationUnit` with
  domain-supplied readiness, active, direction, and peer-neighborhood arrays.
- Added a quick manifest with baseline, negative-control, and main variants.
- Ran the manifest and preserved run CSV, summary, and findings artifacts.
- Added a second adapter-only congestion/capacity manifest where the target is
  balanced allocation rather than threshold cascade.

Result:

- The binary policy lifecycle and readiness propagation can support a new
  adapter-only domain smoke test without changing generic unit code or adding
  hidden domain semantics to `src/neural_abm`.
- The quick manifest passes: no-seed baseline/main preserve zero adoption,
  thresholdless negative control self-excites, sparse-seed baseline stalls at
  the seed, and the adapter-threshold-readiness main reaches full adoption.
- The congestion manifest also passes: imitation and global pressure overcrowd,
  while the adapter capacity policy reaches zero capacity error in symmetric,
  asymmetric, and noisy-preference cases.
- This strengthens the v1 extensibility claim beyond smoke level and beyond a
  threshold-isomorphic holdout, but both cases remain tiny scripted binary
  domains. They are not evidence that the framework is general-purpose.

Completion condition:

- Quick evidence complete for two tiny adapter-only binary holdouts. A stronger
  generality claim still requires a richer holdout domain and manuscript-ready
  analysis.

### Gate 6: Stochastic Endogenous Holdout Evidence

Goal: test whether a source-free adapter can use the shared binary lifecycle in
a closed-loop ABM where actions change future environment state.

Status: quick evidence complete.

Artifacts:

- `experiments/evidence/adapter_only_stochastic_commons_quick.yaml`
- `scripts/run_adapter_stochastic_commons_holdout_evidence.py`
- `experiments/results/nabm_effect_matrix/adapter_only_stochastic_commons_quick_runs.csv`
- `experiments/evidence/results/adapter_only_stochastic_commons_quick.summary.md`
- `experiments/results/nabm_effect_matrix/adapter_only_stochastic_commons_quick_findings.md`

Completed work:

- Added a source-free stochastic commons runner outside `src/neural_abm`.
- Routed binary harvest/conserve decisions through
  `run_binary_policy_learning_step(...)` using domain-owned callbacks.
- Let actions deplete local resources, conservation regenerate resources, and
  stochastic shocks perturb local stocks.
- Compared greedy harvest, global-pressure, and local-resource adapter variants.
- Preserved CSV, JSON, Markdown summary, and findings artifacts.

Result:

- The manifest passes in steady-regeneration, localized-shock, and
  heterogeneous-need cases.
- Greedy harvest collapses in every case, and the global-pressure negative
  control also dips below the resource floor.
- The local-resource main avoids collapse in every seed, keeps mean harvest
  active rather than degenerating to always-conserve, and recovers after shock
  cases.
- This is stronger than the fixed threshold and capacity holdouts because the
  state is endogenous. It remains a compact scripted binary commons, not a
  general-purpose ABM framework proof.

Completion condition:

- Quick evidence complete for a closed-loop adapter-only binary ABM. A stronger
  claim still requires a less scripted holdout or migration of a larger
  existing toy through the same unit contract.

### Gate 7A: Existing-Toy Migration Candidate Audit

Goal: choose the next migration target from existing Toy6-10 runners before
adding another scripted holdout or starting a risky full rewrite.

Status: audit complete.

Artifacts:

- `docs/nabm-unit-v1-migration-candidate-audit.md`
- `src/neural_abm/toy_async.py`
- `src/neural_abm/toy_resource.py`
- `src/neural_abm/toy_heterogeneous.py`
- `tests/test_toy8_runner.py`
- `tests/test_toy7_runner.py`
- `tests/test_toy9_runner.py`

Completed work:

- Audited Toy8, Toy9, and Toy7 as candidate existing-toy migration targets.
- Ranked candidates by existing-toy pressure, scriptedness reduction, domain
  boundary clarity, unit-surface fit, parity-test feasibility, and claim safety.
- Selected Toy8 async social-hazard parity as the next Gate 7B target.
- Recorded Toy9 heterogeneous binary probability mixing as the fallback target.
- Deferred Toy7 continuous resource intensity until a continuous scalar unit
  contract gap is explicitly recorded.

Result:

- Toy8 is the best next target because event scheduling and stale-event
  invalidation are structurally different from the adapter-only binary
  holdouts, while the scalar activation-propensity social-mixing slice remains
  separable from event semantics.
- The next implementation should preserve Toy8 event-time RNG, event queue,
  event validity, event application, counters, and CSV/result contracts.
- The first Toy8 claim should be migration parity plus diagnostics boundary, not
  performance improvement.

Completion condition:

- Audit complete. The next implementation slice is Gate 7B: Toy8 async
  social-hazard parity through a unit-backed scalar social mixing path, with
  Toy9 as the fallback if Toy8 requires a generic contract expansion.

### Gate 7B: Toy8 Async Social-Hazard Parity

Goal: migrate one existing Toy8 social-hazard slice through a unit-backed path
without changing Toy8 event scheduling, event validity, event application, RNG
order, counters, or CSV/result contracts.

Status: parity slice complete.

Artifacts:

- `src/neural_abm/mixers.py::apply_scalar_output_average`
- `src/neural_abm/toy_async.py::apply_output_average`
- `tests/test_social_block.py::test_scalar_output_average_unit_helper_matches_common_block`
- `tests/test_toy8_runner.py::test_toy8_output_average_matches_unit_scalar_parity`
- `tests/test_toy8_runner.py::test_toy8_output_average_routes_through_unit_scalar_helper`

Completed work:

- Added a semantic-free scalar social helper backed by `NABMStep`.
- Routed Toy8 activation-propensity output averaging through that helper.
- Preserved Toy8 hazard construction, event queue scheduling, stale-event
  invalidation, event application, event counters, and aggregate/micro field
  names.
- Added parity coverage for mixed activation propensities, social losses, and
  update norms.

Result:

- The Toy8 social-hazard mixing slice now enters the reusable unit surface
  without moving event-time semantics into the unit.
- The test suite confirms parity against the old scalar social block behavior
  and confirms Toy8 calls the unit-backed helper with the `activation_propensity`
  channel and `event_hazard_commit` commit mode.
- This is existing-toy migration parity, not new Toy8 performance evidence.

Completion condition:

- First existing-toy migration parity slice complete. The next Toy8 step should
  add runner-level artifact parity only if needed; otherwise the next structural
  pressure point is Toy9 fallback migration or a recorded Toy7 continuous-scalar
  contract gap.

### Gate 7C: Toy9 Heterogeneous Probability Parity

Goal: reuse the same unit-backed scalar social path in a second existing toy
without moving Toy9 group assignment, local rules, payoff computation,
propensity learning, payoff EMA, or coordination gating into the unit.

Status: parity slice complete.

Artifacts:

- `src/neural_abm/mixers.py::apply_scalar_output_average`
- `src/neural_abm/toy_heterogeneous.py::apply_output_average`
- `tests/test_toy9_runner.py::test_toy9_output_average_matches_unit_scalar_parity`
- `tests/test_toy9_runner.py::test_toy9_output_average_routes_through_unit_scalar_helper`

Completed work:

- Routed Toy9 heterogeneous action-probability output averaging through the
  `NABMStep`-backed scalar helper.
- Preserved group assignment, group-specific local-rule semantics, payoff
  computation, propensity updates, payoff EMA, action sampling, and CSV/result
  field names.
- Added parity coverage for mixed action probabilities, social losses, and
  update norms.

Result:

- The scalar unit path is now reused by Toy8 and Toy9 existing-toy migration
  slices.
- The Toy9 slice confirms the helper is called with the
  `heterogeneous_action_probability` channel and
  `group_gated_probability_commit` commit mode.
- This is existing-toy migration parity, not evidence that Toy9 is a full NABM
  claim path.

Completion condition:

- Second existing-toy migration parity slice complete. Further scalar migration
  work should be justified by reducing real duplication. The next structural
  pressure point is either Toy7 continuous-scalar contract-gap documentation or
  manuscript consolidation of the current migration evidence.

### Gate 7D: Continuous-Scalar Contract Decision

Goal: decide whether Toy7 continuous extraction intensity can reuse the current
scalar probability path before implementation.

Status: contract decision complete.

Artifacts:

- `docs/decisions/0011-continuous-scalar-unit-contract.md`
- `docs/nabm-unit-v1-migration-candidate-audit.md`
- `src/neural_abm/toy_resource.py`
- `tests/test_toy7_runner.py`
- `tests/test_nabm_unit_docs.py`

Completed work:

- Recorded Toy7 extraction intensity as a bounded continuous scalar, not a
  probability.
- Rejected `SCALAR_PROBABILITY_CHANNEL` as the Toy7 migration path because it
  would blur channel semantics.
- Defined the next preferred API shape as `BOUNDED_SCALAR_CHANNEL` plus a
  semantic-free `mix_bounded_scalars(...)` helper or equivalent.
- Kept resource stock dynamics, payoff construction, noisy intensity sampling,
  and evidence interpretation in Toy7.

Result:

- Toy7 remains deferred until a bounded continuous scalar contract exists and
  is tested outside Toy7.
- The next Toy7 implementation should be parity-only: first add the bounded
  scalar helper and guard tests, then route only Toy7 social intensity mixing
  through it.
- This gate does not promote Toy7 to a full NABM evidence case and does not
  claim continuous-action policy learning is solved.

Completion condition:

- Decision complete. The next implementation slice is Gate 7E: bounded-scalar
  unit contract prototype plus Toy7 social-intensity parity, if the project
  chooses to keep migrating existing compatible toys before manuscript
  consolidation.

### Gate 7E: Toy7 Bounded-Scalar Intensity Parity

Goal: implement the bounded continuous scalar contract and route only Toy7
social extraction-intensity mixing through it.

Status: parity slice complete.

Artifacts:

- `src/neural_abm/social.py::BOUNDED_SCALAR_CHANNEL`
- `src/neural_abm/social.py::mix_bounded_scalars`
- `src/neural_abm/social.py::select_bounded_scalar_output_peers`
- `src/neural_abm/mixers.py::apply_bounded_scalar_output_average`
- `src/neural_abm/toy_resource.py::apply_output_average`
- `tests/test_social_block.py::test_bounded_scalar_output_average_unit_helper_matches_common_block`
- `tests/test_toy7_runner.py::test_toy7_output_average_matches_unit_bounded_scalar_parity`
- `tests/test_toy7_runner.py::test_toy7_output_average_routes_through_unit_bounded_scalar_helper`

Completed work:

- Added a semantic-free bounded scalar channel and validation path that rejects
  non-finite values and values outside declared bounds.
- Added bounded scalar output-similarity peer selection without naming the
  values probabilities.
- Added a `NABMStep`-backed bounded scalar output-average helper.
- Routed Toy7 extraction-intensity social averaging through that helper with
  bounds `[0, 1]`, channel `extraction_intensity`, and commit mode
  `continuous_intensity_commit`.
- Preserved Toy7 resource dynamics, payoff construction, noisy intensity
  sampling, propensity updates, and CSV/result field names.

Result:

- Toy7 is no longer using `SCALAR_PROBABILITY_CHANNEL` for extraction
  intensity.
- The bounded scalar path is tested independently of Toy7 and then covered by a
  Toy7 parity/routing test.
- This is existing-toy migration parity only. It is not evidence that Toy7 is a
  full NABM claim path and not evidence that continuous-action NABMs are solved.

Completion condition:

- Gate 7E is complete for the social-intensity slice. Further Toy7 work should
  require a new decision if it moves from bounded scalar social mixing into
  continuous local policy learning or resource-control evidence.

### Gate 7F: Toy10 Market/Ecology Channel Parity

Goal: decide whether Toy10 needs a new multi-channel vector contract before
migration, then route the smallest safe slice through the existing bounded
scalar unit path.

Status: parity slice complete.

Artifacts:

- `src/neural_abm/toy_market.py::select_peer_ids`
- `src/neural_abm/toy_market.py::mix_channel`
- `src/neural_abm/mixers.py::apply_bounded_scalar_output_average`
- `tests/test_toy10_runner.py::test_toy10_output_similarity_selects_bounded_scalar_composite`
- `tests/test_toy10_runner.py::test_toy10_mix_channel_matches_unit_bounded_scalar_parity`
- `tests/test_toy10_runner.py::test_toy10_mix_channel_routes_through_unit_bounded_scalar_helper`

Completed work:

- Audited Toy10 as two bounded scalar social channels, `price_expectation` and
  `conservation_norm`, plus a Toy-owned composite similarity selector.
- Reused `select_bounded_scalar_output_peers(...)` for the composite
  market/ecology similarity score instead of treating it as a probability.
- Routed each Toy10 social channel through
  `apply_bounded_scalar_output_average(...)` with bounds `[0, 1]` and commit
  mode `multi_channel_market_commit`.
- Kept multi-channel aggregation, harvest construction, market price,
  resource dynamics, payoff updates, dynamic rewiring, and evidence
  interpretation in Toy10.

Result:

- Toy10 no longer uses scalar probability mixing for price/conservation social
  channels.
- A new vector-valued multi-channel message contract is not needed for this
  parity slice because Toy10 can apply the bounded scalar unit path once per
  channel.
- This is not a general multi-channel NABM claim. A future vector/mapping
  channel would require a new decision if multiple domains need atomic
  multi-field message mixing.

Completion condition:

- Gate 7F is complete for Toy10 price/conservation social mixing. Further
  Toy10 work should not move market, resource, payoff, or dynamic rewiring
  semantics into the unit.

### Gate 7G: Toy6 Categorical Distribution Parity

Goal: decide whether Toy6 categorical social mixing needs a new categorical
policy channel or can reuse the existing probability-distribution channel.

Status: parity slice complete.

Artifacts:

- `src/neural_abm/mixers.py::apply_distribution_output_average`
- `src/neural_abm/toy_categorical.py::apply_output_average`
- `tests/test_social_block.py::test_distribution_output_average_unit_helper_matches_common_block`
- `tests/test_toy6_runner.py::test_toy6_output_average_matches_unit_distribution_parity`
- `tests/test_toy6_runner.py::test_toy6_output_average_routes_through_unit_distribution_helper`

Completed work:

- Audited Toy6 categorical policy output as a row-stochastic probability
  distribution, not a separate generic categorical-policy semantic.
- Added a `NABMStep`-backed distribution output-average helper over
  `PROBABILITY_DISTRIBUTION_CHANNEL`.
- Routed Toy6 `strategy_distribution` social averaging through that helper
  with commit mode `categorical_probability_commit`.
- Preserved cyclic payoff construction, local logit updates, action sampling,
  payoff EMA, strategy entropy metrics, and artifact fields in Toy6.

Result:

- Toy6 does not need a new categorical-specific unit contract for this parity
  slice.
- The existing distribution channel is sufficient because the unit sees only
  finite row-stochastic distributions; Toy6 owns strategy meaning and payoff
  semantics.
- This is existing-toy migration parity only. It does not promote Toy6 to full
  NABM evidence and does not claim categorical ABMs are solved.

Completion condition:

- Gate 7G is complete for Toy6 categorical output averaging. Future Toy6 work
  should require a separate decision if it moves local categorical policy
  learning, payoff construction, or categorical evidence criteria into the
  generic unit.

### Gate 7H: Existing-Toy Migration Consolidation

Goal: summarize Gates 7B-7G as one engineering boundary before starting new
parity-only migration work.

Status: documentation consolidation complete.

Artifacts:

- `docs/decisions/0012-existing-toy-migration-parity-consolidation.md`
- `docs/toy-models/capability-matrix.md`
- `docs/nabm-unit-v1-migration-candidate-audit.md`
- `src/neural_abm/README.md`
- `tests/test_nabm_unit_docs.py`

Completed work:

- Consolidated Toy6-10 social-exchange parity into one decision record.
- Kept the Toy6-10 status as `compatible`, not `full`.
- Recorded the unit-owned surfaces separately from toy-owned domain semantics.
- Updated the capability matrix so future readers can see which typed channel
  each compatible toy now exercises.

Result:

- The engineering claim is now bounded: existing compatible toys can reuse
  typed NABM Unit social exchange surfaces without moving domain equations into
  the generic layer.
- The research claim is still conservative: these parity slices are not
  performance evidence and do not make Toy6-10 evidence-default.

Completion condition:

- Gate 7H is complete. Further engineering work should either consolidate
  manuscript architecture claims or target runner lifecycle duplication rather
  than adding more parity-only social mixing slices.
- Runner lifecycle consolidation audit is the next recorded Gate 8A slice.

### Gate 8A: Runner Lifecycle Consolidation Audit

Goal: decide whether the next structural step should be a full runner rewrite,
more unit migration, or a smaller extraction around repeated adapter lifecycle
work.

Status: audit complete.

Artifacts:

- `docs/nabm-unit-v1-runner-lifecycle-audit.md`
- `src/neural_abm/domain_runner.py`
- `src/neural_abm/toy_categorical.py`
- `src/neural_abm/toy_resource.py`
- `src/neural_abm/toy_async.py`
- `src/neural_abm/toy_heterogeneous.py`
- `src/neural_abm/toy_market.py`
- `tests/test_nabm_unit_docs.py`

Completed work:

- Audited `DomainRunSettings`, `DomainToyAdapter`, and `DomainToyRunner`.
- Recorded that `DomainToyRunner` already owns run directory creation,
  metadata artifacts, `aggregate_metrics.csv`, `micro_state.csv`, adapter
  initialization, epoch loop, fallback, final summary writing, and result
  envelope creation.
- Compared Toy6, Toy7, Toy8, Toy9, and Toy10 inner `step(...)` lifecycles
  after typed social-exchange parity.
- Identified the repeated row-mapping surface around `peer_count`,
  `mean_peer_count`, `mean_social_loss`, and `mean_social_update_norm`.

Result:

- A full runner rewrite is not the right next step. The common outer lifecycle
  is already consolidated enough for compatible Toy6-10 runners.
- The remaining duplication is mostly in toy adapters, especially social
  diagnostics row mapping, while payoff, resource, event, market, group, and
  categorical semantics must remain toy-owned.
- Do not unify Toy6-10 step order; the phase order is part of each domain's
  semantics.

Completion condition:

- Gate 8A is complete. The next implementation slice should be Gate 8B: Social
  Diagnostics Mapper Prototype, not a full runner rewrite.

### Gate 8B: Social Diagnostics Mapper Prototype

Goal: extract only the repeated peer/social row-mapping fields from compatible
toy adapters, without changing runner ownership or moving domain semantics into
the shared layer.

Status: complete for Toy6-10 diagnostics rows.

Artifacts:

- `src/neural_abm/domain_social_diagnostics.py`
- `src/neural_abm/toy_categorical.py`
- `src/neural_abm/toy_resource.py`
- `src/neural_abm/toy_async.py`
- `src/neural_abm/toy_heterogeneous.py`
- `src/neural_abm/toy_market.py`
- `tests/test_domain_social_diagnostics.py`
- `tests/test_toy6_runner.py::test_toy6_rows_route_social_diagnostics_through_mapper`
- `tests/test_toy7_runner.py::test_toy7_rows_route_social_diagnostics_through_mapper`
- `tests/test_toy8_runner.py::test_toy8_rows_route_social_diagnostics_through_mapper`
- `tests/test_toy9_runner.py::test_toy9_rows_route_social_diagnostics_through_mapper`
- `tests/test_toy10_runner.py::test_toy10_rows_route_social_diagnostics_through_mapper`
- `docs/nabm-unit-v1-runner-lifecycle-audit.md`
- `src/neural_abm/README.md`

Completed work:

- Added `aggregate_social_diagnostic_fields(...)` for `mean_peer_count`,
  `mean_social_loss`, and `mean_social_update_norm`.
- Added `micro_social_diagnostic_fields(...)` for `peer_ids`, `peer_count`,
  optional toy-supplied `component_id`, `social_loss`, and
  `social_update_norm`.
- Migrated Toy6 categorical, Toy7 resource, Toy8 async, Toy9 heterogeneous,
  and Toy10 market/ecology adapters to the mapper.
- Added toy-independent mapper tests plus Toy6-Toy10 routing tests.

Result:

- The repeated peer/social diagnostics surface is now shared across Toy6-10
  without changing `DomainToyRunner`, `DomainToyAdapter`, or toy `step(...)`
  phase order.
- Payoff, resource, event, group, market, categorical strategy, action
  sampling, graph construction, and evidence semantics remain toy-owned.
- This is an engineering consolidation slice, not performance evidence and not
  a claim that Toy6-10 are full NABM evidence cases.

Completion condition:

- Gate 8B is complete for Toy6-10 compatible-toy diagnostics rows.

## Recommended Next Slice

The next implementation slice should avoid another broad parameter sweep. Two
paths are now useful:

- Compatible-toy adapter thinness audit: inspect whether any remaining
  Toy6-10 adapter boilerplate can be safely reduced without changing
  `step(...)` phase order, domain row schemas, event queues, group logic,
  market/resource updates, or dynamic-graph semantics.
- Shared artifact-contract tests: compare representative Toy6-10 aggregate and
  micro rows against stable expected field sets so future cleanup does not
  silently change CSV contracts.
