# NABM Unit v1 Boundary Audit

Date: 2026-05-21

## Purpose

This audit translates Decision 0010 into concrete Toy2/Toy4 ownership rules.
The goal is to prevent the generic NABM unit from absorbing domain semantics
while still making repeated lifecycle code reusable.

## Shared Unit Surfaces

| Surface | Code path | Current role |
| --- | --- | --- |
| Generic local/social unit | `src/neural_abm/unit.py` | Owns generic local update delegation, social-message validation, typed social mix, commit adapters, and unit reports. |
| Binary policy unit | `src/neural_abm/spatial_binary.py::BinaryPolicyLearningUnit` | Owns binary neural policy readout, action-probability construction, sampling callback, local commit callback, refresh callback, and post-readout ordering. |
| Binary policy callbacks | `src/neural_abm/spatial_binary.py::BinaryPolicyLearningCallbacks` | Defines the adapter boundary for domain-specific policy learning. |
| Binary revision unit | `src/neural_abm/binary_revision.py::BinaryRevisionLearningUnit` | Owns optional stay/switch lifecycle when enabled. |
| Binary revision callbacks | `src/neural_abm/binary_revision.py::BinaryRevisionLearningCallbacks` | Keeps revision signal meaning and sampling semantics in domain callbacks. |
| Readiness propagation | `src/neural_abm/readiness.py::BinaryReadinessPropagationUnit` | Owns peer-readiness aggregation only after a domain has defined readiness. |
| Binary social distillation | `src/neural_abm/spatial_binary.py::run_binary_output_distribution_distillation` | Owns typed output-distribution social mix and backend commit dispatch. |
| Backend commit adapters | `src/neural_abm/spatial_binary.py` and `src/neural_abm/unit.py` | Hide loop, batched, and tensor-runtime commit differences without owning objectives. |

## Toy2 Boundary

Toy2 is a full NABM toy using the binary unit lifecycle, but the game remains
domain-owned.

### Unit-Owned in Toy2

- policy readout and post-local readout ordering through
  `BinaryPolicyLearningUnit`;
- optional stay/switch sequencing through `BinaryRevisionLearningUnit`;
- action-probability to revision-probability mapping when
  `revision_operator_source=policy_probability`;
- backend local-update commit dispatch through binary local update adapters;
- output-distribution social distillation through the shared binary social
  distillation path;
- common aggregate/micro fields for policy probabilities, revision choices,
  readiness, social channel, commit mode, and update norms.

### Toy2-Owned and Must Stay Domain-Specific

- graph and interaction-neighbor construction;
- Prisoner's Dilemma or stag-hunt payoff family and payoff thresholds;
- pairwise payoff recomputation after sampled actions or revision choices;
- counterfactual cooperation/defection advantage construction;
- state-continuation component semantics;
- basin-credit handoff, counterfactual basin scoring, learned-basin runtime,
  and basin training diagnostics;
- decision bootstrap and distill bootstrap teacher construction from
  reputation imitation;
- teacher-alignment diagnostics and gradient-conflict checks;
- evidence case definitions and ceiling criteria.

### Toy2 Extraction Candidates

These may be extracted later only if the extracted surface remains semantic-free:

- a small helper for constructing policy-probability revision callbacks;
- a shared diagnostic merge helper for revision aggregate/micro fields;
- a common bootstrap-diagnostic plumbing helper that does not choose the
  teacher policy or the objective components.

Do not extract Toy2 counterfactual payoff logic into the unit.

## Toy4 Boundary

Toy4 uses the same binary lifecycle but has stronger domain coupling because
public-goods rewards and resource dynamics affect both observations and
advantages.

### Unit-Owned in Toy4

- policy readout and post-local readout ordering through
  `BinaryPolicyLearningUnit`;
- optional stay/switch sequencing through `BinaryRevisionLearningUnit`;
- policy-probability to stay/switch probability mapping for the opt-in
  revision path;
- backend local-update commit dispatch for loop, batched, and tensor-batched
  branches;
- output-distribution social distillation through the shared binary social
  distillation path;
- readiness peer aggregation after Toy4 has defined readiness and direction;
- common diagnostics for local, social, revision, and readiness fields.

### Toy4-Owned and Must Stay Domain-Specific

- public-goods group construction and group payoff computation;
- resource stock update, recovery, extraction, heterogeneity, and collapse
  threshold;
- `resource_observation_mode`, including `global`, `hidden`, and
  `local_sustain`;
- local versus population resource-threshold meaning;
- contribution advantage, welfare, environment, and threshold components;
- reputation-imitation teacher construction used by bootstrap diagnostics;
- basin-credit handoff, counterfactual basin scoring, learned-basin runtime,
  and basin training diagnostics;
- teacher-alignment diagnostics and gradient-conflict checks;
- evidence case definitions and ceiling criteria.

### Toy4 Extraction Candidates

These may be extracted later only if they reduce lifecycle duplication without
moving resource semantics:

- a shared callback builder for opt-in policy-probability revision paths;
- a shared post-local diagnostic merge helper;
- a generic "domain extras to aggregate/micro rows" protocol.

Do not extract resource-threshold, local-sustain, or resource-transition logic
into the unit.

## Cross-Toy Duplication Assessment

| Repeated pattern | Extract now? | Reason |
| --- | --- | --- |
| `BinaryPolicyLearningCallbacks` construction shape | Already extracted enough | The callback container is the right boundary; deeper extraction risks hiding domain objective construction. |
| Policy-probability revision callback shape | Maybe later | Toy2 and Toy4 duplicate the same structural mapping, but action sampling still mutates different domain state. |
| Bootstrap teacher scheduling and diagnostics | No | The helper functions are shared, but each toy must choose teacher probabilities and objective components. |
| Basin-credit diagnostics handoff | No | Shared diagnostic helpers are enough; scoring and replay selection depend on domain phase representations. |
| Resource-threshold logic | No | This is Toy4 mechanism evidence, not a generic NABM unit behavior. |
| Counterfactual advantage logic | No | This is Toy2 game semantics, not lifecycle infrastructure. |
| Social distillation backend dispatch | Already extracted | The unit path owns typed social mix and commit adapter dispatch. |
| Readiness peer evidence aggregation | Already extracted enough | The unit can aggregate readiness values, but domains define readiness and direction. |

## Holdout Migration Gate

The next meaningful structural test should not be another Toy4 parameter
stress. It should ask whether a domain can plug into the v1 unit without
changing generic unit code.

Minimum holdout acceptance criteria:

1. The holdout uses `BinaryPolicyLearningUnit` or `NABMUnit` directly.
2. Domain code supplies observations, objective inputs, local commit callback,
   social message values, peer rule, and diagnostics mapper.
3. The generic unit code does not gain new Toy-specific config fields.
4. The quick manifest includes a hand-coded baseline, the unit path, and a
   negative control.
5. The findings document classifies the result as speed, stability,
   robustness, or interpretability evidence.

If the holdout requires changing the generic unit, the change must be recorded
as a contract gap before running a larger evidence workflow.

## Practical Next Slice

The next implementation slice should be one of:

1. Migrate a holdout binary path that already has clear domain callbacks.
2. Add a typed adapter protocol for diagnostics mapping if Toy2/Toy4/Toy5
   continue duplicating aggregate/micro merge code.
3. Add a small contract test that instantiates a toy-independent
   `BinaryPolicyLearningUnit` fixture and verifies callback ordering, backend
   refresh, post-readout, and diagnostics without Toy2 or Toy4 imports.

The preferred order is 3, then 1, then 2. This gives a cleaner contract test
before adding another migration.

## 2026-05-21 Contract Fixture Slice

Added a toy-independent `BinaryPolicyLearningUnit` contract fixture in
`tests/test_spatial_binary_runner.py`.

The fixture verifies:

- policy readout -> decision probability construction -> action sampling ->
  local commit -> cache refresh -> post-local readout ordering;
- context timing stages for `policy_readout`, `decision_selection`,
  `local_training`, `cache_refresh`, and `post_local_readout`;
- backend synchronization hooks around each timed stage;
- decision probabilities passed through to sampling without domain imports;
- result preservation for pre-readout probabilities, decision probabilities,
  selected actions, local losses, post-local probabilities, and copied extras.

This completes the contract-test step before holdout migration. The next
implementation slice should choose a holdout path and attempt an adapter-only
migration without changing the generic unit code.

## 2026-05-21 Toy5 Holdout Adapter Slice

Promoted Toy5 as the first v1 holdout path.

Implementation guard:

- Added a Toy5 local-step spy test that patches `toy_contagion.BinaryPolicyLearningUnit`
  and verifies the neural Toy5 path instantiates the shared unit with
  `BinaryPolicyLearningCallbacks`.
- The test confirms Toy5 supplies domain-owned callbacks for policy readout,
  decision probability construction, action sampling, local update, and cache
  refresh, while preserving unit-owned result fields such as decision
  probabilities, post-local probabilities, and observations.
- No generic unit code was changed.

Evidence guard:

- Re-ran `experiments/evidence/toy5_readiness_propagation_holdout_quick.yaml`.
- Gate status: `pass`.
- Main variant: `neural_readiness_propagation_w1p0`.
- Final ceiling hits: `3/3`.
- Mean time to ceiling: `1.333`.
- Baseline improvement: `false`, because `neural_output_average` is already at
  endpoint ceiling.

Mechanism interpretation:

- This is not a Toy5 endpoint-performance win.
- It is a structural holdout confirmation that the unit/readiness lifecycle
  works outside Toy2/Toy4.
- The observable difference is internal readiness timing:
  `neural_precommitment_evidence` reaches all-ready at mean epoch `21.0`,
  while `neural_readiness_propagation_w1p0` reaches all-ready at mean epoch
  `4.667`.

The next holdout should be harder than this saturated Toy5 setting. It should
stress a case where the neural output-average baseline is not already at the
endpoint ceiling, so the unit lifecycle can be evaluated for robustness rather
than only internal timing.

## 2026-05-21 Toy5 Hard Holdout Stress Slice

Reused the existing threshold-aware wavefront stress as the first non-saturated
Toy5 v1 holdout evidence.

Artifacts:

- Manifest:
  `experiments/evidence/toy5_neural_threshold_target_threshold_aware_wavefront_quick.yaml`
- Gate summary:
  `experiments/evidence/results/toy5_neural_threshold_target_threshold_aware_wavefront_quick.summary.md`
- Profile:
  `experiments/results/nabm_effect_matrix/toy5_neural_threshold_target_threshold_aware_wavefront_quick_profile.md`
- Profile index rows:
  `experiments/results/nabm_effect_matrix/evidence_profile_index_calibration.csv`

Evidence guard:

- Re-ran the manifest after the Toy5 unit adapter guard.
- Gate status: `pass`.
- No generic unit code was changed.
- The calibration profile index now includes this manifest, raising the index
  to `21` case rows.

Hard-holdout result:

| Case | Baseline final hits | Main final hits | Main mean TtC | Main metric |
| --- | ---: | ---: | ---: | ---: |
| `toy5_threshold_aware_wavefront_no_seed_heterogeneous_safety` | 5/5 | 5/5 | 0.0 | 1.0 non-adoption |
| `toy5_threshold_aware_lattice_k4_heterogeneous_h0p85_spread` | 0/5 | 5/5 | 36.2 | 100 cascade size |
| `toy5_threshold_aware_lattice_k6_heterogeneous_h0p95_spread` | 0/5 | 5/5 | 25.0 | 100 cascade size |
| `toy5_threshold_aware_rewired_p0p10_heterogeneous_h0p95_spread` | 0/5 | 5/5 | 10.0 | 100 cascade size |

Interpretation:

- This is the first Toy5 holdout slice in this audit where the output-average
  baseline is not already saturated on the spread cases.
- The positive evidence is robustness evidence for a domain-owned
  threshold-aware wavefront adapter running through the shared binary policy
  lifecycle, not a claim that the generic unit owns threshold semantics.
- The no-seed case is a safety guard: output averaging is already safe there,
  but the non-directional diagnostic fails, so the direction gate still
  separates real exposure from self-excitation.
- The threshold-aware main is slightly slower than exposure-only diagnostics in
  lattice cases, which is expected because the direction score subtracts each
  agent's adoption threshold.

Claim boundary:

- The supported claim is bounded to the tested Watts-Strogatz slices:
  `k=4, high=0.85`, `k=6, high=0.95`, and `rewire_probability=0.10,
  high=0.95`.
- Broader Toy5 claims still need a larger topology/threshold grid and
  seed-neighborhood diagnostics. This slice is enough for the v1 unit holdout
  gate because it tests adapter reuse under a non-saturated domain condition.

## 2026-05-21 Domain Diagnostic Plumbing Slice

Extracted the first small adapter-boilerplate helper after the Toy5 hard
holdout gate.

Implementation:

- Added `src/neural_abm/domain_learning_diagnostics.py`.
- The helper centralizes Toy2/Toy4 aggregate and micro CSV plumbing for
  state-continuation, basin-credit, learned-basin, bootstrap, decision replay,
  distill-bootstrap, and teacher-alignment diagnostics.
- Updated Toy2 and Toy4 to call `domain_learning_aggregate_fields(...)` and
  `domain_learning_micro_fields(...)` after computing their own domain fields.

Boundary:

- The helper does not compute payoff, resource, graph, threshold, teacher,
  replay, or basin semantics.
- Toy2 still owns game-family payoff semantics and neighbor payoff/action
  summaries.
- Toy4 still owns public-goods/resource semantics and exploitation/resource
  summaries.
- The helper only maps existing `step_result.extras` payloads through existing
  diagnostic formatter functions.

Verification:

- Added `tests/test_domain_learning_diagnostics.py` to compare the helper
  output against the explicit formatter composition it replaced.
- Ran `uv run ruff check src tests scripts`.
- Ran `uv run pytest tests/test_domain_learning_diagnostics.py
  tests/test_toy2_runner.py tests/test_toy4_runner.py tests/test_toy5_runner.py
  tests/test_spatial_binary_runner.py -q`.

Result:

- `385 passed`.
- No generic unit behavior changed.

## 2026-05-21 Domain Diagnostic Field-List Slice

Followed the diagnostic plumbing extraction with the matching schema cleanup.

Implementation:

- Added `DOMAIN_LEARNING_AGGREGATE_FIELDS` and
  `DOMAIN_LEARNING_MICRO_FIELDS` to
  `src/neural_abm/domain_learning_diagnostics.py`.
- Updated Toy2 and Toy4 CSV field lists to include the shared domain-learning
  field block after their own domain-specific fields.
- Kept Toy2-specific fields such as game family, payoff parameters, policy
  consensus, and neighbor payoff/action summaries in Toy2.
- Kept Toy4-specific fields such as payoff variance, resource state, collapse
  time, exploitation index, and local group summaries in Toy4.

Boundary:

- This is schema plumbing only. It reduces field-list drift between Toy2 and
  Toy4 but does not move domain objective, teacher, replay, basin, payoff, or
  resource semantics into the shared unit.
- The shared field list is generated from the existing formatter outputs, so a
  formatter-level schema change has one source of truth.

Verification:

- Extended `tests/test_domain_learning_diagnostics.py` to assert that Toy2 and
  Toy4 compose their CSV schemas from the shared field-list constants plus
  their domain-specific prefixes.
- Ran `uv run ruff check src tests scripts`.
- Ran `uv run pytest tests/test_domain_learning_diagnostics.py
  tests/test_toy2_runner.py tests/test_toy4_runner.py tests/test_toy5_runner.py
  tests/test_spatial_binary_runner.py -q`.

Result:

- `386 passed`.
- No generic unit behavior changed.
