# Neural ABM Module Workspace

This directory contains reusable implementation modules.

Current modules:

- `accelerator`: torch device resolution plus batched per-agent MLP policy
  inference kernels for GPU-friendly neural/social toy execution.
- `api`: stable v0 facade for reusable lifecycle, typed social exchange,
  compatible runner, diagnostics, result, readiness, and toy feature-taxonomy
  surfaces.
- `api_lite`: torch-free facade seed for compatible runner, diagnostics,
  result, readiness, toy feature-taxonomy, NumPy-only social surfaces, and
  lightweight lifecycle report/local-step surfaces that do not load `torch` at
  import time.
- `binary_neural`: shared binary-policy neural accelerator helpers, including
  the `TensorPolicyRuntime` contract used by capability-gated `tensor_batched`
  runtimes.
- `config`: Pydantic config schemas and YAML loader.
- `core`: common neural classification agent contract and MLP utilities.
- `domain_runner`: compatible-toy outer lifecycle runner for Toy6-10-style
  domain adapters, including run directories, metadata, CSV writers, fallback,
  final summary, result envelopes, and settings-based
  `make_domain_run_dir` / `write_domain_run_metadata` artifact helpers.
- `domain_social_diagnostics`: semantic-free peer/social row mapping helpers
  for compatible domain toy adapters.
- `graphs`: static graph and peer graph helpers.
- `logging`: micro-state and aggregate CSV logging helpers.
- `metrics`: task and social-dynamics metrics.
- `metrics_core`: torch-free numerical metrics shared by lightweight package
  surfaces.
- `mixers`: peer selection plus output, latent, and parameter averaging.
- `mobility`: fixed-cell local-quality mobility and state-channel swaps.
- `readiness`: binary readiness-propagation coordination unit that converts
  prior ready-state scores into peer evidence increments before hard
  commitment.
- `reputation`: action-history EMA and reputation-driven imitation helpers.
- `social`: reusable `SocialBlock`/`SocialChannel` contract, invariants,
  scalar probability, bounded scalar, and distribution-valued output peer
  selection, plus probability, bounded scalar, tensor, and parameter-state
  channel mixing.
- `social_core`: torch-free `SocialChannel`, peer/result dataclasses, peer-id
  utilities, NumPy validators, similarity helpers, peer selection helpers, and
  scalar/bounded scalar mix helpers used by `api_lite` and re-exported by
  `social`.
- `spatial_binary`: shared hook runner for Toy 2, Toy 4, and Toy 5; binary
  action/probability validation; social-mode helpers for probability mixing and
  policy distillation; NumPy/torch state helpers; and common aggregate/micro
  logging helpers.
- `toy_classification`: Toy 1 environment and runner.
- `toy_pd`: Toy 2 Neural Spatial Prisoner's Dilemma environment and runner.
  Its `neural_policy` + `tensor_batched` backend keeps actions, payoffs,
  payoff EMA, and reputation state on the configured torch device while
  preserving the public CSV/result contract. The tensor runtime is initialized
  directly from batched model parameters and zero Adam state for this backend,
  without per-agent module or optimizer construction. Local and social update
  passes use detached trainable views over runtime-owned parameter tensors, and
  sampled local updates pass canonical active-agent ids directly to avoid
  rebuilding all-active update masks. Binary local policy-gradient losses use
  a two-logit `logsigmoid` formulation instead of the general 2-class
  `log_softmax` graph. KL distillation losses keep the reported KL value while
  detaching target entropy from the optimizer graph.
- `toy_opinion`: Toy 3 opinion dynamics, rewiring, and runner.
- `toy_public_goods`: Toy 4 public-goods, commons, and runner.
- `toy_contagion`: Toy 5 contagion, threshold adoption, and runner.
- `toy_categorical`: Toy 6 multi-action categorical spatial game and runner.
- `toy_resource`: Toy 7 continuous extraction-intensity resource ABM and
  runner.
- `toy_async`: Toy 8 asynchronous event-driven adoption/failure/recovery ABM
  and runner.
- `toy_heterogeneous`: Toy 9 heterogeneous-agent binary adoption ABM with
  group-specific local rules and coordination gating.
- `toy_market`: Toy 10 dynamic-network market/ecology ABM with multi-channel
  price and conservation messages plus topology churn.
- `unit`: public NABM lifecycle protocol, `NABMUnit`, `NABMStep`, commit
  adapters, message-to-channel value builders, and social diagnostics.
- `unit_core`: torch-free commit reports, social diagnostics, local-update
  reports, local-step adapter wrapper, and callback type aliases used by
  `api_lite` and re-exported by `unit`.

Torch device selection:

- `simulation.device: cpu` keeps all neural modules on CPU.
- `simulation.device: cuda` or `cuda:0` requires CUDA and fails fast if it is
  unavailable.
- `simulation.device: auto` selects CUDA, then MPS, then CPU. This is useful
  for local accelerator experiments but should be pinned for reproducible
  benchmark reports.

Expected future modules:

- `graphs`: static, dynamic, and peer-similarity graph utilities.
- learned edge mixers and parameter alignment helpers.

Implementation should stay reusable here. Experiment-specific orchestration
belongs in `scripts/` and `experiments/configs/`.

Current agent contract methods:

- `observe`
- `act_or_predict`
- `local_update`
- `social_message`
- `log_state`

Reusable NABM unit lifecycle:

```text
Environment / Probe State
        |
        v
NABMAgent.observe / local_update / social_message
        |
        v
Peer Selection -> PeerSelectionResult(peer_ids, similarity)
        |
        v
SocialBlock.mix(SocialChannel, values, peer_ids)
        |
        v
SocialMixResult(channel, commit_mode, losses, update_norms)
        |
        v
CommitAdapter.commit(...)
        |
        v
Updated Agent State + SocialDiagnostics
```

`NABMUnit` is the migration target for toy runners. It wraps a sequence of
`NABMAgent` instances plus injected peer selection and social-value extraction
callbacks, then returns `NABMUnitReport` with local losses, messages, peer ids,
social step diagnostics, aggregate rows, and micro rows. Existing toy runners
can adopt this without moving domain payoff or environment transition code.
Toy1 `output_average` is the first runner-visible migration slice and now uses
`NABMUnit` internally while preserving its existing CSV/result contract.
Toy2, Toy4, and Toy5 loop-based output-distribution distillation helpers route
through the shared `spatial_binary.apply_binary_output_distribution_distillation`
primitive, which owns the `NABMUnit`/`NABMStep` construction for that binary
social update. The runner-facing path can now carry `BinaryOutputDistillationReport`
diagnostics into `BinarySocialStepResult.extras`, allowing common aggregate and
micro rows to expose unit-level social channel, commit mode, and update-norm
fields. Toy2/Toy4/Toy5 batched and tensor-batched policy-distillation branches
now run behind `NABMUnit` via dedicated commit adapters, so their update-norm
fields come from the shared social mix diagnostics. Batched execution commits
through `BatchedDistributionDistillationAdapter`; tensor-batched execution
commits through `TensorRuntimeDistributionDistillationAdapter` while keeping
runtime-owned parameters and deferred agent synchronization outside the generic
unit. Toy2/Toy4/Toy5 batched and tensor-batched local policy-gradient commits
also flow through `NABMLocalStep` adapters after each domain has computed its own
actions, advantages, and active-agent set. Domain payoff logic and objective
construction remain runner-owned. Toy2, Toy4, and Toy5 now use
`BinaryPolicyLearningUnit` to own the repeated neural-policy lifecycle around
those commits: pre-readout, decision probabilities, action sampling, local
commit, cache refresh, and post-local readout. Toy2 keeps payoff-context,
counterfactual advantage, bootstrap, basin, and teacher-alignment semantics in
domain callbacks; Toy4 keeps public-goods bootstrap, basin, and
teacher-alignment semantics in domain callbacks. The unit only sequences the
reusable policy-learning lifecycle. Its optional post-readout callback exists
for domains that need raw pre-readout diagnostics but temperature-adjusted
post-local probabilities. The callbacks are grouped in
`BinaryPolicyLearningCallbacks`, with typed protocols for readout, decision
probability construction, action sampling, local update commit, cache refresh,
and optional post-readout collection. This keeps the lifecycle contract explicit
without moving domain objectives into the shared unit.

## NABM Unit v1 Contract Freeze

The v1 unit contract is frozen around lifecycle sequencing, typed exchange,
backend dispatch, and diagnostics. Generic unit APIs may grow only when the
change can be classified as lifecycle, typed exchange, backend dispatch,
diagnostics, or explicit contract-gap remediation.

Domain semantics stay outside the unit. New shared unit code must not construct
rewards, payoffs, thresholds, teacher signals, basin credit, readiness meaning,
revision pressure meaning, or evidence criteria. Domain adapters supply those
meanings through callbacks and then use the unit to execute the reusable
sequence.

Contract changes must update the boundary docs in the same patch:

- `docs/decisions/0010-nabm-unit-v1-contract.md`
- `docs/decisions/0011-continuous-scalar-unit-contract.md`
- `docs/decisions/0012-existing-toy-migration-parity-consolidation.md`
- `docs/decisions/0013-public-api-v0-contract.md`
- `docs/decisions/0014-package-dependency-policy.md`
- `docs/nabm-unit-v1-boundary-audit.md`
- `docs/api-surface-audit.md`
- `docs/nabm-unit-v1-completeness-checklist.md`
- `docs/nabm-unit-v1-migration-candidate-audit.md` when selecting the next
  existing-toy migration target.
- `docs/nabm-unit-v1-runner-lifecycle-audit.md` when changing
  compatible-toy runner ownership or adapter lifecycle boundaries.

The active guard tests for the current binary unit migration are:

- `tests/test_spatial_binary_runner.py` for unit-level binary lifecycle and
  helper contracts;
- `tests/test_toy2_runner.py`, `tests/test_toy4_runner.py`, and
  `tests/test_toy5_runner.py` for Toy2/Toy4/Toy5 policy-unit adoption;
- `tests/test_nabm_unit_adapter_holdout.py` for adapter-only holdout
  coverage that uses the binary policy and readiness units without adding a
  new `src/neural_abm` toy;
- `tests/test_toy8_runner.py` for the first existing-toy migration parity slice
  where Toy8 social-hazard mixing routes through a unit-backed scalar path;
- `tests/test_toy9_runner.py` for the second existing-toy migration parity
  slice where Toy9 heterogeneous probability mixing uses the same scalar path;
- `tests/test_toy7_runner.py` for Toy7 compatibility guardrails while Toy7
  continuous extraction-intensity social mixing routes through the bounded
  scalar path without using probability semantics;
- `tests/test_toy10_runner.py` for Toy10 market/ecology channel parity where
  price expectation and conservation norm social mixing reuse the bounded
  scalar path while composite peer selection and dynamic rewiring stay
  Toy-owned;
- `tests/test_toy6_runner.py` for Toy6 categorical distribution parity where
  strategy-distribution social mixing reuses the probability-distribution path
  while cyclic payoff and strategy semantics stay Toy-owned;
- `tests/test_domain_toy_artifact_contracts.py` for exact Toy6-Toy10
  `aggregate_metrics.csv` and `micro_state.csv` header contracts;
- `tests/test_readiness.py` for readiness propagation boundaries;
- `tests/test_nabm_unit_docs.py` for this documentation boundary.

Toy runners may keep thin compatibility wrappers, but reusable social updates
should flow through `NABMStep` when they need both mix and commit diagnostics,
reusable local learning commits should flow through `NABMLocalStep`, and binary
neural policy lifecycle wiring should flow through `BinaryPolicyLearningUnit`
once the domain has supplied observations and objective callbacks.
Toy7's continuous extraction intensity must not be routed through
`SCALAR_PROBABILITY_CHANNEL`; its social mixing slice uses the bounded
continuous scalar contract, `BOUNDED_SCALAR_CHANNEL` plus
`mix_bounded_scalars(...)`, while resource and payoff semantics remain in
Toy7.
Toy10's price expectation and conservation norm channels use the same bounded
scalar path one channel at a time; multi-channel aggregation, market price,
resource dynamics, and dynamic rewiring remain Toy10-owned.
Toy6's categorical strategy distribution uses the probability-distribution
path; strategy identity, cyclic payoff semantics, action sampling, and
categorical evidence interpretation remain Toy6-owned.
Together, the Toy6-10 parity slices cover probability-distribution, scalar, and
bounded-scalar social exchange in existing compatible toys. They are migration
parity evidence for typed social exchange reuse, not performance evidence and
not an upgrade of Toy6-10 to full NABM status.
`DomainToyRunner` already owns compatible-toy run directories, metadata,
`aggregate_metrics.csv`, `micro_state.csv`, fallback handling, final summary
writing, and the result envelope. Toy6-10 step(...) phase ordering remains
toy-owned because payoff, resource, event, market, group, and categorical
semantics live in that order. The next generic extraction target is diagnostics
mapping around peer/social fields, not a full runner rewrite.
The first diagnostics mapping slice lives in
`domain_social_diagnostics.aggregate_social_diagnostic_fields` and
`domain_social_diagnostics.micro_social_diagnostic_fields`; Toy6-Toy10 use it
for `peer_count`, `mean_peer_count`, `mean_social_loss`, and
`mean_social_update_norm` row fields while their domain rows remain toy-owned.

Unit contract changes should update the docs that define the boundary:

- `docs/decisions/0010-nabm-unit-v1-contract.md`
- `docs/decisions/0011-continuous-scalar-unit-contract.md`
- `docs/decisions/0012-existing-toy-migration-parity-consolidation.md`
- `docs/decisions/0013-public-api-v0-contract.md`
- `docs/decisions/0014-package-dependency-policy.md`
- `docs/nabm-unit-v1-boundary-audit.md`
- `docs/nabm-unit-v1-completeness-checklist.md`
- `docs/nabm-unit-v1-migration-candidate-audit.md` for Gate 7 existing-toy
  migration target selection.
- `docs/nabm-unit-v1-runner-lifecycle-audit.md` for compatible-toy runner and
  diagnostics mapping boundaries.

In particular, new shared helpers such as
`spatial_binary.run_binary_policy_learning_step` should remain lifecycle
plumbing only. They may wire domain-supplied callbacks into the reusable unit,
but they must not construct rewards, payoffs, thresholds, teacher signals,
basin credit, or evidence criteria.

## Public API v0 Boundary

The current `neural_abm.__init__` export list is broader than the intended v0
public API. It remains a lazy compatibility surface for existing module-path
imports, not the final public contract.

The next API implementation should prefer a narrow `neural_abm.api` facade that
exports stable lifecycle, typed social exchange, compatible-toy runner,
semantic-free diagnostics, result-envelope, and readiness-aggregation surfaces.
Binary policy/revision lifecycles, accelerator/runtime helpers, mobility,
reputation, evidence manifests, and paper diagnostics should stay experimental,
paper-only, or module-path imports until their contracts are explicitly
accepted.

The first facade slice is now `src/neural_abm/api.py`; it intentionally excludes
toy runners, evidence gates, binary revision/policy internals, and accelerator
runtime helpers from the stable v0 namespace.
The first torch-free profile seed is `src/neural_abm/api_lite.py`; it excludes
`NABMUnit`, `SocialBlock`, tensor/state-dict social messages, and all other
torch-backed lifecycle surfaces while retaining compatible runner, diagnostics,
result, readiness utilities, and NumPy-only social primitives from
`src/neural_abm/social_core.py`, plus lightweight lifecycle report/local-step
primitives from `src/neural_abm/unit_core.py`. Its `SocialChannel` metadata is
limited to scalar/bounded scalar mix channels; distribution helpers remain
standalone, and tensor/state mixing requires the torch-backed API.

## Package Dependency Boundary

The default package profile is the lightweight no-torch `api_lite` boundary.
The full stable `neural_abm.api` module still imports `unit` and `social`, and
those modules load `torch` at import time. Torch-free social primitives now live
in `src/neural_abm/social_core.py` and are re-exported through `api_lite`;
`api_lite.SocialChannel` accepts only scalar/bounded scalar mix channel kinds.
Distribution helpers remain standalone in the lite facade, while `social`
continues to own torch-backed tensor/state-dict mixing and `SocialBlock`.
Torch-free lifecycle reports, social diagnostics, and `NABMLocalStep` live in
`src/neural_abm/unit_core.py`; `unit` continues to own `ObservationSpec`,
`SocialMessageSpec`, tensor value builders, torch-backed adapters, `NABMStep`,
and `NABMUnit`.

Decision 0014 records the dependency policy. Torch-backed lifecycle work and
research workflows now require explicit extras such as `torch`, `research`,
`plot`, `cli`, or `full`. Future package-readiness work must decide whether v0
remains explicitly torch-backed for lifecycle or whether more lifecycle surfaces
should be split into torch-free modules.
`tests/test_public_api_lite.py` is the first import-time guard for the split: it
blocks `torch` in a subprocess and imports both the package root and
`neural_abm.api_lite`.
