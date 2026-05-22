# Neural ABM Module Workspace

This directory contains reusable implementation modules.

Current modules:

- `accelerator`: torch device resolution plus batched per-agent MLP policy
  inference kernels for GPU-friendly neural/social toy execution.
- `binary_neural`: shared binary-policy neural accelerator helpers, including
  the `TensorPolicyRuntime` contract used by capability-gated `tensor_batched`
  runtimes.
- `config`: Pydantic config schemas and YAML loader.
- `core`: common neural classification agent contract and MLP utilities.
- `graphs`: static graph and peer graph helpers.
- `logging`: micro-state and aggregate CSV logging helpers.
- `metrics`: task and social-dynamics metrics.
- `mixers`: peer selection plus output, latent, and parameter averaging.
- `mobility`: fixed-cell local-quality mobility and state-channel swaps.
- `readiness`: binary readiness-propagation coordination unit that converts
  prior ready-state scores into peer evidence increments before hard
  commitment.
- `reputation`: action-history EMA and reputation-driven imitation helpers.
- `social`: reusable `SocialBlock`/`SocialChannel` contract, invariants,
  scalar and distribution-valued output peer selection, plus probability,
  tensor, and parameter-state channel mixing.
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
- `docs/nabm-unit-v1-boundary-audit.md`
- `docs/nabm-unit-v1-completeness-checklist.md`

The active guard tests for the current binary unit migration are:

- `tests/test_spatial_binary_runner.py` for unit-level binary lifecycle and
  helper contracts;
- `tests/test_toy2_runner.py`, `tests/test_toy4_runner.py`, and
  `tests/test_toy5_runner.py` for Toy2/Toy4/Toy5 policy-unit adoption;
- `tests/test_nabm_unit_adapter_holdout.py` for adapter-only holdout smoke
  coverage that uses the binary policy and readiness units without adding a
  new `src/neural_abm` toy;
- `tests/test_readiness.py` for readiness propagation boundaries;
- `tests/test_nabm_unit_docs.py` for this documentation boundary.

Toy runners may keep thin compatibility wrappers, but reusable social updates
should flow through `NABMStep` when they need both mix and commit diagnostics,
reusable local learning commits should flow through `NABMLocalStep`, and binary
neural policy lifecycle wiring should flow through `BinaryPolicyLearningUnit`
once the domain has supplied observations and objective callbacks.

Unit contract changes should update the docs that define the boundary:

- `docs/decisions/0010-nabm-unit-v1-contract.md`
- `docs/nabm-unit-v1-boundary-audit.md`
- `docs/nabm-unit-v1-completeness-checklist.md`

In particular, new shared helpers such as
`spatial_binary.run_binary_policy_learning_step` should remain lifecycle
plumbing only. They may wire domain-supplied callbacks into the reusable unit,
but they must not construct rewards, payoffs, thresholds, teacher signals,
basin credit, or evidence criteria.
