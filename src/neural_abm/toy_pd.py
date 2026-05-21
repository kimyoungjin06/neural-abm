"""Toy 2: Neural Spatial Prisoner's Dilemma runner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import networkx as nx
import numpy as np
import torch
from torch import nn

from neural_abm.accelerator import (
    BatchedAdamStateCache,
    BatchedMLPParameters,
    BatchedMLPPolicyCache,
    BatchedMLPUpdateResult,
    NeuralUpdateBackend,
    NeuralUpdateBackendRequest,
    TensorBatchedMLPRuntime,
    batched_mlp_policy_probs,
    resolve_neural_update_backend,
    resolve_torch_device,
)
from neural_abm.binary_neural import (
    apply_batched_output_average_distillation_update,
    apply_tensor_output_average_distillation_update,
)
from neural_abm.binary_revision import (
    REVISION_STAY,
    BinaryRevisionLearningCallbacks,
    BinaryRevisionLearningUnit,
    apply_binary_revision_choices,
    binary_revision_probabilities_from_action_probs,
    normalize_binary_revision_probabilities,
)
from neural_abm.basin_phase_critic import (
    LearnedBasinPhaseCriticBundle,
    LearnedBasinReplayWeightScorer,
    LearnedBasinRuntimeDiagnostics,
    learned_basin_credit_signal,
    learned_basin_runtime_diagnostics,
    load_learned_basin_replay_weight_scorer,
    load_learned_basin_phase_critic_bundle,
)
from neural_abm.basin_transition_samples import (
    annotate_terminal_outcomes,
    basin_transition_sample_rows,
    write_basin_transition_samples,
)
from neural_abm.config import Toy2Config, validate_toy2_payoff_threshold
from neural_abm.domain_learning_diagnostics import (
    DOMAIN_LEARNING_AGGREGATE_FIELDS,
    DOMAIN_LEARNING_MICRO_FIELDS,
    domain_learning_aggregate_fields,
    domain_learning_micro_fields,
)
from neural_abm.losses import LossVector, loss_values_at
from neural_abm.logging import CsvLogWriter
from neural_abm.mobility import (
    MobilityParams,
    MobilityStepResult,
)
from neural_abm.reputation import (
    ReputationParams,
    reputation_imitation_cooperation_probs as apply_reputation_imitation,
    reputation_observation_extra_dim,
    reputation_observation_features,
)
from neural_abm.results import write_binary_summary_artifact, write_run_metadata_artifacts
from neural_abm.social import (
    PeerIndexCache,
    SocialBlock,
    scalar_output_similarity_matrix,
)
from neural_abm.spatial_binary import (
    BINARY_AGGREGATE_COMMON_FIELDS,
    BINARY_MICRO_COMMON_FIELDS,
    BinaryLocalStepResult,
    BatchedDistributionDistillationAdapter,
    BinaryOutputDistillationReport,
    BinaryPolicyLearningCallbacks,
    BinaryPolicyLearningUnit,
    BinaryPolicyStepResult,
    BinarySocialStepResult,
    BinarySpatialRunner,
    BinarySpatialState,
    BinaryStepContext,
    apply_binary_output_distribution_distillation,
    binary_aggregate_common_fields,
    binary_action_probs_from_policy,
    binary_peer_metrics,
    binary_policy_matrix,
    BinaryToyDomainBase,
    BinaryToyResult,
    StateArray,
    mix_binary_output_average,
    run_batched_policy_gradient_local_update,
    run_binary_output_distribution_distillation,
    run_tensor_runtime_policy_gradient_local_update,
    timed_context_stage,
    TensorRuntimeDistributionDistillationAdapter,
    to_numpy_view,
)
from neural_abm.state_continuation import (
    BasinCreditDiagnostics,
    PrototypePhaseBasinCritic,
    StateContinuationComponents,
    basin_credit_diagnostics_with_training_passes,
    basin_credit_effective_learned_replay_min_selected_rate,
    basin_credit_effective_training_passes,
    basin_credit_learned_model_path,
    basin_credit_needs_learned_runtime,
    basin_credit_preserves_objective,
    basin_credit_training_candidate_mask,
    blend_basin_credit_components,
    blend_domain_decision_bootstrap_probabilities,
    blend_domain_bootstrap_components,
    build_basin_credit_diagnostics,
    build_basin_phase_representation,
    build_domain_decision_replay_diagnostics,
    build_domain_teacher_alignment_diagnostics,
    domain_decision_bootstrap_weight,
    domain_decision_replay_weight,
    domain_distill_bootstrap_diagnostic_components,
    domain_distill_bootstrap_weight,
    domain_bootstrap_weight,
    combine_state_continuation_advantages,
    gradient_gate_mask,
    objective_teacher_sign_alignment,
    stable_teacher_probability_mask,
    selected_credit_to_action1_advantage,
    teacher_policy_bce,
)
from neural_abm.unit import (
    ObservationSpec,
    SocialMessageSpec,
)


TOY2_MICRO_STATE_FIELDS = [
    *BINARY_MICRO_COMMON_FIELDS,
    "domain_game_family",
    "domain_neighbor_action_rate",
    "domain_neighbor_mean_payoff",
    *DOMAIN_LEARNING_MICRO_FIELDS,
]


TOY2_AGGREGATE_FIELDS = [
    *BINARY_AGGREGATE_COMMON_FIELDS,
    "edge_entropy",
    "domain_game_family",
    "domain_payoff_T",
    "domain_payoff_R",
    "domain_payoff_P",
    "domain_payoff_S",
    "domain_policy_consensus",
    "domain_action_components",
    "domain_largest_action_cluster_fraction",
    *DOMAIN_LEARNING_AGGREGATE_FIELDS,
]


class PolicyMLP(nn.Module):
    """Small policy network for Toy 2."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, x: torch.Tensor, return_hidden: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.activation(self.fc1(x))
        logits = self.fc2(hidden)
        if return_hidden:
            return logits, hidden
        return logits


@dataclass
class NeuralPDAgent:
    agent_id: int
    model: PolicyMLP
    optimizer: torch.optim.Optimizer
    payoff_ema: float = 0.0
    previous_payoff_ema: float = 0.0

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="pd_observation",
            tensor_shape=(None, self.model.fc1.in_features),
            dtype=torch.float32,
        )

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec(
            required_keys=(
                "agent_id",
                "policy_probs",
                "probe_probs",
                "latent_summary",
                "confidence",
                "param_norm",
                "payoff_ema",
            ),
            tensor_keys=("policy_probs", "probe_probs", "latent_summary"),
            probability_keys=("policy_probs", "probe_probs"),
        )

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def act_or_predict(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(observation), dim=-1)

    def local_update(self, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        raise NotImplementedError("Toy 2 local update requires payoff context")

    def hidden_on(self, observation: torch.Tensor) -> torch.Tensor:
        _, hidden = self.model(observation, return_hidden=True)
        return hidden

    @torch.no_grad()
    def social_message(self, observation: torch.Tensor) -> dict[str, Any]:
        observed = self.observe(observation)
        logits, hidden = self.model(observed, return_hidden=True)
        probs = torch.softmax(logits, dim=-1)
        latent_summary = hidden.mean(dim=0) if hidden.ndim > 1 else hidden
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = torch.log(
            torch.tensor(float(probs.shape[-1]), dtype=probs.dtype, device=probs.device)
        )
        confidence = torch.clamp(1.0 - entropy / max_entropy, 0.0, 1.0)
        params = torch.cat([param.detach().flatten() for param in self.model.parameters()])
        return {
            "agent_id": self.agent_id,
            "policy_probs": probs.detach().clone(),
            "probe_probs": probs.detach().clone(),
            "latent_summary": latent_summary.detach().clone(),
            "confidence": float(confidence.detach().cpu()),
            "param_norm": float(torch.linalg.vector_norm(params).cpu()),
            "payoff_ema": float(self.payoff_ema),
        }

    @torch.no_grad()
    def log_state(self, observation: torch.Tensor) -> dict[str, Any]:
        message = self.social_message(observation)
        return {
            "agent_id": self.agent_id,
            "confidence": message["confidence"],
            "param_norm": message["param_norm"],
            "latent_norm": float(
                torch.linalg.vector_norm(message["latent_summary"]).cpu()
            ),
            "payoff_ema": message["payoff_ema"],
        }


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def payoff_scale(config: Toy2Config) -> float:
    payoff = config.game.payoff
    return max(
        abs(payoff.T),
        abs(payoff.R),
        abs(payoff.P),
        abs(payoff.S),
        1.0,
    )


def transformed_advantage(value: float, config: Toy2Config) -> float:
    if config.policy.domain.payoff_transform == "linear":
        transformed = value
    elif config.policy.domain.payoff_transform == "tanh":
        transformed = float(np.tanh(value))
    else:
        raise ValueError(
            f"Unsupported payoff transform: {config.policy.domain.payoff_transform}"
        )
    return config.policy.selection_strength * transformed


def make_model(config: Toy2Config) -> PolicyMLP:
    model_config = config.agents.model
    model = PolicyMLP(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=model_config.output_dim,
    )
    policy_prior = config.agents.policy_prior_action_probability
    if policy_prior is not None:
        initialize_policy_head_prior(model, policy_prior)
    return model


def initialize_policy_head_prior(
    model: PolicyMLP,
    action_probability: float,
) -> None:
    if model.fc2.out_features != 2:
        raise ValueError("Toy 2 policy prior expects model.output_dim=2")
    prior = np.array(
        [1.0 - action_probability, action_probability],
        dtype=np.float64,
    )
    dtype = model.fc2.bias.dtype
    device = model.fc2.bias.device
    tiny = torch.finfo(dtype).tiny
    log_prior = torch.log(
        torch.as_tensor(np.clip(prior, tiny, 1.0), dtype=dtype, device=device)
    )
    with torch.no_grad():
        model.fc2.weight.zero_()
        model.fc2.bias.copy_(log_prior)


def make_optimizer(model: torch.nn.Module, config: Toy2Config) -> torch.optim.Optimizer:
    if config.agents.optimizer.name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.agents.optimizer.learning_rate
        )
    raise ValueError(f"Unsupported optimizer: {config.agents.optimizer.name}")


def create_initial_models(config: Toy2Config, device: torch.device) -> list[PolicyMLP]:
    count = config.environment.grid_width * config.environment.grid_height
    base_state = None
    if config.agents.init_mode == "same_init":
        torch.manual_seed(config.run.seed)
        base_state = clone_state_dict(make_model(config))

    models = []
    for agent_id in range(count):
        if config.agents.init_mode == "independent_init":
            torch.manual_seed(config.run.seed * 1000 + agent_id)
        model = make_model(config).to(device)
        if base_state is not None:
            model.load_state_dict(base_state)
        models.append(model)
    return models


def create_agents(config: Toy2Config, device: torch.device) -> list[NeuralPDAgent]:
    models = create_initial_models(config, device)
    agents = []
    for agent_id, model in enumerate(models):
        agents.append(
            NeuralPDAgent(
                agent_id=agent_id,
                model=model,
                optimizer=make_optimizer(model, config),
            )
        )
    return agents


def _make_linear_parameter_tensors(
    in_features: int,
    out_features: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.empty((out_features, in_features), dtype=torch.float32)
    bias = torch.empty(out_features, dtype=torch.float32)
    _initialize_linear_parameters_(weight, bias, in_features)
    return weight, bias


def _initialize_linear_parameters_(
    weight: torch.Tensor,
    bias: torch.Tensor,
    in_features: int,
    generator: torch.Generator | None = None,
) -> None:
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5), generator=generator)
    bound = 1 / math.sqrt(in_features) if in_features > 0 else 0.0
    nn.init.uniform_(bias, -bound, bound, generator=generator)


def _make_policy_parameter_tensors(
    config: Toy2Config,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model_config = config.agents.model
    fc1_weight, fc1_bias = _make_linear_parameter_tensors(
        model_config.input_dim,
        model_config.hidden_dim,
    )
    fc2_weight, fc2_bias = _make_linear_parameter_tensors(
        model_config.hidden_dim,
        model_config.output_dim,
    )
    _apply_policy_prior_to_tensors_(config, fc2_weight, fc2_bias)
    return fc1_weight, fc1_bias, fc2_weight, fc2_bias


def _apply_policy_prior_to_tensors_(
    config: Toy2Config,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
) -> None:
    model_config = config.agents.model
    policy_prior = config.agents.policy_prior_action_probability
    if policy_prior is not None:
        if model_config.output_dim != 2:
            raise ValueError("Toy 2 policy prior expects model.output_dim=2")
        prior = np.array(
            [1.0 - policy_prior, policy_prior],
            dtype=np.float64,
        )
        tiny = torch.finfo(fc2_bias.dtype).tiny
        log_prior = torch.log(
            torch.as_tensor(
                np.clip(prior, tiny, 1.0),
                dtype=fc2_bias.dtype,
            )
        )
        fc2_weight.zero_()
        fc2_bias.copy_(log_prior)


def _initialize_policy_parameters_(
    config: Toy2Config,
    fc1_weight: torch.Tensor,
    fc1_bias: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_bias: torch.Tensor,
    generator: torch.Generator | None = None,
) -> None:
    model_config = config.agents.model
    _initialize_linear_parameters_(
        fc1_weight,
        fc1_bias,
        model_config.input_dim,
        generator,
    )
    _initialize_linear_parameters_(
        fc2_weight,
        fc2_bias,
        model_config.hidden_dim,
        generator,
    )
    _apply_policy_prior_to_tensors_(config, fc2_weight, fc2_bias)


def create_initial_batched_parameters(
    config: Toy2Config,
    device: torch.device,
) -> BatchedMLPParameters:
    count = config.environment.grid_width * config.environment.grid_height
    if config.agents.init_mode == "same_init":
        torch.manual_seed(config.run.seed)
        base_parameters = _make_policy_parameter_tensors(config)
        for _agent_id in range(count):
            _make_policy_parameter_tensors(config)
        return BatchedMLPParameters(
            fc1_weight=base_parameters[0].to(device).unsqueeze(0).repeat(count, 1, 1),
            fc1_bias=base_parameters[1].to(device).unsqueeze(0).repeat(count, 1),
            fc2_weight=base_parameters[2].to(device).unsqueeze(0).repeat(count, 1, 1),
            fc2_bias=base_parameters[3].to(device).unsqueeze(0).repeat(count, 1),
        )

    model_config = config.agents.model
    fc1_weight = torch.empty(
        (count, model_config.hidden_dim, model_config.input_dim),
        dtype=torch.float32,
    )
    fc1_bias = torch.empty((count, model_config.hidden_dim), dtype=torch.float32)
    fc2_weight = torch.empty(
        (count, model_config.output_dim, model_config.hidden_dim),
        dtype=torch.float32,
    )
    fc2_bias = torch.empty((count, model_config.output_dim), dtype=torch.float32)
    last_generator: torch.Generator | None = None
    for agent_id in range(count):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.run.seed * 1000 + agent_id)
        _initialize_policy_parameters_(
            config,
            fc1_weight[agent_id],
            fc1_bias[agent_id],
            fc2_weight[agent_id],
            fc2_bias[agent_id],
            generator,
        )
        last_generator = generator
    if last_generator is not None:
        torch.random.set_rng_state(last_generator.get_state())
    return BatchedMLPParameters(
        fc1_weight=fc1_weight.to(device),
        fc1_bias=fc1_bias.to(device),
        fc2_weight=fc2_weight.to(device),
        fc2_bias=fc2_bias.to(device),
    )


def _zero_like_batched_parameters(
    parameters: BatchedMLPParameters,
) -> BatchedMLPParameters:
    return BatchedMLPParameters(
        fc1_weight=torch.zeros_like(parameters.fc1_weight),
        fc1_bias=torch.zeros_like(parameters.fc1_bias),
        fc2_weight=torch.zeros_like(parameters.fc2_weight),
        fc2_bias=torch.zeros_like(parameters.fc2_bias),
    )


def create_tensor_batched_runtime(
    config: Toy2Config,
    device: torch.device,
) -> TensorBatchedMLPRuntime:
    if config.agents.optimizer.name != "adam":
        raise ValueError("Toy 2 tensor_batched requires Adam optimizer")
    parameters = create_initial_batched_parameters(config, device)
    steps = tuple(
        torch.zeros(
            parameters.agent_count,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for tensor in parameters.tensors()
    )
    return TensorBatchedMLPRuntime(
        parameters=parameters,
        exp_avg=_zero_like_batched_parameters(parameters),
        exp_avg_sq=_zero_like_batched_parameters(parameters),
        steps=steps,
        lr=float(config.agents.optimizer.learning_rate),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        shared_step_groups=True,
    )


def build_spatial_graph(config: Toy2Config) -> nx.Graph:
    env = config.environment
    if env.neighborhood != "von_neumann":
        raise ValueError(f"Unsupported neighborhood: {env.neighborhood}")
    graph = nx.grid_2d_graph(
        env.grid_height,
        env.grid_width,
        periodic=env.periodic,
    )
    mapping = {
        (row, col): row * env.grid_width + col
        for row in range(env.grid_height)
        for col in range(env.grid_width)
    }
    return nx.relabel_nodes(graph, mapping)


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [
        sorted(int(node) for node in graph.neighbors(i)) for i in range(agent_count)
    ]


def uniform_peer_index(peer_ids: list[list[int]]) -> np.ndarray | None:
    """Return a dense peer index when all peer lists have the same length."""

    if not peer_ids:
        return np.zeros((0, 0), dtype=np.int64)
    peer_count = len(peer_ids[0])
    if any(len(peers) != peer_count for peers in peer_ids[1:]):
        return None
    if peer_count == 0:
        return np.zeros((len(peer_ids), 0), dtype=np.int64)
    return np.asarray(peer_ids, dtype=np.int64)


def well_mixed_peer_ids(
    neighbors: list[list[int]],
    agent_count: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    peer_ids: list[list[int]] = []
    for agent_id, spatial_peers in enumerate(neighbors):
        peer_count = min(len(spatial_peers), agent_count - 1)
        if peer_count <= 0:
            peer_ids.append([])
            continue
        candidates = np.concatenate(
            [
                np.arange(0, agent_id, dtype=np.int64),
                np.arange(agent_id + 1, agent_count, dtype=np.int64),
            ]
        )
        sampled = rng.choice(candidates, size=peer_count, replace=False)
        peer_ids.append(sorted(int(peer_id) for peer_id in sampled))
    return peer_ids


def neural_context_peer_ids(
    config: Toy2Config,
    neighbors: list[list[int]],
    agent_count: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    if config.policy.domain.neural_peer_mode == "spatial":
        return neighbors
    if config.policy.domain.neural_peer_mode == "well_mixed":
        return well_mixed_peer_ids(
            neighbors=neighbors,
            agent_count=agent_count,
            rng=rng,
        )
    raise ValueError(
        f"Unsupported neural peer mode: {config.policy.domain.neural_peer_mode}"
    )


def interaction_peer_ids(
    config: Toy2Config,
    neighbors: list[list[int]],
    agent_count: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    if config.policy.domain.interaction_mode == "spatial":
        return neighbors
    if config.policy.domain.interaction_mode == "well_mixed_resampled":
        return well_mixed_peer_ids(
            neighbors=neighbors,
            agent_count=agent_count,
            rng=rng,
        )
    raise ValueError(
        f"Unsupported interaction mode: {config.policy.domain.interaction_mode}"
    )


def initialize_actions(config: Toy2Config, rng: np.random.Generator) -> np.ndarray:
    count = config.environment.grid_width * config.environment.grid_height
    return (
        rng.random(count) < config.environment.initial_action_probability
    ).astype(np.int64)


def _tensor_vector(
    values: StateArray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    return torch.as_tensor(values, dtype=dtype, device=device)


def _tensor_peer_index(
    peer_index: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(peer_index, torch.Tensor):
        return peer_index.to(device=device, dtype=torch.long)
    return torch.as_tensor(peer_index, dtype=torch.long, device=device)


def _ragged_neighbor_mean(
    values: torch.Tensor,
    neighbors: list[list[int]],
) -> torch.Tensor:
    means: list[torch.Tensor] = []
    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    for peers in neighbors:
        if not peers:
            means.append(zero)
            continue
        peer_index = torch.as_tensor(peers, dtype=torch.long, device=values.device)
        means.append(values.index_select(0, peer_index).mean())
    if not means:
        return torch.zeros(0, dtype=values.dtype, device=values.device)
    return torch.stack(means)


def build_observations_tensor(
    actions: StateArray,
    payoffs: StateArray,
    agents: list[NeuralPDAgent],
    neighbors: list[list[int]],
    payoff_normalizer: float,
    device: torch.device,
    reputation: StateArray | None = None,
    reputation_observation_mode: str = "none",
    peer_index: np.ndarray | torch.Tensor | None = None,
    payoff_ema: StateArray | None = None,
    previous_payoff_ema: StateArray | None = None,
) -> torch.Tensor:
    safe_scale = max(payoff_normalizer, 1e-8)
    action_values = _tensor_vector(actions, dtype=torch.float64, device=device)
    payoff_values = _tensor_vector(payoffs, dtype=torch.float64, device=device)
    agent_count = int(action_values.shape[0])
    if payoff_values.shape[0] != agent_count:
        raise ValueError("payoffs length must match action count")
    if payoff_ema is None:
        payoff_ema_values = torch.as_tensor(
            [agent.payoff_ema for agent in agents],
            dtype=torch.float64,
            device=device,
        )
    else:
        payoff_ema_values = _tensor_vector(
            payoff_ema,
            dtype=torch.float64,
            device=device,
        )
    if previous_payoff_ema is None:
        previous_payoff_ema_values = torch.as_tensor(
            [agent.previous_payoff_ema for agent in agents],
            dtype=torch.float64,
            device=device,
        )
    else:
        previous_payoff_ema_values = _tensor_vector(
            previous_payoff_ema,
            dtype=torch.float64,
            device=device,
        )
    if (
        payoff_ema_values.shape[0] != agent_count
        or previous_payoff_ema_values.shape[0] != agent_count
    ):
        raise ValueError("payoff EMA arrays must match action count")
    if peer_index is not None:
        peer_index_tensor = _tensor_peer_index(peer_index, device=device)
        if peer_index_tensor.shape[0] != agent_count:
            raise ValueError("peer_index first dimension must match action count")
        if peer_index_tensor.shape[1] == 0:
            neighbor_coop_rate = torch.zeros(
                agent_count,
                dtype=torch.float64,
                device=device,
            )
            neighbor_mean_payoff = torch.zeros(
                agent_count,
                dtype=torch.float64,
                device=device,
            )
        else:
            neighbor_coop_rate = action_values[peer_index_tensor].mean(dim=1)
            neighbor_mean_payoff = payoff_values[peer_index_tensor].mean(dim=1)
    else:
        if len(neighbors) != agent_count:
            raise ValueError("neighbors length must match action count")
        neighbor_coop_rate = _ragged_neighbor_mean(action_values, neighbors)
        neighbor_mean_payoff = _ragged_neighbor_mean(payoff_values, neighbors)
    observation_tensor = torch.stack(
        [
            action_values,
            neighbor_coop_rate,
            payoff_ema_values / safe_scale,
            (payoff_ema_values - previous_payoff_ema_values) / safe_scale,
            neighbor_mean_payoff / safe_scale,
            torch.ones(agent_count, dtype=torch.float64, device=device),
        ],
        dim=1,
    )
    if reputation_observation_mode != "none":
        if reputation is None:
            raise ValueError("reputation observations require reputation state")
        reputation_values = _tensor_vector(
            reputation,
            dtype=torch.float64,
            device=device,
        )
        if reputation_values.shape[0] != agent_count:
            raise ValueError("reputation length must match action count")
        if reputation_observation_mode != "self_neighbor_mean":
            raise ValueError(
                f"unsupported reputation observation mode: {reputation_observation_mode}"
            )
        if peer_index is not None:
            peer_index_tensor = _tensor_peer_index(peer_index, device=device)
            if peer_index_tensor.shape[1] == 0:
                peer_means = torch.zeros(
                    agent_count,
                    dtype=torch.float64,
                    device=device,
                )
            else:
                peer_means = reputation_values[peer_index_tensor].mean(dim=1)
        else:
            peer_means = _ragged_neighbor_mean(reputation_values, neighbors)
        observation_tensor = torch.cat(
            [observation_tensor, torch.stack([reputation_values, peer_means], dim=1)],
            dim=1,
        )
    return observation_tensor.to(dtype=torch.float32)


def build_observations(
    actions: StateArray,
    payoffs: StateArray,
    agents: list[NeuralPDAgent],
    neighbors: list[list[int]],
    payoff_normalizer: float,
    device: torch.device,
    reputation: StateArray | None = None,
    reputation_observation_mode: str = "none",
    peer_index: np.ndarray | torch.Tensor | None = None,
    payoff_ema: StateArray | None = None,
    previous_payoff_ema: StateArray | None = None,
) -> torch.Tensor:
    if any(
        isinstance(values, torch.Tensor)
        for values in (
            actions,
            payoffs,
            reputation,
            peer_index,
            payoff_ema,
            previous_payoff_ema,
        )
    ):
        return build_observations_tensor(
            actions=actions,
            payoffs=payoffs,
            agents=agents,
            neighbors=neighbors,
            payoff_normalizer=payoff_normalizer,
            device=device,
            reputation=reputation,
            reputation_observation_mode=reputation_observation_mode,
            peer_index=peer_index,
            payoff_ema=payoff_ema,
            previous_payoff_ema=previous_payoff_ema,
        )
    safe_scale = max(payoff_normalizer, 1e-8)
    if peer_index is not None:
        action_values = np.asarray(actions, dtype=np.float64)
        payoff_values = np.asarray(payoffs, dtype=np.float64)
        agent_count = len(action_values)
        if peer_index.shape[0] != agent_count:
            raise ValueError("peer_index first dimension must match action count")
        if payoff_ema is None:
            payoff_ema_values = np.asarray(
                [agent.payoff_ema for agent in agents],
                dtype=np.float64,
            )
        else:
            payoff_ema_values = np.asarray(payoff_ema, dtype=np.float64)
        if previous_payoff_ema is None:
            previous_payoff_ema_values = np.asarray(
                [agent.previous_payoff_ema for agent in agents],
                dtype=np.float64,
            )
        else:
            previous_payoff_ema_values = np.asarray(
                previous_payoff_ema,
                dtype=np.float64,
            )
        if (
            payoff_ema_values.shape[0] != agent_count
            or previous_payoff_ema_values.shape[0] != agent_count
        ):
            raise ValueError("payoff EMA arrays must match action count")
        if peer_index.shape[1] == 0:
            neighbor_coop_rate = np.zeros(agent_count, dtype=np.float64)
            neighbor_mean_payoff = np.zeros(agent_count, dtype=np.float64)
        else:
            neighbor_coop_rate = action_values[peer_index].mean(axis=1)
            neighbor_mean_payoff = payoff_values[peer_index].mean(axis=1)
        observation_array = np.column_stack(
            [
                action_values,
                neighbor_coop_rate,
                payoff_ema_values / safe_scale,
                (payoff_ema_values - previous_payoff_ema_values) / safe_scale,
                neighbor_mean_payoff / safe_scale,
                np.ones(agent_count, dtype=np.float64),
            ]
        )
    else:
        observations = []
        for agent_id, peers in enumerate(neighbors):
            neighbor_actions = actions[peers] if peers else np.array([], dtype=np.int64)
            neighbor_payoffs = payoffs[peers] if peers else np.array([], dtype=np.float64)
            neighbor_coop_rate = (
                float(neighbor_actions.mean()) if len(neighbor_actions) else 0.0
            )
            neighbor_mean_payoff = (
                float(neighbor_payoffs.mean()) if len(neighbor_payoffs) else 0.0
            )
            agent_payoff_ema = agents[agent_id].payoff_ema
            payoff_trend = agent_payoff_ema - agents[agent_id].previous_payoff_ema
            observations.append(
                [
                    float(actions[agent_id]),
                    neighbor_coop_rate,
                    agent_payoff_ema / safe_scale,
                    payoff_trend / safe_scale,
                    neighbor_mean_payoff / safe_scale,
                    1.0,
                ]
            )
        observation_array = np.asarray(observations, dtype=np.float64)
    if reputation_observation_mode != "none":
        if reputation is None:
            raise ValueError("reputation observations require reputation state")
        observation_array = np.column_stack(
            [
                observation_array,
                reputation_observation_features(
                    reputation=reputation,
                    peer_ids=neighbors,
                    mode=reputation_observation_mode,
                ),
            ]
        )
    return torch.as_tensor(observation_array, dtype=torch.float32, device=device)


@torch.no_grad()
def collect_policy_probs(
    agents: list[NeuralPDAgent],
    observations: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return batched_mlp_policy_probs(
        [agent.model for agent in agents],
        observations,
        temperature=temperature,
    )


def apply_action_temperature(
    policy_probs: torch.Tensor,
    action_temperature: float,
) -> torch.Tensor:
    if action_temperature == 1.0:
        return policy_probs
    tiny = torch.finfo(policy_probs.dtype).tiny
    logits = torch.log(policy_probs.clamp_min(tiny)) / action_temperature
    return torch.softmax(logits, dim=-1)


def apply_policy_exploration(
    policy_probs: torch.Tensor,
    exploration_epsilon: float,
) -> torch.Tensor:
    if exploration_epsilon <= 0.0:
        return policy_probs
    class_count = policy_probs.shape[1]
    return (1.0 - exploration_epsilon) * policy_probs + (
        exploration_epsilon / class_count
    )


def apply_payoff_threshold_calibration(
    policy_probs: torch.Tensor,
    action_temperature: float,
    calibration_strength: float,
    decision_threshold: float,
) -> torch.Tensor:
    if policy_probs.shape[1] != 2:
        raise ValueError("Payoff-threshold calibration requires binary actions")
    eps = torch.finfo(policy_probs.dtype).eps
    cooperation_prior = policy_probs[:, 1].clamp(min=eps, max=1.0 - eps)
    threshold = torch.as_tensor(
        decision_threshold,
        dtype=policy_probs.dtype,
        device=policy_probs.device,
    )
    score = (
        calibration_strength
        * (torch.logit(cooperation_prior) - torch.logit(threshold))
        / action_temperature
    )
    calibrated_cooperation = torch.sigmoid(score)
    return torch.stack(
        (1.0 - calibrated_cooperation, calibrated_cooperation),
        dim=1,
    )


def sample_actions(
    policy_probs: torch.Tensor,
    exploration_epsilon: float,
    action_temperature: float = 1.0,
) -> np.ndarray:
    action_probs = apply_action_temperature(
        policy_probs=policy_probs,
        action_temperature=action_temperature,
    )
    action_probs = apply_policy_exploration(
        policy_probs=action_probs,
        exploration_epsilon=exploration_epsilon,
    )
    sampled = torch.multinomial(action_probs, num_samples=1).squeeze(1)
    return sampled.detach().cpu().numpy().astype(np.int64)


def terminal_argmax_active(
    *,
    epoch: int,
    total_epochs: int,
    terminal_argmax_epochs: int,
) -> bool:
    if terminal_argmax_epochs <= 0:
        return False
    return epoch > max(0, total_epochs - terminal_argmax_epochs)


@dataclass(frozen=True)
class DecisionKernel:
    mode: str = "sampled"
    action_temperature: float = 1.0
    exploration_epsilon: float = 0.0
    calibration_mode: str = "none"
    calibration_strength: float = 4.0
    decision_threshold: float | None = None
    terminal_argmax_epochs: int = 0

    @classmethod
    def from_config(cls, config: Toy2Config) -> "DecisionKernel":
        decision = config.policy.decision
        calibration = decision.calibration
        decision_threshold = (
            validate_toy2_payoff_threshold(config.game.payoff)
            if calibration.mode == "payoff_threshold"
            else None
        )
        return cls(
            mode=decision.mode,
            action_temperature=decision.action_temperature,
            exploration_epsilon=decision.exploration_epsilon,
            calibration_mode=calibration.mode,
            calibration_strength=calibration.strength,
            decision_threshold=decision_threshold,
            terminal_argmax_epochs=decision.terminal_argmax_epochs,
        )

    def mode_for_epoch(self, *, epoch: int, total_epochs: int) -> str:
        if self.mode == "sampled" and terminal_argmax_active(
            epoch=epoch,
            total_epochs=total_epochs,
            terminal_argmax_epochs=self.terminal_argmax_epochs,
        ):
            return "argmax"
        return self.mode

    def for_epoch(self, *, epoch: int, total_epochs: int) -> "DecisionKernel":
        mode = self.mode_for_epoch(epoch=epoch, total_epochs=total_epochs)
        if mode == self.mode:
            return self
        return replace(self, mode=mode)

    def select_all(self, policy_probs: torch.Tensor) -> np.ndarray:
        if self.mode == "sampled":
            action_probs = self.action_probs(policy_probs)
            sampled = torch.multinomial(action_probs, num_samples=1).squeeze(1)
            return sampled.detach().cpu().numpy().astype(np.int64)
        if self.mode == "argmax":
            selected = torch.argmax(policy_probs, dim=1)
            return selected.detach().cpu().numpy().astype(np.int64)
        raise ValueError(f"Unsupported decision mode: {self.mode}")

    def select_all_tensor(self, policy_probs: torch.Tensor) -> torch.Tensor:
        if self.mode == "sampled":
            action_probs = self.action_probs(policy_probs)
            return torch.multinomial(action_probs, num_samples=1).squeeze(1)
        if self.mode == "argmax":
            return torch.argmax(policy_probs, dim=1)
        raise ValueError(f"Unsupported decision mode: {self.mode}")

    def select_all_from_action_probs(self, action_probs: torch.Tensor) -> np.ndarray:
        if self.mode == "sampled":
            sampled = torch.multinomial(action_probs, num_samples=1).squeeze(1)
            return sampled.detach().cpu().numpy().astype(np.int64)
        if self.mode == "argmax":
            selected = torch.argmax(action_probs, dim=1)
            return selected.detach().cpu().numpy().astype(np.int64)
        raise ValueError(f"Unsupported decision mode: {self.mode}")

    def select_all_tensor_from_action_probs(
        self,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "sampled":
            return torch.multinomial(action_probs, num_samples=1).squeeze(1)
        if self.mode == "argmax":
            return torch.argmax(action_probs, dim=1)
        raise ValueError(f"Unsupported decision mode: {self.mode}")

    def action_probs(self, policy_probs: torch.Tensor) -> torch.Tensor:
        if self.mode == "sampled":
            if self.calibration_mode == "none":
                action_probs = apply_action_temperature(
                    policy_probs=policy_probs,
                    action_temperature=self.action_temperature,
                )
            elif self.calibration_mode == "payoff_threshold":
                if self.decision_threshold is None:
                    raise ValueError("Payoff-threshold calibration missing threshold")
                action_probs = apply_payoff_threshold_calibration(
                    policy_probs=policy_probs,
                    action_temperature=self.action_temperature,
                    calibration_strength=self.calibration_strength,
                    decision_threshold=self.decision_threshold,
                )
            else:
                raise ValueError(
                    f"Unsupported decision calibration mode: {self.calibration_mode}"
                )
            return apply_policy_exploration(
                policy_probs=action_probs,
                exploration_epsilon=self.exploration_epsilon,
            )
        if self.mode == "argmax":
            selected = torch.argmax(policy_probs, dim=1)
            return nn.functional.one_hot(
                selected,
                num_classes=policy_probs.shape[1],
            ).to(dtype=policy_probs.dtype)
        raise ValueError(f"Unsupported decision mode: {self.mode}")

    def select(
        self,
        current_actions: np.ndarray,
        policy_probs: torch.Tensor,
        revision_mask: np.ndarray,
    ) -> np.ndarray:
        if bool(np.all(revision_mask)):
            return self.select_all(policy_probs)
        if not bool(np.any(revision_mask)):
            return np.array(current_actions, dtype=np.int64, copy=True)
        next_actions = np.array(current_actions, dtype=np.int64, copy=True)
        revised_indices = np.flatnonzero(revision_mask)
        next_actions[revised_indices] = self.select_all(policy_probs[revised_indices])
        return next_actions

    def select_from_action_probs(
        self,
        current_actions: np.ndarray,
        action_probs: torch.Tensor,
        revision_mask: np.ndarray,
    ) -> np.ndarray:
        if bool(np.all(revision_mask)):
            return self.select_all_from_action_probs(action_probs)
        if not bool(np.any(revision_mask)):
            return np.array(current_actions, dtype=np.int64, copy=True)
        next_actions = np.array(current_actions, dtype=np.int64, copy=True)
        revised_indices = np.flatnonzero(revision_mask)
        next_actions[revised_indices] = self.select_all_from_action_probs(
            action_probs[revised_indices]
        )
        return next_actions

    def select_tensor(
        self,
        current_actions: StateArray,
        policy_probs: torch.Tensor,
        revision_mask: np.ndarray,
    ) -> torch.Tensor:
        current_tensor = _tensor_vector(
            current_actions,
            dtype=torch.long,
            device=policy_probs.device,
        )
        if bool(np.all(revision_mask)):
            return self.select_all_tensor(policy_probs)
        if not bool(np.any(revision_mask)):
            return current_tensor.clone()
        next_actions = current_tensor.clone()
        revised_indices = torch.as_tensor(
            np.flatnonzero(revision_mask),
            dtype=torch.long,
            device=policy_probs.device,
        )
        revised_probs = policy_probs.index_select(0, revised_indices)
        next_actions.index_copy_(
            0,
            revised_indices,
            self.select_all_tensor(revised_probs),
        )
        return next_actions

    def select_tensor_from_action_probs(
        self,
        current_actions: StateArray,
        action_probs: torch.Tensor,
        revision_mask: np.ndarray,
    ) -> torch.Tensor:
        current_tensor = _tensor_vector(
            current_actions,
            dtype=torch.long,
            device=action_probs.device,
        )
        if bool(np.all(revision_mask)):
            return self.select_all_tensor_from_action_probs(action_probs)
        if not bool(np.any(revision_mask)):
            return current_tensor.clone()
        next_actions = current_tensor.clone()
        revised_indices = torch.as_tensor(
            np.flatnonzero(revision_mask),
            dtype=torch.long,
            device=action_probs.device,
        )
        revised_probs = action_probs.index_select(0, revised_indices)
        next_actions.index_copy_(
            0,
            revised_indices,
            self.select_all_tensor_from_action_probs(revised_probs),
        )
        return next_actions


def payoff_pair(
    action_i: int, action_j: int, config: Toy2Config
) -> tuple[float, float]:
    payoff = config.game.payoff
    if action_i == 1 and action_j == 1:
        return payoff.R, payoff.R
    if action_i == 1 and action_j == 0:
        return payoff.S, payoff.T
    if action_i == 0 and action_j == 1:
        return payoff.T, payoff.S
    return payoff.P, payoff.P


def payoff_matrix(config: Toy2Config) -> np.ndarray:
    payoff = config.game.payoff
    return np.asarray(
        [
            [payoff.P, payoff.T],
            [payoff.S, payoff.R],
        ],
        dtype=np.float64,
    )


def payoff_matrix_tensor(
    config: Toy2Config,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(payoff_matrix(config), dtype=torch.float64, device=device)


def compute_payoffs(
    actions: np.ndarray,
    graph: nx.Graph,
    config: Toy2Config,
) -> np.ndarray:
    totals = np.zeros(len(actions), dtype=np.float64)
    counts = np.zeros(len(actions), dtype=np.float64)
    for source, target in graph.edges():
        source_payoff, target_payoff = payoff_pair(
            int(actions[source]),
            int(actions[target]),
            config,
        )
        totals[source] += source_payoff
        totals[target] += target_payoff
        counts[source] += 1.0
        counts[target] += 1.0
    return totals / np.maximum(counts, 1.0)


def compute_payoffs_from_peer_index(
    actions: StateArray,
    peer_index: np.ndarray | torch.Tensor,
    config: Toy2Config,
) -> StateArray:
    if isinstance(actions, torch.Tensor) or isinstance(peer_index, torch.Tensor):
        device = actions.device if isinstance(actions, torch.Tensor) else torch.device("cpu")
        action_values = _tensor_vector(actions, dtype=torch.long, device=device)
        peer_index_tensor = _tensor_peer_index(peer_index, device=device)
        if peer_index_tensor.shape[0] != action_values.shape[0]:
            raise ValueError("peer_index first dimension must match action count")
        if peer_index_tensor.shape[1] == 0:
            return torch.zeros(
                action_values.shape[0],
                dtype=torch.float64,
                device=device,
            )
        matrix = payoff_matrix_tensor(config, device=device)
        peer_actions = action_values[peer_index_tensor]
        return matrix[action_values[:, None], peer_actions].mean(dim=1)

    action_values = np.asarray(actions, dtype=np.int64)
    if peer_index.shape[0] != action_values.shape[0]:
        raise ValueError("peer_index first dimension must match action count")
    if peer_index.shape[1] == 0:
        return np.zeros(action_values.shape[0], dtype=np.float64)
    matrix = payoff_matrix(config)
    peer_actions = action_values[peer_index]
    return matrix[action_values[:, None], peer_actions].mean(axis=1)


def compute_payoffs_from_peer_ids(
    actions: StateArray,
    peer_ids: list[list[int]],
    config: Toy2Config,
) -> StateArray:
    if isinstance(actions, torch.Tensor):
        action_values = actions.to(dtype=torch.long)
        matrix = payoff_matrix_tensor(config, device=action_values.device)
        payoffs: list[torch.Tensor] = []
        zero = torch.zeros((), dtype=torch.float64, device=action_values.device)
        for agent_id, peers in enumerate(peer_ids):
            if not peers:
                payoffs.append(zero)
                continue
            peer_index = torch.as_tensor(
                peers,
                dtype=torch.long,
                device=action_values.device,
            )
            peer_actions = action_values.index_select(0, peer_index)
            agent_action = action_values[agent_id]
            payoffs.append(matrix[agent_action, peer_actions].mean())
        if not payoffs:
            return torch.zeros(0, dtype=torch.float64, device=action_values.device)
        return torch.stack(payoffs)

    totals = np.zeros(len(actions), dtype=np.float64)
    counts = np.zeros(len(actions), dtype=np.float64)
    for agent_id, peers in enumerate(peer_ids):
        for peer_id in peers:
            agent_payoff, _ = payoff_pair(
                int(actions[agent_id]),
                int(actions[peer_id]),
                config,
            )
            totals[agent_id] += agent_payoff
            counts[agent_id] += 1.0
    return totals / np.maximum(counts, 1.0)


def train_local_policy(
    agent: NeuralPDAgent,
    observation: torch.Tensor,
    action: int,
    payoff: float,
    config: Toy2Config,
) -> float:
    logits = agent.model(observation.unsqueeze(0))
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    action_tensor = torch.tensor([action], dtype=torch.long, device=observation.device)
    log_prob = log_probs.gather(1, action_tensor.unsqueeze(1)).squeeze()
    entropy = -(probs * log_probs).sum()
    safe_scale = max(payoff_scale(config), 1e-8)
    advantage = transformed_advantage((payoff - agent.payoff_ema) / safe_scale, config)
    loss = -float(advantage) * log_prob - config.environment.entropy_beta * entropy
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    return float(loss.detach().cpu())


def counterfactual_action_payoffs(
    peer_actions: np.ndarray,
    config: Toy2Config,
) -> tuple[float, float]:
    if len(peer_actions) == 0:
        return 0.0, 0.0
    defect_payoffs = [
        payoff_pair(0, int(peer_action), config)[0] for peer_action in peer_actions
    ]
    cooperate_payoffs = [
        payoff_pair(1, int(peer_action), config)[0] for peer_action in peer_actions
    ]
    return float(np.mean(defect_payoffs)), float(np.mean(cooperate_payoffs))


def _toy2_social_continuation_advantages(
    action_count: int,
    config: Toy2Config,
) -> np.ndarray:
    if not config.state.reputation.enabled:
        return np.zeros(action_count, dtype=np.float64)
    return np.full(
        action_count,
        1.0 - config.state.reputation.decay,
        dtype=np.float64,
    )


def _toy2_counterfactual_component_arrays(
    actions: np.ndarray,
    peer_ids: list[list[int]],
    config: Toy2Config,
    peer_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    safe_scale = max(payoff_scale(config), 1e-8)
    action_values = np.asarray(actions, dtype=np.int64)
    material_deltas = np.zeros(action_values.shape[0], dtype=np.float64)
    welfare_deltas = np.zeros(action_values.shape[0], dtype=np.float64)
    matrix = payoff_matrix(config)

    if peer_index is not None:
        if peer_index.shape[0] != action_values.shape[0]:
            raise ValueError("peer_index first dimension must match action count")
        if peer_index.shape[1] == 0:
            return material_deltas, welfare_deltas
        peer_actions = action_values[peer_index]
        defect_payoffs = matrix[0, peer_actions].mean(axis=1)
        cooperate_payoffs = matrix[1, peer_actions].mean(axis=1)
        raw_material = cooperate_payoffs - defect_payoffs
        peer_welfare_delta = matrix[peer_actions, 1] - matrix[peer_actions, 0]
        raw_welfare = (
            raw_material + peer_welfare_delta.sum(axis=1)
        ) / float(peer_index.shape[1] + 1)
        return raw_material / safe_scale, raw_welfare / safe_scale

    for agent_id, peers in enumerate(peer_ids):
        if not peers:
            continue
        peer_actions = action_values[peers]
        defect_payoff, cooperate_payoff = counterfactual_action_payoffs(
            peer_actions=peer_actions,
            config=config,
        )
        raw_material = cooperate_payoff - defect_payoff
        peer_welfare_delta = matrix[peer_actions, 1] - matrix[peer_actions, 0]
        raw_welfare = (raw_material + float(np.sum(peer_welfare_delta))) / (
            len(peers) + 1
        )
        material_deltas[agent_id] = raw_material / safe_scale
        welfare_deltas[agent_id] = raw_welfare / safe_scale
    return material_deltas, welfare_deltas


def counterfactual_policy_advantage_components(
    actions: np.ndarray,
    peer_ids: list[list[int]],
    config: Toy2Config,
    peer_index: np.ndarray | None = None,
) -> StateContinuationComponents:
    material_deltas, welfare_deltas = _toy2_counterfactual_component_arrays(
        actions=actions,
        peer_ids=peer_ids,
        config=config,
        peer_index=peer_index,
    )
    raw_components = combine_state_continuation_advantages(
        material=material_deltas,
        social=_toy2_social_continuation_advantages(len(material_deltas), config),
        welfare=welfare_deltas,
        objective=config.policy.domain.objective,
    )
    return raw_components.with_effective(
        transformed_advantages(raw_components.effective, config)
    )


def toy2_target_basin_payoff(config: Toy2Config) -> float:
    """Return the Toy 2 v1 ceiling target used by basin credit."""

    return float(config.game.payoff.R)


def train_counterfactual_policy(
    agent: NeuralPDAgent,
    observation: torch.Tensor,
    peer_actions: np.ndarray,
    config: Toy2Config,
    signed_advantage: float | None = None,
    teacher_distill_probability: float | None = None,
    teacher_distill_weight: float = 0.0,
    teacher_distill_loss: str = "bce",
    base_loss_weight: float = 1.0,
) -> float:
    logits = agent.model(observation.unsqueeze(0))
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    if signed_advantage is None:
        defect_payoff, cooperate_payoff = counterfactual_action_payoffs(
            peer_actions=peer_actions,
            config=config,
        )
        safe_scale = max(payoff_scale(config), 1e-8)
        advantage = transformed_advantage(
            (cooperate_payoff - defect_payoff) / safe_scale,
            config,
        )
    else:
        advantage = float(signed_advantage)
    if advantage >= 0.0:
        policy_loss = -float(advantage) * log_probs[0, 1]
    else:
        policy_loss = float(advantage) * log_probs[0, 0]
    loss = float(base_loss_weight) * (
        policy_loss - config.environment.entropy_beta * entropy
    )
    if teacher_distill_probability is not None and teacher_distill_weight > 0.0:
        loss = loss + float(teacher_distill_weight) * teacher_distillation_loss(
            log_probs=log_probs[0],
            teacher_probability=teacher_distill_probability,
            loss_type=teacher_distill_loss,
        )
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    return float(loss.detach().cpu())


def teacher_distillation_loss(
    *,
    log_probs: torch.Tensor,
    teacher_probability: float,
    loss_type: str,
) -> torch.Tensor:
    target = torch.as_tensor(
        float(teacher_probability),
        dtype=log_probs.dtype,
        device=log_probs.device,
    ).clamp(
        min=torch.finfo(log_probs.dtype).eps,
        max=1.0 - torch.finfo(log_probs.dtype).eps,
    )
    bce = -(target * log_probs[1] + (1.0 - target) * log_probs[0])
    if loss_type == "bce":
        return bce
    if loss_type == "kl":
        teacher_entropy = -(
            target * torch.log(target)
            + (1.0 - target) * torch.log(1.0 - target)
        )
        return bce - teacher_entropy
    raise ValueError(f"Unsupported teacher distillation loss: {loss_type}")


def _flatten_loss_gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    flattened: list[torch.Tensor] = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if gradient is None:
            flattened.append(torch.zeros_like(parameter).reshape(-1))
        else:
            flattened.append(gradient.reshape(-1))
    if not flattened:
        return torch.zeros(0, dtype=loss.dtype, device=loss.device)
    return torch.cat(flattened)


def _gradient_conflict_values(
    *,
    base_loss: torch.Tensor,
    distill_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[float, float, float, float, float]:
    base_gradient = _flatten_loss_gradients(
        base_loss,
        parameters,
        retain_graph=True,
    )
    distill_gradient = _flatten_loss_gradients(
        distill_loss,
        parameters,
        retain_graph=False,
    )
    base_norm = torch.linalg.vector_norm(base_gradient)
    distill_norm = torch.linalg.vector_norm(distill_gradient)
    if float(base_norm.detach().cpu()) <= 0.0 or float(
        distill_norm.detach().cpu()
    ) <= 0.0:
        cosine = float("nan")
    else:
        cosine = float(
            (
                torch.dot(base_gradient, distill_gradient)
                / (base_norm * distill_norm)
            )
            .detach()
            .cpu()
        )
    return (
        float(base_loss.detach().cpu()),
        float(distill_loss.detach().cpu()),
        float(base_norm.detach().cpu()),
        float(distill_norm.detach().cpu()),
        cosine,
    )


def teacher_distill_gradient_conflict_values(
    *,
    agent: NeuralPDAgent,
    observation: torch.Tensor,
    peer_actions: np.ndarray,
    config: Toy2Config,
    signed_advantage: float | None,
    teacher_probability: float,
) -> tuple[float, float, float, float, float]:
    logits = agent.model(observation.unsqueeze(0))
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    if signed_advantage is None:
        defect_payoff, cooperate_payoff = counterfactual_action_payoffs(
            peer_actions=peer_actions,
            config=config,
        )
        safe_scale = max(payoff_scale(config), 1e-8)
        advantage = transformed_advantage(
            (cooperate_payoff - defect_payoff) / safe_scale,
            config,
        )
    else:
        advantage = float(signed_advantage)
    if advantage >= 0.0:
        policy_loss = -float(advantage) * log_probs[0, 1]
    else:
        policy_loss = float(advantage) * log_probs[0, 0]
    base_loss = policy_loss - config.environment.entropy_beta * entropy
    distill_loss = teacher_distillation_loss(
        log_probs=log_probs[0],
        teacher_probability=teacher_probability,
        loss_type=config.policy.domain.bootstrap.distill_loss,
    )
    parameters = [parameter for parameter in agent.model.parameters() if parameter.requires_grad]
    return _gradient_conflict_values(
        base_loss=base_loss,
        distill_loss=distill_loss,
        parameters=parameters,
    )


def train_neural_local_policy(
    agent: NeuralPDAgent,
    observation: torch.Tensor,
    action: int,
    payoff: float,
    peer_actions: np.ndarray,
    config: Toy2Config,
    signed_counterfactual_advantage: float | None = None,
    teacher_distill_probability: float | None = None,
    teacher_distill_weight: float = 0.0,
    base_loss_weight: float = 1.0,
) -> float:
    if config.policy.domain.local_update_rule == "sampled_policy_gradient":
        return train_local_policy(
            agent=agent,
            observation=observation,
            action=action,
            payoff=payoff,
            config=config,
        )
    if config.policy.domain.local_update_rule == "counterfactual_advantage":
        return train_counterfactual_policy(
            agent=agent,
            observation=observation,
            peer_actions=peer_actions,
            config=config,
            signed_advantage=signed_counterfactual_advantage,
            teacher_distill_probability=teacher_distill_probability,
            teacher_distill_weight=teacher_distill_weight,
            teacher_distill_loss=config.policy.domain.bootstrap.distill_loss,
            base_loss_weight=base_loss_weight,
        )
    raise ValueError(
        f"Unsupported Toy 2 local update rule: {config.policy.domain.local_update_rule}"
    )


def transformed_advantages(values: np.ndarray, config: Toy2Config) -> np.ndarray:
    if config.policy.domain.payoff_transform == "linear":
        transformed = values
    elif config.policy.domain.payoff_transform == "tanh":
        transformed = np.tanh(values)
    else:
        raise ValueError(
            f"Unsupported payoff transform: {config.policy.domain.payoff_transform}"
        )
    return config.policy.selection_strength * transformed


def transformed_advantages_tensor(
    values: torch.Tensor,
    config: Toy2Config,
) -> torch.Tensor:
    if config.policy.domain.payoff_transform == "linear":
        transformed = values
    elif config.policy.domain.payoff_transform == "tanh":
        transformed = torch.tanh(values)
    else:
        raise ValueError(
            f"Unsupported payoff transform: {config.policy.domain.payoff_transform}"
        )
    return config.policy.selection_strength * transformed


def sampled_policy_gradient_advantages(
    payoffs: np.ndarray,
    payoff_baseline: np.ndarray,
    config: Toy2Config,
) -> np.ndarray:
    safe_scale = max(payoff_scale(config), 1e-8)
    return transformed_advantages((payoffs - payoff_baseline) / safe_scale, config)


def sampled_policy_gradient_advantages_tensor(
    payoffs: StateArray,
    payoff_baseline: StateArray,
    config: Toy2Config,
    *,
    device: torch.device,
) -> torch.Tensor:
    safe_scale = max(payoff_scale(config), 1e-8)
    payoff_values = _tensor_vector(payoffs, dtype=torch.float64, device=device)
    baseline_values = _tensor_vector(
        payoff_baseline,
        dtype=torch.float64,
        device=device,
    )
    return transformed_advantages_tensor(
        (payoff_values - baseline_values) / safe_scale,
        config,
    )


def counterfactual_policy_targets_and_advantages(
    actions: np.ndarray,
    peer_ids: list[list[int]],
    config: Toy2Config,
    peer_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    components = counterfactual_policy_advantage_components(
        actions=actions,
        peer_ids=peer_ids,
        config=config,
        peer_index=peer_index,
    )
    signed_advantages = components.effective
    target_actions = np.where(signed_advantages >= 0.0, 1, 0).astype(np.int64)
    return target_actions, np.abs(signed_advantages)


def counterfactual_policy_targets_and_advantages_tensor(
    actions: StateArray,
    peer_ids: list[list[int]],
    config: Toy2Config,
    *,
    peer_index: np.ndarray | torch.Tensor | None = None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe_scale = max(payoff_scale(config), 1e-8)
    action_values = _tensor_vector(actions, dtype=torch.long, device=device)
    if peer_index is not None:
        peer_index_tensor = _tensor_peer_index(peer_index, device=device)
        if peer_index_tensor.shape[0] != action_values.shape[0]:
            raise ValueError("peer_index first dimension must match action count")
        if peer_index_tensor.shape[1] == 0:
            payoff_deltas = torch.zeros(
                action_values.shape[0],
                dtype=torch.float64,
                device=device,
            )
        else:
            matrix = payoff_matrix_tensor(config, device=device)
            peer_actions = action_values[peer_index_tensor]
            defect_payoffs = matrix[0, peer_actions].mean(dim=1)
            cooperate_payoffs = matrix[1, peer_actions].mean(dim=1)
            payoff_deltas = (cooperate_payoffs - defect_payoffs) / safe_scale
    else:
        deltas: list[torch.Tensor] = []
        matrix = payoff_matrix_tensor(config, device=device)
        zero = torch.zeros((), dtype=torch.float64, device=device)
        for peers in peer_ids:
            if not peers:
                deltas.append(zero)
                continue
            peer_index_tensor = torch.as_tensor(peers, dtype=torch.long, device=device)
            peer_actions = action_values.index_select(0, peer_index_tensor)
            defect_payoff = matrix[0, peer_actions].mean()
            cooperate_payoff = matrix[1, peer_actions].mean()
            deltas.append((cooperate_payoff - defect_payoff) / safe_scale)
        if deltas:
            payoff_deltas = torch.stack(deltas)
        else:
            payoff_deltas = torch.zeros(0, dtype=torch.float64, device=device)
    signed_advantages = transformed_advantages_tensor(payoff_deltas, config)
    target_actions = torch.where(
        signed_advantages >= 0.0,
        torch.ones_like(action_values),
        torch.zeros_like(action_values),
    )
    return target_actions.to(dtype=torch.long), torch.abs(signed_advantages)


def _active_agent_ids_from_revision_mask(
    revision_mask: np.ndarray,
) -> list[int] | None:
    mask = np.asarray(revision_mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("revision_mask must be 1D")
    if bool(mask.all()):
        return None
    return [int(agent_id) for agent_id in np.flatnonzero(mask)]


def neural_local_policy_loss_inputs(
    *,
    agents: list[NeuralPDAgent],
    actions: np.ndarray,
    payoffs: np.ndarray,
    peer_ids: list[list[int]],
    revision_mask: np.ndarray,
    config: Toy2Config,
    peer_index: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
    if config.policy.domain.local_update_rule == "sampled_policy_gradient":
        payoff_baseline = np.asarray(
            [agent.payoff_ema for agent in agents],
            dtype=np.float64,
        )
        return (
            actions.astype(np.int64, copy=False),
            sampled_policy_gradient_advantages(payoffs, payoff_baseline, config),
            _active_agent_ids_from_revision_mask(revision_mask),
        )
    if config.policy.domain.local_update_rule == "counterfactual_advantage":
        target_actions, advantages = counterfactual_policy_targets_and_advantages(
            actions=actions,
            peer_ids=peer_ids,
            config=config,
            peer_index=peer_index,
        )
        return target_actions, advantages, None
    raise ValueError(
        f"Unsupported Toy 2 local update rule: {config.policy.domain.local_update_rule}"
    )


def neural_local_policy_loss_inputs_tensor(
    *,
    actions: StateArray,
    payoffs: StateArray,
    payoff_baseline: StateArray,
    peer_ids: list[list[int]],
    revision_mask: np.ndarray,
    config: Toy2Config,
    device: torch.device,
    peer_index: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[int] | None]:
    if config.policy.domain.local_update_rule == "sampled_policy_gradient":
        return (
            _tensor_vector(actions, dtype=torch.long, device=device),
            sampled_policy_gradient_advantages_tensor(
                payoffs,
                payoff_baseline,
                config,
                device=device,
            ),
            _active_agent_ids_from_revision_mask(revision_mask),
        )
    if config.policy.domain.local_update_rule == "counterfactual_advantage":
        target_actions, advantages = counterfactual_policy_targets_and_advantages_tensor(
            actions=actions,
            peer_ids=peer_ids,
            config=config,
            peer_index=peer_index,
            device=device,
        )
        return target_actions, advantages, None
    raise ValueError(
        f"Unsupported Toy 2 local update rule: {config.policy.domain.local_update_rule}"
    )


def train_neural_local_policies_batched(
    agents: list[NeuralPDAgent],
    observations: torch.Tensor,
    actions: np.ndarray,
    payoffs: np.ndarray,
    peer_ids: list[list[int]],
    revision_mask: np.ndarray,
    config: Toy2Config,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
    timing_context: BinaryStepContext | None = None,
    peer_index: np.ndarray | None = None,
) -> LossVector:
    return train_neural_local_policies_batched_update(
        agents=agents,
        observations=observations,
        actions=actions,
        payoffs=payoffs,
        peer_ids=peer_ids,
        revision_mask=revision_mask,
        config=config,
        parameters=parameters,
        adam_state_cache=adam_state_cache,
        synchronize_model_parameters=synchronize_model_parameters,
        synchronize_optimizer_states=synchronize_optimizer_states,
        timing_context=timing_context,
        peer_index=peer_index,
    ).losses


def train_neural_local_policies_batched_update(
    agents: list[NeuralPDAgent],
    observations: torch.Tensor,
    actions: np.ndarray,
    payoffs: np.ndarray,
    peer_ids: list[list[int]],
    revision_mask: np.ndarray,
    config: Toy2Config,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
    timing_context: BinaryStepContext | None = None,
    peer_index: np.ndarray | None = None,
) -> BatchedMLPUpdateResult:
    loss_actions, advantages, active_agent_ids = neural_local_policy_loss_inputs(
        agents=agents,
        actions=actions,
        payoffs=payoffs,
        peer_ids=peer_ids,
        revision_mask=revision_mask,
        config=config,
        peer_index=peer_index,
    )
    report = run_batched_policy_gradient_local_update(
        agents=agents,
        observations=observations,
        actions=loss_actions,
        advantages=advantages,
        active_agent_ids=active_agent_ids,
        entropy_beta=config.environment.entropy_beta,
        parameters=parameters,
        adam_state_cache=adam_state_cache,
        synchronize_model_parameters=synchronize_model_parameters,
        synchronize_optimizer_states=synchronize_optimizer_states,
        timing_context=timing_context,
    )
    if report.update_result is None:
        raise RuntimeError("batched local update adapter did not produce a result")
    return report.update_result


def train_neural_local_policies_tensor_batched_update(
    *,
    runtime: TensorBatchedMLPRuntime,
    agents: list[NeuralPDAgent],
    observations: torch.Tensor,
    actions: StateArray,
    payoffs: StateArray,
    peer_ids: list[list[int]],
    revision_mask: np.ndarray,
    config: Toy2Config,
    timing_context: BinaryStepContext | None = None,
    peer_index: np.ndarray | torch.Tensor | None = None,
    payoff_baseline: StateArray | None = None,
) -> BatchedMLPUpdateResult:
    if isinstance(actions, torch.Tensor) or isinstance(payoffs, torch.Tensor):
        baseline = (
            torch.as_tensor(
                [agent.payoff_ema for agent in agents],
                dtype=torch.float64,
                device=runtime.device,
            )
            if payoff_baseline is None
            else payoff_baseline
        )
        loss_actions, advantages, active_agent_ids = (
            neural_local_policy_loss_inputs_tensor(
                actions=actions,
                payoffs=payoffs,
                payoff_baseline=baseline,
                peer_ids=peer_ids,
                revision_mask=revision_mask,
                config=config,
                device=runtime.device,
                peer_index=peer_index,
            )
        )
    else:
        loss_actions, advantages, active_agent_ids = neural_local_policy_loss_inputs(
            agents=agents,
            actions=actions,
            payoffs=payoffs,
            peer_ids=peer_ids,
            revision_mask=revision_mask,
            config=config,
            peer_index=peer_index,
        )
    report = run_tensor_runtime_policy_gradient_local_update(
        runtime=runtime,
        observations=observations,
        actions=loss_actions,
        advantages=advantages,
        active_agent_ids=active_agent_ids,
        entropy_beta=config.environment.entropy_beta,
        timing_context=timing_context,
    )
    if report.update_result is None:
        raise RuntimeError("tensor local update adapter did not produce a result")
    return report.update_result


def output_similarity_matrix(policy_probs: torch.Tensor) -> np.ndarray:
    coop_probs = policy_probs[:, 1].detach().cpu().numpy()
    return scalar_output_similarity_matrix(coop_probs)


def select_peers(
    neighbors: list[list[int]],
    peer_rule: str,
    threshold: float,
    policy_probs: torch.Tensor,
) -> tuple[list[list[int]], np.ndarray | None]:
    coop_probs = policy_probs[:, 1].detach().cpu().numpy()
    try:
        result = SocialBlock(alpha=0.0).select_scalar_output_peers(
            neighbors=neighbors,
            values=coop_probs,
            peer_rule=peer_rule,
            threshold=threshold,
        )
    except ValueError as exc:
        if "Unsupported peer rule" in str(exc):
            raise ValueError(f"Unsupported Toy 2 peer rule: {peer_rule}") from exc
        raise
    return result.peer_ids, result.similarity


def apply_output_average(
    agents: list[NeuralPDAgent],
    peer_ids: list[list[int]],
    alpha: float,
    observations: torch.Tensor,
    previous_probs: torch.Tensor,
) -> list[float]:
    return apply_binary_output_distribution_distillation(
        agents=agents,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        logits_fn=lambda agent, agent_id, observed: agent.model(
            observed[agent_id].unsqueeze(0)
        ),
        loss_mode="kl",
    )


def apply_output_average_distillation_batched(
    agents: list[NeuralPDAgent],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    validate_peers: bool = True,
    timing_context: BinaryStepContext | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
) -> LossVector:
    return apply_output_average_distillation_batched_update(
        agents=agents,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        parameters=parameters,
        adam_state_cache=adam_state_cache,
        validate_peers=validate_peers,
        timing_context=timing_context,
        synchronize_model_parameters=synchronize_model_parameters,
        synchronize_optimizer_states=synchronize_optimizer_states,
    ).losses


def apply_output_average_distillation_batched_update(
    agents: list[NeuralPDAgent],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    validate_peers: bool = True,
    timing_context: BinaryStepContext | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
) -> BatchedMLPUpdateResult:
    return apply_batched_output_average_distillation_update(
        agents=agents,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        parameters=parameters,
        adam_state_cache=adam_state_cache,
        validate_peers=validate_peers,
        timing_context=timing_context,
        loss_mode="kl",
        synchronize_model_parameters=synchronize_model_parameters,
        synchronize_optimizer_states=synchronize_optimizer_states,
    )


def apply_output_average_distillation_tensor_batched_update(
    runtime: TensorBatchedMLPRuntime,
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    uniform_peer_count: int | None = None,
    uniform_peer_index: torch.Tensor | None = None,
    peer_index_cache: PeerIndexCache | None = None,
    validate_peers: bool = True,
    timing_context: BinaryStepContext | None = None,
) -> BatchedMLPUpdateResult:
    return apply_tensor_output_average_distillation_update(
        runtime=runtime,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        uniform_peer_count=uniform_peer_count,
        uniform_peer_index=uniform_peer_index,
        peer_index_cache=peer_index_cache,
        validate_peers=validate_peers,
        timing_context=timing_context,
        loss_mode="kl",
    )


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def cooperation_probs_to_policy_tensor(
    cooperation_probs: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        binary_policy_matrix(cooperation_probs),
        dtype=torch.float32,
        device=device,
    )


def sample_cooperation_probs(
    cooperation_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    clipped = np.clip(cooperation_probs, 0.0, 1.0)
    return (rng.random(len(clipped)) < clipped).astype(np.int64)


def fermi_imitation_cooperation_probs(
    actions: np.ndarray,
    payoffs: np.ndarray,
    neighbors: list[list[int]],
    revision_mask: np.ndarray,
    rng: np.random.Generator,
    config: Toy2Config,
) -> np.ndarray:
    safe_scale = max(payoff_scale(config), 1e-8)
    cooperation_probs = actions.astype(np.float64)
    for agent_id, peer_ids in enumerate(neighbors):
        if not revision_mask[agent_id]:
            continue
        if not peer_ids:
            continue
        peer_id = int(rng.choice(peer_ids))
        payoff_delta = (float(payoffs[peer_id]) - float(payoffs[agent_id])) / safe_scale
        adoption_probability = sigmoid(transformed_advantage(payoff_delta, config))
        cooperation_probs[agent_id] = adoption_probability * float(actions[peer_id]) + (
            1.0 - adoption_probability
        ) * float(actions[agent_id])
    return cooperation_probs


def reputation_params_from_config(config: Toy2Config) -> ReputationParams:
    return ReputationParams(
        enabled=config.state.reputation.enabled,
        decay=config.state.reputation.decay,
        peer_rule=config.state.reputation.peer_rule,
        temperature=config.state.reputation.temperature,
        noise=config.state.reputation.noise,
    )


def neural_observation_input_dim(config: Toy2Config) -> int:
    return 6 + reputation_observation_extra_dim(config.state.reputation.observation_mode)


def validate_tensor_batched_backend_config(config: Toy2Config) -> None:
    if config.policy.rule != "neural_policy":
        raise ValueError("Toy 2 tensor_batched requires policy.rule='neural_policy'")
    if config.agents.optimizer.name != "adam":
        raise ValueError("Toy 2 tensor_batched requires Adam optimizer")
    model = config.agents.model
    if model.activation != "relu" or model.output_dim != 2:
        raise ValueError(
            "Toy 2 tensor_batched requires the standard one-hidden-layer ReLU MLP"
        )


def mobility_params_from_config(config: Toy2Config) -> MobilityParams:
    return MobilityParams(
        enabled=config.state.mobility.enabled,
        rate=config.state.mobility.rate,
        candidate_pool_size=config.state.mobility.candidate_pool_size,
        selection_rule=config.state.mobility.selection_rule,
        move_cost=config.state.mobility.move_cost,
    )


def reputation_peer_ids(
    config: Toy2Config,
    neighbors: list[list[int]],
    agent_count: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    if config.state.reputation.peer_rule == "spatial":
        return neighbors
    if config.state.reputation.peer_rule == "well_mixed":
        return well_mixed_peer_ids(
            neighbors=neighbors,
            agent_count=agent_count,
            rng=rng,
        )
    raise ValueError(
        f"Unsupported Toy 2 reputation peer rule: {config.state.reputation.peer_rule}"
    )


def reputation_imitation_cooperation_probs(
    actions: np.ndarray,
    reputation: np.ndarray,
    neighbors: list[list[int]],
    revision_mask: np.ndarray,
    rng: np.random.Generator,
    config: Toy2Config,
) -> np.ndarray:
    return apply_reputation_imitation(
        actions=actions,
        reputation=reputation,
        peer_ids=neighbors,
        revision_mask=revision_mask,
        rng=rng,
        params=reputation_params_from_config(config),
    )


def sample_revised_cooperation_probs(
    current_actions: np.ndarray,
    cooperation_probs: np.ndarray,
    revision_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if bool(np.all(revision_mask)):
        return sample_cooperation_probs(cooperation_probs, rng=rng)
    if not bool(np.any(revision_mask)):
        return np.array(current_actions, dtype=np.int64, copy=True)
    next_actions = np.array(current_actions, dtype=np.int64, copy=True)
    revised_indices = np.flatnonzero(revision_mask)
    revised_actions = sample_cooperation_probs(
        cooperation_probs[revised_indices],
        rng=rng,
    )
    next_actions[revised_indices] = revised_actions
    return next_actions


def apply_output_average_to_cooperation_probs(
    cooperation_probs: np.ndarray,
    peer_ids: list[list[int]],
    alpha: float,
) -> tuple[np.ndarray, list[float]]:
    """Compatibility wrapper for the shared binary output-average mixer."""

    return mix_binary_output_average(cooperation_probs, peer_ids, alpha)


def policy_consensus(policy_probs: torch.Tensor) -> float:
    coop = policy_probs[:, 1].detach()
    count = int(coop.shape[0])
    if count <= 1:
        return 1.0
    sorted_coop = torch.sort(coop.to(dtype=torch.float64)).values
    coefficients = (
        2.0
        * torch.arange(count, dtype=torch.float64, device=sorted_coop.device)
        - float(count - 1)
    )
    pair_count = count * (count - 1) / 2.0
    mean_abs_difference = torch.sum(coefficients * sorted_coop) / pair_count
    return float((1.0 - mean_abs_difference).cpu())


def cooperation_cluster_metrics_from_edges(
    actions: StateArray,
    edges: np.ndarray,
) -> tuple[int, float]:
    action_values = to_numpy_view(actions, dtype=np.int64)
    cooperator_mask = action_values == 1
    cooperator_count = int(np.sum(cooperator_mask))
    if cooperator_count == 0:
        return 0, 0.0
    agent_count = len(action_values)
    parent = np.arange(agent_count, dtype=np.int64)
    sizes = np.ones(agent_count, dtype=np.int64)
    if len(edges) > 0:
        active_edges = edges[
            cooperator_mask[edges[:, 0]] & cooperator_mask[edges[:, 1]]
        ]
    else:
        active_edges = edges

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    for source, target in active_edges:
        source_root = find(int(source))
        target_root = find(int(target))
        if source_root == target_root:
            continue
        if sizes[source_root] < sizes[target_root]:
            source_root, target_root = target_root, source_root
        parent[target_root] = source_root
        sizes[source_root] += sizes[target_root]

    roots = np.fromiter(
        (find(int(agent_id)) for agent_id in np.flatnonzero(cooperator_mask)),
        dtype=np.int64,
        count=cooperator_count,
    )
    _, component_sizes = np.unique(roots, return_counts=True)
    return int(len(component_sizes)), float(int(component_sizes.max()) / agent_count)


def cooperation_cluster_metrics(
    actions: StateArray, graph: nx.Graph
) -> tuple[int, float]:
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)
    return cooperation_cluster_metrics_from_edges(actions, edges)


def make_run_dir(config: Toy2Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        config.run.output_dir
        / f"{timestamp}_{config.run.name}_seed{config.run.seed:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def canonical_decision_metadata(config: Toy2Config) -> dict[str, object]:
    if config.policy.rule == "neural_policy":
        decision = config.policy.decision.model_dump()
        decision["decision_threshold"] = (
            validate_toy2_payoff_threshold(config.game.payoff)
            if decision["calibration"]["mode"] == "payoff_threshold"
            else None
        )
        return decision
    return {
        "mode": "sampled",
        "action_temperature": 1.0,
        "exploration_epsilon": 0.0,
        "terminal_argmax_epochs": 0,
        "calibration": {
            "mode": "none",
            "strength": 4.0,
        },
        "decision_threshold": None,
    }


def write_run_metadata(config_path: Path, config: Toy2Config, run_dir: Path) -> None:
    decision = canonical_decision_metadata(config)
    write_run_metadata_artifacts(
        config_path=config_path,
        config=config,
        run_dir=run_dir,
        toy="toy2",
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "domain_game_family": config.game.family,
            "domain_payoff": config.game.payoff.model_dump(),
            "policy_rule": config.policy.rule,
            "selection_strength": config.policy.selection_strength,
            "policy_temperature": config.policy.temperature,
            "local_update_rule": config.policy.domain.local_update_rule,
            "neural_peer_mode": config.policy.domain.neural_peer_mode,
            "interaction_mode": config.policy.domain.interaction_mode,
            "decision": decision,
            "decision_mode": decision["mode"],
            "action_temperature": decision["action_temperature"],
            "terminal_argmax_epochs": decision["terminal_argmax_epochs"],
            "decision_calibration_mode": decision["calibration"]["mode"],
            "decision_calibration_strength": decision["calibration"]["strength"],
            "decision_threshold": decision["decision_threshold"],
            "payoff_transform": config.policy.domain.payoff_transform,
            "exploration_epsilon": decision["exploration_epsilon"],
            "learning_enabled": config.policy.learning_enabled,
            "policy_revision_rate": config.policy.revision_rate,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "reputation": config.state.reputation.model_dump(),
            "mobility": config.state.mobility.model_dump(),
            "init_mode": config.agents.init_mode,
            "policy_prior_action_probability": (
                config.agents.policy_prior_action_probability
            ),
            "grid_width": config.environment.grid_width,
            "grid_height": config.environment.grid_height,
        },
    )


def aggregate_context(config: Toy2Config) -> dict[str, object]:
    payoff = config.game.payoff
    return {
        "domain_game_family": config.game.family,
        "domain_payoff_T": payoff.T,
        "domain_payoff_R": payoff.R,
        "domain_payoff_P": payoff.P,
        "domain_payoff_S": payoff.S,
        "policy_rule": config.policy.rule,
    }


def micro_context(config: Toy2Config) -> dict[str, object]:
    return {
        "domain_game_family": config.game.family,
        "policy_rule": config.policy.rule,
    }


def restrict_peer_ids_to_revised(
    peer_ids: list[list[int]],
    revision_mask: np.ndarray,
) -> list[list[int]]:
    return [
        list(peers) if bool(revision_mask[agent_id]) else []
        for agent_id, peers in enumerate(peer_ids)
    ]


def peer_ids_for_aggregate(
    config: Toy2Config,
    neighbors: list[list[int]],
    policy_probs: torch.Tensor,
    agent_count: int,
) -> list[list[int]]:
    if config.coordination.mixer == "none":
        return [[] for _ in range(agent_count)]
    peer_ids, _ = select_peers(
        neighbors=neighbors,
        peer_rule=config.coordination.peer_rule,
        threshold=config.coordination.threshold,
        policy_probs=policy_probs,
    )
    return peer_ids


def spatial_aggregate_row(
    config: Toy2Config,
    epoch: int,
    actions: np.ndarray,
    payoffs: np.ndarray,
    policy_probs: torch.Tensor,
    peer_ids: list[list[int]],
    graph: nx.Graph,
    agent_count: int,
    realized_revision_rate: float,
    reputation: np.ndarray | None = None,
    mobility_result: MobilityStepResult | None = None,
    policy_probs_pre_revision: torch.Tensor | None = None,
    policy_probs_post_local: torch.Tensor | None = None,
    local_losses: list[float] | None = None,
    revised_local_losses: list[float] | None = None,
    social_losses: list[float] | None = None,
) -> dict[str, object]:
    del agent_count
    cooperation_components, largest_coop_fraction = cooperation_cluster_metrics(
        actions,
        graph,
    )
    return {
        **binary_aggregate_common_fields(
            config=config,
            toy="toy2",
            epoch=epoch,
            actions=actions,
            payoffs=payoffs,
            policy_probs=policy_probs,
            peer_ids=peer_ids,
            realized_revision_rate=realized_revision_rate,
            reputation=reputation,
            mobility_result=mobility_result,
            policy_probs_pre_revision=policy_probs_pre_revision,
            policy_probs_post_local=policy_probs_post_local,
            local_losses=local_losses,
            revised_local_losses=revised_local_losses,
            social_losses=social_losses,
            include_edge_entropy=True,
        ),
        **aggregate_context(config),
        "domain_policy_consensus": policy_consensus(policy_probs),
        "domain_action_components": cooperation_components,
        "domain_largest_action_cluster_fraction": largest_coop_fraction,
    }


def run_toy2_rd_well_mixed(
    config: Toy2Config,
    config_path: Path,
) -> BinaryToyResult:
    run_dir = make_run_dir(config)
    write_run_metadata(config_path=config_path, config=config, run_dir=run_dir)
    micro_writer = CsvLogWriter(run_dir / "micro_state.csv", TOY2_MICRO_STATE_FIELDS)
    aggregate_writer = CsvLogWriter(
        run_dir / "aggregate_metrics.csv",
        TOY2_AGGREGATE_FIELDS,
    )

    payoff = config.game.payoff
    x = float(config.environment.initial_action_probability)
    rd_domain_fields = {
        "domain_game_family": config.game.family,
        "domain_payoff_T": payoff.T,
        "domain_payoff_R": payoff.R,
        "domain_payoff_P": payoff.P,
        "domain_payoff_S": payoff.S,
        "domain_policy_consensus": 1.0,
        "domain_action_components": 0,
        "domain_largest_action_cluster_fraction": 0.0,
    }

    final_mean_payoff = 0.0
    try:
        if config.logging.aggregate_metrics:
            initial_cooperator_payoff = x * payoff.R + (1.0 - x) * payoff.S
            initial_defector_payoff = x * payoff.T + (1.0 - x) * payoff.P
            initial_mean_payoff = (
                x * initial_cooperator_payoff + (1.0 - x) * initial_defector_payoff
            )
            aggregate_writer.write(
                {
                    **aggregate_context(config),
                    "run_id": config.run.name,
                    "seed": config.run.seed,
                    "epoch": 0,
                    "coordination_mixer": config.coordination.mixer,
                    "coordination_peer_rule": config.coordination.peer_rule,
                    "policy_revision_rate": config.policy.revision_rate,
                    "realized_revision_rate": 0.0,
                    "toy": "toy2",
                    "action_rate": x,
                    "mean_payoff": initial_mean_payoff,
                    "mean_policy_action_probability": x,
                    "mean_policy_action_probability_pre_revision": x,
                    "mean_policy_action_probability_post_local": x,
                    "mean_policy_action_probability_post_social": x,
                    "mean_local_loss": 0.0,
                    "mean_revised_local_loss": 0.0,
                    "mean_social_loss": 0.0,
                    "mean_reputation": 0.0,
                    "reputation_dispersion": 0.0,
                    "mobility_rate": 0.0,
                    "mean_mobility_gain": 0.0,
                    "fragmentation_components": 0,
                    "mean_peer_count": 0.0,
                    "edge_entropy": 0.0,
                    **rd_domain_fields,
                }
            )
        for epoch in range(1, config.simulation.epochs + 1):
            cooperator_payoff = x * payoff.R + (1.0 - x) * payoff.S
            defector_payoff = x * payoff.T + (1.0 - x) * payoff.P
            final_mean_payoff = x * cooperator_payoff + (1.0 - x) * defector_payoff
            advantage = (cooperator_payoff - defector_payoff) / payoff_scale(config)
            delta = x * (1.0 - x) * transformed_advantage(advantage, config)
            x = float(np.clip(x + delta, 0.0, 1.0))
            if config.logging.aggregate_metrics:
                aggregate_writer.write(
                    {
                        **aggregate_context(config),
                        "run_id": config.run.name,
                        "seed": config.run.seed,
                        "epoch": epoch,
                        "coordination_mixer": config.coordination.mixer,
                        "coordination_peer_rule": config.coordination.peer_rule,
                        "policy_revision_rate": config.policy.revision_rate,
                        "realized_revision_rate": 1.0,
                        "toy": "toy2",
                        "action_rate": x,
                        "mean_payoff": final_mean_payoff,
                        "mean_policy_action_probability": x,
                        "mean_policy_action_probability_pre_revision": x,
                        "mean_policy_action_probability_post_local": x,
                        "mean_policy_action_probability_post_social": x,
                        "mean_local_loss": 0.0,
                        "mean_revised_local_loss": 0.0,
                        "mean_social_loss": 0.0,
                        "mean_reputation": 0.0,
                        "reputation_dispersion": 0.0,
                        "mobility_rate": 0.0,
                        "mean_mobility_gain": 0.0,
                        "fragmentation_components": 0,
                        "mean_peer_count": 0.0,
                        "edge_entropy": 0.0,
                        **rd_domain_fields,
                    }
                )
    finally:
        micro_writer.close()
        aggregate_writer.close()

    domain_metrics = rd_domain_fields
    write_binary_summary_artifact(
        run_dir=run_dir,
        toy="toy2",
        final_action_rate=x,
        final_mean_payoff=final_mean_payoff,
        final_fragmentation_components=0,
        final_mean_policy_action_probability=x,
        final_mean_reputation=0.0,
        final_reputation_dispersion=0.0,
        domain_metrics=domain_metrics,
    )
    return BinaryToyResult(
        run_dir=run_dir,
        toy="toy2",
        final_action_rate=x,
        final_mean_payoff=final_mean_payoff,
        final_fragmentation_components=0,
        final_mean_policy_action_probability=x,
        final_mean_reputation=0.0,
        final_reputation_dispersion=0.0,
        domain_metrics=domain_metrics,
    )


@dataclass
class Toy2SpatialDomain(BinaryToyDomainBase):
    """Toy 2 adapter for the shared binary spatial lifecycle runner."""

    config: Toy2Config
    config_path: Path
    rng: np.random.Generator
    neural_peer_rng: np.random.Generator
    interaction_rng: np.random.Generator
    reputation_rng: np.random.Generator
    mobility_rng: np.random.Generator
    device: torch.device
    neural_update_backend: NeuralUpdateBackend = "loop"
    policy_cache: BatchedMLPPolicyCache | None = field(default=None, init=False)
    tensor_runtime: TensorBatchedMLPRuntime | None = field(default=None, init=False)
    _adam_state_cache: BatchedAdamStateCache | None = field(default=None, init=False)
    _pending_policy_cache_parameters: BatchedMLPParameters | None = field(
        default=None,
        init=False,
    )
    _spatial_peer_index: np.ndarray | None = field(default=None, init=False)
    _spatial_peer_index_tensor: torch.Tensor | None = field(default=None, init=False)
    _uniform_neighbor_peer_count: int | None = field(default=None, init=False)
    _neighbor_peer_index_cache: PeerIndexCache | None = field(default=None, init=False)
    _static_peer_metrics: dict[str, float | int] = field(
        default_factory=dict,
        init=False,
    )
    _graph_edges: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int64),
        init=False,
    )
    _basin_critic: PrototypePhaseBasinCritic = field(
        default_factory=PrototypePhaseBasinCritic,
        init=False,
    )
    _basin_transition_samples: list[dict[str, object]] = field(
        default_factory=list,
        init=False,
    )
    _learned_basin_critic_bundle: LearnedBasinPhaseCriticBundle | None = field(
        default=None,
        init=False,
    )
    _learned_basin_replay_weight_scorer: LearnedBasinReplayWeightScorer | None = field(
        default=None,
        init=False,
    )

    micro_state_fields: ClassVar[list[str]] = TOY2_MICRO_STATE_FIELDS
    aggregate_fields: ClassVar[list[str]] = TOY2_AGGREGATE_FIELDS
    toy: ClassVar[str] = "toy2"
    include_edge_entropy: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.neural_update_backend not in {"loop", "batched", "tensor_batched"}:
            raise ValueError(
                "Toy 2 neural_update_backend must be 'loop', 'batched', "
                "or 'tensor_batched'"
            )
        if self.neural_update_backend == "tensor_batched":
            validate_tensor_batched_backend_config(self.config)
        if (
            self.config.policy.domain.objective.uses_state_continuation()
            and self.neural_update_backend != "loop"
        ):
            raise ValueError(
                "Toy 2 state_continuation objective requires "
                "neural_update_backend='loop'"
            )
        if (
            self.config.policy.domain.basin_credit.enabled
            and self.neural_update_backend != "loop"
        ):
            raise ValueError("Toy 2 basin credit requires neural_update_backend='loop'")
        self.graph = build_spatial_graph(self.config)
        self.agent_count = self.config.environment.grid_width * self.config.environment.grid_height
        self._graph_edges = np.asarray(list(self.graph.edges()), dtype=np.int64)
        if self._graph_edges.size == 0:
            self._graph_edges = np.empty((0, 2), dtype=np.int64)
        self.neighbors = graph_neighbors(self.graph, self.agent_count)
        self._spatial_peer_index = uniform_peer_index(self.neighbors)
        self._uniform_neighbor_peer_count = (
            None
            if self._spatial_peer_index is None
            else int(self._spatial_peer_index.shape[1])
        )
        self._spatial_peer_index_tensor = (
            None
            if self._spatial_peer_index is None
            else torch.as_tensor(
                self._spatial_peer_index,
                dtype=torch.long,
                device=self.device,
            )
        )
        self._neighbor_peer_index_cache = (
            PeerIndexCache.from_peer_ids(self.neighbors, device=self.device)
            if self._uniform_neighbor_peer_count is None
            else None
        )
        self._static_peer_metrics = binary_peer_metrics(
            peer_ids=self.neighbors,
            agent_count=self.agent_count,
            include_edge_entropy=True,
        )
        self.decision_kernel = (
            DecisionKernel.from_config(self.config)
            if self.config.policy.rule == "neural_policy"
            else DecisionKernel()
        )

    def can_reuse_static_peer_ids(self) -> bool:
        return (
            self.config.policy.rule == "neural_policy"
            and self.config.coordination.peer_rule == "none"
            and self.config.coordination.mixer == "output_average"
        )

    def aggregate_peer_metrics(
        self,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, float | int] | None:
        del state
        if self.can_reuse_static_peer_ids() and step_result.peer_ids == self.neighbors:
            return self._static_peer_metrics
        return None

    def select_initial_peers(
        self,
        state: BinarySpatialState,
        policy_probs: Any,
    ) -> list[list[int]]:
        if self.config.coordination.mixer == "none":
            return [[] for _ in range(self.agent_count)]
        if self.can_reuse_static_peer_ids():
            return self.neighbors
        return super().select_initial_peers(state, policy_probs)

    def _uses_torch_state(self) -> bool:
        return (
            self.config.policy.rule == "neural_policy"
            and self.neural_update_backend == "tensor_batched"
        )

    def _spatial_neural_peer_index(self) -> np.ndarray | torch.Tensor | None:
        if self.config.policy.domain.neural_peer_mode != "spatial":
            return None
        if self._uses_torch_state():
            return self._spatial_peer_index_tensor
        return self._spatial_peer_index

    def _spatial_interaction_peer_index(self) -> np.ndarray | torch.Tensor | None:
        if self.config.policy.domain.interaction_mode != "spatial":
            return None
        if self._uses_torch_state():
            return self._spatial_peer_index_tensor
        return self._spatial_peer_index

    def _payoffs_for_peer_context(
        self,
        *,
        actions: StateArray,
        peer_ids: list[list[int]],
        peer_index: np.ndarray | torch.Tensor | None,
    ) -> StateArray:
        if peer_index is not None:
            return compute_payoffs_from_peer_index(
                actions=actions,
                peer_index=peer_index,
                config=self.config,
            )
        return compute_payoffs_from_peer_ids(
            actions=actions,
            peer_ids=peer_ids,
            config=self.config,
        )

    def make_run_dir(self) -> Path:
        return make_run_dir(self.config)

    def write_metadata(self, run_dir: Path) -> None:
        write_run_metadata(
            config_path=self.config_path,
            config=self.config,
            run_dir=run_dir,
        )

    def initial_state(self) -> BinarySpatialState:
        actions_np = initialize_actions(self.config, self.rng)
        if self.config.policy.rule == "neural_policy":
            if self._uses_torch_state():
                agents: list[NeuralPDAgent] = []
                self.tensor_runtime = create_tensor_batched_runtime(
                    self.config,
                    self.device,
                )
                self.policy_cache = None
            else:
                agents = create_agents(self.config, device=self.device)
                self.refresh_policy_cache(agents)
        else:
            agents = []
            self.tensor_runtime = None
            self.policy_cache = None
        if self._uses_torch_state():
            actions: StateArray = torch.as_tensor(
                actions_np,
                dtype=torch.long,
                device=self.device,
            )
            reputation: StateArray = (
                actions.to(dtype=torch.float64)
                if self.config.state.reputation.enabled
                else torch.zeros(self.agent_count, dtype=torch.float64, device=self.device)
            )
            payoffs = torch.zeros(self.agent_count, dtype=torch.float64, device=self.device)
            payoff_ema = torch.zeros(
                self.agent_count,
                dtype=torch.float64,
                device=self.device,
            )
            previous_payoff_ema = torch.zeros(
                self.agent_count,
                dtype=torch.float64,
                device=self.device,
            )
        else:
            actions = actions_np
            reputation = (
                actions_np.astype(np.float64)
                if self.config.state.reputation.enabled
                else np.zeros(self.agent_count, dtype=np.float64)
            )
            payoffs = np.zeros(self.agent_count, dtype=np.float64)
            payoff_ema = np.zeros(self.agent_count, dtype=np.float64)
            previous_payoff_ema = np.zeros(self.agent_count, dtype=np.float64)
        return BinarySpatialState(
            actions=actions,
            payoffs=payoffs,
            payoff_ema=payoff_ema,
            previous_payoff_ema=previous_payoff_ema,
            reputation=reputation,
            agents=agents,
        )

    def initial_step_result(
        self,
        state: BinarySpatialState,
    ) -> BinaryPolicyStepResult:
        initial_interaction_neighbors = interaction_peer_ids(
            config=self.config,
            neighbors=self.neighbors,
            agent_count=self.agent_count,
            rng=self.interaction_rng,
        )
        initial_payoffs = self._payoffs_for_peer_context(
            actions=state.actions,
            peer_ids=initial_interaction_neighbors,
            peer_index=self._spatial_interaction_peer_index(),
        )
        if self.config.policy.rule == "neural_policy":
            initial_context_neighbors = neural_context_peer_ids(
                config=self.config,
                neighbors=self.neighbors,
                agent_count=self.agent_count,
                rng=self.neural_peer_rng,
            )
            initial_observations = build_observations(
                actions=state.actions,
                payoffs=state.payoffs,
                agents=state.agents or [],
                neighbors=initial_context_neighbors,
                payoff_normalizer=payoff_scale(self.config),
                device=self.device,
                reputation=state.reputation,
                reputation_observation_mode=self.config.state.reputation.observation_mode,
                peer_index=self._spatial_neural_peer_index(),
                payoff_ema=state.payoff_ema,
                previous_payoff_ema=state.previous_payoff_ema,
            )
            initial_probs = self.collect_policy_probs(
                state.agents or [],
                initial_observations,
                temperature=self.config.policy.temperature,
            )
        else:
            initial_probs = cooperation_probs_to_policy_tensor(
                state.actions.astype(np.float64),
                device=self.device,
            )
        initial_peer_ids = peer_ids_for_aggregate(
            config=self.config,
            neighbors=self.neighbors,
            policy_probs=initial_probs,
            agent_count=self.agent_count,
        )
        return BinaryPolicyStepResult(
            pre_revision_probs=initial_probs,
            post_local_probs=initial_probs,
            post_social_probs=initial_probs,
            local_losses=[0.0 for _ in range(self.agent_count)],
            social_losses=[0.0 for _ in range(self.agent_count)],
            peer_ids=initial_peer_ids,
            revision_mask=np.zeros(self.agent_count, dtype=bool),
            mobility_result=MobilityStepResult.none(self.agent_count),
            realized_revision_rate=0.0,
            extras={
                "aggregate_payoffs": initial_payoffs,
                "decision_action_probs": initial_probs,
                "revised_local_losses": [],
                "logged_neighbors": self.neighbors,
            },
        )

    def build_step_context(
        self,
        epoch: int,
        state: BinarySpatialState,
        revision_mask: np.ndarray,
    ) -> BinaryStepContext:
        del state
        current_interaction_neighbors = interaction_peer_ids(
            config=self.config,
            neighbors=self.neighbors,
            agent_count=self.agent_count,
            rng=self.interaction_rng,
        )
        extras: dict[str, object] = {
            "current_interaction_neighbors": current_interaction_neighbors,
        }
        current_interaction_peer_index = self._spatial_interaction_peer_index()
        if current_interaction_peer_index is not None:
            extras["current_interaction_peer_index"] = current_interaction_peer_index
        return BinaryStepContext(
            epoch=epoch,
            revision_mask=revision_mask,
            extras=extras,
        )

    def basin_credit_diagnostics_for_actions(
        self,
        *,
        actions: np.ndarray,
        payoffs: np.ndarray,
        peer_ids: list[list[int]],
        revision_mask: np.ndarray,
        action_probabilities: np.ndarray,
        peer_index: np.ndarray | None = None,
    ) -> BasinCreditDiagnostics | None:
        basin_credit = self.config.policy.domain.basin_credit
        if not basin_credit.enabled:
            return None

        action_values = to_numpy_view(actions, dtype=np.int64)
        payoff_values = to_numpy_view(payoffs, dtype=np.float64)
        probability_values = np.asarray(action_probabilities, dtype=np.float64)
        target_payoff = toy2_target_basin_payoff(self.config)
        observed_embedding = build_basin_phase_representation(
            actions=action_values,
            payoffs=payoff_values,
            target_payoff=target_payoff,
            action_probabilities=probability_values,
        )
        observed_output = self._basin_critic.evaluate(
            observed_embedding,
            temperature=basin_credit.critic_temperature,
        )
        observed_score = float(observed_output.target_basin_score[0])
        observed_confidence = float(observed_output.phase_confidence[0])

        applied_mask = np.asarray(revision_mask, dtype=bool)
        selected_credit = np.zeros(len(action_values), dtype=np.float64)
        counterfactual_scores = np.full(
            len(action_values),
            np.nan,
            dtype=np.float64,
        )
        for agent_id in np.flatnonzero(applied_mask):
            counterfactual_actions = action_values.copy()
            counterfactual_actions[int(agent_id)] = 1 - counterfactual_actions[
                int(agent_id)
            ]
            counterfactual_payoffs = self._payoffs_for_peer_context(
                actions=counterfactual_actions,
                peer_ids=peer_ids,
                peer_index=peer_index,
            )
            counterfactual_probabilities = probability_values.copy()
            counterfactual_probabilities[int(agent_id)] = float(
                counterfactual_actions[int(agent_id)]
            )
            counterfactual_embedding = build_basin_phase_representation(
                actions=counterfactual_actions,
                payoffs=counterfactual_payoffs,
                target_payoff=target_payoff,
                action_probabilities=counterfactual_probabilities,
            )
            counterfactual_output = self._basin_critic.evaluate(
                counterfactual_embedding,
                temperature=basin_credit.critic_temperature,
            )
            counterfactual_score = float(counterfactual_output.target_basin_score[0])
            counterfactual_scores[int(agent_id)] = counterfactual_score
            selected_credit[int(agent_id)] = observed_score - counterfactual_score

        self._basin_critic.update(
            observed_embedding,
            target_reached=(
                float(np.mean(payoff_values)) >= target_payoff - 1e-12
                and float(np.mean(action_values)) >= 0.5
            ),
            decay=basin_credit.prototype_decay,
        )
        return build_basin_credit_diagnostics(
            basin_credit=basin_credit,
            selected_action_credit=selected_credit,
            score_observed=np.full(len(action_values), observed_score, dtype=np.float64),
            score_counterfactual=counterfactual_scores,
            applied_mask=applied_mask,
            phase_confidence=np.full(
                len(action_values),
                observed_confidence,
                dtype=np.float64,
            ),
        )

    def learned_basin_diagnostics_for_actions(
        self,
        *,
        actions: np.ndarray,
        payoffs: np.ndarray,
        action_probabilities: np.ndarray,
        prototype_diagnostics: BasinCreditDiagnostics | None,
    ) -> LearnedBasinRuntimeDiagnostics | None:
        basin_credit = self.config.policy.domain.basin_credit
        if not basin_credit.enabled or not basin_credit_needs_learned_runtime(
            basin_credit
        ):
            return None
        learned_model_path = basin_credit_learned_model_path(basin_credit)
        if learned_model_path is None:
            return None
        if self._learned_basin_critic_bundle is None:
            self._learned_basin_critic_bundle = load_learned_basin_phase_critic_bundle(
                learned_model_path
            )
        margin_threshold = (
            basin_credit.learned_credit_abstention_margin_threshold
            if basin_credit.learned_credit_enabled
            else basin_credit.learned_diagnostic_abstention_margin_threshold
        )
        uncertainty_threshold = (
            basin_credit.learned_credit_uncertainty_threshold
            if basin_credit.learned_credit_enabled
            else basin_credit.learned_diagnostic_uncertainty_threshold
        )
        prototype_advantage = (
            None
            if prototype_diagnostics is None
            else np.where(
                np.asarray(actions, dtype=np.int64) == 1,
                prototype_diagnostics.selected_action_credit,
                -prototype_diagnostics.selected_action_credit,
            )
        )
        return learned_basin_runtime_diagnostics(
            self._learned_basin_critic_bundle,
            actions=actions,
            payoffs=payoffs,
            action_probabilities=action_probabilities,
            target_payoff=toy2_target_basin_payoff(self.config),
            abstention_margin_threshold=margin_threshold,
            uncertainty_threshold=uncertainty_threshold,
            prototype_action1_advantage=prototype_advantage,
        )

    def learned_basin_replay_weight_scorer(
        self,
    ) -> LearnedBasinReplayWeightScorer | None:
        basin_credit = self.config.policy.domain.basin_credit
        path = basin_credit.learned_credit_replay_weight_model_path
        if path is None:
            return None
        if self._learned_basin_replay_weight_scorer is None:
            self._learned_basin_replay_weight_scorer = (
                load_learned_basin_replay_weight_scorer(path)
            )
        return self._learned_basin_replay_weight_scorer

    def local_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
    ) -> BinaryLocalStepResult:
        if self.config.policy.rule == "neural_policy":
            return self._local_step_neural(state=state, context=context)
        return self._local_step_classical(state=state, context=context)

    def _local_step_neural(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
    ) -> BinaryLocalStepResult:
        config = self.config
        agents = state.agents or []
        with timed_context_stage(context, "neural_context_peers"):
            context_neighbors = neural_context_peer_ids(
                config=config,
                neighbors=self.neighbors,
                agent_count=self.agent_count,
                rng=self.neural_peer_rng,
            )
            context_peer_index = self._spatial_neural_peer_index()
        with timed_context_stage(context, "build_observations"):
            observations = build_observations(
                actions=state.actions,
                payoffs=state.payoffs,
                agents=agents,
                neighbors=context_neighbors,
                payoff_normalizer=payoff_scale(config),
                device=self.device,
                reputation=state.reputation,
                reputation_observation_mode=config.state.reputation.observation_mode,
                peer_index=context_peer_index,
                payoff_ema=state.payoff_ema,
                previous_payoff_ema=state.previous_payoff_ema,
            )
        decision_probs: torch.Tensor | None = None

        def collect_pre_policy_probs(
            agents_arg: list[NeuralPDAgent],
            observations_arg: torch.Tensor,
            *,
            temperature: float,
        ) -> torch.Tensor:
            del temperature
            nonlocal decision_probs
            policy_logits = self.collect_policy_logits(agents_arg, observations_arg)
            decision_probs = torch.softmax(policy_logits, dim=-1)
            return torch.softmax(policy_logits / config.policy.temperature, dim=-1)

        def collect_post_policy_probs(
            agents_arg: list[NeuralPDAgent],
            observations_arg: torch.Tensor,
            *,
            temperature: float,
        ) -> torch.Tensor:
            del temperature
            return self.collect_policy_probs(
                agents_arg,
                observations_arg,
                temperature=config.policy.temperature,
            )

        decision_kernel = self.decision_kernel.for_epoch(
            epoch=context.epoch,
            total_epochs=config.simulation.epochs,
        )
        decision_bootstrap_diagnostics = None
        diagnostic_reputation_rng = np.random.default_rng(
            int(config.run.seed) + 1_000_003 * int(context.epoch) + 17
        )
        alignment_teacher_neighbors = reputation_peer_ids(
            config=config,
            neighbors=self.neighbors,
            agent_count=self.agent_count,
            rng=diagnostic_reputation_rng,
        )
        alignment_teacher_pre_probs = reputation_imitation_cooperation_probs(
            actions=to_numpy_view(state.actions, dtype=np.int64),
            reputation=to_numpy_view(state.reputation, dtype=np.float64),
            neighbors=alignment_teacher_neighbors,
            revision_mask=np.ones(self.agent_count, dtype=bool),
            rng=diagnostic_reputation_rng,
            config=config,
        )
        actions: StateArray = state.actions
        payoffs: StateArray = state.payoffs

        def build_decision_action_probs(
            pre_revision_probs: torch.Tensor,
        ) -> torch.Tensor:
            del pre_revision_probs
            nonlocal decision_bootstrap_diagnostics
            if decision_probs is None:
                raise RuntimeError("Toy 2 decision policy readout is missing")
            candidate_probs = decision_kernel.action_probs(decision_probs)
            if config.policy.domain.bootstrap.decision_enabled:
                scheduled_weight = domain_decision_bootstrap_weight(
                    config.policy.domain.bootstrap,
                    context.epoch,
                )
                teacher_probs = np.asarray([], dtype=np.float64)
                if scheduled_weight > 0.0:
                    teacher_neighbors = reputation_peer_ids(
                        config=config,
                        neighbors=self.neighbors,
                        agent_count=self.agent_count,
                        rng=self.reputation_rng,
                    )
                    teacher_probs = reputation_imitation_cooperation_probs(
                        actions=to_numpy_view(state.actions, dtype=np.int64),
                        reputation=to_numpy_view(state.reputation, dtype=np.float64),
                        neighbors=teacher_neighbors,
                        revision_mask=np.ones(self.agent_count, dtype=bool),
                        rng=self.reputation_rng,
                        config=config,
                    )
                (
                    bootstrapped_probabilities,
                    decision_bootstrap_diagnostics,
                ) = blend_domain_decision_bootstrap_probabilities(
                    candidate_probs[:, 1].detach().cpu().numpy(),
                    teacher_probabilities=teacher_probs,
                    bootstrap=config.policy.domain.bootstrap,
                    epoch=context.epoch,
                )
                if decision_bootstrap_diagnostics is not None:
                    bootstrapped_cooperation = torch.as_tensor(
                        bootstrapped_probabilities,
                        dtype=candidate_probs.dtype,
                        device=candidate_probs.device,
                    )
                    candidate_probs = torch.stack(
                        (1.0 - bootstrapped_cooperation, bootstrapped_cooperation),
                        dim=1,
                    )
            return self.apply_precommitment_decision_feedback(state, candidate_probs)

        def sample_policy_actions(action_probs: torch.Tensor) -> StateArray:
            nonlocal actions, payoffs
            if self._uses_torch_state():
                selected_actions = decision_kernel.select_tensor_from_action_probs(
                    current_actions=state.actions,
                    action_probs=action_probs,
                    revision_mask=context.revision_mask,
                )
            else:
                selected_actions = decision_kernel.select_from_action_probs(
                    current_actions=state.actions,
                    action_probs=action_probs,
                    revision_mask=context.revision_mask,
                )
            payoffs = self._payoffs_for_peer_context(
                actions=selected_actions,
                peer_ids=context.extras["current_interaction_neighbors"],
                peer_index=context.extras.get("current_interaction_peer_index"),
            )
            actions = selected_actions
            return selected_actions

        def actions_like_state(values: np.ndarray) -> StateArray:
            if isinstance(state.actions, torch.Tensor):
                return torch.as_tensor(
                    values,
                    dtype=state.actions.dtype,
                    device=state.actions.device,
                )
            return values

        def sample_revision_choices(
            revision_probs: torch.Tensor | np.ndarray,
            current_actions: np.ndarray,
        ) -> np.ndarray:
            nonlocal actions, payoffs
            probabilities = normalize_binary_revision_probabilities(revision_probs)
            probability_array = (
                probabilities.detach().cpu().numpy()
                if isinstance(probabilities, torch.Tensor)
                else np.asarray(probabilities, dtype=np.float64)
            )
            choices = np.full(len(current_actions), REVISION_STAY, dtype=np.int64)
            active_indices = np.flatnonzero(context.revision_mask)
            if decision_kernel.mode == "sampled":
                for agent_id in active_indices:
                    choices[int(agent_id)] = int(
                        self.rng.choice(3, p=probability_array[int(agent_id)])
                    )
            elif decision_kernel.mode == "argmax":
                choices[active_indices] = np.argmax(
                    probability_array[active_indices],
                    axis=1,
                )
            else:
                raise ValueError(
                    f"Unsupported decision mode: {decision_kernel.mode}"
                )
            selected_action_values = apply_binary_revision_choices(
                current_actions,
                choices,
            )
            selected_actions = actions_like_state(selected_action_values)
            payoffs = self._payoffs_for_peer_context(
                actions=selected_actions,
                peer_ids=context.extras["current_interaction_neighbors"],
                peer_index=context.extras.get("current_interaction_peer_index"),
            )
            actions = selected_actions
            return choices

        objective_components: StateContinuationComponents | None = None
        training_components: StateContinuationComponents | None = None
        basin_diagnostics = None
        bootstrap_diagnostics = None
        distill_diagnostics = None
        distill_teacher_probs = np.asarray([], dtype=np.float64)
        distill_teacher_post_action_probs = np.asarray([], dtype=np.float64)
        distill_weight = 0.0
        alignment_base_losses = np.asarray([], dtype=np.float64)
        alignment_distill_losses = np.asarray([], dtype=np.float64)
        alignment_base_grad_norms = np.asarray([], dtype=np.float64)
        alignment_distill_grad_norms = np.asarray([], dtype=np.float64)
        alignment_grad_cosines = np.asarray([], dtype=np.float64)
        alignment_distill_candidate_mask = np.asarray([], dtype=bool)
        alignment_distill_stable_teacher_mask = np.asarray([], dtype=bool)
        alignment_distill_gradient_gate_mask = np.asarray([], dtype=bool)
        alignment_distill_applied_mask = np.asarray([], dtype=bool)

        def commit_local_update(updated_actions: StateArray) -> LossVector:
            nonlocal actions, distill_diagnostics, distill_teacher_probs
            nonlocal distill_teacher_post_action_probs, distill_weight
            nonlocal objective_components, training_components
            nonlocal bootstrap_diagnostics
            nonlocal alignment_base_losses, alignment_distill_losses
            nonlocal alignment_base_grad_norms, alignment_distill_grad_norms
            nonlocal alignment_grad_cosines, alignment_distill_candidate_mask
            nonlocal alignment_distill_stable_teacher_mask
            nonlocal alignment_distill_gradient_gate_mask
            nonlocal alignment_distill_applied_mask
            actions = updated_actions
            if config.policy.domain.bootstrap.distill_enabled:
                if decision_probs is None:
                    raise RuntimeError("Toy 2 decision policy readout is missing")
                distill_weight = domain_distill_bootstrap_weight(
                    config.policy.domain.bootstrap,
                    context.epoch,
                )
                if distill_weight > 0.0:
                    distill_teacher_neighbors = reputation_peer_ids(
                        config=config,
                        neighbors=self.neighbors,
                        agent_count=self.agent_count,
                        rng=self.reputation_rng,
                    )
                    distill_teacher_probs = reputation_imitation_cooperation_probs(
                        actions=to_numpy_view(state.actions, dtype=np.int64),
                        reputation=to_numpy_view(state.reputation, dtype=np.float64),
                        neighbors=distill_teacher_neighbors,
                        revision_mask=np.ones(self.agent_count, dtype=bool),
                        rng=self.reputation_rng,
                        config=config,
                    )
                    if config.policy.domain.bootstrap.distill_stable_teacher_only:
                        distill_teacher_post_action_probs = (
                            reputation_imitation_cooperation_probs(
                                actions=to_numpy_view(actions, dtype=np.int64),
                                reputation=to_numpy_view(
                                    state.reputation,
                                    dtype=np.float64,
                                ),
                                neighbors=distill_teacher_neighbors,
                                revision_mask=np.ones(self.agent_count, dtype=bool),
                                rng=self.reputation_rng,
                                config=config,
                            )
                        )
                distill_diagnostics = domain_distill_bootstrap_diagnostic_components(
                    neural_probabilities=decision_probs[:, 1].detach().cpu().numpy(),
                    teacher_probabilities=distill_teacher_probs,
                    realized_actions=to_numpy_view(actions, dtype=np.int64),
                    bootstrap=config.policy.domain.bootstrap,
                    epoch=context.epoch,
                )
            local_losses = [0.0 for _ in range(self.agent_count)]
            if config.policy.learning_enabled:
                if self.neural_update_backend == "tensor_batched":
                    update_result = train_neural_local_policies_tensor_batched_update(
                        runtime=self._require_tensor_runtime(agents),
                        agents=agents,
                        observations=observations,
                        actions=actions,
                        payoffs=payoffs,
                        peer_ids=context_neighbors,
                        revision_mask=context.revision_mask,
                        config=config,
                        timing_context=context,
                        peer_index=context_peer_index,
                        payoff_baseline=state.payoff_ema,
                    )
                    local_losses = update_result.losses
                elif self.neural_update_backend == "batched":
                    with timed_context_stage(
                        context,
                        "local_trainable_parameters",
                    ):
                        update_parameters = (
                            self._require_policy_cache(agents)
                            .parameters
                            .trainable_clone()
                        )
                    with timed_context_stage(context, "local_loss_update"):
                        update_result = train_neural_local_policies_batched_update(
                            agents=agents,
                            observations=observations,
                            actions=actions,
                            payoffs=payoffs,
                            peer_ids=context_neighbors,
                            revision_mask=context.revision_mask,
                            config=config,
                            parameters=update_parameters,
                            adam_state_cache=self._require_adam_state_cache(agents),
                            timing_context=context,
                            peer_index=context_peer_index,
                        )
                    local_losses = update_result.losses
                    self._pending_policy_cache_parameters = (
                        update_result.updated_parameters
                    )
                    if not update_result.used_batched_optimizer:
                        self._adam_state_cache = None
                else:
                    if (
                        config.policy.domain.local_update_rule
                        == "counterfactual_advantage"
                    ):
                        objective_components = counterfactual_policy_advantage_components(
                            actions=actions,
                            peer_ids=context_neighbors,
                            config=config,
                            peer_index=context_peer_index,
                        )
                        training_components = objective_components
                        if config.policy.domain.bootstrap.enabled:
                            scheduled_weight = domain_bootstrap_weight(
                                config.policy.domain.bootstrap,
                                context.epoch,
                            )
                            if scheduled_weight > 0.0:
                                teacher_neighbors = reputation_peer_ids(
                                    config=config,
                                    neighbors=self.neighbors,
                                    agent_count=self.agent_count,
                                    rng=self.reputation_rng,
                                )
                                teacher_probs = reputation_imitation_cooperation_probs(
                                    actions=actions,
                                    reputation=state.reputation,
                                    neighbors=teacher_neighbors,
                                    revision_mask=np.ones(
                                        self.agent_count,
                                        dtype=bool,
                                    ),
                                    rng=self.reputation_rng,
                                    config=config,
                                )
                                (
                                    training_components,
                                    bootstrap_diagnostics,
                                ) = blend_domain_bootstrap_components(
                                    objective_components,
                                    teacher_probabilities=teacher_probs,
                                    bootstrap=config.policy.domain.bootstrap,
                                    epoch=context.epoch,
                                )
                            else:
                                (
                                    training_components,
                                    bootstrap_diagnostics,
                                ) = blend_domain_bootstrap_components(
                                    objective_components,
                                    teacher_probabilities=np.asarray(
                                        [],
                                        dtype=np.float64,
                                    ),
                                    bootstrap=config.policy.domain.bootstrap,
                                    epoch=context.epoch,
                                )
                        if (
                            config.policy.domain.basin_credit.enabled
                            and not basin_credit_preserves_objective(
                                config.policy.domain.basin_credit
                            )
                        ):
                            local_update_agent_ids = np.flatnonzero(
                                context.revision_mask
                            )
                        else:
                            local_update_agent_ids = range(self.agent_count)
                    else:
                        local_update_agent_ids = np.flatnonzero(context.revision_mask)
                    alignment_base_losses = np.full(
                        self.agent_count,
                        np.nan,
                        dtype=np.float64,
                    )
                    alignment_distill_losses = np.full(
                        self.agent_count,
                        np.nan,
                        dtype=np.float64,
                    )
                    alignment_base_grad_norms = np.full(
                        self.agent_count,
                        np.nan,
                        dtype=np.float64,
                    )
                    alignment_distill_grad_norms = np.full(
                        self.agent_count,
                        np.nan,
                        dtype=np.float64,
                    )
                    alignment_grad_cosines = np.full(
                        self.agent_count,
                        np.nan,
                        dtype=np.float64,
                    )
                    alignment_distill_candidate_mask = np.zeros(
                        self.agent_count,
                        dtype=bool,
                    )
                    alignment_distill_stable_teacher_mask = np.zeros(
                        self.agent_count,
                        dtype=bool,
                    )
                    alignment_distill_gradient_gate_mask = np.zeros(
                        self.agent_count,
                        dtype=bool,
                    )
                    alignment_distill_applied_mask = np.zeros(
                        self.agent_count,
                        dtype=bool,
                    )
                    stable_teacher_mask = np.ones(self.agent_count, dtype=bool)
                    if (
                        config.policy.domain.bootstrap.distill_stable_teacher_only
                        and distill_teacher_probs.size > 0
                        and distill_teacher_post_action_probs.size > 0
                    ):
                        stable_teacher_mask = stable_teacher_probability_mask(
                            distill_teacher_probs,
                            distill_teacher_post_action_probs,
                            margin_min=(
                                config.policy.domain.bootstrap.distill_teacher_margin_min
                            ),
                        )
                    for agent_id in local_update_agent_ids:
                        signed_advantage = (
                            None
                            if training_components is None
                            else float(training_components.effective[int(agent_id)])
                        )
                        use_distill = (
                            distill_weight > 0.0
                            and distill_teacher_probs.size > 0
                            and (
                                config.policy.domain.bootstrap.distill_scope == "all"
                                or bool(context.revision_mask[int(agent_id)])
                            )
                        )
                        if use_distill:
                            alignment_distill_candidate_mask[int(agent_id)] = True
                        stable_teacher = bool(stable_teacher_mask[int(agent_id)])
                        if use_distill and stable_teacher:
                            alignment_distill_stable_teacher_mask[int(agent_id)] = True
                            (
                                alignment_base_losses[int(agent_id)],
                                alignment_distill_losses[int(agent_id)],
                                alignment_base_grad_norms[int(agent_id)],
                                alignment_distill_grad_norms[int(agent_id)],
                                alignment_grad_cosines[int(agent_id)],
                            ) = teacher_distill_gradient_conflict_values(
                                agent=agents[int(agent_id)],
                                observation=observations[int(agent_id)],
                                peer_actions=actions[context_neighbors[int(agent_id)]],
                                config=config,
                                signed_advantage=signed_advantage,
                                teacher_probability=float(
                                    distill_teacher_probs[int(agent_id)]
                                ),
                            )
                            gradient_pass = (
                                not config.policy.domain.bootstrap.distill_gradient_gate_enabled
                            )
                            if config.policy.domain.bootstrap.distill_gradient_gate_enabled:
                                gradient_pass = bool(
                                    gradient_gate_mask(
                                        np.asarray(
                                            [alignment_grad_cosines[int(agent_id)]],
                                            dtype=np.float64,
                                        ),
                                        min_cosine=(
                                            config.policy.domain.bootstrap.distill_gradient_min_cosine
                                        ),
                                    )[0]
                                )
                            if gradient_pass:
                                alignment_distill_gradient_gate_mask[
                                    int(agent_id)
                                ] = True
                            use_distill = gradient_pass
                        else:
                            use_distill = False
                        if use_distill:
                            alignment_distill_applied_mask[int(agent_id)] = True
                        local_losses[int(agent_id)] = train_neural_local_policy(
                            agent=agents[int(agent_id)],
                            observation=observations[int(agent_id)],
                            action=int(actions[int(agent_id)]),
                            payoff=float(payoffs[int(agent_id)]),
                            peer_actions=actions[context_neighbors[int(agent_id)]],
                            config=config,
                            signed_counterfactual_advantage=signed_advantage,
                            teacher_distill_probability=(
                                float(distill_teacher_probs[int(agent_id)])
                                if use_distill
                                else None
                            ),
                            teacher_distill_weight=distill_weight if use_distill else 0.0,
                            base_loss_weight=(
                                0.0
                                if (
                                    config.policy.domain.basin_credit.enabled
                                    and not basin_credit_preserves_objective(
                                        config.policy.domain.basin_credit
                                    )
                                )
                                else 1.0
                            ),
                        )
            return local_losses

        revision_operator_aggregate: dict[str, object] | None = None
        revision_operator_micro: list[dict[str, object]] | None = None
        if config.coordination.revision_operator_enabled:
            with timed_context_stage(context, "policy_readout"):
                pre_revision_probs = collect_pre_policy_probs(
                    agents,
                    observations,
                    temperature=config.policy.temperature,
                )
            revision_decision_action_probs: torch.Tensor | None = None

            def collect_revision_signals(
                agents_arg: list[NeuralPDAgent],
                observations_arg: torch.Tensor,
                current_actions_arg: np.ndarray,
            ) -> dict[str, object]:
                del agents_arg, observations_arg, current_actions_arg
                return {
                    "source": config.coordination.revision_operator_source,
                    "revision_mask": context.revision_mask.copy(),
                }

            def collect_revision_probs(
                agents_arg: list[NeuralPDAgent],
                observations_arg: torch.Tensor,
                current_actions_arg: np.ndarray,
                revision_signals: dict[str, object],
                *,
                temperature: float,
            ) -> torch.Tensor:
                del agents_arg, observations_arg, revision_signals, temperature
                nonlocal revision_decision_action_probs
                revision_decision_action_probs = build_decision_action_probs(
                    pre_revision_probs
                )
                return binary_revision_probabilities_from_action_probs(
                    revision_decision_action_probs,
                    current_actions_arg,
                )

            def collect_post_revision_probs(
                agents_arg: list[NeuralPDAgent],
                observations_arg: torch.Tensor,
                current_actions_arg: np.ndarray,
                revision_signals: dict[str, object],
                *,
                temperature: float,
            ) -> torch.Tensor:
                del agents_arg, observations_arg, revision_signals, temperature
                if revision_decision_action_probs is None:
                    raise RuntimeError("Toy 2 revision decision probabilities missing")
                return binary_revision_probabilities_from_action_probs(
                    revision_decision_action_probs,
                    current_actions_arg,
                )

            revision_result = BinaryRevisionLearningUnit(
                agents=agents,
                observations=observations,
                current_actions=state.actions,
                temperature=config.policy.temperature,
                callbacks=BinaryRevisionLearningCallbacks(
                    collect_revision_signals=collect_revision_signals,
                    collect_revision_probs=collect_revision_probs,
                    sample_revision_choices=sample_revision_choices,
                    local_update=lambda selected_actions: commit_local_update(
                        actions_like_state(selected_actions)
                    ),
                    refresh_revision_cache=self.refresh_policy_cache,
                    post_collect_revision_probs=collect_post_revision_probs,
                ),
            ).run()
            if revision_decision_action_probs is None:
                raise RuntimeError("Toy 2 revision decision probabilities missing")
            decision_action_probs = revision_decision_action_probs
            actions = actions_like_state(revision_result.actions_after_revision)
            local_losses = revision_result.local_losses
            with timed_context_stage(context, "post_local_readout"):
                post_local_probs = collect_post_policy_probs(
                    agents,
                    observations,
                    temperature=config.policy.temperature,
                )
            revision_operator_aggregate = {
                **revision_result.aggregate_row(),
                "revision_operator_enabled": True,
                "revision_operator_source": (
                    config.coordination.revision_operator_source
                ),
            }
            revision_operator_micro = [
                {
                    **row,
                    "revision_operator_enabled": True,
                    "revision_operator_source": (
                        config.coordination.revision_operator_source
                    ),
                }
                for row in revision_result.micro_rows()
            ]
        else:
            learning_result = BinaryPolicyLearningUnit(
                agents=agents,
                observations=observations,
                temperature=config.policy.temperature,
                callbacks=BinaryPolicyLearningCallbacks(
                    collect_policy_probs=collect_pre_policy_probs,
                    decision_action_probs=build_decision_action_probs,
                    sample_actions=sample_policy_actions,
                    local_update=commit_local_update,
                    refresh_policy_cache=self.refresh_policy_cache,
                    post_collect_policy_probs=collect_post_policy_probs,
                ),
                context=context,
            ).run()
            pre_revision_probs = learning_result.pre_revision_probs
            decision_action_probs = learning_result.decision_action_probs
            actions = learning_result.actions_after_revision
            local_losses = learning_result.local_losses
            post_local_probs = learning_result.post_local_probs
        if decision_probs is None:
            raise RuntimeError("Toy 2 decision policy readout is missing")
        revised_local_losses = loss_values_at(
            local_losses,
            np.flatnonzero(context.revision_mask),
        )
        context.extras["context_neighbors"] = context_neighbors
        if context_peer_index is not None:
            context.extras["context_peer_index"] = context_peer_index
        return BinaryLocalStepResult(
            pre_revision_probs=pre_revision_probs,
            candidate_action_probs=decision_action_probs[:, 1].detach().cpu().numpy(),
            post_local_probs=post_local_probs,
            local_losses=local_losses,
            social_mode="policy_distill",
            actions_after_revision=actions,
            extras={
                "decision_action_probs": decision_action_probs,
                "revised_local_losses": revised_local_losses,
                "logged_neighbors": context_neighbors,
                "state_continuation_components": objective_components,
                "basin_credit_diagnostics": basin_diagnostics,
                "domain_bootstrap_diagnostics": bootstrap_diagnostics,
                "domain_distill_bootstrap_diagnostics": distill_diagnostics,
                "domain_decision_bootstrap_diagnostics": (
                    decision_bootstrap_diagnostics
                ),
                **(
                    {"revision_operator_aggregate": revision_operator_aggregate}
                    if revision_operator_aggregate is not None
                    else {}
                ),
                **(
                    {"revision_operator_micro": revision_operator_micro}
                    if revision_operator_micro is not None
                    else {}
                ),
                "_domain_teacher_alignment_inputs": {
                    "teacher_pre_action": alignment_teacher_pre_probs,
                    "teacher_neighbors": alignment_teacher_neighbors,
                    "pre_local": decision_probs[:, 1].detach().cpu().numpy(),
                    "post_local": binary_action_probs_from_policy(post_local_probs),
                    "objective_effective": (
                        np.asarray([], dtype=np.float64)
                        if objective_components is None
                        else objective_components.effective
                    ),
                    "base_losses": alignment_base_losses,
                    "distill_losses": alignment_distill_losses,
                    "base_grad_norms": alignment_base_grad_norms,
                    "distill_grad_norms": alignment_distill_grad_norms,
                    "grad_cosines": alignment_grad_cosines,
                    "distill_candidate_mask": alignment_distill_candidate_mask,
                    "distill_stable_teacher_mask": (
                        alignment_distill_stable_teacher_mask
                    ),
                    "distill_gradient_gate_mask": (
                        alignment_distill_gradient_gate_mask
                    ),
                    "distill_applied_mask": alignment_distill_applied_mask,
                },
                "_observations": observations,
                "_payoffs_after_actions": payoffs,
            },
        )

    def _local_step_classical(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
    ) -> BinaryLocalStepResult:
        config = self.config
        pre_revision_probs = cooperation_probs_to_policy_tensor(
            state.actions.astype(np.float64),
            device=self.device,
        )
        payoffs = self._payoffs_for_peer_context(
            actions=state.actions,
            peer_ids=context.extras["current_interaction_neighbors"],
            peer_index=context.extras.get("current_interaction_peer_index"),
        )
        if config.policy.rule == "fermi_imitation":
            base_coop_probs = fermi_imitation_cooperation_probs(
                actions=state.actions,
                payoffs=payoffs,
                neighbors=context.extras["current_interaction_neighbors"],
                revision_mask=context.revision_mask,
                rng=self.rng,
                config=config,
            )
        elif config.policy.rule == "reputation_imitation":
            reputation_neighbors = reputation_peer_ids(
                config=config,
                neighbors=self.neighbors,
                agent_count=self.agent_count,
                rng=self.reputation_rng,
            )
            base_coop_probs = reputation_imitation_cooperation_probs(
                actions=state.actions,
                reputation=state.reputation,
                neighbors=reputation_neighbors,
                revision_mask=context.revision_mask,
                rng=self.reputation_rng,
                config=config,
            )
        else:
            raise ValueError(
                f"Unsupported Toy 2 update rule: {config.policy.rule}"
            )
        base_probs = cooperation_probs_to_policy_tensor(
            base_coop_probs,
            device=self.device,
        )
        return BinaryLocalStepResult(
            pre_revision_probs=pre_revision_probs,
            candidate_action_probs=base_coop_probs,
            post_local_probs=base_probs,
            local_losses=[0.0 for _ in range(self.agent_count)],
            social_mode="probability_mix",
            extras={
                "decision_action_probs": base_probs,
                "revised_local_losses": [
                    0.0 for _ in np.flatnonzero(context.revision_mask)
                ],
                "logged_neighbors": self.neighbors,
            },
        )

    def select_peers(
        self,
        action_probs: np.ndarray,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> list[list[int]]:
        del state
        if self.config.coordination.mixer == "none":
            return [[] for _ in range(self.agent_count)]
        if (
            local_result.social_mode == "policy_distill"
            and self.config.coordination.peer_rule == "none"
        ):
            return self.neighbors
        if local_result.social_mode == "policy_distill":
            policy_probs = local_result.post_local_probs
            neighbors = self.neighbors
        else:
            policy_probs = cooperation_probs_to_policy_tensor(
                action_probs,
                device=self.device,
            )
            neighbors = context.extras["current_interaction_neighbors"]
        peer_ids, _ = select_peers(
            neighbors=neighbors,
            peer_rule=self.config.coordination.peer_rule,
            threshold=self.config.coordination.threshold,
            policy_probs=policy_probs,
        )
        if self.config.coordination.mixer == "output_average":
            if local_result.social_mode == "probability_mix":
                return restrict_peer_ids_to_revised(peer_ids, context.revision_mask)
            return peer_ids
        raise ValueError(f"Unsupported Toy 2 mixer: {self.config.coordination.mixer}")

    def policy_tensor_from_action_probs(
        self,
        action_probs: np.ndarray,
        device_like: torch.Tensor,
    ) -> torch.Tensor:
        return cooperation_probs_to_policy_tensor(action_probs, device=device_like.device)

    def sample_actions(
        self,
        state: BinarySpatialState,
        action_probs: np.ndarray,
        revision_mask: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> np.ndarray:
        del context, local_result
        return sample_revised_cooperation_probs(
            current_actions=state.actions,
            cooperation_probs=action_probs,
            revision_mask=revision_mask,
            rng=self.rng,
        )

    def collect_policy_probs(
        self,
        agents: list[NeuralPDAgent],
        observations: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        if self.neural_update_backend == "tensor_batched":
            return self._require_tensor_runtime(agents).probabilities(
                observations,
                temperature=temperature,
            )
        return self._require_policy_cache(agents).probabilities(
            observations,
            temperature=temperature,
        )

    def collect_policy_logits(
        self,
        agents: list[NeuralPDAgent],
        observations: torch.Tensor,
    ) -> torch.Tensor:
        if self.neural_update_backend == "tensor_batched":
            return self._require_tensor_runtime(agents).logits(observations)
        return self._require_policy_cache(agents).logits(observations)

    def refresh_policy_cache(self, agents: list[NeuralPDAgent]) -> None:
        if self.neural_update_backend == "tensor_batched":
            if self.tensor_runtime is None and agents:
                self.tensor_runtime = TensorBatchedMLPRuntime.from_agents(
                    agents,
                    device=self.device,
                )
            self._pending_policy_cache_parameters = None
            return
        if self._pending_policy_cache_parameters is not None:
            if self.policy_cache is None:
                self.policy_cache = BatchedMLPPolicyCache(
                    self._pending_policy_cache_parameters,
                )
            else:
                self.policy_cache.parameters = self._pending_policy_cache_parameters
            self._pending_policy_cache_parameters = None
            return
        if self.policy_cache is None:
            self.policy_cache = BatchedMLPPolicyCache.from_agents(
                agents,
                device=self.device,
            )
            return
        self.policy_cache.refresh(agents)

    def _require_policy_cache(
        self,
        agents: list[NeuralPDAgent],
    ) -> BatchedMLPPolicyCache:
        if self.policy_cache is None:
            self.refresh_policy_cache(agents)
        if self.policy_cache is None:
            raise RuntimeError("Toy 2 policy cache is not initialized")
        return self.policy_cache

    def _require_tensor_runtime(
        self,
        agents: list[NeuralPDAgent],
    ) -> TensorBatchedMLPRuntime:
        if self.tensor_runtime is None:
            self.tensor_runtime = TensorBatchedMLPRuntime.from_agents(
                agents,
                device=self.device,
            )
        return self.tensor_runtime

    def _require_adam_state_cache(
        self,
        agents: list[NeuralPDAgent],
    ) -> BatchedAdamStateCache:
        if self._adam_state_cache is None:
            self._adam_state_cache = BatchedAdamStateCache.from_agents(
                agents,
                device=self.device,
            )
        return self._adam_state_cache

    def flush_tensor_runtime_to_agents(
        self,
        agents: list[NeuralPDAgent],
    ) -> None:
        if self.tensor_runtime is not None and agents:
            self.tensor_runtime.flush_to_agents(agents)

    def write_summary(
        self,
        run_dir: Path,
        final_row: dict[str, object],
        state: BinarySpatialState,
    ) -> BinaryToyResult:
        agents = list(state.agents or [])
        if (
            self.config.policy.rule == "neural_policy"
            and self._uses_torch_state()
            and agents
        ):
            self._sync_agent_payoff_ema(state)
        self.flush_tensor_runtime_to_agents(agents)
        if self._basin_transition_samples:
            write_basin_transition_samples(
                run_dir,
                annotate_terminal_outcomes(
                    self._basin_transition_samples,
                    final_mean_payoff=float(final_row["mean_payoff"]),
                    target_payoff=toy2_target_basin_payoff(self.config),
                ),
            )
        return super().write_summary(run_dir, final_row, state)

    def distill_policy(
        self,
        agents: list[NeuralPDAgent],
        observations: torch.Tensor,
        peer_ids: list[list[int]],
        alpha: float,
        previous_probs: torch.Tensor,
        context: BinaryStepContext | None = None,
        confidence_weighting: str = "none",
        confidence_weight_floor: float = 0.0,
        confidence_weight_power: float = 1.0,
        social_direction_scores: np.ndarray | None = None,
        precommitment_readiness: np.ndarray | None = None,
        precommitment_readiness_weight: float = 0.0,
    ) -> LossVector | BinaryOutputDistillationReport:
        if self.neural_update_backend == "tensor_batched":
            adapter = TensorRuntimeDistributionDistillationAdapter(
                runtime=self._require_tensor_runtime(agents),
                observations=observations,
                loss_mode="kl",
                timing_context=context,
            )
            return run_binary_output_distribution_distillation(
                agents=agents,
                observations=observations,
                peer_ids=peer_ids,
                alpha=alpha,
                previous_probs=previous_probs,
                commit_adapter=adapter,
                confidence_weighting=confidence_weighting,
                confidence_weight_floor=confidence_weight_floor,
                confidence_weight_power=confidence_weight_power,
                social_direction_scores=social_direction_scores,
                precommitment_readiness=precommitment_readiness,
                precommitment_readiness_weight=precommitment_readiness_weight,
            )
        if self.neural_update_backend == "batched":
            if context is None:
                update_parameters = (
                    self._require_policy_cache(agents)
                    .parameters
                    .trainable_clone()
                )
            else:
                with timed_context_stage(context, "social_trainable_parameters"):
                    update_parameters = (
                        self._require_policy_cache(agents)
                        .parameters
                        .trainable_clone()
                    )
            adapter = BatchedDistributionDistillationAdapter(
                agents=agents,
                observations=observations,
                parameters=update_parameters,
                adam_state_cache=self._require_adam_state_cache(agents),
                loss_mode="kl",
                timing_context=context,
            )
            report = run_binary_output_distribution_distillation(
                agents=agents,
                observations=observations,
                peer_ids=peer_ids,
                alpha=alpha,
                previous_probs=previous_probs,
                commit_adapter=adapter,
                confidence_weighting=confidence_weighting,
                confidence_weight_floor=confidence_weight_floor,
                confidence_weight_power=confidence_weight_power,
                social_direction_scores=social_direction_scores,
                precommitment_readiness=precommitment_readiness,
                precommitment_readiness_weight=precommitment_readiness_weight,
            )
            if adapter.update_result is None:
                raise RuntimeError("batched distillation adapter did not produce a result")
            update_result = adapter.update_result
            self._pending_policy_cache_parameters = update_result.updated_parameters
            if not update_result.used_batched_optimizer:
                self._adam_state_cache = None
            return report
        del context
        return run_binary_output_distribution_distillation(
            agents=agents,
            observations=observations,
            peer_ids=peer_ids,
            alpha=alpha,
            previous_probs=previous_probs,
            logits_fn=lambda agent, agent_id, observed: agent.model(
                observed[agent_id].unsqueeze(0)
            ),
            loss_mode="kl",
            confidence_weighting=confidence_weighting,
            confidence_weight_floor=confidence_weight_floor,
            confidence_weight_power=confidence_weight_power,
            social_direction_scores=social_direction_scores,
            precommitment_readiness=precommitment_readiness,
            precommitment_readiness_weight=precommitment_readiness_weight,
        )

    def commit_actions(
        self,
        state: BinarySpatialState,
        actions: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ) -> dict[str, object]:
        del social_result
        local_actions = local_result.actions_after_revision
        actions_changed_after_local = (
            local_actions is not None
            and not np.array_equal(
                to_numpy_view(actions, dtype=np.int64),
                to_numpy_view(local_actions, dtype=np.int64),
            )
        )
        if (
            local_result.social_mode == "policy_distill"
            and not actions_changed_after_local
        ):
            payoffs = local_result.extras["_payoffs_after_actions"]
        else:
            payoff_interaction_neighbors = (
                context.extras["current_interaction_neighbors"]
                if local_result.social_mode == "policy_distill"
                else interaction_peer_ids(
                    config=self.config,
                    neighbors=self.neighbors,
                    agent_count=self.agent_count,
                    rng=self.interaction_rng,
                )
            )
            if local_result.social_mode != "policy_distill":
                context.extras["payoff_interaction_neighbors"] = (
                    payoff_interaction_neighbors
                )
            payoff_interaction_peer_index = (
                context.extras.get("current_interaction_peer_index")
                if local_result.social_mode == "policy_distill"
                else self._spatial_interaction_peer_index()
            )
            if payoff_interaction_peer_index is not None:
                context.extras["payoff_interaction_peer_index"] = (
                    payoff_interaction_peer_index
                )
            payoffs = self._payoffs_for_peer_context(
                actions=actions,
                peer_ids=payoff_interaction_neighbors,
                peer_index=payoff_interaction_peer_index,
            )
        state.actions = actions
        state.payoffs = payoffs
        return {}

    def payoff_ema_decay(self) -> float | None:
        return self.config.environment.reward_ema_decay

    def mobility_params(self) -> MobilityParams | None:
        return mobility_params_from_config(self.config)

    def mobility_neighbors(self) -> list[list[int]] | None:
        return self.neighbors

    def mobility_random_generator(self) -> np.random.Generator | None:
        return self.mobility_rng

    def _tensor_runtime_mobility_arrays(self) -> dict[str, torch.Tensor]:
        runtime = self.tensor_runtime
        if runtime is None:
            return {}
        arrays: dict[str, torch.Tensor] = {}
        names = ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")
        for prefix, parameters in (
            ("parameters", runtime.parameters),
            ("exp_avg", runtime.exp_avg),
            ("exp_avg_sq", runtime.exp_avg_sq),
        ):
            for name, tensor in zip(names, parameters.tensors(), strict=True):
                arrays[f"tensor_runtime_{prefix}_{name}"] = tensor
        for index, tensor in enumerate(runtime.steps):
            arrays[f"tensor_runtime_step_{index}"] = tensor
        return arrays

    def post_step_state_update(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
    ):
        policy = super().post_step_state_update(
            state=state,
            context=context,
            local_result=local_result,
            social_result=social_result,
        )
        if (
            self._uses_torch_state()
            and policy.mobility_params is not None
            and policy.mobility_params.enabled
            and policy.mobility_params.rate > 0.0
        ):
            policy.mobility_extra_state_arrays = self._tensor_runtime_mobility_arrays()
        return policy

    def finalize_hook_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
    ) -> dict[str, object]:
        updates: dict[str, object] = {"extras": {}}
        if self.config.policy.rule == "neural_policy" and not self._uses_torch_state():
            self._sync_agent_payoff_ema(state)
        if bool(np.any(mobility_result.moved)):
            payoff_neighbors = (
                context.extras["current_interaction_neighbors"]
                if self.config.policy.rule == "neural_policy"
                else context.extras["payoff_interaction_neighbors"]
            )
            payoff_peer_index = (
                context.extras.get("current_interaction_peer_index")
                if self.config.policy.rule == "neural_policy"
                else context.extras.get("payoff_interaction_peer_index")
            )
            payoffs = self._payoffs_for_peer_context(
                actions=state.actions,
                peer_ids=payoff_neighbors,
                peer_index=payoff_peer_index,
            )
            state.payoffs = payoffs
            if self.config.policy.rule == "neural_policy":
                observations = build_observations(
                    actions=state.actions,
                    payoffs=payoffs,
                    agents=state.agents or [],
                    neighbors=context.extras["context_neighbors"],
                    payoff_normalizer=payoff_scale(self.config),
                    device=self.device,
                    reputation=state.reputation,
                    reputation_observation_mode=(
                        self.config.state.reputation.observation_mode
                    ),
                    peer_index=context.extras.get("context_peer_index"),
                    payoff_ema=state.payoff_ema,
                    previous_payoff_ema=state.previous_payoff_ema,
                )
                self.refresh_policy_cache(list(state.agents or []))
                updates["post_social_probs"] = self.collect_policy_probs(
                    state.agents or [],
                    observations,
                    temperature=self.config.policy.temperature,
                )
        alignment_inputs = local_result.extras.get("_domain_teacher_alignment_inputs")
        if isinstance(alignment_inputs, dict):
            diagnostic_reputation_rng = np.random.default_rng(
                int(self.config.run.seed) + 1_000_003 * int(context.epoch) + 19
            )
            teacher_neighbors = alignment_inputs.get("teacher_neighbors")
            if teacher_neighbors is None:
                teacher_neighbors = reputation_peer_ids(
                    config=self.config,
                    neighbors=self.neighbors,
                    agent_count=self.agent_count,
                    rng=diagnostic_reputation_rng,
                )
            teacher_post_action = reputation_imitation_cooperation_probs(
                actions=to_numpy_view(state.actions, dtype=np.int64),
                reputation=to_numpy_view(state.reputation, dtype=np.float64),
                neighbors=teacher_neighbors,
                revision_mask=np.ones(self.agent_count, dtype=bool),
                rng=diagnostic_reputation_rng,
                config=self.config,
            )
            post_social_probs = updates.get(
                "post_social_probs",
                social_result.post_social_probs,
            )
            bootstrap = self.config.policy.domain.bootstrap
            if bootstrap.replay_enabled:
                replay_weight = domain_decision_replay_weight(
                    bootstrap,
                    context.epoch,
                )
                teacher_pre_action = np.asarray(
                    alignment_inputs["teacher_pre_action"],
                    dtype=np.float64,
                )
                objective_effective = np.asarray(
                    alignment_inputs["objective_effective"],
                    dtype=np.float64,
                )
                stable_mask: np.ndarray | None = None
                if bootstrap.replay_stable_teacher_only:
                    stable_mask = stable_teacher_probability_mask(
                        teacher_pre_action,
                        teacher_post_action,
                        margin_min=bootstrap.replay_teacher_margin_min,
                    )
                objective_agreement_mask: np.ndarray | None = None
                if bootstrap.replay_require_objective_agreement:
                    objective_agreement_mask = (
                        np.zeros_like(teacher_pre_action, dtype=bool)
                        if objective_effective.size == 0
                        else objective_teacher_sign_alignment(
                            objective_effective,
                            teacher_pre_action,
                        )[0].astype(bool)
                    )
                postsocial_improvement_mask: np.ndarray | None = None
                if bootstrap.replay_require_postsocial_alignment_improvement:
                    pre_bce = teacher_policy_bce(
                        np.asarray(alignment_inputs["pre_local"], dtype=np.float64),
                        teacher_pre_action,
                    )
                    post_social_action_probs = binary_action_probs_from_policy(
                        post_social_probs
                    )
                    post_bce = teacher_policy_bce(
                        post_social_action_probs,
                        teacher_pre_action,
                    )
                    postsocial_improvement_mask = post_bce <= pre_bce + 1e-12
                replay_diagnostics = build_domain_decision_replay_diagnostics(
                    weight=replay_weight,
                    teacher_probabilities=teacher_pre_action,
                    realized_actions=to_numpy_view(state.actions, dtype=np.int64),
                    revision_mask=context.revision_mask,
                    stable_teacher_mask=stable_mask,
                    objective_agreement_mask=objective_agreement_mask,
                    postsocial_improvement_mask=postsocial_improvement_mask,
                    teacher=bootstrap.replay_teacher,
                )
                updates["extras"]["domain_decision_replay_diagnostics"] = (
                    replay_diagnostics
                )
                observations = local_result.extras.get("_observations")
                agents = list(state.agents or [])
                if (
                    replay_weight > 0.0
                    and replay_diagnostics.applied_mask.size > 0
                    and observations is not None
                    and agents
                ):
                    context_neighbors = context.extras.get(
                        "context_neighbors",
                        self.neighbors,
                    )
                    actions = to_numpy_view(state.actions, dtype=np.int64)
                    payoffs = to_numpy_view(state.payoffs, dtype=np.float64)
                    for agent_id in np.flatnonzero(replay_diagnostics.applied_mask):
                        train_neural_local_policy(
                            agent=agents[int(agent_id)],
                            observation=observations[int(agent_id)],
                            action=int(actions[int(agent_id)]),
                            payoff=float(payoffs[int(agent_id)]),
                            peer_actions=actions[context_neighbors[int(agent_id)]],
                            config=self.config,
                            signed_counterfactual_advantage=0.0,
                            teacher_distill_probability=float(
                                replay_diagnostics.replay_actions[int(agent_id)]
                            ),
                            teacher_distill_weight=replay_weight,
                            base_loss_weight=0.0,
                        )
                    with timed_context_stage(context, "decision_replay_cache_refresh"):
                        self.refresh_policy_cache(agents)
                    with timed_context_stage(context, "decision_replay_readout"):
                        post_social_probs = self.collect_policy_probs(
                            agents,
                            observations,
                            temperature=self.config.policy.temperature,
                        )
                    updates["post_social_probs"] = post_social_probs
            updates["extras"]["domain_teacher_alignment_diagnostics"] = (
                build_domain_teacher_alignment_diagnostics(
                    teacher_pre_action=alignment_inputs["teacher_pre_action"],
                    teacher_post_action=teacher_post_action,
                    pre_local=alignment_inputs["pre_local"],
                    post_local=alignment_inputs["post_local"],
                    post_social=binary_action_probs_from_policy(post_social_probs),
                    realized_actions=to_numpy_view(state.actions, dtype=np.int64),
                    objective_effective=alignment_inputs["objective_effective"],
                    base_losses=alignment_inputs["base_losses"],
                    distill_losses=alignment_inputs["distill_losses"],
                    base_grad_norms=alignment_inputs["base_grad_norms"],
                    distill_grad_norms=alignment_inputs["distill_grad_norms"],
                    grad_cosines=alignment_inputs["grad_cosines"],
                    distill_candidate_mask=alignment_inputs[
                        "distill_candidate_mask"
                    ],
                    distill_stable_teacher_mask=alignment_inputs[
                        "distill_stable_teacher_mask"
                    ],
                    distill_gradient_gate_mask=alignment_inputs[
                        "distill_gradient_gate_mask"
                    ],
                    distill_applied_mask=alignment_inputs["distill_applied_mask"],
                    teacher="reputation_imitation",
                )
            )
        return updates

    def post_social_policy_update(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
        post_social_probs: Any,
    ) -> dict[str, object]:
        del social_result, mobility_result
        config = self.config
        basin_credit = config.policy.domain.basin_credit
        if config.policy.rule != "neural_policy" or not basin_credit.enabled:
            return {}

        actions = to_numpy_view(state.actions, dtype=np.int64)
        payoffs = to_numpy_view(state.payoffs, dtype=np.float64)
        action_probabilities = binary_action_probs_from_policy(post_social_probs)
        basin_training_mask = basin_credit_training_candidate_mask(
            agent_count=len(actions),
            revision_mask=context.revision_mask,
            training_scope=basin_credit.training_scope,
        )
        payoff_peer_ids = context.extras.get(
            "current_interaction_neighbors",
            local_result.extras.get("logged_neighbors", self.neighbors),
        )
        payoff_peer_index = context.extras.get("current_interaction_peer_index")
        basin_diagnostics = self.basin_credit_diagnostics_for_actions(
            actions=actions,
            payoffs=payoffs,
            peer_ids=payoff_peer_ids,
            revision_mask=basin_training_mask,
            action_probabilities=action_probabilities,
            peer_index=payoff_peer_index,
        )
        if basin_diagnostics is None:
            return {}
        effective_training_passes = basin_credit_effective_training_passes(
            basin_credit=basin_credit,
            diagnostics=basin_diagnostics,
        )
        basin_diagnostics = basin_credit_diagnostics_with_training_passes(
            basin_diagnostics,
            training_passes=effective_training_passes,
        )

        context_neighbors = context.extras.get(
            "context_neighbors",
            local_result.extras.get("logged_neighbors", self.neighbors),
        )
        context_peer_index = context.extras.get("context_peer_index")
        objective_components = counterfactual_policy_advantage_components(
            actions=actions,
            peer_ids=context_neighbors,
            config=config,
            peer_index=context_peer_index,
        )
        learned_diagnostics = self.learned_basin_diagnostics_for_actions(
            actions=actions,
            payoffs=payoffs,
            action_probabilities=action_probabilities,
            prototype_diagnostics=basin_diagnostics,
        )
        prototype_action1_advantage = selected_credit_to_action1_advantage(
            basin_diagnostics.selected_action_credit,
            actions,
        )
        learned_replay_min_selected_rate = (
            basin_credit_effective_learned_replay_min_selected_rate(
                basin_credit=basin_credit,
                epoch=context.epoch,
            )
        )
        learned_signal = (
            learned_basin_credit_signal(
                learned_diagnostics,
                prototype_action1_advantage=prototype_action1_advantage,
                fallback=basin_credit.learned_credit_fallback,
                replay_selection=basin_credit.learned_credit_replay_selection,
                replay_mode=basin_credit.learned_credit_replay_mode,
                replay_min_selected_rate=learned_replay_min_selected_rate,
                replay_floor_source=basin_credit.learned_credit_replay_floor_source,
                replay_soft_min_weight=(
                    basin_credit.learned_credit_replay_soft_min_weight
                ),
                replay_soft_disagreement_weight=(
                    basin_credit.learned_credit_replay_soft_disagreement_weight
                ),
                replay_weight_scorer=self.learned_basin_replay_weight_scorer(),
                eligible_mask=basin_training_mask,
            )
            if basin_credit.learned_credit_enabled
            else learned_basin_credit_signal(
                None,
                prototype_action1_advantage=prototype_action1_advantage,
                eligible_mask=basin_training_mask,
            )
        )
        training_components = blend_basin_credit_components(
            objective_components,
            diagnostics=basin_diagnostics,
            basin_credit=basin_credit,
            actions=actions,
            basin_action1_advantage=learned_signal.action1_advantage,
        )
        self._basin_transition_samples.extend(
            basin_transition_sample_rows(
                toy=self.toy,
                run_id=config.run.name,
                seed=config.run.seed,
                epoch=context.epoch,
                actions=actions,
                payoffs=payoffs,
                action_probabilities=action_probabilities,
                target_payoff=toy2_target_basin_payoff(config),
                diagnostics=basin_diagnostics,
                objective_components=objective_components,
                training_components=training_components,
                training_action1_advantage=learned_signal.action1_advantage,
                training_credit_source=learned_signal.source,
                training_replay_selection=learned_signal.replay_selection,
                training_replay_min_selected_rate=(
                    learned_signal.replay_min_selected_rate
                ),
                training_replay_mask=basin_training_mask & learned_signal.replay_mask,
                training_replay_weight=basin_training_mask * learned_signal.replay_weight,
                learned_credit_used_mask=learned_signal.learned_credit_used_mask,
                domain_fields={
                    "domain_game_family": config.game.family,
                },
            )
        )
        updates: dict[str, object] = {
            "extras": {
                "state_continuation_components": objective_components,
                "basin_credit_diagnostics": basin_diagnostics,
                "basin_credit_training_components": training_components,
                "basin_credit_training_action1_advantage": (
                    learned_signal.action1_advantage
                ),
                "basin_credit_training_credit_source": learned_signal.source,
                "basin_credit_training_replay_selection": (
                    learned_signal.replay_selection
                ),
                "basin_credit_training_replay_min_selected_rate": (
                    learned_signal.replay_min_selected_rate
                ),
                "basin_credit_training_replay_mask": (
                    basin_training_mask & learned_signal.replay_mask
                ),
                "basin_credit_training_replay_weight": (
                    basin_training_mask * learned_signal.replay_weight
                ),
                "basin_credit_training_learned_credit_used_mask": (
                    learned_signal.learned_credit_used_mask
                ),
                "basin_learned_diagnostics": learned_diagnostics,
            }
        }
        if (
            not config.policy.learning_enabled
            or basin_credit_preserves_objective(basin_credit)
        ):
            return updates

        agents = list(state.agents or [])
        if not agents:
            return updates

        with timed_context_stage(context, "post_social_basin_observations"):
            observations = build_observations(
                actions=state.actions,
                payoffs=state.payoffs,
                agents=agents,
                neighbors=context_neighbors,
                payoff_normalizer=payoff_scale(config),
                device=self.device,
                reputation=state.reputation,
                reputation_observation_mode=config.state.reputation.observation_mode,
                peer_index=context_peer_index,
                payoff_ema=state.payoff_ema,
                previous_payoff_ema=state.previous_payoff_ema,
            )
        with timed_context_stage(context, "post_social_basin_training"):
            replay_weight = basin_training_mask * learned_signal.replay_weight
            candidate_agent_ids = np.flatnonzero(replay_weight > 0.0)
            for _ in range(effective_training_passes):
                for agent_id in candidate_agent_ids:
                    train_neural_local_policy(
                        agent=agents[int(agent_id)],
                        observation=observations[int(agent_id)],
                        action=int(actions[int(agent_id)]),
                        payoff=float(payoffs[int(agent_id)]),
                        peer_actions=actions[context_neighbors[int(agent_id)]],
                        config=config,
                        signed_counterfactual_advantage=float(
                            training_components.effective[int(agent_id)]
                        ),
                        base_loss_weight=float(replay_weight[int(agent_id)]),
                    )
        with timed_context_stage(context, "post_social_basin_cache_refresh"):
            self.refresh_policy_cache(agents)
        with timed_context_stage(context, "post_social_basin_readout"):
            updates["post_social_probs"] = self.collect_policy_probs(
                agents,
                observations,
                temperature=config.policy.temperature,
            )
        return updates

    def _sync_agent_payoff_ema(self, state: BinarySpatialState) -> None:
        agents = state.agents or []
        if not agents:
            return
        for agent, payoff_ema, previous_payoff_ema in zip(
            agents,
            state.payoff_ema,
            state.previous_payoff_ema,
            strict=True,
        ):
            if isinstance(payoff_ema, torch.Tensor):
                agent.payoff_ema = float(payoff_ema.detach().cpu())
            else:
                agent.payoff_ema = float(payoff_ema)
            if isinstance(previous_payoff_ema, torch.Tensor):
                agent.previous_payoff_ema = float(previous_payoff_ema.detach().cpu())
            else:
                agent.previous_payoff_ema = float(previous_payoff_ema)

    def aggregate_payoffs(
        self,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> np.ndarray:
        return step_result.extras.get("aggregate_payoffs", state.payoffs)

    def domain_aggregate_fields(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, object]:
        del epoch
        cooperation_components, largest_coop_fraction = cooperation_cluster_metrics_from_edges(
            state.actions,
            self._graph_edges,
        )
        payoff = self.config.game.payoff
        return {
            "domain_game_family": self.config.game.family,
            "domain_payoff_T": payoff.T,
            "domain_payoff_R": payoff.R,
            "domain_payoff_P": payoff.P,
            "domain_payoff_S": payoff.S,
            "domain_policy_consensus": policy_consensus(step_result.post_social_probs),
            "domain_action_components": cooperation_components,
            "domain_largest_action_cluster_fraction": largest_coop_fraction,
            **domain_learning_aggregate_fields(
                extras=step_result.extras,
                actions=to_numpy_view(state.actions, dtype=np.int64),
            ),
        }

    def domain_micro_fields(
        self,
        agent_id: int,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, object]:
        del epoch
        logged_neighbors = step_result.extras.get("logged_neighbors", self.neighbors)
        actions = to_numpy_view(state.actions, dtype=np.float64)
        payoffs = to_numpy_view(state.payoffs, dtype=np.float64)
        neighbor_actions = actions[logged_neighbors[agent_id]]
        neighbor_payoffs = payoffs[logged_neighbors[agent_id]]
        return {
            "domain_game_family": self.config.game.family,
            "domain_neighbor_action_rate": float(neighbor_actions.mean()),
            "domain_neighbor_mean_payoff": float(neighbor_payoffs.mean()),
            **domain_learning_micro_fields(
                extras=step_result.extras,
                actions=to_numpy_view(state.actions, dtype=np.int64),
                agent_id=agent_id,
            ),
        }


def run_toy2(
    config: Toy2Config,
    config_path: Path,
    timing_rows: list[dict[str, object]] | None = None,
    neural_update_backend: NeuralUpdateBackendRequest | None = None,
) -> BinaryToyResult:
    """Run the Toy 2 simulation from a validated config."""

    if config.policy.rule == "rd_well_mixed":
        return run_toy2_rd_well_mixed(config=config, config_path=config_path)
    if config.policy.rule not in {
        "neural_policy",
        "fermi_imitation",
        "reputation_imitation",
    }:
        raise ValueError(
            f"Unsupported Toy 2 update rule: {config.policy.rule}"
        )
    if config.policy.rule == "neural_policy":
        expected_input_dim = neural_observation_input_dim(config)
        if config.agents.model.input_dim != expected_input_dim:
            raise ValueError(
                "Toy 2 neural_policy expects "
                f"model.input_dim={expected_input_dim}"
            )
        if config.agents.model.output_dim != 2:
            raise ValueError("Toy 2 neural_policy expects model.output_dim=2")

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    device = resolve_torch_device(config.simulation.device)
    backend: NeuralUpdateBackend = "loop"
    if config.policy.rule == "neural_policy":
        backend = resolve_neural_update_backend(
            (
                config.policy.neural_update_backend
                if neural_update_backend is None
                else neural_update_backend
            ),
            device=device,
            agent_count=(
                config.environment.grid_width * config.environment.grid_height
            ),
        )
    domain = Toy2SpatialDomain(
        config=config,
        config_path=config_path,
        rng=rng,
        neural_peer_rng=np.random.default_rng(config.run.seed + 1_000_003),
        interaction_rng=np.random.default_rng(config.run.seed + 2_000_003),
        reputation_rng=np.random.default_rng(config.run.seed + 3_000_003),
        mobility_rng=np.random.default_rng(config.run.seed + 4_000_003),
        device=device,
        neural_update_backend=backend,
    )
    return BinarySpatialRunner(
        domain=domain,
        epochs=config.simulation.epochs,
        revision_rate=config.policy.revision_rate,
        revision_rng=rng,
        logging_interval=config.logging.interval,
        log_micro_state=config.logging.micro_state,
        log_aggregate_metrics=config.logging.aggregate_metrics,
        timing_rows=timing_rows,
    ).run()
