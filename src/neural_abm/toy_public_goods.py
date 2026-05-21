"""Toy 4: neural public-goods and commons runner."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    can_defer_static_output_average_agent_sync,
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
from neural_abm.config import Toy4Config
from neural_abm.domain_learning_diagnostics import (
    DOMAIN_LEARNING_AGGREGATE_FIELDS,
    DOMAIN_LEARNING_MICRO_FIELDS,
    domain_learning_aggregate_fields,
    domain_learning_micro_fields,
)
from neural_abm.losses import LossVector, loss_values_at
from neural_abm.mobility import (
    MobilityParams,
    MobilityStepResult,
)
from neural_abm.reputation import (
    ReputationParams,
    reputation_imitation_cooperation_probs,
    reputation_observation_extra_dim,
    reputation_observation_features,
)
from neural_abm.results import write_run_metadata_artifacts
from neural_abm.social import (
    PeerIndexCache,
    uniform_peer_count,
    validate_peer_ids,
)
from neural_abm.spatial_binary import (
    BINARY_AGGREGATE_COMMON_FIELDS,
    BINARY_MICRO_COMMON_FIELDS,
    BatchedDistributionDistillationAdapter,
    BinaryLocalStepResult,
    BinaryOutputDistillationReport,
    BinaryPolicyLearningUnit,
    BinaryPolicyStepResult,
    BinarySocialStepResult,
    BinarySpatialRunner,
    BinarySpatialState,
    BinaryStepContext,
    apply_binary_output_distribution_distillation,
    binary_aggregate_common_fields,
    binary_action_probs_from_policy,
    binary_policy_matrix,
    BinaryToyDomainBase,
    BinaryToyResult,
    mix_binary_output_average,
    peer_ids_for_binary_mixer,
    run_batched_policy_gradient_local_update,
    run_binary_policy_learning_step,
    run_binary_output_distribution_distillation,
    run_tensor_runtime_policy_gradient_local_update,
    select_binary_output_similarity_peers,
    StateArray,
    TensorRuntimeDistributionDistillationAdapter,
    timed_context_stage,
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
    combine_state_continuation_advantages,
    domain_decision_bootstrap_weight,
    domain_decision_replay_weight,
    domain_distill_bootstrap_diagnostic_components,
    domain_distill_bootstrap_weight,
    domain_bootstrap_weight,
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


TOY4_MICRO_STATE_FIELDS = [
    *BINARY_MICRO_COMMON_FIELDS,
    "domain_local_action_rate",
    "domain_group_payoff_mean",
    "domain_resource_level",
    *DOMAIN_LEARNING_MICRO_FIELDS,
]


TOY4_AGGREGATE_FIELDS = [
    *BINARY_AGGREGATE_COMMON_FIELDS,
    "domain_payoff_variance",
    "domain_payoff_gini",
    "domain_resource_enabled",
    "domain_resource_level",
    "domain_resource_fraction",
    "domain_collapse_time",
    "domain_action_components",
    "domain_largest_action_cluster_fraction",
    "domain_exploitation_index",
    *DOMAIN_LEARNING_AGGREGATE_FIELDS,
]

class PublicGoodsMLP(nn.Module):
    """Small binary contribution policy network for Toy 4."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


@dataclass
class NeuralPublicGoodsAgent:
    agent_id: int
    model: PublicGoodsMLP
    optimizer: torch.optim.Optimizer

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="public_goods_observation",
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
        raise NotImplementedError("Toy 4 local update requires payoff context")

    def hidden_on(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model.activation(self.model.fc1(observation))

    @torch.no_grad()
    def social_message(self, observation: torch.Tensor) -> dict[str, Any]:
        observed = self.observe(observation)
        logits = self.model(observed)
        probs = torch.softmax(logits, dim=-1)
        hidden = self.hidden_on(observed)
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
        }


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def make_model(config: Toy4Config) -> PublicGoodsMLP:
    model_config = config.agents.model
    return PublicGoodsMLP(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=model_config.output_dim,
    )


def make_optimizer(model: torch.nn.Module, config: Toy4Config) -> torch.optim.Optimizer:
    if config.agents.optimizer.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.agents.optimizer.learning_rate,
        )
    raise ValueError(f"Unsupported optimizer: {config.agents.optimizer.name}")


def create_agents(
    config: Toy4Config,
    device: torch.device,
) -> list[NeuralPublicGoodsAgent]:
    base_state = None
    if config.agents.init_mode == "same_init":
        torch.manual_seed(config.run.seed)
        base_state = clone_state_dict(make_model(config))

    agents: list[NeuralPublicGoodsAgent] = []
    for agent_id in range(config.agent_count):
        if config.agents.init_mode == "independent_init":
            torch.manual_seed(config.run.seed * 1000 + agent_id)
        model = make_model(config).to(device)
        if base_state is not None:
            model.load_state_dict(base_state)
        agents.append(
            NeuralPublicGoodsAgent(
                agent_id=agent_id,
                model=model,
                optimizer=make_optimizer(model, config),
            )
        )
    return agents


def build_public_goods_graph(config: Toy4Config) -> nx.Graph:
    if config.graph.type != "grid":
        raise ValueError(f"Unsupported Toy 4 graph type: {config.graph.type}")
    if config.graph.neighborhood != "von_neumann":
        raise ValueError(f"Unsupported Toy 4 neighborhood: {config.graph.neighborhood}")
    graph = nx.grid_2d_graph(
        config.environment.grid_height,
        config.environment.grid_width,
        periodic=config.graph.periodic,
    )
    mapping = {
        (row, col): row * config.environment.grid_width + col
        for row in range(config.environment.grid_height)
        for col in range(config.environment.grid_width)
    }
    return nx.relabel_nodes(graph, mapping)


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [sorted(int(node) for node in graph.neighbors(i)) for i in range(agent_count)]


def local_groups(neighbors: list[list[int]]) -> list[list[int]]:
    return [
        sorted([agent_id, *peers])
        for agent_id, peers in enumerate(neighbors)
    ]


def initialize_actions(config: Toy4Config, rng: np.random.Generator) -> np.ndarray:
    return (
        rng.random(config.agent_count)
        < config.environment.initial_action_probability
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


def _tensor_index(
    values: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=torch.long)
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _ragged_group_mean_tensor(
    values: torch.Tensor,
    groups: list[list[int]],
) -> torch.Tensor:
    means: list[torch.Tensor] = []
    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    for members in groups:
        if not members:
            means.append(zero)
            continue
        member_index = torch.as_tensor(
            members,
            dtype=torch.long,
            device=values.device,
        )
        means.append(values.index_select(0, member_index).mean())
    if not means:
        return torch.zeros(0, dtype=values.dtype, device=values.device)
    return torch.stack(means)


def resource_fraction(config: Toy4Config, resource_level: float) -> float:
    if not config.environment.resource_enabled:
        return 1.0
    return float(
        np.clip(resource_level / config.environment.resource_carrying_capacity, 0.0, 1.0)
    )


def resource_extraction_rates(config: Toy4Config, agent_count: int) -> np.ndarray:
    """Return per-agent defector extraction rates for Toy4 resource dynamics."""

    base_rate = float(config.environment.resource_extraction_per_defector)
    if agent_count <= 0:
        return np.zeros(0, dtype=np.float64)
    heterogeneity = float(config.environment.resource_extraction_heterogeneity)
    if heterogeneity <= 0.0 or config.environment.resource_extraction_heterogeneity_mode == "none":
        return np.full(agent_count, base_rate, dtype=np.float64)
    if config.environment.resource_extraction_heterogeneity_mode != "checkerboard":
        raise ValueError(
            "unsupported Toy4 resource_extraction_heterogeneity_mode: "
            f"{config.environment.resource_extraction_heterogeneity_mode}"
        )
    index = np.arange(agent_count, dtype=np.int64)
    width = int(config.environment.grid_width)
    rows = index // width
    cols = index % width
    factors = np.where(
        (rows + cols) % 2 == 0,
        1.0 + heterogeneity,
        1.0 - heterogeneity,
    )
    return base_rate * factors.astype(np.float64)


def _resource_sustain_action_rate_for_members(
    *,
    config: Toy4Config,
    members: list[int],
    extraction_rates: np.ndarray,
) -> float:
    if not config.environment.resource_enabled or not members:
        return 0.0
    recovery = float(config.environment.resource_recovery_rate)
    member_indices = np.asarray(members, dtype=np.int64)
    extraction = float(np.mean(extraction_rates[member_indices]))
    denominator = recovery + extraction
    if denominator <= 0.0:
        return 0.0
    return float(np.clip(extraction / denominator, 0.0, 1.0))


def resource_sustain_action_rate(config: Toy4Config) -> float:
    """Return the action-1 rate needed to keep the resource from declining."""

    if not config.environment.resource_enabled:
        return 0.0
    members = list(range(config.agent_count))
    return _resource_sustain_action_rate_for_members(
        config=config,
        members=members,
        extraction_rates=resource_extraction_rates(config, config.agent_count),
    )


def resource_break_even_fraction(config: Toy4Config) -> float:
    """Return the resource fraction where contribution welfare breaks even."""

    multiplier = float(config.game.multiplier)
    if multiplier <= 0.0:
        return 1.0
    return float(
        np.clip(float(config.game.contribution_cost) / multiplier, 0.0, 1.0)
    )


def resource_observation_values(
    *,
    actions: StateArray,
    groups: list[list[int]],
    resource_level: float,
    config: Toy4Config,
) -> np.ndarray:
    """Return the Toy4 resource feature exposed to neural observations."""

    action_values = to_numpy_view(actions, dtype=np.int64)
    if len(action_values) == 0:
        return np.zeros(0, dtype=np.float64)
    mode = config.environment.resource_observation_mode
    if mode == "global":
        return np.full(
            len(action_values),
            resource_fraction(config, resource_level),
            dtype=np.float64,
        )
    if mode == "hidden" or not config.environment.resource_enabled:
        return np.ones(len(action_values), dtype=np.float64)
    if mode != "local_sustain":
        raise ValueError(f"unsupported Toy4 resource_observation_mode: {mode}")

    extraction_rates = resource_extraction_rates(config, len(action_values))
    memberships = _toy4_group_memberships(len(action_values), groups)
    values = np.ones(len(action_values), dtype=np.float64)
    for focal_id, focal_groups in enumerate(memberships):
        if not focal_groups:
            continue
        group_values: list[float] = []
        for group_id in focal_groups:
            members = groups[group_id]
            if not members:
                continue
            sustain_rate = _resource_sustain_action_rate_for_members(
                config=config,
                members=members,
                extraction_rates=extraction_rates,
            )
            if sustain_rate <= 0.0:
                group_values.append(1.0)
                continue
            local_rate = float(np.mean(action_values[np.asarray(members, dtype=np.int64)]))
            group_values.append(float(np.clip(local_rate / sustain_rate, 0.0, 1.0)))
        if group_values:
            values[focal_id] = float(np.mean(group_values))
    return values


def resource_environment_continuation_advantages(
    *,
    actions: StateArray,
    groups: list[list[int]],
    config: Toy4Config,
    resource_level: float,
) -> np.ndarray:
    """Return a resource-maintenance pressure signal for action 1.

    The signal is diagnostic by default because current Toy4 objective profiles
    set ``environment_weight=0``. It becomes behaviorally active only when a
    resource-aware objective explicitly weights the environment component.
    """

    action_values = to_numpy_view(actions, dtype=np.float64)
    if len(action_values) == 0 or not config.environment.resource_enabled:
        return np.zeros(len(action_values), dtype=np.float64)

    sustain_rate = resource_sustain_action_rate(config)
    action_rate = float(np.mean(action_values))
    maintain_pressure = (
        max(0.0, sustain_rate - action_rate) / max(sustain_rate, 1e-8)
        if sustain_rate > 0.0
        else 0.0
    )

    break_even_fraction = resource_break_even_fraction(config)
    current_fraction = resource_fraction(config, resource_level)
    resource_pressure = (
        max(0.0, break_even_fraction - current_fraction)
        / max(break_even_fraction, 1e-8)
        if break_even_fraction > 0.0
        else 0.0
    )

    pressure = maintain_pressure + resource_pressure
    pressure_values = np.full(len(action_values), pressure, dtype=np.float64)
    return (
        config.policy.domain.resource_environment_pressure_weight * pressure_values
        + config.policy.domain.resource_environment_lookahead_weight
        * resource_action_lookahead_advantages(
            actions=action_values,
            config=config,
            resource_level=resource_level,
        )
        + config.policy.domain.resource_environment_threshold_weight
        * resource_threshold_continuation_advantages(
            actions=action_values,
            groups=groups,
            config=config,
        )
    )


def resource_action_lookahead_advantages(
    *,
    actions: StateArray,
    config: Toy4Config,
    resource_level: float,
) -> np.ndarray:
    """Return action-1 resource stock value in one-step counterfactual form.

    This is an opt-in Toy4-specific continuation signal. It compares the next
    resource stock if each focal agent chooses action 1 versus action 0 while
    other actions stay fixed, then translates the stock delta into population
    payoff capacity at the current or sustainable contribution rate.
    """

    action_values = to_numpy_view(actions, dtype=np.int64)
    if len(action_values) == 0 or not config.environment.resource_enabled:
        return np.zeros(len(action_values), dtype=np.float64)

    normalizer = max(payoff_normalizer(config), 1e-8)
    capacity = max(float(config.environment.resource_carrying_capacity), 1e-8)
    action_rate = float(np.mean(action_values))
    valuation_rate = max(action_rate, resource_sustain_action_rate(config))
    values = np.zeros(len(action_values), dtype=np.float64)
    for focal_id in range(len(action_values)):
        action_one = action_values.copy()
        action_zero = action_values.copy()
        action_one[focal_id] = 1
        action_zero[focal_id] = 0
        resource_one = update_resource_level(resource_level, action_one, config)
        resource_zero = update_resource_level(resource_level, action_zero, config)
        resource_fraction_delta = max(0.0, resource_one - resource_zero) / capacity
        values[focal_id] = (
            resource_fraction_delta
            * len(action_values)
            * float(config.game.multiplier)
            * valuation_rate
            / normalizer
        )
    return values


def resource_threshold_continuation_advantages(
    *,
    actions: StateArray,
    groups: list[list[int]],
    config: Toy4Config,
) -> np.ndarray:
    """Return a coordinated sustain-threshold value for action 1.

    Unlike one-step stock lookahead, this signal stays active after collapse
    because it values an agent's role in moving a population or local
    neighborhood toward the resource sustain contribution rate.
    """

    action_values = to_numpy_view(actions, dtype=np.int64)
    if len(action_values) == 0 or not config.environment.resource_enabled:
        return np.zeros(len(action_values), dtype=np.float64)
    sustain_rate = resource_sustain_action_rate(config)
    if sustain_rate <= 0.0:
        return np.zeros(len(action_values), dtype=np.float64)
    extraction_rates = resource_extraction_rates(config, len(action_values))

    if config.policy.domain.resource_environment_threshold_scope == "population":
        return _resource_threshold_values_for_cohort(
            action_values=action_values,
            members=list(range(len(action_values))),
            sustain_rate=_resource_sustain_action_rate_for_members(
                config=config,
                members=list(range(len(action_values))),
                extraction_rates=extraction_rates,
            ),
        )

    memberships = _toy4_group_memberships(len(action_values), groups)
    values = np.zeros(len(action_values), dtype=np.float64)
    for focal_id, focal_groups in enumerate(memberships):
        if not focal_groups:
            continue
        group_values = [
            _resource_threshold_value_for_member(
                action_values=action_values,
                members=groups[group_id],
                focal_id=focal_id,
                sustain_rate=_resource_sustain_action_rate_for_members(
                    config=config,
                    members=groups[group_id],
                    extraction_rates=extraction_rates,
                ),
            )
            for group_id in focal_groups
            if groups[group_id]
        ]
        if group_values:
            values[focal_id] = float(np.mean(group_values))
    return values


def _resource_threshold_values_for_cohort(
    *,
    action_values: np.ndarray,
    members: list[int],
    sustain_rate: float,
) -> np.ndarray:
    values = np.zeros(len(action_values), dtype=np.float64)
    for focal_id in members:
        values[focal_id] = _resource_threshold_value_for_member(
            action_values=action_values,
            members=members,
            focal_id=focal_id,
            sustain_rate=sustain_rate,
        )
    return values


def _resource_threshold_value_for_member(
    *,
    action_values: np.ndarray,
    members: list[int],
    focal_id: int,
    sustain_rate: float,
) -> float:
    if not members:
        return 0.0
    member_indices = np.asarray(members, dtype=np.int64)
    cohort_size = float(len(member_indices))
    current_sum = float(np.sum(action_values[member_indices]))
    focal_action = float(action_values[focal_id])
    rate_zero = (current_sum - focal_action) / cohort_size
    rate_one = (current_sum - focal_action + 1.0) / cohort_size
    urgency = max(0.0, sustain_rate - rate_zero) / max(sustain_rate, 1e-8)
    threshold_gain = (
        max(0.0, min(rate_one, sustain_rate) - min(rate_zero, sustain_rate))
        / max(sustain_rate, 1e-8)
    )
    return float(urgency * threshold_gain * cohort_size)


def compute_public_goods_payoffs(
    actions: StateArray,
    groups: list[list[int]],
    multiplier: float,
    contribution_cost: float,
    resource_multiplier: float = 1.0,
    group_member_index: np.ndarray | torch.Tensor | None = None,
) -> StateArray:
    """Compute overlapping local public-goods payoffs."""

    if group_member_index is not None:
        return compute_public_goods_payoffs_from_index(
            actions=actions,
            group_member_index=group_member_index,
            multiplier=multiplier,
            contribution_cost=contribution_cost,
            resource_multiplier=resource_multiplier,
        )

    if isinstance(actions, torch.Tensor):
        actions_np = to_numpy_view(actions, dtype=np.int64)
        payoffs_np = compute_public_goods_payoffs(
            actions=actions_np,
            groups=groups,
            multiplier=multiplier,
            contribution_cost=contribution_cost,
            resource_multiplier=resource_multiplier,
        )
        return torch.as_tensor(
            payoffs_np,
            dtype=torch.float64,
            device=actions.device,
        )

    totals = np.zeros(len(actions), dtype=np.float64)
    counts = np.zeros(len(actions), dtype=np.float64)
    for members in groups:
        member_actions = actions[members]
        group_size = len(members)
        contribution = float(np.sum(member_actions))
        share = resource_multiplier * multiplier * contribution / group_size
        for member in members:
            totals[member] += share - contribution_cost * float(actions[member])
            counts[member] += 1.0
    return totals / np.maximum(counts, 1.0)


def compute_public_goods_payoffs_from_index(
    actions: StateArray,
    group_member_index: np.ndarray | torch.Tensor,
    multiplier: float,
    contribution_cost: float,
    resource_multiplier: float = 1.0,
) -> StateArray:
    """Compute overlapping public-goods payoffs for dense uniform groups."""

    if isinstance(actions, torch.Tensor) or isinstance(group_member_index, torch.Tensor):
        device = (
            actions.device
            if isinstance(actions, torch.Tensor)
            else group_member_index.device
        )
        action_values = _tensor_vector(actions, dtype=torch.long, device=device)
        member_index = _tensor_index(group_member_index, device=device)
        if member_index.ndim != 2:
            raise ValueError("group_member_index must have shape [groups, members]")
        if member_index.shape[1] == 0:
            return torch.zeros(
                int(action_values.shape[0]),
                dtype=torch.float64,
                device=device,
            )
        actions_float = action_values.to(dtype=torch.float64)
        group_size = int(member_index.shape[1])
        group_actions = actions_float[member_index]
        group_contributions = group_actions.sum(dim=1)
        group_shares = (
            resource_multiplier * multiplier * group_contributions / group_size
        )

        flat_members = member_index.reshape(-1)
        flat_payoffs = (
            group_shares.repeat_interleave(group_size)
            - contribution_cost * actions_float.index_select(0, flat_members)
        )
        totals = torch.zeros(
            int(action_values.shape[0]),
            dtype=torch.float64,
            device=device,
        )
        totals.index_add_(0, flat_members, flat_payoffs)
        counts = torch.zeros_like(totals)
        counts.index_add_(0, flat_members, torch.ones_like(flat_payoffs))
        return totals / counts.clamp_min(1.0)

    if group_member_index.ndim != 2:
        raise ValueError("group_member_index must have shape [groups, members]")
    if group_member_index.shape[1] == 0:
        return np.zeros(len(actions), dtype=np.float64)
    actions_float = actions.astype(np.float64, copy=False)
    group_size = int(group_member_index.shape[1])
    group_actions = actions_float[group_member_index]
    group_contributions = np.sum(group_actions, axis=1, dtype=np.float64)
    group_shares = resource_multiplier * multiplier * group_contributions / group_size

    flat_members = group_member_index.ravel()
    flat_payoffs = (
        np.repeat(group_shares, group_size)
        - contribution_cost * actions_float[flat_members]
    )
    totals = np.bincount(flat_members, weights=flat_payoffs, minlength=len(actions))
    counts = np.bincount(flat_members, minlength=len(actions)).astype(np.float64)
    return totals / np.maximum(counts, 1.0)


def update_resource_level(
    resource_level: float,
    actions: StateArray,
    config: Toy4Config,
) -> float:
    if not config.environment.resource_enabled:
        return config.environment.resource_carrying_capacity
    action_values = to_numpy_view(actions, dtype=np.float64)
    contributors = float(np.sum(action_values))
    agent_count = len(action_values)
    defectors = float(agent_count - contributors)
    recovery = config.environment.resource_recovery_rate * contributors
    if config.environment.resource_extraction_heterogeneity <= 0.0:
        extraction = config.environment.resource_extraction_per_defector * defectors
    else:
        extraction = float(
            np.dot(
                resource_extraction_rates(config, agent_count),
                1.0 - action_values,
            )
        )
    return float(
        np.clip(
            resource_level + recovery - extraction,
            0.0,
            config.environment.resource_carrying_capacity,
        )
    )


def local_contribution_rates(
    actions: StateArray,
    groups: list[list[int]],
) -> np.ndarray:
    if isinstance(actions, torch.Tensor):
        values = actions.to(dtype=torch.float64)
        return _ragged_group_mean_tensor(values, groups).detach().cpu().numpy()
    return np.asarray(
        [float(np.mean(actions[members])) for members in groups],
        dtype=np.float64,
    )


def group_payoff_means(payoffs: StateArray, groups: list[list[int]]) -> np.ndarray:
    if isinstance(payoffs, torch.Tensor):
        values = payoffs.to(dtype=torch.float64)
        return _ragged_group_mean_tensor(values, groups).detach().cpu().numpy()
    return np.asarray(
        [float(np.mean(payoffs[members])) for members in groups],
        dtype=np.float64,
    )


def uniform_group_member_index(groups: list[list[int]]) -> np.ndarray | None:
    """Return a dense member index when every local group has the same size."""

    if not groups:
        return None
    group_size = len(groups[0])
    if group_size == 0 or any(len(group) != group_size for group in groups):
        return None
    return np.asarray(groups, dtype=np.int64)


def local_contribution_rates_from_index(
    actions: StateArray,
    group_member_index: np.ndarray | torch.Tensor,
) -> np.ndarray:
    if isinstance(actions, torch.Tensor) or isinstance(group_member_index, torch.Tensor):
        device = (
            actions.device
            if isinstance(actions, torch.Tensor)
            else group_member_index.device
        )
        action_values = _tensor_vector(actions, dtype=torch.float64, device=device)
        member_index = _tensor_index(group_member_index, device=device)
        return action_values[member_index].mean(dim=1).detach().cpu().numpy()
    return np.mean(actions[group_member_index], axis=1, dtype=np.float64)


def group_payoff_means_from_index(
    payoffs: StateArray,
    group_member_index: np.ndarray | torch.Tensor,
) -> np.ndarray:
    if isinstance(payoffs, torch.Tensor) or isinstance(group_member_index, torch.Tensor):
        device = (
            payoffs.device
            if isinstance(payoffs, torch.Tensor)
            else group_member_index.device
        )
        payoff_values = _tensor_vector(payoffs, dtype=torch.float64, device=device)
        member_index = _tensor_index(group_member_index, device=device)
        return payoff_values[member_index].mean(dim=1).detach().cpu().numpy()
    return np.mean(payoffs[group_member_index], axis=1, dtype=np.float64)


def payoff_normalizer(config: Toy4Config) -> float:
    return max(
        config.game.multiplier,
        config.game.contribution_cost,
        1.0,
    )


def _toy4_group_memberships(
    agent_count: int,
    groups: list[list[int]],
) -> list[list[int]]:
    memberships = [[] for _ in range(agent_count)]
    for group_id, members in enumerate(groups):
        for member in members:
            memberships[int(member)].append(group_id)
    return memberships


def _toy4_social_continuation_advantages(
    action_count: int,
    config: Toy4Config,
) -> np.ndarray:
    if not config.state.reputation.enabled:
        return np.zeros(action_count, dtype=np.float64)
    return np.full(
        action_count,
        1.0 - config.state.reputation.decay,
        dtype=np.float64,
    )


def contribution_advantage_components(
    *,
    actions: StateArray,
    groups: list[list[int]],
    config: Toy4Config,
    resource_level: float,
) -> StateContinuationComponents:
    action_values = to_numpy_view(actions, dtype=np.int64)
    normalizer = max(payoff_normalizer(config), 1e-8)
    resource_multiplier = resource_fraction(config, resource_level)
    memberships = _toy4_group_memberships(len(action_values), groups)
    group_counts = np.asarray(
        [max(len(agent_groups), 1) for agent_groups in memberships],
        dtype=np.float64,
    )
    material = np.zeros(len(action_values), dtype=np.float64)
    welfare = np.zeros(len(action_values), dtype=np.float64)

    for focal_id, focal_groups in enumerate(memberships):
        if not focal_groups:
            continue
        payoff_delta = np.zeros(len(action_values), dtype=np.float64)
        for group_id in focal_groups:
            members = groups[group_id]
            if not members:
                continue
            share_delta = resource_multiplier * config.game.multiplier / len(members)
            for member_id in members:
                payoff_delta[int(member_id)] += (
                    share_delta / group_counts[int(member_id)]
                )
            payoff_delta[focal_id] -= (
                config.game.contribution_cost / group_counts[focal_id]
            )
        material[focal_id] = payoff_delta[focal_id] / normalizer
        welfare[focal_id] = float(np.sum(payoff_delta)) / normalizer

    raw_components = combine_state_continuation_advantages(
        material=material,
        social=_toy4_social_continuation_advantages(len(action_values), config),
        welfare=welfare,
        environment=resource_environment_continuation_advantages(
            actions=action_values,
            groups=groups,
            config=config,
            resource_level=resource_level,
        ),
        objective=config.policy.domain.objective,
    )
    return raw_components


def toy4_target_basin_payoff(config: Toy4Config) -> float:
    """Return the Toy 4 v1 ceiling target used by basin credit."""

    return float(config.game.multiplier - config.game.contribution_cost)


def build_observations(
    actions: StateArray,
    payoffs: StateArray,
    payoff_ema: StateArray,
    groups: list[list[int]],
    resource_level: float,
    config: Toy4Config,
    device: torch.device,
    reputation: StateArray | None = None,
    group_member_index: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    if any(
        isinstance(values, torch.Tensor)
        for values in (actions, payoffs, payoff_ema, reputation, group_member_index)
    ):
        return build_observations_tensor(
            actions=actions,
            payoffs=payoffs,
            payoff_ema=payoff_ema,
            groups=groups,
            resource_level=resource_level,
            config=config,
            device=device,
            reputation=reputation,
            group_member_index=group_member_index,
        )
    if group_member_index is None:
        local_rates = local_contribution_rates(actions, groups)
        group_means = group_payoff_means(payoffs, groups)
    else:
        local_rates = local_contribution_rates_from_index(
            actions,
            group_member_index,
        )
        group_means = group_payoff_means_from_index(payoffs, group_member_index)
    normalizer = payoff_normalizer(config)
    observations = np.column_stack(
        [
            actions.astype(np.float64),
            local_rates,
            payoff_ema / normalizer,
            group_means / normalizer,
            resource_observation_values(
                actions=actions,
                groups=groups,
                resource_level=resource_level,
                config=config,
            ),
            np.ones(len(actions), dtype=np.float64),
        ]
    )
    if config.state.reputation.observation_mode != "none":
        if reputation is None:
            raise ValueError("reputation observations require reputation state")
        if (
            group_member_index is not None
            and config.state.reputation.observation_mode == "self_neighbor_mean"
        ):
            reputation_features = np.column_stack(
                [
                    reputation.astype(np.float64),
                    group_payoff_means_from_index(reputation, group_member_index),
                ]
            )
        else:
            reputation_features = reputation_observation_features(
                reputation=reputation,
                peer_ids=groups,
                mode=config.state.reputation.observation_mode,
            )
        observations = np.column_stack(
            [
                observations,
                reputation_features,
            ]
        )
    return torch.as_tensor(observations, dtype=torch.float32, device=device)


def build_observations_tensor(
    actions: StateArray,
    payoffs: StateArray,
    payoff_ema: StateArray,
    groups: list[list[int]],
    resource_level: float,
    config: Toy4Config,
    device: torch.device,
    reputation: StateArray | None = None,
    group_member_index: np.ndarray | torch.Tensor | None = None,
) -> torch.Tensor:
    action_values = _tensor_vector(actions, dtype=torch.float64, device=device)
    payoff_values = _tensor_vector(payoffs, dtype=torch.float64, device=device)
    payoff_ema_values = _tensor_vector(payoff_ema, dtype=torch.float64, device=device)
    agent_count = int(action_values.shape[0])
    if payoff_values.shape[0] != agent_count or payoff_ema_values.shape[0] != agent_count:
        raise ValueError("payoff arrays must match action count")
    if group_member_index is None:
        local_rates = _ragged_group_mean_tensor(action_values, groups)
        group_means = _ragged_group_mean_tensor(payoff_values, groups)
    else:
        member_index = _tensor_index(group_member_index, device=device)
        if member_index.shape[0] != agent_count:
            raise ValueError("group_member_index first dimension must match action count")
        if member_index.shape[1] == 0:
            local_rates = torch.zeros(agent_count, dtype=torch.float64, device=device)
            group_means = torch.zeros(agent_count, dtype=torch.float64, device=device)
        else:
            local_rates = action_values[member_index].mean(dim=1)
            group_means = payoff_values[member_index].mean(dim=1)
    normalizer = payoff_normalizer(config)
    resource_observations = torch.as_tensor(
        resource_observation_values(
            actions=action_values,
            groups=groups,
            resource_level=resource_level,
            config=config,
        ),
        dtype=torch.float64,
        device=device,
    )
    observations = torch.stack(
        [
            action_values,
            local_rates,
            payoff_ema_values / normalizer,
            group_means / normalizer,
            resource_observations,
            torch.ones(agent_count, dtype=torch.float64, device=device),
        ],
        dim=1,
    )
    if config.state.reputation.observation_mode != "none":
        if reputation is None:
            raise ValueError("reputation observations require reputation state")
        reputation_values = _tensor_vector(reputation, dtype=torch.float64, device=device)
        if reputation_values.shape[0] != agent_count:
            raise ValueError("reputation length must match action count")
        if config.state.reputation.observation_mode != "self_neighbor_mean":
            raise ValueError(
                "unsupported reputation observation mode: "
                f"{config.state.reputation.observation_mode}"
            )
        if group_member_index is None:
            reputation_means = _ragged_group_mean_tensor(reputation_values, groups)
        else:
            member_index = _tensor_index(group_member_index, device=device)
            if member_index.shape[1] == 0:
                reputation_means = torch.zeros(
                    agent_count,
                    dtype=torch.float64,
                    device=device,
                )
            else:
                reputation_means = reputation_values[member_index].mean(dim=1)
        observations = torch.cat(
            [
                observations,
                torch.stack([reputation_values, reputation_means], dim=1),
            ],
            dim=1,
        )
    return observations.to(dtype=torch.float32)


@torch.no_grad()
def collect_policy_probs(
    agents: list[NeuralPublicGoodsAgent],
    observations: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    return batched_mlp_policy_probs(
        [agent.model for agent in agents],
        observations,
        temperature=temperature,
    )


def toy4_terminal_argmax_active(
    *,
    epoch: int,
    total_epochs: int,
    terminal_argmax_epochs: int,
) -> bool:
    if terminal_argmax_epochs <= 0:
        return False
    return epoch > max(0, total_epochs - terminal_argmax_epochs)


def toy4_decision_mode_for_epoch(config: Toy4Config, epoch: int) -> str:
    decision = config.policy.decision
    if decision.mode == "sampled" and toy4_terminal_argmax_active(
        epoch=epoch,
        total_epochs=config.simulation.epochs,
        terminal_argmax_epochs=decision.terminal_argmax_epochs,
    ):
        return "argmax"
    return decision.mode


def decision_action_probs(
    policy_probs: torch.Tensor,
    config: Toy4Config,
    *,
    mode: str | None = None,
) -> torch.Tensor:
    decision = config.policy.decision
    resolved_mode = decision.mode if mode is None else mode
    if resolved_mode == "argmax":
        selected = torch.argmax(policy_probs, dim=-1)
        return torch.nn.functional.one_hot(selected, num_classes=2).to(policy_probs)
    if resolved_mode != "sampled":
        raise ValueError(f"Unsupported decision mode: {resolved_mode}")
    adjusted = policy_probs
    if decision.action_temperature != 1.0:
        logits = torch.log(torch.clamp(policy_probs, min=torch.finfo(policy_probs.dtype).tiny))
        adjusted = torch.softmax(logits / decision.action_temperature, dim=-1)
    if decision.exploration_epsilon > 0.0:
        adjusted = (
            (1.0 - decision.exploration_epsilon) * adjusted
            + decision.exploration_epsilon * 0.5
        )
    return adjusted


def select_actions_from_probs(
    current_actions: StateArray,
    action_probs: torch.Tensor,
    revision_mask: np.ndarray,
    rng: np.random.Generator,
    mode: str = "sampled",
) -> StateArray:
    current_array = to_numpy_view(current_actions, dtype=np.int64)
    next_actions = current_array.copy()
    probabilities = action_probs.detach().cpu().numpy()
    for agent_id in np.flatnonzero(revision_mask):
        if mode == "sampled":
            next_actions[int(agent_id)] = int(
                rng.random() < probabilities[int(agent_id), 1]
            )
        elif mode == "argmax":
            next_actions[int(agent_id)] = int(np.argmax(probabilities[int(agent_id)]))
        else:
            raise ValueError(f"Unsupported decision mode: {mode}")
    if isinstance(current_actions, torch.Tensor):
        return torch.as_tensor(
            next_actions,
            dtype=current_actions.dtype,
            device=current_actions.device,
        )
    return next_actions


def contribution_probs_to_policy_tensor(
    contribution_probs: StateArray,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(contribution_probs, torch.Tensor):
        probs = contribution_probs.to(device=device, dtype=torch.float32)
        return torch.stack([1.0 - probs, probs], dim=1)
    return torch.as_tensor(
        binary_policy_matrix(contribution_probs),
        dtype=torch.float32,
        device=device,
    )


def imitation_candidate_probabilities(
    actions: np.ndarray,
    payoffs: np.ndarray,
    neighbors: list[list[int]],
    revision_mask: np.ndarray,
    selection_strength: float,
) -> np.ndarray:
    probabilities = actions.astype(np.float64).copy()
    if selection_strength <= 0.0:
        return probabilities
    for agent_id in np.flatnonzero(revision_mask):
        candidates = [int(agent_id), *neighbors[int(agent_id)]]
        best = max(candidates, key=lambda candidate: payoffs[candidate])
        payoff_delta = float(payoffs[best] - payoffs[int(agent_id)])
        if payoff_delta > 0.0:
            logit = np.clip(selection_strength * payoff_delta, -60.0, 60.0)
            copy_probability = float(1.0 / (1.0 + np.exp(-logit)))
            probabilities[int(agent_id)] = (
                (1.0 - copy_probability) * float(actions[int(agent_id)])
                + copy_probability * float(actions[best])
            )
    return probabilities


def select_peers_by_output_similarity(
    neighbors: list[list[int]],
    contribution_probs: np.ndarray,
    peer_rule: str,
    threshold: float,
) -> list[list[int]]:
    return select_binary_output_similarity_peers(
        neighbors=neighbors,
        action_probs=contribution_probs,
        peer_rule=peer_rule,
        threshold=threshold,
        error_label="Toy 4",
    )


def peer_ids_for_mixer(
    peer_ids: list[list[int]],
    mixer: str,
    agent_count: int,
) -> list[list[int]]:
    return peer_ids_for_binary_mixer(
        peer_ids=peer_ids,
        mixer=mixer,
        agent_count=agent_count,
        error_label="Toy 4",
    )


def apply_output_average_to_contribution_probs(
    contribution_probs: np.ndarray,
    peer_ids: list[list[int]],
    alpha: float,
) -> tuple[np.ndarray, list[float]]:
    """Compatibility wrapper for the shared binary output-average mixer."""

    return mix_binary_output_average(contribution_probs, peer_ids, alpha)


def train_neural_local_policy(
    agent: NeuralPublicGoodsAgent,
    observation: torch.Tensor,
    action: int,
    payoff: float,
    payoff_baseline: float,
    config: Toy4Config,
    advantage: float | None = None,
    teacher_distill_probability: float | None = None,
    teacher_distill_weight: float = 0.0,
    base_loss_weight: float = 1.0,
) -> float:
    logits = agent.model(observation)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    if advantage is None:
        advantage = (payoff - payoff_baseline) / payoff_normalizer(config)
    base_loss = (
        -float(advantage) * log_probs[int(action)]
        - config.environment.entropy_beta * entropy
    )
    loss = float(base_loss_weight) * base_loss
    if teacher_distill_probability is not None and teacher_distill_weight > 0.0:
        loss = loss + float(teacher_distill_weight) * teacher_distillation_loss(
            log_probs=log_probs,
            teacher_probability=teacher_distill_probability,
            loss_type=config.policy.domain.bootstrap.distill_loss,
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
    agent: NeuralPublicGoodsAgent,
    observation: torch.Tensor,
    action: int,
    payoff: float,
    payoff_baseline: float,
    config: Toy4Config,
    advantage: float | None,
    teacher_probability: float,
) -> tuple[float, float, float, float, float]:
    logits = agent.model(observation)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    if advantage is None:
        advantage = (payoff - payoff_baseline) / payoff_normalizer(config)
    base_loss = (
        -float(advantage) * log_probs[int(action)]
        - config.environment.entropy_beta * entropy
    )
    distill_loss = teacher_distillation_loss(
        log_probs=log_probs,
        teacher_probability=teacher_probability,
        loss_type=config.policy.domain.bootstrap.distill_loss,
    )
    parameters = [parameter for parameter in agent.model.parameters() if parameter.requires_grad]
    return _gradient_conflict_values(
        base_loss=base_loss,
        distill_loss=distill_loss,
        parameters=parameters,
    )


def train_neural_local_policies_batched(
    agents: list[NeuralPublicGoodsAgent],
    observations: torch.Tensor,
    actions: np.ndarray,
    payoffs: np.ndarray,
    payoff_baseline: np.ndarray,
    revision_mask: np.ndarray,
    config: Toy4Config,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
    timing_context: BinaryStepContext | None = None,
) -> LossVector:
    return train_neural_local_policies_batched_update(
        agents=agents,
        observations=observations,
        actions=actions,
        payoffs=payoffs,
        payoff_baseline=payoff_baseline,
        revision_mask=revision_mask,
        config=config,
        parameters=parameters,
        adam_state_cache=adam_state_cache,
        synchronize_model_parameters=synchronize_model_parameters,
        synchronize_optimizer_states=synchronize_optimizer_states,
        timing_context=timing_context,
    ).losses


def train_neural_local_policies_batched_update(
    agents: list[NeuralPublicGoodsAgent],
    observations: torch.Tensor,
    actions: np.ndarray,
    payoffs: np.ndarray,
    payoff_baseline: np.ndarray,
    revision_mask: np.ndarray,
    config: Toy4Config,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    synchronize_model_parameters: bool = True,
    synchronize_optimizer_states: bool = True,
    timing_context: BinaryStepContext | None = None,
) -> BatchedMLPUpdateResult:
    active_agent_ids = [int(agent_id) for agent_id in np.flatnonzero(revision_mask)]
    report = run_batched_policy_gradient_local_update(
        agents=agents,
        observations=observations,
        actions=actions,
        advantages=(payoffs - payoff_baseline) / payoff_normalizer(config),
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


def apply_output_average_distillation(
    agents: list[NeuralPublicGoodsAgent],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
) -> LossVector:
    return apply_binary_output_distribution_distillation(
        agents=agents,
        observations=observations,
        peer_ids=peer_ids,
        alpha=alpha,
        previous_probs=previous_probs,
        logits_fn=lambda agent, agent_id, observed: agent.model(observed[agent_id]),
        loss_mode="cross_entropy",
    )


def apply_output_average_distillation_batched(
    agents: list[NeuralPublicGoodsAgent],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    uniform_peer_count: int | None = None,
    uniform_peer_index: torch.Tensor | None = None,
    peer_index_cache: PeerIndexCache | None = None,
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
        uniform_peer_count=uniform_peer_count,
        uniform_peer_index=uniform_peer_index,
        peer_index_cache=peer_index_cache,
        validate_peers=validate_peers,
        timing_context=timing_context,
        synchronize_model_parameters=synchronize_model_parameters,
        synchronize_optimizer_states=synchronize_optimizer_states,
    ).losses


def apply_output_average_distillation_batched_update(
    agents: list[NeuralPublicGoodsAgent],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
    previous_probs: torch.Tensor,
    parameters: BatchedMLPParameters | None = None,
    adam_state_cache: BatchedAdamStateCache | None = None,
    uniform_peer_count: int | None = None,
    uniform_peer_index: torch.Tensor | None = None,
    peer_index_cache: PeerIndexCache | None = None,
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
        uniform_peer_count=uniform_peer_count,
        uniform_peer_index=uniform_peer_index,
        peer_index_cache=peer_index_cache,
        validate_peers=validate_peers,
        timing_context=timing_context,
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
    )


def reputation_params_from_config(config: Toy4Config) -> ReputationParams:
    return ReputationParams(
        enabled=config.state.reputation.enabled,
        decay=config.state.reputation.decay,
        peer_rule=config.state.reputation.peer_rule,
        temperature=config.state.reputation.temperature,
        noise=config.state.reputation.noise,
    )


def neural_observation_input_dim(config: Toy4Config) -> int:
    return 6 + reputation_observation_extra_dim(config.state.reputation.observation_mode)


def mobility_params_from_config(config: Toy4Config) -> MobilityParams:
    return MobilityParams(
        enabled=config.state.mobility.enabled,
        rate=config.state.mobility.rate,
        candidate_pool_size=config.state.mobility.candidate_pool_size,
        selection_rule=config.state.mobility.selection_rule,
        move_cost=config.state.mobility.move_cost,
    )


def payoff_gini(payoffs: StateArray) -> float:
    payoff_array = to_numpy_view(payoffs, dtype=np.float64)
    if len(payoff_array) == 0:
        return 0.0
    shifted = payoff_array - np.min(payoff_array)
    mean = float(np.mean(shifted))
    if mean == 0.0:
        return 0.0
    differences = np.abs(shifted[:, None] - shifted[None, :])
    return float(np.mean(differences) / (2.0 * mean))


def contributor_cluster_metrics(
    actions: StateArray,
    graph: nx.Graph,
) -> tuple[int, float]:
    action_array = to_numpy_view(actions, dtype=np.int64)
    contributors = [
        int(agent_id) for agent_id, action in enumerate(action_array) if action == 1
    ]
    if not contributors:
        return 0, 0.0
    subgraph = graph.subgraph(contributors)
    components = list(nx.connected_components(subgraph))
    largest_fraction = max(len(component) for component in components) / len(action_array)
    return len(components), float(largest_fraction)


def exploitation_index(actions: StateArray, payoffs: StateArray) -> float:
    action_array = to_numpy_view(actions, dtype=np.int64)
    payoff_array = to_numpy_view(payoffs, dtype=np.float64)
    free_riders = payoff_array[action_array == 0]
    if len(free_riders) == 0:
        return 0.0
    return float(np.mean(free_riders) - np.mean(payoff_array))


def aggregate_row(
    config: Toy4Config,
    epoch: int,
    actions: np.ndarray,
    payoffs: np.ndarray,
    policy_probs: torch.Tensor,
    peer_ids: list[list[int]],
    graph: nx.Graph,
    resource_level: float,
    collapse_time: int | None,
    realized_revision_rate: float,
    reputation: np.ndarray | None = None,
    mobility_result: MobilityStepResult | None = None,
    policy_probs_pre_revision: torch.Tensor | None = None,
    policy_probs_post_local: torch.Tensor | None = None,
    local_losses: list[float] | None = None,
    revised_local_losses: list[float] | None = None,
    social_losses: list[float] | None = None,
) -> dict[str, object]:
    contributor_components, largest_contributor_fraction = contributor_cluster_metrics(
        actions,
        graph,
    )
    return {
        **binary_aggregate_common_fields(
            config=config,
            toy="toy4",
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
        ),
        "domain_payoff_variance": float(np.var(payoffs)),
        "domain_payoff_gini": payoff_gini(payoffs),
        "domain_resource_enabled": config.environment.resource_enabled,
        "domain_resource_level": resource_level,
        "domain_resource_fraction": resource_fraction(config, resource_level),
        "domain_collapse_time": "" if collapse_time is None else collapse_time,
        "domain_action_components": contributor_components,
        "domain_largest_action_cluster_fraction": largest_contributor_fraction,
        "domain_exploitation_index": exploitation_index(actions, payoffs),
    }


def make_run_dir(config: Toy4Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        config.run.output_dir
        / f"{timestamp}_{config.run.name}_seed{config.run.seed:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(config_path: Path, config: Toy4Config, run_dir: Path) -> None:
    write_run_metadata_artifacts(
        config_path=config_path,
        config=config,
        run_dir=run_dir,
        toy="toy4",
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "policy_rule": config.policy.rule,
            "decision": config.policy.decision.model_dump(),
            "decision_mode": config.policy.decision.mode,
            "terminal_argmax_epochs": config.policy.decision.terminal_argmax_epochs,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_multiplier": config.game.multiplier,
            "domain_contribution_cost": config.game.contribution_cost,
            "domain_resource_enabled": config.environment.resource_enabled,
            "reputation": config.state.reputation.model_dump(),
            "mobility": config.state.mobility.model_dump(),
            "domain_grid_width": config.environment.grid_width,
            "domain_grid_height": config.environment.grid_height,
        },
    )


def validate_tensor_batched_backend_config(config: Toy4Config) -> None:
    if config.policy.rule != "neural_policy":
        raise ValueError("Toy 4 tensor_batched requires policy.rule='neural_policy'")
    if config.coordination.peer_rule not in {"none", "output_similarity"}:
        raise ValueError(
            "Toy 4 tensor_batched requires coordination.peer_rule to be "
            "'none' or 'output_similarity'"
        )
    if config.coordination.mixer not in {"none", "output_average"}:
        raise ValueError(
            "Toy 4 tensor_batched requires coordination.mixer to be "
            "'none' or 'output_average'"
        )
    if config.agents.optimizer.name != "adam":
        raise ValueError("Toy 4 tensor_batched requires Adam optimizer")
    model = config.agents.model
    if model.activation != "relu" or model.output_dim != 2:
        raise ValueError(
            "Toy 4 tensor_batched requires the standard one-hidden-layer ReLU MLP"
        )


@dataclass
class Toy4SpatialDomain(BinaryToyDomainBase):
    """Toy 4 adapter for the shared binary spatial lifecycle runner."""

    config: Toy4Config
    config_path: Path
    rng: np.random.Generator
    reputation_rng: np.random.Generator
    mobility_rng: np.random.Generator
    device: torch.device
    neural_update_backend: NeuralUpdateBackend = "loop"
    policy_cache: BatchedMLPPolicyCache | None = field(default=None, init=False)
    tensor_runtime: TensorBatchedMLPRuntime | None = field(default=None, init=False)
    _adam_state_cache: BatchedAdamStateCache | None = field(default=None, init=False)
    _uniform_neighbor_peer_count: int | None = field(default=None, init=False)
    _uniform_neighbor_peer_index: torch.Tensor | None = field(default=None, init=False)
    _neighbor_peer_index_cache: PeerIndexCache | None = field(default=None, init=False)
    _group_member_index: np.ndarray | None = field(default=None, init=False)
    _group_member_index_tensor: torch.Tensor | None = field(default=None, init=False)
    _pending_policy_cache_parameters: BatchedMLPParameters | None = field(
        default=None,
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

    micro_state_fields: ClassVar[list[str]] = TOY4_MICRO_STATE_FIELDS
    aggregate_fields: ClassVar[list[str]] = TOY4_AGGREGATE_FIELDS
    toy: ClassVar[str] = "toy4"

    def __post_init__(self) -> None:
        if self.neural_update_backend not in {"loop", "batched", "tensor_batched"}:
            raise ValueError(
                "Toy 4 neural_update_backend must be 'loop', 'batched', "
                "or 'tensor_batched'"
            )
        if self.neural_update_backend == "tensor_batched":
            validate_tensor_batched_backend_config(self.config)
        if (
            self.config.policy.domain.objective.uses_state_continuation()
            and self.neural_update_backend != "loop"
        ):
            raise ValueError(
                "Toy 4 state_continuation objective requires "
                "neural_update_backend='loop'"
            )
        if (
            self.config.policy.domain.basin_credit.enabled
            and self.neural_update_backend != "loop"
        ):
            raise ValueError("Toy 4 basin credit requires neural_update_backend='loop'")
        self.graph = build_public_goods_graph(self.config)
        self.neighbors = graph_neighbors(self.graph, self.config.agent_count)
        validate_peer_ids(self.neighbors, self.config.agent_count)
        self._uniform_neighbor_peer_count = uniform_peer_count(self.neighbors)
        self._uniform_neighbor_peer_index = (
            torch.as_tensor(self.neighbors, dtype=torch.long, device=self.device)
            if self._uniform_neighbor_peer_count is not None
            and self._uniform_neighbor_peer_count > 0
            else None
        )
        self._neighbor_peer_index_cache = (
            PeerIndexCache.from_peer_ids(self.neighbors, device=self.device)
            if self._uniform_neighbor_peer_count is None
            else None
        )
        self.groups = local_groups(self.neighbors)
        self._group_member_index = uniform_group_member_index(self.groups)
        self._group_member_index_tensor = (
            None
            if self._group_member_index is None
            else torch.as_tensor(
                self._group_member_index,
                dtype=torch.long,
                device=self.device,
            )
        )
        self.collapse_time: int | None = None

    def _uses_torch_state(self) -> bool:
        return (
            self.config.policy.rule == "neural_policy"
            and self.neural_update_backend == "tensor_batched"
        )

    def _observation_group_member_index(self) -> np.ndarray | torch.Tensor | None:
        if self._uses_torch_state():
            return self._group_member_index_tensor
        return self._group_member_index

    def can_reuse_static_peer_ids(self) -> bool:
        return (
            self.config.coordination.peer_rule == "none"
            and self.config.coordination.mixer == "output_average"
        )

    def can_defer_local_agent_sync(self) -> bool:
        return can_defer_static_output_average_agent_sync(
            peer_rule=self.config.coordination.peer_rule,
            mixer=self.config.coordination.mixer,
            alpha=self.config.coordination.alpha,
            uniform_neighbor_peer_count=self._uniform_neighbor_peer_count,
            peer_index_cache=self._neighbor_peer_index_cache,
            agent_count=self.config.agent_count,
        )

    def can_defer_local_adam_state_sync(self) -> bool:
        return self.can_defer_local_agent_sync()

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
        resource_level = (
            self.config.environment.resource_initial
            if self.config.environment.resource_enabled
            else self.config.environment.resource_carrying_capacity
        )
        agents = (
            create_agents(self.config, device=self.device)
            if self.config.policy.rule == "neural_policy"
            else []
        )
        if agents and self.neural_update_backend == "tensor_batched":
            self.tensor_runtime = TensorBatchedMLPRuntime.from_agents(
                agents,
                device=self.device,
            )
            self.policy_cache = None
        elif agents:
            self.refresh_policy_cache(agents)
        else:
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
                else torch.zeros(
                    self.config.agent_count,
                    dtype=torch.float64,
                    device=self.device,
                )
            )
            payoffs = self._compute_payoffs(actions, resource_level)
            payoff_ema = torch.zeros(
                self.config.agent_count,
                dtype=torch.float64,
                device=self.device,
            )
            previous_payoff_ema = torch.zeros(
                self.config.agent_count,
                dtype=torch.float64,
                device=self.device,
            )
        else:
            actions = actions_np
            reputation = (
                actions_np.astype(np.float64)
                if self.config.state.reputation.enabled
                else np.zeros(self.config.agent_count, dtype=np.float64)
            )
            payoffs = self._compute_payoffs(actions, resource_level)
            payoff_ema = np.zeros(self.config.agent_count, dtype=np.float64)
            previous_payoff_ema = np.zeros(self.config.agent_count, dtype=np.float64)
        return BinarySpatialState(
            actions=actions,
            payoffs=payoffs,
            payoff_ema=payoff_ema,
            previous_payoff_ema=previous_payoff_ema,
            reputation=reputation,
            agents=agents,
            extras={"resource_level": resource_level},
        )

    def initial_step_result(
        self,
        state: BinarySpatialState,
    ) -> BinaryPolicyStepResult:
        if self.config.policy.rule == "neural_policy":
            agents = state.agents or []
            observations = build_observations(
                actions=state.actions,
                payoffs=state.payoffs,
                payoff_ema=state.payoff_ema,
                groups=self.groups,
                resource_level=state.extras["resource_level"],
                config=self.config,
                device=self.device,
                reputation=state.reputation,
                group_member_index=self._observation_group_member_index(),
            )
            initial_probs = self.collect_policy_probs(
                agents,
                observations,
                temperature=self.config.policy.temperature,
            )
        else:
            initial_probs = contribution_probs_to_policy_tensor(
                to_numpy_view(state.actions, dtype=np.float64),
                device=self.device,
            )
        initial_peer_ids = select_peers_by_output_similarity(
            neighbors=self.neighbors,
            contribution_probs=initial_probs[:, 1].detach().cpu().numpy(),
            peer_rule=self.config.coordination.peer_rule,
            threshold=self.config.coordination.threshold,
        )
        initial_peer_ids = peer_ids_for_mixer(
            initial_peer_ids,
            mixer=self.config.coordination.mixer,
            agent_count=self.config.agent_count,
        )
        return BinaryPolicyStepResult(
            pre_revision_probs=initial_probs,
            post_local_probs=initial_probs,
            post_social_probs=initial_probs,
            local_losses=[0.0 for _ in range(self.config.agent_count)],
            social_losses=[0.0 for _ in range(self.config.agent_count)],
            peer_ids=initial_peer_ids,
            revision_mask=np.zeros(self.config.agent_count, dtype=bool),
            mobility_result=MobilityStepResult.none(self.config.agent_count),
            realized_revision_rate=0.0,
            extras={"revised_local_losses": []},
        )

    def build_step_context(
        self,
        epoch: int,
        state: BinarySpatialState,
        revision_mask: np.ndarray,
    ) -> BinaryStepContext:
        return BinaryStepContext(
            epoch=epoch,
            revision_mask=revision_mask,
            extras={"resource_level": float(state.extras["resource_level"])},
        )

    def basin_credit_diagnostics_for_actions(
        self,
        *,
        actions: StateArray,
        payoffs: StateArray,
        resource_level: float,
        revision_mask: np.ndarray,
        action_probabilities: np.ndarray,
    ) -> BasinCreditDiagnostics | None:
        basin_credit = self.config.policy.domain.basin_credit
        if not basin_credit.enabled:
            return None

        action_values = to_numpy_view(actions, dtype=np.int64)
        payoff_values = to_numpy_view(payoffs, dtype=np.float64)
        probability_values = np.asarray(action_probabilities, dtype=np.float64)
        target_payoff = toy4_target_basin_payoff(self.config)
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
            counterfactual_resource = update_resource_level(
                resource_level,
                counterfactual_actions,
                self.config,
            )
            counterfactual_payoffs = self._compute_payoffs(
                counterfactual_actions,
                counterfactual_resource,
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
        actions: StateArray,
        payoffs: StateArray,
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
        action_values = to_numpy_view(actions, dtype=np.int64)
        prototype_advantage = (
            None
            if prototype_diagnostics is None
            else np.where(
                action_values == 1,
                prototype_diagnostics.selected_action_credit,
                -prototype_diagnostics.selected_action_credit,
            )
        )
        return learned_basin_runtime_diagnostics(
            self._learned_basin_critic_bundle,
            actions=action_values,
            payoffs=to_numpy_view(payoffs, dtype=np.float64),
            action_probabilities=action_probabilities,
            target_payoff=toy4_target_basin_payoff(self.config),
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
        config = self.config
        actions = state.actions
        payoffs = state.payoffs
        resource_level = float(context.extras["resource_level"])
        local_losses = [0.0 for _ in range(config.agent_count)]
        agents = state.agents or []

        if config.policy.rule == "neural_policy":
            with timed_context_stage(context, "build_observations"):
                observations = build_observations(
                    actions=actions,
                    payoffs=payoffs,
                    payoff_ema=state.payoff_ema,
                    groups=self.groups,
                    resource_level=resource_level,
                    config=config,
                    device=self.device,
                    reputation=state.reputation,
                    group_member_index=self._observation_group_member_index(),
            )
            raw_policy_probs: torch.Tensor | None = None

            def collect_pre_policy_probs(
                agents_arg: list[NeuralPublicGoodsAgent],
                observations_arg: torch.Tensor,
                *,
                temperature: float,
            ) -> torch.Tensor:
                del temperature
                nonlocal raw_policy_probs
                raw_policy_probs = self.collect_policy_probs(
                    agents_arg,
                    observations_arg,
                    temperature=1.0,
                )
                if config.policy.temperature == 1.0:
                    return raw_policy_probs
                return self.collect_policy_probs(
                    agents_arg,
                    observations_arg,
                    temperature=config.policy.temperature,
                )

            def collect_post_policy_probs(
                agents_arg: list[NeuralPublicGoodsAgent],
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
            decision_bootstrap_diagnostics = None
            decision_mode = toy4_decision_mode_for_epoch(config, context.epoch)
            diagnostic_reputation_rng = np.random.default_rng(
                int(config.run.seed) + 1_000_003 * int(context.epoch) + 31
            )
            alignment_teacher_pre_probs = reputation_imitation_cooperation_probs(
                actions=to_numpy_view(state.actions, dtype=np.int64),
                reputation=to_numpy_view(
                    state.reputation,
                    dtype=np.float64,
                ),
                peer_ids=self.neighbors,
                revision_mask=np.ones(config.agent_count, dtype=bool),
                rng=diagnostic_reputation_rng,
                params=reputation_params_from_config(config),
            )
            def build_decision_action_probs(
                pre_revision_probs: torch.Tensor,
            ) -> torch.Tensor:
                nonlocal decision_bootstrap_diagnostics
                candidate_probs = decision_action_probs(
                    pre_revision_probs,
                    config,
                    mode=decision_mode,
                )
                if config.policy.domain.bootstrap.decision_enabled:
                    scheduled_weight = domain_decision_bootstrap_weight(
                        config.policy.domain.bootstrap,
                        context.epoch,
                    )
                    teacher_probs = np.asarray([], dtype=np.float64)
                    if scheduled_weight > 0.0:
                        teacher_probs = reputation_imitation_cooperation_probs(
                            actions=to_numpy_view(state.actions, dtype=np.int64),
                            reputation=to_numpy_view(
                                state.reputation,
                                dtype=np.float64,
                            ),
                            peer_ids=self.neighbors,
                            revision_mask=np.ones(config.agent_count, dtype=bool),
                            rng=self.reputation_rng,
                            params=reputation_params_from_config(config),
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
                        bootstrapped_contribution = torch.as_tensor(
                            bootstrapped_probabilities,
                            dtype=candidate_probs.dtype,
                            device=candidate_probs.device,
                        )
                        candidate_probs = torch.stack(
                            (
                                1.0 - bootstrapped_contribution,
                                bootstrapped_contribution,
                            ),
                            dim=1,
                        )
                return self.apply_precommitment_decision_feedback(
                    state,
                    candidate_probs,
                )

            def sample_policy_actions(action_probs: torch.Tensor) -> StateArray:
                nonlocal actions, payoffs, resource_level
                selected_actions = select_actions_from_probs(
                    current_actions=actions,
                    action_probs=action_probs,
                    revision_mask=context.revision_mask,
                    rng=self.rng,
                    mode=decision_mode,
                )
                resource_level = update_resource_level(
                    resource_level,
                    selected_actions,
                    config,
                )
                payoffs = self._compute_payoffs(selected_actions, resource_level)
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
                nonlocal actions, payoffs, resource_level
                probabilities = normalize_binary_revision_probabilities(
                    revision_probs
                )
                probability_array = (
                    probabilities.detach().cpu().numpy()
                    if isinstance(probabilities, torch.Tensor)
                    else np.asarray(probabilities, dtype=np.float64)
                )
                choices = np.full(len(current_actions), REVISION_STAY, dtype=np.int64)
                active_indices = np.flatnonzero(context.revision_mask)
                if decision_mode == "sampled":
                    for agent_id in active_indices:
                        choices[int(agent_id)] = int(
                            self.rng.choice(3, p=probability_array[int(agent_id)])
                        )
                elif decision_mode == "argmax":
                    choices[active_indices] = np.argmax(
                        probability_array[active_indices],
                        axis=1,
                    )
                else:
                    raise ValueError(f"Unsupported decision mode: {decision_mode}")
                selected_action_values = apply_binary_revision_choices(
                    current_actions,
                    choices,
                )
                selected_actions = actions_like_state(selected_action_values)
                resource_level = update_resource_level(
                    resource_level,
                    selected_actions,
                    config,
                )
                payoffs = self._compute_payoffs(selected_actions, resource_level)
                actions = selected_actions
                return choices
            distill_diagnostics = None
            distill_teacher_probs = np.asarray([], dtype=np.float64)
            distill_teacher_post_action_probs = np.asarray([], dtype=np.float64)
            distill_weight = 0.0
            objective_components: StateContinuationComponents | None = None
            training_components: StateContinuationComponents | None = None
            bootstrap_diagnostics = None
            basin_diagnostics = None
            selected_action_advantages: np.ndarray | None = None
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
                nonlocal bootstrap_diagnostics, selected_action_advantages
                nonlocal alignment_base_losses, alignment_distill_losses
                nonlocal alignment_base_grad_norms, alignment_distill_grad_norms
                nonlocal alignment_grad_cosines, alignment_distill_candidate_mask
                nonlocal alignment_distill_stable_teacher_mask
                nonlocal alignment_distill_gradient_gate_mask
                nonlocal alignment_distill_applied_mask
                actions = updated_actions
                if config.policy.domain.bootstrap.distill_enabled:
                    if raw_policy_probs is None:
                        raise RuntimeError("Toy 4 raw policy readout is missing")
                    distill_weight = domain_distill_bootstrap_weight(
                        config.policy.domain.bootstrap,
                        context.epoch,
                    )
                    if distill_weight > 0.0:
                        distill_teacher_probs = reputation_imitation_cooperation_probs(
                            actions=to_numpy_view(state.actions, dtype=np.int64),
                            reputation=to_numpy_view(
                                state.reputation,
                                dtype=np.float64,
                            ),
                            peer_ids=self.neighbors,
                            revision_mask=np.ones(config.agent_count, dtype=bool),
                            rng=self.reputation_rng,
                            params=reputation_params_from_config(config),
                        )
                        if config.policy.domain.bootstrap.distill_stable_teacher_only:
                            distill_teacher_post_action_probs = (
                                reputation_imitation_cooperation_probs(
                                    actions=to_numpy_view(actions, dtype=np.int64),
                                    reputation=to_numpy_view(
                                        state.reputation,
                                        dtype=np.float64,
                                    ),
                                    peer_ids=self.neighbors,
                                    revision_mask=np.ones(
                                        config.agent_count,
                                        dtype=bool,
                                    ),
                                    rng=self.reputation_rng,
                                    params=reputation_params_from_config(config),
                                )
                            )
                    distill_diagnostics = (
                        domain_distill_bootstrap_diagnostic_components(
                            neural_probabilities=(
                                raw_policy_probs[:, 1].detach().cpu().numpy()
                            ),
                            teacher_probabilities=distill_teacher_probs,
                            realized_actions=to_numpy_view(actions, dtype=np.int64),
                            bootstrap=config.policy.domain.bootstrap,
                            epoch=context.epoch,
                        )
                    )
                if config.policy.domain.objective.uses_state_continuation():
                    objective_components = contribution_advantage_components(
                        actions=actions,
                        groups=self.groups,
                        config=config,
                        resource_level=resource_level,
                    )
                    training_components = objective_components
                    if config.policy.domain.bootstrap.enabled:
                        scheduled_weight = domain_bootstrap_weight(
                            config.policy.domain.bootstrap,
                            context.epoch,
                        )
                        if scheduled_weight > 0.0:
                            teacher_probs = reputation_imitation_cooperation_probs(
                                actions=to_numpy_view(actions, dtype=np.int64),
                                reputation=to_numpy_view(
                                    state.reputation,
                                    dtype=np.float64,
                                ),
                                peer_ids=self.neighbors,
                                revision_mask=np.ones(config.agent_count, dtype=bool),
                                rng=self.reputation_rng,
                                params=reputation_params_from_config(config),
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
                                teacher_probabilities=np.asarray([], dtype=np.float64),
                                bootstrap=config.policy.domain.bootstrap,
                                epoch=context.epoch,
                            )
                    action_values = to_numpy_view(actions, dtype=np.int64)
                    selected_action_advantages = np.where(
                        action_values == 1,
                        training_components.effective,
                        -training_components.effective,
                    )
                local_training_losses = [0.0 for _ in range(config.agent_count)]
                if config.policy.learning_enabled:
                    if self.neural_update_backend == "tensor_batched":
                        local_report = run_tensor_runtime_policy_gradient_local_update(
                            runtime=self._require_tensor_runtime(agents),
                            observations=observations,
                            actions=updated_actions,
                            advantages=(
                                (payoffs - state.payoff_ema)
                                / payoff_normalizer(config)
                            ),
                            active_agent_ids=[
                                int(agent_id)
                                for agent_id in np.flatnonzero(context.revision_mask)
                            ],
                            entropy_beta=config.environment.entropy_beta,
                            timing_context=context,
                        )
                        if local_report.update_result is None:
                            raise RuntimeError(
                                "tensor local update adapter did not produce a result",
                            )
                        update_result = local_report.update_result
                        local_training_losses = update_result.losses
                    elif self.neural_update_backend == "batched":
                        defer_agent_sync = self.can_defer_local_agent_sync()
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
                                actions=updated_actions,
                                payoffs=payoffs,
                                payoff_baseline=state.payoff_ema,
                                revision_mask=context.revision_mask,
                                config=config,
                                parameters=update_parameters,
                                adam_state_cache=self._require_adam_state_cache(agents),
                                synchronize_model_parameters=not defer_agent_sync,
                                synchronize_optimizer_states=not defer_agent_sync,
                                timing_context=context,
                            )
                        local_training_losses = update_result.losses
                        self._pending_policy_cache_parameters = (
                            update_result.updated_parameters
                        )
                        if not update_result.used_batched_optimizer:
                            self._adam_state_cache = None
                    else:
                        update_agent_ids = (
                            range(config.agent_count)
                            if (
                                distill_weight > 0.0
                                and config.policy.domain.bootstrap.distill_scope
                                == "all"
                            )
                            else np.flatnonzero(context.revision_mask)
                        )
                        stable_teacher_mask = np.ones(config.agent_count, dtype=bool)
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
                        for agent_id in update_agent_ids:
                            local_advantage = (
                                None
                                if selected_action_advantages is None
                                else float(selected_action_advantages[int(agent_id)])
                            )
                            revised = bool(context.revision_mask[int(agent_id)])
                            use_distill = (
                                distill_weight > 0.0
                                and distill_teacher_probs.size > 0
                                and (
                                    config.policy.domain.bootstrap.distill_scope
                                    == "all"
                                    or revised
                                )
                            )
                            if (
                                use_distill
                                and alignment_base_losses.size == 0
                            ):
                                alignment_base_losses = np.full(
                                    config.agent_count,
                                    np.nan,
                                    dtype=np.float64,
                                )
                                alignment_distill_losses = np.full(
                                    config.agent_count,
                                    np.nan,
                                    dtype=np.float64,
                                )
                                alignment_base_grad_norms = np.full(
                                    config.agent_count,
                                    np.nan,
                                    dtype=np.float64,
                                )
                                alignment_distill_grad_norms = np.full(
                                    config.agent_count,
                                    np.nan,
                                    dtype=np.float64,
                                )
                                alignment_grad_cosines = np.full(
                                    config.agent_count,
                                    np.nan,
                                    dtype=np.float64,
                                )
                                alignment_distill_candidate_mask = np.zeros(
                                    config.agent_count,
                                    dtype=bool,
                                )
                                alignment_distill_stable_teacher_mask = np.zeros(
                                    config.agent_count,
                                    dtype=bool,
                                )
                                alignment_distill_gradient_gate_mask = np.zeros(
                                    config.agent_count,
                                    dtype=bool,
                                )
                                alignment_distill_applied_mask = np.zeros(
                                    config.agent_count,
                                    dtype=bool,
                                )
                            if use_distill:
                                alignment_distill_candidate_mask[int(agent_id)] = True
                            stable_teacher = bool(stable_teacher_mask[int(agent_id)])
                            if use_distill and stable_teacher:
                                alignment_distill_stable_teacher_mask[
                                    int(agent_id)
                                ] = True
                                (
                                    alignment_base_losses[int(agent_id)],
                                    alignment_distill_losses[int(agent_id)],
                                    alignment_base_grad_norms[int(agent_id)],
                                    alignment_distill_grad_norms[int(agent_id)],
                                    alignment_grad_cosines[int(agent_id)],
                                ) = teacher_distill_gradient_conflict_values(
                                    agent=agents[int(agent_id)],
                                    observation=observations[int(agent_id)],
                                    action=int(updated_actions[int(agent_id)]),
                                    payoff=float(payoffs[int(agent_id)]),
                                    payoff_baseline=float(
                                        state.payoff_ema[int(agent_id)]
                                    ),
                                    config=config,
                                    advantage=local_advantage,
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
                                                [
                                                    alignment_grad_cosines[
                                                        int(agent_id)
                                                    ]
                                                ],
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
                            local_training_losses[int(agent_id)] = (
                                train_neural_local_policy(
                                    agent=agents[int(agent_id)],
                                    observation=observations[int(agent_id)],
                                    action=int(updated_actions[int(agent_id)]),
                                    payoff=float(payoffs[int(agent_id)]),
                                    payoff_baseline=float(
                                        state.payoff_ema[int(agent_id)]
                                    ),
                                    config=config,
                                    advantage=local_advantage,
                                    teacher_distill_probability=(
                                        float(distill_teacher_probs[int(agent_id)])
                                        if use_distill
                                        else None
                                    ),
                                    teacher_distill_weight=(
                                        distill_weight if use_distill else 0.0
                                    ),
                                    base_loss_weight=(
                                        0.0
                                        if (
                                            config.policy.domain.basin_credit.enabled
                                            and not basin_credit_preserves_objective(
                                                config.policy.domain.basin_credit
                                            )
                                        )
                                        else 1.0 if revised else 0.0
                                    ),
                                )
                            )
                return local_training_losses

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
                    agents_arg: list[NeuralPublicGoodsAgent],
                    observations_arg: torch.Tensor,
                    current_actions_arg: np.ndarray,
                ) -> dict[str, object]:
                    del agents_arg, observations_arg, current_actions_arg
                    return {
                        "source": config.coordination.revision_operator_source,
                        "revision_mask": context.revision_mask.copy(),
                    }

                def collect_revision_probs(
                    agents_arg: list[NeuralPublicGoodsAgent],
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
                    agents_arg: list[NeuralPublicGoodsAgent],
                    observations_arg: torch.Tensor,
                    current_actions_arg: np.ndarray,
                    revision_signals: dict[str, object],
                    *,
                    temperature: float,
                ) -> torch.Tensor:
                    del agents_arg, observations_arg, revision_signals, temperature
                    if revision_decision_action_probs is None:
                        raise RuntimeError("Toy 4 revision decision probabilities missing")
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
                    raise RuntimeError("Toy 4 revision decision probabilities missing")
                action_probs = revision_decision_action_probs
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
                learning_result = run_binary_policy_learning_step(
                    agents=agents,
                    observations=observations,
                    temperature=config.policy.temperature,
                    collect_policy_probs=collect_pre_policy_probs,
                    decision_action_probs=build_decision_action_probs,
                    sample_actions=sample_policy_actions,
                    local_update=commit_local_update,
                    refresh_policy_cache=self.refresh_policy_cache,
                    post_collect_policy_probs=collect_post_policy_probs,
                    context=context,
                    unit_type=BinaryPolicyLearningUnit,
                )
                pre_revision_probs = learning_result.pre_revision_probs
                action_probs = learning_result.decision_action_probs
                actions = learning_result.actions_after_revision
                local_losses = learning_result.local_losses
                post_local_probs = learning_result.post_local_probs
            if raw_policy_probs is None:
                raise RuntimeError("Toy 4 raw policy readout is missing")
            revised_local_losses = loss_values_at(
                local_losses,
                np.flatnonzero(context.revision_mask),
            )
            return BinaryLocalStepResult(
                pre_revision_probs=pre_revision_probs,
                candidate_action_probs=action_probs[:, 1].detach().cpu().numpy(),
                post_local_probs=post_local_probs,
                local_losses=local_losses,
                social_mode="policy_distill",
                actions_after_revision=actions,
                extras={
                    "decision_action_probs": action_probs,
                    "revised_local_losses": revised_local_losses,
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
                        "pre_local": raw_policy_probs[:, 1].detach().cpu().numpy(),
                        "post_local": binary_action_probs_from_policy(
                            post_local_probs
                        ),
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
                    "_resource_level_after_actions": resource_level,
                    "_payoffs_after_actions": payoffs,
                },
            )
        elif config.policy.rule in {"imitation", "reputation_imitation"}:
            pre_revision_probs = contribution_probs_to_policy_tensor(
                to_numpy_view(actions, dtype=np.float64),
                device=self.device,
            )
            if config.policy.rule == "imitation":
                candidate_probs = imitation_candidate_probabilities(
                    actions=actions,
                    payoffs=payoffs,
                    neighbors=self.neighbors,
                    revision_mask=context.revision_mask,
                    selection_strength=config.policy.selection_strength,
                )
            else:
                candidate_probs = reputation_imitation_cooperation_probs(
                    actions=actions,
                    reputation=state.reputation,
                    peer_ids=self.neighbors,
                    revision_mask=context.revision_mask,
                    rng=self.reputation_rng,
                    params=reputation_params_from_config(config),
                )
            post_local_probs = contribution_probs_to_policy_tensor(
                candidate_probs,
                device=self.device,
            )
            return BinaryLocalStepResult(
                pre_revision_probs=pre_revision_probs,
                candidate_action_probs=candidate_probs,
                post_local_probs=post_local_probs,
                local_losses=local_losses,
                social_mode="probability_mix",
                extras={
                    "decision_action_probs": post_local_probs,
                    "revised_local_losses": [
                        0.0 for _ in np.flatnonzero(context.revision_mask)
                    ],
                },
            )
        else:
            raise ValueError(
                f"Unsupported Toy 4 update rule: {config.policy.rule}"
            )

    def policy_tensor_from_action_probs(
        self,
        action_probs: np.ndarray,
        device_like: torch.Tensor,
    ) -> torch.Tensor:
        return contribution_probs_to_policy_tensor(
            action_probs,
            device=device_like.device,
        )

    def sample_actions(
        self,
        state: BinarySpatialState,
        action_probs: np.ndarray,
        revision_mask: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> StateArray:
        del context, local_result
        return select_actions_from_probs(
            current_actions=state.actions,
            action_probs=contribution_probs_to_policy_tensor(
                action_probs,
                device=self.device,
            ),
            revision_mask=revision_mask,
            rng=self.rng,
        )

    def collect_policy_probs(
        self,
        agents: list[NeuralPublicGoodsAgent],
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

    def refresh_policy_cache(self, agents: list[NeuralPublicGoodsAgent]) -> None:
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
        agents: list[NeuralPublicGoodsAgent],
    ) -> BatchedMLPPolicyCache:
        if self.policy_cache is None:
            self.refresh_policy_cache(agents)
        if self.policy_cache is None:
            raise RuntimeError("Toy 4 policy cache is not initialized")
        return self.policy_cache

    def _require_tensor_runtime(
        self,
        agents: list[NeuralPublicGoodsAgent],
    ) -> TensorBatchedMLPRuntime:
        if self.tensor_runtime is None:
            self.tensor_runtime = TensorBatchedMLPRuntime.from_agents(
                agents,
                device=self.device,
            )
        return self.tensor_runtime

    def _require_adam_state_cache(
        self,
        agents: list[NeuralPublicGoodsAgent],
    ) -> BatchedAdamStateCache:
        if self._adam_state_cache is None:
            self._adam_state_cache = BatchedAdamStateCache.from_agents(
                agents,
                device=self.device,
            )
        return self._adam_state_cache

    def flush_tensor_runtime_to_agents(
        self,
        agents: list[NeuralPublicGoodsAgent],
    ) -> None:
        if self.tensor_runtime is not None and agents:
            self.tensor_runtime.flush_to_agents(agents)

    def write_summary(
        self,
        run_dir: Path,
        final_row: dict[str, object],
        state: BinarySpatialState,
    ) -> BinaryToyResult:
        self.flush_tensor_runtime_to_agents(list(state.agents or []))
        if self._basin_transition_samples:
            write_basin_transition_samples(
                run_dir,
                annotate_terminal_outcomes(
                    self._basin_transition_samples,
                    final_mean_payoff=float(final_row["mean_payoff"]),
                    target_payoff=toy4_target_basin_payoff(self.config),
                ),
            )
        return super().write_summary(run_dir, final_row, state)

    def distill_policy(
        self,
        agents: list[NeuralPublicGoodsAgent],
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
                loss_mode="cross_entropy",
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
        return run_binary_output_distribution_distillation(
            agents=agents,
            observations=observations,
            peer_ids=peer_ids,
            alpha=alpha,
            previous_probs=previous_probs,
            logits_fn=lambda agent, agent_id, observed: agent.model(observed[agent_id]),
            loss_mode="cross_entropy",
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
            resource_level = float(local_result.extras["_resource_level_after_actions"])
            payoffs = local_result.extras["_payoffs_after_actions"]
        else:
            resource_level = update_resource_level(
                float(context.extras["resource_level"]),
                actions,
                self.config,
            )
            payoffs = self._compute_payoffs(actions, resource_level)
        state.actions = actions
        state.payoffs = payoffs
        state.extras["resource_level"] = resource_level
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
        resource_level = float(state.extras["resource_level"])
        updates: dict[str, object] = {"extras": {}}
        if bool(np.any(mobility_result.moved)):
            payoffs = self._compute_payoffs(state.actions, resource_level)
            state.payoffs = payoffs
            if self.config.policy.rule == "neural_policy":
                observations = build_observations(
                    actions=state.actions,
                    payoffs=payoffs,
                    payoff_ema=state.payoff_ema,
                    groups=self.groups,
                    resource_level=resource_level,
                    config=self.config,
                    device=self.device,
                    reputation=state.reputation,
                    group_member_index=self._observation_group_member_index(),
                )
                self.refresh_policy_cache(list(state.agents or []))
                updates["post_social_probs"] = self.collect_policy_probs(
                    list(state.agents or []),
                    observations,
                    temperature=self.config.policy.temperature,
                )
        if (
            self.config.environment.resource_enabled
            and self.collapse_time is None
            and resource_level <= self.config.environment.resource_collapse_threshold
        ):
            self.collapse_time = context.epoch
        alignment_inputs = local_result.extras.get("_domain_teacher_alignment_inputs")
        if isinstance(alignment_inputs, dict):
            diagnostic_reputation_rng = np.random.default_rng(
                int(self.config.run.seed) + 1_000_003 * int(context.epoch) + 37
            )
            teacher_post_action = reputation_imitation_cooperation_probs(
                actions=to_numpy_view(state.actions, dtype=np.int64),
                reputation=to_numpy_view(
                    state.reputation,
                    dtype=np.float64,
                ),
                peer_ids=self.neighbors,
                revision_mask=np.ones(self.config.agent_count, dtype=bool),
                rng=diagnostic_reputation_rng,
                params=reputation_params_from_config(self.config),
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
                    actions = to_numpy_view(state.actions, dtype=np.int64)
                    payoffs = to_numpy_view(state.payoffs, dtype=np.float64)
                    payoff_ema = to_numpy_view(state.payoff_ema, dtype=np.float64)
                    for agent_id in np.flatnonzero(replay_diagnostics.applied_mask):
                        train_neural_local_policy(
                            agent=agents[int(agent_id)],
                            observation=observations[int(agent_id)],
                            action=int(actions[int(agent_id)]),
                            payoff=float(payoffs[int(agent_id)]),
                            payoff_baseline=float(payoff_ema[int(agent_id)]),
                            config=self.config,
                            advantage=0.0,
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
        del local_result, social_result, mobility_result
        config = self.config
        basin_credit = config.policy.domain.basin_credit
        if config.policy.rule != "neural_policy" or not basin_credit.enabled:
            return {}

        actions = to_numpy_view(state.actions, dtype=np.int64)
        payoffs = to_numpy_view(state.payoffs, dtype=np.float64)
        resource_level_before_actions = float(context.extras["resource_level"])
        resource_level = float(state.extras["resource_level"])
        action_probabilities = binary_action_probs_from_policy(post_social_probs)
        basin_training_mask = basin_credit_training_candidate_mask(
            agent_count=len(actions),
            revision_mask=context.revision_mask,
            training_scope=basin_credit.training_scope,
        )
        basin_diagnostics = self.basin_credit_diagnostics_for_actions(
            actions=actions,
            payoffs=payoffs,
            resource_level=resource_level_before_actions,
            revision_mask=basin_training_mask,
            action_probabilities=action_probabilities,
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

        objective_components = contribution_advantage_components(
            actions=actions,
            groups=self.groups,
            config=config,
            resource_level=resource_level,
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
                target_payoff=toy4_target_basin_payoff(config),
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
                    "domain_resource_enabled": config.environment.resource_enabled,
                    "domain_resource_level": resource_level,
                    "domain_resource_fraction": resource_fraction(
                        config,
                        resource_level,
                    ),
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
                payoff_ema=state.payoff_ema,
                groups=self.groups,
                resource_level=resource_level,
                config=config,
                device=self.device,
                reputation=state.reputation,
                group_member_index=self._observation_group_member_index(),
            )
        with timed_context_stage(context, "post_social_basin_training"):
            replay_weight = basin_training_mask * learned_signal.replay_weight
            candidate_agent_ids = np.flatnonzero(replay_weight > 0.0)
            for _ in range(effective_training_passes):
                for agent_id in candidate_agent_ids:
                    selected_advantage = (
                        training_components.effective[int(agent_id)]
                        if actions[int(agent_id)] == 1
                        else -training_components.effective[int(agent_id)]
                    )
                    train_neural_local_policy(
                        agent=agents[int(agent_id)],
                        observation=observations[int(agent_id)],
                        action=int(actions[int(agent_id)]),
                        payoff=float(payoffs[int(agent_id)]),
                        payoff_baseline=float(state.payoff_ema[int(agent_id)]),
                        config=config,
                        advantage=float(selected_advantage),
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

    def _compute_payoffs(
        self,
        actions: StateArray,
        resource_level: float,
    ) -> StateArray:
        return compute_public_goods_payoffs(
            actions=actions,
            groups=self.groups,
            multiplier=self.config.game.multiplier,
            contribution_cost=self.config.game.contribution_cost,
            resource_multiplier=resource_fraction(self.config, resource_level),
            group_member_index=self._observation_group_member_index(),
        )

    def domain_aggregate_fields(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, object]:
        del epoch
        resource_level = float(state.extras["resource_level"])
        payoffs = to_numpy_view(state.payoffs, dtype=np.float64)
        contributor_components, largest_contributor_fraction = (
            contributor_cluster_metrics(
                state.actions,
                self.graph,
            )
        )
        return {
            "domain_payoff_variance": float(np.var(payoffs)),
            "domain_payoff_gini": payoff_gini(payoffs),
            "domain_resource_enabled": self.config.environment.resource_enabled,
            "domain_resource_level": resource_level,
            "domain_resource_fraction": resource_fraction(
                self.config,
                resource_level,
            ),
            "domain_collapse_time": (
                "" if self.collapse_time is None else self.collapse_time
            ),
            "domain_action_components": contributor_components,
            "domain_largest_action_cluster_fraction": largest_contributor_fraction,
            "domain_exploitation_index": exploitation_index(
                state.actions,
                payoffs,
            ),
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
        local_rates = local_contribution_rates(state.actions, self.groups)
        group_means = group_payoff_means(state.payoffs, self.groups)
        return {
            "domain_local_action_rate": float(local_rates[agent_id]),
            "domain_group_payoff_mean": float(group_means[agent_id]),
            "domain_resource_level": float(state.extras["resource_level"]),
            **domain_learning_micro_fields(
                extras=step_result.extras,
                actions=to_numpy_view(state.actions, dtype=np.int64),
                agent_id=agent_id,
            ),
        }


def run_toy4(
    config: Toy4Config,
    config_path: Path,
    timing_rows: list[dict[str, object]] | None = None,
    neural_update_backend: NeuralUpdateBackendRequest | None = None,
) -> BinaryToyResult:
    """Run Toy 4 from a validated config."""

    if config.policy.rule == "neural_policy":
        expected_input_dim = neural_observation_input_dim(config)
        if config.agents.model.input_dim != expected_input_dim:
            raise ValueError(
                "Toy 4 neural_policy expects "
                f"model.input_dim={expected_input_dim}"
            )
        if config.agents.model.output_dim != 2:
            raise ValueError("Toy 4 neural_policy expects model.output_dim=2")

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    device = resolve_torch_device(config.simulation.device)
    backend = resolve_neural_update_backend(
        (
            config.policy.neural_update_backend
            if neural_update_backend is None
            else neural_update_backend
        ),
        device=device,
        agent_count=config.agent_count,
    )
    domain = Toy4SpatialDomain(
        config=config,
        config_path=config_path,
        rng=rng,
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
