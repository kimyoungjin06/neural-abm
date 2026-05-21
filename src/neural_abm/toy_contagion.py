"""Toy 5: contagion and threshold adoption runner."""

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
from neural_abm.config import Toy5Config
from neural_abm.losses import LossVector
from neural_abm.mobility import MobilityStepResult
from neural_abm.reputation import (
    ReputationParams,
    reputation_imitation_cooperation_probs,
    reputation_observation_extra_dim,
    reputation_observation_features,
)
from neural_abm.readiness import binary_peer_aggregate_values
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
from neural_abm.unit import (
    ObservationSpec,
    SocialMessageSpec,
)


TOY5_MICRO_STATE_FIELDS = [
    *BINARY_MICRO_COMMON_FIELDS,
    "domain_threshold",
    "domain_threshold_group",
    "domain_neighbor_action_rate",
    "domain_repeated_exposure_count",
    "domain_degree",
    "domain_utility_proxy",
    "domain_newly_adopted",
]


TOY5_AGGREGATE_FIELDS = [
    *BINARY_AGGREGATE_COMMON_FIELDS,
    "domain_threshold_mode",
    "domain_cascade_size",
    "domain_non_adoption_rate",
    "domain_time_to_50_action",
    "domain_failed_cascade",
    "domain_action_cluster_count",
    "domain_largest_action_cluster_fraction",
    "domain_mean_neighbor_action_rate",
    "domain_mean_repeated_exposure_count",
    "domain_homogeneous_action_rate",
    "domain_low_threshold_action_rate",
    "domain_high_threshold_action_rate",
]

class AdoptionMLP(nn.Module):
    """Small binary adoption policy network for Toy 5."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


@dataclass
class NeuralAdoptionAgent:
    agent_id: int
    model: AdoptionMLP
    optimizer: torch.optim.Optimizer

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="adoption_observation",
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
        raise NotImplementedError("Toy 5 local update requires utility context")

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


def make_model(config: Toy5Config) -> AdoptionMLP:
    model_config = config.agents.model
    model = AdoptionMLP(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=model_config.output_dim,
    )
    policy_prior = config.agents.policy_prior_action_probability
    if policy_prior is not None:
        initialize_policy_head_prior(model, policy_prior)
    return model


def initialize_policy_head_prior(
    model: AdoptionMLP,
    action_probability: float,
) -> None:
    if model.fc2.out_features != 2:
        raise ValueError("Toy 5 policy prior expects model.output_dim=2")
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


def make_optimizer(model: torch.nn.Module, config: Toy5Config) -> torch.optim.Optimizer:
    if config.agents.optimizer.name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=config.agents.optimizer.learning_rate,
        )
    raise ValueError(f"Unsupported optimizer: {config.agents.optimizer.name}")


def create_agents(config: Toy5Config, device: torch.device) -> list[NeuralAdoptionAgent]:
    base_state = None
    if config.agents.init_mode == "same_init":
        torch.manual_seed(config.run.seed)
        base_state = clone_state_dict(make_model(config))

    agents: list[NeuralAdoptionAgent] = []
    for agent_id in range(config.agents.count):
        if config.agents.init_mode == "independent_init":
            torch.manual_seed(config.run.seed * 1000 + agent_id)
        model = make_model(config).to(device)
        if base_state is not None:
            model.load_state_dict(base_state)
        agents.append(
            NeuralAdoptionAgent(
                agent_id=agent_id,
                model=model,
                optimizer=make_optimizer(model, config),
            )
        )
    return agents


def build_contagion_graph(config: Toy5Config) -> nx.Graph:
    if config.graph.type == "watts_strogatz":
        graph = nx.watts_strogatz_graph(
            n=config.agents.count,
            k=config.graph.k,
            p=config.graph.rewire_probability,
            seed=config.run.seed,
        )
        graph.add_nodes_from(range(config.agents.count))
        return graph
    raise ValueError(f"Unsupported Toy 5 graph type: {config.graph.type}")


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [sorted(int(node) for node in graph.neighbors(i)) for i in range(agent_count)]


def initialize_adoptions(
    config: Toy5Config,
    rng: np.random.Generator,
) -> np.ndarray:
    agent_count = config.agents.count
    initial_fraction = config.environment.initial_action_fraction
    target_count = int(round(agent_count * initial_fraction))
    if initial_fraction > 0.0 and target_count == 0:
        target_count = 1
    target_count = min(max(target_count, 0), agent_count)
    adopted = np.zeros(agent_count, dtype=np.int64)
    if target_count == 0:
        return adopted
    if config.environment.seed_selection == "first_agent":
        adopted[:target_count] = 1
    elif config.environment.seed_selection == "random":
        seeds = rng.choice(agent_count, size=target_count, replace=False)
        adopted[seeds] = 1
    else:
        raise ValueError(f"Unsupported seed selection: {config.environment.seed_selection}")
    return adopted


def initialize_thresholds(
    config: Toy5Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str]]:
    if config.environment.threshold_mode == "homogeneous":
        return (
            np.full(config.agents.count, config.environment.homogeneous_threshold),
            ["homogeneous" for _ in range(config.agents.count)],
        )
    if config.environment.threshold_mode != "heterogeneous":
        raise ValueError(f"Unsupported threshold mode: {config.environment.threshold_mode}")
    groups = np.asarray(["low"] * (config.agents.count // 2))
    if len(groups) < config.agents.count:
        groups = np.concatenate(
            [groups, np.asarray(["high"] * (config.agents.count - len(groups)))]
        )
    rng.shuffle(groups)
    thresholds = np.where(
        groups == "low",
        config.environment.heterogeneous_threshold_low,
        config.environment.heterogeneous_threshold_high,
    )
    return thresholds.astype(np.float64), [str(group) for group in groups]


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


def _ragged_neighbor_sum_tensor(
    values: torch.Tensor,
    neighbors: list[list[int]],
) -> torch.Tensor:
    sums: list[torch.Tensor] = []
    zero = torch.zeros((), dtype=values.dtype, device=values.device)
    for peers in neighbors:
        if not peers:
            sums.append(zero)
            continue
        peer_index = torch.as_tensor(peers, dtype=torch.long, device=values.device)
        sums.append(values.index_select(0, peer_index).sum())
    if not sums:
        return torch.zeros(0, dtype=values.dtype, device=values.device)
    return torch.stack(sums)


def _ragged_neighbor_mean_tensor(
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


def neighbor_adoption_rates(
    adopted: StateArray,
    neighbors: list[list[int]],
    peer_index: np.ndarray | torch.Tensor | None = None,
) -> StateArray:
    if isinstance(adopted, torch.Tensor) or isinstance(peer_index, torch.Tensor):
        device = adopted.device if isinstance(adopted, torch.Tensor) else peer_index.device
        adopted_values = _tensor_vector(adopted, dtype=torch.float64, device=device)
        if peer_index is not None:
            peer_index_tensor = _tensor_peer_index(peer_index, device=device)
            if peer_index_tensor.shape[1] == 0:
                return torch.zeros(
                    int(adopted_values.shape[0]),
                    dtype=torch.float64,
                    device=device,
                )
            return adopted_values[peer_index_tensor].mean(dim=1)
        return _ragged_neighbor_mean_tensor(adopted_values, neighbors)

    rates = np.zeros(len(adopted), dtype=np.float64)
    for agent_id, peers in enumerate(neighbors):
        if peers:
            rates[agent_id] = float(np.mean(adopted[peers]))
    return rates


def adopted_neighbor_counts(
    adopted: StateArray,
    neighbors: list[list[int]],
    peer_index: np.ndarray | torch.Tensor | None = None,
) -> StateArray:
    if isinstance(adopted, torch.Tensor) or isinstance(peer_index, torch.Tensor):
        device = adopted.device if isinstance(adopted, torch.Tensor) else peer_index.device
        adopted_values = _tensor_vector(adopted, dtype=torch.float64, device=device)
        if peer_index is not None:
            peer_index_tensor = _tensor_peer_index(peer_index, device=device)
            if peer_index_tensor.shape[1] == 0:
                return torch.zeros(
                    int(adopted_values.shape[0]),
                    dtype=torch.float64,
                    device=device,
                )
            return adopted_values[peer_index_tensor].sum(dim=1)
        return _ragged_neighbor_sum_tensor(adopted_values, neighbors)

    counts = np.zeros(len(adopted), dtype=np.float64)
    for agent_id, peers in enumerate(neighbors):
        if peers:
            counts[agent_id] = float(np.sum(adopted[peers]))
    return counts


def update_exposure_counts(
    previous_counts: StateArray,
    adopted: StateArray,
    neighbors: list[list[int]],
    decay: float,
    peer_index: np.ndarray | torch.Tensor | None = None,
) -> StateArray:
    if any(isinstance(values, torch.Tensor) for values in (previous_counts, adopted, peer_index)):
        device = (
            previous_counts.device
            if isinstance(previous_counts, torch.Tensor)
            else adopted.device
            if isinstance(adopted, torch.Tensor)
            else peer_index.device
        )
        previous_values = _tensor_vector(
            previous_counts,
            dtype=torch.float64,
            device=device,
        )
        count_values = adopted_neighbor_counts(
            adopted,
            neighbors,
            peer_index=peer_index,
        )
        count_tensor = _tensor_vector(count_values, dtype=torch.float64, device=device)
        return decay * previous_values + count_tensor
    return decay * previous_counts + adopted_neighbor_counts(adopted, neighbors)


def simple_contagion_adoption_probabilities(
    adopted: np.ndarray,
    neighbors: list[list[int]],
    exposure_probability: float,
    adoption_is_absorbing: bool = True,
) -> np.ndarray:
    counts = adopted_neighbor_counts(adopted, neighbors)
    probabilities = 1.0 - np.power(1.0 - exposure_probability, counts)
    if adoption_is_absorbing:
        probabilities = np.where(adopted == 1, 1.0, probabilities)
    return np.clip(probabilities, 0.0, 1.0)


def complex_threshold_adoption_probabilities(
    adopted: np.ndarray,
    neighbors: list[list[int]],
    thresholds: np.ndarray,
    adoption_is_absorbing: bool = True,
) -> np.ndarray:
    rates = neighbor_adoption_rates(adopted, neighbors)
    probabilities = (rates >= thresholds).astype(np.float64)
    if adoption_is_absorbing:
        probabilities = np.where(adopted == 1, 1.0, probabilities)
    return probabilities


def sample_adoptions_from_probabilities(
    current_adoptions: StateArray,
    adoption_probs: np.ndarray,
    revision_mask: np.ndarray,
    adoption_is_absorbing: bool,
    rng: np.random.Generator,
) -> StateArray:
    current_array = to_numpy_view(current_adoptions, dtype=np.int64)
    next_adoptions = current_array.copy()
    for agent_id in np.flatnonzero(revision_mask):
        if adoption_is_absorbing and current_array[int(agent_id)] == 1:
            next_adoptions[int(agent_id)] = 1
        else:
            next_adoptions[int(agent_id)] = int(rng.random() < adoption_probs[int(agent_id)])
    if isinstance(current_adoptions, torch.Tensor):
        return torch.as_tensor(
            next_adoptions,
            dtype=current_adoptions.dtype,
            device=current_adoptions.device,
        )
    return next_adoptions


def select_peers_by_output_similarity(
    neighbors: list[list[int]],
    adoption_probs: np.ndarray,
    peer_rule: str,
    threshold: float,
) -> list[list[int]]:
    return select_binary_output_similarity_peers(
        neighbors=neighbors,
        action_probs=adoption_probs,
        peer_rule=peer_rule,
        threshold=threshold,
        error_label="Toy 5",
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
        error_label="Toy 5",
    )


def apply_output_average_to_adoption_probs(
    adoption_probs: np.ndarray,
    peer_ids: list[list[int]],
    alpha: float,
) -> tuple[np.ndarray, list[float]]:
    """Compatibility wrapper for the shared binary output-average mixer."""

    return mix_binary_output_average(adoption_probs, peer_ids, alpha)


def reputation_params_from_config(config: Toy5Config) -> ReputationParams:
    return ReputationParams(
        enabled=config.state.reputation.enabled,
        decay=config.state.reputation.decay,
        peer_rule=config.state.reputation.peer_rule,
        temperature=config.state.reputation.temperature,
        noise=config.state.reputation.noise,
    )


def neural_observation_input_dim(config: Toy5Config) -> int:
    return 6 + reputation_observation_extra_dim(config.state.reputation.observation_mode)


def adoption_probs_to_policy_tensor(
    adoption_probs: StateArray,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(adoption_probs, torch.Tensor):
        probs = adoption_probs.to(device=device, dtype=torch.float32)
        return torch.stack([1.0 - probs, probs], dim=1)
    return torch.as_tensor(
        binary_policy_matrix(adoption_probs),
        dtype=torch.float32,
        device=device,
    )


def build_observations(
    adopted: StateArray,
    neighbors: list[list[int]],
    thresholds: StateArray,
    exposure_counts: StateArray,
    device: torch.device,
    reputation: StateArray | None = None,
    reputation_observation_mode: str = "none",
    peer_index: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, StateArray, StateArray]:
    if any(
        isinstance(values, torch.Tensor)
        for values in (adopted, thresholds, exposure_counts, reputation, peer_index)
    ):
        return build_observations_tensor(
            adopted=adopted,
            neighbors=neighbors,
            thresholds=thresholds,
            exposure_counts=exposure_counts,
            device=device,
            reputation=reputation,
            reputation_observation_mode=reputation_observation_mode,
            peer_index=peer_index,
        )
    rates = neighbor_adoption_rates(adopted, neighbors)
    degrees = np.asarray([len(peers) for peers in neighbors], dtype=np.float64)
    max_degree = max(float(np.max(degrees)), 1.0)
    normalized_exposures = exposure_counts / np.maximum(degrees, 1.0)
    utility_proxy = rates - thresholds
    observations = np.column_stack(
        [
            adopted.astype(np.float64),
            rates,
            normalized_exposures,
            degrees / max_degree,
            utility_proxy,
            np.ones(len(adopted), dtype=np.float64),
        ]
    )
    if reputation_observation_mode != "none":
        if reputation is None:
            raise ValueError("reputation observations require reputation state")
        observations = np.column_stack(
            [
                observations,
                reputation_observation_features(
                    reputation=reputation,
                    peer_ids=neighbors,
                    mode=reputation_observation_mode,
                ),
            ]
        )
    return (
        torch.as_tensor(observations, dtype=torch.float32, device=device),
        rates,
        utility_proxy,
    )


def build_observations_tensor(
    adopted: StateArray,
    neighbors: list[list[int]],
    thresholds: StateArray,
    exposure_counts: StateArray,
    device: torch.device,
    reputation: StateArray | None = None,
    reputation_observation_mode: str = "none",
    peer_index: np.ndarray | torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adopted_values = _tensor_vector(adopted, dtype=torch.float64, device=device)
    threshold_values = _tensor_vector(thresholds, dtype=torch.float64, device=device)
    exposure_values = _tensor_vector(exposure_counts, dtype=torch.float64, device=device)
    agent_count = int(adopted_values.shape[0])
    if (
        threshold_values.shape[0] != agent_count
        or exposure_values.shape[0] != agent_count
    ):
        raise ValueError("threshold and exposure arrays must match action count")
    if peer_index is not None:
        peer_index_tensor = _tensor_peer_index(peer_index, device=device)
        if peer_index_tensor.shape[0] != agent_count:
            raise ValueError("peer_index first dimension must match action count")
        degrees = torch.full(
            (agent_count,),
            float(peer_index_tensor.shape[1]),
            dtype=torch.float64,
            device=device,
        )
        if peer_index_tensor.shape[1] == 0:
            rates = torch.zeros(agent_count, dtype=torch.float64, device=device)
        else:
            rates = adopted_values[peer_index_tensor].mean(dim=1)
    else:
        rates = _ragged_neighbor_mean_tensor(adopted_values, neighbors)
        degrees = torch.as_tensor(
            [len(peers) for peers in neighbors],
            dtype=torch.float64,
            device=device,
        )
    max_degree = degrees.max().clamp_min(1.0) if degrees.numel() else torch.tensor(
        1.0,
        dtype=torch.float64,
        device=device,
    )
    normalized_exposures = exposure_values / degrees.clamp_min(1.0)
    utility_proxy = rates - threshold_values
    observations = torch.stack(
        [
            adopted_values,
            rates,
            normalized_exposures,
            degrees / max_degree,
            utility_proxy,
            torch.ones(agent_count, dtype=torch.float64, device=device),
        ],
        dim=1,
    )
    if reputation_observation_mode != "none":
        if reputation is None:
            raise ValueError("reputation observations require reputation state")
        reputation_values = _tensor_vector(reputation, dtype=torch.float64, device=device)
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
            peer_means = _ragged_neighbor_mean_tensor(reputation_values, neighbors)
        observations = torch.cat(
            [
                observations,
                torch.stack([reputation_values, peer_means], dim=1),
            ],
            dim=1,
        )
    return observations.to(dtype=torch.float32), rates, utility_proxy


@torch.no_grad()
def collect_policy_probs(
    agents: list[NeuralAdoptionAgent],
    observations: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    return batched_mlp_policy_probs(
        [agent.model for agent in agents],
        observations,
        temperature=temperature,
    )


def decision_action_probs(policy_probs: torch.Tensor, config: Toy5Config) -> torch.Tensor:
    decision = config.policy.decision
    if decision.mode == "argmax":
        selected = torch.argmax(policy_probs, dim=-1)
        return torch.nn.functional.one_hot(selected, num_classes=2).to(policy_probs)
    adjusted = policy_probs
    if decision.action_temperature != 1.0:
        tiny = torch.finfo(policy_probs.dtype).tiny
        logits = torch.log(torch.clamp(policy_probs, min=tiny))
        adjusted = torch.softmax(logits / decision.action_temperature, dim=-1)
    if decision.exploration_epsilon > 0.0:
        adjusted = (
            (1.0 - decision.exploration_epsilon) * adjusted
            + decision.exploration_epsilon * 0.5
        )
    return adjusted


def clipped_adoption_utility_advantages(utility_proxy: StateArray) -> StateArray:
    if isinstance(utility_proxy, torch.Tensor):
        return torch.clamp(utility_proxy, -1.0, 1.0)
    return np.clip(utility_proxy, -1.0, 1.0)


def threshold_target_policy_advantages(
    *,
    actions: StateArray,
    utility_proxy: StateArray,
    current_actions: StateArray,
    adoption_is_absorbing: bool,
) -> StateArray:
    if isinstance(utility_proxy, torch.Tensor) or isinstance(actions, torch.Tensor):
        if not isinstance(utility_proxy, torch.Tensor):
            utility_values = torch.as_tensor(utility_proxy, dtype=torch.float64)
        else:
            utility_values = utility_proxy
        action_values = (
            actions.to(device=utility_values.device)
            if isinstance(actions, torch.Tensor)
            else torch.as_tensor(actions, device=utility_values.device)
        )
        current_values = (
            current_actions.to(device=utility_values.device)
            if isinstance(current_actions, torch.Tensor)
            else torch.as_tensor(current_actions, device=utility_values.device)
        )
        target = utility_values >= 0.0
        if adoption_is_absorbing:
            target = target | (current_values.to(dtype=torch.long) == 1)
        matches = action_values.to(dtype=torch.long) == target.to(dtype=torch.long)
        magnitude = torch.clamp(torch.abs(utility_values), 0.0, 1.0)
        return torch.where(matches, magnitude, -magnitude)

    utility_values = np.asarray(utility_proxy, dtype=np.float64)
    action_values = np.asarray(actions, dtype=np.int64)
    current_values = np.asarray(current_actions, dtype=np.int64)
    target = utility_values >= 0.0
    if adoption_is_absorbing:
        target = target | (current_values == 1)
    matches = action_values == target.astype(np.int64)
    magnitude = np.clip(np.abs(utility_values), 0.0, 1.0)
    return np.where(matches, magnitude, -magnitude)


def toy5_local_policy_advantages(
    *,
    actions: StateArray,
    utility_proxy: StateArray,
    current_actions: StateArray,
    local_update_rule: str,
    adoption_is_absorbing: bool,
) -> StateArray:
    if local_update_rule == "adoption_utility":
        return clipped_adoption_utility_advantages(utility_proxy)
    if local_update_rule == "threshold_target":
        return threshold_target_policy_advantages(
            actions=actions,
            utility_proxy=utility_proxy,
            current_actions=current_actions,
            adoption_is_absorbing=adoption_is_absorbing,
        )
    raise ValueError(f"Unsupported Toy 5 local update rule: {local_update_rule}")


def train_neural_local_policy(
    agent: NeuralAdoptionAgent,
    observation: torch.Tensor,
    action: int,
    utility_proxy: float,
    advantage: float | None = None,
) -> float:
    logits = agent.model(observation)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum()
    advantage_value = (
        float(np.clip(utility_proxy, -1.0, 1.0))
        if advantage is None
        else float(advantage)
    )
    loss = -advantage_value * log_probs[int(action)] - 0.01 * entropy
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
    return float(loss.detach().cpu())


def train_neural_local_policies_batched(
    agents: list[NeuralAdoptionAgent],
    observations: torch.Tensor,
    actions: np.ndarray,
    utility_proxy: np.ndarray,
    revision_mask: np.ndarray,
    advantages: StateArray | None = None,
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
        utility_proxy=utility_proxy,
        revision_mask=revision_mask,
        advantages=advantages,
        parameters=parameters,
        adam_state_cache=adam_state_cache,
        synchronize_model_parameters=synchronize_model_parameters,
        synchronize_optimizer_states=synchronize_optimizer_states,
        timing_context=timing_context,
    ).losses


def train_neural_local_policies_batched_update(
    agents: list[NeuralAdoptionAgent],
    observations: torch.Tensor,
    actions: np.ndarray,
    utility_proxy: np.ndarray,
    revision_mask: np.ndarray,
    advantages: StateArray | None = None,
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
        advantages=(
            clipped_adoption_utility_advantages(utility_proxy)
            if advantages is None
            else advantages
        ),
        active_agent_ids=active_agent_ids,
        entropy_beta=0.01,
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
    agents: list[NeuralAdoptionAgent],
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
        logits_fn=lambda agent, _agent_id, observed: agent.model(observed[agent.agent_id]),
        loss_mode="cross_entropy",
    )


def apply_output_average_distillation_batched(
    agents: list[NeuralAdoptionAgent],
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
    agents: list[NeuralAdoptionAgent],
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


def adoption_cluster_metrics(adopted: np.ndarray, graph: nx.Graph) -> tuple[int, float]:
    adopted_array = to_numpy_view(adopted, dtype=np.int64)
    adopted_nodes = [
        int(agent_id) for agent_id, value in enumerate(adopted_array) if value == 1
    ]
    if not adopted_nodes:
        return 0, 0.0
    subgraph = graph.subgraph(adopted_nodes)
    components = list(nx.connected_components(subgraph))
    largest_fraction = max(len(component) for component in components) / len(
        adopted_array
    )
    return len(components), float(largest_fraction)


def threshold_group_adoption_rate(
    adopted: np.ndarray,
    threshold_groups: list[str],
    group_name: str,
) -> float | None:
    mask = np.asarray([group == group_name for group in threshold_groups], dtype=bool)
    if not np.any(mask):
        return None
    return float(np.mean(adopted[mask]))


def aggregate_row(
    config: Toy5Config,
    epoch: int,
    adopted: np.ndarray,
    graph: nx.Graph,
    peer_ids: list[list[int]],
    neighbors: list[list[int]],
    exposure_counts: np.ndarray,
    threshold_groups: list[str],
    time_to_50: int | None,
    realized_revision_rate: float,
    policy_probs: torch.Tensor,
    policy_probs_pre_revision: torch.Tensor | None = None,
    policy_probs_post_local: torch.Tensor | None = None,
    reputation: np.ndarray | None = None,
    local_losses: list[float] | None = None,
    social_losses: list[float] | None = None,
) -> dict[str, object]:
    cluster_count, largest_cluster_fraction = adoption_cluster_metrics(adopted, graph)
    adoption_rate = float(np.mean(adopted))
    homogeneous_rate = threshold_group_adoption_rate(
        adopted,
        threshold_groups,
        "homogeneous",
    )
    low_rate = threshold_group_adoption_rate(adopted, threshold_groups, "low")
    high_rate = threshold_group_adoption_rate(adopted, threshold_groups, "high")
    return {
        **binary_aggregate_common_fields(
            config=config,
            toy="toy5",
            epoch=epoch,
            actions=adopted,
            payoffs=np.zeros(len(adopted), dtype=np.float64),
            policy_probs=policy_probs,
            peer_ids=peer_ids,
            realized_revision_rate=realized_revision_rate,
            reputation=reputation,
            mobility_result=MobilityStepResult.none(len(adopted)),
            policy_probs_pre_revision=policy_probs_pre_revision,
            policy_probs_post_local=policy_probs_post_local,
            local_losses=local_losses,
            social_losses=social_losses,
        ),
        "domain_threshold_mode": config.environment.threshold_mode,
        "domain_cascade_size": int(np.sum(adopted)),
        "domain_time_to_50_action": "" if time_to_50 is None else time_to_50,
        "domain_failed_cascade": adoption_rate < 0.5,
        "domain_action_cluster_count": cluster_count,
        "domain_largest_action_cluster_fraction": largest_cluster_fraction,
        "domain_mean_neighbor_action_rate": float(
            np.mean(neighbor_adoption_rates(adopted, neighbors))
        ),
        "domain_mean_repeated_exposure_count": float(np.mean(exposure_counts)),
        "domain_homogeneous_action_rate": (
            "" if homogeneous_rate is None else homogeneous_rate
        ),
        "domain_low_threshold_action_rate": "" if low_rate is None else low_rate,
        "domain_high_threshold_action_rate": "" if high_rate is None else high_rate,
    }


def make_run_dir(config: Toy5Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        config.run.output_dir
        / f"{timestamp}_{config.run.name}_seed{config.run.seed:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(config_path: Path, config: Toy5Config, run_dir: Path) -> None:
    write_run_metadata_artifacts(
        config_path=config_path,
        config=config,
        run_dir=run_dir,
        toy="toy5",
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "policy_rule": config.policy.rule,
            "domain_threshold_mode": config.environment.threshold_mode,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_agent_count": config.agents.count,
            "domain_graph_type": config.graph.type,
            "domain_graph_k": config.graph.k,
            "domain_graph_rewire_probability": config.graph.rewire_probability,
            "reputation": config.state.reputation.model_dump(),
        },
    )


def validate_tensor_batched_backend_config(config: Toy5Config) -> None:
    if config.policy.rule != "neural_policy":
        raise ValueError("Toy 5 tensor_batched requires policy.rule='neural_policy'")
    if config.coordination.peer_rule not in {"none", "output_similarity"}:
        raise ValueError(
            "Toy 5 tensor_batched requires coordination.peer_rule to be "
            "'none' or 'output_similarity'"
        )
    if config.coordination.mixer not in {"none", "output_average"}:
        raise ValueError(
            "Toy 5 tensor_batched requires coordination.mixer to be "
            "'none' or 'output_average'"
        )
    if config.agents.optimizer.name != "adam":
        raise ValueError("Toy 5 tensor_batched requires Adam optimizer")
    model = config.agents.model
    if model.activation != "relu" or model.output_dim != 2:
        raise ValueError(
            "Toy 5 tensor_batched requires the standard one-hidden-layer ReLU MLP"
        )


@dataclass
class Toy5SpatialDomain(BinaryToyDomainBase):
    """Toy 5 adapter for the shared binary spatial lifecycle runner."""

    config: Toy5Config
    config_path: Path
    rng: np.random.Generator
    device: torch.device
    neural_update_backend: NeuralUpdateBackend = "loop"
    policy_cache: BatchedMLPPolicyCache | None = field(default=None, init=False)
    tensor_runtime: TensorBatchedMLPRuntime | None = field(default=None, init=False)
    _adam_state_cache: BatchedAdamStateCache | None = field(default=None, init=False)
    _uniform_neighbor_peer_count: int | None = field(default=None, init=False)
    _uniform_neighbor_peer_index: torch.Tensor | None = field(default=None, init=False)
    _neighbor_peer_index_cache: PeerIndexCache | None = field(default=None, init=False)
    _pending_policy_cache_parameters: BatchedMLPParameters | None = field(
        default=None,
        init=False,
    )

    micro_state_fields: ClassVar[list[str]] = TOY5_MICRO_STATE_FIELDS
    aggregate_fields: ClassVar[list[str]] = TOY5_AGGREGATE_FIELDS
    toy: ClassVar[str] = "toy5"

    def __post_init__(self) -> None:
        if self.neural_update_backend not in {"loop", "batched", "tensor_batched"}:
            raise ValueError(
                "Toy 5 neural_update_backend must be 'loop', 'batched', "
                "or 'tensor_batched'"
            )
        if self.neural_update_backend == "tensor_batched":
            validate_tensor_batched_backend_config(self.config)
        self.graph = build_contagion_graph(self.config)
        self.neighbors = graph_neighbors(self.graph, self.config.agents.count)
        validate_peer_ids(self.neighbors, self.config.agents.count)
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
        self.thresholds: np.ndarray | None = None
        self.threshold_groups: list[str] = []
        self.time_to_50: int | None = None

    def _uses_torch_state(self) -> bool:
        return (
            self.config.policy.rule == "neural_policy"
            and self.neural_update_backend == "tensor_batched"
        )

    def _observation_peer_index(self) -> torch.Tensor | None:
        if self._uses_torch_state():
            return self._uniform_neighbor_peer_index
        return None

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
            agent_count=self.config.agents.count,
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
        adopted_np = initialize_adoptions(self.config, self.rng)
        self.thresholds, self.threshold_groups = initialize_thresholds(
            self.config,
            self.rng,
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
            adopted: StateArray = torch.as_tensor(
                adopted_np,
                dtype=torch.long,
                device=self.device,
            )
            reputation: StateArray = (
                adopted.to(dtype=torch.float64)
                if self.config.state.reputation.enabled
                else torch.zeros(
                    self.config.agents.count,
                    dtype=torch.float64,
                    device=self.device,
                )
            )
            exposure_counts: StateArray = torch.zeros(
                self.config.agents.count,
                dtype=torch.float64,
                device=self.device,
            )
            payoffs = torch.zeros(
                self.config.agents.count,
                dtype=torch.float64,
                device=self.device,
            )
            payoff_ema = torch.zeros(
                self.config.agents.count,
                dtype=torch.float64,
                device=self.device,
            )
            previous_payoff_ema = torch.zeros(
                self.config.agents.count,
                dtype=torch.float64,
                device=self.device,
            )
        else:
            adopted = adopted_np
            reputation = (
                adopted_np.astype(np.float64)
                if self.config.state.reputation.enabled
                else np.zeros(self.config.agents.count, dtype=np.float64)
            )
            exposure_counts = np.zeros(self.config.agents.count, dtype=np.float64)
            payoffs = np.zeros(self.config.agents.count, dtype=np.float64)
            payoff_ema = np.zeros(self.config.agents.count, dtype=np.float64)
            previous_payoff_ema = np.zeros(self.config.agents.count, dtype=np.float64)
        self.time_to_50 = (
            0
            if float(np.mean(to_numpy_view(adopted, dtype=np.float64))) >= 0.5
            else None
        )
        return BinarySpatialState(
            actions=adopted,
            payoffs=payoffs,
            payoff_ema=payoff_ema,
            previous_payoff_ema=previous_payoff_ema,
            reputation=reputation,
            agents=agents,
            extras={"exposure_counts": exposure_counts},
        )

    def initial_step_result(
        self,
        state: BinarySpatialState,
    ) -> BinaryPolicyStepResult:
        initial_probs = adoption_probs_to_policy_tensor(
            state.actions.to(dtype=torch.float32)
            if isinstance(state.actions, torch.Tensor)
            else state.actions.astype(np.float64),
            device=self.device,
        )
        initial_peer_ids = select_peers_by_output_similarity(
            neighbors=self.neighbors,
            adoption_probs=initial_probs[:, 1].detach().cpu().numpy(),
            peer_rule=self.config.coordination.peer_rule,
            threshold=self.config.coordination.threshold,
        )
        initial_peer_ids = peer_ids_for_mixer(
            initial_peer_ids,
            mixer=self.config.coordination.mixer,
            agent_count=self.config.agents.count,
        )
        return BinaryPolicyStepResult(
            pre_revision_probs=initial_probs,
            post_local_probs=initial_probs,
            post_social_probs=initial_probs,
            local_losses=[0.0 for _ in range(self.config.agents.count)],
            social_losses=[0.0 for _ in range(self.config.agents.count)],
            peer_ids=initial_peer_ids,
            revision_mask=np.zeros(self.config.agents.count, dtype=bool),
            mobility_result=MobilityStepResult.none(self.config.agents.count),
            realized_revision_rate=0.0,
            extras={"final_action_probs": initial_probs[:, 1].detach().cpu().numpy()},
        )

    def build_step_context(
        self,
        epoch: int,
        state: BinarySpatialState,
        revision_mask: np.ndarray,
    ) -> BinaryStepContext:
        if self.thresholds is None:
            raise RuntimeError("Toy 5 thresholds were not initialized")
        adopted = state.actions
        previous_adopted = (
            adopted.detach().clone() if isinstance(adopted, torch.Tensor) else adopted.copy()
        )
        exposure_counts = update_exposure_counts(
            previous_counts=state.extras["exposure_counts"],
            adopted=adopted,
            neighbors=self.neighbors,
            decay=self.config.policy.domain.repeated_exposure_decay,
            peer_index=self._observation_peer_index(),
        )
        state.extras["exposure_counts"] = exposure_counts
        return BinaryStepContext(
            epoch=epoch,
            revision_mask=revision_mask,
            extras={
                "previous_adopted": previous_adopted,
                "exposure_counts": exposure_counts,
            },
        )

    def local_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
    ) -> BinaryLocalStepResult:
        if self.thresholds is None:
            raise RuntimeError("Toy 5 thresholds were not initialized")
        config = self.config
        adopted = state.actions
        reputation = state.reputation
        exposure_counts = context.extras["exposure_counts"]
        local_losses = [0.0 for _ in range(config.agents.count)]
        pre_revision_probs = adoption_probs_to_policy_tensor(
            adopted.to(dtype=torch.float32)
            if isinstance(adopted, torch.Tensor)
            else adopted.astype(np.float64),
            device=self.device,
        )

        if config.policy.rule == "simple_contagion":
            adoption_probs = simple_contagion_adoption_probabilities(
                adopted=adopted,
                neighbors=self.neighbors,
                exposure_probability=config.environment.simple_contagion_probability,
                adoption_is_absorbing=config.policy.domain.adoption_is_absorbing,
            )
            post_local_probs = adoption_probs_to_policy_tensor(
                adoption_probs,
                device=self.device,
            )
            return BinaryLocalStepResult(
                pre_revision_probs=pre_revision_probs,
                candidate_action_probs=adoption_probs,
                post_local_probs=post_local_probs,
                local_losses=local_losses,
                social_mode="probability_mix",
            )
        elif config.policy.rule == "complex_threshold":
            adoption_probs = complex_threshold_adoption_probabilities(
                adopted=adopted,
                neighbors=self.neighbors,
                thresholds=self.thresholds,
                adoption_is_absorbing=config.policy.domain.adoption_is_absorbing,
            )
            post_local_probs = adoption_probs_to_policy_tensor(
                adoption_probs,
                device=self.device,
            )
            return BinaryLocalStepResult(
                pre_revision_probs=pre_revision_probs,
                candidate_action_probs=adoption_probs,
                post_local_probs=post_local_probs,
                local_losses=local_losses,
                social_mode="probability_mix",
            )
        elif config.policy.rule == "reputation_imitation":
            adoption_probs = reputation_imitation_cooperation_probs(
                actions=adopted,
                reputation=reputation,
                peer_ids=self.neighbors,
                revision_mask=context.revision_mask,
                rng=self.rng,
                params=reputation_params_from_config(config),
            )
            if config.policy.domain.adoption_is_absorbing:
                adoption_probs = np.where(adopted == 1, 1.0, adoption_probs)
            post_local_probs = adoption_probs_to_policy_tensor(
                adoption_probs,
                device=self.device,
            )
            return BinaryLocalStepResult(
                pre_revision_probs=pre_revision_probs,
                candidate_action_probs=adoption_probs,
                post_local_probs=post_local_probs,
                local_losses=local_losses,
                social_mode="probability_mix",
            )
        elif config.policy.rule == "neural_policy":
            agents = list(state.agents or [])
            with timed_context_stage(context, "build_observations"):
                observations, _, utility_proxy = build_observations(
                    adopted=adopted,
                    neighbors=self.neighbors,
                    thresholds=self.thresholds,
                    exposure_counts=exposure_counts,
                    device=self.device,
                    reputation=reputation,
                    reputation_observation_mode=config.state.reputation.observation_mode,
                    peer_index=self._observation_peer_index(),
                )
            def sample_policy_actions(action_probs: torch.Tensor) -> StateArray:
                return sample_adoptions_from_probabilities(
                    current_adoptions=adopted,
                    adoption_probs=action_probs[:, 1].detach().cpu().numpy(),
                    revision_mask=context.revision_mask,
                    adoption_is_absorbing=config.policy.domain.adoption_is_absorbing,
                    rng=self.rng,
                )

            def commit_local_update(updated_actions: StateArray) -> LossVector:
                local_losses = [0.0 for _ in range(config.agents.count)]
                if config.policy.learning_enabled:
                    advantages = toy5_local_policy_advantages(
                        actions=updated_actions,
                        utility_proxy=utility_proxy,
                        current_actions=adopted,
                        local_update_rule=config.policy.domain.local_update_rule,
                        adoption_is_absorbing=(
                            config.policy.domain.adoption_is_absorbing
                        ),
                    )
                    if self.neural_update_backend == "tensor_batched":
                        local_report = run_tensor_runtime_policy_gradient_local_update(
                            runtime=self._require_tensor_runtime(agents),
                            observations=observations,
                            actions=updated_actions,
                            advantages=advantages,
                            active_agent_ids=[
                                int(agent_id)
                                for agent_id in np.flatnonzero(context.revision_mask)
                            ],
                            entropy_beta=0.01,
                            timing_context=context,
                        )
                        if local_report.update_result is None:
                            raise RuntimeError(
                                "tensor local update adapter did not produce a result",
                            )
                        update_result = local_report.update_result
                        local_losses = update_result.losses
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
                                utility_proxy=utility_proxy,
                                revision_mask=context.revision_mask,
                                advantages=advantages,
                                parameters=update_parameters,
                                adam_state_cache=self._require_adam_state_cache(agents),
                                synchronize_model_parameters=not defer_agent_sync,
                                synchronize_optimizer_states=not defer_agent_sync,
                                timing_context=context,
                            )
                        local_losses = update_result.losses
                        self._pending_policy_cache_parameters = (
                            update_result.updated_parameters
                        )
                        if not update_result.used_batched_optimizer:
                            self._adam_state_cache = None
                    else:
                        for agent_id in np.flatnonzero(context.revision_mask):
                            local_losses[int(agent_id)] = train_neural_local_policy(
                                agent=agents[int(agent_id)],
                                observation=observations[int(agent_id)],
                                action=int(updated_actions[int(agent_id)]),
                                utility_proxy=float(utility_proxy[int(agent_id)]),
                                advantage=float(advantages[int(agent_id)]),
                            )
                return local_losses

            learning_result = run_binary_policy_learning_step(
                agents=agents,
                observations=observations,
                temperature=config.policy.temperature,
                collect_policy_probs=self.collect_policy_probs,
                decision_action_probs=lambda probs: decision_action_probs(
                    probs,
                    config,
                ),
                sample_actions=sample_policy_actions,
                local_update=commit_local_update,
                refresh_policy_cache=self.refresh_policy_cache,
                context=context,
                unit_type=BinaryPolicyLearningUnit,
            )
            action_probs = learning_result.decision_action_probs
            return BinaryLocalStepResult(
                pre_revision_probs=learning_result.pre_revision_probs,
                candidate_action_probs=action_probs[:, 1].detach().cpu().numpy(),
                post_local_probs=learning_result.post_local_probs,
                local_losses=learning_result.local_losses,
                social_mode="policy_distill",
                actions_after_revision=learning_result.actions_after_revision,
                extras={
                    "decision_action_probs": action_probs,
                    "_observations": observations,
                },
            )
        else:
            raise ValueError(
                f"Unsupported Toy 5 update rule: {config.policy.rule}"
            )

    def policy_tensor_from_action_probs(
        self,
        action_probs: np.ndarray,
        device_like: torch.Tensor,
    ) -> torch.Tensor:
        return adoption_probs_to_policy_tensor(action_probs, device=device_like.device)

    def sample_actions(
        self,
        state: BinarySpatialState,
        action_probs: np.ndarray,
        revision_mask: np.ndarray,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
    ) -> StateArray:
        del context, local_result
        return sample_adoptions_from_probabilities(
            current_adoptions=state.actions,
            adoption_probs=action_probs,
            revision_mask=revision_mask,
            adoption_is_absorbing=self.config.policy.domain.adoption_is_absorbing,
            rng=self.rng,
        )

    def collect_policy_probs(
        self,
        agents: list[NeuralAdoptionAgent],
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

    def refresh_policy_cache(self, agents: list[NeuralAdoptionAgent]) -> None:
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
        agents: list[NeuralAdoptionAgent],
    ) -> BatchedMLPPolicyCache:
        if self.policy_cache is None:
            self.refresh_policy_cache(agents)
        if self.policy_cache is None:
            raise RuntimeError("Toy 5 policy cache is not initialized")
        return self.policy_cache

    def _require_tensor_runtime(
        self,
        agents: list[NeuralAdoptionAgent],
    ) -> TensorBatchedMLPRuntime:
        if self.tensor_runtime is None:
            self.tensor_runtime = TensorBatchedMLPRuntime.from_agents(
                agents,
                device=self.device,
            )
        return self.tensor_runtime

    def _require_adam_state_cache(
        self,
        agents: list[NeuralAdoptionAgent],
    ) -> BatchedAdamStateCache:
        if self._adam_state_cache is None:
            self._adam_state_cache = BatchedAdamStateCache.from_agents(
                agents,
                device=self.device,
            )
        return self._adam_state_cache

    def flush_tensor_runtime_to_agents(
        self,
        agents: list[NeuralAdoptionAgent],
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
        return super().write_summary(run_dir, final_row, state)

    def precommitment_direction_scores(
        self,
        state: BinarySpatialState,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        action_probs: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray | None:
        source = self.config.coordination.precommitment_direction_source
        if source == "social":
            return super().precommitment_direction_scores(
                state=state,
                local_result=local_result,
                social_result=social_result,
                action_probs=action_probs,
                active=active,
            )
        del local_result, action_probs, active
        if self.thresholds is None:
            raise RuntimeError("Toy 5 thresholds were not initialized")
        actions = to_numpy_view(state.actions, dtype=np.int64)
        thresholds = to_numpy_view(self.thresholds, dtype=np.float64)
        active_neighbor_rate = np.asarray(
            neighbor_adoption_rates(actions, self.neighbors),
            dtype=np.float64,
        )
        if source == "local_threshold":
            return active_neighbor_rate - thresholds
        if source not in {
            "readiness_augmented_threshold",
            "readiness_augmented_threshold_with_action_anchor",
            "readiness_exposure_with_action_anchor",
        }:
            raise ValueError(
                f"Unsupported Toy 5 precommitment direction source: {source}"
            )
        previous_readiness = self._precommitment_decision_feedback_scores(state)
        if source in {
            "readiness_augmented_threshold_with_action_anchor",
            "readiness_exposure_with_action_anchor",
        }:
            previous_readiness = np.maximum(
                previous_readiness,
                actions.astype(np.float64),
            )
        peer_readiness = binary_peer_aggregate_values(
            peer_ids=social_result.peer_ids,
            values=previous_readiness,
            aggregation=self.coordination_precommitment_peer_readiness_aggregation(),
        )
        if source == "readiness_exposure_with_action_anchor":
            exposure_scores = (
                active_neighbor_rate
                + self.config.coordination.precommitment_readiness_direction_weight
                * peer_readiness
            )
            exposure_scores = np.where(
                actions == 1,
                np.maximum(exposure_scores, 1.0),
                exposure_scores,
            )
            return np.where(exposure_scores > 0.0, exposure_scores, -1.0)
        scores = (
            active_neighbor_rate
            + self.config.coordination.precommitment_readiness_direction_weight
            * peer_readiness
            - thresholds
        )
        if source == "readiness_augmented_threshold_with_action_anchor":
            scores = np.where(actions == 1, np.maximum(scores, 1.0), scores)
        return scores

    def distill_policy(
        self,
        agents: list[NeuralAdoptionAgent],
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
            logits_fn=lambda agent, _agent_id, observed: agent.model(
                observed[agent.agent_id]
            ),
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
        del local_result, social_result
        adoption_rate = float(np.mean(to_numpy_view(actions, dtype=np.float64)))
        state.actions = actions
        if self.time_to_50 is None and adoption_rate >= 0.5:
            self.time_to_50 = context.epoch
        return {
            "previous_adopted": context.extras["previous_adopted"],
            "action_rate": adoption_rate,
        }

    def finalize_hook_step(
        self,
        state: BinarySpatialState,
        context: BinaryStepContext,
        local_result: BinaryLocalStepResult,
        social_result: BinarySocialStepResult,
        mobility_result: MobilityStepResult,
    ) -> dict[str, object]:
        del state, local_result, mobility_result
        final_adoption_probs = social_result.final_action_probs
        if (
            self.config.policy.rule == "reputation_imitation"
            and self.config.policy.domain.adoption_is_absorbing
        ):
            previous_adopted = to_numpy_view(
                context.extras["previous_adopted"],
                dtype=np.int64,
            )
            final_adoption_probs = np.where(
                previous_adopted == 1,
                1.0,
                final_adoption_probs,
            )
            return {
                "post_social_probs": adoption_probs_to_policy_tensor(
                    final_adoption_probs,
                    device=self.device,
                ),
                "extras": {"final_action_probs": final_adoption_probs},
            }
        return {"extras": {"final_action_probs": final_adoption_probs}}

    def domain_aggregate_fields(
        self,
        epoch: int,
        state: BinarySpatialState,
        step_result: BinaryPolicyStepResult,
    ) -> dict[str, object]:
        del epoch, step_result
        actions = to_numpy_view(state.actions, dtype=np.int64)
        exposure_counts = to_numpy_view(
            state.extras["exposure_counts"],
            dtype=np.float64,
        )
        cluster_count, largest_cluster_fraction = adoption_cluster_metrics(
            actions,
            self.graph,
        )
        action_rate = float(np.mean(actions))
        homogeneous_rate = threshold_group_adoption_rate(
            actions,
            self.threshold_groups,
            "homogeneous",
        )
        low_rate = threshold_group_adoption_rate(
            actions,
            self.threshold_groups,
            "low",
        )
        high_rate = threshold_group_adoption_rate(
            actions,
            self.threshold_groups,
            "high",
        )
        return {
            "domain_threshold_mode": self.config.environment.threshold_mode,
            "domain_cascade_size": int(np.sum(actions)),
            "domain_non_adoption_rate": 1.0 - action_rate,
            "domain_time_to_50_action": (
                "" if self.time_to_50 is None else self.time_to_50
            ),
            "domain_failed_cascade": action_rate < 0.5,
            "domain_action_cluster_count": cluster_count,
            "domain_largest_action_cluster_fraction": largest_cluster_fraction,
            "domain_mean_neighbor_action_rate": float(
                np.mean(neighbor_adoption_rates(actions, self.neighbors))
            ),
            "domain_mean_repeated_exposure_count": float(
                np.mean(exposure_counts)
            ),
            "domain_homogeneous_action_rate": (
                "" if homogeneous_rate is None else homogeneous_rate
            ),
            "domain_low_threshold_action_rate": (
                "" if low_rate is None else low_rate
            ),
            "domain_high_threshold_action_rate": (
                "" if high_rate is None else high_rate
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
        if self.thresholds is None:
            raise RuntimeError("Toy 5 thresholds were not initialized")
        actions = to_numpy_view(state.actions, dtype=np.int64)
        exposure_counts = to_numpy_view(
            state.extras["exposure_counts"],
            dtype=np.float64,
        )
        rates = neighbor_adoption_rates(actions, self.neighbors)
        utility_proxy = rates - self.thresholds
        previous_adopted = step_result.extras.get(
            "previous_adopted",
            actions.copy(),
        )
        previous_adopted_array = to_numpy_view(previous_adopted, dtype=np.int64)
        newly_adopted = (actions == 1) & (previous_adopted_array == 0)
        return {
            "domain_threshold": float(self.thresholds[agent_id]),
            "domain_threshold_group": self.threshold_groups[agent_id],
            "domain_neighbor_action_rate": float(rates[agent_id]),
            "domain_repeated_exposure_count": float(
                exposure_counts[agent_id]
            ),
            "domain_degree": len(self.neighbors[agent_id]),
            "domain_utility_proxy": float(utility_proxy[agent_id]),
            "domain_newly_adopted": bool(newly_adopted[agent_id]),
        }


def run_toy5(
    config: Toy5Config,
    config_path: Path,
    timing_rows: list[dict[str, object]] | None = None,
    neural_update_backend: NeuralUpdateBackendRequest | None = None,
) -> BinaryToyResult:
    """Run Toy 5 from a validated config."""

    if config.policy.rule == "neural_policy":
        expected_input_dim = neural_observation_input_dim(config)
        if config.agents.model.input_dim != expected_input_dim:
            raise ValueError(
                "Toy 5 neural_policy expects "
                f"model.input_dim={expected_input_dim}"
            )
        if config.agents.model.output_dim != 2:
            raise ValueError("Toy 5 neural_policy expects model.output_dim=2")

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
        agent_count=config.agents.count,
    )
    domain = Toy5SpatialDomain(
        config=config,
        config_path=config_path,
        rng=rng,
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
