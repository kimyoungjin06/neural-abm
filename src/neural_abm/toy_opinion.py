"""Toy 3: opinion dynamics with endogenous rewiring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import torch
from torch import nn

from neural_abm.accelerator import resolve_torch_device
from neural_abm.config import Toy3Config
from neural_abm.graphs import component_map, graph_from_peer_ids
from neural_abm.logging import CsvLogWriter
from neural_abm.results import (
    DomainToyResult,
    write_domain_summary_artifact,
    write_run_metadata_artifacts,
)
from neural_abm.social import (
    SCALAR_PROBABILITY_CHANNEL,
    SocialBlock,
    SocialChannel,
    select_scalar_output_peers,
)
from neural_abm.unit import ObservationSpec, SocialMessageSpec


TOY3_MICRO_STATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "policy_rule",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_opinion",
    "domain_opinion_pre_update",
    "domain_opinion_delta",
    "domain_neighbor_opinion_mean",
    "domain_neighbor_opinion_std",
    "domain_local_disagreement",
    "domain_degree",
    "peer_ids",
    "peer_count",
    "component_id",
    "domain_edge_disagreement",
    "domain_acceptance_probability_pre_social",
    "domain_acceptance_probability_post_social",
    "revised",
    "domain_rewired",
]


TOY3_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "policy_rule",
    "domain_confidence_threshold",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_rewiring_enabled",
    "domain_rewiring_threshold",
    "domain_rewiring_rate",
    "domain_realized_rewiring_rate",
    "domain_rewired_edge_count",
    "domain_cumulative_rewired_edge_count",
    "domain_opinion_mean",
    "domain_opinion_variance",
    "domain_polarization_index",
    "domain_opinion_cluster_count",
    "domain_mean_edge_disagreement",
    "domain_high_disagreement_edge_fraction",
    "fragmentation_components",
    "domain_largest_connected_component_fraction",
    "mean_peer_count",
    "domain_opinion_assortativity",
]


class OpinionMLP(nn.Module):
    """Small acceptance model for Toy 3 neural opinion updates."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


@dataclass
class NeuralOpinionAgent:
    agent_id: int
    model: OpinionMLP
    optimizer: torch.optim.Optimizer

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="opinion_observation",
            tensor_shape=(None, self.model.fc1.in_features),
            dtype=torch.float32,
        )

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec(
            required_keys=(
                "agent_id",
                "acceptance_probability",
                "probe_probs",
                "latent_summary",
                "confidence",
                "param_norm",
            ),
            tensor_keys=("acceptance_probability", "probe_probs", "latent_summary"),
            probability_keys=("acceptance_probability", "probe_probs"),
        )

    def observe(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @torch.no_grad()
    def act_or_predict(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(observation))

    def local_update(self, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        raise NotImplementedError("Toy 3 local update requires opinion context")

    def hidden_on(self, observation: torch.Tensor) -> torch.Tensor:
        return self.model.activation(self.model.fc1(observation))

    @torch.no_grad()
    def social_message(self, observation: torch.Tensor) -> dict[str, Any]:
        observed = self.observe(observation)
        acceptance = self.act_or_predict(observed).reshape(-1)
        probe_probs = torch.stack([1.0 - acceptance, acceptance], dim=-1)
        hidden = self.hidden_on(observed)
        latent_summary = hidden.mean(dim=0) if hidden.ndim > 1 else hidden
        entropy = -(probe_probs * torch.log(probe_probs + 1e-8)).sum(dim=-1).mean()
        max_entropy = torch.log(
            torch.tensor(2.0, dtype=probe_probs.dtype, device=probe_probs.device)
        )
        confidence = torch.clamp(1.0 - entropy / max_entropy, 0.0, 1.0)
        params = torch.cat(
            [param.detach().flatten() for param in self.model.parameters()]
        )
        return {
            "agent_id": self.agent_id,
            "acceptance_probability": acceptance.detach().clone(),
            "probe_probs": probe_probs.detach().clone(),
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


@dataclass(frozen=True)
class RewireStats:
    rewired_edge_count: int
    considered_edge_count: int
    rewired_agents: set[int]


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def make_model(config: Toy3Config) -> OpinionMLP:
    model_config = config.agents.model
    return OpinionMLP(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=model_config.output_dim,
    )


def make_optimizer(model: torch.nn.Module, config: Toy3Config) -> torch.optim.Optimizer:
    if config.agents.optimizer.name == "adam":
        learning_rate = (
            config.dynamics.neural_learning_rate
            if config.dynamics.neural_learning_rate > 0.0
            else config.agents.optimizer.learning_rate
        )
        return torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
    raise ValueError(f"Unsupported optimizer: {config.agents.optimizer.name}")


def create_agents(config: Toy3Config, device: torch.device) -> list[NeuralOpinionAgent]:
    base_state = None
    if config.agents.init_mode == "same_init":
        torch.manual_seed(config.run.seed)
        base_state = clone_state_dict(make_model(config))

    agents: list[NeuralOpinionAgent] = []
    for agent_id in range(config.agents.count):
        if config.agents.init_mode == "independent_init":
            torch.manual_seed(config.run.seed * 1000 + agent_id)
        model = make_model(config).to(device)
        if base_state is not None:
            model.load_state_dict(base_state)
        agents.append(
            NeuralOpinionAgent(
                agent_id=agent_id,
                model=model,
                optimizer=make_optimizer(model, config),
            )
        )
    return agents


def build_opinion_graph(config: Toy3Config) -> nx.Graph:
    if config.graph.type == "watts_strogatz":
        graph = nx.watts_strogatz_graph(
            n=config.agents.count,
            k=config.graph.k,
            p=config.graph.rewire_probability,
            seed=config.run.seed,
        )
        graph.add_nodes_from(range(config.agents.count))
        return graph
    raise ValueError(f"Unsupported Toy 3 graph type: {config.graph.type}")


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [
        sorted(int(node) for node in graph.neighbors(i)) for i in range(agent_count)
    ]


def initialize_opinions(
    config: Toy3Config,
    rng: np.random.Generator,
) -> np.ndarray:
    env = config.environment
    if env.initial_opinion_mode == "uniform":
        opinions = rng.uniform(env.opinion_min, env.opinion_max, config.agents.count)
    elif env.initial_opinion_mode == "two_clusters":
        assignments = np.arange(config.agents.count) % 2
        rng.shuffle(assignments)
        centers = np.asarray(env.cluster_centers, dtype=np.float64)
        opinions = rng.normal(centers[assignments], env.cluster_std)
    else:
        raise ValueError(
            f"Unsupported initial opinion mode: {env.initial_opinion_mode}"
        )
    return np.clip(opinions.astype(np.float64), env.opinion_min, env.opinion_max)


def compatible_peer_ids(
    opinions: np.ndarray,
    neighbors: list[list[int]],
    confidence_threshold: float,
) -> list[list[int]]:
    return [
        [
            int(peer)
            for peer in peers
            if abs(float(opinions[agent_id]) - float(opinions[peer]))
            <= confidence_threshold
        ]
        for agent_id, peers in enumerate(neighbors)
    ]


def select_neural_peer_ids(
    opinions: np.ndarray,
    neighbors: list[list[int]],
    peer_rule: str,
    confidence_threshold: float,
    output_similarity_threshold: float = 0.0,
    acceptance_probs: torch.Tensor | np.ndarray | None = None,
) -> list[list[int]]:
    """Select Toy 3 neural peers using bounded confidence or output similarity."""

    if peer_rule == "none":
        return [[int(peer) for peer in peers] for peers in neighbors]
    if peer_rule == "bounded_confidence":
        return compatible_peer_ids(opinions, neighbors, confidence_threshold)
    if peer_rule == "output_similarity":
        if acceptance_probs is None:
            raise ValueError("Toy 3 output_similarity peer selection requires outputs")
        values = (
            acceptance_probs.detach().cpu().numpy()
            if isinstance(acceptance_probs, torch.Tensor)
            else np.asarray(acceptance_probs)
        )
        return select_scalar_output_peers(
            neighbors=neighbors,
            values=np.asarray(values, dtype=np.float64).reshape(-1),
            peer_rule="output_similarity",
            threshold=output_similarity_threshold,
        ).peer_ids
    raise ValueError(f"Unsupported Toy 3 peer rule: {peer_rule}")


def hk_update_opinions(
    opinions: np.ndarray,
    neighbors: list[list[int]],
    confidence_threshold: float,
    influence_rate: float = 1.0,
    opinion_min: float = -1.0,
    opinion_max: float = 1.0,
) -> tuple[np.ndarray, list[list[int]]]:
    """Apply one graph-local Hegselmann-Krause bounded-confidence step."""

    peer_ids = compatible_peer_ids(opinions, neighbors, confidence_threshold)
    next_opinions = opinions.astype(np.float64).copy()
    for agent_id, peers in enumerate(peer_ids):
        if not peers:
            continue
        target = float(np.mean(np.concatenate(([opinions[agent_id]], opinions[peers]))))
        next_opinions[agent_id] = opinions[agent_id] + influence_rate * (
            target - opinions[agent_id]
        )
    return np.clip(next_opinions, opinion_min, opinion_max), peer_ids


def deffuant_pair_update(
    opinion_i: float,
    opinion_j: float,
    confidence_threshold: float,
    mu: float,
) -> tuple[float, float]:
    """Apply a symmetric Deffuant pair update."""

    if abs(opinion_i - opinion_j) > confidence_threshold:
        return opinion_i, opinion_j
    return (
        opinion_i + mu * (opinion_j - opinion_i),
        opinion_j + mu * (opinion_i - opinion_j),
    )


def deffuant_update_opinions(
    opinions: np.ndarray,
    graph: nx.Graph,
    confidence_threshold: float,
    mu: float,
    rng: np.random.Generator,
    opinion_min: float = -1.0,
    opinion_max: float = 1.0,
) -> tuple[np.ndarray, list[list[int]]]:
    neighbors = graph_neighbors(graph, len(opinions))
    peer_ids = compatible_peer_ids(opinions, neighbors, confidence_threshold)
    next_opinions = opinions.astype(np.float64).copy()
    edges = [(int(i), int(j)) for i, j in graph.edges()]
    rng.shuffle(edges)
    for source, target in edges:
        next_source, next_target = deffuant_pair_update(
            float(next_opinions[source]),
            float(next_opinions[target]),
            confidence_threshold=confidence_threshold,
            mu=mu,
        )
        next_opinions[source] = next_source
        next_opinions[target] = next_target
    return np.clip(next_opinions, opinion_min, opinion_max), peer_ids


def neighbor_stats(
    opinions: np.ndarray,
    neighbors: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    agent_count = len(opinions)
    means = np.zeros(agent_count, dtype=np.float64)
    stds = np.zeros(agent_count, dtype=np.float64)
    disagreements = np.zeros(agent_count, dtype=np.float64)
    for agent_id, peers in enumerate(neighbors):
        if not peers:
            means[agent_id] = opinions[agent_id]
            continue
        peer_opinions = opinions[peers]
        means[agent_id] = float(np.mean(peer_opinions))
        stds[agent_id] = float(np.std(peer_opinions))
        disagreements[agent_id] = float(
            np.mean(np.abs(peer_opinions - opinions[agent_id]))
        )
    return means, stds, disagreements


def build_observations(
    opinions: np.ndarray,
    neighbors: list[list[int]],
    recent_drift: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    means, stds, disagreements = neighbor_stats(opinions, neighbors)
    observations = np.column_stack(
        [
            opinions,
            means,
            stds,
            disagreements,
            recent_drift,
            np.ones(len(opinions), dtype=np.float64),
        ]
    )
    return torch.as_tensor(observations, dtype=torch.float32, device=device)


@torch.no_grad()
def collect_acceptance_probabilities(
    agents: list[NeuralOpinionAgent],
    observations: torch.Tensor,
) -> torch.Tensor:
    logits = torch.cat(
        [agent.model(observations[agent.agent_id]).reshape(1) for agent in agents]
    )
    return torch.sigmoid(logits)


def apply_output_average_to_acceptance_probs(
    acceptance_probs: torch.Tensor,
    peer_ids: list[list[int]],
    alpha: float,
) -> torch.Tensor:
    if alpha == 0.0:
        return acceptance_probs.detach().clone()
    result = SocialBlock(alpha=alpha).mix(
        channel=SocialChannel(
            name="acceptance_probability",
            kind=SCALAR_PROBABILITY_CHANNEL,
            commit_mode="tensor_probability",
        ),
        values=acceptance_probs.detach().cpu().numpy(),
        peer_ids=peer_ids,
    )
    return torch.as_tensor(
        result.mixed_values,
        dtype=acceptance_probs.dtype,
        device=acceptance_probs.device,
    )


def train_acceptance_models(
    agents: list[NeuralOpinionAgent],
    observations: torch.Tensor,
    peer_ids: list[list[int]],
    neighbors: list[list[int]],
) -> list[float]:
    losses: list[float] = []
    for agent in agents:
        degree = max(len(neighbors[agent.agent_id]), 1)
        target_value = len(peer_ids[agent.agent_id]) / degree
        target = torch.tensor(
            [target_value],
            dtype=torch.float32,
            device=observations.device,
        )
        logit = agent.model(observations[agent.agent_id])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logit.reshape(1),
            target,
        )
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return losses


def neural_update_opinions(
    opinions: np.ndarray,
    neighbors: list[list[int]],
    agents: list[NeuralOpinionAgent],
    confidence_threshold: float,
    peer_rule: str,
    output_similarity_threshold: float,
    influence_rate: float,
    social_mixer: str,
    social_alpha: float,
    recent_drift: np.ndarray,
    device: torch.device,
    opinion_min: float,
    opinion_max: float,
    learning_enabled: bool,
    delta_scale: float,
) -> tuple[np.ndarray, list[list[int]], torch.Tensor, torch.Tensor, list[float]]:
    observations = build_observations(
        opinions=opinions,
        neighbors=neighbors,
        recent_drift=recent_drift,
        device=device,
    )
    acceptance_pre = collect_acceptance_probabilities(agents, observations)
    peer_ids = select_neural_peer_ids(
        opinions=opinions,
        neighbors=neighbors,
        peer_rule=peer_rule,
        confidence_threshold=confidence_threshold,
        output_similarity_threshold=output_similarity_threshold,
        acceptance_probs=acceptance_pre,
    )
    if social_mixer == "none":
        acceptance_post = acceptance_pre.detach().clone()
    elif social_mixer == "output_average":
        acceptance_post = apply_output_average_to_acceptance_probs(
            acceptance_probs=acceptance_pre,
            peer_ids=peer_ids,
            alpha=social_alpha,
        )
    else:
        raise ValueError(f"Unsupported Toy 3 social mixer: {social_mixer}")

    next_opinions = opinions.astype(np.float64).copy()
    acceptance_np = acceptance_post.detach().cpu().numpy()
    for agent_id, peers in enumerate(peer_ids):
        if not peers:
            continue
        target = float(np.mean(opinions[peers]))
        delta = influence_rate * acceptance_np[agent_id] * (target - opinions[agent_id])
        next_opinions[agent_id] = opinions[agent_id] + float(
            np.clip(delta, -delta_scale, delta_scale)
        )

    losses = (
        train_acceptance_models(
            agents=agents,
            observations=observations,
            peer_ids=peer_ids,
            neighbors=neighbors,
        )
        if learning_enabled
        else [0.0 for _ in agents]
    )
    return (
        np.clip(next_opinions, opinion_min, opinion_max),
        peer_ids,
        acceptance_pre,
        acceptance_post,
        losses,
    )


def rewire_disagreeing_edges(
    graph: nx.Graph,
    opinions: np.ndarray,
    threshold: float,
    rate: float,
    candidate_pool_size: int,
    rng: np.random.Generator,
) -> RewireStats:
    """Drop high-disagreement edges and reconnect to homophilous candidates."""

    if rate <= 0.0:
        return RewireStats(
            rewired_edge_count=0,
            considered_edge_count=graph.number_of_edges(),
            rewired_agents=set(),
        )

    edges = [(int(i), int(j)) for i, j in graph.edges()]
    rewired_count = 0
    rewired_agents: set[int] = set()
    for source, target in edges:
        if not graph.has_edge(source, target):
            continue
        if abs(float(opinions[source]) - float(opinions[target])) <= threshold:
            continue
        if rng.random() > rate:
            continue

        graph.remove_edge(source, target)
        available = [
            candidate
            for candidate in graph.nodes
            if int(candidate) != source
            and int(candidate) != target
            and not graph.has_edge(source, int(candidate))
        ]
        if not available:
            graph.add_edge(source, target)
            continue

        pool_size = min(candidate_pool_size, len(available))
        pool = rng.choice(
            np.asarray(available, dtype=np.int64), size=pool_size, replace=False
        )
        replacement = int(
            min(
                pool,
                key=lambda node: abs(float(opinions[source]) - float(opinions[node])),
            )
        )
        if replacement == source or graph.has_edge(source, replacement):
            graph.add_edge(source, target)
            continue
        graph.add_edge(source, replacement)
        rewired_count += 1
        rewired_agents.update({source, target, replacement})

    return RewireStats(
        rewired_edge_count=rewired_count,
        considered_edge_count=len(edges),
        rewired_agents=rewired_agents,
    )


def edge_disagreements(graph: nx.Graph, opinions: np.ndarray) -> list[float]:
    return [
        abs(float(opinions[int(source)]) - float(opinions[int(target)]))
        for source, target in graph.edges()
    ]


def node_edge_disagreement(
    opinions: np.ndarray,
    neighbors: list[list[int]],
) -> np.ndarray:
    disagreements = np.zeros(len(opinions), dtype=np.float64)
    for agent_id, peers in enumerate(neighbors):
        if not peers:
            continue
        disagreements[agent_id] = float(
            np.mean(np.abs(opinions[peers] - opinions[agent_id]))
        )
    return disagreements


def opinion_cluster_count(opinions: np.ndarray, gap_threshold: float) -> int:
    if len(opinions) == 0:
        return 0
    sorted_opinions = np.sort(opinions)
    gaps = np.diff(sorted_opinions)
    return int(1 + np.sum(gaps > gap_threshold))


def opinion_assortativity(graph: nx.Graph, opinions: np.ndarray) -> float:
    if graph.number_of_edges() < 2:
        return 0.0
    source_values = np.asarray(
        [opinions[int(source)] for source, _ in graph.edges()],
        dtype=np.float64,
    )
    target_values = np.asarray(
        [opinions[int(target)] for _, target in graph.edges()],
        dtype=np.float64,
    )
    if np.std(source_values) == 0.0 or np.std(target_values) == 0.0:
        return 1.0
    return float(np.corrcoef(source_values, target_values)[0, 1])


def aggregate_metrics(
    config: Toy3Config,
    epoch: int,
    opinions: np.ndarray,
    graph: nx.Graph,
    peer_ids: list[list[int]],
    rewired_edge_count: int,
    cumulative_rewired_edge_count: int,
    considered_edge_count: int | None = None,
) -> dict[str, object]:
    disagreements = edge_disagreements(graph, opinions)
    mean_edge_disagreement = float(np.mean(disagreements)) if disagreements else 0.0
    high_disagreement_fraction = (
        float(np.mean(np.asarray(disagreements) > config.rewiring.threshold))
        if disagreements
        else 0.0
    )
    components = list(nx.connected_components(graph))
    largest_component_fraction = (
        max(len(component) for component in components) / config.agents.count
        if components
        else 0.0
    )
    opinion_range = config.environment.opinion_max - config.environment.opinion_min
    max_variance = (opinion_range / 2.0) ** 2
    variance = float(np.var(opinions))
    rewiring_denominator = (
        graph.number_of_edges()
        if considered_edge_count is None
        else considered_edge_count
    )
    realized_rewiring_rate = (
        rewired_edge_count / rewiring_denominator if rewiring_denominator > 0 else 0.0
    )
    return {
        "run_id": config.run.name,
        "seed": config.run.seed,
        "epoch": epoch,
        "policy_rule": config.policy.update_rule,
        "domain_confidence_threshold": config.policy.confidence_threshold,
        "coordination_mixer": config.coordination.mixer,
        "coordination_peer_rule": config.coordination.peer_rule,
        "domain_rewiring_enabled": config.rewiring.enabled,
        "domain_rewiring_threshold": config.rewiring.threshold,
        "domain_rewiring_rate": config.rewiring.rate,
        "domain_realized_rewiring_rate": realized_rewiring_rate,
        "domain_rewired_edge_count": rewired_edge_count,
        "domain_cumulative_rewired_edge_count": cumulative_rewired_edge_count,
        "domain_opinion_mean": float(np.mean(opinions)),
        "domain_opinion_variance": variance,
        "domain_polarization_index": float(np.clip(variance / max_variance, 0.0, 1.0)),
        "domain_opinion_cluster_count": opinion_cluster_count(
            opinions,
            config.policy.confidence_threshold,
        ),
        "domain_mean_edge_disagreement": mean_edge_disagreement,
        "domain_high_disagreement_edge_fraction": high_disagreement_fraction,
        "fragmentation_components": nx.number_connected_components(graph),
        "domain_largest_connected_component_fraction": largest_component_fraction,
        "mean_peer_count": float(np.mean([len(peers) for peers in peer_ids])),
        "domain_opinion_assortativity": opinion_assortativity(graph, opinions),
    }


def make_run_dir(config: Toy3Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (
        config.run.output_dir
        / f"{timestamp}_{config.run.name}_seed{config.run.seed:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(config_path: Path, config: Toy3Config, run_dir: Path) -> None:
    write_run_metadata_artifacts(
        config_path=config_path,
        config=config,
        run_dir=run_dir,
        toy="toy3",
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "toy": "toy3",
            "policy_rule": config.policy.update_rule,
            "domain_confidence_threshold": config.policy.confidence_threshold,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_rewiring_enabled": config.rewiring.enabled,
            "domain_rewiring_threshold": config.rewiring.threshold,
            "domain_rewiring_rate": config.rewiring.rate,
            "agent_count": config.agents.count,
            "domain_graph_type": config.graph.type,
            "domain_graph_k": config.graph.k,
            "domain_graph_rewire_probability": config.graph.rewire_probability,
        },
    )


def run_toy3(config: Toy3Config, config_path: Path) -> DomainToyResult:
    """Run Toy 3 from a validated config."""

    if config.dynamics.update_rule == "neural_policy":
        if config.agents.model.input_dim != 6:
            raise ValueError("Toy 3 neural_policy expects model.input_dim=6")
        if config.agents.model.output_dim != 1:
            raise ValueError("Toy 3 neural_policy expects model.output_dim=1")

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    rewire_rng = np.random.default_rng(config.run.seed + 1_000_003)
    device = resolve_torch_device(config.simulation.device)

    graph = build_opinion_graph(config)
    opinions = initialize_opinions(config, rng)
    previous_opinions = opinions.copy()
    agents = (
        create_agents(config, device=device)
        if config.dynamics.update_rule == "neural_policy"
        else []
    )

    run_dir = make_run_dir(config)
    write_run_metadata(config_path=config_path, config=config, run_dir=run_dir)
    micro_writer = CsvLogWriter(run_dir / "micro_state.csv", TOY3_MICRO_STATE_FIELDS)
    aggregate_writer = CsvLogWriter(
        run_dir / "aggregate_metrics.csv",
        TOY3_AGGREGATE_FIELDS,
    )

    cumulative_rewired_edge_count = 0
    final_row: dict[str, object] | None = None

    try:
        neighbors = graph_neighbors(graph, config.agents.count)
        if config.dynamics.update_rule == "neural_policy":
            initial_observations = build_observations(
                opinions=opinions,
                neighbors=neighbors,
                recent_drift=np.zeros_like(opinions),
                device=device,
            )
            initial_acceptance = collect_acceptance_probabilities(
                agents,
                initial_observations,
            )
            peer_ids = select_neural_peer_ids(
                opinions=opinions,
                neighbors=neighbors,
                peer_rule=config.social.peer_rule,
                confidence_threshold=config.dynamics.confidence_threshold,
                output_similarity_threshold=config.social.threshold,
                acceptance_probs=initial_acceptance,
            )
        else:
            peer_ids = compatible_peer_ids(
                opinions,
                neighbors,
                config.dynamics.confidence_threshold,
            )
        if config.logging.aggregate_metrics:
            initial_row = aggregate_metrics(
                config=config,
                epoch=0,
                opinions=opinions,
                graph=graph,
                peer_ids=peer_ids,
                rewired_edge_count=0,
                cumulative_rewired_edge_count=0,
                considered_edge_count=graph.number_of_edges(),
            )
            aggregate_writer.write(initial_row)
            final_row = initial_row

        for epoch in range(1, config.simulation.epochs + 1):
            neighbors = graph_neighbors(graph, config.agents.count)
            opinion_pre_update = opinions.copy()
            recent_drift = opinion_pre_update - previous_opinions

            acceptance_pre: torch.Tensor | None = None
            acceptance_post: torch.Tensor | None = None
            if config.dynamics.update_rule == "hk":
                opinions, peer_ids = hk_update_opinions(
                    opinions=opinions,
                    neighbors=neighbors,
                    confidence_threshold=config.dynamics.confidence_threshold,
                    influence_rate=config.dynamics.influence_rate,
                    opinion_min=config.environment.opinion_min,
                    opinion_max=config.environment.opinion_max,
                )
            elif config.dynamics.update_rule == "deffuant":
                opinions, peer_ids = deffuant_update_opinions(
                    opinions=opinions,
                    graph=graph,
                    confidence_threshold=config.dynamics.confidence_threshold,
                    mu=config.dynamics.deffuant_mu,
                    rng=rng,
                    opinion_min=config.environment.opinion_min,
                    opinion_max=config.environment.opinion_max,
                )
            elif config.dynamics.update_rule == "neural_policy":
                opinions, peer_ids, acceptance_pre, acceptance_post, _ = (
                    neural_update_opinions(
                        opinions=opinions,
                        neighbors=neighbors,
                        agents=agents,
                        confidence_threshold=config.dynamics.confidence_threshold,
                        peer_rule=config.social.peer_rule,
                        output_similarity_threshold=config.social.threshold,
                        influence_rate=config.dynamics.influence_rate,
                        social_mixer=config.social.mixer,
                        social_alpha=config.social.alpha,
                        recent_drift=recent_drift,
                        device=device,
                        opinion_min=config.environment.opinion_min,
                        opinion_max=config.environment.opinion_max,
                        learning_enabled=config.dynamics.neural_learning_rate > 0.0,
                        delta_scale=config.dynamics.neural_delta_scale,
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported Toy 3 update rule: {config.dynamics.update_rule}"
                )

            rewire_stats = (
                rewire_disagreeing_edges(
                    graph=graph,
                    opinions=opinions,
                    threshold=config.rewiring.threshold,
                    rate=config.rewiring.rate,
                    candidate_pool_size=config.rewiring.candidate_pool_size,
                    rng=rewire_rng,
                )
                if config.rewiring.enabled
                else RewireStats(
                    rewired_edge_count=0,
                    considered_edge_count=graph.number_of_edges(),
                    rewired_agents=set(),
                )
            )
            cumulative_rewired_edge_count += rewire_stats.rewired_edge_count

            neighbors = graph_neighbors(graph, config.agents.count)
            if (
                config.dynamics.update_rule == "neural_policy"
                and acceptance_post is not None
            ):
                peer_ids = select_neural_peer_ids(
                    opinions=opinions,
                    neighbors=neighbors,
                    peer_rule=config.social.peer_rule,
                    confidence_threshold=config.dynamics.confidence_threshold,
                    output_similarity_threshold=config.social.threshold,
                    acceptance_probs=acceptance_post,
                )
            else:
                peer_ids = compatible_peer_ids(
                    opinions,
                    neighbors,
                    config.dynamics.confidence_threshold,
                )
            row = aggregate_metrics(
                config=config,
                epoch=epoch,
                opinions=opinions,
                graph=graph,
                peer_ids=peer_ids,
                rewired_edge_count=rewire_stats.rewired_edge_count,
                cumulative_rewired_edge_count=cumulative_rewired_edge_count,
                considered_edge_count=rewire_stats.considered_edge_count,
            )
            final_row = row
            if config.logging.aggregate_metrics:
                aggregate_writer.write(row)

            if config.logging.micro_state and epoch % config.logging.interval == 0:
                peer_graph = graph_from_peer_ids(config.agents.count, peer_ids)
                components = component_map(peer_graph)
                means, stds, local_disagreements = neighbor_stats(opinions, neighbors)
                node_disagreements = node_edge_disagreement(opinions, neighbors)
                opinion_deltas = opinions - opinion_pre_update
                revised_mask = np.abs(opinion_deltas) > 1e-12
                acceptance_pre_np = (
                    acceptance_pre.detach().cpu().numpy()
                    if acceptance_pre is not None
                    else None
                )
                acceptance_post_np = (
                    acceptance_post.detach().cpu().numpy()
                    if acceptance_post is not None
                    else None
                )
                for agent_id in range(config.agents.count):
                    micro_writer.write(
                        {
                            "run_id": config.run.name,
                            "seed": config.run.seed,
                            "epoch": epoch,
                            "agent_id": agent_id,
                            "policy_rule": config.policy.update_rule,
                            "coordination_mixer": config.coordination.mixer,
                            "coordination_peer_rule": config.coordination.peer_rule,
                            "domain_opinion": float(opinions[agent_id]),
                            "domain_opinion_pre_update": float(
                                opinion_pre_update[agent_id]
                            ),
                            "domain_opinion_delta": float(opinion_deltas[agent_id]),
                            "domain_neighbor_opinion_mean": float(means[agent_id]),
                            "domain_neighbor_opinion_std": float(stds[agent_id]),
                            "domain_local_disagreement": float(
                                local_disagreements[agent_id]
                            ),
                            "domain_degree": len(neighbors[agent_id]),
                            "peer_ids": peer_ids[agent_id],
                            "peer_count": len(peer_ids[agent_id]),
                            "component_id": components.get(agent_id, -1),
                            "domain_edge_disagreement": float(
                                node_disagreements[agent_id]
                            ),
                            "domain_acceptance_probability_pre_social": (
                                float(acceptance_pre_np[agent_id])
                                if acceptance_pre_np is not None
                                else ""
                            ),
                            "domain_acceptance_probability_post_social": (
                                float(acceptance_post_np[agent_id])
                                if acceptance_post_np is not None
                                else ""
                            ),
                            "revised": bool(revised_mask[agent_id]),
                            "domain_rewired": agent_id in rewire_stats.rewired_agents,
                        }
                    )

            previous_opinions = opinion_pre_update
    finally:
        micro_writer.close()
        aggregate_writer.close()

    if final_row is None:
        final_row = aggregate_metrics(
            config=config,
            epoch=0,
            opinions=opinions,
            graph=graph,
            peer_ids=peer_ids,
            rewired_edge_count=0,
            cumulative_rewired_edge_count=cumulative_rewired_edge_count,
            considered_edge_count=graph.number_of_edges(),
        )

    domain_metrics = {
        "domain_final_opinion_mean": final_row["domain_opinion_mean"],
        "domain_final_opinion_variance": final_row["domain_opinion_variance"],
        "domain_final_polarization_index": final_row["domain_polarization_index"],
        "domain_final_opinion_cluster_count": final_row["domain_opinion_cluster_count"],
        "domain_final_mean_edge_disagreement": final_row[
            "domain_mean_edge_disagreement"
        ],
        "domain_final_largest_connected_component_fraction": final_row[
            "domain_largest_connected_component_fraction"
        ],
        "domain_cumulative_rewired_edge_count": cumulative_rewired_edge_count,
    }
    write_domain_summary_artifact(
        run_dir=run_dir,
        toy="toy3",
        final_fragmentation_components=final_row["fragmentation_components"],
        domain_metrics=domain_metrics,
    )
    return DomainToyResult(
        run_dir=run_dir,
        toy="toy3",
        final_fragmentation_components=int(final_row["fragmentation_components"]),
        domain_metrics=domain_metrics,
    )
