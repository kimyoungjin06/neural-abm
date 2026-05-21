"""Neural HK Classification toy model runner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from neural_abm.accelerator import resolve_torch_device
from neural_abm.config import Toy1Config
from neural_abm.core import (
    ClassificationMLP,
    NeuralClassificationAgent,
    clone_state_dict,
    flatten_parameters,
    parameter_delta_norm,
)
from neural_abm.graphs import build_graph, component_map, graph_from_peer_ids
from neural_abm.logging import AGGREGATE_FIELDS, MICRO_STATE_FIELDS, CsvLogWriter
from neural_abm.metrics import (
    accuracy_from_probs,
    consensus,
    cross_entropy_from_probs,
    edge_entropy,
    entropy_mean,
    js_divergence_np,
    pairwise_output_js,
    polarization_clusters,
)
from neural_abm.mixers import (
    apply_latent_average,
    apply_output_average,
    apply_parameter_aligned_average,
    apply_parameter_average,
    select_peers,
)
from neural_abm.results import (
    DomainToyResult,
    write_domain_summary_artifact,
    write_run_metadata_artifacts,
)


@dataclass
class DatasetSplit:
    x: np.ndarray
    y: np.ndarray


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def boundary_value(x1: np.ndarray) -> np.ndarray:
    return 0.35 * np.sin(3.0 * math.pi * x1)


def make_split(size: int, label_noise: float, rng: np.random.Generator) -> DatasetSplit:
    x = rng.uniform(-1.0, 1.0, size=(size, 2)).astype(np.float32)
    y = (x[:, 1] > boundary_value(x[:, 0])).astype(np.int64)
    if label_noise > 0:
        flips = rng.random(size) < label_noise
        y[flips] = 1 - y[flips]
    return DatasetSplit(x=x, y=y)


def select_indices_for_group(
    group_name: str,
    pool: DatasetSplit,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    x = pool.x
    if group_name == "left_region":
        candidates = np.flatnonzero(x[:, 0] < -0.2)
    elif group_name == "right_region":
        candidates = np.flatnonzero(x[:, 0] > 0.2)
    elif group_name == "boundary_region":
        distance = np.abs(x[:, 1] - boundary_value(x[:, 0]))
        cutoff = np.quantile(distance, 0.35)
        candidates = np.flatnonzero(distance <= cutoff)
    elif group_name in {"noisy_labels", "small_balanced"}:
        candidates = np.arange(len(x))
    else:
        raise ValueError(f"Unsupported shard group: {group_name}")
    replace = len(candidates) < count
    return rng.choice(candidates, size=count, replace=replace)


def apply_extra_label_noise(
    y: np.ndarray, label_noise: float | None, rng: np.random.Generator
) -> np.ndarray:
    copied = y.copy()
    if label_noise is None or label_noise <= 0:
        return copied
    flips = rng.random(len(copied)) < label_noise
    copied[flips] = 1 - copied[flips]
    return copied


def make_model(config: Toy1Config) -> ClassificationMLP:
    model_config = config.agents.model
    return ClassificationMLP(
        input_dim=model_config.input_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=model_config.output_dim,
    )


def make_optimizer(model: torch.nn.Module, config: Toy1Config) -> torch.optim.Optimizer:
    if config.agents.optimizer.name == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=config.agents.optimizer.learning_rate
        )
    raise ValueError(f"Unsupported optimizer: {config.agents.optimizer.name}")


def create_agents(
    config: Toy1Config,
    train_pool: DatasetSplit,
    device: torch.device,
    rng: np.random.Generator,
) -> list[NeuralClassificationAgent]:
    expected = sum(group.count for group in config.agents.shards.groups.values())
    if expected != config.agents.count:
        raise ValueError(
            f"Shard group count ({expected}) does not match agent count "
            f"({config.agents.count})"
        )

    base_state = None
    if config.agents.init_mode == "same_init":
        torch.manual_seed(config.run.seed)
        base_state = clone_state_dict(make_model(config))

    agents: list[NeuralClassificationAgent] = []
    agent_id = 0
    for group_name, group in config.agents.shards.groups.items():
        for _ in range(group.count):
            if config.agents.init_mode == "independent_init":
                torch.manual_seed(config.run.seed * 1000 + agent_id)
            model = make_model(config).to(device)
            if base_state is not None:
                model.load_state_dict(base_state)
            optimizer = make_optimizer(model, config)
            indices = select_indices_for_group(
                group_name=group_name,
                pool=train_pool,
                count=group.samples_per_agent,
                rng=rng,
            )
            shard_x = train_pool.x[indices]
            shard_y = apply_extra_label_noise(
                train_pool.y[indices], group.label_noise, rng
            )
            agent = NeuralClassificationAgent(
                agent_id=agent_id,
                shard_group=group_name,
                model=model,
                optimizer=optimizer,
                train_x=torch.as_tensor(shard_x, device=device),
                train_y=torch.as_tensor(shard_y, dtype=torch.long, device=device),
            )
            agents.append(agent)
            agent_id += 1
    return agents


@torch.no_grad()
def collect_probe_state(
    agents: list[NeuralClassificationAgent], probe_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = []
    hidden = []
    latent_summary = []
    params = []
    for agent in agents:
        logits, hidden_values = agent.model(probe_x, return_hidden=True)
        agent_probs = torch.softmax(logits, dim=-1)
        probs.append(agent_probs)
        hidden.append(hidden_values)
        latent_summary.append(hidden_values.mean(dim=0))
        params.append(flatten_parameters(agent.model))
    return (
        torch.stack(probs, dim=0),
        torch.stack(hidden, dim=0),
        torch.stack(latent_summary, dim=0),
        torch.stack(params, dim=0),
    )


@torch.no_grad()
def evaluate_agents(
    agents: list[NeuralClassificationAgent],
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[list[float], list[float]]:
    accuracies = []
    losses = []
    for agent in agents:
        probs = agent.predict_proba(x)
        accuracies.append(accuracy_from_probs(probs, y))
        losses.append(cross_entropy_from_probs(probs, y))
    return accuracies, losses


def make_run_dir(config: Toy1Config) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (
        config.run.output_dir
        / f"{timestamp}_{config.run.name}_seed{config.run.seed:02d}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_run_metadata(config_path: Path, config: Toy1Config, run_dir: Path) -> None:
    write_run_metadata_artifacts(
        config_path=config_path,
        config=config,
        run_dir=run_dir,
        toy="toy1",
        resolved_config_format="json",
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "toy": "toy1",
            "policy_rule": "neural_hk",
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "model_init_mode": config.agents.init_mode,
        },
    )


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [
        sorted(int(node) for node in graph.neighbors(i)) for i in range(agent_count)
    ]


def write_probe_prediction_snapshot(
    run_dir: Path,
    epoch: int,
    probe_probs: np.ndarray,
    agents: list[NeuralClassificationAgent],
    peer_ids: list[list[int]],
    components: dict[int, int],
) -> None:
    """Write per-agent probe predictions for behavior differentiation analyses."""

    snapshot_dir = run_dir / "probe_predictions"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    max_peer_count = max((len(peers) for peers in peer_ids), default=0)
    peer_matrix = np.full((len(agents), max_peer_count), fill_value=-1, dtype=np.int64)
    for agent_id, peers in enumerate(peer_ids):
        if peers:
            peer_matrix[agent_id, : len(peers)] = peers

    np.savez_compressed(
        snapshot_dir / f"epoch_{epoch:04d}.npz",
        epoch=np.array(epoch, dtype=np.int64),
        probe_probs=probe_probs.astype(np.float32),
        agent_ids=np.array([agent.agent_id for agent in agents], dtype=np.int64),
        shard_groups=np.array([agent.shard_group for agent in agents]),
        component_ids=np.array(
            [components.get(agent.agent_id, -1) for agent in agents],
            dtype=np.int64,
        ),
        peer_ids=peer_matrix,
    )


def run_toy1(config: Toy1Config, config_path: Path) -> DomainToyResult:
    """Run the Toy 1 simulation from a validated config."""

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    device = resolve_torch_device(config.simulation.device)

    train_pool = make_split(
        size=config.data.train_pool_size,
        label_noise=config.data.label_noise,
        rng=rng,
    )
    probe = make_split(size=config.data.probe_size, label_noise=0.0, rng=rng)
    test = make_split(size=config.data.test_size, label_noise=0.0, rng=rng)

    probe_x = torch.as_tensor(probe.x, device=device)
    probe_y = torch.as_tensor(probe.y, dtype=torch.long, device=device)
    test_x = torch.as_tensor(test.x, device=device)
    test_y = torch.as_tensor(test.y, dtype=torch.long, device=device)

    agents = create_agents(config=config, train_pool=train_pool, device=device, rng=rng)
    candidate_graph = build_graph(config.graph, config.agents.count, config.run.seed)
    neighbors = graph_neighbors(candidate_graph, config.agents.count)

    run_dir = make_run_dir(config)
    write_run_metadata(config_path=config_path, config=config, run_dir=run_dir)

    micro_writer = CsvLogWriter(run_dir / "micro_state.csv", MICRO_STATE_FIELDS)
    aggregate_writer = CsvLogWriter(run_dir / "aggregate_metrics.csv", AGGREGATE_FIELDS)

    final_global_accuracy = 0.0
    final_consensus = 0.0
    final_fragmentation = 0

    try:
        for epoch in range(1, config.simulation.epochs + 1):
            local_losses = [
                agent.local_update(
                    batch_size=config.agents.training.local_batch_size,
                    steps=config.agents.training.local_steps_per_epoch,
                )
                for agent in agents
            ]

            previous_states = [clone_state_dict(agent.model) for agent in agents]
            previous_probs, previous_hidden, latent_vectors, state_vectors = (
                collect_probe_state(agents, probe_x)
            )

            peer_ids, _ = select_peers(
                graph_neighbors=neighbors,
                peer_rule=config.social.peer_rule,
                threshold=config.social.threshold,
                state_vectors=state_vectors,
                latent_vectors=latent_vectors,
                probe_probs=previous_probs.detach().cpu().numpy(),
                parameter_states=previous_states,
            )
            if config.social.mixer == "none":
                peer_ids = [[] for _ in agents]

            social_step_result = None
            if config.social.mixer == "parameter_average":
                social_step_result = apply_parameter_average(
                    agents=agents,
                    peer_ids=peer_ids,
                    alpha=config.social.alpha,
                    previous_states=previous_states,
                )
            elif config.social.mixer == "parameter_aligned_average":
                social_step_result = apply_parameter_aligned_average(
                    agents=agents,
                    peer_ids=peer_ids,
                    alpha=config.social.alpha,
                    previous_states=previous_states,
                )
            elif config.social.mixer == "output_average":
                social_step_result = apply_output_average(
                    agents=agents,
                    peer_ids=peer_ids,
                    alpha=config.social.alpha,
                    probe_x=probe_x,
                    previous_probs=previous_probs.detach(),
                )
            elif config.social.mixer == "latent_average":
                social_step_result = apply_latent_average(
                    agents=agents,
                    peer_ids=peer_ids,
                    alpha=config.social.alpha,
                    probe_x=probe_x,
                    previous_hidden=previous_hidden.detach(),
                )
            elif config.social.mixer == "none":
                pass
            else:
                raise ValueError(f"Unsupported mixer: {config.social.mixer}")

            final_probs, _, final_latents, _ = collect_probe_state(agents, probe_x)
            probe_probs_np = final_probs.detach().cpu().numpy()
            js_matrix = pairwise_output_js(probe_probs_np)
            consensus_value = consensus(probe_probs_np)
            mean_js = float(np.mean(js_matrix[np.triu_indices(len(agents), k=1)]))

            peer_graph = graph_from_peer_ids(config.agents.count, peer_ids)
            components = component_map(peer_graph)
            fragmentation = nx.number_connected_components(peer_graph)
            cluster_count = polarization_clusters(
                js_matrix=js_matrix,
                similarity_threshold=config.social.threshold,
            )
            peer_counts = [len(peers) for peers in peer_ids]
            entropy_value = edge_entropy(peer_ids, config.agents.count)
            social_aggregate = (
                social_step_result.diagnostics.aggregate_row()
                if social_step_result is not None
                else {
                    "social_channel": "",
                    "commit_mode": "none",
                    "mean_social_loss": 0.0,
                    "mean_social_update_norm": 0.0,
                    "max_social_update_norm": 0.0,
                    "active_social_agent_count": 0,
                }
            )

            global_accuracies, _ = evaluate_agents(agents, test_x, test_y)
            probe_accuracies, _ = evaluate_agents(agents, probe_x, probe_y)
            final_global_accuracy = float(np.mean(global_accuracies))
            final_consensus = consensus_value
            final_fragmentation = fragmentation

            aggregate_writer.write(
                {
                    "run_id": config.run.name,
                    "seed": config.run.seed,
                    "epoch": epoch,
                    "coordination_mixer": config.coordination.mixer,
                    "coordination_peer_rule": config.coordination.peer_rule,
                    "model_init_mode": config.agents.init_mode,
                    "domain_mean_global_accuracy": final_global_accuracy,
                    "domain_mean_probe_accuracy": float(np.mean(probe_accuracies)),
                    "domain_mean_consensus": consensus_value,
                    "domain_mean_output_js": mean_js,
                    "domain_polarization_clusters": cluster_count,
                    "fragmentation_components": fragmentation,
                    "mean_peer_count": float(np.mean(peer_counts)),
                    **social_aggregate,
                    "edge_entropy": entropy_value,
                }
            )

            if (
                config.logging.probe_predictions
                and epoch % config.logging.probe_prediction_interval == 0
            ):
                write_probe_prediction_snapshot(
                    run_dir=run_dir,
                    epoch=epoch,
                    probe_probs=probe_probs_np,
                    agents=agents,
                    peer_ids=peer_ids,
                    components=components,
                )

            if config.logging.micro_state and epoch % config.logging.interval == 0:
                population_mean = probe_probs_np.mean(axis=0)
                for i, agent in enumerate(agents):
                    after_state = clone_state_dict(agent.model)
                    param_vector = flatten_parameters(agent.model)
                    peer_count = len(peer_ids[i])
                    edge_weights = (
                        [1.0 / peer_count for _ in peer_ids[i]]
                        if peer_count > 0
                        else []
                    )
                    output_js_to_mean = js_divergence_np(
                        probe_probs_np[i], population_mean
                    )
                    probe_entropy = entropy_mean(final_probs[i])
                    social_micro = (
                        social_step_result.diagnostics.micro_row(i)
                        if social_step_result is not None
                        else {
                            "social_channel": "",
                            "commit_mode": "none",
                            "social_loss": 0.0,
                            "social_update_norm": 0.0,
                        }
                    )
                    micro_writer.write(
                        {
                            "run_id": config.run.name,
                            "seed": config.run.seed,
                            "epoch": epoch,
                            "agent_id": agent.agent_id,
                            "domain_shard_group": agent.shard_group,
                            "coordination_mixer": config.coordination.mixer,
                            "coordination_peer_rule": config.coordination.peer_rule,
                            "model_init_mode": config.agents.init_mode,
                            **social_micro,
                            "local_loss": local_losses[i],
                            "domain_global_accuracy": global_accuracies[i],
                            "domain_probe_accuracy": probe_accuracies[i],
                            "domain_probe_entropy": probe_entropy,
                            "domain_confidence": 1.0 - probe_entropy / math.log(2.0),
                            "peer_ids": peer_ids[i],
                            "edge_weights": edge_weights,
                            "peer_count": peer_count,
                            "component_id": components.get(i, -1),
                            "message_norm": float(
                                torch.linalg.vector_norm(final_probs[i]).cpu()
                            ),
                            "latent_norm": float(
                                torch.linalg.vector_norm(final_latents[i]).cpu()
                            ),
                            "param_norm": float(
                                torch.linalg.vector_norm(param_vector).cpu()
                            ),
                            "param_delta_norm": parameter_delta_norm(
                                previous_states[i], after_state
                            ),
                            "domain_output_js_to_population_mean": output_js_to_mean,
                        }
                    )
    finally:
        micro_writer.close()
        aggregate_writer.close()

    domain_metrics = {
        "domain_final_mean_global_accuracy": final_global_accuracy,
        "domain_final_mean_consensus": final_consensus,
    }
    write_domain_summary_artifact(
        run_dir=run_dir,
        toy="toy1",
        final_fragmentation_components=final_fragmentation,
        domain_metrics=domain_metrics,
    )
    return DomainToyResult(
        run_dir=run_dir,
        toy="toy1",
        final_fragmentation_components=final_fragmentation,
        domain_metrics=domain_metrics,
    )
