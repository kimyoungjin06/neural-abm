"""Toy 10: dynamic-network market/ecology ABM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from neural_abm.accelerator import resolve_torch_device
from neural_abm.config import Toy10Config
from neural_abm.domain_runner import (
    DomainRunSettings,
    DomainToyRunner,
    make_timestamped_run_dir,
)
from neural_abm.graphs import build_graph, component_map, graph_from_peer_ids
from neural_abm.logging import CsvLogWriter
from neural_abm.results import (
    DomainToyResult,
    write_run_metadata_artifacts,
)
from neural_abm.mixers import apply_bounded_scalar_output_average
from neural_abm.social import (
    empty_peers,
    select_bounded_scalar_output_peers,
)


TOY10_MICRO_STATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_harvest_intensity",
    "domain_price_expectation",
    "domain_conservation_norm",
    "domain_payoff",
    "domain_payoff_ema",
    "domain_market_price",
    "domain_resource_level",
    "domain_resource_fraction",
    "domain_local_price_expectation",
    "domain_local_conservation_norm",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]


TOY10_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_resource_level",
    "domain_resource_fraction",
    "domain_market_price",
    "domain_market_imbalance",
    "domain_mean_harvest_intensity",
    "domain_harvest_variance",
    "domain_mean_price_expectation",
    "domain_mean_conservation_norm",
    "domain_mean_payoff",
    "domain_payoff_variance",
    "domain_cumulative_rewired_edge_count",
    "fragmentation_components",
    "mean_peer_count",
    "mean_social_loss",
    "mean_social_update_norm",
]


@dataclass(frozen=True)
class Toy10StepResult:
    harvest_intensities: np.ndarray
    price_expectations: np.ndarray
    conservation_norms: np.ndarray
    local_price_expectations: np.ndarray
    local_conservation_norms: np.ndarray
    payoffs: np.ndarray
    payoff_ema: np.ndarray
    resource_level: float
    market_price: float
    market_imbalance: float
    cumulative_rewired_edge_count: int
    peer_ids: list[list[int]]
    social_losses: list[float]
    social_update_norms: list[float]


@dataclass
class Toy10RunState:
    rng: np.random.Generator
    graph: nx.Graph
    price_expectations: np.ndarray
    conservation_norms: np.ndarray
    resource_level: float
    payoff_ema: np.ndarray
    cumulative_rewired_edge_count: int = 0


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [
        sorted(int(node) for node in graph.neighbors(agent_id))
        for agent_id in range(agent_count)
    ]


def initialize_channel(
    *,
    mean: float,
    std: float,
    count: int,
    same_init: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    if same_init:
        return np.full(count, mean, dtype=np.float64)
    return np.clip(rng.normal(loc=mean, scale=std, size=count), 0.0, 1.0).astype(
        np.float64
    )


def select_peer_ids(
    price_expectations: np.ndarray,
    conservation_norms: np.ndarray,
    neighbors: list[list[int]],
    config: Toy10Config,
) -> list[list[int]]:
    if config.coordination.mixer == "none":
        return empty_peers(len(price_expectations))
    composite = np.clip(0.5 * (price_expectations + conservation_norms), 0.0, 1.0)
    return select_bounded_scalar_output_peers(
        neighbors=neighbors,
        values=composite,
        peer_rule=config.coordination.peer_rule,
        threshold=config.coordination.threshold,
        lower_bound=0.0,
        upper_bound=1.0,
    ).peer_ids


def mix_channel(
    values: np.ndarray,
    peer_ids: list[list[int]],
    config: Toy10Config,
    *,
    channel: str,
) -> tuple[np.ndarray, list[float], list[float]]:
    if config.coordination.mixer == "none":
        return (
            values.copy(),
            [0.0 for _ in range(len(values))],
            [0.0 for _ in range(len(values))],
        )
    result = apply_bounded_scalar_output_average(
        values=values,
        peer_ids=peer_ids,
        alpha=config.coordination.alpha,
        lower_bound=0.0,
        upper_bound=1.0,
        channel=channel,
        commit_mode="multi_channel_market_commit",
    )
    return (
        np.asarray(result.mix.mixed_values, dtype=np.float64),
        result.commit.losses,
        result.mix.update_norms,
    )


def harvest_from_channels(
    price_expectations: np.ndarray,
    conservation_norms: np.ndarray,
    rng: np.random.Generator,
    config: Toy10Config,
    *,
    base_price_expectations: np.ndarray | None = None,
    base_conservation_norms: np.ndarray | None = None,
) -> np.ndarray:
    harvest = price_expectations * (
        1.0 - config.policy.conservation_harvest_weight * conservation_norms
    )
    if (
        config.coordination.mixer != "none"
        and base_price_expectations is not None
        and base_conservation_norms is not None
    ):
        base_harvest = base_price_expectations * (
            1.0
            - config.policy.conservation_harvest_weight * base_conservation_norms
        )
        if config.policy.social_harvest_gain != 1.0:
            harvest = base_harvest + config.policy.social_harvest_gain * (
                harvest - base_harvest
            )
        social_disagreement = 0.5 * (
            np.abs(price_expectations - base_price_expectations)
            + np.abs(conservation_norms - base_conservation_norms)
        )
        harvest -= config.policy.social_disagreement_penalty * social_disagreement
    if config.policy.exploration_std > 0.0:
        harvest += rng.normal(0.0, config.policy.exploration_std, size=len(harvest))
    return np.clip(harvest, 0.0, 1.0)


def market_price(
    harvest_intensities: np.ndarray,
    price_expectations: np.ndarray,
    config: Toy10Config,
) -> tuple[float, float]:
    demand = float(np.mean(price_expectations))
    supply = float(np.mean(harvest_intensities))
    imbalance = demand - supply
    price = (
        config.environment.base_price
        + config.environment.demand_sensitivity * demand
        - config.environment.supply_sensitivity * supply
    )
    return float(np.clip(price, 0.0, 1.0)), imbalance


def compute_payoffs(
    harvest_intensities: np.ndarray,
    resource_level: float,
    market_price_value: float,
    config: Toy10Config,
) -> np.ndarray:
    resource_fraction = resource_level / config.environment.resource_carrying_capacity
    revenue = market_price_value * harvest_intensities * resource_fraction
    costs = config.environment.extraction_cost * np.square(harvest_intensities)
    scarcity_penalty = (1.0 - resource_fraction) * harvest_intensities * 0.20
    return revenue - costs - scarcity_penalty


def update_resource(
    resource_level: float,
    harvest_intensities: np.ndarray,
    config: Toy10Config,
) -> float:
    recovered = resource_level + config.environment.resource_recovery_rate * (
        config.environment.resource_carrying_capacity - resource_level
    )
    extracted = config.environment.extraction_scale * float(np.mean(harvest_intensities))
    return float(
        np.clip(
            recovered - extracted,
            0.0,
            config.environment.resource_carrying_capacity,
        )
    )


def local_channel_means(values: np.ndarray, neighbors: list[list[int]]) -> np.ndarray:
    means = np.zeros(len(values), dtype=np.float64)
    for agent_id, peers in enumerate(neighbors):
        if peers:
            means[agent_id] = float(np.mean(values[peers]))
        else:
            means[agent_id] = float(values[agent_id])
    return means


def update_channels(
    price_expectations: np.ndarray,
    conservation_norms: np.ndarray,
    market_price_value: float,
    resource_level: float,
    payoffs: np.ndarray,
    payoff_ema: np.ndarray,
    config: Toy10Config,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    revise = rng.random(config.agents.count) < config.policy.revision_rate
    advantages = payoffs - payoff_ema
    resource_fraction = resource_level / config.environment.resource_carrying_capacity
    scarcity_target = np.clip(1.0 - resource_fraction, 0.0, 1.0)
    next_price = price_expectations.copy()
    next_conservation = conservation_norms.copy()
    for agent_id in np.flatnonzero(revise):
        price_delta = config.policy.learning_rate * (
            market_price_value - next_price[int(agent_id)]
        )
        payoff_delta = 0.05 * advantages[int(agent_id)]
        next_price[int(agent_id)] += price_delta + payoff_delta
        conservation_delta = config.policy.learning_rate * (
            scarcity_target - next_conservation[int(agent_id)]
        )
        next_conservation[int(agent_id)] += conservation_delta - payoff_delta
    return (
        np.clip(next_price, 0.0, 1.0),
        np.clip(next_conservation, 0.0, 1.0),
    )


def rewire_graph(
    graph: nx.Graph,
    payoffs: np.ndarray,
    price_expectations: np.ndarray,
    conservation_norms: np.ndarray,
    config: Toy10Config,
    rng: np.random.Generator,
) -> int:
    if config.network.dynamic_rewire_rate <= 0.0:
        return 0
    churn = 0
    agent_count = config.agents.count
    composite = np.clip(0.5 * (price_expectations + conservation_norms), 0.0, 1.0)
    for agent_id in range(agent_count):
        if rng.random() >= config.network.dynamic_rewire_rate:
            continue
        neighbors = list(graph.neighbors(agent_id))
        if not neighbors:
            continue
        drop = min(
            neighbors,
            key=lambda peer_id: (
                payoffs[int(peer_id)] - abs(composite[agent_id] - composite[int(peer_id)])
            ),
        )
        candidates = [
            node
            for node in range(agent_count)
            if node != agent_id and not graph.has_edge(agent_id, node)
        ]
        if not candidates:
            continue
        pool_size = min(config.network.candidate_pool_size, len(candidates))
        pool = rng.choice(candidates, size=pool_size, replace=False)
        add = max(
            (int(node) for node in pool),
            key=lambda node: payoffs[node] - abs(composite[agent_id] - composite[node]),
        )
        graph.remove_edge(agent_id, int(drop))
        graph.add_edge(agent_id, add)
        churn += 1
    return churn


def aggregate_row(
    config: Toy10Config,
    epoch: int,
    step: Toy10StepResult,
) -> dict[str, object]:
    peer_graph = graph_from_peer_ids(config.agents.count, step.peer_ids)
    resource_fraction = (
        step.resource_level / config.environment.resource_carrying_capacity
    )
    return {
        "run_id": config.run.name,
        "seed": config.run.seed,
        "epoch": epoch,
        "coordination_mixer": config.coordination.mixer,
        "coordination_peer_rule": config.coordination.peer_rule,
        "domain_resource_level": step.resource_level,
        "domain_resource_fraction": resource_fraction,
        "domain_market_price": step.market_price,
        "domain_market_imbalance": step.market_imbalance,
        "domain_mean_harvest_intensity": float(np.mean(step.harvest_intensities)),
        "domain_harvest_variance": float(np.var(step.harvest_intensities)),
        "domain_mean_price_expectation": float(np.mean(step.price_expectations)),
        "domain_mean_conservation_norm": float(np.mean(step.conservation_norms)),
        "domain_mean_payoff": float(np.mean(step.payoffs)),
        "domain_payoff_variance": float(np.var(step.payoffs)),
        "domain_cumulative_rewired_edge_count": step.cumulative_rewired_edge_count,
        "fragmentation_components": nx.number_connected_components(peer_graph),
        "mean_peer_count": float(np.mean([len(peers) for peers in step.peer_ids])),
        "mean_social_loss": float(np.mean(step.social_losses)),
        "mean_social_update_norm": float(np.mean(step.social_update_norms)),
    }


def domain_run_settings(config: Toy10Config, config_path: Path) -> DomainRunSettings:
    return DomainRunSettings(
        toy="toy10",
        config=config,
        config_path=config_path,
        output_dir=config.run.output_dir,
        run_name=config.run.name,
        seed=config.run.seed,
        micro_state_fields=TOY10_MICRO_STATE_FIELDS,
        aggregate_fields=TOY10_AGGREGATE_FIELDS,
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "toy": "toy10",
            "policy_rule": config.policy.rule,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_agent_count": config.agents.count,
            "domain_network_type": config.network.type,
            "domain_network_k": config.network.k,
            "domain_network_rewire_probability": (
                config.network.rewire_probability
            ),
            "domain_dynamic_rewire_rate": config.network.dynamic_rewire_rate,
            "domain_conservation_harvest_weight": (
                config.policy.conservation_harvest_weight
            ),
            "domain_social_harvest_gain": config.policy.social_harvest_gain,
            "domain_social_disagreement_penalty": (
                config.policy.social_disagreement_penalty
            ),
        },
        logging_interval=config.logging.interval,
        log_micro_state=config.logging.micro_state,
        log_aggregate_metrics=config.logging.aggregate_metrics,
        no_step_error="Toy 10 simulation produced no steps",
    )


def make_run_dir(config: Toy10Config) -> Path:
    settings = domain_run_settings(config, Path("<unknown-config>"))
    return make_timestamped_run_dir(
        output_dir=settings.output_dir,
        run_name=settings.run_name,
        seed=settings.seed,
    )


def write_run_metadata(config_path: Path, config: Toy10Config, run_dir: Path) -> None:
    settings = domain_run_settings(config, config_path)
    write_run_metadata_artifacts(
        config_path=settings.config_path,
        config=settings.config,
        run_dir=run_dir,
        toy=settings.toy,
        metadata=settings.metadata,
        strict_capability=settings.strict_capability,
    )


def micro_rows(
    config: Toy10Config,
    epoch: int,
    step: Toy10StepResult,
) -> list[dict[str, object]]:
    peer_graph = graph_from_peer_ids(config.agents.count, step.peer_ids)
    components = component_map(peer_graph)
    resource_fraction = (
        step.resource_level / config.environment.resource_carrying_capacity
    )
    return [
        {
            "run_id": config.run.name,
            "seed": config.run.seed,
            "epoch": epoch,
            "agent_id": agent_id,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_harvest_intensity": float(
                step.harvest_intensities[agent_id]
            ),
            "domain_price_expectation": float(step.price_expectations[agent_id]),
            "domain_conservation_norm": float(step.conservation_norms[agent_id]),
            "domain_payoff": float(step.payoffs[agent_id]),
            "domain_payoff_ema": float(step.payoff_ema[agent_id]),
            "domain_market_price": step.market_price,
            "domain_resource_level": step.resource_level,
            "domain_resource_fraction": resource_fraction,
            "domain_local_price_expectation": float(
                step.local_price_expectations[agent_id]
            ),
            "domain_local_conservation_norm": float(
                step.local_conservation_norms[agent_id]
            ),
            "peer_ids": step.peer_ids[agent_id],
            "peer_count": len(step.peer_ids[agent_id]),
            "component_id": components.get(agent_id, -1),
            "social_loss": step.social_losses[agent_id],
            "social_update_norm": step.social_update_norms[agent_id],
        }
        for agent_id in range(config.agents.count)
    ]


def write_micro_state(
    writer: CsvLogWriter,
    config: Toy10Config,
    epoch: int,
    step: Toy10StepResult,
) -> None:
    for row in micro_rows(config, epoch, step):
        writer.write(row)


@dataclass
class Toy10Adapter:
    config: Toy10Config
    rng: np.random.Generator

    def initialize(self) -> Toy10RunState:
        same_init = self.config.agents.init_mode == "same_init"
        return Toy10RunState(
            rng=self.rng,
            graph=build_graph(
                self.config.network,
                self.config.agents.count,
                self.config.run.seed,
            ),
            price_expectations=initialize_channel(
                mean=self.config.environment.initial_price_expectation_mean,
                std=self.config.environment.initial_price_expectation_std,
                count=self.config.agents.count,
                same_init=same_init,
                rng=self.rng,
            ),
            conservation_norms=initialize_channel(
                mean=self.config.environment.initial_conservation_norm_mean,
                std=self.config.environment.initial_conservation_norm_std,
                count=self.config.agents.count,
                same_init=same_init,
                rng=self.rng,
            ),
            resource_level=self.config.environment.resource_initial,
            payoff_ema=np.zeros(self.config.agents.count, dtype=np.float64),
        )

    def step_epochs(self, state: Toy10RunState) -> range:
        return range(1, self.config.simulation.epochs + 1)

    def step(self, epoch: int, state: Toy10RunState) -> Toy10StepResult:
        neighbors = graph_neighbors(state.graph, self.config.agents.count)
        peer_ids = select_peer_ids(
            state.price_expectations,
            state.conservation_norms,
            neighbors,
            self.config,
        )
        mixed_price, price_losses, price_update_norms = mix_channel(
            state.price_expectations,
            peer_ids,
            self.config,
            channel="price_expectation",
        )
        mixed_conservation, conservation_losses, conservation_update_norms = (
            mix_channel(
                state.conservation_norms,
                peer_ids,
                self.config,
                channel="conservation_norm",
            )
        )
        social_losses = [
            0.5 * (price_losses[i] + conservation_losses[i])
            for i in range(self.config.agents.count)
        ]
        social_update_norms = [
            0.5 * (price_update_norms[i] + conservation_update_norms[i])
            for i in range(self.config.agents.count)
        ]
        harvest_intensities = harvest_from_channels(
            mixed_price,
            mixed_conservation,
            state.rng,
            self.config,
            base_price_expectations=state.price_expectations,
            base_conservation_norms=state.conservation_norms,
        )
        market_price_value, imbalance = market_price(
            harvest_intensities,
            mixed_price,
            self.config,
        )
        payoffs = compute_payoffs(
            harvest_intensities,
            state.resource_level,
            market_price_value,
            self.config,
        )
        state.resource_level = update_resource(
            state.resource_level,
            harvest_intensities,
            self.config,
        )
        state.price_expectations, state.conservation_norms = update_channels(
            mixed_price,
            mixed_conservation,
            market_price_value,
            state.resource_level,
            payoffs,
            state.payoff_ema,
            self.config,
            state.rng,
        )
        state.payoff_ema = (
            self.config.policy.reward_ema_decay * state.payoff_ema
            + (1.0 - self.config.policy.reward_ema_decay) * payoffs
        )
        state.cumulative_rewired_edge_count += rewire_graph(
            state.graph,
            payoffs,
            state.price_expectations,
            state.conservation_norms,
            self.config,
            state.rng,
        )
        next_neighbors = graph_neighbors(state.graph, self.config.agents.count)
        return Toy10StepResult(
            harvest_intensities=harvest_intensities.copy(),
            price_expectations=state.price_expectations.copy(),
            conservation_norms=state.conservation_norms.copy(),
            local_price_expectations=local_channel_means(
                state.price_expectations,
                next_neighbors,
            ),
            local_conservation_norms=local_channel_means(
                state.conservation_norms,
                next_neighbors,
            ),
            payoffs=payoffs.copy(),
            payoff_ema=state.payoff_ema.copy(),
            resource_level=state.resource_level,
            market_price=market_price_value,
            market_imbalance=imbalance,
            cumulative_rewired_edge_count=state.cumulative_rewired_edge_count,
            peer_ids=peer_ids,
            social_losses=social_losses,
            social_update_norms=social_update_norms,
        )

    def fallback_step(self, state: Toy10RunState) -> None:
        return None

    def aggregate_row(
        self,
        epoch: int,
        state: Toy10RunState,
        step: Toy10StepResult,
    ) -> dict[str, object]:
        return aggregate_row(self.config, epoch, step)

    def micro_rows(
        self,
        epoch: int,
        state: Toy10RunState,
        step: Toy10StepResult,
    ) -> list[dict[str, object]]:
        return micro_rows(self.config, epoch, step)

    def final_epoch(self, state: Toy10RunState, step: Toy10StepResult) -> int:
        return self.config.simulation.epochs

    def domain_metrics(
        self,
        final_row: dict[str, object],
        state: Toy10RunState,
        step: Toy10StepResult,
    ) -> dict[str, object]:
        return {
            "domain_final_resource_level": final_row["domain_resource_level"],
            "domain_final_resource_fraction": final_row["domain_resource_fraction"],
            "domain_final_market_price": final_row["domain_market_price"],
            "domain_final_market_imbalance": final_row["domain_market_imbalance"],
            "domain_final_mean_harvest_intensity": final_row[
                "domain_mean_harvest_intensity"
            ],
            "domain_final_mean_price_expectation": final_row[
                "domain_mean_price_expectation"
            ],
            "domain_final_mean_conservation_norm": final_row[
                "domain_mean_conservation_norm"
            ],
            "domain_final_mean_payoff": final_row["domain_mean_payoff"],
            "domain_cumulative_rewired_edge_count": final_row[
                "domain_cumulative_rewired_edge_count"
            ],
        }


def run_toy10(config: Toy10Config, config_path: Path) -> DomainToyResult:
    """Run Toy 10 from a validated config."""

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    resolve_torch_device(config.simulation.device)
    return DomainToyRunner(
        Toy10Adapter(config=config, rng=rng),
        domain_run_settings(config, config_path),
    ).run()
