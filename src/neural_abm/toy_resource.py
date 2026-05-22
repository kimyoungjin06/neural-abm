"""Toy 7: continuous extraction-intensity resource ABM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from neural_abm.accelerator import resolve_torch_device
from neural_abm.config import Toy7Config
from neural_abm.domain_runner import (
    DomainRunSettings,
    DomainToyRunner,
    make_domain_run_dir,
    write_domain_run_metadata,
)
from neural_abm.domain_social_diagnostics import (
    aggregate_social_diagnostic_fields,
    micro_social_diagnostic_fields,
)
from neural_abm.graphs import build_graph, component_map, graph_from_peer_ids
from neural_abm.mixers import apply_bounded_scalar_output_average
from neural_abm.results import DomainToyResult
from neural_abm.social import (
    empty_peers,
    select_bounded_scalar_output_peers,
)


TOY7_MICRO_STATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_intensity",
    "domain_propensity",
    "domain_payoff",
    "domain_payoff_ema",
    "domain_resource_level",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]


TOY7_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_resource_level",
    "domain_resource_fraction",
    "domain_mean_intensity",
    "domain_intensity_variance",
    "domain_mean_payoff",
    "fragmentation_components",
    "mean_peer_count",
    "mean_social_loss",
    "mean_social_update_norm",
]


@dataclass(frozen=True)
class Toy7StepResult:
    intensities: np.ndarray
    propensities: np.ndarray
    payoffs: np.ndarray
    payoff_ema: np.ndarray
    resource_level: float
    peer_ids: list[list[int]]
    social_losses: list[float]
    social_update_norms: list[float]


@dataclass
class Toy7RunState:
    rng: np.random.Generator
    neighbors: list[list[int]]
    propensities: np.ndarray
    payoff_ema: np.ndarray
    resource_level: float


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [
        sorted(int(node) for node in graph.neighbors(i)) for i in range(agent_count)
    ]


def initialize_propensities(config: Toy7Config, rng: np.random.Generator) -> np.ndarray:
    if config.agents.init_mode == "same_init":
        return np.full(config.agents.count, config.environment.initial_intensity_mean)
    return np.clip(
        rng.normal(
            loc=config.environment.initial_intensity_mean,
            scale=config.environment.initial_intensity_std,
            size=config.agents.count,
        ),
        0.0,
        1.0,
    ).astype(np.float64)


def select_peer_ids(
    values: np.ndarray,
    neighbors: list[list[int]],
    config: Toy7Config,
) -> list[list[int]]:
    if config.coordination.mixer == "none":
        return empty_peers(len(values))
    return select_bounded_scalar_output_peers(
        neighbors=neighbors,
        values=values,
        peer_rule=config.coordination.peer_rule,
        threshold=config.coordination.threshold,
        lower_bound=0.0,
        upper_bound=1.0,
    ).peer_ids


def apply_output_average(
    values: np.ndarray,
    peer_ids: list[list[int]],
    config: Toy7Config,
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
        channel="extraction_intensity",
        commit_mode="continuous_intensity_commit",
    )
    return (
        np.asarray(result.mix.mixed_values, dtype=np.float64),
        result.commit.losses,
        result.mix.update_norms,
    )


def compute_payoffs(
    intensities: np.ndarray,
    resource_level: float,
    config: Toy7Config,
) -> np.ndarray:
    resource_fraction = resource_level / config.environment.resource_carrying_capacity
    gross = intensities * resource_fraction
    costs = config.environment.extraction_cost * np.square(intensities)
    return gross - costs


def update_resource(
    resource_level: float,
    intensities: np.ndarray,
    config: Toy7Config,
) -> float:
    env = config.environment
    recovered = resource_level + env.resource_recovery_rate * (
        env.resource_carrying_capacity - resource_level
    )
    extracted = env.extraction_scale * float(np.mean(intensities))
    return float(np.clip(recovered - extracted, 0.0, env.resource_carrying_capacity))


def adaptive_target(resource_level: float, config: Toy7Config) -> float:
    resource_fraction = resource_level / config.environment.resource_carrying_capacity
    if config.environment.extraction_cost <= 0.0:
        return 1.0
    return float(np.clip(resource_fraction / (2.0 * config.environment.extraction_cost), 0.0, 1.0))


def update_propensities(
    propensities: np.ndarray,
    intensities: np.ndarray,
    payoffs: np.ndarray,
    payoff_ema: np.ndarray,
    resource_level: float,
    config: Toy7Config,
    rng: np.random.Generator,
) -> np.ndarray:
    updated = propensities.copy()
    revision_mask = rng.random(config.agents.count) < config.policy.revision_rate
    target = adaptive_target(resource_level, config)
    advantages = payoffs - payoff_ema
    for agent_id in np.flatnonzero(revision_mask):
        current = updated[int(agent_id)]
        target_blend = config.policy.learning_rate * (target - current)
        reinforcement = 0.05 * advantages[int(agent_id)] * (
            intensities[int(agent_id)] - current
        )
        updated[int(agent_id)] = current + target_blend + reinforcement
    return np.clip(updated, 0.0, 1.0)


def aggregate_row(
    config: Toy7Config,
    epoch: int,
    step: Toy7StepResult,
) -> dict[str, object]:
    peer_graph = graph_from_peer_ids(config.agents.count, step.peer_ids)
    return {
        "run_id": config.run.name,
        "seed": config.run.seed,
        "epoch": epoch,
        "coordination_mixer": config.coordination.mixer,
        "coordination_peer_rule": config.coordination.peer_rule,
        "domain_resource_level": step.resource_level,
        "domain_resource_fraction": (
            step.resource_level / config.environment.resource_carrying_capacity
        ),
        "domain_mean_intensity": float(np.mean(step.intensities)),
        "domain_intensity_variance": float(np.var(step.intensities)),
        "domain_mean_payoff": float(np.mean(step.payoffs)),
        "fragmentation_components": nx.number_connected_components(peer_graph),
        **aggregate_social_diagnostic_fields(
            peer_ids=step.peer_ids,
            social_losses=step.social_losses,
            social_update_norms=step.social_update_norms,
        ),
    }


def domain_run_settings(config: Toy7Config, config_path: Path) -> DomainRunSettings:
    return DomainRunSettings(
        toy="toy7",
        config=config,
        config_path=config_path,
        output_dir=config.run.output_dir,
        run_name=config.run.name,
        seed=config.run.seed,
        micro_state_fields=TOY7_MICRO_STATE_FIELDS,
        aggregate_fields=TOY7_AGGREGATE_FIELDS,
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "toy": "toy7",
            "policy_rule": config.policy.rule,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_agent_count": config.agents.count,
            "domain_graph_type": config.graph.type,
            "domain_graph_k": config.graph.k,
            "domain_graph_rewire_probability": config.graph.rewire_probability,
        },
        logging_interval=config.logging.interval,
        log_micro_state=config.logging.micro_state,
        log_aggregate_metrics=config.logging.aggregate_metrics,
        no_step_error="Toy 7 produced no simulation steps",
    )


def make_run_dir(config: Toy7Config) -> Path:
    return make_domain_run_dir(domain_run_settings(config, Path("<unknown-config>")))


def write_run_metadata(config_path: Path, config: Toy7Config, run_dir: Path) -> None:
    write_domain_run_metadata(domain_run_settings(config, config_path), run_dir)


def micro_rows(
    config: Toy7Config,
    epoch: int,
    step: Toy7StepResult,
) -> list[dict[str, object]]:
    peer_graph = graph_from_peer_ids(config.agents.count, step.peer_ids)
    components = component_map(peer_graph)
    return [
        {
            "run_id": config.run.name,
            "seed": config.run.seed,
            "epoch": epoch,
            "agent_id": agent_id,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_intensity": float(step.intensities[agent_id]),
            "domain_propensity": float(step.propensities[agent_id]),
            "domain_payoff": float(step.payoffs[agent_id]),
            "domain_payoff_ema": float(step.payoff_ema[agent_id]),
            "domain_resource_level": step.resource_level,
            **micro_social_diagnostic_fields(
                agent_id=agent_id,
                peer_ids=step.peer_ids,
                social_losses=step.social_losses,
                social_update_norms=step.social_update_norms,
                component_id=components.get(agent_id, -1),
            ),
        }
        for agent_id in range(config.agents.count)
    ]


@dataclass
class Toy7Adapter:
    config: Toy7Config
    rng: np.random.Generator

    def initialize(self) -> Toy7RunState:
        graph = build_graph(self.config.graph, self.config.agents.count, self.config.run.seed)
        return Toy7RunState(
            rng=self.rng,
            neighbors=graph_neighbors(graph, self.config.agents.count),
            propensities=initialize_propensities(self.config, self.rng),
            payoff_ema=np.zeros(self.config.agents.count, dtype=np.float64),
            resource_level=float(self.config.environment.resource_initial),
        )

    def step_epochs(self, state: Toy7RunState) -> range:
        return range(1, self.config.simulation.epochs + 1)

    def step(self, epoch: int, state: Toy7RunState) -> Toy7StepResult:
        peer_ids = select_peer_ids(state.propensities, state.neighbors, self.config)
        social_values, social_losses, social_update_norms = apply_output_average(
            state.propensities,
            peer_ids,
            self.config,
        )
        intensities = np.clip(
            social_values
            + state.rng.normal(
                loc=0.0,
                scale=self.config.policy.exploration_std,
                size=self.config.agents.count,
            ),
            0.0,
            1.0,
        )
        payoffs = compute_payoffs(intensities, state.resource_level, self.config)
        state.payoff_ema = (
            self.config.policy.reward_ema_decay * state.payoff_ema
            + (1.0 - self.config.policy.reward_ema_decay) * payoffs
        )
        state.resource_level = update_resource(
            state.resource_level,
            intensities,
            self.config,
        )
        state.propensities = update_propensities(
            propensities=social_values,
            intensities=intensities,
            payoffs=payoffs,
            payoff_ema=state.payoff_ema,
            resource_level=state.resource_level,
            config=self.config,
            rng=state.rng,
        )
        return Toy7StepResult(
            intensities=intensities,
            propensities=state.propensities,
            payoffs=payoffs,
            payoff_ema=state.payoff_ema,
            resource_level=state.resource_level,
            peer_ids=peer_ids,
            social_losses=social_losses,
            social_update_norms=social_update_norms,
        )

    def fallback_step(self, state: Toy7RunState) -> None:
        return None

    def aggregate_row(
        self,
        epoch: int,
        state: Toy7RunState,
        step: Toy7StepResult,
    ) -> dict[str, object]:
        return aggregate_row(self.config, epoch, step)

    def micro_rows(
        self,
        epoch: int,
        state: Toy7RunState,
        step: Toy7StepResult,
    ) -> list[dict[str, object]]:
        return micro_rows(self.config, epoch, step)

    def final_epoch(self, state: Toy7RunState, step: Toy7StepResult) -> int:
        return self.config.simulation.epochs

    def domain_metrics(
        self,
        final_row: dict[str, object],
        state: Toy7RunState,
        step: Toy7StepResult,
    ) -> dict[str, object]:
        return {
            "domain_final_resource_level": final_row["domain_resource_level"],
            "domain_final_resource_fraction": final_row["domain_resource_fraction"],
            "domain_final_mean_intensity": final_row["domain_mean_intensity"],
            "domain_final_intensity_variance": final_row[
                "domain_intensity_variance"
            ],
            "domain_final_mean_payoff": final_row["domain_mean_payoff"],
        }


def run_toy7(config: Toy7Config, config_path: Path) -> DomainToyResult:
    """Run Toy 7 from a validated config."""

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    resolve_torch_device(config.simulation.device)
    return DomainToyRunner(
        Toy7Adapter(config=config, rng=rng),
        domain_run_settings(config, config_path),
    ).run()
