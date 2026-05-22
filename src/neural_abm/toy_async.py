"""Toy 8: asynchronous event-driven adoption, failure, and recovery ABM."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from neural_abm.accelerator import resolve_torch_device
from neural_abm.config import Toy8Config
from neural_abm.domain_runner import (
    DomainRunSettings,
    DomainToyRunner,
    make_timestamped_run_dir,
)
from neural_abm.graphs import build_graph, component_map, graph_from_peer_ids
from neural_abm.logging import CsvLogWriter
from neural_abm.mixers import apply_scalar_output_average
from neural_abm.results import (
    DomainToyResult,
    write_run_metadata_artifacts,
)
from neural_abm.social import (
    empty_peers,
    select_scalar_output_peers,
)


STATE_INACTIVE = 0
STATE_ACTIVE = 1
STATE_FAILED = 2

EVENT_ACTIVATION = "activation"
EVENT_FAILURE = "failure"
EVENT_RECOVERY = "recovery"
EVENT_NONE = "none"


TOY8_MICRO_STATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "domain_time",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_state",
    "domain_activation_rate",
    "domain_failure_rate",
    "domain_recovery_rate",
    "domain_neighbor_active_fraction",
    "domain_activation_propensity",
    "event_type",
    "event_agent_id",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]


TOY8_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "domain_time",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_inactive_fraction",
    "domain_active_fraction",
    "domain_failed_fraction",
    "domain_event_type",
    "domain_event_agent_id",
    "domain_cumulative_activation_events",
    "domain_cumulative_failure_events",
    "domain_cumulative_recovery_events",
    "fragmentation_components",
    "mean_peer_count",
    "mean_activation_rate",
    "mean_failure_rate",
    "mean_recovery_rate",
    "mean_social_loss",
    "mean_social_update_norm",
]


@dataclass(order=True, frozen=True)
class ScheduledEvent:
    time: float
    event_id: int
    agent_id: int
    event_type: str
    version: int


@dataclass(frozen=True)
class Toy8RateSnapshot:
    activation_rates: np.ndarray
    failure_rates: np.ndarray
    recovery_rates: np.ndarray
    neighbor_active_fraction: np.ndarray
    activation_propensities: np.ndarray
    peer_ids: list[list[int]]
    social_losses: list[float]
    social_update_norms: list[float]


@dataclass(frozen=True)
class Toy8StepResult:
    states: np.ndarray
    event_time: float
    event_type: str
    event_agent_id: int
    activation_events: int
    failure_events: int
    recovery_events: int
    snapshot: Toy8RateSnapshot


@dataclass
class Toy8RunState:
    rng: np.random.Generator
    neighbors: list[list[int]]
    states: np.ndarray
    versions: np.ndarray
    event_queue: list[ScheduledEvent]
    current_time: float
    next_event_id: int
    activation_events: int = 0
    failure_events: int = 0
    recovery_events: int = 0


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [
        sorted(int(node) for node in graph.neighbors(agent_id))
        for agent_id in range(agent_count)
    ]


def initialize_states(config: Toy8Config, rng: np.random.Generator) -> np.ndarray:
    states = np.full(config.agents.count, STATE_INACTIVE, dtype=np.int64)
    active_count = int(round(config.agents.count * config.environment.initial_active_fraction))
    failed_count = int(round(config.agents.count * config.environment.initial_failed_fraction))
    selected_count = min(active_count + failed_count, config.agents.count)
    if selected_count == 0:
        return states
    selected = rng.choice(config.agents.count, size=selected_count, replace=False)
    if active_count:
        states[selected[:active_count]] = STATE_ACTIVE
    if failed_count:
        states[selected[active_count : active_count + failed_count]] = STATE_FAILED
    return states


def neighbor_active_fraction(
    states: np.ndarray,
    neighbors: list[list[int]],
) -> np.ndarray:
    fractions = np.zeros(len(states), dtype=np.float64)
    for agent_id, peers in enumerate(neighbors):
        if peers:
            fractions[agent_id] = float(np.mean(states[peers] == STATE_ACTIVE))
    return fractions


def select_peer_ids(
    activation_propensities: np.ndarray,
    neighbors: list[list[int]],
    config: Toy8Config,
) -> list[list[int]]:
    if config.coordination.mixer == "none":
        return empty_peers(len(activation_propensities))
    return select_scalar_output_peers(
        neighbors=neighbors,
        values=activation_propensities,
        peer_rule=config.coordination.peer_rule,
        threshold=config.coordination.threshold,
    ).peer_ids


def apply_output_average(
    activation_propensities: np.ndarray,
    peer_ids: list[list[int]],
    config: Toy8Config,
) -> tuple[np.ndarray, list[float], list[float]]:
    if config.coordination.mixer == "none":
        return (
            activation_propensities.copy(),
            [0.0 for _ in range(len(activation_propensities))],
            [0.0 for _ in range(len(activation_propensities))],
        )
    result = apply_scalar_output_average(
        values=activation_propensities,
        peer_ids=peer_ids,
        alpha=config.coordination.alpha,
        channel="activation_propensity",
        commit_mode="event_hazard_commit",
    )
    return (
        np.asarray(result.mix.mixed_values, dtype=np.float64),
        result.commit.losses,
        result.mix.update_norms,
    )


def compute_rate_snapshot(
    states: np.ndarray,
    neighbors: list[list[int]],
    config: Toy8Config,
) -> Toy8RateSnapshot:
    env = config.environment
    active_fraction = neighbor_active_fraction(states, neighbors)
    max_activation_rate = env.base_activation_rate + env.peer_activation_rate
    local_activation_rates = env.base_activation_rate + (
        env.peer_activation_rate * active_fraction
    )
    if max_activation_rate > 0.0:
        local_propensities = np.clip(
            local_activation_rates / max_activation_rate,
            0.0,
            1.0,
        )
    else:
        local_propensities = np.zeros(len(states), dtype=np.float64)

    peer_ids = select_peer_ids(local_propensities, neighbors, config)
    mixed_propensities, social_losses, social_update_norms = apply_output_average(
        local_propensities,
        peer_ids,
        config,
    )
    if config.coordination.mixer == "none":
        activation_rates = local_activation_rates
    else:
        activation_rates = max_activation_rate * mixed_propensities
    activation_rates = np.where(states == STATE_INACTIVE, activation_rates, 0.0)
    failure_rates = np.where(
        states == STATE_ACTIVE,
        env.failure_rate + env.overload_failure_rate * active_fraction,
        0.0,
    )
    recovery_rates = np.where(states == STATE_FAILED, env.recovery_rate, 0.0)
    return Toy8RateSnapshot(
        activation_rates=activation_rates.astype(np.float64),
        failure_rates=failure_rates.astype(np.float64),
        recovery_rates=recovery_rates.astype(np.float64),
        neighbor_active_fraction=active_fraction,
        activation_propensities=mixed_propensities,
        peer_ids=peer_ids,
        social_losses=social_losses,
        social_update_norms=social_update_norms,
    )


def schedule_all_events(
    *,
    queue: list[ScheduledEvent],
    versions: np.ndarray,
    states: np.ndarray,
    neighbors: list[list[int]],
    config: Toy8Config,
    rng: np.random.Generator,
    current_time: float,
    next_event_id: int,
) -> int:
    versions += 1
    snapshot = compute_rate_snapshot(states, neighbors, config)
    for agent_id, state in enumerate(states):
        if state == STATE_INACTIVE:
            rate = snapshot.activation_rates[agent_id]
            event_type = EVENT_ACTIVATION
        elif state == STATE_ACTIVE:
            rate = snapshot.failure_rates[agent_id]
            event_type = EVENT_FAILURE
        else:
            rate = snapshot.recovery_rates[agent_id]
            event_type = EVENT_RECOVERY
        if rate <= 0.0:
            continue
        event_time = current_time + float(rng.exponential(scale=1.0 / rate))
        heapq.heappush(
            queue,
            ScheduledEvent(
                time=event_time,
                event_id=next_event_id,
                agent_id=int(agent_id),
                event_type=event_type,
                version=int(versions[agent_id]),
            ),
        )
        next_event_id += 1
    return next_event_id


def valid_event(event: ScheduledEvent, states: np.ndarray, versions: np.ndarray) -> bool:
    if event.version != int(versions[event.agent_id]):
        return False
    state = int(states[event.agent_id])
    return (
        (event.event_type == EVENT_ACTIVATION and state == STATE_INACTIVE)
        or (event.event_type == EVENT_FAILURE and state == STATE_ACTIVE)
        or (event.event_type == EVENT_RECOVERY and state == STATE_FAILED)
    )


def apply_event(states: np.ndarray, event: ScheduledEvent) -> None:
    if event.event_type == EVENT_ACTIVATION:
        states[event.agent_id] = STATE_ACTIVE
    elif event.event_type == EVENT_FAILURE:
        states[event.agent_id] = STATE_FAILED
    elif event.event_type == EVENT_RECOVERY:
        states[event.agent_id] = STATE_INACTIVE
    else:
        raise ValueError(f"Unsupported Toy 8 event type: {event.event_type}")


def aggregate_row(
    config: Toy8Config,
    epoch: int,
    step: Toy8StepResult,
) -> dict[str, object]:
    peer_graph = graph_from_peer_ids(config.agents.count, step.snapshot.peer_ids)
    inactive_fraction = float(np.mean(step.states == STATE_INACTIVE))
    active_fraction = float(np.mean(step.states == STATE_ACTIVE))
    failed_fraction = float(np.mean(step.states == STATE_FAILED))
    return {
        "run_id": config.run.name,
        "seed": config.run.seed,
        "epoch": epoch,
        "domain_time": step.event_time,
        "coordination_mixer": config.coordination.mixer,
        "coordination_peer_rule": config.coordination.peer_rule,
        "domain_inactive_fraction": inactive_fraction,
        "domain_active_fraction": active_fraction,
        "domain_failed_fraction": failed_fraction,
        "domain_event_type": step.event_type,
        "domain_event_agent_id": step.event_agent_id,
        "domain_cumulative_activation_events": step.activation_events,
        "domain_cumulative_failure_events": step.failure_events,
        "domain_cumulative_recovery_events": step.recovery_events,
        "fragmentation_components": nx.number_connected_components(peer_graph),
        "mean_peer_count": float(
            np.mean([len(peers) for peers in step.snapshot.peer_ids])
        ),
        "mean_activation_rate": float(np.mean(step.snapshot.activation_rates)),
        "mean_failure_rate": float(np.mean(step.snapshot.failure_rates)),
        "mean_recovery_rate": float(np.mean(step.snapshot.recovery_rates)),
        "mean_social_loss": float(np.mean(step.snapshot.social_losses)),
        "mean_social_update_norm": float(np.mean(step.snapshot.social_update_norms)),
    }


def domain_run_settings(config: Toy8Config, config_path: Path) -> DomainRunSettings:
    return DomainRunSettings(
        toy="toy8",
        config=config,
        config_path=config_path,
        output_dir=config.run.output_dir,
        run_name=config.run.name,
        seed=config.run.seed,
        micro_state_fields=TOY8_MICRO_STATE_FIELDS,
        aggregate_fields=TOY8_AGGREGATE_FIELDS,
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "toy": "toy8",
            "policy_rule": config.policy.rule,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_agent_count": config.agents.count,
            "domain_graph_type": config.graph.type,
            "domain_graph_k": config.graph.k,
            "domain_graph_rewire_probability": config.graph.rewire_probability,
            "domain_max_time": config.environment.max_time,
        },
        logging_interval=config.logging.interval,
        log_micro_state=config.logging.micro_state,
        log_aggregate_metrics=config.logging.aggregate_metrics,
        no_step_error="Toy 8 produced no simulation steps",
    )


def make_run_dir(config: Toy8Config) -> Path:
    settings = domain_run_settings(config, Path("<unknown-config>"))
    return make_timestamped_run_dir(
        output_dir=settings.output_dir,
        run_name=settings.run_name,
        seed=settings.seed,
    )


def write_run_metadata(config_path: Path, config: Toy8Config, run_dir: Path) -> None:
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
    config: Toy8Config,
    epoch: int,
    step: Toy8StepResult,
) -> list[dict[str, object]]:
    peer_graph = graph_from_peer_ids(config.agents.count, step.snapshot.peer_ids)
    components = component_map(peer_graph)
    return [
        {
            "run_id": config.run.name,
            "seed": config.run.seed,
            "epoch": epoch,
            "domain_time": step.event_time,
            "agent_id": agent_id,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_state": int(step.states[agent_id]),
            "domain_activation_rate": float(
                step.snapshot.activation_rates[agent_id]
            ),
            "domain_failure_rate": float(step.snapshot.failure_rates[agent_id]),
            "domain_recovery_rate": float(step.snapshot.recovery_rates[agent_id]),
            "domain_neighbor_active_fraction": float(
                step.snapshot.neighbor_active_fraction[agent_id]
            ),
            "domain_activation_propensity": float(
                step.snapshot.activation_propensities[agent_id]
            ),
            "event_type": step.event_type,
            "event_agent_id": step.event_agent_id,
            "peer_ids": step.snapshot.peer_ids[agent_id],
            "peer_count": len(step.snapshot.peer_ids[agent_id]),
            "component_id": components.get(agent_id, -1),
            "social_loss": step.snapshot.social_losses[agent_id],
            "social_update_norm": step.snapshot.social_update_norms[agent_id],
        }
        for agent_id in range(config.agents.count)
    ]


def write_micro_state(
    writer: CsvLogWriter,
    config: Toy8Config,
    epoch: int,
    step: Toy8StepResult,
) -> None:
    for row in micro_rows(config, epoch, step):
        writer.write(row)


def state_fraction(states: np.ndarray, state: int) -> float:
    return float(np.mean(states == state))


@dataclass
class Toy8Adapter:
    config: Toy8Config
    rng: np.random.Generator

    def initialize(self) -> Toy8RunState:
        graph = build_graph(self.config.graph, self.config.agents.count, self.config.run.seed)
        neighbors = graph_neighbors(graph, self.config.agents.count)
        states = initialize_states(self.config, self.rng)
        versions = np.zeros(self.config.agents.count, dtype=np.int64)
        event_queue: list[ScheduledEvent] = []
        next_event_id = schedule_all_events(
            queue=event_queue,
            versions=versions,
            states=states,
            neighbors=neighbors,
            config=self.config,
            rng=self.rng,
            current_time=0.0,
            next_event_id=0,
        )
        return Toy8RunState(
            rng=self.rng,
            neighbors=neighbors,
            states=states,
            versions=versions,
            event_queue=event_queue,
            current_time=0.0,
            next_event_id=next_event_id,
        )

    def step_epochs(self, state: Toy8RunState) -> range:
        return range(1, self.config.simulation.epochs + 1)

    def step(self, epoch: int, state: Toy8RunState) -> Toy8StepResult | None:
        event: ScheduledEvent | None = None
        while state.event_queue:
            candidate = heapq.heappop(state.event_queue)
            if valid_event(candidate, state.states, state.versions):
                event = candidate
                break
        if event is None:
            return None
        if event.time > self.config.environment.max_time:
            state.current_time = self.config.environment.max_time
            return None

        state.current_time = event.time
        apply_event(state.states, event)
        if event.event_type == EVENT_ACTIVATION:
            state.activation_events += 1
        elif event.event_type == EVENT_FAILURE:
            state.failure_events += 1
        elif event.event_type == EVENT_RECOVERY:
            state.recovery_events += 1

        state.next_event_id = schedule_all_events(
            queue=state.event_queue,
            versions=state.versions,
            states=state.states,
            neighbors=state.neighbors,
            config=self.config,
            rng=state.rng,
            current_time=state.current_time,
            next_event_id=state.next_event_id,
        )
        snapshot = compute_rate_snapshot(state.states, state.neighbors, self.config)
        return Toy8StepResult(
            states=state.states.copy(),
            event_time=state.current_time,
            event_type=event.event_type,
            event_agent_id=event.agent_id,
            activation_events=state.activation_events,
            failure_events=state.failure_events,
            recovery_events=state.recovery_events,
            snapshot=snapshot,
        )

    def fallback_step(self, state: Toy8RunState) -> Toy8StepResult:
        snapshot = compute_rate_snapshot(state.states, state.neighbors, self.config)
        return Toy8StepResult(
            states=state.states.copy(),
            event_time=state.current_time,
            event_type=EVENT_NONE,
            event_agent_id=-1,
            activation_events=state.activation_events,
            failure_events=state.failure_events,
            recovery_events=state.recovery_events,
            snapshot=snapshot,
        )

    def aggregate_row(
        self,
        epoch: int,
        state: Toy8RunState,
        step: Toy8StepResult,
    ) -> dict[str, object]:
        return aggregate_row(self.config, epoch, step)

    def micro_rows(
        self,
        epoch: int,
        state: Toy8RunState,
        step: Toy8StepResult,
    ) -> list[dict[str, object]]:
        return micro_rows(self.config, epoch, step)

    def final_epoch(self, state: Toy8RunState, step: Toy8StepResult) -> int:
        return step.activation_events + step.failure_events + step.recovery_events

    def domain_metrics(
        self,
        final_row: dict[str, object],
        state: Toy8RunState,
        step: Toy8StepResult,
    ) -> dict[str, object]:
        total_events = self.final_epoch(state, step)
        return {
            "domain_final_time": step.event_time,
            "domain_final_inactive_fraction": state_fraction(
                step.states,
                STATE_INACTIVE,
            ),
            "domain_final_active_fraction": state_fraction(
                step.states,
                STATE_ACTIVE,
            ),
            "domain_final_failed_fraction": state_fraction(
                step.states,
                STATE_FAILED,
            ),
            "domain_total_events": total_events,
            "domain_activation_events": step.activation_events,
            "domain_failure_events": step.failure_events,
            "domain_recovery_events": step.recovery_events,
            "domain_absorbed": total_events < self.config.simulation.epochs,
        }


def run_toy8(config: Toy8Config, config_path: Path) -> DomainToyResult:
    """Run Toy 8 from a validated config."""

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    resolve_torch_device(config.simulation.device)
    return DomainToyRunner(
        Toy8Adapter(config=config, rng=rng),
        domain_run_settings(config, config_path),
    ).run()
