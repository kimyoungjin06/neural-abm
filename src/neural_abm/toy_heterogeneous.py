"""Toy 9: heterogeneous-agent binary adoption ABM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from neural_abm.accelerator import resolve_torch_device
from neural_abm.config import Toy9Config
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
from neural_abm.logging import CsvLogWriter
from neural_abm.mixers import apply_scalar_output_average
from neural_abm.results import DomainToyResult
from neural_abm.social import (
    empty_peers,
    select_scalar_output_peers,
)


TOY9_MICRO_STATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_agent_group",
    "domain_local_rule",
    "domain_coordination_enabled",
    "domain_action",
    "domain_action_probability",
    "domain_propensity",
    "domain_payoff",
    "domain_payoff_ema",
    "domain_neighbor_action_rate",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]


TOY9_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_action_rate",
    "domain_mean_action_probability",
    "domain_mean_payoff",
    "domain_payoff_variance",
    "domain_threshold_group_action_rate",
    "domain_payoff_learning_group_action_rate",
    "domain_coordination_enabled_action_rate",
    "domain_coordination_disabled_action_rate",
    "domain_group_action_rate_gap",
    "fragmentation_components",
    "mean_peer_count",
    "mean_social_loss",
    "mean_social_update_norm",
]


@dataclass(frozen=True)
class Toy9StepResult:
    actions: np.ndarray
    action_probabilities: np.ndarray
    propensities: np.ndarray
    payoffs: np.ndarray
    payoff_ema: np.ndarray
    neighbor_action_rates: np.ndarray
    peer_ids: list[list[int]]
    social_losses: list[float]
    social_update_norms: list[float]
    group_ids: np.ndarray
    group_names: list[str]
    local_rules: list[str]
    coordination_enabled: np.ndarray


@dataclass
class Toy9RunState:
    rng: np.random.Generator
    neighbors: list[list[int]]
    group_ids: np.ndarray
    group_names: list[str]
    local_rules: list[str]
    coordination_enabled: np.ndarray
    propensities: np.ndarray
    payoff_ema: np.ndarray


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)


def graph_neighbors(graph: nx.Graph, agent_count: int) -> list[list[int]]:
    return [
        sorted(int(node) for node in graph.neighbors(agent_id))
        for agent_id in range(agent_count)
    ]


def group_counts(config: Toy9Config) -> np.ndarray:
    fractions = np.asarray([group.fraction for group in config.agents.groups], dtype=float)
    normalized = fractions / fractions.sum()
    raw_counts = normalized * config.agents.count
    counts = np.floor(raw_counts).astype(np.int64)
    remainder = config.agents.count - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw_counts - counts))
        for group_id in order[:remainder]:
            counts[int(group_id)] += 1
    return counts


def assign_agent_groups(config: Toy9Config, rng: np.random.Generator) -> np.ndarray:
    ids: list[int] = []
    for group_id, count in enumerate(group_counts(config)):
        ids.extend([group_id] * int(count))
    group_ids = np.asarray(ids, dtype=np.int64)
    rng.shuffle(group_ids)
    return group_ids


def group_lookup(config: Toy9Config) -> tuple[list[str], list[str], np.ndarray]:
    names = [group.name for group in config.agents.groups]
    local_rules = [group.local_rule for group in config.agents.groups]
    coordination_enabled = np.asarray(
        [group.coordination_enabled for group in config.agents.groups],
        dtype=bool,
    )
    return names, local_rules, coordination_enabled


def initialize_propensities(
    config: Toy9Config,
    group_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    values = np.zeros(config.agents.count, dtype=np.float64)
    for group_id, group in enumerate(config.agents.groups):
        prior = (
            config.environment.initial_action_probability
            if group.initial_action_probability is None
            else group.initial_action_probability
        )
        values[group_ids == group_id] = prior
    if config.agents.init_mode == "independent_init":
        values += rng.normal(loc=0.0, scale=0.04, size=config.agents.count)
    return np.clip(values, 0.0, 1.0)


def action_probabilities(config: Toy9Config, propensities: np.ndarray) -> np.ndarray:
    values = np.clip(propensities, 1e-6, 1.0 - 1e-6)
    logits = np.log(values / (1.0 - values))
    scaled = 1.0 / (1.0 + np.exp(-(logits / config.policy.temperature)))
    if config.policy.exploration_epsilon > 0.0:
        epsilon = config.policy.exploration_epsilon
        scaled = (1.0 - epsilon) * scaled + epsilon * 0.5
    return np.clip(scaled, 0.0, 1.0)


def sample_actions(
    probabilities: np.ndarray,
    config: Toy9Config,
    rng: np.random.Generator,
) -> np.ndarray:
    if config.policy.decision_mode == "argmax":
        return (probabilities >= 0.5).astype(np.int64)
    return (rng.random(len(probabilities)) < probabilities).astype(np.int64)


def neighbor_action_rates(
    actions: np.ndarray,
    neighbors: list[list[int]],
) -> np.ndarray:
    rates = np.zeros(len(actions), dtype=np.float64)
    for agent_id, peers in enumerate(neighbors):
        if peers:
            rates[agent_id] = float(np.mean(actions[peers]))
    return rates


def compute_payoffs(
    actions: np.ndarray,
    local_rates: np.ndarray,
    config: Toy9Config,
) -> np.ndarray:
    cooperative_return = config.environment.benefit * local_rates
    active_payoff = cooperative_return - config.environment.action_cost
    inactive_payoff = 0.25 * config.environment.benefit * (1.0 - local_rates)
    return np.where(actions == 1, active_payoff, inactive_payoff).astype(np.float64)


def select_peer_ids(
    probabilities: np.ndarray,
    neighbors: list[list[int]],
    group_ids: np.ndarray,
    coordination_enabled_by_group: np.ndarray,
    config: Toy9Config,
) -> list[list[int]]:
    if config.coordination.mixer == "none":
        return empty_peers(len(probabilities))
    selected = select_scalar_output_peers(
        neighbors=neighbors,
        values=probabilities,
        peer_rule=config.coordination.peer_rule,
        threshold=config.coordination.threshold,
    ).peer_ids
    gated: list[list[int]] = []
    for agent_id, peers in enumerate(selected):
        if coordination_enabled_by_group[int(group_ids[agent_id])]:
            gated.append(peers)
        else:
            gated.append([])
    return gated


def apply_output_average(
    probabilities: np.ndarray,
    peer_ids: list[list[int]],
    config: Toy9Config,
) -> tuple[np.ndarray, list[float], list[float]]:
    if config.coordination.mixer == "none":
        return (
            probabilities.copy(),
            [0.0 for _ in range(len(probabilities))],
            [0.0 for _ in range(len(probabilities))],
        )
    result = apply_scalar_output_average(
        values=probabilities,
        peer_ids=peer_ids,
        alpha=config.coordination.alpha,
        channel="heterogeneous_action_probability",
        commit_mode="group_gated_probability_commit",
    )
    return (
        np.asarray(result.mix.mixed_values, dtype=np.float64),
        result.commit.losses,
        result.mix.update_norms,
    )


def update_propensities(
    propensities: np.ndarray,
    actions: np.ndarray,
    payoffs: np.ndarray,
    payoff_ema: np.ndarray,
    local_rates: np.ndarray,
    group_ids: np.ndarray,
    config: Toy9Config,
    rng: np.random.Generator,
) -> np.ndarray:
    updated = propensities.copy()
    advantages = payoffs - payoff_ema
    for agent_id in range(config.agents.count):
        group = config.agents.groups[int(group_ids[agent_id])]
        if rng.random() >= group.revision_rate:
            continue
        current = updated[agent_id]
        if group.local_rule == "threshold":
            threshold = config.environment.threshold if group.threshold is None else group.threshold
            target = 1.0 if local_rates[agent_id] >= threshold else 0.0
            delta = group.learning_rate * (target - current)
        elif group.local_rule == "payoff_learning":
            action_direction = float(actions[agent_id]) - current
            delta = group.learning_rate * advantages[agent_id] * action_direction
        else:
            raise ValueError(f"Unsupported Toy 9 local rule: {group.local_rule}")
        updated[agent_id] = current + delta
    return np.clip(updated, 0.0, 1.0)


def masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not bool(np.any(mask)):
        return 0.0
    return float(np.mean(values[mask]))


def aggregate_row(config: Toy9Config, epoch: int, step: Toy9StepResult) -> dict[str, object]:
    peer_graph = graph_from_peer_ids(config.agents.count, step.peer_ids)
    threshold_mask = np.asarray(
        [step.local_rules[int(group_id)] == "threshold" for group_id in step.group_ids],
        dtype=bool,
    )
    payoff_mask = np.asarray(
        [
            step.local_rules[int(group_id)] == "payoff_learning"
            for group_id in step.group_ids
        ],
        dtype=bool,
    )
    coordination_mask = step.coordination_enabled[step.group_ids]
    enabled_rate = masked_mean(step.actions, coordination_mask)
    disabled_rate = masked_mean(step.actions, ~coordination_mask)
    return {
        "run_id": config.run.name,
        "seed": config.run.seed,
        "epoch": epoch,
        "coordination_mixer": config.coordination.mixer,
        "coordination_peer_rule": config.coordination.peer_rule,
        "domain_action_rate": float(np.mean(step.actions)),
        "domain_mean_action_probability": float(np.mean(step.action_probabilities)),
        "domain_mean_payoff": float(np.mean(step.payoffs)),
        "domain_payoff_variance": float(np.var(step.payoffs)),
        "domain_threshold_group_action_rate": masked_mean(step.actions, threshold_mask),
        "domain_payoff_learning_group_action_rate": masked_mean(step.actions, payoff_mask),
        "domain_coordination_enabled_action_rate": enabled_rate,
        "domain_coordination_disabled_action_rate": disabled_rate,
        "domain_group_action_rate_gap": abs(enabled_rate - disabled_rate),
        "fragmentation_components": nx.number_connected_components(peer_graph),
        **aggregate_social_diagnostic_fields(
            peer_ids=step.peer_ids,
            social_losses=step.social_losses,
            social_update_norms=step.social_update_norms,
        ),
    }


def domain_run_settings(config: Toy9Config, config_path: Path) -> DomainRunSettings:
    return DomainRunSettings(
        toy="toy9",
        config=config,
        config_path=config_path,
        output_dir=config.run.output_dir,
        run_name=config.run.name,
        seed=config.run.seed,
        micro_state_fields=TOY9_MICRO_STATE_FIELDS,
        aggregate_fields=TOY9_AGGREGATE_FIELDS,
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "toy": "toy9",
            "policy_rule": config.policy.rule,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_agent_count": config.agents.count,
            "domain_graph_type": config.graph.type,
            "domain_graph_k": config.graph.k,
            "domain_graph_rewire_probability": config.graph.rewire_probability,
            "domain_group_count": len(config.agents.groups),
        },
        logging_interval=config.logging.interval,
        log_micro_state=config.logging.micro_state,
        log_aggregate_metrics=config.logging.aggregate_metrics,
        no_step_error="Toy 9 simulation produced no steps",
    )


def make_run_dir(config: Toy9Config) -> Path:
    return make_domain_run_dir(domain_run_settings(config, Path("<unknown-config>")))


def write_run_metadata(config_path: Path, config: Toy9Config, run_dir: Path) -> None:
    write_domain_run_metadata(domain_run_settings(config, config_path), run_dir)


def micro_rows(
    config: Toy9Config,
    epoch: int,
    step: Toy9StepResult,
) -> list[dict[str, object]]:
    peer_graph = graph_from_peer_ids(config.agents.count, step.peer_ids)
    components = component_map(peer_graph)
    rows: list[dict[str, object]] = []
    for agent_id in range(config.agents.count):
        group_id = int(step.group_ids[agent_id])
        rows.append(
            {
                "run_id": config.run.name,
                "seed": config.run.seed,
                "epoch": epoch,
                "agent_id": agent_id,
                "coordination_mixer": config.coordination.mixer,
                "coordination_peer_rule": config.coordination.peer_rule,
                "domain_agent_group": step.group_names[group_id],
                "domain_local_rule": step.local_rules[group_id],
                "domain_coordination_enabled": bool(
                    step.coordination_enabled[group_id]
                ),
                "domain_action": int(step.actions[agent_id]),
                "domain_action_probability": float(
                    step.action_probabilities[agent_id]
                ),
                "domain_propensity": float(step.propensities[agent_id]),
                "domain_payoff": float(step.payoffs[agent_id]),
                "domain_payoff_ema": float(step.payoff_ema[agent_id]),
                "domain_neighbor_action_rate": float(
                    step.neighbor_action_rates[agent_id]
                ),
                **micro_social_diagnostic_fields(
                    agent_id=agent_id,
                    peer_ids=step.peer_ids,
                    social_losses=step.social_losses,
                    social_update_norms=step.social_update_norms,
                    component_id=components.get(agent_id, -1),
                ),
            }
        )
    return rows


def write_micro_state(
    writer: CsvLogWriter,
    config: Toy9Config,
    epoch: int,
    step: Toy9StepResult,
) -> None:
    for row in micro_rows(config, epoch, step):
        writer.write(row)


@dataclass
class Toy9Adapter:
    config: Toy9Config
    rng: np.random.Generator

    def initialize(self) -> Toy9RunState:
        graph = build_graph(self.config.graph, self.config.agents.count, self.config.run.seed)
        group_ids = assign_agent_groups(self.config, self.rng)
        group_names, local_rules, coordination_enabled = group_lookup(self.config)
        return Toy9RunState(
            rng=self.rng,
            neighbors=graph_neighbors(graph, self.config.agents.count),
            group_ids=group_ids,
            group_names=group_names,
            local_rules=local_rules,
            coordination_enabled=coordination_enabled,
            propensities=initialize_propensities(self.config, group_ids, self.rng),
            payoff_ema=np.zeros(self.config.agents.count, dtype=np.float64),
        )

    def step_epochs(self, state: Toy9RunState) -> range:
        return range(1, self.config.simulation.epochs + 1)

    def step(self, epoch: int, state: Toy9RunState) -> Toy9StepResult:
        base_probabilities = action_probabilities(self.config, state.propensities)
        peer_ids = select_peer_ids(
            base_probabilities,
            state.neighbors,
            state.group_ids,
            state.coordination_enabled,
            self.config,
        )
        mixed_probabilities, social_losses, social_update_norms = (
            apply_output_average(base_probabilities, peer_ids, self.config)
        )
        actions = sample_actions(mixed_probabilities, self.config, state.rng)
        local_rates = neighbor_action_rates(actions, state.neighbors)
        payoffs = compute_payoffs(actions, local_rates, self.config)
        state.propensities = update_propensities(
            state.propensities,
            actions,
            payoffs,
            state.payoff_ema,
            local_rates,
            state.group_ids,
            self.config,
            state.rng,
        )
        state.payoff_ema = (
            self.config.environment.payoff_ema_decay * state.payoff_ema
            + (1.0 - self.config.environment.payoff_ema_decay) * payoffs
        )
        return Toy9StepResult(
            actions=actions.copy(),
            action_probabilities=mixed_probabilities.copy(),
            propensities=state.propensities.copy(),
            payoffs=payoffs.copy(),
            payoff_ema=state.payoff_ema.copy(),
            neighbor_action_rates=local_rates.copy(),
            peer_ids=peer_ids,
            social_losses=social_losses,
            social_update_norms=social_update_norms,
            group_ids=state.group_ids.copy(),
            group_names=state.group_names,
            local_rules=state.local_rules,
            coordination_enabled=state.coordination_enabled.copy(),
        )

    def fallback_step(self, state: Toy9RunState) -> None:
        return None

    def aggregate_row(
        self,
        epoch: int,
        state: Toy9RunState,
        step: Toy9StepResult,
    ) -> dict[str, object]:
        return aggregate_row(self.config, epoch, step)

    def micro_rows(
        self,
        epoch: int,
        state: Toy9RunState,
        step: Toy9StepResult,
    ) -> list[dict[str, object]]:
        return micro_rows(self.config, epoch, step)

    def final_epoch(self, state: Toy9RunState, step: Toy9StepResult) -> int:
        return self.config.simulation.epochs

    def domain_metrics(
        self,
        final_row: dict[str, object],
        state: Toy9RunState,
        step: Toy9StepResult,
    ) -> dict[str, object]:
        return {
            "domain_final_action_rate": final_row["domain_action_rate"],
            "domain_final_mean_action_probability": final_row[
                "domain_mean_action_probability"
            ],
            "domain_final_mean_payoff": final_row["domain_mean_payoff"],
            "domain_final_payoff_variance": final_row["domain_payoff_variance"],
            "domain_final_group_action_rate_gap": final_row[
                "domain_group_action_rate_gap"
            ],
            "domain_final_coordination_enabled_action_rate": final_row[
                "domain_coordination_enabled_action_rate"
            ],
            "domain_final_coordination_disabled_action_rate": final_row[
                "domain_coordination_disabled_action_rate"
            ],
        }


def run_toy9(config: Toy9Config, config_path: Path) -> DomainToyResult:
    """Run Toy 9 from a validated config."""

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    resolve_torch_device(config.simulation.device)
    return DomainToyRunner(
        Toy9Adapter(config=config, rng=rng),
        domain_run_settings(config, config_path),
    ).run()
