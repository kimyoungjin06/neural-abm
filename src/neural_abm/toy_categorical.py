"""Toy 6: multi-action categorical spatial game."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from neural_abm.accelerator import resolve_torch_device
from neural_abm.config import Toy6Config
from neural_abm.domain_runner import (
    DomainRunSettings,
    DomainToyRunner,
    make_timestamped_run_dir,
)
from neural_abm.domain_social_diagnostics import (
    aggregate_social_diagnostic_fields,
    micro_social_diagnostic_fields,
)
from neural_abm.graphs import component_map, graph_from_peer_ids
from neural_abm.mixers import apply_distribution_output_average
from neural_abm.results import (
    DomainToyResult,
    write_run_metadata_artifacts,
)
from neural_abm.social import (
    empty_peers,
    select_distribution_output_peers,
)


TOY6_MICRO_STATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "agent_id",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_strategy",
    "domain_payoff",
    "domain_payoff_ema",
    "domain_strategy_probability",
    "domain_dominant_strategy",
    "peer_ids",
    "peer_count",
    "component_id",
    "social_loss",
    "social_update_norm",
]


TOY6_AGGREGATE_FIELDS = [
    "run_id",
    "seed",
    "epoch",
    "coordination_mixer",
    "coordination_peer_rule",
    "domain_mean_payoff",
    "domain_strategy_entropy",
    "domain_dominant_strategy",
    "domain_dominant_strategy_fraction",
    "fragmentation_components",
    "mean_peer_count",
    "mean_social_loss",
    "mean_social_update_norm",
]


@dataclass(frozen=True)
class Toy6StepResult:
    actions: np.ndarray
    payoffs: np.ndarray
    probabilities: np.ndarray
    peer_ids: list[list[int]]
    social_losses: list[float]
    social_update_norms: list[float]


@dataclass
class Toy6RunState:
    rng: np.random.Generator
    device: torch.device
    neighbors: list[list[int]]
    logits: np.ndarray
    payoff_ema: np.ndarray


def set_global_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def grid_neighbors(
    width: int,
    height: int,
    *,
    periodic: bool,
) -> list[list[int]]:
    neighbors: list[list[int]] = []
    for y in range(height):
        for x in range(width):
            peers: list[int] = []
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                nx_pos = x + dx
                ny_pos = y + dy
                if periodic:
                    nx_pos %= width
                    ny_pos %= height
                if 0 <= nx_pos < width and 0 <= ny_pos < height:
                    peers.append(ny_pos * width + nx_pos)
            neighbors.append(peers)
    return neighbors


def strategy_prior(config: Toy6Config) -> np.ndarray:
    if config.environment.initial_strategy_probabilities is None:
        return np.full(config.game.strategy_count, 1.0 / config.game.strategy_count)
    values = np.asarray(config.environment.initial_strategy_probabilities, dtype=float)
    return values / values.sum()


def initialize_logits(config: Toy6Config, rng: np.random.Generator) -> np.ndarray:
    prior = np.log(np.clip(strategy_prior(config), 1e-8, 1.0))
    logits = np.tile(prior, (config.agent_count, 1))
    if config.agents.init_mode == "independent_init" and config.agents.logit_noise > 0.0:
        logits += rng.normal(
            loc=0.0,
            scale=config.agents.logit_noise,
            size=logits.shape,
        )
    return logits.astype(np.float64)


def softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def sample_actions(
    probabilities: np.ndarray,
    rng: np.random.Generator,
    *,
    mode: str,
) -> np.ndarray:
    normalized = normalize_probabilities(probabilities)
    if mode == "argmax":
        return np.argmax(normalized, axis=1).astype(np.int64)
    return np.asarray(
        [
            rng.choice(normalized.shape[1], p=normalized[agent_id])
            for agent_id in range(normalized.shape[0])
        ],
        dtype=np.int64,
    )


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    return clipped / np.sum(clipped, axis=1, keepdims=True)


def compute_cyclic_payoffs(
    actions: np.ndarray,
    neighbors: list[list[int]],
    *,
    strategy_count: int,
    win_payoff: float,
    loss_payoff: float,
    draw_payoff: float,
) -> np.ndarray:
    payoffs = np.zeros(len(actions), dtype=np.float64)
    for agent_id, peers in enumerate(neighbors):
        if not peers:
            continue
        total = 0.0
        for peer_id in peers:
            diff = (int(actions[agent_id]) - int(actions[peer_id])) % strategy_count
            if diff == 0:
                total += draw_payoff
            elif diff == 1:
                total += win_payoff
            elif diff == strategy_count - 1:
                total += loss_payoff
            else:
                total += draw_payoff
        payoffs[agent_id] = total / len(peers)
    return payoffs


def update_logits(
    logits: np.ndarray,
    actions: np.ndarray,
    payoffs: np.ndarray,
    payoff_ema: np.ndarray,
    revision_mask: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    updated = logits.copy()
    advantages = payoffs - payoff_ema
    for agent_id in np.flatnonzero(revision_mask):
        updated[int(agent_id), int(actions[int(agent_id)])] += (
            learning_rate * advantages[int(agent_id)]
        )
    updated -= np.mean(updated, axis=1, keepdims=True)
    return np.clip(updated, -20.0, 20.0)


def select_peer_ids(
    probabilities: np.ndarray,
    neighbors: list[list[int]],
    config: Toy6Config,
) -> list[list[int]]:
    if config.coordination.mixer == "none":
        return empty_peers(len(probabilities))
    selected = select_distribution_output_peers(
        neighbors=neighbors,
        probe_probs=probabilities,
        peer_rule=config.coordination.peer_rule,
        threshold=config.coordination.threshold,
    ).peer_ids
    return selected


def apply_output_average(
    probabilities: np.ndarray,
    peer_ids: list[list[int]],
    config: Toy6Config,
    device: torch.device,
) -> tuple[np.ndarray, list[float], list[float]]:
    if config.coordination.mixer == "none":
        return (
            probabilities.copy(),
            [0.0 for _ in range(len(probabilities))],
            [0.0 for _ in range(len(probabilities))],
        )
    result = apply_distribution_output_average(
        values=torch.as_tensor(probabilities, dtype=torch.float32, device=device),
        peer_ids=peer_ids,
        alpha=config.coordination.alpha,
        channel="strategy_distribution",
        commit_mode="categorical_probability_commit",
    )
    mixed = normalize_probabilities(result.mix.mixed_values.detach().cpu().numpy())
    return (
        mixed,
        result.commit.losses,
        result.mix.update_norms,
    )


def strategy_entropy(actions: np.ndarray, strategy_count: int) -> float:
    counts = np.bincount(actions, minlength=strategy_count).astype(np.float64)
    probs = counts / max(float(np.sum(counts)), 1.0)
    active = probs[probs > 0.0]
    if len(active) == 0:
        return 0.0
    return float(-(active * np.log(active)).sum() / np.log(strategy_count))


def aggregate_row(
    config: Toy6Config,
    epoch: int,
    step: Toy6StepResult,
) -> dict[str, object]:
    counts = np.bincount(step.actions, minlength=config.game.strategy_count)
    dominant_strategy = int(np.argmax(counts))
    peer_graph = graph_from_peer_ids(config.agent_count, step.peer_ids)
    return {
        "run_id": config.run.name,
        "seed": config.run.seed,
        "epoch": epoch,
        "coordination_mixer": config.coordination.mixer,
        "coordination_peer_rule": config.coordination.peer_rule,
        "domain_mean_payoff": float(np.mean(step.payoffs)),
        "domain_strategy_entropy": strategy_entropy(
            step.actions,
            config.game.strategy_count,
        ),
        "domain_dominant_strategy": dominant_strategy,
        "domain_dominant_strategy_fraction": float(
            counts[dominant_strategy] / len(step.actions)
        ),
        "fragmentation_components": nx.number_connected_components(peer_graph),
        **aggregate_social_diagnostic_fields(
            peer_ids=step.peer_ids,
            social_losses=step.social_losses,
            social_update_norms=step.social_update_norms,
        ),
    }


def domain_run_settings(config: Toy6Config, config_path: Path) -> DomainRunSettings:
    return DomainRunSettings(
        toy="toy6",
        config=config,
        config_path=config_path,
        output_dir=config.run.output_dir,
        run_name=config.run.name,
        seed=config.run.seed,
        micro_state_fields=TOY6_MICRO_STATE_FIELDS,
        aggregate_fields=TOY6_AGGREGATE_FIELDS,
        metadata={
            "run_name": config.run.name,
            "seed": config.run.seed,
            "toy": "toy6",
            "policy_rule": config.policy.rule,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_strategy_count": config.game.strategy_count,
            "domain_grid_width": config.environment.grid_width,
            "domain_grid_height": config.environment.grid_height,
        },
        logging_interval=config.logging.interval,
        log_micro_state=config.logging.micro_state,
        log_aggregate_metrics=config.logging.aggregate_metrics,
        no_step_error="Toy 6 produced no simulation steps",
    )


def make_run_dir(config: Toy6Config) -> Path:
    settings = domain_run_settings(config, Path("<unknown-config>"))
    return make_timestamped_run_dir(
        output_dir=settings.output_dir,
        run_name=settings.run_name,
        seed=settings.seed,
    )


def write_run_metadata(config_path: Path, config: Toy6Config, run_dir: Path) -> None:
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
    config: Toy6Config,
    epoch: int,
    step: Toy6StepResult,
    payoff_ema: np.ndarray,
) -> list[dict[str, object]]:
    peer_graph = graph_from_peer_ids(config.agent_count, step.peer_ids)
    components = component_map(peer_graph)
    dominant = np.argmax(step.probabilities, axis=1)
    return [
        {
            "run_id": config.run.name,
            "seed": config.run.seed,
            "epoch": epoch,
            "agent_id": agent_id,
            "coordination_mixer": config.coordination.mixer,
            "coordination_peer_rule": config.coordination.peer_rule,
            "domain_strategy": int(step.actions[agent_id]),
            "domain_payoff": float(step.payoffs[agent_id]),
            "domain_payoff_ema": float(payoff_ema[agent_id]),
            "domain_strategy_probability": step.probabilities[agent_id].tolist(),
            "domain_dominant_strategy": int(dominant[agent_id]),
            **micro_social_diagnostic_fields(
                agent_id=agent_id,
                peer_ids=step.peer_ids,
                social_losses=step.social_losses,
                social_update_norms=step.social_update_norms,
                component_id=components.get(agent_id, -1),
            ),
        }
        for agent_id in range(config.agent_count)
    ]


@dataclass
class Toy6Adapter:
    config: Toy6Config
    rng: np.random.Generator
    device: torch.device

    def initialize(self) -> Toy6RunState:
        return Toy6RunState(
            rng=self.rng,
            device=self.device,
            neighbors=grid_neighbors(
                self.config.environment.grid_width,
                self.config.environment.grid_height,
                periodic=self.config.graph.periodic,
            ),
            logits=initialize_logits(self.config, self.rng),
            payoff_ema=np.zeros(self.config.agent_count, dtype=np.float64),
        )

    def step_epochs(self, state: Toy6RunState) -> range:
        return range(1, self.config.simulation.epochs + 1)

    def step(self, epoch: int, state: Toy6RunState) -> Toy6StepResult:
        local_probs = softmax(state.logits, self.config.policy.temperature)
        local_actions = sample_actions(
            local_probs,
            state.rng,
            mode=self.config.policy.decision_mode,
        )
        payoffs = compute_cyclic_payoffs(
            local_actions,
            state.neighbors,
            strategy_count=self.config.game.strategy_count,
            win_payoff=self.config.game.win_payoff,
            loss_payoff=self.config.game.loss_payoff,
            draw_payoff=self.config.game.draw_payoff,
        )
        revision_mask = (
            state.rng.random(self.config.agent_count) < self.config.policy.revision_rate
        )
        state.logits = update_logits(
            logits=state.logits,
            actions=local_actions,
            payoffs=payoffs,
            payoff_ema=state.payoff_ema,
            revision_mask=revision_mask,
            learning_rate=self.config.policy.learning_rate,
        )
        state.payoff_ema = (
            self.config.environment.reward_ema_decay * state.payoff_ema
            + (1.0 - self.config.environment.reward_ema_decay) * payoffs
        )
        candidate_probs = softmax(state.logits, self.config.policy.temperature)
        peer_ids = select_peer_ids(candidate_probs, state.neighbors, self.config)
        final_probs, social_losses, social_update_norms = apply_output_average(
            candidate_probs,
            peer_ids,
            self.config,
            state.device,
        )
        state.logits = np.log(np.clip(final_probs, 1e-8, 1.0))
        actions = sample_actions(
            final_probs,
            state.rng,
            mode=self.config.policy.decision_mode,
        )
        return Toy6StepResult(
            actions=actions,
            payoffs=payoffs,
            probabilities=final_probs,
            peer_ids=peer_ids,
            social_losses=social_losses,
            social_update_norms=social_update_norms,
        )

    def fallback_step(self, state: Toy6RunState) -> None:
        return None

    def aggregate_row(
        self,
        epoch: int,
        state: Toy6RunState,
        step: Toy6StepResult,
    ) -> dict[str, object]:
        return aggregate_row(self.config, epoch, step)

    def micro_rows(
        self,
        epoch: int,
        state: Toy6RunState,
        step: Toy6StepResult,
    ) -> list[dict[str, object]]:
        return micro_rows(self.config, epoch, step, state.payoff_ema)

    def final_epoch(self, state: Toy6RunState, step: Toy6StepResult) -> int:
        return self.config.simulation.epochs

    def domain_metrics(
        self,
        final_row: dict[str, object],
        state: Toy6RunState,
        step: Toy6StepResult,
    ) -> dict[str, object]:
        return {
            "domain_final_mean_payoff": final_row["domain_mean_payoff"],
            "domain_final_strategy_entropy": final_row["domain_strategy_entropy"],
            "domain_final_dominant_strategy": final_row["domain_dominant_strategy"],
            "domain_final_dominant_strategy_fraction": final_row[
                "domain_dominant_strategy_fraction"
            ],
        }


def run_toy6(config: Toy6Config, config_path: Path) -> DomainToyResult:
    """Run Toy 6 from a validated config."""

    set_global_seeds(config.run.seed)
    rng = np.random.default_rng(config.run.seed)
    device = resolve_torch_device(config.simulation.device)
    return DomainToyRunner(
        Toy6Adapter(
            config=config,
            rng=rng,
            device=device,
        ),
        domain_run_settings(config, config_path),
    ).run()
