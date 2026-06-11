"""Minimal stable-API NABM example.

This demo imports only from ``neural_abm.api``. It is a release-smoke example
for the public v0 facade, not a paper evidence case.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from neural_abm.api import (
    SCALAR_PROBABILITY_CHANNEL,
    CommitReport,
    NABMAgent,
    NABMStep,
    NABMUnit,
    ObservationSpec,
    SocialBlock,
    SocialChannel,
    SocialMessageSpec,
    SocialMixResult,
    scalar_message_values,
)


@dataclass
class BeliefAgent:
    agent_id: int
    belief_probability: float
    local_target: float
    local_learning_rate: float = 0.25

    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec(
            name="minimal_belief_context",
            tensor_shape=(1,),
            dtype=torch.float32,
            description="target belief probability",
        )

    def social_message_spec(self) -> SocialMessageSpec:
        return SocialMessageSpec(
            required_keys=(
                "agent_id",
                "belief_probability",
                "latent_summary",
                "confidence",
                "param_norm",
            ),
            tensor_keys=("latent_summary",),
            probability_keys=("belief_probability",),
        )

    def observe(self, x: Mapping[str, float] | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.detach().clone().to(dtype=torch.float32)
        return torch.tensor([float(x["local_target"])], dtype=torch.float32)

    def act_or_predict(self, observation: torch.Tensor) -> torch.Tensor:
        observed = self.observe(observation)
        target = torch.clamp(observed[0], 0.0, 1.0)
        current = torch.tensor(float(self.belief_probability), dtype=torch.float32)
        return torch.clamp(0.75 * current + 0.25 * target, 0.0, 1.0).reshape(1)

    def local_update(self, observation: torch.Tensor | None = None) -> float:
        if observation is None:
            observation = torch.tensor([self.local_target], dtype=torch.float32)
        before = float(self.belief_probability)
        target = float(self.act_or_predict(observation)[0])
        self.belief_probability = float(
            np.clip(
                (1.0 - self.local_learning_rate) * before
                + self.local_learning_rate * target,
                0.0,
                1.0,
            )
        )
        return abs(self.belief_probability - before)

    def social_message(self, observation: torch.Tensor | None = None) -> dict[str, Any]:
        del observation
        return {
            "agent_id": self.agent_id,
            "belief_probability": float(self.belief_probability),
            "latent_summary": torch.tensor(
                [self.belief_probability, self.local_target],
                dtype=torch.float32,
            ),
            "confidence": 1.0,
            "param_norm": 0.0,
        }

    def log_state(self, observation: torch.Tensor | None = None) -> dict[str, Any]:
        del observation
        return {
            "agent_id": self.agent_id,
            "belief_probability": float(self.belief_probability),
            "local_target": float(self.local_target),
        }


@dataclass
class BeliefProbabilityCommit:
    agents: Sequence[BeliefAgent]

    def commit(self, mix_result: SocialMixResult) -> CommitReport:
        values = np.asarray(mix_result.mixed_values, dtype=np.float64)
        committed: list[int] = []
        losses = [0.0 for _agent in self.agents]
        for agent_id, agent in enumerate(self.agents):
            before = float(agent.belief_probability)
            after = float(values[agent_id])
            agent.belief_probability = after
            committed.append(agent.agent_id)
            losses[agent_id] = abs(after - before)
        return CommitReport.from_mix_result(
            mix_result=mix_result,
            committed_agent_ids=committed,
            losses=losses,
        )


def make_agents(agent_count: int, seed: int) -> list[BeliefAgent]:
    rng = np.random.default_rng(seed)
    targets = np.linspace(0.15, 0.85, agent_count)
    return [
        BeliefAgent(
            agent_id=agent_id,
            belief_probability=float(rng.uniform(0.15, 0.85)),
            local_target=float(targets[agent_id]),
        )
        for agent_id in range(agent_count)
    ]


def ring_peer_selector(messages: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    agent_count = len(messages)
    return [
        [int((agent_id - 1) % agent_count), int((agent_id + 1) % agent_count)]
        for agent_id in range(agent_count)
    ]


def run_demo(*, seed: int = 1, steps: int = 5, agent_count: int = 8) -> dict[str, Any]:
    agents = make_agents(agent_count=agent_count, seed=seed)
    channel = SocialChannel(
        name="belief_probability",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="belief_probability_commit",
    )
    unit = NABMUnit(
        agents=agents,
        step=NABMStep(
            social_block=SocialBlock(alpha=0.4),
            channel=channel,
            commit_adapter=BeliefProbabilityCommit(agents),
        ),
        peer_selector=ring_peer_selector,
        social_value_builder=scalar_message_values("belief_probability"),
    )

    history: list[dict[str, Any]] = []
    for _step in range(steps):
        report = unit.run()
        aggregate = report.aggregate_row()
        history.append(
            {
                "mean_local_loss": aggregate["mean_local_loss"],
                "mean_social_loss": aggregate["mean_social_loss"],
                "mean_social_update_norm": aggregate["mean_social_update_norm"],
                "social_channel": aggregate["social_channel"],
                "commit_mode": aggregate["commit_mode"],
            }
        )

    beliefs = np.asarray(
        [agent.belief_probability for agent in agents],
        dtype=np.float64,
    )
    return {
        "agent_count": agent_count,
        "steps": steps,
        "social_channel": "belief_probability",
        "commit_mode": "belief_probability_commit",
        "mean_belief_probability": float(np.mean(beliefs)),
        "belief_dispersion": float(np.std(beliefs)),
        "history": history,
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), indent=2, sort_keys=True))

