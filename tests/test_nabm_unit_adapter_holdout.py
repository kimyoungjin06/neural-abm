from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from neural_abm.readiness import BinaryReadinessPropagationUnit
from neural_abm.spatial_binary import run_binary_policy_learning_step


@dataclass(frozen=True)
class _AdapterOnlyAgent:
    agent_id: int


@dataclass
class _AdapterOnlyThresholdDomain:
    agents: list[_AdapterOnlyAgent]
    calls: list[str] = field(default_factory=list)
    refreshed_agent_ids: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.actions = np.zeros(len(self.agents), dtype=np.int64)

    def observations(self) -> torch.Tensor:
        return torch.tensor(
            [
                [0.80, 0.10, 0.40],
                [0.10, 0.10, 0.50],
                [0.90, 0.20, 0.70],
                [0.20, 0.00, 0.70],
            ],
            dtype=torch.float32,
        )

    def collect_policy_probs(
        self,
        agents: list[_AdapterOnlyAgent],
        observations: torch.Tensor,
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert agents == self.agents
        self.calls.append("readout")
        action_bonus = torch.as_tensor(self.actions, dtype=torch.float32) * 0.20
        logits = (
            observations[:, 0]
            + observations[:, 1]
            - observations[:, 2]
            + action_bonus
        ) / float(temperature)
        adopt_probs = torch.sigmoid(logits)
        return torch.stack((1.0 - adopt_probs, adopt_probs), dim=1)

    def decision_action_probs(self, policy_probs: torch.Tensor) -> torch.Tensor:
        self.calls.append("decision")
        return policy_probs

    def sample_actions(self, action_probs: torch.Tensor) -> np.ndarray:
        self.calls.append("sample")
        return (action_probs[:, 1].detach().cpu().numpy() >= 0.5).astype(np.int64)

    def local_update(self, actions: np.ndarray) -> list[float]:
        self.calls.append("local")
        self.actions = np.asarray(actions, dtype=np.int64)
        target_actions = np.asarray([1, 0, 1, 0], dtype=np.int64)
        return np.abs(self.actions - target_actions).astype(float).tolist()

    def refresh_policy_cache(self, agents: list[_AdapterOnlyAgent]) -> None:
        self.calls.append("refresh")
        self.refreshed_agent_ids = [agent.agent_id for agent in agents]


def test_adapter_only_threshold_domain_uses_binary_policy_unit_without_src_changes() -> None:
    domain = _AdapterOnlyThresholdDomain(
        agents=[_AdapterOnlyAgent(agent_id) for agent_id in range(4)]
    )

    result = run_binary_policy_learning_step(
        agents=domain.agents,
        observations=domain.observations(),
        temperature=1.0,
        collect_policy_probs=domain.collect_policy_probs,
        decision_action_probs=domain.decision_action_probs,
        sample_actions=domain.sample_actions,
        local_update=domain.local_update,
        refresh_policy_cache=domain.refresh_policy_cache,
        extras={"domain": "adapter_only_threshold_holdout"},
    )

    assert domain.calls == [
        "readout",
        "decision",
        "sample",
        "local",
        "refresh",
        "readout",
    ]
    assert domain.refreshed_agent_ids == [0, 1, 2, 3]
    np.testing.assert_array_equal(result.actions_after_revision, [1, 0, 1, 0])
    assert result.local_losses == [0.0, 0.0, 0.0, 0.0]
    assert result.extras == {"domain": "adapter_only_threshold_holdout"}
    assert result.post_local_probs[0, 1] > result.pre_revision_probs[0, 1]
    assert result.post_local_probs[2, 1] > result.pre_revision_probs[2, 1]


def test_adapter_only_threshold_domain_uses_readiness_unit_without_src_changes() -> None:
    previous_readiness = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    active = np.asarray([True, False, True, False], dtype=bool)
    direction_ok = np.ones(4, dtype=bool)

    report = BinaryReadinessPropagationUnit(
        enabled=True,
        weight=0.5,
        aggregation="max",
    ).propagate(
        peer_ids=[[1], [0, 2], [1], [0, 2]],
        previous_readiness=previous_readiness,
        active=active,
        direction_ok=direction_ok,
    )

    np.testing.assert_allclose(report.peer_readiness, [0.0, 1.0, 0.0, 1.0])
    np.testing.assert_allclose(
        report.peer_evidence_increment,
        [0.0, 0.5, 0.0, 0.5],
    )
    assert report.aggregate_row() == {
        "precommitment_peer_evidence_enabled": True,
        "precommitment_peer_evidence_weight": 0.5,
        "precommitment_peer_readiness_aggregation": "max",
        "precommitment_peer_readiness_mean": 0.5,
        "precommitment_peer_readiness_active_rate": 0.5,
        "precommitment_peer_evidence_increment_mean": 0.25,
    }
