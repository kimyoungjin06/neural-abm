"""Readiness coordination units shared by NABM binary runners."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def binary_peer_mean_values(
    *,
    peer_ids: list[list[int]],
    values: np.ndarray,
) -> np.ndarray:
    """Return each agent's mean peer value, using zero for peerless agents."""

    agent_count = len(values)
    if len(peer_ids) != agent_count:
        raise ValueError("binary peer value aggregation length mismatch")
    means = np.zeros(agent_count, dtype=np.float64)
    for agent_id, peers in enumerate(peer_ids):
        if not peers:
            continue
        peer_array = np.asarray(peers, dtype=np.int64)
        if np.any((peer_array < 0) | (peer_array >= agent_count)):
            raise ValueError("binary peer value aggregation peer id out of range")
        means[agent_id] = float(np.mean(values[peer_array]))
    return means


def binary_peer_aggregate_values(
    *,
    peer_ids: list[list[int]],
    values: np.ndarray,
    aggregation: str = "mean",
) -> np.ndarray:
    """Return each agent's peer-readiness aggregate."""

    if aggregation == "mean":
        return binary_peer_mean_values(peer_ids=peer_ids, values=values)
    if aggregation != "max":
        raise ValueError("binary readiness aggregation must be one of: mean, max")
    agent_count = len(values)
    if len(peer_ids) != agent_count:
        raise ValueError("binary peer value aggregation length mismatch")
    maximums = np.zeros(agent_count, dtype=np.float64)
    for agent_id, peers in enumerate(peer_ids):
        if not peers:
            continue
        peer_array = np.asarray(peers, dtype=np.int64)
        if np.any((peer_array < 0) | (peer_array >= agent_count)):
            raise ValueError("binary peer value aggregation peer id out of range")
        maximums[agent_id] = float(np.max(values[peer_array]))
    return maximums


@dataclass(frozen=True)
class BinaryReadinessPropagationReport:
    """One peer-readiness propagation step for binary precommitment evidence."""

    enabled: bool
    weight: float
    aggregation: str
    peer_readiness: np.ndarray
    peer_evidence_increment: np.ndarray

    def aggregate_row(self) -> dict[str, object]:
        return {
            "precommitment_peer_evidence_enabled": self.enabled,
            "precommitment_peer_evidence_weight": self.weight,
            "precommitment_peer_readiness_aggregation": self.aggregation,
            "precommitment_peer_readiness_mean": (
                float(np.mean(self.peer_readiness))
                if len(self.peer_readiness)
                else 0.0
            ),
            "precommitment_peer_readiness_active_rate": (
                float(np.mean(self.peer_readiness > 0.0))
                if len(self.peer_readiness)
                else 0.0
            ),
            "precommitment_peer_evidence_increment_mean": (
                float(np.mean(self.peer_evidence_increment))
                if len(self.peer_evidence_increment)
                else 0.0
            ),
        }

    def micro_row(self, agent_id: int) -> dict[str, object]:
        peer_readiness = (
            self.peer_readiness[agent_id]
            if len(self.peer_readiness) > agent_id
            else 0.0
        )
        peer_evidence_increment = (
            self.peer_evidence_increment[agent_id]
            if len(self.peer_evidence_increment) > agent_id
            else 0.0
        )
        return {
            "precommitment_peer_readiness": float(peer_readiness),
            "precommitment_peer_evidence_increment": float(
                peer_evidence_increment
            ),
        }


@dataclass(frozen=True)
class BinaryReadinessPropagationUnit:
    """Propagate ready-state evidence over the binary social neighborhood."""

    enabled: bool = False
    weight: float = 0.0
    aggregation: str = "mean"

    def propagate(
        self,
        *,
        peer_ids: list[list[int]],
        previous_readiness: np.ndarray,
        active: np.ndarray,
        direction_ok: np.ndarray,
    ) -> BinaryReadinessPropagationReport:
        readiness_values = np.asarray(previous_readiness, dtype=np.float64)
        active_values = np.asarray(active, dtype=bool)
        direction_values = np.asarray(direction_ok, dtype=bool)
        agent_count = len(readiness_values)
        if len(active_values) != agent_count or len(direction_values) != agent_count:
            raise ValueError("binary readiness propagation length mismatch")
        if not self.enabled or self.weight <= 0.0:
            zeros = np.zeros(agent_count, dtype=np.float64)
            return BinaryReadinessPropagationReport(
                enabled=self.enabled,
                weight=0.0,
                aggregation=self.aggregation,
                peer_readiness=zeros,
                peer_evidence_increment=zeros.copy(),
            )
        peer_readiness = binary_peer_aggregate_values(
            peer_ids=peer_ids,
            values=readiness_values,
            aggregation=self.aggregation,
        )
        eligible = (~active_values) & direction_values
        peer_evidence_increment = (
            peer_readiness * float(self.weight) * eligible.astype(np.float64)
        )
        return BinaryReadinessPropagationReport(
            enabled=self.enabled,
            weight=float(self.weight),
            aggregation=self.aggregation,
            peer_readiness=peer_readiness,
            peer_evidence_increment=peer_evidence_increment,
        )
