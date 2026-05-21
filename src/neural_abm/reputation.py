"""Reusable reputation state and reputation-driven imitation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReputationParams:
    enabled: bool = True
    decay: float = 0.9
    peer_rule: str = "spatial"
    temperature: float = 1.0
    noise: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("reputation decay must lie in [0, 1)")
        if self.peer_rule not in {"spatial", "well_mixed"}:
            raise ValueError(f"unsupported reputation peer rule: {self.peer_rule}")
        if self.temperature <= 0.0:
            raise ValueError("reputation temperature must be positive")
        if self.noise < 0.0:
            raise ValueError("reputation noise must be non-negative")


def update_action_reputation(
    reputation: np.ndarray,
    actions: np.ndarray,
    decay: float,
) -> None:
    """Update reputation in-place as an EMA of binary cooperative actions."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("reputation decay must lie in [0, 1)")
    reputation[:] = decay * reputation + (1.0 - decay) * actions.astype(np.float64)


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exp_values = np.exp(shifted)
    return exp_values / float(np.sum(exp_values))


def reputation_imitation_cooperation_probs(
    actions: np.ndarray,
    reputation: np.ndarray,
    peer_ids: list[list[int]],
    revision_mask: np.ndarray,
    rng: np.random.Generator,
    params: ReputationParams | None = None,
) -> np.ndarray:
    """Return cooperation probabilities induced by reputation-ranked peers.

    With zero noise this is deterministic: revised agents copy the action of the
    highest-reputation peer, breaking ties by lower peer id. With positive noise,
    noisy peer reputations are softmaxed into a weighted peer-action probability.
    """

    reputation_params = params or ReputationParams()
    cooperation_probs = actions.astype(np.float64)
    for agent_id, peers in enumerate(peer_ids):
        if not revision_mask[agent_id] or not peers:
            continue
        peer_array = np.asarray(peers, dtype=np.int64)
        if reputation_params.noise <= 0.0:
            best_peer = min(
                (int(peer_id) for peer_id in peer_array),
                key=lambda peer_id: (-float(reputation[peer_id]), peer_id),
            )
            cooperation_probs[agent_id] = float(actions[best_peer])
            continue
        logits = (
            reputation[peer_array].astype(np.float64)
            / reputation_params.temperature
        )
        logits = logits + rng.normal(
            loc=0.0,
            scale=reputation_params.noise,
            size=len(peer_array),
        )
        weights = softmax(logits)
        cooperation_probs[agent_id] = float(
            np.dot(weights, actions[peer_array].astype(np.float64))
        )
    np.clip(cooperation_probs, 0.0, 1.0, out=cooperation_probs)
    return cooperation_probs


def reputation_summary(reputation: np.ndarray) -> dict[str, float]:
    return {
        "mean_reputation": float(np.mean(reputation)),
        "reputation_dispersion": float(np.std(reputation)),
    }


def reputation_observation_extra_dim(mode: str) -> int:
    if mode == "none":
        return 0
    if mode == "self_neighbor_mean":
        return 2
    raise ValueError(f"unsupported reputation observation mode: {mode}")


def reputation_observation_features(
    reputation: np.ndarray,
    peer_ids: list[list[int]],
    mode: str,
) -> np.ndarray:
    """Return optional reputation features for neural observations."""

    if mode == "none":
        return np.zeros((len(reputation), 0), dtype=np.float64)
    if mode != "self_neighbor_mean":
        raise ValueError(f"unsupported reputation observation mode: {mode}")
    if len(peer_ids) != len(reputation):
        raise ValueError("reputation peer_ids length must match reputation length")
    peer_means = np.asarray(
        [
            float(np.mean(reputation[peers])) if peers else 0.0
            for peers in peer_ids
        ],
        dtype=np.float64,
    )
    return np.column_stack([reputation.astype(np.float64), peer_means])
