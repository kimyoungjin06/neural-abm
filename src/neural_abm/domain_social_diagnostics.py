"""Semantic-free social diagnostics row helpers for domain toy adapters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def aggregate_social_diagnostic_fields(
    *,
    peer_ids: Sequence[Sequence[int]],
    social_losses: Sequence[float],
    social_update_norms: Sequence[float],
) -> dict[str, float]:
    """Return common aggregate peer/social diagnostic fields."""

    agent_count = _validate_social_diagnostic_lengths(
        peer_ids=peer_ids,
        social_losses=social_losses,
        social_update_norms=social_update_norms,
    )
    if agent_count == 0:
        raise ValueError("peer_ids must contain at least one agent")
    return {
        "mean_peer_count": float(np.mean([len(peers) for peers in peer_ids])),
        "mean_social_loss": float(np.mean(social_losses)),
        "mean_social_update_norm": float(np.mean(social_update_norms)),
    }


def micro_social_diagnostic_fields(
    *,
    agent_id: int,
    peer_ids: Sequence[Sequence[int]],
    social_losses: Sequence[float],
    social_update_norms: Sequence[float],
    component_id: int | None = None,
) -> dict[str, object]:
    """Return common per-agent peer/social diagnostic fields."""

    agent_count = _validate_social_diagnostic_lengths(
        peer_ids=peer_ids,
        social_losses=social_losses,
        social_update_norms=social_update_norms,
    )
    if not 0 <= agent_id < agent_count:
        raise IndexError(f"agent_id {agent_id} out of range for {agent_count} agents")
    fields: dict[str, object] = {
        "peer_ids": list(peer_ids[agent_id]),
        "peer_count": len(peer_ids[agent_id]),
    }
    if component_id is not None:
        fields["component_id"] = int(component_id)
    fields["social_loss"] = float(social_losses[agent_id])
    fields["social_update_norm"] = float(social_update_norms[agent_id])
    return fields


def _validate_social_diagnostic_lengths(
    *,
    peer_ids: Sequence[Sequence[int]],
    social_losses: Sequence[float],
    social_update_norms: Sequence[float],
) -> int:
    agent_count = len(peer_ids)
    if len(social_losses) != agent_count:
        raise ValueError(
            "social_losses length must match peer_ids length: "
            f"{len(social_losses)} != {agent_count}"
        )
    if len(social_update_norms) != agent_count:
        raise ValueError(
            "social_update_norms length must match peer_ids length: "
            f"{len(social_update_norms)} != {agent_count}"
        )
    return agent_count
