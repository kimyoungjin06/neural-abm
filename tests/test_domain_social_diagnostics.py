from __future__ import annotations

import pytest

from neural_abm.domain_social_diagnostics import (
    aggregate_social_diagnostic_fields,
    micro_social_diagnostic_fields,
)


def test_aggregate_social_diagnostic_fields_maps_common_values() -> None:
    fields = aggregate_social_diagnostic_fields(
        peer_ids=[[1, 2], [], [0]],
        social_losses=[0.3, 0.0, 0.6],
        social_update_norms=[0.1, 0.0, 0.2],
    )

    assert fields == {
        "mean_peer_count": pytest.approx(1.0),
        "mean_social_loss": pytest.approx(0.3),
        "mean_social_update_norm": pytest.approx(0.1),
    }


def test_micro_social_diagnostic_fields_maps_common_values() -> None:
    fields = micro_social_diagnostic_fields(
        agent_id=2,
        peer_ids=[[1, 2], [], [0]],
        social_losses=[0.3, 0.0, 0.6],
        social_update_norms=[0.1, 0.0, 0.2],
        component_id=4,
    )

    assert fields == {
        "peer_ids": [0],
        "peer_count": 1,
        "component_id": 4,
        "social_loss": pytest.approx(0.6),
        "social_update_norm": pytest.approx(0.2),
    }


def test_social_diagnostic_fields_reject_misaligned_lengths() -> None:
    with pytest.raises(ValueError, match="social_losses length"):
        aggregate_social_diagnostic_fields(
            peer_ids=[[1], [0]],
            social_losses=[0.1],
            social_update_norms=[0.2, 0.3],
        )

    with pytest.raises(ValueError, match="social_update_norms length"):
        micro_social_diagnostic_fields(
            agent_id=0,
            peer_ids=[[1], [0]],
            social_losses=[0.1, 0.2],
            social_update_norms=[0.3],
        )


def test_micro_social_diagnostic_fields_rejects_invalid_agent_id() -> None:
    with pytest.raises(IndexError, match="agent_id 3 out of range"):
        micro_social_diagnostic_fields(
            agent_id=3,
            peer_ids=[[1], [0]],
            social_losses=[0.1, 0.2],
            social_update_norms=[0.3, 0.4],
        )
