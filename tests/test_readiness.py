from __future__ import annotations

import numpy as np
import pytest

from neural_abm.readiness import BinaryReadinessPropagationUnit


def test_binary_readiness_propagation_unit_returns_zero_when_inactive() -> None:
    unit = BinaryReadinessPropagationUnit(enabled=False, weight=1.0)

    report = unit.propagate(
        peer_ids=[[1], [0], [0]],
        previous_readiness=np.asarray([1.0, 0.0, 0.0]),
        active=np.zeros(3, dtype=bool),
        direction_ok=np.ones(3, dtype=bool),
    )

    assert report.enabled is False
    assert report.weight == pytest.approx(0.0)
    np.testing.assert_allclose(report.peer_readiness, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(report.peer_evidence_increment, [0.0, 0.0, 0.0])
    assert report.aggregate_row() == {
        "precommitment_peer_evidence_enabled": False,
        "precommitment_peer_evidence_weight": 0.0,
        "precommitment_peer_readiness_aggregation": "mean",
        "precommitment_peer_readiness_mean": 0.0,
        "precommitment_peer_readiness_active_rate": 0.0,
        "precommitment_peer_evidence_increment_mean": 0.0,
    }


def test_binary_readiness_propagation_unit_accumulates_peer_evidence() -> None:
    unit = BinaryReadinessPropagationUnit(enabled=True, weight=0.5)

    report = unit.propagate(
        peer_ids=[[1], [0], [0], [0]],
        previous_readiness=np.asarray([1.0, 0.0, 0.0, 0.5]),
        active=np.asarray([False, False, True, False]),
        direction_ok=np.asarray([True, True, True, False]),
    )

    assert report.enabled is True
    assert report.weight == pytest.approx(0.5)
    np.testing.assert_allclose(report.peer_readiness, [0.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        report.peer_evidence_increment,
        [0.0, 0.5, 0.0, 0.0],
    )
    assert report.aggregate_row() == {
        "precommitment_peer_evidence_enabled": True,
        "precommitment_peer_evidence_weight": 0.5,
        "precommitment_peer_readiness_aggregation": "mean",
        "precommitment_peer_readiness_mean": 0.75,
        "precommitment_peer_readiness_active_rate": 0.75,
        "precommitment_peer_evidence_increment_mean": 0.125,
    }
    assert report.micro_row(1) == {
        "precommitment_peer_readiness": 1.0,
        "precommitment_peer_evidence_increment": 0.5,
    }


def test_binary_readiness_propagation_unit_can_use_max_peer_readiness() -> None:
    unit = BinaryReadinessPropagationUnit(
        enabled=True,
        weight=0.5,
        aggregation="max",
    )

    report = unit.propagate(
        peer_ids=[[1, 2], [0, 2], [0, 1]],
        previous_readiness=np.asarray([1.0, 0.25, 0.0]),
        active=np.asarray([False, False, False]),
        direction_ok=np.asarray([True, True, False]),
    )

    assert report.aggregate_row()["precommitment_peer_readiness_aggregation"] == "max"
    np.testing.assert_allclose(report.peer_readiness, [0.25, 1.0, 1.0])
    np.testing.assert_allclose(
        report.peer_evidence_increment,
        [0.125, 0.5, 0.0],
    )
