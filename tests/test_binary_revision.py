from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from neural_abm.binary_revision import (
    REVISION_STAY,
    REVISION_SWITCH_TO_ONE,
    REVISION_SWITCH_TO_ZERO,
    BinaryRevisionLearningCallbacks,
    BinaryRevisionLearningUnit,
    apply_binary_revision_choices,
    binary_revision_effective_stay_probabilities,
    binary_revision_probabilities_from_action_probs,
    binary_revision_switch_probabilities,
    normalize_binary_revision_probabilities,
)


def test_binary_revision_learning_unit_runs_revision_lifecycle() -> None:
    agents = ["a0", "a1", "a2"]
    observations = torch.eye(3, dtype=torch.float32)
    current_actions = np.asarray([0, 1, 0], dtype=np.int64)
    calls: list[str] = []

    def collect_signals(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        actions_arg: np.ndarray,
    ) -> dict[str, np.ndarray]:
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        np.testing.assert_array_equal(actions_arg, [0, 1, 0])
        calls.append("signals")
        return {"revision_pressure": np.asarray([0.8, 0.7, 0.1])}

    def collect_probs(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        actions_arg: np.ndarray,
        signals: dict[str, np.ndarray],
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        np.testing.assert_array_equal(actions_arg, [0, 1, 0])
        np.testing.assert_allclose(signals["revision_pressure"], [0.8, 0.7, 0.1])
        assert temperature == pytest.approx(0.5)
        calls.append("readout")
        return torch.tensor(
            [
                [0.2, 0.7, 0.1],
                [0.4, 0.2, 0.4],
                [0.8, 0.2, 0.0],
            ],
            dtype=torch.float32,
        )

    def sample_choices(
        revision_probs: torch.Tensor,
        actions_arg: np.ndarray,
    ) -> np.ndarray:
        np.testing.assert_array_equal(actions_arg, [0, 1, 0])
        torch.testing.assert_close(
            revision_probs.sum(dim=1),
            torch.ones(3, dtype=torch.float32),
        )
        calls.append("sample")
        return np.asarray(
            [REVISION_SWITCH_TO_ONE, REVISION_SWITCH_TO_ZERO, REVISION_STAY],
            dtype=np.int64,
        )

    def local_update(actions_arg: np.ndarray) -> list[float]:
        np.testing.assert_array_equal(actions_arg, [1, 0, 0])
        calls.append("local")
        return [0.1, 0.2, 0.0]

    def refresh(agents_arg: list[str]) -> None:
        assert agents_arg == agents
        calls.append("refresh")

    def collect_post_signals(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        actions_arg: np.ndarray,
    ) -> dict[str, np.ndarray]:
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        np.testing.assert_array_equal(actions_arg, [1, 0, 0])
        calls.append("post_signals")
        return {"revision_pressure": np.asarray([0.1, 0.2, 0.1])}

    def collect_post_probs(
        agents_arg: list[str],
        observations_arg: torch.Tensor,
        actions_arg: np.ndarray,
        signals: dict[str, np.ndarray],
        *,
        temperature: float,
    ) -> torch.Tensor:
        assert agents_arg == agents
        torch.testing.assert_close(observations_arg, observations)
        np.testing.assert_array_equal(actions_arg, [1, 0, 0])
        np.testing.assert_allclose(signals["revision_pressure"], [0.1, 0.2, 0.1])
        assert temperature == pytest.approx(0.5)
        calls.append("post_readout")
        return torch.tensor(
            [
                [0.9, 0.1, 0.0],
                [0.8, 0.2, 0.0],
                [0.9, 0.1, 0.0],
            ],
            dtype=torch.float32,
        )

    result = BinaryRevisionLearningUnit(
        agents=agents,
        observations=observations,
        current_actions=current_actions,
        temperature=0.5,
        callbacks=BinaryRevisionLearningCallbacks(
            collect_revision_signals=collect_signals,
            collect_revision_probs=collect_probs,
            sample_revision_choices=sample_choices,
            local_update=local_update,
            refresh_revision_cache=refresh,
            post_collect_revision_signals=collect_post_signals,
            post_collect_revision_probs=collect_post_probs,
        ),
    ).run()

    assert calls == [
        "signals",
        "readout",
        "sample",
        "local",
        "refresh",
        "post_signals",
        "post_readout",
    ]
    np.testing.assert_array_equal(result.actions_after_revision, [1, 0, 0])
    np.testing.assert_array_equal(result.revision_mask, [True, True, False])
    assert result.realized_revision_rate == pytest.approx(2.0 / 3.0)
    aggregate = result.aggregate_row()
    assert aggregate["mean_revision_switch_probability"] == pytest.approx(
        (0.7 + 0.4 + 0.2) / 3.0
    )
    assert aggregate["revision_choice_switch_to_one_rate"] == pytest.approx(1.0 / 3.0)
    assert aggregate["revision_choice_switch_to_zero_rate"] == pytest.approx(1.0 / 3.0)
    assert aggregate["action_rate_after_revision"] == pytest.approx(1.0 / 3.0)
    assert aggregate["mean_revision_local_loss"] == pytest.approx(0.1)
    rows = result.micro_rows()
    assert rows[0]["revision_choice"] == "switch_to_1"
    assert rows[0]["revision_switch_probability"] == pytest.approx(0.7)
    assert rows[1]["revision_choice"] == "switch_to_0"
    assert rows[1]["revision_switch_probability"] == pytest.approx(0.4)
    assert rows[2]["revision_choice"] == "stay"
    assert rows[2]["revised"] is False


def test_binary_revision_helpers_normalize_and_apply_choices() -> None:
    probs = normalize_binary_revision_probabilities(
        np.asarray(
            [
                [1.0, 3.0, 0.0],
                [2.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
    )
    np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
    current = np.asarray([0, 1], dtype=np.int64)
    np.testing.assert_allclose(
        binary_revision_switch_probabilities(probs, current),
        [0.75, 0.25],
    )
    np.testing.assert_allclose(
        binary_revision_effective_stay_probabilities(probs, current),
        [0.25, 0.75],
    )
    revised = apply_binary_revision_choices(
        current,
        np.asarray([REVISION_SWITCH_TO_ONE, REVISION_STAY], dtype=np.int64),
    )
    np.testing.assert_array_equal(revised, [1, 1])


def test_binary_revision_probabilities_from_action_probs_preserve_action_meaning() -> None:
    current = np.asarray([0, 1, 0, 1], dtype=np.int64)
    action_probs = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.1, 0.9],
            [0.6, 0.4],
        ],
        dtype=np.float64,
    )

    revision_probs = binary_revision_probabilities_from_action_probs(
        action_probs,
        current,
    )

    np.testing.assert_allclose(
        revision_probs,
        [
            [0.8, 0.2, 0.0],
            [0.7, 0.0, 0.3],
            [0.1, 0.9, 0.0],
            [0.4, 0.0, 0.6],
        ],
    )
    np.testing.assert_allclose(
        binary_revision_switch_probabilities(revision_probs, current),
        [0.2, 0.3, 0.9, 0.6],
    )


@pytest.mark.parametrize(
    "values",
    [
        [[0.5, 0.5]],
        [[0.5, -0.1, 0.6]],
        [[0.0, 0.0, 0.0]],
        [[0.1, float("nan"), 0.9]],
    ],
)
def test_normalize_binary_revision_probabilities_rejects_invalid_values(
    values: list[list[float]],
) -> None:
    with pytest.raises(ValueError):
        normalize_binary_revision_probabilities(values)


def test_binary_revision_learning_unit_rejects_invalid_sampled_choices() -> None:
    def collect_probs(*args: Any, **kwargs: Any) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)

    unit = BinaryRevisionLearningUnit(
        agents=["a0"],
        observations=np.asarray([[1.0]], dtype=np.float64),
        current_actions=np.asarray([0], dtype=np.int64),
        temperature=1.0,
        callbacks=BinaryRevisionLearningCallbacks(
            collect_revision_probs=collect_probs,
            sample_revision_choices=lambda probs, actions: np.asarray([3]),
            local_update=lambda actions: [0.0],
        ),
    )

    with pytest.raises(ValueError, match="stay/switch_to_1/switch_to_0"):
        unit.run()
