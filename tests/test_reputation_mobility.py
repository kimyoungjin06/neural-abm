from __future__ import annotations

import numpy as np
import pytest

from neural_abm.mobility import (
    MobilityParams,
    apply_local_quality_mobility,
    local_quality,
)
from neural_abm.reputation import (
    ReputationParams,
    reputation_imitation_cooperation_probs,
    reputation_observation_extra_dim,
    reputation_observation_features,
    reputation_summary,
    update_action_reputation,
)


def test_action_reputation_ema_is_domain_agnostic() -> None:
    reputation = np.array([0.2, 0.8, 0.0], dtype=np.float64)
    actions = np.array([1, 0, 1], dtype=np.int64)

    update_action_reputation(reputation=reputation, actions=actions, decay=0.5)

    assert reputation.tolist() == pytest.approx([0.6, 0.4, 0.5])
    summary = reputation_summary(reputation)
    assert summary["mean_reputation"] == pytest.approx(0.5)
    assert summary["reputation_dispersion"] == pytest.approx(np.std(reputation))


def test_reputation_imitation_uses_supplied_peer_ids_without_toy_config() -> None:
    actions = np.array([0, 1, 0], dtype=np.int64)
    reputation = np.array([0.4, 0.8, 0.9], dtype=np.float64)
    peer_ids = [[1, 2], [0, 2], [0, 1]]

    cooperation_probs = reputation_imitation_cooperation_probs(
        actions=actions,
        reputation=reputation,
        peer_ids=peer_ids,
        revision_mask=np.ones(3, dtype=bool),
        rng=np.random.default_rng(7),
        params=ReputationParams(noise=0.0),
    )

    assert cooperation_probs.tolist() == [0.0, 0.0, 1.0]


def test_noisy_reputation_imitation_clips_floating_roundoff_to_probability_bounds() -> None:
    actions = np.array(
        [0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
        dtype=np.int64,
    )
    reputation = np.array(
        [0.0, 0.0, 0.9, 0.1, 1.0, 1.0, 0.1, 1.0, 0.0, 1.0],
        dtype=np.float64,
    )
    peer_ids = [
        [1, 9, 2, 3],
        [0, 2, 3, 4],
        [1, 3, 4, 5],
        [2, 4, 5, 6],
        [3, 5, 6, 7],
        [4, 6, 7, 8],
        [5, 7, 8, 9],
        [6, 8, 9, 0],
        [7, 9, 0, 1],
        [8, 0, 1, 2],
    ]

    cooperation_probs = reputation_imitation_cooperation_probs(
        actions=actions,
        reputation=reputation,
        peer_ids=peer_ids,
        revision_mask=np.ones(len(actions), dtype=bool),
        rng=np.random.default_rng(3_000_004),
        params=ReputationParams(noise=1.0),
    )

    assert np.all((0.0 <= cooperation_probs) & (cooperation_probs <= 1.0))


def test_reputation_observation_features_append_self_and_peer_mean() -> None:
    reputation = np.array([0.2, 0.8, 0.5], dtype=np.float64)
    peer_ids = [[1, 2], [0], []]

    features = reputation_observation_features(
        reputation=reputation,
        peer_ids=peer_ids,
        mode="self_neighbor_mean",
    )

    assert reputation_observation_extra_dim("none") == 0
    assert reputation_observation_extra_dim("self_neighbor_mean") == 2
    np.testing.assert_allclose(
        features,
        np.asarray(
            [
                [0.2, 0.65],
                [0.8, 0.2],
                [0.5, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_local_quality_mobility_swaps_all_named_state_channels() -> None:
    actions = np.array([0, 1], dtype=np.int64)
    reputation = np.array([0.1, 0.9], dtype=np.float64)
    payoff_ema = np.array([0.0, 5.0], dtype=np.float64)
    agents: list[object] = ["agent-a", "agent-b"]

    result = apply_local_quality_mobility(
        state_arrays={
            "actions": actions,
            "reputation": reputation,
            "payoff_ema": payoff_ema,
        },
        quality_signal=payoff_ema,
        neighbors=[[], []],
        rng=np.random.default_rng(13),
        params=MobilityParams(enabled=True, rate=1.0, candidate_pool_size=1),
        state_lists={"agents": agents},
    )

    assert result.moved.tolist() == [True, False]
    assert result.targets.tolist() == [1, -1]
    assert result.gains[0] == pytest.approx(5.0)
    assert actions.tolist() == [1, 0]
    assert reputation.tolist() == [0.9, 0.1]
    assert payoff_ema.tolist() == [5.0, 0.0]
    assert agents == ["agent-b", "agent-a"]


def test_local_quality_averages_self_and_neighbors() -> None:
    values = np.array([1.0, 3.0, 5.0], dtype=np.float64)

    quality = local_quality(quality_signal=values, neighbors=[[1], [0, 2], []])

    assert quality.tolist() == pytest.approx([2.0, 3.0, 5.0])
