from __future__ import annotations

import numpy as np
import pytest
import torch

from neural_abm.core import ClassificationMLP, NeuralClassificationAgent, clone_state_dict
from neural_abm.mixers import (
    align_hidden_layer_state,
    apply_bounded_scalar_output_average,
    apply_distribution_output_average,
    apply_parameter_aligned_average,
    apply_parameter_average,
    apply_scalar_output_average,
    output_similarity_matrix,
    select_peers,
)
from neural_abm.mobility import MobilityStepResult
from neural_abm.social import (
    BOUNDED_SCALAR_CHANNEL,
    PeerIndexCache,
    PROBABILITY_DISTRIBUTION_CHANNEL,
    SCALAR_PROBABILITY_CHANNEL,
    STATE_DICT_CHANNEL,
    TENSOR_CHANNEL,
    SocialBlock,
    SocialChannel,
    bounded_scalar_similarity_matrix,
    distribution_output_similarity_matrix,
    empty_peers,
    mix_bounded_scalars,
    mix_probability_distributions,
    mix_scalar_probabilities,
    mix_state_dict_channel,
    mix_tensor_channel,
    peer_ids_for_mixer,
    scalar_output_similarity_matrix,
    select_bounded_scalar_output_peers,
    select_distribution_output_peers,
    select_scalar_output_peers,
    uniform_peer_count,
    validate_bounded_scalar_vector,
    validate_peer_ids,
    validate_probability_distributions,
    validate_probability_matrix,
    validate_probability_tensor,
    validate_probability_vector,
    validate_state_dicts,
)
from neural_abm.spatial_binary import BinaryPolicyStepResult
from neural_abm.toy_contagion import (
    apply_output_average_to_adoption_probs,
    select_peers_by_output_similarity as select_toy5_peers,
)
from neural_abm.toy_pd import apply_output_average_to_cooperation_probs
from neural_abm.toy_opinion import apply_output_average_to_acceptance_probs
from neural_abm.toy_public_goods import (
    apply_output_average_to_contribution_probs,
    select_peers_by_output_similarity as select_toy4_peers,
)


def test_empty_peers_for_none_mixer() -> None:
    peers = [[1], [0, 2], [1]]

    assert empty_peers(3) == [[], [], []]
    assert peer_ids_for_mixer(peers, mixer="none", agent_count=3) == [[], [], []]
    assert (
        peer_ids_for_mixer(
            peers,
            mixer="output_average",
            agent_count=3,
            copy_peers=False,
        )
        is peers
    )
    assert (
        peer_ids_for_mixer(
            [[3], [0]],
            mixer="output_average",
            agent_count=2,
            copy_peers=False,
            validate_peers=False,
        )
        == [[3], [0]]
    )


def test_invalid_peer_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid peer id"):
        validate_peer_ids([[1], [0, 3], [1]], agent_count=3)

    with pytest.raises(ValueError, match="must equal agent_count"):
        validate_peer_ids([[1], [0]], agent_count=3)


def test_uniform_peer_count_detects_fixed_degree_peer_lists() -> None:
    assert uniform_peer_count([[1, 2], [0, 2], [0, 1]]) == 2
    assert uniform_peer_count([[], [], []]) == 0
    assert uniform_peer_count([[1, 2], [0], []]) is None


def test_social_channel_contract_rejects_invalid_channels() -> None:
    SocialChannel(
        name="cooperation_probability",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="scalar_probability_sample",
    )

    with pytest.raises(ValueError, match="Unsupported social channel kind"):
        SocialChannel(name="bad", kind="unknown", commit_mode="sample")

    with pytest.raises(ValueError, match="align_state"):
        SocialChannel(
            name="hidden",
            kind=TENSOR_CHANNEL,
            commit_mode="distillation_step",
            align_state=lambda candidate, reference: candidate,
        )


def test_probability_vector_bounds_are_checked() -> None:
    validate_probability_vector(np.asarray([0.0, 0.4, 1.0]))

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_probability_vector(np.asarray([0.2, 1.2]))

    with pytest.raises(ValueError, match="finite"):
        validate_probability_vector(np.asarray([0.2, np.nan]))


def test_probability_matrix_requires_row_stochastic_values() -> None:
    validate_probability_matrix(np.asarray([[0.2, 0.8], [1.0, 0.0]]))

    with pytest.raises(ValueError, match="rows must sum to 1"):
        validate_probability_matrix(np.asarray([[0.2, 0.7], [1.0, 0.0]]))


def test_probability_distribution_array_and_tensor_contract() -> None:
    values = np.asarray(
        [
            [[0.2, 0.8], [0.5, 0.5]],
            [[0.9, 0.1], [0.4, 0.6]],
        ]
    )

    validate_probability_distributions(values)
    validate_probability_tensor(torch.as_tensor(values, dtype=torch.float32))

    with pytest.raises(ValueError, match="final axis"):
        validate_probability_distributions(values * 0.5)


def test_state_dict_contract_rejects_incompatible_states() -> None:
    validate_state_dicts(
        [
            {"w": torch.ones(2, 2), "b": torch.zeros(2)},
            {"w": torch.zeros(2, 2), "b": torch.ones(2)},
        ]
    )

    with pytest.raises(ValueError, match="same keys"):
        validate_state_dicts([{"w": torch.ones(2)}, {"v": torch.ones(2)}])

    with pytest.raises(ValueError, match="floating point"):
        validate_state_dicts([{"w": torch.ones(2, dtype=torch.int64)}])


def test_scalar_output_similarity_peer_selection_threshold() -> None:
    neighbors = [[1, 2], [0, 2], [0, 1]]
    values = np.asarray([0.1, 0.2, 0.9])

    similarity = scalar_output_similarity_matrix(values)
    selection = select_scalar_output_peers(
        neighbors=neighbors,
        values=values,
        peer_rule="output_similarity",
        threshold=0.8,
    )

    np.testing.assert_allclose(
        similarity,
        np.asarray(
            [
                [1.0, 0.9, 0.2],
                [0.9, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ]
        ),
    )
    assert selection.peer_ids == [[1], [0], []]
    assert selection.peer_counts == [1, 1, 0]


def test_scalar_output_peer_selection_can_reuse_none_rule_neighbors() -> None:
    neighbors = [[1, 2], [0, 2], [0, 1]]
    values = np.asarray([0.1, 0.2, 0.9])

    result = select_scalar_output_peers(
        neighbors=neighbors,
        values=values,
        peer_rule="none",
        threshold=0.0,
        copy_peers=False,
    )

    assert result.peer_ids is neighbors
    assert result.similarity is None

    unvalidated = select_scalar_output_peers(
        neighbors=[[3], [0]],
        values=np.asarray([0.1, 0.2]),
        peer_rule="none",
        threshold=0.0,
        copy_peers=False,
        validate_peers=False,
    )
    assert unvalidated.peer_ids == [[3], [0]]


def test_distribution_output_similarity_peer_selection_matches_toy1_wrapper() -> None:
    neighbors = [[1, 2], [0, 2], [0, 1]]
    probe_probs = np.asarray(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.85, 0.15], [0.75, 0.25]],
            [[0.1, 0.9], [0.2, 0.8]],
        ],
        dtype=np.float64,
    )
    threshold = 0.9

    expected = select_distribution_output_peers(
        neighbors=neighbors,
        probe_probs=probe_probs,
        peer_rule="output_similarity",
        threshold=threshold,
    )
    state_vectors = torch.eye(3)
    latent_vectors = torch.eye(3)
    selected, similarity = select_peers(
        graph_neighbors=neighbors,
        peer_rule="output_similarity",
        threshold=threshold,
        state_vectors=state_vectors,
        latent_vectors=latent_vectors,
        probe_probs=probe_probs,
    )

    assert selected == expected.peer_ids
    np.testing.assert_allclose(
        output_similarity_matrix(probe_probs),
        distribution_output_similarity_matrix(probe_probs),
    )
    assert similarity is not None
    np.testing.assert_allclose(similarity, expected.similarity)


def test_scalar_probability_mix_matches_formula() -> None:
    values = np.asarray([0.1, 0.9, 0.4])
    peers = [[1, 2], [0, 2], []]

    result = mix_scalar_probabilities(values, peers, alpha=0.25)

    assert result.mixed_values.tolist() == pytest.approx(
        [
            0.1 * 0.75 + ((0.9 + 0.4) / 2.0) * 0.25,
            0.9 * 0.75 + ((0.1 + 0.4) / 2.0) * 0.25,
            0.4,
        ]
    )
    assert result.losses == pytest.approx(np.abs(result.mixed_values - values))
    assert result.update_norms == pytest.approx(result.losses)
    assert result.channel == "scalar_probability"


def test_binary_policy_step_result_revision_rate_default_only_computes_when_missing() -> None:
    revision_mask = np.asarray([True, False, True])
    result = BinaryPolicyStepResult(
        pre_revision_probs=np.zeros(3),
        post_local_probs=np.zeros(3),
        post_social_probs=np.zeros(3),
        local_losses=[],
        social_losses=[],
        peer_ids=[[], [], []],
        revision_mask=revision_mask,
        mobility_result=MobilityStepResult.none(3),
    )
    explicit_zero = BinaryPolicyStepResult(
        pre_revision_probs=np.zeros(3),
        post_local_probs=np.zeros(3),
        post_social_probs=np.zeros(3),
        local_losses=[],
        social_losses=[],
        peer_ids=[[], [], []],
        revision_mask=revision_mask,
        mobility_result=MobilityStepResult.none(3),
        realized_revision_rate=0.0,
    )

    assert result.realized_revision_rate == pytest.approx(2.0 / 3.0)
    assert explicit_zero.realized_revision_rate == 0.0


def test_probability_distribution_mix_matches_formula() -> None:
    values = torch.as_tensor(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.9], [0.2, 0.8]],
            [[0.5, 0.5], [0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    peers = [[1, 2], [0], []]

    result = mix_probability_distributions(values, peers, alpha=0.25)

    expected_0 = 0.75 * values[0] + 0.25 * values[[1, 2]].mean(dim=0)
    expected_1 = 0.75 * values[1] + 0.25 * values[0]
    expected_2 = values[2]
    expected = torch.stack([expected_0, expected_1, expected_2], dim=0)
    expected = expected / expected.sum(dim=-1, keepdim=True)

    assert torch.allclose(result.mixed_values, expected)
    assert result.losses[0] > 0.0
    assert result.losses[1] > 0.0
    assert result.losses[2] == 0.0
    assert result.channel == "output_distribution"
    assert result.commit_mode == "distillation_step"
    assert result.active_agent_ids == [0, 1]
    assert result.peer_ids == peers
    assert result.peer_ids is not peers

    no_copy_result = mix_probability_distributions(
        values,
        peers,
        alpha=0.25,
        copy_peers=False,
    )
    assert no_copy_result.peer_ids is peers
    assert no_copy_result.active_agent_ids == [0, 1]


def test_probability_distribution_mix_uniform_peer_count_matches_formula() -> None:
    values = torch.as_tensor(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.9], [0.2, 0.8]],
            [[0.5, 0.5], [0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    peers = [[1, 2], [0, 2], [0, 1]]

    result = mix_probability_distributions(
        values,
        peers,
        alpha=0.25,
        uniform_peer_count=2,
    )
    indexed_result = mix_probability_distributions(
        values,
        peers,
        alpha=0.25,
        uniform_peer_index=torch.as_tensor(peers, dtype=torch.long),
        validate_peers=False,
    )

    expected_values = []
    for agent_id, agent_peers in enumerate(peers):
        peer_mean = values[agent_peers].mean(dim=0)
        mixed = 0.75 * values[agent_id] + 0.25 * peer_mean
        expected_values.append(mixed / mixed.sum(dim=-1, keepdim=True))
    expected = torch.stack(expected_values, dim=0)

    assert torch.allclose(result.mixed_values, expected)
    assert torch.allclose(indexed_result.mixed_values, expected)
    assert result.active_agent_ids == [0, 1, 2]
    assert indexed_result.active_agent_ids == [0, 1, 2]
    assert all(loss > 0.0 for loss in result.losses)

    with pytest.raises(ValueError, match="uniform_peer_count"):
        mix_probability_distributions(
            values,
            [[1, 2], [0], [0, 1]],
            0.25,
            uniform_peer_count=2,
        )
    with pytest.raises(ValueError, match="uniform_peer_index"):
        mix_probability_distributions(
            values,
            peers,
            0.25,
            uniform_peer_index=torch.as_tensor([1, 2, 0], dtype=torch.long),
        )


def test_probability_distribution_mix_matches_loop_reference() -> None:
    torch.manual_seed(14)
    values = torch.softmax(torch.randn(11, 3, 4), dim=-1)
    peers = [
        [1, 2, 3],
        [],
        [0, 4],
        [2],
        [1, 5, 6],
        [4],
        [7, 8],
        [6],
        [9, 10],
        [8],
        [],
    ]
    alpha = 0.35

    result = mix_probability_distributions(values, peers, alpha=alpha)
    cached_result = mix_probability_distributions(
        values,
        peers,
        alpha=alpha,
        peer_index_cache=PeerIndexCache.from_peer_ids(peers, device=values.device),
        validate_peers=False,
    )
    expected_values = []
    expected_norms = []
    original = values.detach()
    for agent_id, agent_peers in enumerate(peers):
        if not agent_peers:
            mixed = original[agent_id].clone()
            update_norm = 0.0
        else:
            peer_mean = original[agent_peers].mean(dim=0)
            mixed = (1.0 - alpha) * original[agent_id] + alpha * peer_mean
            mixed = mixed / mixed.sum(dim=-1, keepdim=True)
            update_norm = float(torch.linalg.vector_norm(mixed - original[agent_id]))
        expected_values.append(mixed)
        expected_norms.append(update_norm)

    assert torch.allclose(result.mixed_values, torch.stack(expected_values), atol=1e-6)
    assert torch.allclose(
        cached_result.mixed_values,
        torch.stack(expected_values),
        atol=1e-6,
    )
    assert result.losses == pytest.approx(expected_norms)
    assert cached_result.losses == pytest.approx(expected_norms)
    assert result.update_norms == pytest.approx(expected_norms)
    assert cached_result.update_norms == pytest.approx(expected_norms)
    assert result.active_agent_ids == [0, 2, 3, 4, 5, 6, 7, 8, 9]
    assert cached_result.active_agent_ids == [0, 2, 3, 4, 5, 6, 7, 8, 9]

    cache = PeerIndexCache.from_peer_ids(peers, device=values.device)
    with pytest.raises(ValueError, match="agent_count"):
        mix_probability_distributions(
            values,
            peers,
            alpha=alpha,
            peer_index_cache=PeerIndexCache(
                agent_count=cache.agent_count + 1,
                target_index=cache.target_index,
                peer_index=cache.peer_index,
                counts=cache.counts,
                active_mask=cache.active_mask,
                active_agent_ids=cache.active_agent_ids,
            ),
        )
    mismatched_peers = [[2, 1, 3], *peers[1:]]
    with pytest.raises(ValueError, match="does not match"):
        mix_probability_distributions(
            values,
            peers,
            alpha=alpha,
            peer_index_cache=PeerIndexCache.from_peer_ids(
                mismatched_peers,
                device=values.device,
            ),
        )


def test_social_block_probability_distribution_dispatch_matches_helper() -> None:
    values = torch.as_tensor(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.9], [0.2, 0.8]],
            [[0.5, 0.5], [0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    peers = [[1, 2], [0], []]
    channel = SocialChannel(
        name="probe_output_distribution",
        kind=PROBABILITY_DISTRIBUTION_CHANNEL,
        commit_mode="distillation_step",
    )

    expected = mix_probability_distributions(
        values,
        peers,
        alpha=0.25,
        channel=channel.name,
        commit_mode=channel.commit_mode,
    )
    result = SocialBlock(alpha=0.25).mix(channel, values, peers)

    assert torch.allclose(result.mixed_values, expected.mixed_values)
    assert result.losses == pytest.approx(expected.losses)
    assert result.channel == "probe_output_distribution"
    assert result.active_agent_ids == expected.active_agent_ids


def test_distribution_output_average_unit_helper_matches_common_block() -> None:
    values = torch.as_tensor(
        [
            [[0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.9], [0.2, 0.8]],
            [[0.5, 0.5], [0.4, 0.6]],
        ],
        dtype=torch.float32,
    )
    peers = [[1, 2], [0], []]

    expected = mix_probability_distributions(
        values,
        peers,
        alpha=0.25,
        channel="strategy_distribution",
        commit_mode="categorical_probability_commit",
    )
    result = apply_distribution_output_average(
        values,
        peers,
        alpha=0.25,
        channel="strategy_distribution",
        commit_mode="categorical_probability_commit",
    )

    assert torch.allclose(result.mix.mixed_values, expected.mixed_values)
    assert result.commit.losses == pytest.approx(expected.losses)
    assert result.mix.update_norms == pytest.approx(expected.update_norms)
    assert result.diagnostics.aggregate_row()["social_channel"] == (
        "strategy_distribution"
    )


def test_probability_distribution_mix_can_skip_update_norm_collection() -> None:
    values = torch.as_tensor(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    peers = [[1, 2], [0], []]

    expected = mix_probability_distributions(values, peers, alpha=0.25)
    result = mix_probability_distributions(
        values,
        peers,
        alpha=0.25,
        collect_update_norms=False,
    )

    assert torch.allclose(result.mixed_values, expected.mixed_values)
    assert result.losses == []
    assert result.update_norms == []
    assert result.active_agent_ids == expected.active_agent_ids


def test_tensor_channel_mix_matches_formula() -> None:
    values = torch.as_tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    peers = [[1, 2], [0], []]

    result = mix_tensor_channel(
        values,
        peers,
        alpha=0.5,
        channel="probe_hidden_activation",
        commit_mode="distillation_step",
    )

    expected = torch.stack(
        [
            0.5 * values[0] + 0.5 * values[[1, 2]].mean(dim=0),
            0.5 * values[1] + 0.5 * values[0],
            values[2],
        ],
        dim=0,
    )

    assert torch.allclose(result.mixed_values, expected)
    assert result.losses[0] > 0.0
    assert result.losses[1] > 0.0
    assert result.losses[2] == 0.0
    assert result.channel == "probe_hidden_activation"
    assert result.commit_mode == "distillation_step"


def test_state_dict_channel_mix_matches_formula() -> None:
    states = [
        {"w": torch.as_tensor([1.0, 3.0]), "b": torch.as_tensor([0.0])},
        {"w": torch.as_tensor([5.0, 7.0]), "b": torch.as_tensor([2.0])},
        {"w": torch.as_tensor([2.0, 4.0]), "b": torch.as_tensor([6.0])},
    ]
    peers = [[1, 2], [0], []]

    result = mix_state_dict_channel(states, peers, alpha=0.25)
    mixed_states = result.mixed_values

    assert torch.allclose(
        mixed_states[0]["w"],
        0.75 * states[0]["w"] + 0.25 * torch.stack([states[1]["w"], states[2]["w"]]).mean(dim=0),
    )
    assert torch.allclose(
        mixed_states[1]["b"],
        0.75 * states[1]["b"] + 0.25 * states[0]["b"],
    )
    assert torch.allclose(mixed_states[2]["w"], states[2]["w"])
    assert mixed_states[2]["w"] is not states[2]["w"]
    assert result.losses == pytest.approx(result.update_norms)
    assert result.channel == "parameter_state"
    assert result.commit_mode == "state_dict_load"


def test_state_dict_channel_uses_alignment_callback() -> None:
    states = [
        {"w": torch.as_tensor([0.0, 0.0])},
        {"w": torch.as_tensor([-2.0, -4.0])},
    ]

    def flip_sign_alignment(
        candidate_state: dict[str, torch.Tensor],
        reference_state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        del reference_state
        return {"w": -candidate_state["w"]}

    result = mix_state_dict_channel(
        states,
        peer_ids=[[1], []],
        alpha=0.5,
        align_state=flip_sign_alignment,
        channel="aligned_parameter_state",
    )

    assert torch.allclose(result.mixed_values[0]["w"], torch.as_tensor([1.0, 2.0]))
    assert result.mixed_values[1]["w"].tolist() == pytest.approx([-2.0, -4.0])
    assert result.channel == "aligned_parameter_state"


def test_scalar_probability_alpha_zero_is_noop_copy() -> None:
    values = np.asarray([0.1, 0.9, 0.4])
    peers = [[1, 2], [0, 2], [0, 1]]

    result = mix_scalar_probabilities(values, peers, alpha=0.0)

    assert result.mixed_values is not values
    assert result.mixed_values.tolist() == pytest.approx(values.tolist())
    assert result.losses == [0.0, 0.0, 0.0]


def test_scalar_probability_mix_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        mix_scalar_probabilities(np.asarray([0.1, 0.9]), [[1], [0]], alpha=1.5)


def test_bounded_scalar_vector_validation_uses_declared_bounds() -> None:
    validate_bounded_scalar_vector(
        np.asarray([2.0, 5.0, 8.0]),
        lower_bound=0.0,
        upper_bound=10.0,
    )

    with pytest.raises(ValueError, match="bounded scalar vector"):
        validate_bounded_scalar_vector(np.asarray([[0.1, 0.2]]))
    with pytest.raises(ValueError, match="finite"):
        validate_bounded_scalar_vector(np.asarray([0.1, np.nan]))
    with pytest.raises(ValueError, match=r"\[0, 10\]"):
        validate_bounded_scalar_vector(
            np.asarray([11.0]),
            lower_bound=0.0,
            upper_bound=10.0,
        )
    with pytest.raises(ValueError, match="lower_bound"):
        validate_bounded_scalar_vector(
            np.asarray([1.0]),
            lower_bound=2.0,
            upper_bound=1.0,
        )


def test_bounded_scalar_mix_uses_bounds_without_probability_semantics() -> None:
    values = np.asarray([2.0, 8.0, 4.0])
    peers = [[1, 2], [0], []]

    result = mix_bounded_scalars(
        values,
        peers,
        alpha=0.5,
        lower_bound=0.0,
        upper_bound=10.0,
        channel="extraction_intensity",
        commit_mode="continuous_intensity_commit",
    )

    assert result.mixed_values.tolist() == pytest.approx([4.0, 5.0, 4.0])
    assert result.losses == pytest.approx([2.0, 3.0, 0.0])
    assert result.update_norms == pytest.approx([2.0, 3.0, 0.0])
    assert result.channel == "extraction_intensity"
    assert result.commit_mode == "continuous_intensity_commit"


def test_bounded_scalar_output_similarity_respects_custom_span() -> None:
    values = np.asarray([0.0, 5.0, 10.0])

    matrix = bounded_scalar_similarity_matrix(
        values,
        lower_bound=0.0,
        upper_bound=10.0,
    )
    result = select_bounded_scalar_output_peers(
        neighbors=[[1, 2], [0, 2], [0, 1]],
        values=values,
        peer_rule="output_similarity",
        threshold=0.6,
        lower_bound=0.0,
        upper_bound=10.0,
    )

    np.testing.assert_allclose(
        matrix,
        np.asarray(
            [
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.5],
                [0.0, 0.5, 1.0],
            ]
        ),
    )
    assert result.peer_ids == [[], [], []]


def test_social_block_bounded_scalar_dispatch_matches_helper() -> None:
    values = np.asarray([2.0, 8.0, 4.0])
    peers = [[1, 2], [0], []]
    channel = SocialChannel(
        name="extraction_intensity",
        kind=BOUNDED_SCALAR_CHANNEL,
        commit_mode="continuous_intensity_commit",
        lower_bound=0.0,
        upper_bound=10.0,
    )

    expected = mix_bounded_scalars(
        values,
        peers,
        alpha=0.5,
        lower_bound=0.0,
        upper_bound=10.0,
        channel=channel.name,
        commit_mode=channel.commit_mode,
    )
    result = SocialBlock(alpha=0.5).mix(channel, values, peers)

    assert result.mixed_values.tolist() == pytest.approx(
        expected.mixed_values.tolist()
    )
    assert result.losses == pytest.approx(expected.losses)
    assert result.peer_ids == peers


def test_social_block_scalar_dispatch_matches_helper() -> None:
    values = np.asarray([0.1, 0.9, 0.4])
    peers = [[1, 2], [0, 2], []]
    channel = SocialChannel(
        name="cooperation_probability",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="scalar_probability_sample",
    )

    expected = mix_scalar_probabilities(
        values,
        peers,
        alpha=0.25,
        channel=channel.name,
        commit_mode=channel.commit_mode,
    )
    result = SocialBlock(alpha=0.25).mix(channel, values, peers)

    assert result.mixed_values.tolist() == pytest.approx(
        expected.mixed_values.tolist()
    )
    assert result.losses == pytest.approx(expected.losses)
    assert result.peer_ids == peers


def test_bounded_scalar_output_average_unit_helper_matches_common_block() -> None:
    values = np.asarray([2.0, 8.0, 4.0])
    peers = [[1, 2], [0], []]

    expected = mix_bounded_scalars(
        values,
        peers,
        alpha=0.5,
        lower_bound=0.0,
        upper_bound=10.0,
        channel="extraction_intensity",
        commit_mode="continuous_intensity_commit",
    )
    result = apply_bounded_scalar_output_average(
        values,
        peers,
        alpha=0.5,
        lower_bound=0.0,
        upper_bound=10.0,
        channel="extraction_intensity",
        commit_mode="continuous_intensity_commit",
    )

    assert result.mix.mixed_values.tolist() == pytest.approx(
        expected.mixed_values.tolist()
    )
    assert result.commit.losses == pytest.approx(expected.losses)
    assert result.mix.update_norms == pytest.approx(expected.update_norms)
    assert result.diagnostics.aggregate_row()["social_channel"] == (
        "extraction_intensity"
    )


def test_scalar_output_average_unit_helper_matches_common_block() -> None:
    values = np.asarray([0.1, 0.9, 0.4])
    peers = [[1, 2], [0, 2], []]

    expected = mix_scalar_probabilities(
        values,
        peers,
        alpha=0.25,
        channel="activation_propensity",
        commit_mode="event_hazard_commit",
    )
    result = apply_scalar_output_average(
        values,
        peers,
        alpha=0.25,
        channel="activation_propensity",
        commit_mode="event_hazard_commit",
    )

    assert result.mix.mixed_values.tolist() == pytest.approx(
        expected.mixed_values.tolist()
    )
    assert result.commit.losses == pytest.approx(expected.losses)
    assert result.mix.update_norms == pytest.approx(expected.update_norms)
    assert result.diagnostics.aggregate_row()["social_channel"] == (
        "activation_propensity"
    )


def test_toy2_scalar_probability_wrapper_matches_common_block() -> None:
    values = np.asarray([0.1, 0.9, 0.4])
    peers = [[1, 2], [0, 2], [0, 1]]

    expected = mix_scalar_probabilities(values, peers, alpha=0.25)
    mixed, losses = apply_output_average_to_cooperation_probs(values, peers, alpha=0.25)

    assert mixed.tolist() == pytest.approx(expected.mixed_values.tolist())
    assert losses == pytest.approx(expected.losses)


def test_toy4_scalar_probability_wrapper_matches_common_block() -> None:
    values = np.asarray([0.2, 0.7, 0.4])
    peers = [[1], [0, 2], [1]]

    expected = mix_scalar_probabilities(values, peers, alpha=0.5)
    mixed, losses = apply_output_average_to_contribution_probs(
        values,
        peers,
        alpha=0.5,
    )

    assert mixed.tolist() == pytest.approx(expected.mixed_values.tolist())
    assert losses == pytest.approx(expected.losses)


def test_toy5_scalar_probability_wrapper_matches_common_block() -> None:
    values = np.asarray([0.0, 0.5, 1.0])
    peers = [[1], [0, 2], [1]]

    expected = mix_scalar_probabilities(values, peers, alpha=0.3)
    mixed, losses = apply_output_average_to_adoption_probs(values, peers, alpha=0.3)

    assert mixed.tolist() == pytest.approx(expected.mixed_values.tolist())
    assert losses == pytest.approx(expected.losses)


def test_toy3_acceptance_probability_wrapper_matches_social_block() -> None:
    values = torch.as_tensor([0.1, 0.9, 0.4], dtype=torch.float32)
    peers = [[1, 2], [0], []]
    channel = SocialChannel(
        name="acceptance_probability",
        kind=SCALAR_PROBABILITY_CHANNEL,
        commit_mode="tensor_probability",
    )

    expected = SocialBlock(alpha=0.25).mix(
        channel,
        values.detach().cpu().numpy(),
        peers,
    )
    mixed = apply_output_average_to_acceptance_probs(values, peers, alpha=0.25)

    assert torch.allclose(
        mixed,
        torch.as_tensor(expected.mixed_values, dtype=values.dtype),
    )


def test_toy4_toy5_peer_selection_wrappers_match_common_block() -> None:
    neighbors = [[1, 2], [0, 2], [0, 1]]
    values = np.asarray([0.15, 0.25, 0.95])
    expected = select_scalar_output_peers(
        neighbors=neighbors,
        values=values,
        peer_rule="output_similarity",
        threshold=0.85,
    ).peer_ids

    assert select_toy4_peers(neighbors, values, "output_similarity", 0.85) == expected
    assert select_toy5_peers(neighbors, values, "output_similarity", 0.85) == expected


def make_classification_agent(agent_id: int) -> NeuralClassificationAgent:
    model = ClassificationMLP(input_dim=2, hidden_dim=4, output_dim=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return NeuralClassificationAgent(
        agent_id=agent_id,
        shard_group=f"group_{agent_id}",
        model=model,
        optimizer=optimizer,
        train_x=torch.zeros((4, 2), dtype=torch.float32),
        train_y=torch.zeros(4, dtype=torch.long),
    )


def test_toy1_parameter_average_wrapper_matches_common_block() -> None:
    torch.manual_seed(101)
    agents = [make_classification_agent(agent_id) for agent_id in range(3)]
    previous_states = [clone_state_dict(agent.model) for agent in agents]
    peer_ids = [[1, 2], [0], [0, 1]]

    expected = mix_state_dict_channel(
        previous_states,
        peer_ids,
        alpha=0.3,
        channel="parameter_state",
        commit_mode="state_dict_load",
    ).mixed_values
    block_expected = SocialBlock(alpha=0.3).mix(
        SocialChannel(
            name="parameter_state",
            kind=STATE_DICT_CHANNEL,
            commit_mode="state_dict_load",
        ),
        previous_states,
        peer_ids,
    ).mixed_values
    apply_parameter_average(
        agents=agents,
        peer_ids=peer_ids,
        alpha=0.3,
        previous_states=previous_states,
    )

    for agent_id, agent in enumerate(agents):
        for key, value in agent.model.state_dict().items():
            assert torch.allclose(block_expected[agent_id][key], expected[agent_id][key])
            assert torch.allclose(value, expected[agent_id][key])


def test_toy1_parameter_aligned_average_wrapper_matches_common_block() -> None:
    torch.manual_seed(103)
    agents = [make_classification_agent(agent_id) for agent_id in range(2)]
    previous_states = [clone_state_dict(agent.model) for agent in agents]
    peer_ids = [[1], [0]]

    expected = mix_state_dict_channel(
        previous_states,
        peer_ids,
        alpha=0.4,
        align_state=align_hidden_layer_state,
        channel="aligned_parameter_state",
        commit_mode="state_dict_load",
    ).mixed_values
    apply_parameter_aligned_average(
        agents=agents,
        peer_ids=peer_ids,
        alpha=0.4,
        previous_states=previous_states,
    )

    for agent_id, agent in enumerate(agents):
        for key, value in agent.model.state_dict().items():
            assert torch.allclose(value, expected[agent_id][key])
