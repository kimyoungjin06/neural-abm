from __future__ import annotations

import torch

from neural_abm.core import ClassificationMLP, clone_state_dict
from neural_abm.mixers import align_hidden_layer_state, aligned_state_similarity_matrix


def permute_hidden_units(
    state: dict[str, torch.Tensor],
    permutation: torch.Tensor,
) -> dict[str, torch.Tensor]:
    permuted = {key: value.detach().clone() for key, value in state.items()}
    permuted["fc1.weight"] = state["fc1.weight"][permutation].detach().clone()
    permuted["fc1.bias"] = state["fc1.bias"][permutation].detach().clone()
    permuted["fc2.weight"] = state["fc2.weight"][:, permutation].detach().clone()
    return permuted


def test_align_hidden_layer_state_recovers_permuted_mlp() -> None:
    torch.manual_seed(23)
    model = ClassificationMLP(input_dim=2, hidden_dim=6, output_dim=2)
    reference = clone_state_dict(model)
    candidate = permute_hidden_units(reference, torch.tensor([2, 0, 5, 1, 4, 3]))

    aligned = align_hidden_layer_state(
        candidate_state=candidate,
        reference_state=reference,
    )

    for key, reference_value in reference.items():
        assert torch.allclose(aligned[key], reference_value)


def test_aligned_state_similarity_treats_permutation_as_equivalent() -> None:
    torch.manual_seed(29)
    model = ClassificationMLP(input_dim=2, hidden_dim=6, output_dim=2)
    reference = clone_state_dict(model)
    candidate = permute_hidden_units(reference, torch.tensor([4, 1, 0, 3, 5, 2]))

    similarity = aligned_state_similarity_matrix([reference, candidate])

    assert similarity.shape == (2, 2)
    assert similarity[0, 1] > 0.999
    assert similarity[1, 0] > 0.999
