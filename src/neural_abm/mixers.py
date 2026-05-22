"""Peer selection and social mixing operations."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from neural_abm.core import NeuralClassificationAgent
from neural_abm.social import (
    PROBABILITY_DISTRIBUTION_CHANNEL,
    SCALAR_PROBABILITY_CHANNEL,
    STATE_DICT_CHANNEL,
    TENSOR_CHANNEL,
    SocialBlock,
    SocialChannel,
    distribution_output_similarity_matrix,
)
from neural_abm.unit import (
    DistributionDistillationAdapter,
    NABMUnit,
    NABMStep,
    NABMStepResult,
    StateDictLoadAdapter,
    TensorDistillationAdapter,
)


def cosine_similarity_matrix(vectors: torch.Tensor) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D tensor")
    normalized = torch.nn.functional.normalize(vectors, dim=1, eps=1e-8)
    return (normalized @ normalized.T).detach().cpu().numpy()


def output_similarity_matrix(probe_probs: np.ndarray) -> np.ndarray:
    return distribution_output_similarity_matrix(probe_probs)


def hidden_unit_signatures(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """Build one signature vector per hidden unit for the Toy 1 MLP."""

    required = {"fc1.weight", "fc1.bias", "fc2.weight"}
    missing = required - set(state)
    if missing:
        raise ValueError(f"State dict is missing hidden alignment keys: {missing}")

    fc1_weight = state["fc1.weight"].detach()
    fc1_bias = state["fc1.bias"].detach().unsqueeze(1)
    fc2_outgoing = state["fc2.weight"].detach().T
    return torch.cat([fc1_weight, fc1_bias, fc2_outgoing], dim=1)


def hidden_alignment_permutation(
    reference_state: dict[str, torch.Tensor],
    candidate_state: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return candidate hidden-unit indices ordered to match the reference."""

    reference = nn.functional.normalize(
        hidden_unit_signatures(reference_state), dim=1, eps=1e-8
    )
    candidate = nn.functional.normalize(
        hidden_unit_signatures(candidate_state), dim=1, eps=1e-8
    )
    if reference.shape != candidate.shape:
        raise ValueError(
            "Reference and candidate hidden signatures must have matching shapes"
        )

    scores = (reference @ candidate.T).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(-scores)
    permutation = np.empty(len(row_ind), dtype=np.int64)
    permutation[row_ind] = col_ind
    return torch.as_tensor(
        permutation,
        dtype=torch.long,
        device=candidate_state["fc1.weight"].device,
    )


def align_hidden_layer_state(
    candidate_state: dict[str, torch.Tensor],
    reference_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Permute a Toy 1 peer state into the reference hidden-unit basis."""

    permutation = hidden_alignment_permutation(
        reference_state=reference_state,
        candidate_state=candidate_state,
    )
    aligned = {key: value.detach().clone() for key, value in candidate_state.items()}
    aligned["fc1.weight"] = candidate_state["fc1.weight"][permutation].detach().clone()
    aligned["fc1.bias"] = candidate_state["fc1.bias"][permutation].detach().clone()
    aligned["fc2.weight"] = (
        candidate_state["fc2.weight"][:, permutation].detach().clone()
    )
    return aligned


def flatten_state(state: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.detach().flatten() for value in state.values()])


def aligned_state_similarity_matrix(
    parameter_states: list[dict[str, torch.Tensor]],
) -> np.ndarray:
    """Compute directional cosine similarities after hidden-unit alignment."""

    count = len(parameter_states)
    matrix = np.eye(count, dtype=np.float64)
    for i in range(count):
        reference = parameter_states[i]
        reference_vector = flatten_state(reference)
        for j in range(count):
            if i == j:
                continue
            aligned_candidate = align_hidden_layer_state(
                candidate_state=parameter_states[j],
                reference_state=reference,
            )
            candidate_vector = flatten_state(aligned_candidate)
            similarity = nn.functional.cosine_similarity(
                reference_vector,
                candidate_vector,
                dim=0,
                eps=1e-8,
            )
            matrix[i, j] = float(similarity.detach().cpu())
    return matrix


def select_peers(
    graph_neighbors: list[list[int]],
    peer_rule: str,
    threshold: float,
    state_vectors: torch.Tensor,
    latent_vectors: torch.Tensor,
    probe_probs: np.ndarray,
    parameter_states: list[dict[str, torch.Tensor]] | None = None,
) -> tuple[list[list[int]], np.ndarray | None]:
    """Select peers from graph neighbors using the configured rule."""

    if peer_rule == "none":
        return [list(neighbors) for neighbors in graph_neighbors], None

    if peer_rule == "state_similarity":
        sim = cosine_similarity_matrix(state_vectors)
    elif peer_rule == "aligned_state_similarity":
        if parameter_states is None:
            raise ValueError("aligned_state_similarity requires parameter_states")
        sim = aligned_state_similarity_matrix(parameter_states)
    elif peer_rule == "latent_similarity":
        sim = cosine_similarity_matrix(latent_vectors)
    elif peer_rule == "output_similarity":
        result = SocialBlock(alpha=0.0).select_distribution_output_peers(
            neighbors=graph_neighbors,
            probe_probs=probe_probs,
            peer_rule=peer_rule,
            threshold=threshold,
        )
        return result.peer_ids, result.similarity
    else:
        raise ValueError(f"Unsupported peer rule: {peer_rule}")

    peer_ids: list[list[int]] = []
    for i, neighbors in enumerate(graph_neighbors):
        peers = [j for j in neighbors if sim[i, j] >= threshold]
        peer_ids.append(peers)
    return peer_ids, sim


def apply_scalar_output_average(
    values: np.ndarray,
    peer_ids: list[list[int]],
    alpha: float,
    *,
    channel: str = "scalar_probability",
    commit_mode: str = "scalar_probability_sample",
) -> NABMStepResult:
    """Apply a NABMStep-backed scalar probability social mix."""

    step = NABMStep(
        social_block=SocialBlock(alpha=alpha),
        channel=SocialChannel(
            name=channel,
            kind=SCALAR_PROBABILITY_CHANNEL,
            commit_mode=commit_mode,
        ),
    )
    return step.run(
        values=np.asarray(values, dtype=np.float64),
        peer_ids=peer_ids,
    )


def apply_parameter_average(
    agents: list[NeuralClassificationAgent],
    peer_ids: list[list[int]],
    alpha: float,
    previous_states: list[dict[str, torch.Tensor]],
) -> NABMStepResult:
    """Apply synchronous parameter averaging."""

    step = NABMStep(
        social_block=SocialBlock(alpha=alpha),
        channel=SocialChannel(
            name="parameter_state",
            kind=STATE_DICT_CHANNEL,
            commit_mode="state_dict_load",
        ),
        commit_adapter=StateDictLoadAdapter(agents=agents),
    )
    return step.run(values=previous_states, peer_ids=peer_ids)


def apply_parameter_aligned_average(
    agents: list[NeuralClassificationAgent],
    peer_ids: list[list[int]],
    alpha: float,
    previous_states: list[dict[str, torch.Tensor]],
) -> NABMStepResult:
    """Apply synchronous parameter averaging after hidden-unit alignment."""

    step = NABMStep(
        social_block=SocialBlock(alpha=alpha),
        channel=SocialChannel(
            name="aligned_parameter_state",
            kind=STATE_DICT_CHANNEL,
            commit_mode="state_dict_load",
            align_state=align_hidden_layer_state,
        ),
        commit_adapter=StateDictLoadAdapter(agents=agents),
    )
    return step.run(values=previous_states, peer_ids=peer_ids)


def apply_output_average(
    agents: list[NeuralClassificationAgent],
    peer_ids: list[list[int]],
    alpha: float,
    probe_x: torch.Tensor,
    previous_probs: torch.Tensor,
) -> NABMStepResult:
    """Distill each agent toward mixed peer probe predictions."""

    step = NABMStep(
        social_block=SocialBlock(alpha=alpha),
        channel=SocialChannel(
            name="probe_output_distribution",
            kind=PROBABILITY_DISTRIBUTION_CHANNEL,
            commit_mode="distillation_step",
        ),
        commit_adapter=DistributionDistillationAdapter(
            agents=agents,
            logits_fn=lambda agent, _agent_id: agent.model(probe_x),
            optimizer_fn=lambda agent, _agent_id: agent.optimizer,
            loss_mode="kl",
        ),
    )
    unit = NABMUnit(
        agents=agents,
        step=step,
        peer_selector=lambda _messages: peer_ids,
        social_value_builder=lambda _agents, _messages: previous_probs,
    )
    return unit.run(
        message_args=(probe_x,),
        run_local_update=False,
        collect_logs=False,
    ).social_step


def apply_latent_average(
    agents: list[NeuralClassificationAgent],
    peer_ids: list[list[int]],
    alpha: float,
    probe_x: torch.Tensor,
    previous_hidden: torch.Tensor,
) -> NABMStepResult:
    """Align hidden probe activations toward peer hidden summaries."""

    step = NABMStep(
        social_block=SocialBlock(alpha=alpha),
        channel=SocialChannel(
            name="probe_hidden_activation",
            kind=TENSOR_CHANNEL,
            commit_mode="distillation_step",
        ),
        commit_adapter=TensorDistillationAdapter(
            agents=agents,
            tensor_fn=lambda agent, _agent_id: agent.hidden_on(probe_x),
            optimizer_fn=lambda agent, _agent_id: agent.optimizer,
        ),
    )
    return step.run(values=previous_hidden, peer_ids=peer_ids)
