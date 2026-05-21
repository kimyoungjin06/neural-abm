"""Capability metadata for the toy ABM family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NABMStatus = Literal["full", "compatible", "reference"]
NABM_ARTIFACT_FIELDS = (
    "nabm_status",
    "neural_role",
    "social_channels",
    "reference_policies",
)


@dataclass(frozen=True)
class CoordinationCapability:
    mixer: str
    peer_rules: tuple[str, ...]


@dataclass(frozen=True)
class ToyCapability:
    toy: str
    state_kind: str
    action_space: str
    runner_kind: str
    result_kind: str
    coordination: tuple[CoordinationCapability, ...]
    nabm_status: NABMStatus
    neural_role: str
    social_channels: tuple[str, ...]
    backends: tuple[str, ...] = ("loop",)
    reference_policies: tuple[str, ...] = ()


TOY_CAPABILITIES: dict[str, ToyCapability] = {
    "toy1": ToyCapability(
        toy="toy1",
        state_kind="classification_probe",
        action_space="distribution",
        runner_kind="classification",
        result_kind="domain",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("output_similarity",)),
            CoordinationCapability("latent_average", ("latent_similarity",)),
            CoordinationCapability("parameter_average", ("state_similarity",)),
            CoordinationCapability(
                "parameter_aligned_average",
                ("state_similarity", "aligned_state_similarity"),
            ),
        ),
        nabm_status="full",
        neural_role=(
            "Local classifier training plus social output, latent, and parameter "
            "mixing are the primary experiment path."
        ),
        social_channels=("output_distribution", "latent_state", "parameters"),
    ),
    "toy2": ToyCapability(
        toy="toy2",
        state_kind="binary_spatial",
        action_space="binary",
        runner_kind="binary_spatial",
        result_kind="binary",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("none", "output_similarity")),
        ),
        nabm_status="full",
        neural_role=(
            "Neural policy local learning and social policy distillation are the "
            "primary path."
        ),
        social_channels=("action_probability", "policy_distribution", "reputation"),
        backends=("loop", "batched", "tensor_batched", "auto"),
        reference_policies=(
            "rd_well_mixed",
            "fermi_imitation",
            "reputation_imitation",
        ),
    ),
    "toy3": ToyCapability(
        toy="toy3",
        state_kind="continuous_opinion",
        action_space="continuous",
        runner_kind="opinion",
        result_kind="domain",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability(
                "output_average",
                ("bounded_confidence", "output_similarity"),
            ),
        ),
        nabm_status="full",
        neural_role=(
            "Neural opinion updates and social output mixing drive the "
            "opinion/rewiring experiment path."
        ),
        social_channels=("opinion_output", "bounded_confidence", "peer_graph"),
    ),
    "toy4": ToyCapability(
        toy="toy4",
        state_kind="binary_public_goods",
        action_space="binary",
        runner_kind="binary_spatial",
        result_kind="binary",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("none", "output_similarity")),
        ),
        nabm_status="full",
        neural_role=(
            "Neural contribution policy learning and social distillation are the "
            "primary path."
        ),
        social_channels=(
            "action_probability",
            "policy_distribution",
            "reputation",
            "resource_state",
        ),
        backends=("loop", "batched", "tensor_batched", "auto"),
        reference_policies=("imitation", "reputation_imitation"),
    ),
    "toy5": ToyCapability(
        toy="toy5",
        state_kind="binary_contagion",
        action_space="binary",
        runner_kind="binary_spatial",
        result_kind="binary",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("none", "output_similarity")),
        ),
        nabm_status="full",
        neural_role=(
            "Neural adoption policy learning and social distillation are the "
            "primary path."
        ),
        social_channels=(
            "adoption_probability",
            "policy_distribution",
            "reputation",
            "exposure_state",
        ),
        backends=("loop", "batched", "tensor_batched", "auto"),
        reference_policies=(
            "simple_contagion",
            "complex_threshold",
            "reputation_imitation",
        ),
    ),
    "toy6": ToyCapability(
        toy="toy6",
        state_kind="categorical_spatial",
        action_space="categorical",
        runner_kind="categorical",
        result_kind="domain",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("none", "output_similarity")),
        ),
        nabm_status="compatible",
        neural_role=(
            "Uses the shared config, result, social, and sweep contracts; the "
            "neural path is limited relative to the full NABM toys."
        ),
        social_channels=("categorical_policy", "output_distribution"),
    ),
    "toy7": ToyCapability(
        toy="toy7",
        state_kind="continuous_resource",
        action_space="continuous_scalar",
        runner_kind="resource",
        result_kind="domain",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("none", "output_similarity")),
        ),
        nabm_status="compatible",
        neural_role=(
            "Uses the shared config, result, social, and sweep contracts; resource "
            "dynamics remain the dominant toy-specific mechanism."
        ),
        social_channels=("continuous_action", "resource_state"),
    ),
    "toy8": ToyCapability(
        toy="toy8",
        state_kind="async_event_state",
        action_space="event",
        runner_kind="async_event",
        result_kind="domain",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("none", "output_similarity")),
        ),
        nabm_status="compatible",
        neural_role=(
            "Uses the shared config, result, social, and sweep contracts; event "
            "scheduling remains toy-specific."
        ),
        social_channels=("event_hazard", "event_state"),
    ),
    "toy9": ToyCapability(
        toy="toy9",
        state_kind="heterogeneous_agent_state",
        action_space="binary",
        runner_kind="heterogeneous",
        result_kind="domain",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("none", "output_similarity")),
        ),
        nabm_status="compatible",
        neural_role=(
            "Uses the shared config, result, social, and sweep contracts; "
            "heterogeneous local rules remain toy-specific."
        ),
        social_channels=("binary_action_probability", "group_state"),
    ),
    "toy10": ToyCapability(
        toy="toy10",
        state_kind="dynamic_market_ecology",
        action_space="multi_channel_continuous",
        runner_kind="market_ecology",
        result_kind="domain",
        coordination=(
            CoordinationCapability("none", ("none",)),
            CoordinationCapability("output_average", ("none", "output_similarity")),
        ),
        nabm_status="compatible",
        neural_role=(
            "Uses the shared config, result, social, and sweep contracts; "
            "market/ecology feedback remains toy-specific."
        ),
        social_channels=("harvest_intensity", "price_signal", "conservation_signal"),
    ),
}


def supported_toys() -> tuple[str, ...]:
    return tuple(TOY_CAPABILITIES)


def toy_capability(toy: str) -> ToyCapability:
    try:
        return TOY_CAPABILITIES[toy]
    except KeyError as exc:
        raise KeyError(f"Unknown toy capability: {toy}") from exc


def supported_coordination_pairs(toy: str) -> tuple[tuple[str, str], ...]:
    capability = toy_capability(toy)
    return tuple(
        (coordination.mixer, peer_rule)
        for coordination in capability.coordination
        for peer_rule in coordination.peer_rules
    )


def supports_coordination(toy: str, mixer: str, peer_rule: str) -> bool:
    return (mixer, peer_rule) in supported_coordination_pairs(toy)


def nabm_status(toy: str) -> str:
    return toy_capability(toy).nabm_status


def artifact_capability_metadata(toy: str) -> dict[str, object]:
    capability = toy_capability(toy)
    return {
        "nabm_status": capability.nabm_status,
        "neural_role": capability.neural_role,
        "social_channels": list(capability.social_channels),
        "reference_policies": list(capability.reference_policies),
    }


def optional_artifact_capability_metadata(toy: str) -> dict[str, object]:
    try:
        return artifact_capability_metadata(toy)
    except KeyError:
        return {}


def sweep_capability_metadata(toy: str) -> dict[str, str]:
    capability = toy_capability(toy)
    return {
        "nabm_status": capability.nabm_status,
        "neural_role": capability.neural_role,
        "social_channels": _format_sweep_sequence(capability.social_channels),
        "reference_policies": _format_sweep_sequence(capability.reference_policies),
    }


def _format_sweep_sequence(values: tuple[str, ...]) -> str:
    return ", ".join(values) if values else "none"
