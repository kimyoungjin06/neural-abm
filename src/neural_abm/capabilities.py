"""Capability metadata for the toy ABM family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NABMStatus = Literal["full", "compatible", "reference"]
ToyTaxonomyField = Literal[
    "domain_family",
    "state_family",
    "output_family",
    "topology_family",
    "coordination_family",
    "unit_surface",
    "evidence_role",
]
TOY_TAXONOMY_FIELDS: tuple[ToyTaxonomyField, ...] = (
    "domain_family",
    "state_family",
    "output_family",
    "topology_family",
    "coordination_family",
    "unit_surface",
    "evidence_role",
)
NABM_ARTIFACT_FIELDS = (
    "nabm_status",
    "neural_role",
    "social_channels",
    "reference_policies",
)
TOY_TAXONOMY_ARTIFACT_FIELDS = (
    "toy_display_name",
    *TOY_TAXONOMY_FIELDS,
)


@dataclass(frozen=True)
class CoordinationCapability:
    mixer: str
    peer_rules: tuple[str, ...]


@dataclass(frozen=True)
class ToyCapability:
    toy: str
    display_name: str
    domain_family: str
    state_family: str
    output_family: str
    topology_family: str
    coordination_family: str
    unit_surface: str
    evidence_role: str
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
        display_name="Neural HK Classification",
        domain_family="supervised_social_learning",
        state_family="supervised_probe",
        output_family="categorical_distribution",
        topology_family="static_population",
        coordination_family="output_latent_parameter_social_learning",
        unit_surface="torch_backed_distribution_latent_parameter",
        evidence_role="default_evidence",
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
        display_name="Spatial Prisoner's Dilemma",
        domain_family="binary_spatial_game",
        state_family="binary_spatial",
        output_family="binary_probability",
        topology_family="spatial_grid_mobility",
        coordination_family="output_distillation",
        unit_surface="binary_policy_tensor_backed",
        evidence_role="default_evidence",
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
        display_name="Opinion Rewiring",
        domain_family="continuous_opinion_dynamics",
        state_family="continuous_graph",
        output_family="continuous_scalar",
        topology_family="dynamic_rewiring",
        coordination_family="bounded_confidence_output_social",
        unit_surface="continuous_output_toy_specific",
        evidence_role="default_evidence",
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
        display_name="Public Goods Commons",
        domain_family="binary_public_goods_commons",
        state_family="binary_resource",
        output_family="binary_probability",
        topology_family="spatial_group_resource",
        coordination_family="output_distillation",
        unit_surface="binary_policy_tensor_backed",
        evidence_role="default_evidence",
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
        display_name="Contagion Adoption",
        domain_family="binary_contagion_cascade",
        state_family="binary_threshold",
        output_family="binary_probability",
        topology_family="spatial_exposure",
        coordination_family="output_distillation_readiness",
        unit_surface="binary_policy_tensor_backed",
        evidence_role="default_evidence",
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
        display_name="Categorical Spatial Game",
        domain_family="categorical_spatial_game",
        state_family="categorical_grid",
        output_family="categorical_distribution",
        topology_family="spatial_grid",
        coordination_family="output_distribution_parity",
        unit_surface="probability_distribution",
        evidence_role="parity_coverage",
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
        display_name="Resource Intensity",
        domain_family="continuous_resource_extraction",
        state_family="continuous_resource",
        output_family="bounded_scalar",
        topology_family="spatial_resource",
        coordination_family="bounded_scalar_parity",
        unit_surface="bounded_scalar",
        evidence_role="parity_coverage",
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
        display_name="Async Event ABM",
        domain_family="asynchronous_event_dynamics",
        state_family="event_queue",
        output_family="event_hazard",
        topology_family="event_time_snapshot",
        coordination_family="scalar_probability_parity",
        unit_surface="scalar_probability",
        evidence_role="parity_coverage",
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
        display_name="Heterogeneous Agent Rules",
        domain_family="heterogeneous_rule_dynamics",
        state_family="heterogeneous_group_state",
        output_family="binary_probability",
        topology_family="static_group_network",
        coordination_family="scalar_probability_parity",
        unit_surface="scalar_probability",
        evidence_role="parity_coverage",
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
        display_name="Market Ecology Network",
        domain_family="market_ecology_feedback",
        state_family="multi_channel_continuous",
        output_family="multi_channel_bounded_scalar",
        topology_family="dynamic_network_churn",
        coordination_family="per_channel_bounded_scalar_parity",
        unit_surface="bounded_scalar_per_channel",
        evidence_role="parity_coverage",
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


def toy_display_name(toy: str) -> str:
    return toy_capability(toy).display_name


def toy_taxonomy_metadata(toy: str) -> dict[str, str]:
    capability = toy_capability(toy)
    return {
        "toy": capability.toy,
        "display_name": capability.display_name,
        **{
            field: getattr(capability, field)
            for field in TOY_TAXONOMY_FIELDS
        },
    }


def toys_by_taxonomy(
    field: ToyTaxonomyField,
    value: str,
) -> tuple[str, ...]:
    if field not in TOY_TAXONOMY_FIELDS:
        raise KeyError(f"Unknown toy taxonomy field: {field}")
    return tuple(
        toy
        for toy, capability in TOY_CAPABILITIES.items()
        if getattr(capability, field) == value
    )


def toy_catalog() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            **toy_taxonomy_metadata(toy),
            "nabm_status": capability.nabm_status,
            "neural_role": capability.neural_role,
            "social_channels": list(capability.social_channels),
            "reference_policies": list(capability.reference_policies),
            "backends": list(capability.backends),
            "runner_kind": capability.runner_kind,
            "result_kind": capability.result_kind,
        }
        for toy, capability in TOY_CAPABILITIES.items()
    )


def artifact_taxonomy_metadata(toy: str) -> dict[str, str]:
    taxonomy = toy_taxonomy_metadata(toy)
    return {
        "toy_display_name": taxonomy["display_name"],
        **{
            field: taxonomy[field]
            for field in TOY_TAXONOMY_FIELDS
        },
    }


def nabm_status(toy: str) -> str:
    return toy_capability(toy).nabm_status


def artifact_capability_metadata(toy: str) -> dict[str, object]:
    capability = toy_capability(toy)
    return {
        **artifact_taxonomy_metadata(toy),
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
