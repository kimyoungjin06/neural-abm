"""Torch-free public facade seed for lightweight package profiles.

This module intentionally exports only surfaces that can be imported without
loading ``torch``. It is narrower than ``neural_abm.api`` and does not expose
full unit lifecycle, tensor/state-dict social exchange, or torch-backed agent
protocols.
"""

from __future__ import annotations

from neural_abm.capabilities import (
    TOY_TAXONOMY_FIELDS,
    CoordinationCapability,
    ToyCapability,
    supported_toys,
    toy_catalog,
    toy_capability,
    toy_display_name,
    toy_taxonomy_metadata,
    toys_by_taxonomy,
)
from neural_abm.domain_runner import (
    DomainRunSettings,
    DomainToyAdapter,
    DomainToyRunner,
    make_domain_run_dir,
    write_domain_run_metadata,
)
from neural_abm.domain_social_diagnostics import (
    aggregate_social_diagnostic_fields,
    micro_social_diagnostic_fields,
)
from neural_abm.readiness import (
    BinaryReadinessPropagationReport,
    BinaryReadinessPropagationUnit,
    binary_peer_aggregate_values,
    binary_peer_mean_values,
)
from neural_abm.results import (
    DomainToyResult,
    domain_summary_payload,
    write_domain_summary_artifact,
    write_json_artifact,
)
from neural_abm.scenario_lite import (
    BoundedScalarScenarioResult,
    BoundedScalarScenarioSpec,
    ReplicatedScenarioComparison,
    ReplicatedScenarioResult,
    ReplicationSpec,
    ScenarioComparison,
    ScenarioDefinition,
    ScenarioReplicateContext,
    run_bounded_scalar_scenarios,
    run_replicated_bounded_scalar_scenarios,
)
from neural_abm.social_core import (
    BOUNDED_SCALAR_CHANNEL,
    PROBABILITY_DISTRIBUTION_CHANNEL,
    SCALAR_PROBABILITY_CHANNEL,
    PeerSelectionResult,
    SocialMixResult,
    SocialChannel as _SocialChannel,
    bounded_scalar_similarity_matrix,
    copy_peer_ids,
    distribution_output_similarity_matrix,
    empty_peers,
    mix_bounded_scalars,
    mix_scalar_probabilities,
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
    validate_probability_vector,
)
from neural_abm.unit_core import (
    CommitAdapter,
    CommitReport,
    LocalUpdateAdapter,
    LocalUpdateReport,
    NABMLocalStep,
    NABMStepResult,
    PeerSelector,
    SocialDiagnostics,
    SocialValueBuilder,
    social_diagnostics,
)

LITE_SOCIAL_CHANNEL_KINDS = (
    SCALAR_PROBABILITY_CHANNEL,
    BOUNDED_SCALAR_CHANNEL,
)


class SocialChannel(_SocialChannel):
    """Torch-free social channel metadata for api_lite scalar mix helpers."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind not in LITE_SOCIAL_CHANNEL_KINDS:
            allowed = ", ".join(LITE_SOCIAL_CHANNEL_KINDS)
            raise ValueError(
                "api_lite SocialChannel kind must be one of "
                f"{allowed}; use standalone distribution helpers or install "
                "neural-abm[torch] for tensor/state-dict social channels"
            )


__all__ = [
    "BOUNDED_SCALAR_CHANNEL",
    "BinaryReadinessPropagationReport",
    "BinaryReadinessPropagationUnit",
    "BoundedScalarScenarioResult",
    "BoundedScalarScenarioSpec",
    "CommitAdapter",
    "CommitReport",
    "CoordinationCapability",
    "DomainRunSettings",
    "DomainToyAdapter",
    "DomainToyResult",
    "DomainToyRunner",
    "LITE_SOCIAL_CHANNEL_KINDS",
    "LocalUpdateAdapter",
    "LocalUpdateReport",
    "NABMLocalStep",
    "NABMStepResult",
    "PROBABILITY_DISTRIBUTION_CHANNEL",
    "PeerSelectionResult",
    "PeerSelector",
    "ReplicatedScenarioComparison",
    "ReplicatedScenarioResult",
    "ReplicationSpec",
    "SCALAR_PROBABILITY_CHANNEL",
    "ScenarioComparison",
    "ScenarioDefinition",
    "ScenarioReplicateContext",
    "SocialChannel",
    "SocialDiagnostics",
    "SocialMixResult",
    "SocialValueBuilder",
    "TOY_TAXONOMY_FIELDS",
    "ToyCapability",
    "aggregate_social_diagnostic_fields",
    "binary_peer_aggregate_values",
    "binary_peer_mean_values",
    "bounded_scalar_similarity_matrix",
    "copy_peer_ids",
    "distribution_output_similarity_matrix",
    "domain_summary_payload",
    "empty_peers",
    "make_domain_run_dir",
    "micro_social_diagnostic_fields",
    "mix_bounded_scalars",
    "mix_scalar_probabilities",
    "peer_ids_for_mixer",
    "run_bounded_scalar_scenarios",
    "run_replicated_bounded_scalar_scenarios",
    "scalar_output_similarity_matrix",
    "select_bounded_scalar_output_peers",
    "select_distribution_output_peers",
    "select_scalar_output_peers",
    "social_diagnostics",
    "supported_toys",
    "toy_capability",
    "toy_catalog",
    "toy_display_name",
    "toy_taxonomy_metadata",
    "toys_by_taxonomy",
    "uniform_peer_count",
    "validate_bounded_scalar_vector",
    "validate_peer_ids",
    "validate_probability_distributions",
    "validate_probability_matrix",
    "validate_probability_vector",
    "write_domain_run_metadata",
    "write_domain_summary_artifact",
    "write_json_artifact",
]
