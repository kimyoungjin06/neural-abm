from __future__ import annotations

import numpy as np
import pytest

import neural_abm.api as api


EXPECTED_PUBLIC_API = {
    "BOUNDED_SCALAR_CHANNEL",
    "BinaryReadinessPropagationReport",
    "BinaryReadinessPropagationUnit",
    "CommitAdapter",
    "CommitReport",
    "CoordinationCapability",
    "DomainRunSettings",
    "DomainToyAdapter",
    "DomainToyResult",
    "DomainToyRunner",
    "LocalUpdateAdapter",
    "LocalUpdateReport",
    "NABMAgent",
    "NABMLocalStep",
    "NABMStep",
    "NABMStepResult",
    "NABMUnit",
    "NABMUnitReport",
    "ObservationSpec",
    "PROBABILITY_DISTRIBUTION_CHANNEL",
    "PeerSelectionResult",
    "PeerSelector",
    "SCALAR_PROBABILITY_CHANNEL",
    "SocialBlock",
    "SocialChannel",
    "SocialDiagnostics",
    "SocialMessageSpec",
    "SocialMixResult",
    "SocialValueBuilder",
    "TOY_TAXONOMY_FIELDS",
    "ToyCapability",
    "aggregate_social_diagnostic_fields",
    "binary_peer_aggregate_values",
    "binary_peer_mean_values",
    "copy_peer_ids",
    "domain_summary_payload",
    "empty_peers",
    "make_domain_run_dir",
    "micro_social_diagnostic_fields",
    "mix_bounded_scalars",
    "mix_probability_distributions",
    "mix_scalar_probabilities",
    "peer_ids_for_mixer",
    "scalar_message_values",
    "select_bounded_scalar_output_peers",
    "select_distribution_output_peers",
    "select_scalar_output_peers",
    "social_diagnostics",
    "state_dict_values",
    "supported_toys",
    "tensor_message_values",
    "toy_catalog",
    "toy_capability",
    "toy_display_name",
    "toy_taxonomy_metadata",
    "toys_by_taxonomy",
    "validate_peer_ids",
    "write_domain_run_metadata",
    "write_domain_summary_artifact",
    "write_json_artifact",
}

FORBIDDEN_STABLE_API_NAMES = {
    "BatchedMLPPolicyCache",
    "BinaryPolicyLearningUnit",
    "BinaryRevisionLearningUnit",
    "TensorBatchedMLPRuntime",
    "TensorPolicyRuntime",
    "load_manifest",
    "run_binary_policy_learning_step",
    "run_evidence_gate",
    "run_toy2",
    "run_toy4",
    "run_toy5",
}


def test_public_api_v0_exports_exact_stable_surface() -> None:
    assert set(api.__all__) == EXPECTED_PUBLIC_API
    assert list(api.__all__) == sorted(api.__all__)

    for name in EXPECTED_PUBLIC_API:
        assert hasattr(api, name), name
    for name in FORBIDDEN_STABLE_API_NAMES:
        assert not hasattr(api, name), name


def test_public_api_v0_toy_taxonomy_smoke() -> None:
    assert api.supported_toys() == tuple(f"toy{index}" for index in range(1, 11))
    assert api.toy_display_name("toy10") == "Market Ecology Network"
    assert api.toys_by_taxonomy("output_family", "binary_probability") == (
        "toy2",
        "toy4",
        "toy5",
        "toy9",
    )
    taxonomy = api.toy_taxonomy_metadata("toy8")
    assert taxonomy["display_name"] == "Async Event ABM"
    assert taxonomy["output_family"] == "event_hazard"
    assert set(api.TOY_TAXONOMY_FIELDS).issubset(taxonomy)
    catalog = api.toy_catalog()
    assert len(catalog) == 10
    assert catalog[-1]["toy"] == "toy10"
    assert catalog[-1]["display_name"] == "Market Ecology Network"


def test_public_api_v0_social_mix_smoke() -> None:
    channel = api.SocialChannel(
        name="adoption_probability",
        kind=api.SCALAR_PROBABILITY_CHANNEL,
        commit_mode="probability_commit",
    )
    block = api.SocialBlock(alpha=0.5)

    result = block.mix(
        channel=channel,
        values=np.asarray([0.2, 0.8], dtype=np.float64),
        peer_ids=[[1], [0]],
    )

    assert isinstance(result, api.SocialMixResult)
    np.testing.assert_allclose(result.mixed_values, [0.5, 0.5])
    assert result.channel == "adoption_probability"
    assert result.commit_mode == "probability_commit"


def test_public_api_v0_readiness_and_diagnostic_smoke() -> None:
    peer_ids = [[1], [0, 2], []]
    readiness = api.binary_peer_mean_values(
        peer_ids=peer_ids,
        values=np.asarray([0.0, 1.0, 0.5], dtype=np.float64),
    )
    np.testing.assert_allclose(readiness, [1.0, 0.25, 0.0])

    aggregate = api.aggregate_social_diagnostic_fields(
        peer_ids=peer_ids,
        social_losses=[0.0, 0.2, 0.4],
        social_update_norms=[0.1, 0.2, 0.3],
    )
    assert aggregate["mean_peer_count"] == pytest.approx(1.0)
    assert aggregate["mean_social_loss"] == pytest.approx(0.2)
