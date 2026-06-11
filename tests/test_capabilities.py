from __future__ import annotations

from pathlib import Path

import pytest

from neural_abm.capabilities import (
    TOY_TAXONOMY_FIELDS,
    artifact_capability_metadata,
    nabm_status,
    supported_coordination_pairs,
    supported_toys,
    supports_coordination,
    sweep_capability_metadata,
    toy_catalog,
    toy_capability,
    toy_display_name,
    toy_taxonomy_metadata,
    toys_by_taxonomy,
)

VALID_NABM_STATUSES = {"full", "compatible", "reference"}
ROOT = Path(__file__).resolve().parents[1]


def test_toy1_10_capabilities_are_registered() -> None:
    assert supported_toys() == (
        "toy1",
        "toy2",
        "toy3",
        "toy4",
        "toy5",
        "toy6",
        "toy7",
        "toy8",
        "toy9",
        "toy10",
    )


def test_toy1_10_advertise_valid_nabm_metadata() -> None:
    for toy in supported_toys():
        capability = toy_capability(toy)
        assert capability.display_name
        assert toy_display_name(toy) == capability.display_name
        assert capability.nabm_status in VALID_NABM_STATUSES
        assert nabm_status(toy) == capability.nabm_status
        assert capability.neural_role
        assert isinstance(capability.social_channels, tuple)
        assert isinstance(capability.reference_policies, tuple)
        for field in TOY_TAXONOMY_FIELDS:
            assert getattr(capability, field)


def test_toy_taxonomy_metadata_keeps_stable_ids_separate_from_feature_names() -> None:
    taxonomy = toy_taxonomy_metadata("toy10")

    assert taxonomy == {
        "toy": "toy10",
        "display_name": "Market Ecology Network",
        "domain_family": "market_ecology_feedback",
        "state_family": "multi_channel_continuous",
        "output_family": "multi_channel_bounded_scalar",
        "topology_family": "dynamic_network_churn",
        "coordination_family": "per_channel_bounded_scalar_parity",
        "unit_surface": "bounded_scalar_per_channel",
        "evidence_role": "parity_coverage",
    }


def test_toy_taxonomy_groups_by_feature_axes() -> None:
    assert toys_by_taxonomy("output_family", "binary_probability") == (
        "toy2",
        "toy4",
        "toy5",
        "toy9",
    )
    assert toys_by_taxonomy("output_family", "categorical_distribution") == (
        "toy1",
        "toy6",
    )
    assert toys_by_taxonomy("unit_surface", "binary_policy_tensor_backed") == (
        "toy2",
        "toy4",
        "toy5",
    )
    assert toys_by_taxonomy("evidence_role", "parity_coverage") == (
        "toy6",
        "toy7",
        "toy8",
        "toy9",
        "toy10",
    )


def test_toy_catalog_is_json_friendly_and_feature_first() -> None:
    catalog = toy_catalog()

    assert len(catalog) == 10
    assert catalog[1]["toy"] == "toy2"
    assert catalog[1]["display_name"] == "Spatial Prisoner's Dilemma"
    assert catalog[1]["output_family"] == "binary_probability"
    assert catalog[1]["social_channels"] == [
        "action_probability",
        "policy_distribution",
        "reputation",
    ]
    assert catalog[9]["toy"] == "toy10"
    assert catalog[9]["unit_surface"] == "bounded_scalar_per_channel"
    assert catalog[9]["backends"] == ["loop"]


def test_unknown_toy_taxonomy_field_raises() -> None:
    with pytest.raises(KeyError, match="Unknown toy taxonomy field"):
        toys_by_taxonomy("unknown", "value")  # type: ignore[arg-type]


def test_capability_matrix_documents_feature_taxonomy() -> None:
    matrix = (ROOT / "docs" / "toy-models" / "capability-matrix.md").read_text(
        encoding="utf-8"
    )

    for required in (
        "Stable IDs vs Feature Names",
        "Feature Groups",
        "Market Ecology Network",
        "Spatial Prisoner's Dilemma",
        "toy_display_name",
        "unit_surface",
        "Binary probability: `toy2`, `toy4`, `toy5`, `toy9`",
        "Default evidence families: `toy1`, `toy2`, `toy3`, `toy4`, `toy5`",
        "Parity coverage families: `toy6`, `toy7`, `toy8`, `toy9`, `toy10`",
    ):
        assert required in matrix


def test_artifact_and_sweep_nabm_metadata_are_serializable() -> None:
    artifact = artifact_capability_metadata("toy2")
    sweep = sweep_capability_metadata("toy2")

    assert artifact["toy_display_name"] == "Spatial Prisoner's Dilemma"
    assert artifact["domain_family"] == "binary_spatial_game"
    assert artifact["output_family"] == "binary_probability"
    assert artifact["unit_surface"] == "binary_policy_tensor_backed"
    assert artifact["evidence_role"] == "default_evidence"
    assert artifact["nabm_status"] == "full"
    assert artifact["social_channels"] == [
        "action_probability",
        "policy_distribution",
        "reputation",
    ]
    assert artifact["reference_policies"] == [
        "rd_well_mixed",
        "fermi_imitation",
        "reputation_imitation",
    ]
    assert sweep["nabm_status"] == "full"
    assert sweep["social_channels"] == (
        "action_probability, policy_distribution, reputation"
    )
    assert sweep["reference_policies"] == (
        "rd_well_mixed, fermi_imitation, reputation_imitation"
    )


def test_full_nabm_toys_advertise_social_channels() -> None:
    full_toys = [
        toy for toy in supported_toys() if toy_capability(toy).nabm_status == "full"
    ]

    assert full_toys == ["toy1", "toy2", "toy3", "toy4", "toy5"]
    for toy in full_toys:
        assert toy_capability(toy).social_channels


@pytest.mark.parametrize(
    ("toy", "reference_policies"),
    [
        (
            "toy2",
            ("rd_well_mixed", "fermi_imitation", "reputation_imitation"),
        ),
        ("toy4", ("imitation", "reputation_imitation")),
        (
            "toy5",
            ("simple_contagion", "complex_threshold", "reputation_imitation"),
        ),
    ],
)
def test_binary_toys_advertise_tensor_backend_and_reference_policies(
    toy: str, reference_policies: tuple[str, ...]
) -> None:
    capability = toy_capability(toy)

    assert capability.action_space == "binary"
    assert capability.nabm_status == "full"
    assert "tensor_batched" in capability.backends
    assert capability.reference_policies == reference_policies
    assert supports_coordination(toy, "output_average", "output_similarity")


def test_toy6_to_toy10_advertise_current_coordination_scope() -> None:
    for toy in ("toy6", "toy7", "toy8", "toy9", "toy10"):
        capability = toy_capability(toy)
        assert capability.nabm_status == "compatible"
        assert capability.backends == ("loop",)
        assert supports_coordination(toy, "none", "none")
        assert supports_coordination(toy, "output_average", "none")
        assert supports_coordination(toy, "output_average", "output_similarity")
        assert not supports_coordination(toy, "output_average", "bounded_confidence")


def test_toy3_keeps_opinion_specific_peer_rules() -> None:
    assert supported_coordination_pairs("toy3") == (
        ("none", "none"),
        ("output_average", "bounded_confidence"),
        ("output_average", "output_similarity"),
    )


def test_unknown_toy_capability_raises() -> None:
    with pytest.raises(KeyError, match="Unknown toy capability"):
        toy_capability("toy99")


def test_unknown_toy_nabm_status_raises() -> None:
    with pytest.raises(KeyError, match="Unknown toy capability"):
        nabm_status("toy99")
