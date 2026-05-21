from __future__ import annotations

import pytest

from neural_abm.capabilities import (
    artifact_capability_metadata,
    nabm_status,
    supported_coordination_pairs,
    supported_toys,
    supports_coordination,
    sweep_capability_metadata,
    toy_capability,
)

VALID_NABM_STATUSES = {"full", "compatible", "reference"}


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
        assert capability.nabm_status in VALID_NABM_STATUSES
        assert nabm_status(toy) == capability.nabm_status
        assert capability.neural_role
        assert isinstance(capability.social_channels, tuple)
        assert isinstance(capability.reference_policies, tuple)


def test_artifact_and_sweep_nabm_metadata_are_serializable() -> None:
    artifact = artifact_capability_metadata("toy2")
    sweep = sweep_capability_metadata("toy2")

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
