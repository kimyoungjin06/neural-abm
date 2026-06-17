from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

import neural_abm.api_lite as api_lite


EXPECTED_PUBLIC_API_LITE = {
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
    "LITE_SOCIAL_CHANNEL_KINDS",
    "NABMLocalStep",
    "NABMStepResult",
    "PROBABILITY_DISTRIBUTION_CHANNEL",
    "PeerSelectionResult",
    "PeerSelector",
    "SCALAR_PROBABILITY_CHANNEL",
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
    "scalar_output_similarity_matrix",
    "select_bounded_scalar_output_peers",
    "select_distribution_output_peers",
    "select_scalar_output_peers",
    "social_diagnostics",
    "supported_toys",
    "toy_catalog",
    "toy_capability",
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
}

FORBIDDEN_PUBLIC_API_LITE_NAMES = {
    "NABMAgent",
    "NABMStep",
    "NABMUnit",
    "ObservationSpec",
    "SocialBlock",
    "SocialMessageSpec",
    "mix_probability_distributions",
    "state_dict_values",
    "tensor_message_values",
}


def test_public_api_lite_exports_exact_torch_free_seed_surface() -> None:
    assert set(api_lite.__all__) == EXPECTED_PUBLIC_API_LITE
    assert list(api_lite.__all__) == sorted(api_lite.__all__)

    for name in EXPECTED_PUBLIC_API_LITE:
        assert hasattr(api_lite, name), name
    for name in FORBIDDEN_PUBLIC_API_LITE_NAMES:
        assert not hasattr(api_lite, name), name


def test_public_api_lite_smoke() -> None:
    assert api_lite.toy_display_name("toy7") == "Resource Intensity"
    assert api_lite.toys_by_taxonomy("unit_surface", "scalar_probability") == (
        "toy8",
        "toy9",
    )
    catalog = api_lite.toy_catalog()
    assert len(catalog) == 10
    assert catalog[0]["toy"] == "toy1"
    assert catalog[0]["display_name"] == "Neural HK Classification"

    peer_ids = [[1], [0, 2], []]
    readiness = api_lite.binary_peer_mean_values(
        peer_ids=peer_ids,
        values=np.asarray([0.0, 1.0, 0.5], dtype=np.float64),
    )
    np.testing.assert_allclose(readiness, [1.0, 0.25, 0.0])

    aggregate = api_lite.aggregate_social_diagnostic_fields(
        peer_ids=peer_ids,
        social_losses=[0.0, 0.2, 0.4],
        social_update_norms=[0.1, 0.2, 0.3],
    )
    assert aggregate["mean_peer_count"] == pytest.approx(1.0)
    assert aggregate["mean_social_loss"] == pytest.approx(0.2)

    channel = api_lite.SocialChannel(
        name="readiness",
        kind=api_lite.SCALAR_PROBABILITY_CHANNEL,
        commit_mode="readiness_commit",
    )
    assert channel.kind == "scalar_probability"
    assert api_lite.LITE_SOCIAL_CHANNEL_KINDS == (
        api_lite.SCALAR_PROBABILITY_CHANNEL,
        api_lite.BOUNDED_SCALAR_CHANNEL,
    )
    for unsupported_kind in (
        api_lite.PROBABILITY_DISTRIBUTION_CHANNEL,
        "tensor",
    ):
        with pytest.raises(ValueError, match="api_lite SocialChannel kind"):
            api_lite.SocialChannel(
                name="unsupported",
                kind=unsupported_kind,
                commit_mode="unsupported_commit",
            )

    mixed = api_lite.mix_scalar_probabilities(
        values=np.asarray([0.0, 1.0, 0.5], dtype=np.float64),
        peer_ids=peer_ids,
        alpha=0.5,
        channel=channel.name,
        commit_mode=channel.commit_mode,
    )
    assert isinstance(mixed, api_lite.SocialMixResult)
    np.testing.assert_allclose(mixed.mixed_values, [0.5, 0.625, 0.5])

    commit = api_lite.CommitReport.from_mix_result(
        mixed,
        committed_agent_ids=[0, 1],
    )
    assert commit.channel == "readiness"
    assert commit.committed_agent_ids == [0, 1]
    diagnostics = api_lite.social_diagnostics(mixed)
    assert diagnostics.active_agent_count == 2

    class Adapter:
        def update(self, scale: float) -> api_lite.LocalUpdateReport:
            return api_lite.LocalUpdateReport(
                losses=[scale],
                active_agent_ids=[0],
                diagnostics={"scale": scale},
            )

    local_report = api_lite.NABMLocalStep(Adapter()).run(0.75)
    assert local_report.losses == [0.75]
    assert local_report.diagnostics == {"scale": 0.75}


def test_public_api_lite_imports_when_torch_is_blocked() -> None:
    code = """
import importlib.abc
import json
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch import blocked for api_lite smoke")
        return None

sys.modules.pop("torch", None)
sys.meta_path.insert(0, BlockTorch())

import neural_abm.api_lite as api_lite

channel = api_lite.SocialChannel(
    name="readiness",
    kind=api_lite.SCALAR_PROBABILITY_CHANNEL,
    commit_mode="readiness_commit",
)
mixed = api_lite.mix_scalar_probabilities(
    values=__import__("numpy").asarray([0.0, 1.0], dtype=float),
    peer_ids=[[1], [0]],
    alpha=0.5,
    channel=channel.name,
    commit_mode=channel.commit_mode,
)
commit = api_lite.CommitReport.from_mix_result(mixed, committed_agent_ids=[0, 1])

class Adapter:
    def update(self, scale):
        return api_lite.LocalUpdateReport(
            losses=[scale],
            active_agent_ids=[0],
            diagnostics={"scale": scale},
        )

local_report = api_lite.NABMLocalStep(Adapter()).run(0.25)
try:
    api_lite.SocialChannel(
        name="tensor",
        kind="tensor",
        commit_mode="tensor_commit",
    )
except ValueError:
    rejected_tensor_channel = True
else:
    rejected_tensor_channel = False

print(json.dumps({
    "commit_channel": commit.channel,
    "exports": list(api_lite.__all__),
    "lite_social_channel_kinds": list(api_lite.LITE_SOCIAL_CHANNEL_KINDS),
    "local_losses": local_report.losses,
    "mixed_values": list(mixed.mixed_values),
    "rejected_tensor_channel": rejected_tensor_channel,
    "torch_loaded": "torch" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert set(payload["exports"]) == EXPECTED_PUBLIC_API_LITE
    assert payload["commit_channel"] == "readiness"
    assert payload["lite_social_channel_kinds"] == [
        api_lite.SCALAR_PROBABILITY_CHANNEL,
        api_lite.BOUNDED_SCALAR_CHANNEL,
    ]
    assert payload["local_losses"] == [0.25]
    assert payload["mixed_values"] == [0.5, 0.5]
    assert payload["rejected_tensor_channel"] is True
    assert payload["torch_loaded"] is False


def test_package_root_imports_when_torch_is_blocked() -> None:
    code = """
import importlib.abc
import json
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch import blocked for root package smoke")
        return None

sys.modules.pop("torch", None)
sys.meta_path.insert(0, BlockTorch())

import neural_abm

root_names = [
    "CommitReport",
    "LocalUpdateAdapter",
    "LocalUpdateReport",
    "NABMLocalStep",
    "NABMStepResult",
    "PeerSelectionResult",
    "PeerSelector",
    "SocialDiagnostics",
    "SocialMixResult",
    "SocialValueBuilder",
    "social_diagnostics",
]
resolved_root_names = {}
for name in root_names:
    value = getattr(neural_abm, name)
    resolved_root_names[name] = getattr(value, "__name__", type(value).__name__)

channel = neural_abm.SocialChannel(
    name="readiness",
    kind=neural_abm.SCALAR_PROBABILITY_CHANNEL,
    commit_mode="readiness_commit",
)

class Adapter:
    def update(self, scale):
        return neural_abm.LocalUpdateReport(losses=[scale])

local_report = neural_abm.NABMLocalStep(Adapter()).run(0.5)

print(json.dumps({
    "version": neural_abm.__version__,
    "has_nabm_unit_export": "NABMUnit" in neural_abm.__all__,
    "local_losses": local_report.losses,
    "resolved_root_names": resolved_root_names,
    "social_channel": channel.__class__.__name__,
    "torch_loaded": "torch" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload["version"] == "0.1.0a5"
    assert payload["has_nabm_unit_export"] is True
    assert payload["local_losses"] == [0.5]
    assert set(payload["resolved_root_names"]) == {
        "CommitReport",
        "LocalUpdateAdapter",
        "LocalUpdateReport",
        "NABMLocalStep",
        "NABMStepResult",
        "PeerSelectionResult",
        "PeerSelector",
        "SocialDiagnostics",
        "SocialMixResult",
        "SocialValueBuilder",
        "social_diagnostics",
    }
    assert payload["resolved_root_names"]["LocalUpdateAdapter"] == (
        "LocalUpdateAdapter"
    )
    assert payload["social_channel"] == "SocialChannel"
    assert payload["torch_loaded"] is False
