from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SKELETON = (
    ROOT
    / "experiments/configs/pivot_feedback_g0_measurement_freeze_v1.yaml"
)
PREDECESSOR = (
    ROOT
    / "experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_g0_measurement_freeze_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load(VERIFIER, "g0_measurement_freeze_v1_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_binds_authorization_predecessor_and_inventory() -> None:
    skeleton = yaml.safe_load(SKELETON.read_text(encoding="utf-8"))

    assert skeleton["schema_version"] == "pivot_feedback_g0_measurement_freeze.v2"
    assert skeleton["status"] == (
        "PENDING_G0_SUCCESSOR_SKELETON_TWO_D0_VALUES_SERIALIZED_NO_SEAL"
    )
    for key in (
        "authorization_parent",
        "immutable_predecessor",
        "readiness_inventory_binding",
    ):
        binding = skeleton[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
    assert skeleton["immutable_predecessor"]["sha256"] == (
        VERIFY_MODULE.PREDECESSOR_SHA256
    )
    assert skeleton["immutable_predecessor"]["predecessor_mutation_allowed"] is False


def test_v1_copies_prospective_predecessor_subtrees_exactly() -> None:
    skeleton = yaml.safe_load(SKELETON.read_text(encoding="utf-8"))
    predecessor = yaml.safe_load(PREDECESSOR.read_text(encoding="utf-8"))

    for subtree in VERIFY_MODULE.COPIED_SUBTREES:
        assert skeleton[subtree] == predecessor[subtree]
    for key, value in predecessor["lifecycle"].items():
        assert skeleton["lifecycle"][key] == value
    for key, value in predecessor["completion_contract"].items():
        if key != "validation_status":
            assert skeleton["completion_contract"][key] == value


def test_v1_serializes_exactly_two_values_and_preserves_every_gap() -> None:
    skeleton = yaml.safe_load(SKELETON.read_text(encoding="utf-8"))
    resolution = skeleton["resolution"]

    assert resolution["resolved_measurements"] == VERIFY_MODULE.EXPECTED_RESOLVED
    assert resolution["resolved_measurement_count"] == 2
    assert tuple(resolution["unresolved_quantitative_choices"]) == (
        VERIFY_MODULE.EXPECTED_UNRESOLVED_CHOICES
    )
    assert all(
        value is None
        for value in resolution["unresolved_quantitative_choices"].values()
    )
    assert tuple(resolution["unresolved_required_sections"]) == (
        VERIFY_MODULE.EXPECTED_SECTIONS
    )
    assert resolution["unresolved_quantitative_choice_count"] == 17
    assert resolution["unresolved_required_section_count"] == 15
    assert resolution["freeze_ready_required_section_count"] == 0
    assert tuple(skeleton["readiness_blockers"]) == VERIFY_MODULE.EXPECTED_BLOCKERS
    assert skeleton["g_minus_1_binding"]["state"] == "PENDING"
    assert skeleton["g_minus_1_binding"]["receipt_sha256"] is None
    assert skeleton["holdout_access_binding"]["state"] == "PENDING"
    assert skeleton["holdout_access_binding"]["receipt_sha256"] is None


def test_verifier_is_metadata_only_and_current_skeleton_passes() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})

    result = VERIFY_MODULE.run(SKELETON)

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert result["access_accounting"] == {
        "repository_documents_read": 4,
        "operator_json_metadata_files_read": 0,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    assert result["resolution_replay"]["resolved_measurement_count"] == 2
    assert (
        result["resolution_replay"]["unresolved_quantitative_choice_count"]
        == 17
    )
    assert result["fail_closed_boundary_replay"]["pass_g0"] is False
    assert result["authorization"]["g0_measurement_freeze"] == "NOT_AUTHORIZED"


def test_verifier_rejects_third_resolved_value(tmp_path: Path) -> None:
    skeleton = yaml.safe_load(SKELETON.read_text(encoding="utf-8"))
    skeleton["resolution"]["resolved_measurements"]["attention_surge_threshold"] = {
        "value": 1.5,
        "selected_from_observed_outcomes": False,
        "source": "UNAUTHORIZED",
    }
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(skeleton, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0MeasurementFreezeV1Error,
        match="exact two resolved measurements",
    ):
        VERIFY_MODULE.run(mutated)


def test_verifier_rejects_false_g0_pass(tmp_path: Path) -> None:
    skeleton = yaml.safe_load(SKELETON.read_text(encoding="utf-8"))
    skeleton["lifecycle"]["pass_g0"] = True
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(skeleton, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0MeasurementFreezeV1Error,
        match="lifecycle preservation pass_g0",
    ):
        VERIFY_MODULE.run(mutated)


def test_verifier_rejects_predecessor_subtree_drift(tmp_path: Path) -> None:
    skeleton = yaml.safe_load(SKELETON.read_text(encoding="utf-8"))
    skeleton["fixed_design_choices"]["temporal_clocks"]["response_offsets"] = [
        0,
        1,
    ]
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(skeleton, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0MeasurementFreezeV1Error,
        match="predecessor copied subtree fixed_design_choices",
    ):
        VERIFY_MODULE.run(mutated)
