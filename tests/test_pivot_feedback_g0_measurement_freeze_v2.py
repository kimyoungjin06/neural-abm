from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml"
)
PREDECESSOR = (
    ROOT
    / "experiments/configs/pivot_feedback_g0_measurement_freeze_v1.yaml"
)
VERIFIER = ROOT / "scripts/verify_pivot_feedback_g0_measurement_freeze_v2.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load(VERIFIER, "g0_measurement_freeze_v2_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_binds_authorization_predecessor_and_reconciliation() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["schema_version"] == "pivot_feedback_g0_measurement_freeze.v3"
    assert config["status"] == (
        "PENDING_G0_V2_G_MINUS_1A_BOUND_THREE_BLOCKERS_NO_SEAL"
    )
    for key in (
        "authorization_parent",
        "immutable_predecessor",
        "binding_reconciliation_contract",
    ):
        binding = config[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
    assert config["immutable_predecessor"]["predecessor_mutation_allowed"] is False


def test_v2_preserves_all_eight_authorized_predecessor_subtrees() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    predecessor = yaml.safe_load(PREDECESSOR.read_text(encoding="utf-8"))

    for subtree in VERIFY_MODULE.PRESERVED_SUBTREES:
        assert config[subtree] == predecessor[subtree]
    for key, value in predecessor["lifecycle"].items():
        if key != "successor_skeleton_status":
            assert config["lifecycle"][key] == value
    assert config["lifecycle"]["successor_skeleton_status"] == (
        "G_MINUS_1A_BOUND_THREE_BLOCKERS_PENDING"
    )


def test_v2_retains_legacy_as_pending_lineage_without_pass_inference() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    predecessor = yaml.safe_load(PREDECESSOR.read_text(encoding="utf-8"))
    legacy = config["legacy_g_minus_1_binding"]
    source = predecessor["g_minus_1_binding"]

    assert legacy["state"] == "PENDING_NOT_PASSED_IMMUTABLE_LINEAGE"
    for key in (
        "receipt_relative_path",
        "receipt_sha256",
        "required_schema_version",
        "required_status",
        "result_fingerprint_sha256",
    ):
        assert legacy[key] == source[key]
    assert legacy["lineage_only"] is True
    assert legacy["prerequisite_role_active"] is False
    assert legacy["legacy_pass_inferred"] is False


def test_v2_binds_exact_g_minus_1a_receipt_without_relabeling() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    binding = config["g_minus_1a_binding"]

    assert binding["state"] == "SATISFIED_G_MINUS_1A_BOUNDED_EVIDENCE_ONLY"
    assert binding["receipt_sha256"] == (
        VERIFY_MODULE.EXPECTED_BINDINGS["g_minus_1a_summary"]["sha256"]
    )
    assert binding["required_schema_version"] == (
        "pivot_feedback_g_minus_1a_receipt.v1"
    )
    assert binding["required_status"] == VERIFY_MODULE.G_MINUS_1A_STATUS
    assert binding["receipt_unresolved_boundary_count"] == 14
    assert binding["current_snapshot_proxy_and_author_token_caveats_preserved"]
    for key in (
        "may_pass_or_relabel_legacy_g_minus_1",
        "may_by_itself_pass_g0",
        "may_by_itself_authorize_model_fitting",
        "may_by_itself_open_g4",
        "may_by_itself_establish_scientific_effect",
    ):
        assert binding[key] is False


def test_v2_resolves_only_g_minus_1_absence_blocker() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    readiness = config["readiness"]

    assert readiness["status"] == "G0_NOT_READY_THREE_BLOCKERS_REMAIN"
    assert readiness["resolved_blocker"] == "G_MINUS_1_RECEIPT_ABSENT"
    assert readiness["resolved_blocker_count"] == 1
    assert readiness["remaining_blocker_count"] == 3
    assert tuple(readiness["remaining_blockers"]) == (
        VERIFY_MODULE.EXPECTED_REMAINING_BLOCKERS
    )
    assert readiness["g_minus_1a_bounded_prerequisite_satisfied"] is True
    assert readiness["legacy_g_minus_1_passed"] is False
    assert readiness["g0_freeze_ready"] is False
    assert config["holdout_access_binding"]["receipt_sha256"] is None
    assert config["resolution"]["unresolved_quantitative_choice_count"] == 17
    assert config["resolution"]["unresolved_required_section_count"] == 15


def test_verifier_is_metadata_only_and_current_v2_passes() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})

    result = VERIFY_MODULE.run(CONFIG)

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert result["predecessor_preservation_replay"]["preserved_subtree_count"] == 8
    assert result["legacy_binding_replay"]["legacy_pass_inferred"] is False
    assert result["g_minus_1a_binding_replay"][
        "claim_boundary_preserved_exactly"
    ] is True
    assert result["readiness_replay"]["remaining_blocker_count"] == 3
    assert result["fail_closed_boundary_replay"]["access_accounting"] == {
        "repository_documents_read": 7,
        "operator_json_metadata_files_read": 0,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    assert result["authorization"]["g0_measurement_freeze"] == "NOT_AUTHORIZED"


def test_verifier_rejects_nonbinding_subtree_drift(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["fixed_design_choices"]["temporal_clocks"]["response_offsets"] = [0, 1]
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0MeasurementFreezeV2Error,
        match="predecessor preserved subtree fixed_design_choices",
    ):
        VERIFY_MODULE.run(mutated)


def test_verifier_rejects_false_g0_pass(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["lifecycle"]["pass_g0"] = True
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0MeasurementFreezeV2Error,
        match="lifecycle preservation pass_g0",
    ):
        VERIFY_MODULE.run(mutated)
