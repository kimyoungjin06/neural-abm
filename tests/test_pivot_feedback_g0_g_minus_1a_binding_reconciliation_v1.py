from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_g_minus_1a_binding_reconciliation_v1.yaml"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_g0_g_minus_1a_binding_reconciliation_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load(VERIFIER, "g_minus_1a_binding_reconciliation_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_binds_exact_authorization_and_receipt_chain() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    assert contract["schema_version"] == (
        "pivot_feedback_g0_g_minus_1a_binding_reconciliation.v1"
    )
    assert contract["status"] == (
        "PENDING_INDEPENDENT_VALIDATION_FUTURE_G0_V2_BINDING_ONLY"
    )
    for key in (
        "authorization_parent",
        "immutable_g0_v1_skeleton",
        "validated_closure_plan",
        "gate_sequence",
        "legacy_g_minus_1_registry",
    ):
        binding = contract[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
    receipt = contract["exact_g_minus_1a_receipt"]
    for key in ("summary", "config", "verifier"):
        binding = receipt[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])


def test_contract_forbids_legacy_schema_or_status_relabeling() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    diagnosis = contract["source_binding_diagnosis"]

    assert diagnosis["mismatch_class"] == (
        "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE"
    )
    assert diagnosis["immutable_required_schema_version"] == (
        "pivot_feedback_g_minus_1_data_readiness_outputs.v1"
    )
    assert diagnosis["available_schema_version"] == (
        "pivot_feedback_g_minus_1a_receipt.v1"
    )
    assert diagnosis["schema_aliasing_allowed"] is False
    assert diagnosis["status_relabeling_allowed"] is False
    assert diagnosis["legacy_pass_inference_allowed"] is False


def test_contract_defines_distinct_future_g_minus_1a_binding() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    binding = contract["future_g0_v2_binding_contract"]

    assert binding["future_field_name"] == "g_minus_1a_binding"
    assert binding["legacy_field_name_if_retained"] == "legacy_g_minus_1_binding"
    assert binding["legacy_field_state_if_retained"] == (
        "PENDING_NOT_PASSED_IMMUTABLE_LINEAGE"
    )
    assert binding["binding_state_if_exact"] == (
        "SATISFIED_G_MINUS_1A_BOUNDED_EVIDENCE_ONLY"
    )
    assert binding["required_schema_version"] == (
        "pivot_feedback_g_minus_1a_receipt.v1"
    )
    assert binding["required_status"] == VERIFY_MODULE.G_MINUS_1A_STATUS
    assert binding["may_satisfy_future_g0_v2_g_minus_1a_prerequisite"] is True
    for key in (
        "may_pass_or_relabel_legacy_g_minus_1",
        "may_by_itself_pass_g0",
        "may_by_itself_authorize_model_fitting",
        "may_by_itself_open_g4",
        "may_by_itself_establish_scientific_effect",
    ):
        assert binding[key] is False


def test_contract_preserves_future_g0_v2_boundary() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    future = contract["future_g0_v2_preservation_contract"]

    assert tuple(future["must_preserve_from_g0_v1"]) == (
        VERIFY_MODULE.EXPECTED_PRESERVED_SUBTREES
    )
    assert tuple(future["only_permitted_semantic_delta"]) == (
        VERIFY_MODULE.EXPECTED_DELTAS
    )
    assert future["must_keep_holdout_binding_pending_and_null"] is True
    assert future["must_keep_unresolved_quantitative_choice_count"] == 17
    assert future["must_keep_not_freeze_ready_section_count"] == 15
    assert future["must_keep_pass_g0_false"] is True
    assert future["future_successor_not_authored_by_this_contract"] is True
    assert future["future_successor_requires_separate_authorization_gate"] is True


def test_verifier_is_metadata_only_and_current_contract_passes() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})

    result = VERIFY_MODULE.run(CONTRACT)

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert result["diagnosis_replay"] == {
        "mismatch_class": "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE",
        "schema_aliasing_allowed": False,
        "status_relabeling_allowed": False,
        "legacy_pass_inference_allowed": False,
    }
    assert (
        result["receipt_boundary_preservation_replay"][
            "claim_boundary_preserved_exactly"
        ]
        is True
    )
    assert result["future_g0_v2_preservation_replay"][
        "unresolved_quantitative_choice_count"
    ] == 17
    assert result["fail_closed_boundary_replay"]["access_accounting"] == {
        "repository_documents_read": 9,
        "operator_json_metadata_files_read": 0,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    assert result["authorization"]["actual_g0_v2_authoring_or_modification"] == (
        "NOT_AUTHORIZED"
    )


def test_verifier_rejects_schema_aliasing(tmp_path: Path) -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    contract["source_binding_diagnosis"]["schema_aliasing_allowed"] = True
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.GMinus1ABindingReconciliationError,
        match="diagnosis boundary schema_aliasing_allowed",
    ):
        VERIFY_MODULE.run(mutated)


def test_verifier_rejects_lost_receipt_caveat(tmp_path: Path) -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    contract["preserved_receipt_boundaries"]["unresolved_boundaries"][
        "author_entity_validity"
    ] = "ESTABLISHED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.GMinus1ABindingReconciliationError,
        match="receipt unresolved-boundary preservation",
    ):
        VERIFY_MODULE.run(mutated)


def test_verifier_rejects_unauthorized_g0_v2_authoring(tmp_path: Path) -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    contract["lifecycle"]["g0_v2_authored_or_modified"] = True
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.GMinus1ABindingReconciliationError,
        match="lifecycle boundary g0_v2_authored_or_modified",
    ):
        VERIFY_MODULE.run(mutated)
