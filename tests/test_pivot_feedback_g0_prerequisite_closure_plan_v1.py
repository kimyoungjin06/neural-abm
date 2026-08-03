from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "experiments/configs/pivot_feedback_g0_prerequisite_closure_plan_v1.yaml"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_g0_prerequisite_closure_plan_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load(VERIFIER, "g0_prerequisite_closure_plan_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_binds_exact_authorization_and_evidence_chain() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))

    assert plan["schema_version"] == (
        "pivot_feedback_g0_prerequisite_closure_plan.v1"
    )
    assert plan["status"] == (
        "PENDING_INDEPENDENT_VALIDATION_FOUR_WORKSTREAM_PLAN_NO_EXECUTION"
    )
    for key in (
        "authorization_parent",
        "immutable_g0_v1_skeleton",
        "readiness_inventory",
        "gate_sequence",
        "legacy_g_minus_1_registry",
    ):
        binding = plan[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
    receipt = plan["passed_g_minus_1a_receipt"]
    for key in ("summary", "config", "verifier"):
        binding = receipt[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])


def test_plan_has_exact_acyclic_four_workstream_dag() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    dag = plan["workstream_dag"]

    assert tuple(dag["order"]) == VERIFY_MODULE.EXPECTED_WORKSTREAMS
    assert dag["workstream_count"] == 4
    for workstream in VERIFY_MODULE.EXPECTED_WORKSTREAMS[:3]:
        assert dag["dependencies"][workstream] == []
    assert tuple(dag["dependencies"][VERIFY_MODULE.EXPECTED_WORKSTREAMS[3]]) == (
        VERIFY_MODULE.EXPECTED_WORKSTREAMS[:3]
    )
    assert dag["all_workstreams_currently_complete"] is False
    assert dag["plan_does_not_authorize_execution"] is True


def test_plan_diagnoses_g_minus_1_contract_generation_mismatch() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    workstream = plan["workstreams"]["G_MINUS_1_RECEIPT_CLOSURE"]
    diagnostic = workstream["diagnostic"]

    assert workstream["current_state"] == (
        "PASSED_G_MINUS_1A_RECEIPT_EXISTS_LEGACY_G_MINUS_1_BINDING_INCOMPATIBLE"
    )
    assert diagnostic["mismatch_class"] == (
        "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE"
    )
    assert diagnostic["available_receipt_schema_version"] == (
        "pivot_feedback_g_minus_1a_receipt.v1"
    )
    assert diagnostic["legacy_registry_status"] == "G_MINUS_1_PENDING"
    assert diagnostic["legacy_registry_can_pass"] is False
    deliverable = workstream["next_deliverable"]
    assert deliverable["kind"] == (
        "VERSIONED_METADATA_ONLY_BINDING_RECONCILIATION_CONTRACT"
    )
    assert deliverable["may_mutate_g0_v1"] is False
    assert deliverable["may_pass_legacy_g_minus_1"] is False
    assert workstream["execution_status"] == "NOT_AUTHORIZED_PLAN_ONLY"


def test_plan_preserves_all_choices_and_sections_as_pending() -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    choices = plan["workstreams"][
        "SEVENTEEN_QUANTITATIVE_CHOICE_PRECOMMITMENTS"
    ]
    packages = choices["packages"]

    assert [item["package_id"] for item in packages] == list(
        VERIFY_MODULE.EXPECTED_PACKAGES
    )
    flattened = [
        choice
        for item in packages
        for choice in item["choice_ids"]
    ]
    assert tuple(flattened) == VERIFY_MODULE.EXPECTED_CHOICES
    assert all(
        value is None
        for item in packages
        for value in item["current_values"]
    )
    assert choices["package_count"] == 5
    assert choices["choice_count"] == 17
    assert choices["resolved_value_count"] == 0

    sections = plan["workstreams"][
        "FIFTEEN_REQUIRED_SECTION_COMPLETION_CONTRACTS"
    ]
    assert tuple(item["section_id"] for item in sections["sections"]) == (
        VERIFY_MODULE.EXPECTED_SECTIONS
    )
    assert all(item["state"] == "NOT_FREEZE_READY" for item in sections["sections"])
    assert sections["section_count"] == 15
    assert sections["freeze_ready_section_count"] == 0


def test_verifier_is_metadata_only_and_current_plan_passes() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})

    result = VERIFY_MODULE.run(PLAN)

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
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
    assert result["g_minus_1_route_replay"]["passing_g_minus_1a_receipt_exists"]
    assert result["g_minus_1_route_replay"]["legacy_g_minus_1_remains_pending"]
    assert result["quantitative_choice_route_replay"]["choice_count"] == 17
    assert result["required_section_route_replay"]["section_count"] == 15
    assert result["authorization"]["actual_g0_v2_binding_change"] == (
        "NOT_AUTHORIZED"
    )


def test_verifier_rejects_false_evidence_absence_diagnosis(tmp_path: Path) -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    plan["workstreams"]["G_MINUS_1_RECEIPT_CLOSURE"]["diagnostic"][
        "mismatch_class"
    ] = "EVIDENCE_ABSENT"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0PrerequisiteClosurePlanError,
        match="G-1 mismatch class",
    ):
        VERIFY_MODULE.run(mutated)


def test_verifier_rejects_silently_resolved_choice(tmp_path: Path) -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    plan["workstreams"]["SEVENTEEN_QUANTITATIVE_CHOICE_PRECOMMITMENTS"][
        "packages"
    ][0]["current_values"][0] = 1.5
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0PrerequisiteClosurePlanError,
        match="values must remain null",
    ):
        VERIFY_MODULE.run(mutated)


def test_verifier_rejects_unauthorized_g0_v2_change(tmp_path: Path) -> None:
    plan = yaml.safe_load(PLAN.read_text(encoding="utf-8"))
    plan["next_atomic_step"]["actual_g0_v2_binding_change_authorized"] = True
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0PrerequisiteClosurePlanError,
        match="G0 v2 boundary",
    ):
        VERIFY_MODULE.run(mutated)
