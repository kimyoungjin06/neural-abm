from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments/configs/pivot_feedback_holdout_access_seal_v1.yaml"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_holdout_access_seal_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load(VERIFIER, "holdout_access_seal_contract_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mutated_contract(tmp_path: Path, mutate) -> Path:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    mutate(contract)
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    return path


def test_contract_binds_exact_authorization_g0_v2_and_closure_plan() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    assert contract["schema_version"] == (
        "pivot_feedback_holdout_access_seal_contract.v1"
    )
    assert contract["status"] == (
        "PENDING_INDEPENDENT_VALIDATION_FUTURE_SEAL_CREATION_ONLY"
    )
    for key in (
        "authorization_parent",
        "bound_g0_v2",
        "validated_prerequisite_closure_plan",
    ):
        binding = contract[key]
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
        assert binding["mutation_allowed"] is False


def test_contract_freezes_exact_holdout_partition() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    partition = contract["partition_contract"]

    assert partition == {
        "partition_id": "g4_temporal_holdout",
        "entry_start_year": 2020,
        "entry_end_year": 2022,
        "entry_years": [2020, 2021, 2022],
        "outcome_followup_through_year": 2025,
        "temporal_holdout": True,
        "domain_cold": True,
        "author_cold": True,
        "outcome_blind_definition": True,
        "partition_membership_immutable_after_contract_validation": True,
        "future_seal_must_bind_this_partition_exactly": True,
    }


def test_contract_separates_creation_verification_and_opening() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    phases = contract["phase_separation_contract"]

    assert tuple(phases["ordered_phases"]) == VERIFY_MODULE.EXPECTED_PHASES
    assert phases["contract_validation"]["may_create_or_issue_seal"] is False
    assert phases["future_seal_creation"]["state"] == (
        "REQUIRES_SEPARATE_SUCCESSOR_GATE"
    )
    assert phases["future_seal_creation"]["may_open_seal"] is False
    assert phases["future_seal_independent_verification"]["state"] == (
        "REQUIRED_AFTER_RECEIPT"
    )
    assert phases["future_seal_opening"]["state"] == (
        "REQUIRES_DISTINCT_G4_OPENING_GATE"
    )
    assert phases["future_seal_opening"]["may_be_combined_with_creation_phase"] is False


def test_contract_defines_atomic_manifest_and_receipt_without_creating_them() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    manifest = contract["future_manifest_contract"]
    receipt = contract["future_receipt_contract"]
    lifecycle = contract["lifecycle"]

    assert tuple(manifest["required_top_level_fields"]) == (
        VERIFY_MODULE.EXPECTED_MANIFEST_FIELDS
    )
    assert tuple(receipt["required_top_level_fields"]) == (
        VERIFY_MODULE.EXPECTED_RECEIPT_FIELDS
    )
    assert receipt["required_schema_version"] == (
        "pivot_feedback_holdout_access_seal_outputs.v1"
    )
    assert receipt["required_status"] == (
        "PASS_CONFIRMATORY_PARTITIONS_SEALED_AND_UNOPENED"
    )
    assert receipt["receipt_written_last"] is True
    assert lifecycle["holdout_artifact_paths_enumerated"] is False
    assert lifecycle["holdout_artifact_bytes_hashed"] is False
    assert lifecycle["seal_created_or_issued"] is False
    assert lifecycle["seal_receipt_written"] is False


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
    assert result["partition_replay"]["entry_years"] == [2020, 2021, 2022]
    assert result["phase_separation_replay"] == {
        "ordered_phases": list(VERIFY_MODULE.EXPECTED_PHASES),
        "creation_requires_successor_gate": True,
        "opening_requires_distinct_g4_gate": True,
        "creation_and_opening_separate": True,
    }
    assert result["current_boundary_replay"]["access_accounting"] == {
        "repository_documents_read": 4,
        "operator_json_metadata_files_read": 0,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "holdout_artifact_paths_enumerated": 0,
        "holdout_artifact_bytes_hashed": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    assert result["authorization"]["future_seal_creation_or_receipt_issuance"] == (
        "REQUIRES_SUCCESSOR_GATE"
    )


def test_verifier_rejects_partition_drift(tmp_path: Path) -> None:
    path = _mutated_contract(
        tmp_path,
        lambda contract: contract["partition_contract"].__setitem__(
            "entry_end_year", 2023
        ),
    )
    with pytest.raises(
        VERIFY_MODULE.HoldoutAccessSealContractError,
        match="partition contract exact freeze",
    ):
        VERIFY_MODULE.run(path)


def test_verifier_rejects_creation_without_successor_gate(tmp_path: Path) -> None:
    path = _mutated_contract(
        tmp_path,
        lambda contract: contract["phase_separation_contract"][
            "future_seal_creation"
        ].__setitem__("state", "AUTHORIZED"),
    )
    with pytest.raises(
        VERIFY_MODULE.HoldoutAccessSealContractError,
        match="future creation gate",
    ):
        VERIFY_MODULE.run(path)


def test_verifier_rejects_receipt_before_final_revalidation(tmp_path: Path) -> None:
    path = _mutated_contract(
        tmp_path,
        lambda contract: contract["atomic_creation_protocol"].__setitem__(
            "receipt_must_not_exist_before_final_directory_revalidation", False
        ),
    )
    with pytest.raises(
        VERIFY_MODULE.HoldoutAccessSealContractError,
        match=(
            "atomic protocol requirement "
            "receipt_must_not_exist_before_final_directory_revalidation"
        ),
    ):
        VERIFY_MODULE.run(path)


def test_verifier_rejects_resealing_previously_opened_artifact(tmp_path: Path) -> None:
    path = _mutated_contract(
        tmp_path,
        lambda contract: contract["fail_closed_policy"].__setitem__(
            "previously_opened_artifact_may_be_resealed", True
        ),
    )
    with pytest.raises(
        VERIFY_MODULE.HoldoutAccessSealContractError,
        match="fail-closed boundary previously_opened_artifact_may_be_resealed",
    ):
        VERIFY_MODULE.run(path)
