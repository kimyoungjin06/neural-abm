from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
INVENTORY = (
    ROOT
    / "experiments/configs/pivot_feedback_g0_readiness_inventory_v1.yaml"
)
VERIFIER = ROOT / "scripts/verify_pivot_feedback_g0_readiness_inventory_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load(VERIFIER, "g0_readiness_inventory_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_inventory_binds_v48_and_preserves_pending_overlay() -> None:
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))

    assert inventory["schema_version"] == "pivot_feedback_g0_readiness_inventory.v1"
    assert inventory["status"] == "G0_NOT_READY_VERSIONED_SUCCESSOR_CONTRACT_REQUIRED"
    parent = inventory["authorization_parent"]
    overlay = inventory["pending_g0_overlay"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    assert overlay["sha256"] == _sha256(ROOT / overlay["relative_path"])
    assert overlay["required_fields"]["g_minus_1_binding.state"] == "PENDING"
    assert overlay["required_fields"]["holdout_access_binding.state"] == "PENDING"
    assert overlay["required_fields"]["completion_contract.pending_contract_sealable"] is False


def test_inventory_maps_exact_choice_and_section_gaps() -> None:
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    choices = inventory["quantitative_choice_inventory"]
    sections = inventory["required_section_inventory"]
    totals = inventory["inventory_totals"]

    assert [item["choice_id"] for item in choices] == list(
        VERIFY_MODULE.EXPECTED_CHOICES
    )
    resolved = [
        item
        for item in choices
        if item["disposition"] == VERIFY_MODULE.RESOLVED_DISPOSITION
    ]
    assert [item["choice_id"] for item in resolved] == [
        "author_minimum_history_works",
        "primary_pivot_thresholds",
    ]
    assert all(item["outcome_based_choice"] is False for item in resolved)
    assert [item["section_id"] for item in sections] == list(
        VERIFY_MODULE.EXPECTED_SECTIONS
    )
    assert totals == {
        "quantitative_choices_total": 19,
        "quantitative_choices_resolved_by_d0": 2,
        "quantitative_choices_still_unresolved": 17,
        "required_sections_total": 15,
        "required_sections_freeze_ready": 0,
        "required_sections_partial": 10,
        "required_sections_unresolved": 5,
    }
    assert inventory["readiness_decision"]["g0_freeze_ready"] is False
    assert set(inventory["readiness_decision"]["hard_blockers"]) == (
        VERIFY_MODULE.EXPECTED_HARD_BLOCKERS
    )


def test_verifier_is_metadata_only_and_current_inventory_passes() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})

    result = VERIFY_MODULE.run(INVENTORY, operator_base=Path("/ssd"))

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert result["access_accounting"] == {
        "repository_documents_read": 9,
        "operator_json_metadata_files_read": 8,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    replay = result["inventory_mapping_replay"]
    assert replay["quantitative_choices_resolved_by_d0"] == 2
    assert replay["quantitative_choices_still_unresolved"] == 17
    assert replay["required_sections_freeze_ready"] == 0
    assert result["readiness_decision_replay"]["g0_freeze_ready"] is False
    assert result["authorization"]["g0_measurement_freeze"] == "NOT_AUTHORIZED"
    assert result["authorization"]["g4_open"] == "NOT_AUTHORIZED"


def test_validator_rejects_promoting_an_unresolved_choice(
    tmp_path: Path,
) -> None:
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    inventory["quantitative_choice_inventory"][2][
        "proposed_successor_value"
    ] = 1.5
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(inventory, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0ReadinessInventoryError,
        match="unresolved value",
    ):
        VERIFY_MODULE.run(mutated, operator_base=Path("/ssd"))


def test_validator_rejects_false_freeze_readiness(tmp_path: Path) -> None:
    inventory = yaml.safe_load(INVENTORY.read_text(encoding="utf-8"))
    inventory["readiness_decision"]["g0_freeze_ready"] = True
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(inventory, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFY_MODULE.G0ReadinessInventoryError,
        match="G0 freeze ready",
    ):
        VERIFY_MODULE.run(mutated, operator_base=Path("/ssd"))
