from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
RESOLUTION = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_holdout_production_input_resolution_v1.yaml"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_holdout_production_input_resolution_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFY_MODULE = _load(VERIFIER, "holdout_production_input_resolution_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mutated_resolution(tmp_path: Path, mutate) -> Path:
    resolution = yaml.safe_load(RESOLUTION.read_text(encoding="utf-8"))
    mutate(resolution)
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(resolution, sort_keys=False), encoding="utf-8")
    return path


def test_resolution_binds_exact_repository_metadata_only() -> None:
    resolution = yaml.safe_load(RESOLUTION.read_text(encoding="utf-8"))

    assert resolution["schema_version"] == (
        "pivot_feedback_holdout_production_input_resolution.v1"
    )
    assert resolution["status"] == (
        "BLOCKED_NO_MATERIALIZED_G4_ARTIFACT_SET_NO_SEAL_ACTION"
    )
    bindings = {
        "authorization_parent": resolution["authorization_parent"],
        **resolution["bound_design_sources"],
    }
    assert set(bindings) == set(VERIFY_MODULE.EXPECTED_BINDINGS)
    for binding in bindings.values():
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
        assert binding["mutation_allowed"] is False


def test_resolver_is_metadata_only_and_current_resolution_passes() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})

    result = VERIFY_MODULE.run(RESOLUTION)

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert result["declared_inputs"] == {
        "source_root_bound": False,
        "complete_artifact_list_bound": False,
        "artifact_count": 0,
        "production_preflight_written": False,
        "production_seal_created": False,
        "production_receipt_written": False,
    }
    replay = result["materialization_replay"]
    assert replay["logical_partition_reserved"] is True
    assert replay["g4_domain_selected"] is False
    assert replay["g4_field_selected"] is False
    assert replay["g4_artifact_set_materialized"] is False
    assert replay["available_d0_entry_year_maximum"] == 2012
    assert replay["requested_g4_entry_years"] == [2020, 2021, 2022]
    assert replay["existing_d0_artifact_substitution_valid"] is False


def test_resolution_distinguishes_logical_reservation_from_byte_seal() -> None:
    result = VERIFY_MODULE.run(RESOLUTION)
    decision = result["resolution_decision"]

    assert decision["blocker_class"] == (
        "PREREQUISITE_ARTIFACT_NOT_MATERIALIZED_NOT_MISSING_USER_PATH"
    )
    assert decision["production_status"] == (
        "NOT_RUN_NO_MATERIALIZED_G4_ARTIFACT_SET"
    )
    assert decision["logical_partition_reservation_equals_byte_seal"] is False
    assert decision["required_successor_order"] == (
        VERIFY_MODULE.EXPECTED_SUCCESSOR_ORDER
    )
    assert result["access_accounting"]["operator_paths_searched_or_scanned"] == 0
    assert result["access_accounting"]["holdout_artifact_paths_enumerated"] == 0
    assert result["access_accounting"]["holdout_artifact_bytes_hashed"] == 0
    assert result["access_accounting"]["confirmatory_or_holdout_rows_read"] == 0


def test_resolver_rejects_permission_to_run_preflight_now(tmp_path: Path) -> None:
    path = _mutated_resolution(
        tmp_path,
        lambda resolution: resolution["resolution_decision"].__setitem__(
            "production_preflight_may_run_now", True
        ),
    )
    with pytest.raises(
        VERIFY_MODULE.HoldoutProductionInputResolutionError,
        match="decision boundary production_preflight_may_run_now",
    ):
        VERIFY_MODULE.run(path)


def test_resolver_rejects_relabelling_d0_as_g4(tmp_path: Path) -> None:
    path = _mutated_resolution(
        tmp_path,
        lambda resolution: resolution["materialization_replay"][
            "available_d0_entry_source"
        ].__setitem__("may_be_relabelled_as_g4_holdout_artifact", True),
    )
    with pytest.raises(
        VERIFY_MODULE.HoldoutProductionInputResolutionError,
        match="D0 substitution boundary",
    ):
        VERIFY_MODULE.run(path)
