from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_e_g1_execution_v1.yaml"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_e_g1_execution_v1_20260805.summary.json"
OUTPUT = Path("/ssd/pivot-feedback-e-g1-execution-v1-20260805")
EXECUTOR_PATH = ROOT / "scripts/execute_pivot_feedback_e_g1_v1.py"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_e_g1_execution_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_PATH, "pivot_feedback_e_g1_execution_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_execution_contract_detects_exact_frozen_authorization_conflict() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    block = config["blocking_condition"]

    assert block["block_id"] == "G1_AUTHOR_COLD_GUARD_REQUIRED_BUT_PRIVATE_G4_ACCESS_FORBIDDEN"
    assert block["frozen_guard_application_required"] is True
    assert block["exact_guard_application_pending"] is True
    assert block["preflight_v1_private_g4_artifact_is_execution_input"] is False
    assert block["preflight_v1_g4_artifact_access"] == "NOT_AUTHORIZED"
    assert block["preflight_v1_g4_seal_open"] == "NOT_AUTHORIZED"
    assert block["reconstruct_guard_from_g4_field_or_year_allowed"] is False
    assert block["safe_compliant_execution_exists"] is False
    assert block["resolution_requires_new_authority"] is True


def test_production_terminal_blocked_package_independently_replays() -> None:
    replay = VERIFIER.verify(CONFIG, OUTPUT)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert replay["status"] == summary["status"] == (
        "PASS_INDEPENDENT_VERIFICATION_OF_TERMINAL_BLOCKED_E_G1_NO_CONFIRMATORY_READ"
    )
    assert replay["production_artifact"] == summary["production_artifact"]
    assert replay["blocking_condition_replay"] == summary["blocking_condition_replay"]
    assert replay["terminal_state"] == summary["terminal_state"]
    assert summary["bound_implementation"]["config_sha256"] == _sha(CONFIG)
    assert summary["bound_implementation"]["executor_sha256"] == _sha(EXECUTOR_PATH)
    assert summary["bound_implementation"]["independent_verifier_sha256"] == _sha(VERIFIER_PATH)


def test_blocked_execution_reads_no_analytic_source_and_emits_no_rows() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    access = summary["access_accounting"]

    assert access["operator_private_g4_artifacts_read"] == 0
    assert access["confirmatory_rows_read"] == 0
    assert access["holdout_rows_read"] == 0
    assert access["labels_materialized"] == 0
    assert access["estimators_run"] == 0
    assert access["bootstrap_refits_run"] == 0
    assert access["seal_open_events"] == 0
    assert access["g4_open"] is False
    receipt = json.loads((OUTPUT / "g1_execution.outputs.json").read_text(encoding="utf-8"))
    assert receipt["artifact_count"] == 7
    assert all(row.get("rows", 0) == 0 for row in receipt["artifacts"] if row["relative_path"].endswith(".parquet"))


def test_executor_and_verifier_are_independent_and_do_not_import_analytic_engines() -> None:
    for path in (EXECUTOR_PATH, VERIFIER_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imports.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imports.isdisjoint({"duckdb", "numpy", "pandas", "statsmodels"})
    verifier_imports = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    names = {
        alias.name
        for node in ast.walk(verifier_imports)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "execute_pivot_feedback_e_g1_v1" not in names
