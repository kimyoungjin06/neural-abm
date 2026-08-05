from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = ROOT / "experiments/configs/pivot_feedback_e_g1_guard_only_exception_v1.yaml"
SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_e_g1_guard_only_exception_v1_20260805.summary.json"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_e_g1_guard_only_exception_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_PATH, "pivot_feedback_e_g1_guard_only_exception_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exception_is_exact_guard_only_and_does_not_open_g4() -> None:
    authority = yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))
    source = authority["guard_provenance"]["exact_source"]

    assert authority["authorization_basis"] == "EXPLICIT_USER_APPROVAL_GUARD_ONLY_EXCEPTION"
    assert source["sha256"] == (
        "90f79d46dd16e83088a258aeafd795541d918e4afb1915feadd5648523b5ca3a"
    )
    assert source["rows"] == 8_174_698
    assert source["required_projection_columns"] == [
        "author_numeric_id",
        "exclude_from_g1",
    ]
    assert authority["execution_semantics"]["anti_join_order"] == (
        "BEFORE_LABEL_CONTROL_SAMPLING_WEIGHT_OR_ESTIMATOR_LOGIC"
    )
    assert authority["forbidden_operations"]["sealed_private_root_access"] is True
    assert authority["forbidden_operations"]["holdout_seal_open"] is True
    assert authority["execution_semantics"]["g4_open_after_use"] is False


def test_exception_independent_summary_binds_full_guard_projection() -> None:
    replay = VERIFIER.verify(AUTHORITY)
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert replay == summary
    assert summary["status"] == (
        "PASS_INDEPENDENT_EXACT_GUARD_ONLY_EXCEPTION_G1_RESUME_AUTHORIZED"
    )
    assert summary["authority"]["sha256"] == _sha(AUTHORITY)
    assert summary["guard_replay"]["rows"] == 8_174_698
    assert summary["guard_replay"]["non_true_exclude_from_g1_rows"] == 0
    assert summary["access_accounting"]["confirmatory_rows_read"] == 0
    assert summary["access_accounting"]["other_private_g4_files_read"] == 0
    assert summary["access_accounting"]["seal_open_events"] == 0
    assert summary["access_accounting"]["g4_open"] is False


def test_exception_verifier_has_one_literal_private_g4_source() -> None:
    source = VERIFIER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and "/ssd/pivot-feedback-f-g4-private-membership" in node.value
    }

    assert len(literals) == 1
    assert "g4_author_cold_guard.parquet" in next(iter(literals))
    assert "read_table" not in source
