from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import shutil
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "experiments/configs/pivot_feedback_d0_production_candidate_rule_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-production-candidate-rule-v1-20260729")
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_production_candidate_rule_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_SCRIPT, "production_candidate_rule_verifier_test")


def _document() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_current_production_candidate_rule_replays() -> None:
    result = VERIFIER.validate(CONFIG, OUTPUT)

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["production_candidate_dispositions"] == 4
    assert result["computer_science_field_certificate_issued"] is True
    assert result["medicine_required_topic_id"] == 10543
    assert result["medicine_required_content_state"] == "CONTENT_UNRESOLVED"
    assert result["frontier"]["total_content_pass_rows"] == 3
    assert result["frontier"]["total_content_fail_rows"] == 1
    assert result["frontier"]["total_content_unresolved_rows"] == 1
    assert result["global_selection_certificate_issued"] is False


def test_targeted_sample_execution_stays_closed_in_component(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["targeted_t10543_local_sample_execution"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.ProductionCandidateRuleVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_api_or_global_certificate_cannot_be_opened(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["new_api_request_creation"] = "AUTHORIZED"
    document["authorization"]["global_selection_certificate"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.ProductionCandidateRuleVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    frontier = copied / "adaptive_candidate_frontier_v2.parquet"
    frontier.write_bytes(frontier.read_bytes() + b"\n")
    frontier.chmod(0o600)

    with pytest.raises(
        VERIFIER.ProductionCandidateRuleVerificationError,
        match="artifact SHA",
    ):
        VERIFIER.validate(CONFIG, copied)


def test_verifier_does_not_import_producer() -> None:
    tree = ast.parse(VERIFIER_SCRIPT.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        str(node.module)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "scripts.build_pivot_feedback_d0_production_candidate_rule_v1" not in imports
