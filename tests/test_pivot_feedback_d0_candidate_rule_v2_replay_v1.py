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
    ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_candidate_rule_v2_retrospective_replay_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-candidate-rule-v2-replay-v1-20260729")
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_candidate_rule_v2_replay_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_SCRIPT, "candidate_rule_v2_replay_verifier_test")


def test_current_candidate_rule_v2_replay() -> None:
    result = VERIFIER.validate(CONFIG, OUTPUT)

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["candidate_rows"] == 5
    assert result["unchanged_rows"] == 4
    assert result["changed_rows"] == 1
    assert result["changed_topic_id"] == 10543
    assert result["v2_would_content_pass_rows"] == 3
    assert result["v2_would_content_fail_indexing_anomaly_rows"] == 2
    assert result["v2_would_content_unresolved_rows"] == 0
    assert result["production_candidate_rule_frozen"] is False
    assert result["selection_certificate_issued"] is False


def test_production_freeze_cannot_be_opened(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["authorization"]["production_candidate_rule_freeze"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.CandidateRuleV2ReplayVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_replay_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    replay = copied / "candidate_rule_v2_retrospective_replay.parquet"
    replay.write_bytes(replay.read_bytes() + b"\n")
    replay.chmod(0o600)

    with pytest.raises(
        VERIFIER.CandidateRuleV2ReplayVerificationError,
        match="artifact",
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

    assert "scripts.build_pivot_feedback_d0_candidate_rule_v2_replay_v1" not in imports
