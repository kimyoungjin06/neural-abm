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
    / "experiments/configs/pivot_feedback_d0_t10543_rank3_local_routing_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-t10543-rank3-local-routing-v1-20260729")
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_t10543_rank3_local_routing_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_SCRIPT, "t10543_rank3_local_routing_verifier_test")


def test_current_rank3_local_routing_replays() -> None:
    result = VERIFIER.validate(CONFIG, output_root=OUTPUT)
    candidate = result["candidate"]

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert candidate["recent_accept_rows"] == 21
    assert candidate["recent_ambiguous_rows"] == 16
    assert candidate["recent_opaque_quarantine_rows"] == 27
    assert candidate["recent_opaque_quarantine_fraction"] == 0.421875
    assert candidate["candidate_rule_v2_branch"] == (
        "JOINT_SCORE_AND_SOURCE_SHIFT_INDEXING_ANOMALY"
    )
    assert candidate["candidate_rule_v2_state"] == (
        "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
    )
    assert result["api_requests_created"] == 0
    assert result["production_content_dispositions_created"] == 0


def test_production_disposition_cannot_be_opened(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["authorization"]["production_content_disposition"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.T10543Rank3LocalRoutingVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated, output_root=OUTPUT)


def test_candidate_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    candidate = copied / "t10543_rank3_candidate_rule_v2_dry_run.parquet"
    candidate.write_bytes(candidate.read_bytes() + b"\n")
    candidate.chmod(0o600)

    with pytest.raises(
        VERIFIER.T10543Rank3LocalRoutingVerificationError,
        match="artifact",
    ):
        VERIFIER.validate(CONFIG, output_root=copied)


def test_verifier_does_not_import_rank3_producer() -> None:
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

    assert "build_pivot_feedback_d0_t10543_rank3_local_routing_v1" not in imports
