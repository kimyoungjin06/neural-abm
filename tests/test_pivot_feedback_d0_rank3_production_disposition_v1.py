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
    ROOT / "experiments/configs/pivot_feedback_d0_rank3_production_disposition_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-rank3-production-disposition-v1-20260729")
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_rank3_production_disposition_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(SCRIPT, "rank3_production_disposition_verifier_test")


def test_current_rank3_production_disposition_replays() -> None:
    result = VERIFIER.validate(CONFIG, OUTPUT)

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["cumulative_production_dispositions"] == 6
    assert result["rank3_content_state"] == "CONTENT_FAIL"
    assert result["frontier_v4_required_candidate_rank"] == 4
    assert result["frontier_v4_required_topic_id"] == 12505
    assert result["frontier_v4_required_entry_year"] == 1993
    assert result["global_selection_certificate_issued"] is False


def test_global_certificate_cannot_be_opened(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["authorization"]["global_selection_certificate"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.Rank3ProductionDispositionVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_frontier_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    frontier = copied / "adaptive_candidate_frontier_v4.parquet"
    frontier.write_bytes(frontier.read_bytes() + b"\n")
    frontier.chmod(0o600)

    with pytest.raises(
        VERIFIER.Rank3ProductionDispositionVerificationError,
        match="artifact",
    ):
        VERIFIER.validate(CONFIG, copied)


def test_verifier_does_not_import_producer() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
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

    assert "scripts.build_pivot_feedback_d0_rank3_production_disposition_v1" not in imports
