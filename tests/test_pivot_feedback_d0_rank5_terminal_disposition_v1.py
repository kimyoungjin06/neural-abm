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
    / "experiments/configs/pivot_feedback_d0_rank5_terminal_disposition_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-rank5-terminal-disposition-v1-20260729")
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_rank5_terminal_disposition_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_SCRIPT, "rank5_terminal_disposition_verifier_test")


def test_current_rank5_terminal_disposition_replays() -> None:
    result = VERIFIER.validate(CONFIG, OUTPUT)

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["cumulative_production_dispositions"] == 8
    assert result["computer_science_selected_candidate_ranks"] == [1, 45]
    assert result["medicine_lazy_pair"] is None
    assert result["medicine_blocking_candidate_rank"] == 5
    assert result["medicine_blocking_content_state"] == "CONTENT_UNRESOLVED"
    assert result["global_selection_certificate_issued"] is False
    assert result["deterministic_reselection_executed"] is False


def test_unresolved_cannot_be_changed_to_pass(tmp_path: Path) -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    document["rank5_production_disposition"]["content_state"] = "CONTENT_PASS"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.Rank5TerminalDispositionVerificationError,
        match="rank5 disposition",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_certificate_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    certificate = copied / "terminal_selection_certificate_status.json"
    certificate.write_bytes(certificate.read_bytes() + b"\n")
    certificate.chmod(0o600)

    with pytest.raises(
        VERIFIER.Rank5TerminalDispositionVerificationError,
        match="artifact",
    ):
        VERIFIER.validate(CONFIG, copied)


def test_verifier_does_not_import_terminal_producer() -> None:
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

    assert "build_pivot_feedback_d0_rank5_terminal_disposition_v1" not in imports
