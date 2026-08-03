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
    / "experiments/configs/pivot_feedback_d0_candidate_content_dry_run_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-candidate-content-dry-run-v1-20260729")
PRODUCER_SCRIPT = (
    ROOT / "scripts/build_pivot_feedback_d0_candidate_content_dry_run_v1.py"
)
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_candidate_content_dry_run_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PRODUCER = _load(PRODUCER_SCRIPT, "candidate_content_dry_run_producer_test")
VERIFIER = _load(VERIFIER_SCRIPT, "candidate_content_dry_run_verifier_test")


def _document() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_current_candidate_content_dry_run_replays() -> None:
    result = VERIFIER.validate(CONFIG, OUTPUT)

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["closure"]["candidate_rows"] == 4
    assert result["closure"]["would_content_pass_rows"] == 3
    assert result["closure"]["would_content_fail_indexing_anomaly_rows"] == 1
    assert result["closure"]["indexing_anomaly_topic_ids"] == [12505]
    assert result["production_content_dispositions_created"] == 0
    assert result["production_content_required_rows_resolved"] == 0
    assert result["selection_certificate_issued"] is False
    assert result["new_titles_read"] is False
    assert result["api_requests_submitted"] == 0


def test_indexing_anomaly_precedes_lexical_pass() -> None:
    state, reason = PRODUCER.classify_candidate(
        recent_accept_fraction=0.90,
        recent_reject_fraction=0.00,
        score_drop=0.80,
        source_share_increase=0.50,
        accept_threshold=0.50,
        semantic_reject_threshold=0.50,
        score_drop_threshold=0.15,
        source_share_threshold=0.25,
    )

    assert state == "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
    assert reason == "JOINT_SCORE_DROP_AND_SOURCE_CONCENTRATION_SHIFT"


def test_unknown_titles_do_not_count_as_accept_or_reject() -> None:
    state, _ = PRODUCER.classify_candidate(
        recent_accept_fraction=0.49,
        recent_reject_fraction=0.49,
        score_drop=0.00,
        source_share_increase=0.00,
        accept_threshold=0.50,
        semantic_reject_threshold=0.50,
        score_drop_threshold=0.15,
        source_share_threshold=0.25,
    )

    assert state == "WOULD_CONTENT_UNRESOLVED"


def test_raw_scan_api_or_production_flags_fail_closed(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["raw_title_scan"] = "AUTHORIZED"
    document["authorization"]["api_submission"] = "AUTHORIZED"
    document["authorization"]["production_content_dispositions"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.CandidateContentDryRunVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    estimate = copied / "candidate_frontier_cost_estimate.json"
    estimate.write_text(
        estimate.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    estimate.chmod(0o600)

    with pytest.raises(
        VERIFIER.CandidateContentDryRunVerificationError,
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

    assert (
        "scripts.build_pivot_feedback_d0_candidate_content_dry_run_v1"
        not in imports
    )
