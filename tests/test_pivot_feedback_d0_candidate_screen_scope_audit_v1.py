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
    / "experiments/configs/pivot_feedback_d0_candidate_screen_scope_audit_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-candidate-screen-scope-audit-v1-20260729")
PRODUCER_SCRIPT = (
    ROOT / "scripts/build_pivot_feedback_d0_candidate_screen_scope_audit_v1.py"
)
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_candidate_screen_scope_audit_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_SCRIPT, "pivot_feedback_candidate_scope_verifier_test")


def _document() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_current_candidate_scope_audit_replays() -> None:
    result = VERIFIER.validate(CONFIG, OUTPUT)
    closure = result["closure"]

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert closure["candidate_rows"] == 70_277
    assert closure["distinct_candidate_topics"] == 994
    assert closure["candidate_window_rows"] == 421_662
    assert closure["unique_candidate_topic_year_cells"] == 99_433
    assert closure["unique_candidate_window_linked_work_rows"] == 22_437_851
    assert closure["candidate_window_linked_work_memberships"] == 102_900_039
    assert closure["card_covered_candidate_topics"] == 4
    assert closure["uncovered_candidate_topics"] == 990
    assert closure["card_covered_candidate_rows"] == 236
    assert result["raw_openalex_main_table_read"] is False
    assert result["titles_read"] is False
    assert result["work_ids_materialized"] is False
    assert result["api_requests_submitted"] == 0
    assert result["complete_candidate_work_manifest_materialized"] is False
    assert result["adaptive_frontier_equivalence_proven"] is False
    assert result["deterministic_reselection_authorized"] is False


def test_small_replay_counts_overlapping_memberships() -> None:
    candidates = [
        {
            "field_id": 17,
            "field_candidate_rank": 1,
            "topic_id": 100,
            "topic_name": "Topic",
            "subfield_id": 1701,
            "entry_year": 2002,
        },
        {
            "field_id": 17,
            "field_candidate_rank": 2,
            "topic_id": 100,
            "topic_name": "Topic",
            "subfield_id": 1701,
            "entry_year": 2003,
        },
    ]
    topic_year = [
        {
            "field_id": 17,
            "topic_id": 100,
            "topic_name": "Topic",
            "subfield_id": 1701,
            "subfield_name": "Subfield",
            "publication_year": year,
            "author_linked_works": 1,
        }
        for year in range(1996, 2003)
    ]

    window, topics, closure = VERIFIER._replay(candidates, topic_year, {100})

    assert len(window) == 12
    assert len(topics) == 1
    assert closure["unique_candidate_topic_year_cells"] == 7
    assert closure["unique_candidate_window_linked_work_rows"] == 7
    assert closure["candidate_window_linked_work_memberships"] == 12
    assert closure["card_covered_candidate_rows"] == 2


def test_raw_scan_or_api_authorization_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["execution_boundary"]["raw_openalex_scan"] = "AUTHORIZED"
    document["execution_boundary"]["api_submission"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.CandidateScopeVerificationError,
        match="unsafe scope authorization",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_router_cannot_be_promoted_to_full_coverage(tmp_path: Path) -> None:
    document = _document()
    document["decision_rule"][
        "current_four_card_router_can_screen_complete_candidate_universe"
    ] = True
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.CandidateScopeVerificationError,
        match="router promoted",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    report = copied / "candidate_screen_scope_audit.json"
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    report.chmod(0o600)

    with pytest.raises(
        VERIFIER.CandidateScopeVerificationError,
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
        "scripts.build_pivot_feedback_d0_candidate_screen_scope_audit_v1"
        not in imports
    )
