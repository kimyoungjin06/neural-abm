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
    / "experiments/configs/pivot_feedback_d0_adaptive_candidate_frontier_v1.yaml"
)
OUTPUT = Path("/ssd/pivot-feedback-d0-adaptive-candidate-frontier-v1-20260729")
PRODUCER_SCRIPT = (
    ROOT / "scripts/build_pivot_feedback_d0_adaptive_candidate_frontier_v1.py"
)
VERIFIER_SCRIPT = (
    ROOT / "scripts/verify_pivot_feedback_d0_adaptive_candidate_frontier_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VERIFIER = _load(VERIFIER_SCRIPT, "pivot_feedback_adaptive_frontier_verifier_test")


def _candidates() -> list[dict[str, int]]:
    return [
        {"rank": 1, "subfield_id": 1, "entry_year": 0},
        {"rank": 2, "subfield_id": 1, "entry_year": 9},
        {"rank": 3, "subfield_id": 2, "entry_year": 9},
        {"rank": 4, "subfield_id": 2, "entry_year": 9},
    ]


def _document() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_current_adaptive_frontier_replays() -> None:
    result = VERIFIER.validate(CONFIG, OUTPUT)

    assert result["status"] == VERIFIER.VERIFY_STATUS
    assert result["proof"]["total_fixture_cases"] == 299_584
    assert result["proof"]["eager_pair_cases"] == 118_404
    assert (
        result["proof"]["required_state_unresolved_mutation_cases"] == 331_424
    )
    assert (
        result["proof"]["structurally_incompatible_not_evaluated_cases"]
        == 146_832
    )
    assert result["frontier"]["total_frontier_rows"] == 51
    assert result["frontier"]["total_content_required_rows"] == 4
    assert result["frontier"]["total_structurally_incompatible_rows"] == 47
    assert result["frontier"]["initial_selection_certificate_issued"] is False
    assert result["raw_openalex_main_table_read"] is False
    assert result["titles_read"] is False
    assert result["api_requests_submitted"] == 0
    assert result["content_dispositions_created"] is False
    assert result["deterministic_reselection_executed"] is False


def test_lazy_certificate_matches_eager_pair() -> None:
    candidates = _candidates()
    truth = [
        "CONTENT_PASS",
        "CONTENT_PASS",
        "CONTENT_FAIL",
        "CONTENT_PASS",
    ]
    observed = [
        "CONTENT_PASS",
        "NOT_EVALUATED",
        "CONTENT_FAIL",
        "CONTENT_PASS",
    ]

    assert VERIFIER.eager_select(candidates, truth) == (1, 4)
    assert VERIFIER.issue_lazy_certificate(candidates, observed) == (1, 4)


def test_unresolved_required_candidate_blocks_certificate() -> None:
    candidates = _candidates()
    states = [
        "CONTENT_PASS",
        "NOT_EVALUATED",
        "CONTENT_UNRESOLVED",
        "CONTENT_PASS",
    ]

    assert VERIFIER.issue_lazy_certificate(candidates, states) is None


def test_structurally_incompatible_candidate_need_not_be_evaluated() -> None:
    candidates = _candidates()
    states = [
        "CONTENT_PASS",
        "NOT_EVALUATED",
        "CONTENT_FAIL",
        "CONTENT_PASS",
    ]
    incompatible_promoted = list(states)
    incompatible_promoted[1] = "CONTENT_PASS"

    assert VERIFIER.issue_lazy_certificate(candidates, states) == (1, 4)
    assert (
        VERIFIER.issue_lazy_certificate(candidates, incompatible_promoted)
        == (1, 4)
    )


def test_api_or_content_disposition_authorization_fails_closed(
    tmp_path: Path,
) -> None:
    document = _document()
    document["authorization"]["candidate_content_dispositions"] = "AUTHORIZED"
    document["authorization"]["api_submission"] = "AUTHORIZED"
    mutated = tmp_path / "mutated.yaml"
    mutated.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.AdaptiveFrontierVerificationError,
        match="unsafe authorization",
    ):
        VERIFIER.validate(mutated, OUTPUT)


def test_artifact_tampering_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / OUTPUT.name
    shutil.copytree(OUTPUT, copied)
    proof = copied / "adaptive_frontier_equivalence_proof.json"
    proof.write_text(proof.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    proof.chmod(0o600)

    with pytest.raises(
        VERIFIER.AdaptiveFrontierVerificationError,
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
        "scripts.build_pivot_feedback_d0_adaptive_candidate_frontier_v1"
        not in imports
    )
