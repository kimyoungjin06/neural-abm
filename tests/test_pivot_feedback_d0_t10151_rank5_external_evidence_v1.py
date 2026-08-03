from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_t10151_rank5_external_evidence_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("external_evidence_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_evidence_packet_is_exact_and_identifier_free() -> None:
    result = _module().verify()
    assert (
        result["status"]
        == "PASS_INDEPENDENT_EXACT_14_EXTERNAL_EVIDENCE_EXACT_11_PACKET_VERIFICATION"
    )
    assert result["external_rows"] == 14
    assert result["review_packet_rows"] == 11
    assert result["holdout_rows"] == 3
    assert result["crossref_rows"] == 7
    assert result["pubmed_rows"] == 8
    assert result["semantic_labels_created"] is False
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
