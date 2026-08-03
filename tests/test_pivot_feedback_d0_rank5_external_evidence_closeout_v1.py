from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_rank5_external_evidence_closeout_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("external_closeout_v1", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_closeout_remains_blocked_and_preserves_cs_certificate() -> None:
    result = _module().verify()
    assert (
        result["status"]
        == "PASS_INDEPENDENT_RANK5_EXTERNAL_EVIDENCE_UNRESOLVED_TERMINAL_CLOSEOUT"
    )
    assert result["rank5_counts"] == {"accept": 30, "reject": 25, "unresolved": 9}
    assert result["rank5_content_state"] == "CONTENT_UNRESOLVED"
    assert result["computer_science_selected_candidate_ranks"] == [1, 45]
    assert result["medicine_field_certificate_issued"] is False
    assert result["global_selection_certificate_issued"] is False
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
