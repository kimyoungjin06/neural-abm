from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_rank5_local_abstract_closeout_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("rank5_local_abstract_closeout_verify", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_abstract_closeout_is_independently_verified() -> None:
    result = _load().validate()
    assert result["status"] == "PASS_INDEPENDENT_RANK5_LOCAL_ABSTRACT_UNRESOLVED_TERMINAL_CLOSEOUT"
    assert result["disposition_rows_verified"] == 8
    assert result["frontier_rows_verified"] == 51
    assert result["final_accept_rows"] == 28
    assert result["final_reject_rows"] == 22
    assert result["final_unresolved_rows"] == 14
    assert result["rank5_content_state"] == "CONTENT_UNRESOLVED"
    assert result["local_abstract_evidence_used"] is True
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
    assert result["medicine_field_certificate_issued"] is False
    assert result["global_selection_certificate_issued"] is False
