from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v31.py"


def _load():
    spec = importlib.util.spec_from_file_location("gate_v31", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_v31_authorizes_only_unresolved_production_closeout() -> None:
    result = _load().validate()
    assert result["status"] == "PASS_RANK5_LOCAL_ABSTRACT_UNRESOLVED_PRODUCTION_CLOSEOUT_GATE"
    assert result["final_accept_rows"] == 28
    assert result["final_reject_rows"] == 22
    assert result["final_unresolved_rows"] == 14
    assert result["rank5_content_state_authorized"] == "CONTENT_UNRESOLVED"
    assert result["frontier_v8_authorized"] is True
    assert result["certificate_status_v3_blocked_authorized"] is True
    assert result["selection_certificate_authorized"] is False
    assert result["new_api_requests_authorized"] is False
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
