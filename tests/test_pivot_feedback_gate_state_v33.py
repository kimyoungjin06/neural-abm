from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v33.py"


def _module():
    spec = importlib.util.spec_from_file_location("gate_v33", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_v33_authorizes_only_unresolved_external_closeout() -> None:
    result = _module().validate()
    assert result["status"] == "PASS_RANK5_EXTERNAL_EVIDENCE_UNRESOLVED_PRODUCTION_CLOSEOUT_GATE"
    assert result["final_accept_rows"] == 30
    assert result["final_reject_rows"] == 25
    assert result["final_unresolved_rows"] == 9
    assert result["rank5_content_state_authorized"] == "CONTENT_UNRESOLVED"
    assert result["frontier_v9_authorized"] is True
    assert result["selection_certificate_authorized"] is False
    assert result["new_api_requests_authorized"] is False
