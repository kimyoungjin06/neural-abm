from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v30.py"


def _load():
    spec = importlib.util.spec_from_file_location("gate_v30", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_v30_authorizes_only_local_abstract_recovery_and_five_row_review() -> None:
    result = _load().validate()
    assert result["status"] == "PASS_RANK5_LOCAL_ABSTRACT_RECOVERY_AND_FIVE_ROW_REVIEW_GATE"
    assert result["frozen_unresolved_ids"] == 16
    assert result["expected_reconstructable_abstracts"] == 6
    assert result["eligible_review_rows_authorized"] == 5
    assert result["provider_calls_authorized"] == 2
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
    assert result["production_disposition_authorized"] is False
    assert result["selection_certificate_authorized"] is False
    assert result["outcomes_read_authorized"] is False
    assert result["g4_open"] is False
