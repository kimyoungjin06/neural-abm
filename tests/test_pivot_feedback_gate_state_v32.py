from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_gate_state_v32.py"


def _module():
    spec = importlib.util.spec_from_file_location("gate_v32", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gate_v32_authorizes_exact_external_collection_without_production() -> None:
    result = _module().validate()
    assert result["status"] == "PASS_RANK5_EXACT_14_EXTERNAL_COLLECTION_EXACT_11_REVIEW_GATE"
    assert result["external_collection_rows_authorized"] == 14
    assert result["external_review_rows_authorized"] == 11
    assert result["holdout_rows"] == 3
    assert result["provider_calls_authorized"] == 2
    assert result["retries_authorized"] == 0
    assert result["production_disposition_authorized"] is False
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
