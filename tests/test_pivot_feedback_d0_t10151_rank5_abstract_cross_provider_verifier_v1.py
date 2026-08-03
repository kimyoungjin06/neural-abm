from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("abstract_cross_provider_verify", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepared_abstract_pair_is_independently_verified() -> None:
    result = _load().validate()
    assert result["status"] == "PASS_INDEPENDENT_EXACT_FIVE_LOCAL_ABSTRACT_REQUEST_VERIFICATION"
    assert result["request_rows"] == 5
    assert result["provider_calls"] == 2
    assert result["visible_payload_identical"] is True
    assert result["forbidden_fields_absent"] is True
    assert result["within_authorized_caps"] is True
    assert float(result["cumulative_upper_bound_usd"]) <= 0.10
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
    assert result["production_disposition_authorized"] is False
