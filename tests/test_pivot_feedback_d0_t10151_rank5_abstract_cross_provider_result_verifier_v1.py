from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_v1.py"


def _load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("abstract_cross_provider_result_verify", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abstract_result_is_independently_verified() -> None:
    result = _load().validate()
    assert result["status"] == "PASS_INDEPENDENT_RANK5_ABSTRACT_CROSS_PROVIDER_RESULT_VERIFICATION"
    assert result["dual_high_accept_rows"] == 1
    assert result["dual_high_reject_rows"] == 1
    assert result["reviewed_unresolved_rows"] == 3
    assert result["unavailable_or_fragment_rows"] == 11
    assert result["final_accept_rows"] == 28
    assert result["final_reject_rows"] == 22
    assert result["final_unresolved_rows"] == 14
    assert result["final_content_state"] == "CONTENT_UNRESOLVED"
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
    assert result["production_disposition_authorized"] is False
