from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SCRIPT = SCRIPTS / "verify_pivot_feedback_d0_t10151_rank5_external_cross_provider_v1.py"


def _module():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("external_cross_provider_v1", SCRIPT)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


def test_external_cross_provider_result_reconstructs_from_raw_responses() -> None:
    result = _module().verify()
    assert (
        result["status"]
        == "PASS_INDEPENDENT_EXACT_11_EXTERNAL_CROSS_PROVIDER_RESULT_VERIFICATION"
    )
    assert result["provider_calls"] == 2
    assert result["retries"] == 0
    assert result["dual_high_accept_rows"] == 2
    assert result["dual_high_reject_rows"] == 3
    assert result["reviewed_unresolved_rows"] == 6
    assert result["final_accept_rows"] == 30
    assert result["final_reject_rows"] == 25
    assert result["final_unresolved_rows"] == 9
    assert result["candidate_content_state"] == "CONTENT_UNRESOLVED"
