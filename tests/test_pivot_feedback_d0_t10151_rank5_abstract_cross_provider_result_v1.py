from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_pivot_feedback_d0_t10151_rank5_abstract_cross_provider_result_v1.py"


def _load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("abstract_cross_provider_result", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_abstract_result_reduction_remains_unresolved() -> None:
    module = _load()
    config = module._yaml(module.DEFAULT_CONFIG)
    rows, candidate, summary = module.reduce(config)
    assert len(rows) == 5
    assert summary["dual_high_accept_rows"] == 1
    assert summary["dual_high_reject_rows"] == 1
    assert summary["reviewed_remains_unresolved_rows"] == 3
    assert candidate["final_accept_rows"] == 28
    assert candidate["final_reject_rows"] == 22
    assert candidate["final_unresolved_rows"] == 14
    assert candidate["candidate_content_state"] == "CONTENT_UNRESOLVED"
    assert candidate["human_review"] == "NOT_RUN_BY_DESIGN"
    assert candidate["production_disposition_created"] is False
