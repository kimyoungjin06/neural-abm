from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_null_leak_diagnosis_20261018.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_the_diagnosis_is_exploratory_and_reads_no_study_cell() -> None:
    result = _result()
    assert result["status"] == "EXPLORATORY_DIAGNOSIS_NOT_REGISTERED_NULL_CELLS_ONLY"
    assert result["boundaries"]["study_cells_read"] is False
    assert hashlib.sha256((ROOT / result["code"]).read_bytes()).hexdigest() == result["code_sha256"]


def test_the_mlp_gain_is_confined_to_items_with_a_visible_fact_check() -> None:
    groups = _result()["per_cell_delta_vs_f1"]
    for arm in ("L0-F1", "L0_no_trust-F1"):
        entry = groups["B/partial_mlp/null"][arm]
        assert entry["visible"]["positive_cells"] == 6 and entry["visible"]["mean"] > 0.05
        assert abs(entry["unrevealed"]["mean"]) < 0.01
    assert abs(groups["B/partial_linear_exact/null"]["L0-F1"]["visible"]["mean"]) < 0.01
