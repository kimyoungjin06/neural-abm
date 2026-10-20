from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "info_spread_policy_ladder_d2_diagnosis_20261014.summary.json"


def _groups() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))["groups"]


def test_ensembling_helps_before_the_shift_everywhere() -> None:
    for group in _groups().values():
        assert group["contrasts"]["T0_1 - T0_4 (ensembling)"]["pre_shift"]["mean"] < 0


def test_the_closed_loop_at_matched_evidence_is_mostly_indistinguishable() -> None:
    covering = 0
    for group in _groups().values():
        contrast = group["contrasts"]["L0_full - T0_1 (closed loop at matched evidence)"]
        for cohort in ("pre_shift", "post_shift"):
            low, high = contrast[cohort]["interval_95"]
            covering += low <= 0 <= high
    assert covering == 14


def test_forgetting_pays_after_the_flip_for_mlps_only() -> None:
    groups = _groups()
    forgetting = "L0_window - L0_full (forgetting)"
    for regime in ("A", "B"):
        assert groups[f"POS_flip/{regime}/mlp"]["contrasts"][forgetting]["post_shift"]["interval_95"][0] > 0
        assert groups[f"POS_flip/{regime}/mlp"]["contrasts"][forgetting]["pre_shift"]["mean"] < 0
