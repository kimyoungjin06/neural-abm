from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_ladder_recovery_v2_20261008.summary.json"


def _worlds() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))["registered_result"]["worlds"]


def test_v2_identification_passed_reproducibly() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["precommit"]["commit_sha"] == "0cbe5f24db2ffed37f633a1716be24cd2deafed6"
    assert result["execution"]["byte_identical"] is True
    assert result["registered_result"]["disposition"] == "IDENTIFICATION_PASSES"


def test_anticipation_no_longer_wins_and_learning_beats_it() -> None:
    worlds = _worlds()
    assert worlds["W6_static_aligned_drift"]["nabm_wins"] == 0
    assert worlds["W6_static_aligned_drift"]["beats_b2"] == 8 and worlds["W6_static_aligned_drift"]["beats_anticipatory"] == 0
    for world in ("W1_learning", "W2_learning_confounded", "W1w_weak_learning", "W7_learning_with_anticipation"):
        assert worlds[world]["nabm_wins"] == 8 and worlds[world]["beats_anticipatory"] == 8


def test_the_no_learning_worlds_stay_within_the_limit() -> None:
    worlds = _worlds()
    for world in ("W3_static", "W4_static_confounded", "W5_static_independent_drift", "W6_static_aligned_drift"):
        assert worlds[world]["nabm_wins"] <= 1 and worlds[world]["meets_role"] is True
    assert worlds["W4_static_confounded"]["beats_best_non_learning"] == 5
