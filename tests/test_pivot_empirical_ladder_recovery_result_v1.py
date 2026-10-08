from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_ladder_recovery_v1_20261008.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_identification_passed_reproducibly() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "af43b16722be63fb128614975614c387fab674f2"
    assert result["execution"]["byte_identical"] is True
    assert result["registered_result"]["disposition"] == "IDENTIFICATION_PASSES"


def test_role_counts_and_recovered_coefficients() -> None:
    worlds = _result()["registered_result"]["worlds"]
    for world in ("W1_learning", "W2_learning_confounded"):
        assert worlds[world]["nabm_wins"] == 8 and abs(worlds[world]["median_learned_coefficient"] - 1.0) < 0.05
    for world in ("W3_static", "W4_static_confounded", "W5_static_independent_drift"):
        assert worlds[world]["nabm_wins"] <= 1 and abs(worlds[world]["median_learned_coefficient"]) < 0.1
    assert abs(worlds["W1w_weak_learning"]["median_learned_coefficient"] - 0.5) < 0.05


def test_the_anticipation_limit_and_the_sign_test_are_recorded() -> None:
    result = _result()
    worlds = result["registered_result"]["worlds"]
    assert worlds["W6_static_aligned_drift"]["nabm_wins"] == 8
    assert worlds["W1_learning"]["sign_test_positive"] == 0 and worlds["W2_learning_confounded"]["sign_test_positive"] == 8
    assert "ANTICIPATORY" in result["next"]["N11a_v2"]
