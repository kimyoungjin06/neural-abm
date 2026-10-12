from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments" / "evidence" / "results" / "pivot_empirical_ladder_recovery_v3_20261012.summary.json"


def _result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _worlds() -> dict:
    return _result()["registered_result"]["worlds"]


def test_v3_is_underpowered_and_reproduced_byte_for_byte() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "288062a8df6eb6580659bf79d25ceadcd3d606f4"
    assert result["execution"]["byte_identical"] is True
    assert result["execution"]["tasks"] == 80
    assert result["registered_result"]["disposition"] == "UNDERPOWERED_AT_THE_EMPIRICAL_SHAPE"
    assert result["next"]["N11c"] == result["next"]["N11d"] == "NOT_PROPOSED"


def test_no_learning_world_false_recovers() -> None:
    worlds = _worlds()
    for world, row in worlds.items():
        if row["role"] == "MUST_NOT_FALSE_RECOVER":
            assert row["nabm_wins"] == 0 and row["meets_role"] is True, world
    assert worlds["W8_static_recruitment"]["wins_without_recruitment_censoring"] == 0


def test_the_learning_worlds_lose_to_the_anticipatory_rival() -> None:
    worlds = _worlds()
    assert worlds["W1_learning"]["nabm_wins"] == 1 and worlds["W2_learning_confounded"]["nabm_wins"] == 3
    for world in ("W1_learning", "W2_learning_confounded"):
        assert worlds[world]["meets_role"] is False
        assert worlds[world]["beats_b2"] > worlds["W1_learning"]["beats_anticipatory"]


def test_no_empirical_row_was_read() -> None:
    boundaries = _result()["boundaries"]
    assert boundaries["empirical_rows_read"] == boundaries["empirical_outcome_rows_read"] == 0
    assert boundaries["model_selected_for_data"] == "NO"
