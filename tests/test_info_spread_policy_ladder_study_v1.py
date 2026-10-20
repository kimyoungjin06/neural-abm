from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_info_spread_policy_ladder_study_v1 as study  # noqa: E402


def _contract() -> dict:
    return study.load_contract()


def test_the_contract_is_a_precommit_bound_to_its_sources() -> None:
    contract = _contract()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(study.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert set(contract["precommit"]["precommit_paths"]) == {
        "docs/research/info-spread-policy-ladder-study-v1.md",
        "experiments/configs/info_spread_policy_ladder_study_v1.yaml",
        "scripts/run_info_spread_policy_ladder_study_v1.py",
        "tests/test_info_spread_policy_ladder_study_v1.py",
    }
    assert contract["boundaries"]["llm_calls"] == 0


def test_a_development_run_needs_a_full_sha() -> None:
    with pytest.raises(RuntimeError):
        study.verify_precommit("abc123", _contract())


def _table(a_positive: int, b_positive: int, null_positive: bool = False) -> dict:
    table = {}
    contract = _contract()
    for regime, count in (("A", a_positive), ("B", b_positive)):
        study_groups = [
            f"{regime}/{family}/{geometry}"
            for family in contract["design"]["families"][regime]
            for geometry in contract["design"]["study_geometries"]
        ]
        for index, key in enumerate(study_groups):
            material = index < count
            table[key] = {
                "closed_loop": {"material": material, "mean_cell_delta": 0.05 if material else 0.0},
                "best_table": {"material": False, "mean_cell_delta": 0.0},
                "hand_rule": {"material": False, "mean_cell_delta": 0.0},
            }
        for family in contract["design"]["families"][regime]:
            table[f"{regime}/{family}/null"] = {
                "closed_loop": {"material": False, "mean_cell_delta": 0.0},
                "best_table": {"material": False, "mean_cell_delta": 0.0},
                "hand_rule": {"material": null_positive, "mean_cell_delta": 0.05 if null_positive else 0.0},
            }
    return table


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [(0, 4, "AS_EXPECTED"), (1, 6, "AS_EXPECTED"), (0, 1, "TABLE_SUFFICES_EVEN_IN_B"),
     (5, 3, "CLOSED_LOOP_MATTERS_EVEN_IN_A"), (2, 3, "MIXED")],
)
def test_the_registered_dispositions(a: int, b: int, expected: str) -> None:
    result = study.dispositions(_contract(), _table(a, b))
    assert result["closed_loop"] == expected
    assert result["design_check"] == "NULL_WORLD_PASSES"


def test_a_learner_that_beats_the_hand_rule_in_the_null_world_is_a_defect() -> None:
    result = study.dispositions(_contract(), _table(0, 4, null_positive=True))
    assert result["design_check"] == "DESIGN_DEFECT_IN_NULL_WORLD"


def test_a_smoke_cell_runs_deterministically_for_every_family() -> None:
    contract = _contract()
    study.apply_learner_settings(contract)
    cells = []
    for regime in contract["design"]["regimes"]:
        for family in contract["design"]["families"][regime]:
            first = study.run_cell(contract, "smoke", regime, family, "linear", 0, "cpu")
            second = study.run_cell(contract, "smoke", regime, family, "linear", 0, "cpu")
            assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
            assert set(first["rows"][0]["arms"]) == set(study.ARMS)
            assert "seconds" not in json.dumps(first)  # timing never enters cells.jsonl
            cells.append(first)
    summary = study.summarize(contract, cells)
    assert set(summary["dispositions"]) >= {"closed_loop", "design_check", "best_table_counts"}
