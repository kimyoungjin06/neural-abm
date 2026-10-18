from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_info_spread_policy_ladder_study_v2 as study  # noqa: E402


def _contract() -> dict:
    return study.load_contract()


def test_the_contract_is_a_precommit_bound_to_its_sources() -> None:
    contract = _contract()
    assert contract["status"] == "PRECOMMIT_DESIGN_NO_RESULT_INTERPRETATION"
    assert len(study.verify_source_bindings(contract)) == len(contract["source_bindings"])
    assert set(contract["precommit"]["precommit_paths"]) == {
        "docs/research/info-spread-policy-ladder-study-v2.md",
        "experiments/configs/info_spread_policy_ladder_study_v2.yaml",
        "scripts/run_info_spread_policy_ladder_study_v2.py",
        "tests/test_info_spread_policy_ladder_study_v2.py",
    }
    assert contract["measures"]["primary"] == "unverified_share_precision"
    assert contract["boundaries"]["llm_calls"] == 0


def test_the_world_seeds_are_fresh() -> None:
    import run_info_spread_policy_ladder_study_v1 as v1

    v2_seeds, v1_seeds = _contract()["seeds"], v1.load_contract()["seeds"]
    assert v2_seeds["world_seed_offset"] == 7000 != v1_seeds["world_seed_offset"]
    assert v2_seeds["learner_tag"] != v1_seeds["learner_tag"] and v2_seeds["rule_tag"] != v1_seeds["rule_tag"]


def test_a_development_run_needs_a_full_sha() -> None:
    with pytest.raises(RuntimeError):
        study.verify_precommit("abc123", _contract())


def _table(a_positive: int, b_positive: int, null_positive: bool = False) -> dict:
    table = {}
    contract = _contract()
    for regime, count in (("A", a_positive), ("B", b_positive)):
        study_groups = [f"{regime}/{family}/{geometry}" for family in contract["design"]["families"][regime]
                        for geometry in contract["design"]["study_geometries"]]
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
    [(0, 4, "AS_EXPECTED"), (0, 1, "TABLE_SUFFICES_EVEN_IN_B"), (5, 3, "CLOSED_LOOP_MATTERS_EVEN_IN_A"),
     (2, 3, "MIXED")],
)
def test_the_registered_dispositions(a: int, b: int, expected: str) -> None:
    result = study.dispositions(_contract(), _table(a, b))
    assert result["closed_loop"] == expected
    assert result["design_check"] == "NULL_WORLD_PASSES"
    assert study.dispositions(_contract(), _table(a, b, null_positive=True))["design_check"] == (
        "DESIGN_DEFECT_IN_NULL_WORLD")


def test_a_smoke_cell_runs_deterministically_and_is_judged_on_the_unverified_measure() -> None:
    contract = _contract()
    study.apply_learner_settings(contract)
    cells = []
    for regime in contract["design"]["regimes"]:
        for family in contract["design"]["families"][regime]:
            first = study.run_cell(contract, "smoke", regime, family, "null", 0, "cpu")
            second = study.run_cell(contract, "smoke", regime, family, "null", 0, "cpu")
            assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
            assert set(first["rows"][0]["arms"]) == set(study.ARMS)
            assert "seconds" not in json.dumps(first)
            cells.append(first)
    summary = study.summarize(contract, cells)
    entry = summary["table"]["B/partial_mlp/null"]
    assert set(entry) >= {"unverified_share_precision", "share_precision", "mean_unverified_share_precision"}
    expected = {key: value["unverified_share_precision"] for key, value in summary["table"].items()}
    assert summary["dispositions"] == study.dispositions(contract, expected)
