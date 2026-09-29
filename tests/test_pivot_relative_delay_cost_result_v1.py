from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_relative_delay_cost_v1_20260929.summary.json"
)


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def _cells() -> dict[tuple[float, str], dict]:
    return {(c["q"], c["visibility_id"]): c for c in _result()["registered_cells"]}


def test_result_is_precommitted_nested_and_reproduced() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == (
        "22605d51d778743d4436ad78065776bf3c9a192b"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 2304
    assert execution["deterministic_rerun"]["all_four_artifacts_byte_identical"] is True
    assert result["validation"]["nesting_checks"] == "PASS_EXACT"
    assert result["boundaries"]["model_selected"] == "NO"


def test_active_delays_grow_as_the_entry_wave_widens() -> None:
    test = _result()["registered_relative_delay_test"]
    assert test["active_delays_by_q"] == {
        "1.0": [0],
        "0.5": [0, 1],
        "0.25": [0, 1, 2, 4],
    }
    assert test["disposition"] == "NEITHER_SIMPLE_RULE"
    assert test["separable_by_relative_delay"] is False


def test_the_tree_carries_every_active_cell() -> None:
    for cell in _cells().values():
        if cell["status"] == "ENTRY_CHANNEL_ACTIVE":
            assert "tree_batch/axis_interaction_fit" in cell["passing_family_geometries"]
    mlp_cells = [
        key
        for key, cell in _cells().items()
        if any(pair.startswith("mlp_batch") for pair in cell["passing_family_geometries"])
    ]
    assert mlp_cells == [(0.25, "failure_lag_1")]


def test_post_hoc_normalizations_separate_but_are_flagged_as_hypotheses() -> None:
    result = _result()
    post = result["post_hoc_descriptive"]
    assert post["normalizations_chosen_after_opening_results"] is True
    wave = post["feature_arm_wave_separation"]
    assert wave["max_active"] < wave["min_inactive"]
    share = post["post_failure_entry_share_separation"]
    assert share["min_active"] > share["max_inactive"]
    assessment = result["scientific_assessment"]
    assert assessment["post_hoc_separations_are_hypotheses_not_confirmed"] is True
    assert assessment["fixed_rule_normalization_separates_cells"] is False


def test_cost_ledger_orders_pooled_below_per_agent_learning() -> None:
    records = {
        (r["arm_id"], r["evidence_regime"]): r for r in _result()["cost_ledger"]["records"]
    }
    assert records[("fixed_rule", "none")]["cost_relative_to_fixed_rule"] == 1.0
    for arm in ("linear_batch", "tree_batch", "mlp_batch"):
        pooled = records[(arm, "field_public")]["cost_relative_to_fixed_rule"]
        local = records[(arm, "neighbor_local")]["cost_relative_to_fixed_rule"]
        assert pooled < 10.0 < 40.0 < local
    counts = _result()["llm_reference_counts"]
    assert counts["policy_queries"] == 2304 * 256 * 24
    assert 0 < counts["pre_entry_decisions"] < counts["policy_queries"]
