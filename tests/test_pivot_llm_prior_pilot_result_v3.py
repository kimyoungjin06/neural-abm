from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_llm_prior_pilot_v3_20260929.summary.json"
)
WORLDS = ("linear_fit", "smooth_interaction_fit", "axis_interaction_fit")
FIELD_ARMS = ("linear_batch", "tree_batch", "mlp_batch")
CELLS = [f"{world}/{arm}" for world in WORLDS for arm in FIELD_ARMS]


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_reproduced_and_makes_no_llm_calls() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == (
        "87c90eb24b51e0d8bc472a11d8501f5772dffbfc"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 6 * 3 * 4 * 16
    assert execution["new_llm_calls"] == 0
    assert execution["deterministic_rerun"]["all_four_artifacts_byte_identical"] is True
    assert execution["first_run"]["receipt_sha256"] == (
        execution["deterministic_rerun"]["receipt_sha256"]
    )
    assert result["validation"]["nesting_checks"] == "PASS_EXACT"
    assert result["boundaries"]["paid_api_calls"] == 0


def test_epsilons_hit_the_registered_exploratory_shares() -> None:
    epsilons = _result()["validation"]["epsilon_by_condition"]
    for condition, share in (("hybrid_10", 0.1), ("hybrid_20", 0.2), ("hybrid_40", 0.4)):
        assert abs(1.0 - (1.0 - epsilons[condition]) ** 16 - share) < 1e-12
    assert epsilons["hand"] == epsilons["llm_selective"] == epsilons["flat"] == 0.0


def test_registered_summaries_by_condition() -> None:
    disposition = _result()["registered_result_disposition"]
    expected = {
        "llm_selective": ("DOMINATES_IN_MINORITY", 3),
        "hybrid_10": ("DOMINATES_IN_MAJORITY", 5),
        "hybrid_20": ("DOMINATES_IN_MAJORITY", 7),
        "hybrid_40": ("DOMINATES_NOWHERE", 0),
        "flat": ("DOMINATES_NOWHERE", 0),
    }
    for condition, (summary, count) in expected.items():
        assert disposition[condition]["summary"] == summary
        assert len(disposition[condition]["dominating_cells"]) == count
    assert sorted(set(CELLS) - set(disposition["hybrid_20"]["dominating_cells"])) == [
        "axis_interaction_fit/linear_batch",
        "linear_fit/linear_batch",
    ]


def test_hybrid_20_selects_better_everywhere_and_misses_only_on_linear_learning() -> None:
    cells = _result()["registered_decisions"]["hybrid_20"]["cells"]
    for cell in CELLS:
        precision = cells[cell]["precision_versus_hand"]
        assert precision["pass"] is True
        assert precision["improvement"]["mean"] > 0.026
        assert precision["positive_replicates"] >= 13
    for cell in ("axis_interaction_fit/linear_batch", "linear_fit/linear_batch"):
        learning = cells[cell]["learning_noninferior_versus_hand"]
        assert learning["pass"] is False
        assert -0.008 < learning["hand_minus_condition_probe_brier"]["mean"] < -0.005


def test_the_exploratory_share_trades_precision_for_learning() -> None:
    disposition = _result()["registered_result_disposition"]
    for cell in CELLS:
        assert disposition["llm_selective"]["cells"][cell] in {
            "HYBRID_DOMINATES_HAND",
            "SELECTION_ONLY",
        }
        assert disposition["hybrid_40"]["cells"][cell] in {"LEARNING_ONLY", "NEITHER"}
        assert disposition["flat"]["cells"][cell] in {"LEARNING_ONLY", "NEITHER"}
    decisions = _result()["registered_decisions"]
    flat_precision = [
        decisions["flat"]["cells"][cell]["precision_versus_hand"] for cell in CELLS
    ]
    assert all(entry["positive_replicates"] == 0 for entry in flat_precision)
    monotone = _result()["post_hoc_descriptive"]
    assert monotone["cells_with_precision_falling_and_evidence_rising"] == 9
    assert monotone["cells_with_probe_brier_falling"] == 7


def test_restoration_against_selective_is_in_interaction_world_nonlinear_learners() -> None:
    restored = _result()["post_hoc_descriptive"]["restoration_passes_by_condition"]
    assert restored["hybrid_20"] == [
        "axis_interaction_fit/mlp_batch",
        "axis_interaction_fit/tree_batch",
        "smooth_interaction_fit/mlp_batch",
    ]
    for cells in restored.values():
        assert all(not cell.startswith("linear_fit") for cell in cells)
        assert all(not cell.endswith("linear_batch") for cell in cells)


def test_linear_learner_in_the_linear_world_is_never_noninferior() -> None:
    post = _result()["post_hoc_descriptive"]
    gaps = post["linear_learner_in_linear_world_hand_minus_condition_probe_brier"]
    assert all(gap < -0.005 for gap in gaps.values())
    assert min(gaps, key=gaps.get) == "flat"
    assert post["status"] == "POST_HOC_DESCRIPTIVE_NOT_REGISTERED_NO_NEW_EXECUTION"


def test_profiles_show_realized_exploration_and_evidence_ordering() -> None:
    profiles = _result()["condition_profiles"]
    for world in WORLDS:
        mlp = {condition: profiles[world][condition]["mlp_batch"] for condition in profiles[world]}
        assert mlp["hand"]["exploratory_entry_rate"] == 0.0
        assert 0.05 < mlp["hybrid_10"]["exploratory_entry_rate"] < 0.06
        assert 0.10 < mlp["hybrid_20"]["exploratory_entry_rate"] < 0.12
        assert 0.20 < mlp["hybrid_40"]["exploratory_entry_rate"] < 0.21
        evidence = [
            mlp[condition]["evidence_events_per_learner"]
            for condition in ("llm_selective", "hybrid_10", "hybrid_20", "hybrid_40", "flat")
        ]
        assert evidence == sorted(evidence)
        assert mlp["flat"]["cumulative_entry_rate"] == 1.0


def test_assessment_selects_nothing_and_claims_no_real_knowledge() -> None:
    result = _result()
    assessment = result["scientific_assessment"]
    assert assessment["no_share_selected"] is True
    assert assessment["model_selected"] == "NO"
    assert assessment["llm_knowledge_of_real_pivots"] == "NOT_ESTABLISHED"
    framework = result["framework_assessment"]
    assert framework["generic_api_change_needed"] is False
    assert framework["added_llm_cost"] == 0
