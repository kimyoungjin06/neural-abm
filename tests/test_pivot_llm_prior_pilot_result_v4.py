from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_llm_prior_pilot_v4_20260930.summary.json"
)
WORLDS = ("linear_fit", "smooth_interaction_fit", "axis_interaction_fit")
MODELS = ("qwen3_0_6b", "qwen3_1_7b", "qwen3_4b", "qwen3_4b_2507")


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_reproduced_and_makes_no_paid_calls() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "15dc4d186d1b4c709bdfddc601b086c2e17c26e2"
    execution = result["execution"]
    assert execution["raw_row_count"] == 12 * 3 * 4 * 16
    assert execution["deterministic_rerun"]["all_four_artifacts_byte_identical"] is True
    for model in MODELS:
        assert result["elicitation"][model]["elicitation_and_prior_byte_identical"] is True
    validation = result["validation"]
    assert validation["nesting_checks"] == "PASS_EXACT"
    assert validation["v3_reference_rebuilt_exactly"] is True
    assert result["boundaries"]["paid_api_calls"] == 0
    assert result["boundaries"]["package_llm_dependency"] is False


def test_the_prefilled_readout_is_valid_for_every_model_and_v1_fails_at_1_7b() -> None:
    models = _result()["phase_a"]["models"]
    for model in MODELS:
        assert models[model]["validity"]["median_answer_token_mass"] > 0.999
        assert models[model]["determinism"]["repeat_pass_identical"] is True
    assert models["qwen3_1_7b"]["v1_first_token_readout_validity"]["weak_format"] is True
    assert models["qwen3_4b_2507"]["spearman_readout_versus_v1_readout"] > 0.95


def test_registered_summaries_and_the_size_frontier() -> None:
    result = _result()
    disposition = result["registered_result_disposition"]
    expected = {
        "qwen3_0_6b": ("DOMINATES_IN_MINORITY", 4, 7, 5),
        "qwen3_1_7b": ("DOMINATES_IN_MINORITY", 1, 0, 5),
        "qwen3_4b": ("DOMINATES_IN_MAJORITY", 7, 9, 8),
    }
    for model, (summary, dominated, selecting, matching) in expected.items():
        entry = disposition[model]
        assert entry["hybrid_summary"] == summary
        assert len(entry["dominating_cells"]) == dominated
        assert len(entry["selecting_cells"]) == selecting
        assert len(entry["matching_cells"]) == matching
    assert len(disposition["qwen3_4b_2507"]["dominating_cells"]) == 8
    assert len(disposition["v3_reference"]["dominating_cells"]) == 6
    frontier = result["size_frontier"]
    assert frontier["smallest_same_release_model_dominating_in_majority"] == "qwen3_4b"
    assert frontier["status"] == "DESCRIPTIVE_FRONTIER_NOT_MODEL_SELECTION"
    assert result["v3_replication"] == "V3_HYBRID_20_MAJORITY_REPLICATED_ON_FRESH_SEEDS"


def test_phase_a_shows_the_0_6b_linear_alignment_and_the_flat_1_7b() -> None:
    result = _result()
    auc = {
        model: result["phase_a"]["models"][model]["calibrated_prior_design_cohort_auc"]
        for model in MODELS
    }
    hand = result["phase_a"]["hand_design_cohort_auc"]
    assert auc["qwen3_0_6b"]["linear_fit"] > 0.94
    assert max(auc["qwen3_0_6b"]["smooth_interaction_fit"], auc["qwen3_0_6b"]["axis_interaction_fit"]) < 0.53
    assert auc["qwen3_1_7b"]["linear_fit"] < 0.5
    assert auc["qwen3_4b"]["linear_fit"] > hand["linear_fit"] + 0.15
    profiles = result["condition_profiles"]
    for world in ("smooth_interaction_fit", "axis_interaction_fit"):
        # Slightly lower AUC than the hand rule, yet better selection at the top.
        assert auc["qwen3_4b"][world] < hand[world]
        selective = profiles[world]["selective_qwen3_4b"]["fixed_rule"]
        assert selective["productive_share_of_entrants"] > (
            profiles[world]["hand"]["fixed_rule"]["productive_share_of_entrants"] + 0.09
        )
    flat = result["post_hoc_descriptive"]["qwen3_1_7b_prior_is_nearly_flat"]
    assert flat["max_abs_weight"] < 0.14
    assert flat["fixed_rule_entry"] < 0.07


def test_cost_ledger_gives_a_light_model_floor_only_when_batched() -> None:
    result = _result()
    models = result["phase_a"]["models"]
    batch_32 = {model: models[model]["seconds_per_query"]["batch_32"] for model in MODELS}
    assert batch_32["qwen3_4b"] / batch_32["qwen3_0_6b"] > 4.5
    for model in MODELS:
        assert 0.025 < models[model]["seconds_per_query"]["batch_1"] < 0.05
        assert models[model]["one_time_elicitation_gpu_seconds"] < 50
    hours = result["cost_ledger"]["by_model"]
    assert hours["qwen3_0_6b"]["llm_agent_arm_gpu_hours"]["batch_32"] < 0.4
    assert 1.8 < hours["qwen3_4b"]["llm_agent_arm_gpu_hours"]["batch_32"] < 1.9


def test_post_hoc_checks_are_labelled_and_hold() -> None:
    post = _result()["post_hoc_descriptive"]
    assert post["status"] == "POST_HOC_DESCRIPTIVE_NOT_REGISTERED_NO_NEW_EXECUTION"
    rho = post["phase_a_auc_predicts_abm_selection"]["spearman_design_auc_vs_fixed_rule_selective_precision"]
    assert rho > 0.85
    erosion = post["pooled_learning_erodes_the_0_6b_linear_world_selection"]
    assert erosion["fixed_rule_precision"] > 0.76
    assert max(erosion["field_precision"].values()) < 0.56
    signs = post["weights_versus_the_designer_linear_fit"]["sign_and_largest_weight_by_source"]
    assert signs["qwen3_0_6b"] == {
        "all_six_signs_match": True,
        "largest_abs_weight_feature": "field_opportunity",
    }
    assert signs["qwen3_4b"]["all_six_signs_match"] is True


def test_assessment_selects_nothing_and_claims_no_real_knowledge() -> None:
    result = _result()
    assessment = result["scientific_assessment"]
    assert assessment["model_selected"] == "NO"
    assert assessment["llm_knowledge_of_real_pivots"] == "NOT_ESTABLISHED"
    assert assessment["size_effect"].startswith("NOT_MONOTONE")
    assert result["framework_assessment"]["generic_api_change_needed"] is False
