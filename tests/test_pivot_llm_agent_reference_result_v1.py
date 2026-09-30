from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_llm_agent_reference_v1_20260930.summary.json"
)
WORLDS = ("linear_fit", "smooth_interaction_fit", "axis_interaction_fit")
FIELD_ARMS = ("linear_batch", "tree_batch", "mlp_batch")
CELLS = [f"{world}/{arm}" for world in WORLDS for arm in FIELD_ARMS]


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_reproduced_and_makes_no_paid_calls() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "2a25ab562e160a1e51f8eab6b10a214591574fc5"
    execution = result["execution"]
    assert execution["raw_row_count"] == 3 * 16 * (4 + 4 + 1)
    rerun = execution["deterministic_rerun"]
    assert rerun["all_four_reproducible_artifacts_byte_identical"] is True
    assert rerun["cost_timing_differs_as_registered"] is True
    validation = result["validation"]
    assert validation["nesting_checks"] == "PASS_EXACT"
    assert validation["llm_machinery_nests_the_fixed_rule_in_every_linear_world_replicate"] is True
    assert validation["repeated_llm_forward_passes_identical"] is True
    assert result["boundaries"]["paid_api_calls"] == 0


def test_calibration_is_aggregate_only_and_hits_the_stated_rate() -> None:
    calibration = _result()["llm_calibration"]
    assert abs(calibration["achieved_mean"] - 0.30) < 1e-9
    assert calibration["raw_mean_probability"] < 0.01
    assert 12.0 < calibration["calibration_shift"] < 13.0
    assert calibration["validity"]["weak_format"] is False


def test_nabm_matches_the_llm_agent_everywhere_and_is_mostly_superior() -> None:
    disposition = _result()["registered_result_disposition"]["nabm_versus_llm_agent"]
    assert disposition["summary"] == "NABM_MATCHES_LLM_AGENT_IN_ALL_CELLS"
    assert sorted(disposition["matching_cells"]) == sorted(CELLS)
    assert sorted(disposition["nabm_learning_superior_cells"]) == sorted(CELLS)
    assert set(CELLS) - set(disposition["nabm_precision_superior_cells"]) == {
        "axis_interaction_fit/linear_batch"
    }
    for cell in disposition["cells"].values():
        assert cell["nabm_minus_llm_precision_mean"] > 0.0
        assert cell["llm_minus_nabm_probe_brier_mean"] > 0.04


def test_the_llm_agent_beats_the_hand_rule_nowhere() -> None:
    disposition = _result()["registered_result_disposition"]["llm_agent_versus_hand"]
    assert disposition["summary"] == "DOMINATES_NOWHERE"
    assert disposition["selection_only_cells"] == [
        "axis_interaction_fit/linear_batch",
        "linear_fit/tree_batch",
    ]
    for cell in disposition["cells"].values():
        assert cell["hand_minus_llm_probe_brier_mean"] < -0.04


def test_post_hoc_drift_and_in_context_ranking() -> None:
    post = _result()["post_hoc_descriptive"]
    assert post["status"] == "POST_HOC_DESCRIPTIVE_NOT_REGISTERED_NO_NEW_EXECUTION"
    for world in WORLDS:
        drift = post["calibration_drift_after_evidence"][world]
        assert drift["llm_agent_mean_output_from_step_4"]["min"] > drift["population_productive_rate"] + 0.08
        assert drift["hybrid_mlp_mean_output_range"]["max"] < 0.32
        ranking = post["ranking_in_context"][world]
        assert ranking["llm_agent_probe_auc"] > 0.88
    assert "NEAREST_NEIGHBOUR" in post["ranking_caveat"]


def test_cost_shows_the_llm_agent_needs_gpu_every_run() -> None:
    cost = _result()["cost"]
    assert cost["status"].startswith("TIMING_NOT_BYTE_REPRODUCIBLE")
    gpu = cost["llm_agent_median_gpu_seconds_per_run"]
    cpu = cost["nabm_median_cpu_seconds_per_arm_run"]
    assert 100 < gpu["development"] < 120 and 100 < gpu["reproduction"] < 120
    assert cpu["development"] < 6 and cpu["reproduction"] < 6
    assert cost["nabm_hybrid_one_time_llm_gpu_seconds"] < 50
    floor = cost["cost_floor_from_v4_batch_32"]["qwen3_0_6b"]["gpu_seconds_per_run_at_this_query_count"]
    assert floor > 2 * cpu["development"]
    assert "MIXES_HARDWARE" in cost["hardware_note"]


def test_assessment_bounds_the_llm_agent_claim() -> None:
    assessment = _result()["scientific_assessment"]
    assert assessment["llm_agent_design"] == "ONE_REASONABLE_IN_CONTEXT_DESIGN_NOT_THE_BEST_POSSIBLE_LLM_AGENT"
    assert assessment["model_selected"] == "NO"
    assert assessment["llm_knowledge_of_real_pivots"] == "NOT_ESTABLISHED"
