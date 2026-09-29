from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_llm_prior_pilot_v1_20260929.summary.json"
)
WORLDS = ("linear_fit", "smooth_interaction_fit", "axis_interaction_fit")


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_local_precommitted_and_reproduced() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == (
        "75fbbf7e7f6b0131bd8b13cf7392330187543f2d"
    )
    elicitation = result["elicitation"]
    assert elicitation["model_id"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert elicitation["reproduction_identical"] is True
    assert elicitation["determinism"]["repeat_pass_identical"] is True
    assert elicitation["validity"]["weak_format"] is False
    assert result["execution"]["deterministic_rerun"]["all_four_artifacts_byte_identical"] is True
    assert result["validation"]["nesting_checks"] == "PASS_EXACT"
    assert result["boundaries"]["paid_api_calls"] == 0


def test_hand_prior_wins_every_registered_decision() -> None:
    for world in WORLDS:
        disposition = _result()["registered_result_disposition"][world]
        assert disposition["prior_quality"] == "HAND_PRIOR_BETTER"
        assert disposition["local_rescue"] == "HAND_PRIOR_BETTER_FOR_LOCAL_LEARNING"
        assert disposition["pooled_washout"] == "PRIOR_PERSISTS_UNDER_POOLING"
        assert disposition["entry_precision"] == "HAND_PRIOR_SELECTS_BETTER_ENTRANTS"


def test_the_llm_prior_is_self_sealing() -> None:
    post = _result()["post_hoc_descriptive"]
    assert post["llm_prior_max_entry_rate"] == 0.0
    assert set(post["llm_field_evidence_per_learner_max"].values()) == {0.0}


def test_llm_ranks_like_the_hand_rule_but_is_miscalibrated() -> None:
    ranking = _result()["post_hoc_descriptive"]["design_cohort_ranking_and_calibration"]
    for world in WORLDS:
        entry = ranking[world]
        assert entry["distilled"]["auc"] >= entry["hand"]["auc"] - 0.01
        assert entry["distilled"]["mean_prediction"] < 0.01
        assert entry["primary"]["mean_prediction"] < 0.05
        assert abs(entry["hand"]["mean_prediction"] - entry["base_rate"]) < 0.1
    assert ranking["linear_fit"]["distilled"]["auc"] > ranking["linear_fit"]["hand"]["auc"]


def test_cost_ledger_contrasts_one_time_elicitation_with_per_step_queries() -> None:
    cost = _result()["cost_ledger"]
    assert cost["phase_b_agent_steps"] == 960 * 256 * 18
    assert cost["llm_queries_used_by_the_hybrid"] < 5000
    assert cost["llm_agent_equivalent_gpu_hours_lower_bound"]["batched_32"] > 30.0
    elicitation = cost["elicitation_gpu_seconds"]
    assert sum(elicitation.values()) < 600.0
