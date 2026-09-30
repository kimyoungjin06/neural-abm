from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_llm_persona_agent_v1_20261001.summary.json"
)
WORLDS = ("linear_fit", "smooth_interaction_fit", "axis_interaction_fit")
CELLS = [f"{world}/{arm}" for world in WORLDS for arm in ("linear_batch", "tree_batch", "mlp_batch")]


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_reproduced_and_makes_no_paid_calls() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == "150486aad3a4a4044f30e7ac94acdd9cd7dca90e"
    rerun = result["execution"]["deterministic_rerun"]
    assert rerun["all_four_reproducible_artifacts_byte_identical"] is True
    assert rerun["cost_timing_differs_as_registered"] is True
    assert result["execution"]["raw_row_count"] == 3 * 16 * (4 + 4 + 1)
    assert result["validation"]["decision_machinery_nests_the_fixed_rule_in_every_linear_world_replicate"] is True
    assert result["boundaries"]["paid_api_calls"] == 0


def test_the_persona_is_extremely_cautious_and_self_seals_in_interaction_worlds() -> None:
    result = _result()
    assert result["phase_a"]["zero_evidence_yes_share"] == 0.0
    totals = result["decision_totals"]
    assert totals["pivot_yes"] / totals["pivot_questions"] < 0.002
    assert totals["stay_no"] == totals["stay_questions"]
    sealing = result["registered_result_disposition"]["self_sealing_by_world"]
    assert [sealing[world]["self_sealing"] for world in WORLDS] == [False, True, True]


def test_nabm_matches_the_persona_everywhere_and_is_mostly_superior() -> None:
    disposition = _result()["registered_result_disposition"]
    nabm = disposition["nabm_versus_persona"]
    assert nabm["summary"] == "NABM_MATCHES_PERSONA_AGENT_IN_ALL_CELLS"
    assert set(CELLS) - set(nabm["nabm_precision_superior_cells"]) == {"linear_fit/tree_batch"}
    assert set(CELLS) - set(nabm["nabm_ranking_superior_cells"]) == {
        "axis_interaction_fit/linear_batch",
        "smooth_interaction_fit/linear_batch",
    }
    for cell in nabm["cells"].values():
        assert cell["nabm_minus_persona_precision_mean"] > 0.06
        assert cell["nabm_minus_persona_probe_auc_mean"] > -0.02
    hand = disposition["persona_versus_hand_precision"]
    assert hand["summary"] == "SELECTS_BETTER_NOWHERE"
    assert all(cell["precision_mean"] < 0 for cell in hand["cells"].values())


def test_entry_comes_from_the_floor_and_cost_is_recorded() -> None:
    result = _result()
    floor = result["post_hoc_descriptive"]["entry_comes_from_the_floor"]
    for world in WORLDS:
        assert floor[world]["exploratory"] > floor[world]["decided"]
    assert result["post_hoc_descriptive"]["status"] == "POST_HOC_DESCRIPTIVE_NOT_REGISTERED_NO_NEW_EXECUTION"
    cost = result["cost"]
    assert cost["persona_median_gpu_seconds_per_run"]["development"] > 150
    assert cost["nabm_median_cpu_seconds_per_arm_run"]["development"] < 7
    assert result["scientific_assessment"]["model_selected"] == "NO"
