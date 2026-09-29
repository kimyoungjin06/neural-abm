from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "experiments"
    / "evidence"
    / "results"
    / "pivot_llm_prior_pilot_v2_20260929.summary.json"
)
WORLDS = ("linear_fit", "smooth_interaction_fit", "axis_interaction_fit")


def _result() -> dict[str, object]:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_result_is_precommitted_reproduced_and_makes_no_llm_calls() -> None:
    result = _result()
    assert result["precommit"]["commit_sha"] == (
        "2ddcd9337c7b6c389f45994570ecbea7b3905c1d"
    )
    execution = result["execution"]
    assert execution["raw_row_count"] == 1680
    assert execution["new_llm_calls"] == 0
    assert execution["deterministic_rerun"]["all_five_artifacts_byte_identical"] is True
    diagnostics = result["validation"]["calibration_diagnostics"]
    assert abs(diagnostics["llm_base_rate"]["achieved_mean"] - 0.30) < 1e-9
    assert result["boundaries"]["paid_api_calls"] == 0


def test_calibration_removes_self_sealing() -> None:
    sealing = _result()["registered_result_disposition"]["self_sealing_priors"]
    assert sealing == [f"llm_uncalibrated/{world}" for world in sorted(WORLDS)]


def test_calibrated_prior_selects_better_entrants_everywhere() -> None:
    by_variant = _result()["registered_result_disposition"]["by_variant"]
    for world in WORLDS:
        assert by_variant["llm_base_rate"][world]["entry_precision"] == "VARIANT_BETTER"
        assert by_variant["llm_hand_level"][world]["entry_precision"] == "VARIANT_BETTER"
        assert by_variant["llm_base_rate"][world]["prior_quality"] != "VARIANT_BETTER"
        assert by_variant["llm_verbalized"][world]["local_rescue"] == "VARIANT_BETTER"
        assert by_variant["llm_uncalibrated"][world]["prior_quality"] == "HAND_BETTER"


def test_selectivity_trades_entrant_precision_for_evidence() -> None:
    profiles = _result()["prior_profiles"]
    for world in WORLDS:
        hand, calibrated, verbalized = (
            profiles[world][prior] for prior in ("hand", "llm_base_rate", "llm_verbalized")
        )
        assert calibrated["fixed_entrant_precision"] > hand["fixed_entrant_precision"] + 0.1
        assert calibrated["field_evidence_per_learner"] < hand["field_evidence_per_learner"]
        assert verbalized["field_evidence_per_learner"] > hand["field_evidence_per_learner"]
    for world in ("smooth_interaction_fit", "axis_interaction_fit"):
        pooled = {
            prior: profiles[world][prior]["pooled_probe_brier"]["mlp_batch"]
            for prior in ("hand", "llm_base_rate", "llm_verbalized")
        }
        assert pooled["llm_verbalized"] < pooled["hand"] < pooled["llm_base_rate"]


def test_only_the_verbalized_prior_beats_flat_for_local_learning() -> None:
    post = _result()["post_hoc_descriptive"]["online_probe_brier_improvement_over_flat"]
    for world in WORLDS:
        assert post[world]["llm_verbalized"]["positive_replicates"] == 16
        assert post[world]["hand"]["positive_replicates"] == 0
        assert post[world]["llm_base_rate"]["mean"] < 0.005
    assessment = _result()["scientific_assessment"]
    assert assessment["local_rescue_source"].startswith("MOSTLY_ESCAPING_THE_HAND")
    assert assessment["llm_knowledge_of_real_pivots"] == "NOT_ESTABLISHED"
