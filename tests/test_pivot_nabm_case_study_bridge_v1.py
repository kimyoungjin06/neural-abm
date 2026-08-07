from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_nabm_case_study_bridge_v1.yaml"
DOC = ROOT / "docs/research/pivot-nabm-case-study-bridge-v1.md"


def _config() -> dict[str, object]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bridge_places_nabm_above_pivot_and_h3() -> None:
    config = _config()
    assert config["status"] == "DESIGN_BRIDGE_ONLY_NO_EXECUTION_NO_MODEL_SELECTION"
    hierarchy = config["project_hierarchy"]
    assert hierarchy["primary_goal"] == (
        "COMPLETE_AND_IMPROVE_NABM_AS_A_REUSABLE_INSPECTABLE_ADAPTIVE_ABM_FRAMEWORK"
    )
    assert hierarchy["pivot_role"] == "FLAGSHIP_CASE_STUDY_AND_FRAMEWORK_STRESS_TEST"
    assert hierarchy["empirical_role"] == (
        "CONSTRAIN_SIMULATION_OBSERVABLES_AND_EXTERNAL_PLAUSIBILITY"
    )
    assert hierarchy["lean_h3_role"] == (
        "ONE_SET_VALUED_EMPIRICAL_TIMING_TARGET_NOT_THE_PROJECT_GOAL"
    )
    document = DOC.read_text(encoding="utf-8")
    assert "complete and improve NABM" in document
    assert "Lean H3 is subordinate" in document


def test_bridge_binds_current_studies_measurement_and_capability_context() -> None:
    bindings = _config()["source_bindings"]
    assert len(bindings) == 7
    roles = {binding["role"] for binding in bindings}
    assert roles == {
        "FIXED_RULE_STYLIZED_PREDECESSOR",
        "IMMEDIATE_VISIBILITY_LEARNING_PREDECESSOR",
        "TRACKED_STUDY_1_RESULT_SNAPSHOT",
        "TRACKED_STUDY_2_RESULT_SNAPSHOT",
        "PROSPECTIVE_EMPIRICAL_AND_BASELINE_LADDER_CONTEXT",
        "SET_VALUED_MEASUREMENT_FAILURE_CONTEXT",
        "CURRENT_NABM_CAPABILITY_AND_EXTRACTION_BOUNDARY",
    }
    for binding in bindings:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert binding["sha256"] == _sha256(path)


def test_empirical_map_preserves_set_vectors_and_proxy_boundaries() -> None:
    mapping = {
        row["empirical_object"]: row for row in _config()["empirical_to_simulation_map"]
    }
    assert set(mapping) == {
        "ATTENTION_SURGE_EPISODE_CLOCK",
        "TOPIC_ENTRY_INCIDENCE_CURVE_AND_PRACTICAL_PEAK_SET",
        "LATER_COHORT_PERSISTENCE_CURVE_AND_DECLINE_TIMING",
        "PUBLIC_AVAILABILITY_PROXY",
        "PRE_ENTRY_CAPACITY_AND_BRIDGE_PROXIES",
        "FOUR_FIELD_SET_VALUED_TIMING_VECTOR",
        "MISSING_OR_ZERO_DENOMINATOR_CELLS",
    }
    assert mapping["PUBLIC_AVAILABILITY_PROXY"]["comparison_role"] == (
        "AVAILABILITY_NOT_RECEIPT_OR_BELIEF"
    )
    assert mapping["FOUR_FIELD_SET_VALUED_TIMING_VECTOR"]["comparison_role"] == (
        "PRIMARY_EMPIRICAL_ALIGNMENT_NO_SCALAR_REPLACEMENT"
    )
    assert mapping["MISSING_OR_ZERO_DENOMINATOR_CELLS"]["comparison_role"] == (
        "NO_IMPUTATION_RESCUE"
    )


def test_capability_gaps_improve_nabm_without_absorbing_domain_semantics() -> None:
    inventory = _config()["nabm_capability_inventory"]
    gaps = {
        row["capability"]: row for row in inventory["flagship_required_next"]
    }
    assert set(gaps) == {
        "DELAYED_SELECTIVE_OUTCOME_EVENT_QUEUE",
        "PRIVATE_STATE_PUBLIC_MESSAGE_SEPARATION",
        "MULTI_CHANNEL_ATTENTION_SUCCESS_AND_FAILURE_MESSAGES",
        "NONABSORBING_PIVOT_PERSISTENCE_EXIT_AND_RETURN_STATE_MACHINE",
        "EMPIRICAL_SET_VALUED_OBSERVATION_ADAPTER",
        "FAIR_BASELINE_LADDER_ORCHESTRATION",
    }
    assert gaps["DELAYED_SELECTIVE_OUTCOME_EVENT_QUEUE"]["first_owner"] == (
        "PIVOT_DOMAIN_ADAPTER"
    )
    assert gaps["NONABSORBING_PIVOT_PERSISTENCE_EXIT_AND_RETURN_STATE_MACHINE"][
        "extraction_rule"
    ] == "KEEP_DOMAIN_SEMANTICS_OUTSIDE_NABM_UNIT"
    assert "FRAMEWORK_OWNED_SCIENTIFIC_SUCCESS_CRITERION" in inventory[
        "deferred_unless_diagnostic_requires"
    ]


def test_model_ladder_is_fair_and_can_prefer_the_simpler_model() -> None:
    ladder = _config()["model_ladder"]
    assert [model["id"] for model in ladder["models"]] == [
        "B1_FIXED_STATISTICAL",
        "B2_FIXED_RULE_ABM",
        "B3_EXPLICIT_ADAPTIVE_BASELINE",
        "B4_NABM_LOCAL_ONLY",
        "B5_NABM_IMITATIVE",
        "B6_NABM_CAUTIONARY_OR_MIXED",
    ]
    assert set(ladder["common_inputs"]) == {
        "IDENTICAL_AGENT_POPULATION_OR_EMPIRICAL_RISK_SET",
        "IDENTICAL_ATTENTION_AND_OUTCOME_AVAILABILITY_PROCESS",
        "IDENTICAL_TOPOLOGY_AND_RANDOMNESS_BY_REPLICATE",
        "IDENTICAL_TRAIN_VALIDATION_AND_HOLDOUT_ASSIGNMENTS",
        "IDENTICAL_SCORING_ROWS_AND_COMPUTE_BUDGET_RULE",
    }
    assert "prefer the simpler scientific model" in ladder["decision_rule"]
    assert "NO_NABM_GAIN_SIMPLER_MODEL_PREFERRED" in _config()[
        "registered_result_space"
    ]["families"]


def test_next_build_is_delayed_feedback_prototype_not_empirical_execution() -> None:
    config = _config()
    next_build = config["next_build"]
    assert next_build["name"] == "PIVOT_DELAYED_FEEDBACK_LEARNING_PROTOTYPE_V1"
    assert set(next_build["scope"]) == {
        "DOMAIN_OWNED_SUCCESS_AND_FAILURE_VISIBILITY_DELAYS",
        "SEPARATE_PRIVATE_OUTCOME_AND_PUBLIC_MESSAGE_STATE",
        "FIXED_IMITATIVE_AND_CAUTIONARY_POLICY_ARMS",
        "IMMEDIATE_VISIBILITY_PREDECESSOR_AS_NESTED_BASELINE",
        "NO_EMPIRICAL_FIT",
    }
    assert next_build["additional_governance_gate_required"] is False
    assert next_build["separate_precommit_before_result_interpretation"] is True
    authorization = config["authorization"]
    assert authorization["generic_nabm_api_change"] == "NOT_AUTHORIZED_BY_THIS_BRIDGE"
    for key in (
        "empirical_row_access",
        "opened_challenge_reuse",
        "fresh_blind_generation",
        "model_selection",
        "holdout_opening",
    ):
        assert authorization[key] == "FORBIDDEN_NOT_RUN"
    assert authorization["scientific_effect"] == "NOT_ESTABLISHED"
