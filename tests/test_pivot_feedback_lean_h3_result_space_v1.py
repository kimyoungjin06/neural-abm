from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import stat

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_lean_h3_result_space_v1.yaml"
SIMULATOR = ROOT / "scripts/simulate_pivot_feedback_lean_h3_result_space_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_result_space_v1.py"
SUMMARY = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_result_space_v1_20260806.summary.json"
)


def _module() -> dict[str, object]:
    return runpy.run_path(str(SIMULATOR))


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_contract_is_synthetic_only_and_keeps_empirical_authority_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == (
        "AUTHORIZED_SYNTHETIC_ONLY_NO_EMPIRICAL_ROW_ACCESS_NO_SCIENTIFIC_CLAIM"
    )
    assert config["scientific_object"]["event_time_grid"] == [-2, -1, 0, 1, 2]
    assert config["design"]["estimators"] == [
        "raw_earliest_argmax",
        "kernel_earliest_argmax",
    ]
    authorization = config["authorization"]
    for key in (
        "source_or_empirical_row_read",
        "current_g1_row_read",
        "model_selection_row_read",
        "holdout_or_private_g4_read",
        "empirical_parameter_calibration",
        "confirmatory_execution",
    ):
        assert authorization[key] == "FORBIDDEN"
    assert authorization["scientific_effect"] == "NOT_ESTABLISHED"
    assert config["deferred"] == {
        "hierarchical_partial_pooling": (
            "DEFERRED_UNTIL_RAW_AND_SMOOTHED_COUNTEREXAMPLE_AUDIT"
        ),
        "synthetic_blind_challenge": (
            "DEFERRED_UNTIL_GENERATOR_AND_ESTIMATOR_V1_PASS"
        ),
        "agent_based_generator": (
            "DEFERRED_UNTIL_OBSERVATIONAL_EQUIVALENCE_MAP_EXISTS"
        ),
    }


def test_plan_reconstructs_declared_result_space_without_execution() -> None:
    module = _module()
    plan = module["plan"](CONFIG)
    assert plan["status"] == "PASS_SYNTHETIC_RESULT_SPACE_PLAN_NO_EXECUTION"
    assert plan["canonical_scenario_count"] == 9
    assert plan["event_time_grid"] == [-2, -1, 0, 1, 2]
    assert plan["mechanism_families"] == [
        "delayed_feedback",
        "synchronous_opportunity",
        "fast_correction",
        "selective_composition",
        "capacity_depletion",
        "recruitment_mixture",
        "measurement_artifact",
    ]
    assert plan["source_or_empirical_row_read"] == "FORBIDDEN"
    assert plan["scientific_effect"] == "NOT_ESTABLISHED"


def test_canonical_truth_contains_positive_null_negative_and_mimic_cases() -> None:
    module = _module()
    config = module["load_config"](CONFIG)
    scenarios = {
        scenario["scenario_id"]: scenario
        for scenario in config["canonical_scenarios"]
    }
    expected = {
        "delayed_gap_plus_one_sharp": (1.0, True, True),
        "synchronous_null": (0.0, False, True),
        "fast_correction_negative": (-1.0, False, True),
        "measurement_shift_mimics_positive_gap": (1.0, False, True),
    }
    for scenario_id, (truth, feedback, identifiable) in expected.items():
        scenario = scenarios[scenario_id]
        panel = module["_generate_panel"](config, scenario, 0)
        observed = [episode["observable_delta"] for episode in panel["episodes"]]
        assert observed == [truth] * 18
        assert scenario["feedback_present"] is feedback
        assert scenario["truth_identifiable"] is identifiable


def test_adversarial_manifest_is_deterministic_and_covers_every_family() -> None:
    module = _module()
    config = module["load_config"](CONFIG)
    first = module["_random_scenarios"](config, 14)
    second = module["_random_scenarios"](config, 14)
    assert first == second
    assert {scenario["mechanism_family"] for scenario in first} == set(
        module["MECHANISM_FAMILIES"]
    )
    encoded = json.dumps(first, sort_keys=True).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == (
        hashlib.sha256(
            json.dumps(second, sort_keys=True).encode("utf-8")
        ).hexdigest()
    )


def test_smoke_run_records_calibration_and_counterexamples_without_rows(
    tmp_path: Path,
) -> None:
    module = _module()
    output = tmp_path / "result-space"
    result = module["run"](CONFIG, output, "smoke")
    assert result["status"] == (
        "PASS_SYNTHETIC_RESULT_SPACE_EXECUTION_COUNTEREXAMPLES_EXPECTED_NO_EMPIRICAL_ROWS"
    )
    assert result["scenario_count"] == 17
    assert result["replicate_estimate_rows"] == 208
    assert result["counterexample_rows_total"] > 0

    report = _json(output / "result_space_report.json")
    assert report["access_accounting"] == {
        "source_or_empirical_rows_read": 0,
        "current_g1_rows_read": 0,
        "model_selection_rows_read": 0,
        "holdout_or_private_g4_rows_read": 0,
        "seal_open_events": 0,
        "models_fitted_to_empirical_data": 0,
    }
    canonical = {
        (item["scenario_id"], item["estimator"]): item
        for item in report["canonical_operating_characteristics"]
    }
    for estimator in (
        "raw_earliest_argmax",
        "kernel_earliest_argmax",
    ):
        assert canonical[("delayed_gap_plus_one_sharp", estimator)][
            "pass_rate"
        ] == 1.0
        assert canonical[("synchronous_null", estimator)]["pass_rate"] == 0.0
        assert canonical[("fast_correction_negative", estimator)][
            "pass_rate"
        ] == 0.0
        assert canonical[("measurement_shift_mimics_positive_gap", estimator)][
            "pass_rate"
        ] == 1.0
        assert canonical[("measurement_shift_mimics_positive_gap", estimator)][
            "flag_rates"
        ]["mechanism_overreach_risk"] == 1.0
    assert report["flag_counts"]["estimator_disagreement"] > 0
    assert report["flag_counts"]["confident_wrong_interval"] > 0

    receipt = _json(output / "receipt.json")
    assert receipt["status"] == "PASS_RECEIPT_WRITTEN_LAST_SYNTHETIC_ONLY"
    assert receipt["scientific_effect"] == "NOT_ESTABLISHED"
    assert {item["basename"] for item in receipt["artifacts"]} == {
        "scenario_manifest.jsonl",
        "replicate_estimates.jsonl",
        "counterexamples.json",
        "result_space_report.json",
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o600
        for path in output.iterdir()
        if path.is_file()
    )


def test_existing_output_root_fails_closed(tmp_path: Path) -> None:
    module = _module()
    with pytest.raises(module["ResultSpaceError"], match="already exists"):
        module["run"](CONFIG, tmp_path, "smoke")


def test_independent_verifier_replays_smoke_aggregates_flags_and_ranking(
    tmp_path: Path,
) -> None:
    producer = _module()
    output = tmp_path / "result-space"
    producer["run"](CONFIG, output, "smoke")
    verifier = runpy.run_path(str(VERIFIER))
    result = verifier["verify"](
        config_path=CONFIG,
        producer_path=SIMULATOR,
        output_root=output,
    )
    assert result["status"] == (
        "PASS_INDEPENDENT_AGGREGATE_FLAG_AND_COUNTEREXAMPLE_REPLAY_SYNTHETIC_ONLY"
    )
    assert result["replay_counts"] == {
        "scenario_rows": 17,
        "replicate_estimate_rows": 208,
        "counterexample_rows_total": 94,
        "counterexample_rows_retained": 94,
        "flag_counts": {
            "directional_false_positive": 8,
            "delayed_feedback_false_negative": 0,
            "sign_reversal": 1,
            "confident_wrong_interval": 25,
            "nonidentifiable_truth_pass": 3,
            "mechanism_overreach_risk": 49,
            "aggregation_paradox": 24,
            "estimator_disagreement": 32,
        },
    }
    assert result["access_accounting"]["source_or_empirical_rows_read"] == 0
    assert result["authorization"]["estimator_selected"] == "NO"
    assert result["authorization"]["support_rule_selected"] == "NO"


def test_simulator_has_no_empirical_data_or_row_decode_surface() -> None:
    source = SIMULATOR.read_text(encoding="utf-8") + VERIFIER.read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "pyarrow",
        "duckdb",
        "pandas",
        "download_state.json",
        "g1_risk_set_and_labels.parquet",
        "g4_author_cold_guard.parquet",
        "pq.read_table(",
        ".iter_batches(",
    ):
        assert forbidden not in source


def test_committed_design_pilot_summary_binds_corrected_independent_result() -> None:
    summary = _json(SUMMARY)
    assert summary["status"] == (
        "PASS_INDEPENDENT_SYNTHETIC_RESULT_SPACE_COUNTEREXAMPLES_FOUND_"
        "KERNEL_NOT_ADMISSIBLE_RAW_NOT_YET_SELECTED_NO_EMPIRICAL_ROWS"
    )
    assert summary["canonical_execution"] == {
        "artifact_root": (
            "/ssd/pivot-feedback-lean-h3-result-space-v1-design-pilot-"
            "corrected-20260806"
        ),
        "profile": "design_pilot",
        "canonical_scenarios": 9,
        "adversarial_scenarios": 96,
        "total_scenarios": 105,
        "canonical_replicates": 128,
        "adversarial_replicates": 24,
        "bootstrap_replicates": 199,
        "replicate_estimate_rows": 6912,
        "counterexample_rows_total": 4012,
        "counterexample_rows_retained": 200,
    }
    assert summary["immutable_bindings"]["config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    assert summary["immutable_bindings"]["producer_sha256"] == hashlib.sha256(
        SIMULATOR.read_bytes()
    ).hexdigest()
    assert summary["immutable_bindings"]["verifier_sha256"] == hashlib.sha256(
        VERIFIER.read_bytes()
    ).hexdigest()
    assert summary["replayed_flag_counts"] == {
        "aggregation_paradox": 706,
        "confident_wrong_interval": 1085,
        "delayed_feedback_false_negative": 0,
        "directional_false_positive": 128,
        "estimator_disagreement": 958,
        "mechanism_overreach_risk": 2360,
        "nonidentifiable_truth_pass": 1070,
        "sign_reversal": 103,
    }
    kernel = summary["canonical_findings"]["bimodal_recruitment_kernel"]
    assert kernel["truth_delta"] == 0.0
    assert kernel["pass_rate"] == 1.0
    assert kernel["two_sided_coverage"] == 0.0
    assert summary["design_disposition"] == {
        "kernel_earliest_argmax": "NOT_ADMISSIBLE_AS_V1_PRIMARY",
        "raw_earliest_argmax": "RETAINED_CANDIDATE_NOT_SELECTED",
        "support_rule": "NOT_SELECTED",
        "hierarchical_partial_pooling": "DEFERRED",
        "synthetic_blind_challenge": "NEXT_BEFORE_ANY_ESTIMATOR_FREEZE",
        "agent_based_generator": "DEFERRED",
    }
    assert set(summary["access_accounting"].values()) == {0}
    assert summary["authorization"]["estimator_selected"] == "NO"
    assert summary["authorization"]["support_rule_selected"] == "NO"
    assert summary["authorization"]["scientific_effect"] == "NOT_ESTABLISHED"
    assert summary["superseded_diagnostic_execution"]["status"] == (
        "SUPERSEDED_DIAGNOSTIC_FLAG_SEMANTICS_ONLY_DO_NOT_CITE"
    )
