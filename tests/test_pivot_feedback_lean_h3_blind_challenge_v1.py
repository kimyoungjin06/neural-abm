from __future__ import annotations

import hashlib
import json
from pathlib import Path
import runpy
import stat

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml"
GENERATOR = ROOT / "scripts/generate_pivot_feedback_lean_h3_blind_challenge_v1.py"
ESTIMATOR = ROOT / "scripts/estimate_pivot_feedback_lean_h3_blind_challenge_v1.py"
EVALUATOR = ROOT / "scripts/evaluate_pivot_feedback_lean_h3_blind_challenge_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_blind_challenge_v1.py"
FIXED_SEED = "01" * 32
FAKE_PRECOMMIT = "a" * 40


def _module(path: Path) -> dict[str, object]:
    return runpy.run_path(str(path))


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _generate_smoke(output: Path) -> dict[str, object]:
    module = _module(GENERATOR)
    generate = module["generate"]
    generate.__globals__["_require_precommit"] = lambda commit: None
    return generate(
        CONFIG,
        output,
        "smoke",
        precommit=FAKE_PRECOMMIT,
        seed_hex=FIXED_SEED,
    )


def _estimate_smoke(output: Path) -> dict[str, object]:
    module = _module(ESTIMATOR)
    estimate = module["estimate"]
    estimate.__globals__["_require_precommit"] = lambda commit: None
    return estimate(CONFIG, output)


def test_contract_freezes_single_rubric_and_keeps_empirical_authority_closed() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "PRECOMMITTED_NO_CHALLENGE_SEED_NO_PANEL_NO_RESULT"
    assert config["scientific_object"]["event_time_grid"] == [-2, -1, 0, 1, 2]
    assert config["design"]["estimators"] == [
        "raw_earliest_argmax",
        "kernel_earliest_argmax",
    ]
    assert config["source_result_space"]["disposition"] == {
        "raw_earliest_argmax": "RETAINED_INCUMBENT_NOT_SELECTED",
        "kernel_earliest_argmax": (
            "NEGATIVE_COMPARATOR_NOT_ADMISSIBLE_AS_V1_PRIMARY"
        ),
        "hierarchical_partial_pooling": (
            "DEFERRED_NOT_DEFINED_NOT_PERMITTED_IN_THIS_CHALLENGE"
        ),
    }
    rubric = config["admissibility_rubric"]
    assert rubric["semantics"] == (
        "ALL_CRITERIA_REQUIRED_PER_ESTIMATOR_NO_POST_UNBLIND_THRESHOLD_CHANGE"
    )
    assert rubric["minimum_identifiable_two_sided_coverage"] == 0.90
    assert rubric["minimum_worst_family_identifiable_two_sided_coverage"] == 0.80
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
    assert authorization["estimator_selected"] == "NO"
    assert authorization["support_rule_selected"] == "NO"
    assert authorization["scientific_effect"] == "NOT_ESTABLISHED"


def test_plan_has_no_seed_panel_or_result(tmp_path: Path) -> None:
    module = _module(GENERATOR)
    plan = module["plan"](CONFIG)
    assert plan["status"] == "PASS_BLIND_CHALLENGE_PLAN_NO_SEED_NO_PANEL_NO_RESULT"
    assert plan["held_out_generator_family_count"] == 8
    assert plan["source_or_empirical_row_read"] == "FORBIDDEN"
    assert plan["scientific_effect"] == "NOT_ESTABLISHED"
    assert list(tmp_path.iterdir()) == []


def test_fixed_seed_generation_is_deterministic_and_panels_are_opaque(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_result = _generate_smoke(first)
    second_result = _generate_smoke(second)
    assert first_result["panel_rows"] == 32
    assert first_result["challenge_scenarios"] == 8
    assert first_result["seed_disclosed"] is False
    assert first_result["challenge_panels_sha256"] == second_result[
        "challenge_panels_sha256"
    ]
    assert hashlib.sha256((first / "private_labels.json").read_bytes()).hexdigest() == (
        hashlib.sha256((second / "private_labels.json").read_bytes()).hexdigest()
    )
    panels = _jsonl(first / "challenge_panels.jsonl")
    assert set(panels[0]) == {"challenge_id", "replicate", "grid", "episodes"}
    assert set(panels[0]["episodes"][0]) == {
        "field_id",
        "episode_id",
        "opportunities",
        "entries",
        "persistent",
    }
    serialized = json.dumps(panels)
    for forbidden in (
        "generator_family",
        "feedback_present",
        "truth_identifiable",
        "truth_delta",
        FIXED_SEED,
    ):
        assert forbidden not in serialized
    assert stat.S_IMODE(first.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in first.iterdir())


def test_blind_estimator_source_has_no_private_label_filename_or_empirical_surface() -> None:
    source = ESTIMATOR.read_text(encoding="utf-8")
    for forbidden in (
        "private_labels.json",
        "pyarrow",
        "duckdb",
        "pandas",
        "g1_risk_set_and_labels.parquet",
        "g4_author_cold_guard.parquet",
    ):
        assert forbidden not in source


def test_smoke_estimation_precedes_unblinding_and_reports_no_private_access(
    tmp_path: Path,
) -> None:
    output = tmp_path / "challenge"
    _generate_smoke(output)
    estimation = _estimate_smoke(output)
    assert estimation["status"] == "PASS_BLIND_ESTIMATES_WRITTEN_BEFORE_UNBLINDING"
    assert estimation["panel_rows"] == 32
    assert estimation["estimate_rows"] == 64
    assert estimation["private_label_files_opened"] == 0
    assert not (output / "unblinded_report.json").exists()
    receipt = _json(output / "blind_estimation_receipt.json")
    assert receipt["private_label_files_opened"] == 0
    assert receipt["blind_estimates"]["sha256"] == hashlib.sha256(
        (output / "blind_estimates.jsonl").read_bytes()
    ).hexdigest()


def test_one_time_unblinding_applies_frozen_rubric_without_selection(
    tmp_path: Path,
) -> None:
    output = tmp_path / "challenge"
    _generate_smoke(output)
    _estimate_smoke(output)
    evaluator = _module(EVALUATOR)
    result = evaluator["evaluate"](CONFIG, output)
    assert result["status"] in {
        "PASS_BLIND_CHALLENGE_AT_LEAST_ONE_ESTIMATOR_ELIGIBLE_NOT_SELECTED",
        "BLOCKED_BLIND_CHALLENGE_NO_ESTIMATOR_ADMISSIBLE_NO_FREEZE",
    }
    assert result["estimator_selected"] == "NO"
    assert result["support_rule_selected"] == "NO"
    report = _json(output / "unblinded_report.json")
    assert set(report["estimator_metrics"]) == {
        "raw_earliest_argmax",
        "kernel_earliest_argmax",
    }
    for metrics in report["estimator_metrics"].values():
        assert set(metrics["criteria"]) == {
            "valid_rate",
            "meaningful_positive_detection_rate",
            "nonpositive_directional_false_positive_rate",
            "identifiable_two_sided_coverage",
            "worst_family_identifiable_two_sided_coverage",
            "identifiable_sign_reversal_rate",
            "nonidentifiable_pass_rate",
        }
        assert metrics["admissible"] is all(metrics["criteria"].values())
    assert set(report["access_accounting"].values()) == {0}
    assert report["authorization"]["post_unblind_retuning"] == "FORBIDDEN"
    with pytest.raises(evaluator["BlindEvaluationError"], match="already exists"):
        evaluator["evaluate"](CONFIG, output)


def test_independent_verifier_replays_bindings_metrics_and_disposition(
    tmp_path: Path,
) -> None:
    output = tmp_path / "challenge"
    _generate_smoke(output)
    _estimate_smoke(output)
    _module(EVALUATOR)["evaluate"](CONFIG, output)
    verifier = _module(VERIFIER)
    result = verifier["verify"](CONFIG, output)
    assert result["status"] == (
        "PASS_INDEPENDENT_BLIND_BINDING_AND_ADJUDICATION_REPLAY_SYNTHETIC_ONLY"
    )
    assert result["replay_counts"] == {
        "panel_rows": 32,
        "estimate_rows": 64,
        "label_rows": 32,
    }
    report = _json(output / "unblinded_report.json")
    assert result["estimator_metrics"] == report["estimator_metrics"]
    assert result["eligible_for_later_freeze"] == report[
        "eligible_for_later_freeze"
    ]
    assert result["estimator_selected"] == "NO"
    assert result["support_rule_selected"] == "NO"
    assert set(result["access_accounting"].values()) == {0}
    assert result["scientific_effect"] == "NOT_ESTABLISHED"


def test_existing_output_root_fails_closed(tmp_path: Path) -> None:
    module = _module(GENERATOR)
    generate = module["generate"]
    generate.__globals__["_require_precommit"] = lambda commit: None
    with pytest.raises(module["BlindChallengeError"], match="already exists"):
        generate(
            CONFIG,
            tmp_path,
            "smoke",
            precommit=FAKE_PRECOMMIT,
            seed_hex=FIXED_SEED,
        )
