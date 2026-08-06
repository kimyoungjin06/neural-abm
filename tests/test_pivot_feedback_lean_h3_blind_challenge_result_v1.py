from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_blind_challenge_v1_20260806.summary.json"
)
CONFIG = ROOT / "experiments/configs/pivot_feedback_lean_h3_blind_challenge_v1.yaml"
GENERATOR = ROOT / "scripts/generate_pivot_feedback_lean_h3_blind_challenge_v1.py"
ESTIMATOR = ROOT / "scripts/estimate_pivot_feedback_lean_h3_blind_challenge_v1.py"
EVALUATOR = ROOT / "scripts/evaluate_pivot_feedback_lean_h3_blind_challenge_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_blind_challenge_v1.py"
PRECOMMIT = "5b5f7efd2e9264fe4b9db566b123ffb2a1dda210"


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary_binds_precommitted_code_and_blocked_disposition() -> None:
    summary = _json(SUMMARY)
    assert summary["status"] == (
        "BLOCKED_BLIND_CHALLENGE_NO_ESTIMATOR_ADMISSIBLE_NO_FREEZE"
    )
    assert summary["precommit"]["git_commit"] == PRECOMMIT
    bindings = summary["immutable_bindings"]
    assert bindings["config_sha256"] == _sha256(CONFIG)
    assert bindings["generator_sha256"] == _sha256(GENERATOR)
    assert bindings["estimator_sha256"] == _sha256(ESTIMATOR)
    assert bindings["evaluator_sha256"] == _sha256(EVALUATOR)
    assert bindings["verifier_sha256"] == _sha256(VERIFIER)
    disposition = summary["disposition"]
    assert disposition["eligible_for_later_freeze"] == []
    assert disposition["estimator_selected"] == "NO"
    assert disposition["support_rule_selected"] == "NO"
    assert disposition["post_unblind_retuning_on_this_challenge"] == "FORBIDDEN"
    assert disposition["opened_challenge_future_role"] == (
        "DISCOVERY_DIAGNOSTIC_ONLY_NOT_REUSABLE_AS_BLIND_CONFIRMATION"
    )
    assert set(summary["access_accounting"].values()) == {0}
    assert summary["authorization"]["scientific_effect"] == "NOT_ESTABLISHED"


def test_raw_failure_is_multiple_substantive_misses_not_near_threshold() -> None:
    summary = _json(SUMMARY)
    raw = summary["estimator_results"]["raw_earliest_argmax"]
    assert raw["admissible"] is False
    assert raw["valid_rate"] == 1.0
    assert raw["meaningful_positive_detection_rate"] == 0.9620253164556962
    assert raw["nonpositive_directional_false_positive_rate"] == 0.0
    assert raw["identifiable_sign_reversal_rate"] == 0.022286421791167972
    assert raw["identifiable_two_sided_coverage"] == 0.81328125
    assert raw["worst_family_identifiable_two_sided_coverage"] == 0.53515625
    assert raw["nonidentifiable_pass_rate"] == 0.08333333333333333
    assert raw["key_counts"] == {
        "meaningful_positive_detected": 1216,
        "meaningful_positive_total": 1264,
        "nonpositive_false_positive": 0,
        "nonpositive_total": 802,
        "identifiable_interval_covered": 2082,
        "identifiable_interval_total": 2560,
        "identifiable_sign_reversal": 54,
        "identifiable_signed_total": 2423,
        "nonidentifiable_passed": 128,
        "nonidentifiable_total": 1536,
    }
    proportionality = summary["gate_proportionality"]
    assert proportionality["conclusion"] == (
        "FAILURE_IS_SUBSTANTIVE_NOT_A_NEAR_THRESHOLD_GOVERNANCE_ARTIFACT"
    )
    assert proportionality["evidence"]["raw_failed_criteria_count"] == 3


def test_kernel_remains_inadmissible_and_no_threshold_was_relaxed() -> None:
    summary = _json(SUMMARY)
    kernel = summary["estimator_results"]["kernel_earliest_argmax"]
    assert kernel["admissible"] is False
    assert kernel["identifiable_two_sided_coverage"] == 0.53203125
    assert kernel["worst_family_identifiable_two_sided_coverage"] == 0.24609375
    assert kernel["identifiable_sign_reversal_rate"] == 0.06314486174164259
    assert len(kernel["failed_criteria"]) == 4
    assert summary["frozen_thresholds"] == {
        "minimum_valid_rate": 0.98,
        "minimum_meaningful_positive_detection_rate": 0.8,
        "maximum_nonpositive_directional_false_positive_rate": 0.05,
        "minimum_identifiable_two_sided_coverage": 0.9,
        "minimum_worst_family_identifiable_two_sided_coverage": 0.8,
        "maximum_identifiable_sign_reversal_rate": 0.05,
        "maximum_nonidentifiable_pass_rate": 0.05,
    }


def test_challenge_code_is_byte_identical_to_precommit() -> None:
    for path in (CONFIG, GENERATOR, ESTIMATOR, EVALUATOR, VERIFIER):
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.run(
            ["git", "show", f"{PRECOMMIT}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert committed == path.read_bytes()
