from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_peak_set_development_v1_20260806.summary.json"
)
CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_peak_set_development_v1.yaml"
)
DEVELOPER = ROOT / "scripts/develop_pivot_feedback_lean_h3_peak_set_estimator_v1.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_peak_set_development_v1.py"
PRECOMMIT = "ea9bac8e3a94cba190e23c2a8055b5873105ada1"


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary_binds_precommitted_candidate_and_no_selection() -> None:
    summary = _json(SUMMARY)
    assert summary["status"] == (
        "COMPLETE_OPENED_DISCOVERY_DIAGNOSTIC_"
        "CONDITIONAL_CALIBRATION_LOW_UNEVEN_VALIDITY_NO_SELECTION"
    )
    assert summary["precommit"]["git_commit"] == PRECOMMIT
    bindings = summary["immutable_bindings"]
    assert bindings["config_sha256"] == _sha256(CONFIG)
    assert bindings["developer_sha256"] == _sha256(DEVELOPER)
    assert bindings["verifier_sha256"] == _sha256(VERIFIER)
    assert summary["canonical_execution"]["private_label_files_opened_by_estimator"] == 0
    disposition = summary["disposition"]
    assert disposition["candidate_admissibility"] == "NOT_EVALUATED"
    assert disposition["estimator_selected"] == "NO"
    assert disposition["support_rule_selected"] == "NO"
    assert disposition["retuning_this_candidate_on_opened_challenge"] == "FORBIDDEN"
    assert set(summary["access_accounting"].values()) == {0}
    assert summary["authorization"]["scientific_effect"] == "NOT_ESTABLISHED"


def test_conditional_calibration_is_recorded_with_low_uneven_validity() -> None:
    summary = _json(SUMMARY)
    metrics = summary["diagnostic_metrics"]
    assert metrics["identifiable_valid_rate"] == 0.4234375
    assert metrics["identifiable_two_sided_coverage_among_valid"] == (
        0.9723247232472325
    )
    assert metrics["meaningful_positive_detection_rate_among_valid"] == (
        0.9376026272577996
    )
    assert metrics["nonpositive_false_positive_rate_among_valid"] == 0.0
    assert metrics["identifiable_sign_reversal_rate_among_valid"] == (
        0.007759456838021339
    )
    assert metrics["key_counts"] == {
        "identifiable_valid": 1084,
        "identifiable_total": 2560,
        "nonidentifiable_invalid": 1464,
        "nonidentifiable_total": 1536,
        "meaningful_positive_detected": 571,
        "meaningful_positive_valid": 609,
        "nonpositive_false_positive": 0,
        "nonpositive_valid": 359,
        "identifiable_interval_covered": 1054,
        "identifiable_valid_for_coverage": 1084,
        "identifiable_sign_reversal": 8,
        "identifiable_signed_valid": 1031,
        "no_feedback_passed": 439,
        "no_feedback_valid": 1006,
    }


def test_structural_cases_define_the_next_prospective_targets() -> None:
    summary = _json(SUMMARY)
    families = summary["family_findings"]
    assert families["informative_time_missingness"]["nonidentifiable_invalid_rate"] == 1.0
    assert families["monotone_boundary_process"]["nonidentifiable_invalid_rate"] == 1.0
    assert families["plateau_and_near_tie"]["nonidentifiable_invalid_rate"] == 0.859375
    assert families["plateau_and_near_tie"]["nonidentifiable_valid_rows"] == 72
    assert families["staggered_field_mixture"]["identifiable_valid_rate"] == (
        0.00390625
    )
    assert families["staggered_field_mixture"]["valid_rows"] == 2
    assert families["no_feedback_observational_mimic"]["pass_rate_among_valid"] == (
        0.8782201405152225
    )
    interpretation = summary["methodological_interpretation"]
    assert interpretation["gate_proportionality"] == (
        "NO_NEW_PASS_FAIL_GATE_WAS_APPLIED_"
        "THE_LOW_VALIDITY_IS_AN_ESTIMATOR_OUTPUT_NOT_A_GOVERNANCE_FAILURE"
    )
    assert set(interpretation["unresolved_targets"]) == {
        "PRACTICAL_PEAK_CONTRAST_OR_EQUIVALENCE_RULE_FOR_NEAR_TIES",
        "FIELD_VECTOR_OR_PARTIAL_IDENTIFICATION_FOR_STAGGERED_HETEROGENEITY",
        "EMPIRICALLY_JUSTIFIED_DEPENDENCE_AND_OVERDISPERSION_MODEL",
    }


def test_candidate_code_is_byte_identical_to_precommit() -> None:
    for path in (CONFIG, DEVELOPER, VERIFIER):
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.run(
            ["git", "show", f"{PRECOMMIT}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert committed == path.read_bytes()
