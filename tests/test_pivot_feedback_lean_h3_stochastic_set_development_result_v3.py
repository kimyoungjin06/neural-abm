from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_stochastic_set_development_v3_20260807.summary.json"
)
CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_stochastic_set_development_v3.yaml"
)
DEVELOPER = ROOT / "scripts/develop_pivot_feedback_lean_h3_stochastic_set_v3.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_stochastic_set_v3.py"
PRECOMMIT = "730a6d9f03c77f2eb2b471dd2ee17acf8067959c"


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary_binds_precommit_and_preserves_no_selection() -> None:
    summary = _json(SUMMARY)
    assert summary["status"] == (
        "COMPLETE_PUBLIC_NONBLIND_STOCHASTIC_DEVELOPMENT_"
        "SET_UNDERCOVERAGE_OVERDISPERSION_FAILURE_NO_SELECTION"
    )
    assert summary["precommit"]["git_commit"] == PRECOMMIT
    bindings = summary["immutable_bindings"]
    assert bindings["config_sha256"] == _sha256(CONFIG)
    assert bindings["developer_sha256"] == _sha256(DEVELOPER)
    assert bindings["verifier_sha256"] == _sha256(VERIFIER)
    assert summary["canonical_execution"]["truth_files_read_by_estimator"] == 0
    assert set(summary["access_accounting"].values()) == {0}
    assert summary["authorization"]["scientific_effect"] == "NOT_ESTABLISHED"
    disposition = summary["disposition"]
    assert disposition["candidate_admissibility"] == "NOT_EVALUATED"
    assert disposition["candidate_selected"] == "NO"
    assert disposition["estimator_selected"] == "NO"
    assert disposition["support_rule_selected"] == "NO"
    assert disposition["fresh_blind_authorized"] is False


def test_summary_records_set_undercoverage_with_denominators() -> None:
    metrics = _json(SUMMARY)["descriptive_metrics"]
    assert metrics["denominators"] == {
        "all_rows": 256,
        "truth_valid": 208,
        "truth_invalid": 48,
        "jointly_valid": 191,
    }
    assert metrics["truth_valid_estimator_valid_rate"] == 0.9182692307692307
    assert metrics["truth_invalid_estimator_invalid_rate"] == 1.0
    assert metrics["field_vector_interval_containment_among_jointly_valid"] == (
        0.581151832460733
    )
    assert metrics["identification_state_match_among_jointly_valid"] == (
        0.9947643979057592
    )
    assert metrics["direction_state_match_among_jointly_valid"] == 1.0


def test_scenario_findings_keep_failure_families_explicit() -> None:
    summary = _json(SUMMARY)
    findings = summary["scenario_findings"]
    assert findings["near_tie_threshold_region"][
        "field_vector_interval_containment_among_jointly_valid"
    ] == 0.0
    assert findings["set_geometry"]["adjacent_set_containment"] == 0.0
    assert findings["set_geometry"]["multimodal_set_containment"] == 0.0
    assert findings["structural_invalidity"] == {
        "flat_truth_invalid_estimator_invalid_rate": 1.0,
        "boundary_truth_invalid_estimator_invalid_rate": 1.0,
        "missing_truth_invalid_estimator_invalid_rate": 1.0,
        "interpretation": (
            "REGISTERED_FLAT_BOUNDARY_AND_MISSING_CASES_"
            "FAILED_CLOSED_AS_DESIGNED"
        ),
    }
    overdispersed = findings["beta_binomial_probability_perturbation"]
    assert overdispersed["truth_valid"] == 16
    assert overdispersed["estimator_valid"] == 0
    assert overdispersed["truth_valid_estimator_valid_rate"] == 0.0
    interpretation = summary["methodological_interpretation"]
    assert interpretation["gate_proportionality"] == (
        "NO_PERFORMANCE_THRESHOLD_OR_ADMISSIBILITY_GATE_WAS_APPLIED_"
        "THE_FAILURES_ARE_DESCRIPTIVE_ESTIMATOR_OUTPUTS"
    )


def test_next_step_is_public_redesign_not_fresh_blind_or_new_review_gate() -> None:
    next_step = _json(SUMMARY)["next_methodological_step"]
    assert next_step["status"] == (
        "REDESIGN_REQUIRED_ON_PUBLIC_DEVELOPMENT_BEFORE_ANY_FRESH_BLIND"
    )
    assert next_step["fresh_blind_authorized"] is False
    assert next_step["additional_governance_review_required_now"] is False
    assert set(next_step["required_objects"]) == {
        "JOINT_POSTERIOR_PEAK_OR_GAP_SET_WITH_AN_EXPLICIT_SIMULTANEOUS_UNCERTAINTY_SEMANTIC",
        "SEPARATE_REPORTING_OF_SET_CONTAINMENT_APPLICABILITY_AND_DIRECTION_WITH_DENOMINATORS",
        "HIERARCHICAL_OR_BETA_BINOMIAL_WORKING_MODEL_OR_AN_EXPLICIT_MINIMUM_SUPPORT_DOMAIN",
        "NEW_DISJOINT_PUBLIC_STRESS_GRID_AFTER_V3_INFORMED_REDESIGN",
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
