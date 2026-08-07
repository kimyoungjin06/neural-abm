from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_set_identified_prospectus_v2_20260807.summary.json"
)
CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_set_identified_prospectus_v2.yaml"
)
PRODUCER = ROOT / "scripts/prototype_pivot_feedback_lean_h3_set_identified_v2.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_set_identified_v2.py"
PRECOMMIT = "9747e5b77157af24838f4f79596832050c5d689d"


def _json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_summary_binds_precommit_and_no_selection_disposition() -> None:
    summary = _json(SUMMARY)
    assert summary["status"] == (
        "COMPLETE_SET_IDENTIFIED_ANALYTIC_CONFORMANCE_"
        "NO_STOCHASTIC_CANDIDATE_NO_SELECTION"
    )
    assert summary["precommit"]["git_commit"] == PRECOMMIT
    bindings = summary["immutable_bindings"]
    assert bindings["config_sha256"] == _sha256(CONFIG)
    assert bindings["producer_sha256"] == _sha256(PRODUCER)
    assert bindings["verifier_sha256"] == _sha256(VERIFIER)
    disposition = summary["disposition"]
    assert disposition == {
        "definition_conformance": "PASS",
        "margin_validated_for_peak_location": False,
        "stochastic_candidate_implemented": False,
        "candidate_admissibility": "NOT_EVALUATED",
        "candidate_selected": "NO",
        "estimator_selected": "NO",
        "support_rule_selected": "NO",
        "fresh_blind_authorized": False,
    }
    assert set(summary["access_accounting"].values()) == {0}
    assert summary["authorization"]["scientific_effect"] == "NOT_ESTABLISHED"


def test_fixture_findings_preserve_sets_and_field_heterogeneity() -> None:
    findings = _json(SUMMARY)["fixture_findings"]
    assert findings["adjacent_practical_near_tie"]["field_intervals"] == [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ]
    assert findings["staggered_field_vector"]["field_intervals"] == [
        [-1.0, -1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    assert findings["staggered_field_vector"]["field_balanced_interval_hull"] == [
        0.5,
        0.5,
    ]
    assert findings["staggered_field_vector"]["direction_state"] == (
        "MIXED_OR_PARTIALLY_IDENTIFIED_NO_BINARY_DECISION"
    )
    assert findings["noncontiguous_interior_multimodal"]["field_intervals"] == [
        [0.0, 2.0],
        [0.0, 2.0],
        [0.0, 2.0],
        [0.0, 2.0],
    ]
    assert findings["boundary_censored"]["state"] == "INVALID"
    assert findings["missing_registered_time"]["state"] == "INVALID"
    assert findings["practically_flat"]["state"] == "INVALID"


def test_next_gate_requires_stochastic_precommit_before_any_fresh_blind() -> None:
    summary = _json(SUMMARY)
    next_gate = summary["next_gate"]
    assert next_gate["status"] == (
        "PENDING_SEPARATE_PRECOMMIT_BEFORE_ANY_STOCHASTIC_OR_OPENED_PANEL_RUN"
    )
    assert set(next_gate["required_objects"]) == {
        "ENTRY_AND_PERSISTENCE_MARGIN_SCALE_AUDIT_INDEPENDENT_OF_OPENED_RESULTS",
        "POSTERIOR_PRACTICAL_PEAK_SET_WITH_SIMULTANEOUS_UNCERTAINTY",
        "DEPENDENCE_AND_OVERDISPERSION_MODEL_OR_EXPLICIT_UNRESOLVED_BOUNDARY",
        "DISJOINT_NONBLIND_DEVELOPMENT_GENERATOR_NOT_CALIBRATED_TO_OPENED_METRICS",
        "FRESH_HELD_OUT_BLIND_GENERATOR_ONLY_AFTER_STOCHASTIC_CANDIDATE_FREEZE",
    }
    assert next_gate["opened_challenge_reuse"] == (
        "FORBIDDEN_FOR_PARAMETER_SELECTION_OR_BLIND_CONFIRMATION"
    )
    assert next_gate["fresh_blind_authorized"] is False
    interpretation = summary["scientific_interpretation"]
    assert interpretation["gate_proportionality"] == (
        "NO_ADMISSIBILITY_GATE_OR_PERFORMANCE_THRESHOLD_WAS_CREATED_"
        "FIXTURE_PASS_MEANS_IMPLEMENTATION_CONFORMANCE_ONLY"
    )


def test_prospectus_code_is_byte_identical_to_precommit() -> None:
    for path in (CONFIG, PRODUCER, VERIFIER):
        relative = path.relative_to(ROOT).as_posix()
        committed = subprocess.run(
            ["git", "show", f"{PRECOMMIT}:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        assert committed == path.read_bytes()
