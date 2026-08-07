from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import runpy

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "experiments/configs/pivot_feedback_lean_h3_set_identified_prospectus_v2.yaml"
)
PRODUCER = ROOT / "scripts/prototype_pivot_feedback_lean_h3_set_identified_v2.py"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_lean_h3_set_identified_v2.py"


def _producer() -> dict[str, object]:
    return runpy.run_path(str(PRODUCER))


def _verifier() -> dict[str, object]:
    return runpy.run_path(str(VERIFIER))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_is_prospective_fixture_only_and_outcome_independent() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["mode"] == (
        "ANALYTIC_FIXTURE_ONLY_PROSPECTIVE_DEFINITION_NO_OPENED_PANEL_REPLAY"
    )
    assert config["status"] == (
        "AUTHORIZED_PROSPECTUS_ONLY_NO_CANDIDATE_SELECTION_NO_BLIND_PROMOTION"
    )
    anchor = config["source_anchors"]["outcome_blind_practical_null_anchor"]
    assert anchor["original_symmetric_practical_null"] == [-0.01, 0.01]
    assert anchor["selected_from_observed_outcomes"] is False
    assert "NOT_EMPIRICALLY_VALIDATED_FOR_PEAK_LOCATION" in anchor["reuse_status"]
    opened = config["source_anchors"]["opened_peak_set_diagnostic"]
    assert opened["permitted_role"] == "FAILURE_MODE_NAMES_AND_CLAIM_BOUNDARIES_ONLY"
    assert opened["forbidden_role"] == (
        "METRIC_DRIVEN_PARAMETER_OR_THRESHOLD_SELECTION"
    )
    authorization = config["authorization"]
    for key in (
        "opened_challenge_panel_replay",
        "opened_challenge_private_label_read",
        "source_or_empirical_row_read",
        "current_g1_row_read",
        "model_selection_row_read",
        "holdout_or_private_g4_read",
        "empirical_parameter_calibration",
        "confirmatory_execution",
    ):
        assert authorization[key] == "FORBIDDEN"
    assert authorization["candidate_selected"] == "NO"
    assert authorization["estimator_selected"] == "NO"
    assert authorization["support_rule_selected"] == "NO"


def test_source_anchors_are_hash_bound_and_plan_has_no_fixture_run() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for anchor in config["source_anchors"].values():
        assert anchor["sha256"] == _sha256(ROOT / anchor["relative_path"])
    result = _producer()["plan"](CONFIG)
    assert result == {
        "status": "PASS_SET_IDENTIFIED_PROSPECTUS_PLAN_NO_FIXTURE_RUN",
        "config_sha256": _sha256(CONFIG),
        "candidate_family": "practical_equivalence_set_field_vector_v2",
        "analytic_fixture_count": 7,
        "opened_challenge_panel_replay": "FORBIDDEN_NOT_RUN",
        "candidate_selected": "NO",
        "scientific_effect": "NOT_ESTABLISHED",
    }


def test_practical_peak_set_preserves_adjacent_and_multimodal_sets() -> None:
    peak_set = _producer()["_curve_peak_set"]
    singleton = peak_set([0.01, 0.03, 0.10, 0.02, 0.01], margin=0.01)
    assert singleton["peak_set"] == [0.0]
    assert singleton["shape_state"] == "SINGLETON"
    adjacent = peak_set([0.01, 0.03, 0.10, 0.095, 0.01], margin=0.01)
    assert adjacent["peak_set"] == [0.0, 1.0]
    assert adjacent["shape_state"] == "CONTIGUOUS_SET"
    multimodal = peak_set([0.01, 0.10, 0.02, 0.095, 0.01], margin=0.01)
    assert multimodal["peak_set"] == [-1.0, 1.0]
    assert multimodal["shape_state"] == "MULTIMODAL_SET"


def test_flat_boundary_and_missing_curves_remain_invalid() -> None:
    peak_set = _producer()["_curve_peak_set"]
    flat = peak_set([0.050, 0.052, 0.055, 0.051, 0.050], margin=0.01)
    assert flat["valid"] is False
    assert flat["invalid_reason"] == "PRACTICALLY_FLAT_CURVE"
    boundary = peak_set([0.01, 0.02, 0.03, 0.05, 0.10], margin=0.01)
    assert boundary["valid"] is False
    assert boundary["invalid_reason"] == "BOUNDARY_EQUIVALENT_PEAK_SET"
    missing = peak_set([0.01, 0.03, None, 0.02, 0.01], margin=0.01)
    assert missing["valid"] is False
    assert missing["invalid_reason"] == "MISSING_REGISTERED_TIME"


def test_fixture_run_preserves_vector_and_never_makes_binary_pass_decision() -> None:
    result = _producer()["run"](CONFIG)
    assert result["status"] == "PASS_ANALYTIC_DEFINITION_CONFORMANCE_NO_SELECTION"
    assert result["fixture_counts"] == {
        "total": 7,
        "conformant": 7,
        "valid": 4,
        "invalid": 3,
        "point_identified": 2,
        "set_identified": 2,
    }
    outcomes = {row["fixture_id"]: row for row in result["fixture_outcomes"]}
    assert outcomes["adjacent_practical_near_tie"]["field_intervals"] == [
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
        [1.0, 2.0],
    ]
    assert outcomes["staggered_field_vector"]["field_intervals"] == [
        [-1.0, -1.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ]
    assert outcomes["staggered_field_vector"]["direction_state"] == (
        "MIXED_OR_PARTIALLY_IDENTIFIED_NO_BINARY_DECISION"
    )
    encoded = json.dumps(result, sort_keys=True)
    assert '"passed"' not in encoded
    assert '"admissible": true' not in encoded.lower()
    assert result["disposition"]["candidate_selected"] == "NO"
    assert set(result["access_accounting"].values()) == {0}


def test_independent_oracle_verifier_rejects_mutated_field_vector(tmp_path: Path) -> None:
    result = _producer()["run"](CONFIG)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    verifier = _verifier()
    verified = verifier["verify"](CONFIG, result_path)
    assert verified["status"] == (
        "PASS_INDEPENDENT_ANALYTIC_FIXTURE_ORACLE_REPLAY_NO_SELECTION"
    )
    mutated = copy.deepcopy(result)
    mutated["fixture_outcomes"][0]["field_intervals"][0] = [0.0, 0.0]
    result_path.write_text(json.dumps(mutated), encoding="utf-8")
    error = verifier["SetIdentifiedVerificationError"]
    with pytest.raises(error, match="oracle drift"):
        verifier["verify"](CONFIG, result_path)


def test_tools_expose_no_opened_or_empirical_decode_surface() -> None:
    source = PRODUCER.read_text(encoding="utf-8") + VERIFIER.read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "/ssd/",
        "challenge_panels.jsonl",
        "private_labels.json",
        "g1_risk_set_and_labels.parquet",
        "g4_author_cold_guard.parquet",
        "pyarrow",
        "duckdb",
        "pandas",
    ):
        assert forbidden not in source
