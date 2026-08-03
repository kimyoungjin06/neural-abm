from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml"
)
APPLICATION = (
    ROOT
    / "scripts/apply_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py"
)
FIXTURE_INPUT = (
    ROOT
    / "tests/fixtures/pivot_feedback_g0_g1_replication_tolerance_v1.input.json"
)
FIXTURE_EXPECTED = (
    ROOT
    / "tests/fixtures/pivot_feedback_g0_g1_replication_tolerance_v1.expected.json"
)
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_g0_g1_replication_tolerance_precommitment_20260803.summary.json"
)
GATE_V61 = ROOT / "experiments/configs/pivot_feedback_gate_state_v61.yaml"
G0_V4 = ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml"
PROTOCOL = (
    ROOT
    / "docs/case-studies/researcher-pivot/"
    "pivot-feedback-empirical-nabm-protocol.md"
)
NOTE = (
    ROOT
    / "docs/research/"
    "pivot-feedback-g0-g1-replication-tolerance-precommitment-v1.md"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_contract_binds_v61_and_preserves_g0_v4() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    gate = yaml.safe_load(GATE_V61.read_text(encoding="utf-8"))
    g0 = yaml.safe_load(G0_V4.read_text(encoding="utf-8"))

    assert _sha256(CONTRACT) == (
        "1bcf91d197a9b927286d43f3505528f3f3f7fb022ffd4fe89d95cb24c041184d"
    )
    assert contract["authorization_parent"]["sha256"] == _sha256(GATE_V61)
    assert contract["immutable_g0_v4"]["sha256"] == _sha256(G0_V4)
    assert _sha256(GATE_V61) == (
        "253c661fd529feb0d70733174197be449dfd27403b5881c6510bd7c8257c2715"
    )
    assert _sha256(G0_V4) == (
        "cd49a31ab67e7a66d038b3198493df8b3a309aa14ee9de80410557a194581ce4"
    )
    assert _sha256(PROTOCOL) == (
        "731e39dec4c7be5c29a0f0404b48992ad0caef152062d1d1d7876acc07064746"
    )
    assert gate["authorized_next_precommitment_scope"]["exact_choice_ids"] == [
        "g1_replication_tolerance"
    ]
    assert g0["resolution"]["unresolved_quantitative_choices"][
        "g1_replication_tolerance"
    ] is None
    assert g0["lifecycle"]["pass_g0"] is False


def test_policy_defines_scale_direction_interval_and_failure_rule() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert contract["scope"]["choice_ids"] == ["g1_replication_tolerance"]
    rule = contract["precommitment_choices"]["g1_replication_tolerance"]

    assert rule["value"] == 1.0
    assert rule["unit"] == "COHORT_YEAR"
    assert rule["estimand"]["scale"] == "ADDITIVE_COHORT_YEARS"
    assert rule["estimand"]["replication_contrast"] == (
        "FIELD_VALUE_MINUS_OTHER_FIELDS_REFERENCE"
    )
    assert rule["estimand"]["favorable_direction"] == (
        "LARGER_IS_MORE_POSITIVE_TIMING_GAP"
    )
    assert rule["interval_rule"]["interval"] == (
        "ONE_SIDED_95_PERCENT_LOWER_CONFIDENCE_BOUND"
    )
    assert rule["interval_rule"][
        "lower_noninferiority_margin_cohort_years"
    ] == -1.0
    assert rule["interval_rule"]["interval_boundary_counts_as_pass"] is False
    assert rule["interval_rule"]["this_contract_selects_interval_estimator"] is False
    assert rule["failure_rule"]["post_access_margin_relaxation_allowed"] is False
    assert rule["claim_role"].endswith(
        "NOT_REPLICATION_PROBABILITY_OR_EFFECT_TRUTH"
    )


def test_closed_fixture_application_matches_frozen_expected() -> None:
    completed = _run(str(APPLICATION))

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    expected = json.loads(FIXTURE_EXPECTED.read_text(encoding="utf-8"))
    assert observed == expected
    assert observed["summary"] == {
        "case_count": 7,
        "pass_count": 1,
        "fail_count": 6,
    }
    by_id = {item["case_id"]: item for item in observed["case_results"]}
    assert by_id["FULL_PASS"]["replication_pass"] is True
    assert by_id["NONINFERIORITY_BOUNDARY_FAIL"]["failure_reasons"] == [
        "SYNTHETIC_A:NONINFERIORITY_LOWER_BOUND_NOT_ABOVE_MARGIN"
    ]
    assert by_id["PRIMARY_BRANCH_FAIL_NOT_RESCUED"]["failure_reasons"] == [
        "PRIMARY_POINT_BELOW_ONE_COHORT_YEAR"
    ]


def test_independent_verifier_matches_frozen_summary() -> None:
    completed = _run(str(VERIFIER))

    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    expected = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert observed == expected
    assert observed["status"] == (
        "PASS_ONE_OUTCOME_BLIND_G1_REPLICATION_TOLERANCE_AND_CLOSED_FIXTURE"
    )
    assert observed["policy_replay"]["future_interval_estimator_still_unresolved"]
    assert observed["access_accounting"][
        "g1_confirmatory_or_holdout_rows_read"
    ] == 0
    assert observed["authorization"]["g0_successor_binding"] == "NOT_AUTHORIZED"


def test_forbidden_input_flag_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE_INPUT.read_text(encoding="utf-8"))
    payload["contains_g1_confirmatory_or_holdout_rows"] = True
    mutated = tmp_path / "forbidden.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run(str(APPLICATION), "--input", str(mutated))

    assert completed.returncode != 0
    assert "contains_g1_confirmatory_or_holdout_rows" in completed.stderr


def test_missing_g1_field_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(FIXTURE_INPUT.read_text(encoding="utf-8"))
    payload["cases"][0]["fields"].pop()
    mutated = tmp_path / "two_fields.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")

    completed = _run(str(APPLICATION), "--input", str(mutated))

    assert completed.returncode != 0
    assert "target field count" in completed.stderr


def test_contract_mutation_is_rejected_before_application(tmp_path: Path) -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    contract["precommitment_choices"]["g1_replication_tolerance"]["value"] = 0.5
    mutated = tmp_path / "mutated_contract.yaml"
    mutated.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")

    completed = _run(str(APPLICATION), "--contract", str(mutated))

    assert completed.returncode != 0
    assert "contract hash drift" in completed.stderr


def test_documented_claim_and_access_boundaries_match_artifacts() -> None:
    note = NOTE.read_text(encoding="utf-8")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    for path in (CONTRACT, APPLICATION, VERIFIER, SUMMARY):
        assert _sha256(path) in note
    assert "one cohort-year" in note
    assert "not a literature-mandated constant" in note
    assert "does not complete any of the fifteen required sections" in note
    assert "Gate v61 therefore remains" in note
    assert summary["authorization"]["scientific_effect"] == "NOT_ESTABLISHED"
    assert summary["access_accounting"]["discovery_rows_read"] == 0
    assert summary["access_accounting"]["production_rows_read"] == 0
