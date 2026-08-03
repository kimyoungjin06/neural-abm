#!/usr/bin/env python3
"""Independently verify the outcome-blind G1 replication-tolerance package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml"
)
APPLICATION_TOOL = (
    REPO_ROOT
    / "scripts/apply_pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.py"
)
FIXTURE_INPUT = (
    REPO_ROOT
    / "tests/fixtures/pivot_feedback_g0_g1_replication_tolerance_v1.input.json"
)
FIXTURE_EXPECTED = (
    REPO_ROOT
    / "tests/fixtures/pivot_feedback_g0_g1_replication_tolerance_v1.expected.json"
)
CONTRACT_SHA256 = "1bcf91d197a9b927286d43f3505528f3f3f7fb022ffd4fe89d95cb24c041184d"
APPLICATION_SHA256 = "30d9fe0c6f443bf11eccf9d5eb3b19dd279478538f37624dd58f0b8d5126357a"
FIXTURE_INPUT_SHA256 = "731f268d840f058cd1fc9524eeeb85cdbdd86f9f943b599a0461eb2338ed6045"
FIXTURE_EXPECTED_SHA256 = "bc7782f972646094556298e1668e64703c60da598ff6373422694a35113f00bb"
CONTRACT_SCHEMA = "pivot_feedback_g0_g1_replication_tolerance_precommitment.v1"
CONTRACT_STATUS = (
    "FROZEN_SINGLE_G1_REPLICATION_TOLERANCE_PENDING_INDEPENDENT_VALIDATION_"
    "NO_G0_PROMOTION"
)
VERIFY_SCHEMA = "pivot_feedback_g0_g1_replication_tolerance_precommitment_summary.v1"
VERIFY_STATUS = (
    "PASS_ONE_OUTCOME_BLIND_G1_REPLICATION_TOLERANCE_AND_CLOSED_FIXTURE"
)
EXPECTED_BINDINGS = {
    "authorization_parent": {
        "path": "experiments/configs/pivot_feedback_gate_state_v61.yaml",
        "sha256": (
            "253c661fd529feb0d70733174197be449dfd27403b5881c6510bd7c8257c2715"
        ),
        "schema": "pivot_feedback_gate_state.v61",
        "status": (
            "VALID_G0_V4_TWELVE_RESOLVED_SEVEN_PENDING_G1_TOLERANCE_"
            "PRECOMMITMENT_AUTHORIZED"
        ),
        "format": "yaml",
    },
    "immutable_g0_v4": {
        "path": "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml",
        "sha256": (
            "cd49a31ab67e7a66d038b3198493df8b3a309aa14ee9de80410557a194581ce4"
        ),
        "schema": "pivot_feedback_g0_measurement_freeze.v5",
        "status": (
            "PENDING_G0_V4_TWELVE_RESOLVED_SEVEN_CHOICES_AND_FIFTEEN_"
            "SECTIONS_NO_SEAL"
        ),
        "format": "yaml",
    },
    "prerequisite_closure_plan": {
        "path": (
            "experiments/configs/pivot_feedback_g0_prerequisite_closure_plan_v1.yaml"
        ),
        "sha256": (
            "45db08ea9e51d861270c1be4b8ac675e56e30df7b57268f515fd665708b2c4fc"
        ),
        "schema": "pivot_feedback_g0_prerequisite_closure_plan.v1",
        "status": "PENDING_INDEPENDENT_VALIDATION_FOUR_WORKSTREAM_PLAN_NO_EXECUTION",
        "format": "yaml",
    },
    "empirical_protocol": {
        "path": (
            "docs/case-studies/researcher-pivot/"
            "pivot-feedback-empirical-nabm-protocol.md"
        ),
        "sha256": (
            "731e39dec4c7be5c29a0f0404b48992ad0caef152062d1d1d7876acc07064746"
        ),
        "schema": None,
        "status": None,
        "format": "text",
    },
}
REQUIRED_CASE_IDS = {
    "FULL_PASS",
    "NONINFERIORITY_BOUNDARY_FAIL",
    "FIELD_DIRECTION_ZERO_FAIL",
    "LEAVE_ONE_FIELD_OUT_DIRECTION_FAIL",
    "ALTERNATIVE_PIVOT_DIRECTION_FAIL",
    "PRIMARY_BRANCH_FAIL_NOT_RESCUED",
    "INSUFFICIENT_EPISODE_SUPPORT_FAIL",
}


class G1ReplicationToleranceVerificationError(RuntimeError):
    """Fail-closed G1 replication-tolerance verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G1ReplicationToleranceVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G1ReplicationToleranceVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G1ReplicationToleranceVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _binding_spec(contract: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key in {"authorization_parent", "immutable_g0_v4"}:
        return _mapping(contract.get(key), key)
    sources = _mapping(contract.get("bound_design_sources"), "bound sources")
    return _mapping(sources.get(key), key)


def _validate_binding(contract: Mapping[str, Any], key: str) -> Any:
    declared = _binding_spec(contract, key)
    expected = EXPECTED_BINDINGS[key]
    require(declared.get("relative_path") == expected["path"], f"{key} path")
    require(declared.get("sha256") == expected["sha256"], f"{key} hash")
    require(declared.get("mutation_allowed") is False, f"{key} mutation")
    path = _safe_repo_path(declared["relative_path"], key)
    require(_sha256(path) == expected["sha256"], f"{key} byte hash")
    if expected["format"] == "text":
        return path.read_text(encoding="utf-8")

    document = _read_yaml(path, key)
    require(
        declared.get("required_schema_version") == expected["schema"],
        f"{key} required schema",
    )
    require(document.get("schema_version") == expected["schema"], f"{key} schema")
    require(
        declared.get("required_status") == expected["status"],
        f"{key} required status",
    )
    require(document.get("status") == expected["status"], f"{key} status")
    return document


def _validate_source_semantics(documents: Mapping[str, Any]) -> dict[str, Any]:
    gate = _mapping(documents["authorization_parent"], "gate v61")
    scope = _mapping(
        gate.get("authorized_next_precommitment_scope"), "authorized scope"
    )
    require(scope.get("exact_choice_ids") == ["g1_replication_tolerance"], "scope")
    for key in (
        "must_bind_this_gate_byte_for_byte",
        "must_bind_g0_v4_byte_for_byte",
        "must_keep_value_outcome_blind",
        "must_define_estimand_scale_direction_interval_and_failure_rule",
        "must_label_judgment_based_value_as_policy_not_empirical_truth",
        "must_include_independent_closed_fixture_validation",
    ):
        require(scope.get(key) is True, key)
    for key in (
        "may_write_or_modify_g0_v4",
        "may_bind_value_into_g0_successor",
        "may_resolve_g2_or_g3_choices",
        "may_complete_required_sections",
        "may_apply_rule_to_g1_or_production_data",
        "may_read_g1_confirmatory_or_holdout_rows",
        "may_pass_g0",
        "may_fit_model",
        "may_open_g4",
    ):
        require(scope.get(key) is False, key)

    g0 = _mapping(documents["immutable_g0_v4"], "G0 v4")
    resolution = _mapping(g0.get("resolution"), "G0 resolution")
    unresolved = _mapping(
        resolution.get("unresolved_quantitative_choices"), "unresolved choices"
    )
    require(len(unresolved) == 7, "G0 remaining choice count")
    require(unresolved.get("g1_replication_tolerance") is None, "predecessor choice")
    require(resolution.get("freeze_ready_required_section_count") == 0, "sections")
    lifecycle = _mapping(g0.get("lifecycle"), "G0 lifecycle")
    require(lifecycle.get("pass_g0") is False, "G0 pass")
    gate_result = _mapping(
        _mapping(gate.get("verified_g0_v4_binding"), "verified G0 v4").get(
            "result"
        ),
        "verified G0 v4 result",
    )
    require(
        gate_result.get("confirmatory_or_holdout_rows_read") == 0,
        "G0 access",
    )

    plan = _mapping(documents["prerequisite_closure_plan"], "closure plan")
    workstreams = _mapping(plan.get("workstreams"), "workstreams")
    quantitative = _mapping(
        workstreams.get("SEVENTEEN_QUANTITATIVE_CHOICE_PRECOMMITMENTS"),
        "quantitative workstream",
    )
    packages = quantitative.get("packages")
    require(isinstance(packages, list), "quantitative packages")
    g1_packages = [item for item in packages if item.get("package_id") == "G1_REPLICATION"]
    require(len(g1_packages) == 1, "G1 package")
    require(g1_packages[0].get("choice_ids") == ["g1_replication_tolerance"], "G1 ID")
    require(g1_packages[0].get("current_values") == [None], "G1 null value")

    protocol = documents["empirical_protocol"]
    for token in (
        "The single confirmatory primary is H3's field-balanced timing gap",
        "least one cohort-year",
        "one-sided 95% episode-clustered interval",
        "leave-one-field-out estimates retain",
        "pre-specified alternative pivot construction",
        "A larger effect in one field",
    ):
        require(token in protocol, f"protocol token: {token}")
    return {
        "authorization_gate": "v61",
        "predecessor_unresolved_choice_count": len(unresolved),
        "predecessor_choice_is_null": True,
        "predecessor_pass_g0": False,
    }


def _validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")
    scope = _mapping(contract.get("scope"), "scope")
    require(scope.get("choice_ids") == ["g1_replication_tolerance"], "choice IDs")
    require(scope.get("choice_count") == 1, "choice count")
    require(scope.get("rule_freeze_is_not_interval_estimator_freeze") is True, "estimator boundary")
    require(scope.get("rule_freeze_is_not_g0_promotion") is True, "G0 boundary")

    choices = _mapping(contract.get("precommitment_choices"), "choices")
    require(set(choices) == {"g1_replication_tolerance"}, "single choice")
    rule = _mapping(choices["g1_replication_tolerance"], "tolerance rule")
    require(
        rule.get("rule_id")
        == "G1_FIELD_VERSUS_OTHER_FIELDS_ONE_COHORT_YEAR_NONINFERIORITY_V1",
        "rule ID",
    )
    require(rule.get("resolution_status") == "FROZEN_OUTCOME_BLIND_POLICY_RULE", "resolution")
    require(rule.get("value") == 1.0, "value")
    require(rule.get("unit") == "COHORT_YEAR", "unit")

    estimand = _mapping(rule.get("estimand"), "estimand")
    require(estimand.get("episode_value_symbol") == "delta_j", "episode symbol")
    require(estimand.get("episode_value_grid_cohort_years") == [-2, -1, 0, 1, 2], "grid")
    require(estimand.get("replication_contrast_symbol") == "rho_f", "contrast")
    require(estimand.get("scale") == "ADDITIVE_COHORT_YEARS", "scale")
    require(estimand.get("target_field_count") == 3, "field count")
    require(estimand.get("minimum_independent_episodes_per_field") == 3, "episode count")

    interval = _mapping(rule.get("interval_rule"), "interval")
    require(interval.get("interval") == "ONE_SIDED_95_PERCENT_LOWER_CONFIDENCE_BOUND", "interval")
    require(interval.get("one_sided_alpha") == 0.05, "alpha")
    require(interval.get("lower_noninferiority_margin_cohort_years") == -1.0, "margin")
    require(interval.get("interval_boundary_counts_as_pass") is False, "margin boundary")
    require(interval.get("this_contract_selects_interval_estimator") is False, "estimator selection")

    direction = _mapping(rule.get("direction_rule"), "direction")
    for key in (
        "every_field_point_estimate_strictly_positive",
        "every_leave_one_field_out_point_estimate_strictly_positive",
        "every_alternative_pivot_field_point_estimate_strictly_positive",
    ):
        require(direction.get(key) is True, key)
    require(direction.get("zero_counts_as_positive") is False, "zero direction")

    primary = _mapping(rule.get("primary_branch_guard"), "primary")
    require(primary.get("field_balanced_point_estimate_minimum_cohort_years") == 1.0, "primary minimum")
    require(primary.get("point_boundary_counts_as_pass") is True, "primary boundary")
    require(primary.get("primary_one_sided_95_percent_lower_bound_strictly_greater_than_zero") is True, "primary interval")
    require(primary.get("replication_rule_may_rescue_failed_primary") is False, "primary rescue")
    require(primary.get("primary_rule_may_rescue_failed_replication") is False, "replication rescue")

    policy = _mapping(contract.get("policy_value_boundary"), "policy boundary")
    require(policy.get("one_cohort_year_is_literature_mandated") is False, "mandate")
    require(policy.get("one_cohort_year_is_power_or_precision_guarantee") is False, "power")
    require(policy.get("value_is_precommitted_design_policy_not_empirical_truth") is True, "policy label")

    lifecycle = _mapping(contract.get("lifecycle"), "lifecycle")
    require(lifecycle.get("g0_v4_changed") is False, "G0 changed")
    require(lifecycle.get("unresolved_g0_choice_count_before") == 7, "before count")
    require(lifecycle.get("resolved_choice_count_in_this_contract") == 1, "resolved count")
    require(lifecycle.get("unresolved_g0_choice_count_after_future_binding") == 6, "future count")
    require(lifecycle.get("required_sections_completed") is False, "section completion")
    require(lifecycle.get("confirmatory_or_holdout_rows_read") == 0, "row access")
    require(lifecycle.get("model_fitted") is False, "model")
    require(lifecycle.get("g4_open") is False, "G4")

    authorization = _mapping(contract.get("authorization"), "authorization")
    require(authorization.get("fixture_application_and_validation") == "AUTHORIZED_SYNTHETIC_ONLY", "fixture authorization")
    for key in (
        "discovery_production_application",
        "g1_confirmatory_application",
        "g0_successor_binding",
        "g2_or_g3_choice_precommitment",
        "required_section_completion",
        "confirmatory_or_holdout_access",
        "g0_measurement_freeze",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", key)
    return {
        "choice_id": "g1_replication_tolerance",
        "value_cohort_years": 1.0,
        "contrast": "FIELD_VALUE_MINUS_OTHER_FIELDS_REFERENCE",
        "lower_margin_cohort_years": -1.0,
        "interval": "ONE_SIDED_95_PERCENT_LOWER_CONFIDENCE_BOUND",
        "strict_boundary": True,
        "future_interval_estimator_still_unresolved": True,
    }


def _finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} numeric")
    numeric = float(value)
    require(math.isfinite(numeric), f"{label} finite")
    return numeric


def _independent_case_replay(case_value: Any) -> tuple[str, bool, list[str]]:
    case = _mapping(case_value, "fixture case")
    case_id = case.get("case_id")
    require(isinstance(case_id, str) and case_id, "fixture case ID")
    primary_point = _finite_number(case.get("primary_point_estimate_cohort_years"), "primary point")
    primary_lower = _finite_number(case.get("primary_one_sided_95_lower_bound_cohort_years"), "primary lower")
    failures: list[str] = []
    if primary_point < 1.0:
        failures.append("PRIMARY_POINT_BELOW_ONE_COHORT_YEAR")
    if primary_lower <= 0:
        failures.insert(0, "PRIMARY_LOWER_BOUND_NOT_STRICTLY_POSITIVE")

    fields = case.get("fields")
    require(isinstance(fields, list) and len(fields) == 3, "fixture fields")
    seen: set[str] = set()
    for field_value in fields:
        field = _mapping(field_value, "fixture field")
        field_id = field.get("synthetic_field_id")
        require(isinstance(field_id, str) and field_id.startswith("SYNTHETIC_"), "field ID")
        require(field_id not in seen, "duplicate field ID")
        seen.add(field_id)
        episodes = field.get("independent_episode_count")
        require(type(episodes) is int and episodes >= 0, "episode count")
        point = _finite_number(field.get("field_point_estimate_cohort_years"), "field point")
        lower = _finite_number(field.get("field_minus_other_fields_one_sided_95_lower_bound_cohort_years"), "field lower")
        leave_one_out = _finite_number(field.get("leave_one_field_out_point_estimate_cohort_years"), "leave-one-out")
        alternative = _finite_number(field.get("alternative_pivot_field_point_estimate_cohort_years"), "alternative")
        if episodes < 3:
            failures.append(f"{field_id}:INSUFFICIENT_INDEPENDENT_EPISODES")
        if point <= 0:
            failures.append(f"{field_id}:FIELD_DIRECTION_NOT_STRICTLY_POSITIVE")
        if lower <= -1.0:
            failures.append(f"{field_id}:NONINFERIORITY_LOWER_BOUND_NOT_ABOVE_MARGIN")
        if leave_one_out <= 0:
            failures.append(f"{field_id}:LEAVE_ONE_FIELD_OUT_NOT_STRICTLY_POSITIVE")
        if alternative <= 0:
            failures.append(f"{field_id}:ALTERNATIVE_PIVOT_NOT_STRICTLY_POSITIVE")
    return case_id, not failures, failures


def _validate_fixture(contract: Mapping[str, Any]) -> dict[str, Any]:
    fixture = _mapping(contract.get("fixture_contract"), "fixture contract")
    require(fixture.get("synthetic_only") is True, "synthetic fixture")
    require(fixture.get("may_promote_to_production_evidence") is False, "promotion")
    require(set(fixture.get("must_cover", [])) == REQUIRED_CASE_IDS, "fixture coverage declaration")
    require(_sha256(APPLICATION_TOOL) == APPLICATION_SHA256, "application hash")
    require(_sha256(FIXTURE_INPUT) == FIXTURE_INPUT_SHA256, "fixture input hash")
    require(_sha256(FIXTURE_EXPECTED) == FIXTURE_EXPECTED_SHA256, "fixture expected hash")

    payload = _read_json(FIXTURE_INPUT, "fixture input")
    require(payload.get("status") == "SYNTHETIC_CLOSED_FIXTURE", "fixture status")
    for key in (
        "contains_discovery_rows",
        "contains_g1_confirmatory_or_holdout_rows",
        "contains_field_or_episode_identities",
        "contains_model_scores_or_g4_labels",
    ):
        require(payload.get(key) is False, key)
    cases = payload.get("cases")
    require(isinstance(cases, list), "fixture cases")
    independent = [_independent_case_replay(case) for case in cases]
    require({item[0] for item in independent} == REQUIRED_CASE_IDS, "fixture case IDs")

    completed = subprocess.run(
        [
            sys.executable,
            str(APPLICATION_TOOL),
            "--contract",
            str(DEFAULT_CONTRACT),
            "--input",
            str(FIXTURE_INPUT),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    require(completed.returncode == 0, f"fixture application: {completed.stderr}")
    try:
        observed = _mapping(json.loads(completed.stdout), "application output")
    except json.JSONDecodeError as error:
        raise G1ReplicationToleranceVerificationError(
            f"invalid application output: {error}"
        ) from error
    expected = _read_json(FIXTURE_EXPECTED, "fixture expected")
    require(observed == expected, "application output drift")
    by_id = {
        item["case_id"]: item
        for item in observed.get("case_results", [])
        if isinstance(item, Mapping)
    }
    for case_id, passed, failures in independent:
        require(by_id[case_id].get("replication_pass") is passed, f"pass {case_id}")
        require(by_id[case_id].get("failure_reasons") == failures, f"failures {case_id}")
    summary = _mapping(observed.get("summary"), "fixture summary")
    require(summary == {"case_count": 7, "pass_count": 1, "fail_count": 6}, "summary")
    access = _mapping(observed.get("access_accounting"), "fixture access")
    require(access.get("synthetic_cases_read") == 7, "synthetic cases")
    require(access.get("discovery_rows_read") == 0, "discovery access")
    require(access.get("g1_confirmatory_or_holdout_rows_read") == 0, "G1 access")
    require(access.get("production_rows_read") == 0, "production access")
    require(access.get("model_fitted") is False, "fixture model")
    require(access.get("g4_open") is False, "fixture G4")
    return {
        "case_count": summary["case_count"],
        "pass_count": summary["pass_count"],
        "fail_count": summary["fail_count"],
        "case_ids": [item[0] for item in independent],
        "strict_minus_one_boundary_failure_verified": True,
        "zero_direction_failures_verified": True,
        "primary_failure_not_rescued_verified": True,
        "insufficient_episode_support_failure_verified": True,
    }


def verify(contract_path: Path) -> dict[str, Any]:
    require(contract_path == DEFAULT_CONTRACT.resolve(), "unexpected contract path")
    require(_sha256(contract_path) == CONTRACT_SHA256, "contract hash drift")
    contract = _read_yaml(contract_path, "contract")
    documents = {
        key: _validate_binding(contract, key) for key in EXPECTED_BINDINGS
    }
    source_replay = _validate_source_semantics(documents)
    policy_replay = _validate_contract(contract)
    fixture_replay = _validate_fixture(contract)
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "source_replay": source_replay,
        "policy_replay": policy_replay,
        "fixture_replay": fixture_replay,
        "access_accounting": {
            "repository_yaml_documents_read": 4,
            "repository_text_documents_read": 1,
            "synthetic_fixture_cases_read": 7,
            "discovery_rows_read": 0,
            "g1_confirmatory_or_holdout_rows_read": 0,
            "production_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "repository_artifacts": {
            "contract": {
                "relative_path": contract_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(contract_path),
            },
            "application_tool": {
                "relative_path": APPLICATION_TOOL.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(APPLICATION_TOOL),
            },
            "fixture_input": {
                "relative_path": FIXTURE_INPUT.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(FIXTURE_INPUT),
            },
            "fixture_expected": {
                "relative_path": FIXTURE_EXPECTED.relative_to(REPO_ROOT).as_posix(),
                "sha256": _sha256(FIXTURE_EXPECTED),
            },
        },
        "authorization": {
            "single_choice_precommitment": "COMPLETE_INDEPENDENTLY_VERIFIED",
            "fixture_application": "COMPLETE_SYNTHETIC_ONLY",
            "g0_successor_binding": "NOT_AUTHORIZED",
            "g1_or_production_application": "NOT_AUTHORIZED",
            "g2_or_g3_choice_resolution": "NOT_AUTHORIZED",
            "required_section_completion": "NOT_AUTHORIZED",
            "confirmatory_or_holdout_access": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The independently replayed package freezes one outcome-blind "
            "one-cohort-year G1 field-replication noninferiority tolerance and "
            "validates only its closed synthetic decision logic. The value is "
            "a policy, not an empirical truth, replication probability, power "
            "or precision guarantee, interval-estimator freeze, G1 result, G0 "
            "pass, transport result, causality, or scientific effect. G0 v4 "
            "remains unchanged, no G0 successor is authorized, no discovery, "
            "G1, production, confirmatory, or holdout row was read, no model "
            "was fitted, and G4 remains unopened."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = verify(args.contract.resolve())
    except (G1ReplicationToleranceVerificationError, OSError) as error:
        raise SystemExit(str(error)) from error
    rendered = json.dumps(output, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
