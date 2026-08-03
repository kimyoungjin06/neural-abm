#!/usr/bin/env python3
"""Apply the frozen G1 replication-tolerance rule to a closed fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_g1_replication_tolerance_precommitment_v1.yaml"
)
DEFAULT_FIXTURE = (
    REPO_ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_g1_replication_tolerance_v1.input.json"
)
CONTRACT_SHA256 = "1bcf91d197a9b927286d43f3505528f3f3f7fb022ffd4fe89d95cb24c041184d"
CONTRACT_SCHEMA = "pivot_feedback_g0_g1_replication_tolerance_precommitment.v1"
CONTRACT_STATUS = (
    "FROZEN_SINGLE_G1_REPLICATION_TOLERANCE_PENDING_INDEPENDENT_VALIDATION_"
    "NO_G0_PROMOTION"
)
INPUT_SCHEMA = "pivot_feedback_g0_g1_replication_tolerance_fixture_input.v1"
OUTPUT_SCHEMA = "pivot_feedback_g0_g1_replication_tolerance_fixture_output.v1"
OUTPUT_STATUS = "PASS_CLOSED_SYNTHETIC_G1_REPLICATION_TOLERANCE_RULE_APPLICATION"
INPUT_STATUS = "SYNTHETIC_CLOSED_FIXTURE"
ROUND_DIGITS = 12
ROOT_FIELDS = {
    "schema_version",
    "status",
    "contains_discovery_rows",
    "contains_g1_confirmatory_or_holdout_rows",
    "contains_field_or_episode_identities",
    "contains_model_scores_or_g4_labels",
    "cases",
}
CASE_FIELDS = {
    "case_id",
    "primary_point_estimate_cohort_years",
    "primary_one_sided_95_lower_bound_cohort_years",
    "fields",
}
FIELD_FIELDS = {
    "synthetic_field_id",
    "independent_episode_count",
    "field_point_estimate_cohort_years",
    "field_minus_other_fields_one_sided_95_lower_bound_cohort_years",
    "leave_one_field_out_point_estimate_cohort_years",
    "alternative_pivot_field_point_estimate_cohort_years",
}


class G1ReplicationToleranceApplicationError(RuntimeError):
    """Fail-closed G1 replication-tolerance application error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G1ReplicationToleranceApplicationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G1ReplicationToleranceApplicationError(
            f"invalid {label}: {error}"
        ) from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G1ReplicationToleranceApplicationError(
            f"invalid {label}: {error}"
        ) from error


def _finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    numeric = float(value)
    require(math.isfinite(numeric), f"{label} must be finite")
    return numeric


def _exact_int(value: Any, label: str, *, minimum: int) -> int:
    require(type(value) is int, f"{label} must be an integer")
    require(value >= minimum, f"{label} below minimum")
    return value


def _round(value: float) -> float:
    rounded = round(value, ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def _read_policy(contract_path: Path) -> dict[str, Any]:
    require(_sha256(contract_path) == CONTRACT_SHA256, "contract hash drift")
    contract = _read_yaml(contract_path, "contract")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")

    scope = _mapping(contract.get("scope"), "scope")
    require(scope.get("choice_ids") == ["g1_replication_tolerance"], "choice IDs")
    require(scope.get("choice_count") == 1, "choice count")
    for key in (
        "selection_uses_discovery_outcomes",
        "selection_uses_g1_confirmatory_outcomes",
        "selection_uses_holdout_outcomes",
        "selection_uses_g4_field_identity",
        "selection_uses_model_performance",
    ):
        require(scope.get(key) is False, key)

    choices = _mapping(contract.get("precommitment_choices"), "choices")
    require(set(choices) == {"g1_replication_tolerance"}, "single choice")
    rule = _mapping(choices["g1_replication_tolerance"], "tolerance rule")
    estimand = _mapping(rule.get("estimand"), "estimand")
    interval = _mapping(rule.get("interval_rule"), "interval rule")
    direction = _mapping(rule.get("direction_rule"), "direction rule")
    primary = _mapping(rule.get("primary_branch_guard"), "primary guard")

    require(rule.get("value") == 1.0, "tolerance value")
    require(rule.get("unit") == "COHORT_YEAR", "tolerance unit")
    require(estimand.get("scale") == "ADDITIVE_COHORT_YEARS", "estimand scale")
    require(estimand.get("target_field_count") == 3, "target field count")
    require(
        estimand.get("minimum_independent_episodes_per_field") == 3,
        "episode minimum",
    )
    require(interval.get("one_sided_alpha") == 0.05, "interval alpha")
    require(
        interval.get("lower_noninferiority_margin_cohort_years") == -1.0,
        "lower margin",
    )
    require(interval.get("interval_boundary_counts_as_pass") is False, "boundary")
    require(
        interval.get("this_contract_selects_interval_estimator") is False,
        "interval estimator boundary",
    )
    require(
        set(direction.values()) == {True, False},
        "direction boolean contract",
    )
    require(direction.get("zero_counts_as_positive") is False, "zero direction")
    require(
        primary.get("field_balanced_point_estimate_minimum_cohort_years") == 1.0,
        "primary point minimum",
    )
    require(primary.get("point_boundary_counts_as_pass") is True, "point boundary")
    require(primary.get("replication_rule_may_rescue_failed_primary") is False, "rescue")
    return {
        "tolerance_cohort_years": float(rule["value"]),
        "lower_margin_cohort_years": float(
            interval["lower_noninferiority_margin_cohort_years"]
        ),
        "target_field_count": int(estimand["target_field_count"]),
        "minimum_independent_episodes_per_field": int(
            estimand["minimum_independent_episodes_per_field"]
        ),
        "primary_point_minimum_cohort_years": float(
            primary["field_balanced_point_estimate_minimum_cohort_years"]
        ),
    }


def _evaluate_field(
    value: Any,
    *,
    minimum_episodes: int,
    lower_margin: float,
) -> tuple[dict[str, Any], list[str]]:
    field = _mapping(value, "field")
    require(set(field) == FIELD_FIELDS, "field keys")
    field_id = field.get("synthetic_field_id")
    require(
        isinstance(field_id, str) and field_id.startswith("SYNTHETIC_"),
        "synthetic field ID",
    )
    episodes = _exact_int(
        field.get("independent_episode_count"),
        "independent episode count",
        minimum=0,
    )
    point = _finite_number(
        field.get("field_point_estimate_cohort_years"), "field point"
    )
    lower = _finite_number(
        field.get(
            "field_minus_other_fields_one_sided_95_lower_bound_cohort_years"
        ),
        "field contrast lower bound",
    )
    leave_one_out = _finite_number(
        field.get("leave_one_field_out_point_estimate_cohort_years"),
        "leave-one-field-out point",
    )
    alternative = _finite_number(
        field.get("alternative_pivot_field_point_estimate_cohort_years"),
        "alternative pivot point",
    )

    failures: list[str] = []
    if episodes < minimum_episodes:
        failures.append(f"{field_id}:INSUFFICIENT_INDEPENDENT_EPISODES")
    if point <= 0:
        failures.append(f"{field_id}:FIELD_DIRECTION_NOT_STRICTLY_POSITIVE")
    if lower <= lower_margin:
        failures.append(f"{field_id}:NONINFERIORITY_LOWER_BOUND_NOT_ABOVE_MARGIN")
    if leave_one_out <= 0:
        failures.append(f"{field_id}:LEAVE_ONE_FIELD_OUT_NOT_STRICTLY_POSITIVE")
    if alternative <= 0:
        failures.append(f"{field_id}:ALTERNATIVE_PIVOT_NOT_STRICTLY_POSITIVE")

    return (
        {
            "synthetic_field_id": field_id,
            "episode_support_pass": episodes >= minimum_episodes,
            "field_direction_pass": point > 0,
            "noninferiority_interval_pass": lower > lower_margin,
            "leave_one_field_out_direction_pass": leave_one_out > 0,
            "alternative_pivot_direction_pass": alternative > 0,
        },
        failures,
    )


def _evaluate_case(value: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    case = _mapping(value, "case")
    require(set(case) == CASE_FIELDS, "case keys")
    case_id = case.get("case_id")
    require(isinstance(case_id, str) and case_id, "case ID")
    primary_point = _finite_number(
        case.get("primary_point_estimate_cohort_years"), "primary point"
    )
    primary_lower = _finite_number(
        case.get("primary_one_sided_95_lower_bound_cohort_years"),
        "primary lower bound",
    )
    fields = case.get("fields")
    require(isinstance(fields, list), "fields must be a list")
    require(len(fields) == policy["target_field_count"], "target field count")

    field_results: list[dict[str, Any]] = []
    failures: list[str] = []
    seen_field_ids: set[str] = set()
    for field_value in fields:
        result, field_failures = _evaluate_field(
            field_value,
            minimum_episodes=policy["minimum_independent_episodes_per_field"],
            lower_margin=policy["lower_margin_cohort_years"],
        )
        field_id = result["synthetic_field_id"]
        require(field_id not in seen_field_ids, "duplicate synthetic field ID")
        seen_field_ids.add(field_id)
        field_results.append(result)
        failures.extend(field_failures)

    primary_point_pass = primary_point >= policy["primary_point_minimum_cohort_years"]
    primary_interval_pass = primary_lower > 0
    if not primary_point_pass:
        failures.insert(0, "PRIMARY_POINT_BELOW_ONE_COHORT_YEAR")
    if not primary_interval_pass:
        failures.insert(0, "PRIMARY_LOWER_BOUND_NOT_STRICTLY_POSITIVE")

    return {
        "case_id": case_id,
        "primary_point_estimate_cohort_years": _round(primary_point),
        "primary_one_sided_95_lower_bound_cohort_years": _round(primary_lower),
        "primary_point_pass": primary_point_pass,
        "primary_interval_pass": primary_interval_pass,
        "field_results": field_results,
        "replication_pass": not failures,
        "failure_reasons": failures,
    }


def apply_contract(contract_path: Path, input_path: Path) -> dict[str, Any]:
    policy = _read_policy(contract_path)
    payload = _read_json(input_path, "input")
    require(set(payload) == ROOT_FIELDS, "input root keys")
    require(payload.get("schema_version") == INPUT_SCHEMA, "input schema")
    require(payload.get("status") == INPUT_STATUS, "input status")
    for key in (
        "contains_discovery_rows",
        "contains_g1_confirmatory_or_holdout_rows",
        "contains_field_or_episode_identities",
        "contains_model_scores_or_g4_labels",
    ):
        require(payload.get(key) is False, key)
    cases = payload.get("cases")
    require(isinstance(cases, list) and cases, "cases")

    results: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for value in cases:
        result = _evaluate_case(value, policy)
        case_id = result["case_id"]
        require(case_id not in seen_case_ids, "duplicate case ID")
        seen_case_ids.add(case_id)
        results.append(result)

    pass_count = sum(result["replication_pass"] for result in results)
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "contract": {
            "relative_path": contract_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(contract_path),
        },
        "policy": {
            "choice_id": "g1_replication_tolerance",
            "tolerance_cohort_years": policy["tolerance_cohort_years"],
            "lower_noninferiority_margin_cohort_years": policy[
                "lower_margin_cohort_years"
            ],
            "interval": "ONE_SIDED_95_PERCENT_LOWER_CONFIDENCE_BOUND",
            "boundary_counts_as_pass": False,
            "policy_not_empirical_truth": True,
        },
        "case_results": results,
        "summary": {
            "case_count": len(results),
            "pass_count": pass_count,
            "fail_count": len(results) - pass_count,
        },
        "access_accounting": {
            "synthetic_cases_read": len(results),
            "discovery_rows_read": 0,
            "g1_confirmatory_or_holdout_rows_read": 0,
            "production_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "claim_boundary": (
            "This is a closed synthetic decision-rule application. It validates "
            "a one-cohort-year field noninferiority policy and strict direction "
            "guards, not an interval estimator, G1 result, replication "
            "probability, G0 pass, transport result, causality, or effect."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = apply_contract(args.contract.resolve(), args.input.resolve())
    except (G1ReplicationToleranceApplicationError, OSError) as error:
        raise SystemExit(str(error)) from error
    rendered = json.dumps(output, indent=2, sort_keys=False) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
