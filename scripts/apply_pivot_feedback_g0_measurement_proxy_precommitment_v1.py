#!/usr/bin/env python3
"""Apply frozen G0 measurement/proxy rules to explicit discovery aggregates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_measurement_proxy_precommitment_v1.yaml"
)
DEFAULT_FIXTURE = (
    REPO_ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_measurement_proxy_discovery_v1.input.json"
)
CONTRACT_SHA256 = "2e4d90a9541cd9973b8a69bf5fb34f6f35c3d1570a12a4c46b37cac77cca6b5f"
CONTRACT_SCHEMA = "pivot_feedback_g0_measurement_proxy_precommitment.v1"
CONTRACT_STATUS = (
    "FROZEN_FOUR_MEASUREMENT_PROXY_CHOICES_PENDING_INDEPENDENT_VALIDATION_"
    "NO_G0_PROMOTION"
)
INPUT_SCHEMA = "pivot_feedback_g0_measurement_proxy_discovery_input.v1"
OUTPUT_SCHEMA = "pivot_feedback_g0_measurement_proxy_discovery_output.v1"
OUTPUT_STATUS = "PASS_FROZEN_MEASUREMENT_PROXY_RULE_APPLICATION_DISCOVERY_ONLY"
PERMITTED_INPUT_STATUSES = {
    "SYNTHETIC_CLOSED_FIXTURE",
    "DISCOVERY_ONLY_AGGREGATED_NO_CONFIRMATORY_ROWS",
}
ROUND_DIGITS = 12


class MeasurementProxyApplicationError(RuntimeError):
    """Fail-closed measurement/proxy rule application error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementProxyApplicationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeasurementProxyApplicationError(f"invalid {label}: {error}") from error


def _read_contract(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "invalid contract")
    require(_sha256(path) == CONTRACT_SHA256, "contract hash drift")
    try:
        contract = _mapping(
            yaml.safe_load(path.read_text(encoding="utf-8")), "contract"
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise MeasurementProxyApplicationError(f"invalid contract: {error}") from error
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")
    return contract


def _exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    require(type(value) is int, f"{label} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{label} below minimum")
    return value


def _finite_number(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    numeric = float(value)
    require(math.isfinite(numeric), f"{label} must be finite")
    return numeric


def _round(value: float) -> float:
    rounded = round(value, ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def _attention_rules(contract: Mapping[str, Any]) -> dict[str, Any]:
    choices = _mapping(contract.get("precommitment_choices"), "choices")
    rule = _mapping(choices.get("attention_surge_threshold"), "attention rule")
    require(
        rule.get("rule_id") == "ROBUST_INCUMBENT_ATTENTION_ACCELERATION_MAD_V1",
        "attention rule ID",
    )
    require(rule.get("recent_window_offsets") == [-3, -2, -1], "recent window")
    require(rule.get("reference_window_offsets") == [-6, -5, -4], "reference window")
    require(rule.get("window_year_count") == 3, "window year count")
    require(
        rule.get("acceleration_score")
        == "LOG1P_RECENT_ANNUAL_MEAN_MINUS_LOG1P_REFERENCE_ANNUAL_MEAN",
        "acceleration score",
    )
    require(
        rule.get("normalization_group")
        == "PRIMARY_FIELD_ID_X_DECISION_YEAR_ACROSS_ELIGIBLE_TOPICS",
        "normalization group",
    )
    require(rule.get("location_estimator") == "MEDIAN", "location estimator")
    require(
        rule.get("scale_estimator") == "MEDIAN_ABSOLUTE_DEVIATION",
        "scale estimator",
    )
    require(
        rule.get("mad_zero_action") == "NO_SURGE_FAIL_CLOSED", "MAD zero action"
    )
    return rule


def _citation_rules(contract: Mapping[str, Any]) -> dict[str, Any]:
    choices = _mapping(contract.get("precommitment_choices"), "choices")
    public = _mapping(choices.get("public_success_threshold"), "public rule")
    maturation = _mapping(
        choices.get("citation_maturation_lag_years"), "maturation rule"
    )
    require(
        public.get("rule_id")
        == "FIELD_YEAR_TOP_DECILE_AGE3_CITATION_PROXY_INCLUSIVE_TIES_V1",
        "public success rule ID",
    )
    require(public.get("reference_cohort") == "PRIMARY_FIELD_ID_X_PUBLICATION_YEAR", "citation cohort")
    require(public.get("target_fraction") == 0.10, "target fraction")
    require(
        public.get("boundary_tie_rule") == "INCLUDE_ALL_WORKS_AT_OR_ABOVE_CUTOFF",
        "boundary tie rule",
    )
    require(
        public.get("minimum_reference_cohort_for_decile_calculation") == 10,
        "minimum reference cohort",
    )
    require(maturation.get("value") == 3, "maturation lag")
    require(public.get("citation_horizon_years") == 3, "citation horizon")
    return public


def _margin_rules(contract: Mapping[str, Any]) -> dict[str, Any]:
    choices = _mapping(contract.get("precommitment_choices"), "choices")
    margins = _mapping(
        choices.get("negative_control_equivalence_margins"), "margin rule"
    )
    require(margins.get("symmetric_practical_null_lower") == -0.01, "lower margin")
    require(margins.get("symmetric_practical_null_upper") == 0.01, "upper margin")
    require(margins.get("equivalence_test") == "TWO_ONE_SIDED_TESTS", "TOST rule")
    require(margins.get("one_sided_alpha") == 0.05, "TOST alpha")
    require(
        margins.get("equivalent_confidence_interval") == "TWO_SIDED_90_PERCENT",
        "equivalence interval",
    )
    return margins


def _apply_attention(
    rows_value: Any, rule: Mapping[str, Any]
) -> dict[str, Any]:
    require(isinstance(rows_value, list) and rows_value, "attention rows")
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[int, int, int]] = set()
    for value in rows_value:
        row = _mapping(value, "attention row")
        require(
            set(row)
            == {
                "field_id",
                "topic_id",
                "decision_year",
                "recent_incumbent_attention_count",
                "reference_incumbent_attention_count",
            },
            "attention row fields",
        )
        field_id = _exact_int(row["field_id"], "field ID", minimum=1)
        topic_id = _exact_int(row["topic_id"], "topic ID", minimum=1)
        year = _exact_int(row["decision_year"], "decision year")
        require(year <= 2012, "post-discovery decision year")
        recent = _exact_int(row["recent_incumbent_attention_count"], "recent count", minimum=0)
        reference = _exact_int(
            row["reference_incumbent_attention_count"], "reference count", minimum=0
        )
        key = (field_id, topic_id, year)
        require(key not in seen, "duplicate attention row")
        seen.add(key)
        score = math.log1p(recent / 3) - math.log1p(reference / 3)
        groups[(field_id, year)].append(
            {
                "field_id": field_id,
                "topic_id": topic_id,
                "decision_year": year,
                "recent_incumbent_attention_count": recent,
                "reference_incumbent_attention_count": reference,
                "acceleration_score": score,
            }
        )

    output_groups: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    mad_zero_group_count = 0
    for (field_id, year), values in sorted(groups.items()):
        scores = [item["acceleration_score"] for item in values]
        median = float(statistics.median(scores))
        mad = float(statistics.median(abs(score - median) for score in scores))
        if mad == 0:
            mad_zero_group_count += 1
        surge_count = 0
        for item in sorted(values, key=lambda candidate: candidate["topic_id"]):
            score = float(item["acceleration_score"])
            modified_z = (
                None
                if mad == 0
                else float(rule["modified_z_multiplier"]) * (score - median) / mad
            )
            is_surge = bool(
                modified_z is not None
                and modified_z >= float(rule["primary_modified_z_threshold_inclusive"])
                and score > 0
                and item["recent_incumbent_attention_count"]
                > item["reference_incumbent_attention_count"]
            )
            surge_count += int(is_surge)
            output_rows.append(
                {
                    "field_id": field_id,
                    "topic_id": item["topic_id"],
                    "decision_year": year,
                    "recent_incumbent_attention_count": item[
                        "recent_incumbent_attention_count"
                    ],
                    "reference_incumbent_attention_count": item[
                        "reference_incumbent_attention_count"
                    ],
                    "acceleration_score": _round(score),
                    "modified_z": None if modified_z is None else _round(modified_z),
                    "primary_surge": is_surge,
                    "sensitivity_surge_z2_5": bool(
                        modified_z is not None and modified_z >= 2.5 and score > 0
                    ),
                    "sensitivity_surge_z3_5": bool(
                        modified_z is not None and modified_z >= 3.5 and score > 0
                    ),
                }
            )
        output_groups.append(
            {
                "field_id": field_id,
                "decision_year": year,
                "topic_count": len(values),
                "median_acceleration_score": _round(median),
                "median_absolute_deviation": _round(mad),
                "mad_zero_fail_closed": mad == 0,
                "primary_surge_count": surge_count,
            }
        )
    return {
        "rule_id": rule["rule_id"],
        "group_count": len(output_groups),
        "row_count": len(output_rows),
        "mad_zero_group_count": mad_zero_group_count,
        "primary_surge_count": sum(row["primary_surge"] for row in output_rows),
        "groups": output_groups,
        "rows": output_rows,
    }


def _apply_citations(rows_value: Any, rule: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(rows_value, list) and rows_value, "citation rows")
    groups: dict[tuple[int, int], list[dict[str, int]]] = defaultdict(list)
    seen: set[tuple[int, int, int]] = set()
    for value in rows_value:
        row = _mapping(value, "citation row")
        require(
            set(row)
            == {"field_id", "publication_year", "work_id", "citation_count_age_3"},
            "citation row fields",
        )
        field_id = _exact_int(row["field_id"], "field ID", minimum=1)
        year = _exact_int(row["publication_year"], "publication year")
        work_id = _exact_int(row["work_id"], "work ID", minimum=1)
        count = _exact_int(row["citation_count_age_3"], "citation count", minimum=0)
        key = (field_id, year, work_id)
        require(key not in seen, "duplicate citation work row")
        seen.add(key)
        groups[(field_id, year)].append(
            {
                "field_id": field_id,
                "publication_year": year,
                "work_id": work_id,
                "citation_count_age_3": count,
            }
        )

    output_groups: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    unresolved_group_count = 0
    inclusive_tie_inflation_total = 0
    minimum = int(rule["minimum_reference_cohort_for_decile_calculation"])
    target_fraction = _finite_number(rule["target_fraction"], "target fraction")
    for (field_id, year), values in sorted(groups.items()):
        count = len(values)
        if count < minimum:
            unresolved_group_count += 1
            target_count = None
            cutoff = None
            success_count = None
            inflation = None
            status = "UNRESOLVED_INSUFFICIENT_REFERENCE_COHORT"
        else:
            target_count = math.ceil(count * target_fraction)
            descending = sorted(
                (item["citation_count_age_3"] for item in values), reverse=True
            )
            cutoff = descending[target_count - 1]
            success_count = sum(
                item["citation_count_age_3"] >= cutoff for item in values
            )
            inflation = success_count - target_count
            inclusive_tie_inflation_total += inflation
            status = "CLASSIFIED_TOP_DECILE_ANCHORED_INCLUSIVE_TIES"
        output_groups.append(
            {
                "field_id": field_id,
                "publication_year": year,
                "reference_cohort_count": count,
                "target_count": target_count,
                "citation_cutoff_age_3": cutoff,
                "public_success_count": success_count,
                "inclusive_tie_inflation_count": inflation,
                "realized_public_success_fraction": (
                    None if success_count is None else _round(success_count / count)
                ),
                "status": status,
            }
        )
        for item in sorted(values, key=lambda candidate: candidate["work_id"]):
            output_rows.append(
                {
                    **item,
                    "public_success_proxy": (
                        None
                        if cutoff is None
                        else item["citation_count_age_3"] >= cutoff
                    ),
                }
            )
    return {
        "rule_id": rule["rule_id"],
        "group_count": len(output_groups),
        "row_count": len(output_rows),
        "unresolved_group_count": unresolved_group_count,
        "inclusive_tie_inflation_total": inclusive_tie_inflation_total,
        "groups": output_groups,
        "rows": output_rows,
    }


def run(
    input_path: Path = DEFAULT_FIXTURE, contract_path: Path = DEFAULT_CONTRACT
) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    input_path = input_path.resolve(strict=True)
    contract = _read_contract(contract_path)
    input_document = _read_json(input_path, "discovery input")
    require(input_document.get("schema_version") == INPUT_SCHEMA, "input schema")
    require(input_document.get("status") in PERMITTED_INPUT_STATUSES, "input status")
    require(
        set(input_document)
        == {
            "schema_version",
            "status",
            "attention_topic_year_rows",
            "citation_work_rows",
        },
        "input fields",
    )
    attention_rule = _attention_rules(contract)
    citation_rule = _citation_rules(contract)
    margins = _margin_rules(contract)
    attention = _apply_attention(
        input_document["attention_topic_year_rows"], attention_rule
    )
    citations = _apply_citations(input_document["citation_work_rows"], citation_rule)
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "input_status": input_document["status"],
        "input_sha256": _sha256(input_path),
        "contract_sha256": _sha256(contract_path),
        "attention_surge_application": attention,
        "public_success_application": citations,
        "citation_maturation_lag_years": 3,
        "negative_control_equivalence_policy": {
            "estimand_scale": margins["estimand_scale"],
            "symmetric_practical_null": [-0.01, 0.01],
            "equivalence_test": "TWO_ONE_SIDED_TESTS",
            "one_sided_alpha": 0.05,
            "equivalent_confidence_interval": "TWO_SIDED_90_PERCENT",
            "equivalence_control_ids": margins["equivalence_control_ids"],
            "true_minus_placebo_control_ids": margins[
                "true_minus_placebo_control_ids"
            ],
        },
        "access_accounting": {
            "repository_contract_files_read": 1,
            "explicit_input_files_read": 1,
            "operator_directories_listed_or_scanned": 0,
            "parquet_files_opened": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "claim_boundary": (
            "This application classifies only explicit synthetic or discovery-only "
            "aggregates under frozen policy rules. Citation availability remains a "
            "current-snapshot bibliographic-time proxy, boundary ties may inflate the "
            "realized top-decile share, and MAD-zero groups fail closed. It does not "
            "establish historical observation, researcher visibility, an empirical "
            "effect, G0 passage, a holdout seal, a model result, or G4."
        ),
    }


def _write_create_only(path: Path, result: Mapping[str, Any]) -> None:
    require(path.parent.is_dir(), "output parent")
    require(not path.exists() and not path.is_symlink(), "output exists")
    payload = json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(args.input, args.contract)
    if args.output is None:
        print(json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False))
    else:
        _write_create_only(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
