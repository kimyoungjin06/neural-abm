#!/usr/bin/env python3
"""Apply frozen G0 risk-set, weight, support, and VIF rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_risk_set_weight_support_precommitment_v1.yaml"
)
DEFAULT_FIXTURE = (
    REPO_ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_risk_set_weight_support_discovery_v1.input.json"
)
CONTRACT_SHA256 = "8f38863eba1880ee375d982cbb2ac094d3ae64028ee651e92b59f13d0189f26e"
CONTRACT_SCHEMA = "pivot_feedback_g0_risk_set_weight_support_precommitment.v1"
CONTRACT_STATUS = (
    "FROZEN_SIX_RISK_SET_WEIGHT_SUPPORT_CHOICES_PENDING_INDEPENDENT_"
    "VALIDATION_NO_G0_PROMOTION"
)
INPUT_SCHEMA = "pivot_feedback_g0_risk_set_weight_support_discovery_input.v1"
OUTPUT_SCHEMA = "pivot_feedback_g0_risk_set_weight_support_discovery_output.v1"
OUTPUT_STATUS = "PASS_FROZEN_RISK_SET_WEIGHT_SUPPORT_RULE_APPLICATION_DISCOVERY_ONLY"
PERMITTED_INPUT_STATUSES = {
    "SYNTHETIC_CLOSED_FIXTURE",
    "DISCOVERY_ONLY_AGGREGATED_OR_EXPLICIT_RISK_SET_NO_CONFIRMATORY_ROWS",
}
ROUND_DIGITS = 12


class RiskSetWeightSupportApplicationError(RuntimeError):
    """Fail-closed risk-set and support application error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RiskSetWeightSupportApplicationError(message)


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
        raise RiskSetWeightSupportApplicationError(
            f"invalid {label}: {error}"
        ) from error


def _read_contract(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "invalid contract")
    require(_sha256(path) == CONTRACT_SHA256, "contract hash drift")
    try:
        contract = _mapping(
            yaml.safe_load(path.read_text(encoding="utf-8")), "contract"
        )
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise RiskSetWeightSupportApplicationError(
            f"invalid contract: {error}"
        ) from error
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")
    return contract


def _exact_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    require(type(value) is int, f"{label} must be an integer")
    if minimum is not None:
        require(value >= minimum, f"{label} below minimum")
    return value


def _finite_positive(value: Any, label: str) -> float:
    require(type(value) in (int, float), f"{label} must be numeric")
    numeric = float(value)
    require(math.isfinite(numeric) and numeric > 0, f"{label} must be positive")
    return numeric


def _nonempty_string(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} must be a string")
    return value


def _round(value: float) -> float:
    rounded = round(value, ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def _choice_rules(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    choices = _mapping(contract.get("precommitment_choices"), "choices")
    expected = {
        "negative_sampling_ratio": "STRATIFIED_RISK_SET_TEN_CONTROLS_PER_CASE_V1",
        "inverse_probability_weight_truncation": (
            "STABILIZED_IPW_NEAREST_RANK_P01_P99_WINSORIZATION_V1"
        ),
        "minimum_cohort_size": (
            "FIELD_EPISODE_PARTITION_MINIMUM_500_WITH_100_PER_OUTCOME_CLASS_V1"
        ),
        "small_cell_suppression_threshold": (
            "SUPPRESS_COUNTS_ONE_THROUGH_TEN_AND_DERIVABLE_COMPLEMENTS_V1"
        ),
        "capacity_cell_support_threshold": (
            "FOUR_SIGNAL_CAPACITY_CELLS_MIN_50_TOTAL_11_PER_OUTCOME_CLASS_V1"
        ),
        "collinearity_threshold": (
            "CENTERED_WEIGHTED_EXPOSURE_BLOCK_VIF_STRICTLY_BELOW_FIVE_V1"
        ),
    }
    rules: dict[str, dict[str, Any]] = {}
    for choice_id, rule_id in expected.items():
        rule = _mapping(choices.get(choice_id), choice_id)
        require(rule.get("rule_id") == rule_id, f"{choice_id} rule ID")
        rules[choice_id] = rule
    require(rules["negative_sampling_ratio"].get("ratio_controls_per_case") == 10, "sampling ratio")
    require(rules["inverse_probability_weight_truncation"].get("lower_percentile") == 0.01, "lower percentile")
    require(rules["inverse_probability_weight_truncation"].get("upper_percentile") == 0.99, "upper percentile")
    require(rules["minimum_cohort_size"].get("minimum_total_opportunity_rows") == 500, "cohort total")
    require(rules["minimum_cohort_size"].get("minimum_entry_events") == 100, "cohort events")
    require(rules["minimum_cohort_size"].get("minimum_non_entry_opportunities") == 100, "cohort non-events")
    require(rules["small_cell_suppression_threshold"].get("suppressed_inclusive_upper") == 10, "suppression threshold")
    require(rules["capacity_cell_support_threshold"].get("minimum_total_opportunities_per_cell") == 50, "capacity total")
    require(rules["capacity_cell_support_threshold"].get("minimum_entry_events_per_cell") == 11, "capacity events")
    require(rules["capacity_cell_support_threshold"].get("minimum_non_entry_opportunities_per_cell") == 11, "capacity non-events")
    require(rules["collinearity_threshold"].get("threshold") == 5.0, "VIF threshold")
    return rules


def _apply_sampling(
    values: Any, rule: Mapping[str, Any]
) -> dict[str, Any]:
    require(isinstance(values, list) and values, "sampling strata")
    ratio = int(rule["ratio_controls_per_case"])
    seed = int(rule["component_seed"])
    output: list[dict[str, Any]] = []
    seen_strata: set[str] = set()
    for value in values:
        row = _mapping(value, "sampling stratum")
        require(
            set(row)
            == {
                "stratum_id",
                "analysis_partition_id",
                "field_id",
                "episode_id",
                "decision_year",
                "case_ids",
                "control_ids",
            },
            "sampling stratum fields",
        )
        stratum_id = _nonempty_string(row["stratum_id"], "stratum ID")
        require(stratum_id not in seen_strata, "duplicate sampling stratum")
        seen_strata.add(stratum_id)
        partition = _nonempty_string(row["analysis_partition_id"], "partition")
        field_id = _exact_int(row["field_id"], "field ID", minimum=1)
        episode_id = _exact_int(row["episode_id"], "episode ID", minimum=1)
        year = _exact_int(row["decision_year"], "decision year")
        require(year <= 2012, "post-discovery decision year")
        cases = row["case_ids"]
        controls = row["control_ids"]
        require(isinstance(cases, list) and cases, "case IDs")
        require(isinstance(controls, list) and controls, "control IDs")
        case_ids = [_nonempty_string(item, "case ID") for item in cases]
        control_ids = [_nonempty_string(item, "control ID") for item in controls]
        require(len(case_ids) == len(set(case_ids)), "duplicate case ID")
        require(len(control_ids) == len(set(control_ids)), "duplicate control ID")
        require(set(case_ids).isdisjoint(control_ids), "case-control overlap")
        target = min(len(control_ids), ratio * len(case_ids))
        priority = sorted(
            control_ids,
            key=lambda item: (
                hashlib.sha256(
                    f"{seed}|{stratum_id}|{item}".encode()
                ).hexdigest(),
                item,
            ),
        )
        selected = priority[:target]
        probability = target / len(control_ids)
        output.append(
            {
                "stratum_id": stratum_id,
                "analysis_partition_id": partition,
                "field_id": field_id,
                "episode_id": episode_id,
                "decision_year": year,
                "case_count": len(case_ids),
                "eligible_control_count": len(control_ids),
                "target_control_count": target,
                "selected_control_count": len(selected),
                "all_controls_retained": target == len(control_ids),
                "control_inclusion_probability": _round(probability),
                "control_raw_sampling_weight": _round(1 / probability),
                "selected_control_ids_priority_order": selected,
            }
        )
    return {
        "rule_id": rule["rule_id"],
        "stratum_count": len(output),
        "case_count": sum(item["case_count"] for item in output),
        "eligible_control_count": sum(
            item["eligible_control_count"] for item in output
        ),
        "selected_control_count": sum(
            item["selected_control_count"] for item in output
        ),
        "strata": output,
    }


def _kish_ess(weight_counts: Sequence[tuple[float, int]]) -> float:
    total = sum(weight * count for weight, count in weight_counts)
    squares = sum(weight * weight * count for weight, count in weight_counts)
    return total * total / squares


def _apply_weights(values: Any, rule: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(values, list) and values, "weight groups")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        row = _mapping(value, "weight group")
        require(
            set(row)
            == {
                "group_id",
                "analysis_partition_id",
                "field_id",
                "raw_weight_counts",
            },
            "weight group fields",
        )
        group_id = _nonempty_string(row["group_id"], "weight group ID")
        require(group_id not in seen, "duplicate weight group")
        seen.add(group_id)
        histogram = row["raw_weight_counts"]
        require(isinstance(histogram, list) and histogram, "weight histogram")
        weight_counts: list[tuple[float, int]] = []
        for item in histogram:
            entry = _mapping(item, "weight histogram entry")
            require(set(entry) == {"weight", "count"}, "weight entry fields")
            weight = _finite_positive(entry["weight"], "weight")
            count = _exact_int(entry["count"], "weight count", minimum=1)
            weight_counts.append((weight, count))
        require(
            len({weight for weight, _ in weight_counts}) == len(weight_counts),
            "duplicate histogram weight",
        )
        ordered = sorted(weight_counts)
        n = sum(count for _, count in ordered)
        lower_rank = math.ceil(float(rule["lower_percentile"]) * n)
        upper_rank = math.ceil(float(rule["upper_percentile"]) * n)

        def at_rank(rank: int) -> float:
            cumulative = 0
            for weight, count in ordered:
                cumulative += count
                if cumulative >= rank:
                    return weight
            raise RiskSetWeightSupportApplicationError("percentile rank overflow")

        lower = at_rank(lower_rank)
        upper = at_rank(upper_rank)
        winsorized: list[tuple[float, int]] = []
        clipped_low = 0
        clipped_high = 0
        for weight, count in ordered:
            clipped = min(max(weight, lower), upper)
            clipped_low += count if weight < lower else 0
            clipped_high += count if weight > upper else 0
            winsorized.append((clipped, count))
        winsorized_mean = sum(
            weight * count for weight, count in winsorized
        ) / n
        normalized_counts: dict[float, int] = {}
        for weight, count in winsorized:
            normalized = _round(weight / winsorized_mean)
            normalized_counts[normalized] = normalized_counts.get(normalized, 0) + count
        output.append(
            {
                "group_id": group_id,
                "analysis_partition_id": _nonempty_string(
                    row["analysis_partition_id"], "partition"
                ),
                "field_id": _exact_int(row["field_id"], "field ID", minimum=1),
                "observation_count": n,
                "lower_nearest_rank": lower_rank,
                "upper_nearest_rank": upper_rank,
                "lower_cutoff": _round(lower),
                "upper_cutoff": _round(upper),
                "clipped_low_count": clipped_low,
                "clipped_high_count": clipped_high,
                "raw_minimum": _round(ordered[0][0]),
                "raw_maximum": _round(ordered[-1][0]),
                "winsorized_mean_before_normalization": _round(winsorized_mean),
                "normalized_weight_counts": [
                    {"weight": weight, "count": count}
                    for weight, count in sorted(normalized_counts.items())
                ],
                "raw_kish_effective_sample_size": _round(
                    _kish_ess(weight_counts)
                ),
                "winsorized_kish_effective_sample_size": _round(
                    _kish_ess(winsorized)
                ),
            }
        )
    return {
        "rule_id": rule["rule_id"],
        "group_count": len(output),
        "observation_count": sum(item["observation_count"] for item in output),
        "clipped_low_count": sum(item["clipped_low_count"] for item in output),
        "clipped_high_count": sum(item["clipped_high_count"] for item in output),
        "groups": output,
    }


def _apply_cohort_support(values: Any, rule: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(values, list) and values, "cohort groups")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for value in values:
        row = _mapping(value, "cohort group")
        require(
            set(row)
            == {
                "group_id",
                "analysis_partition_id",
                "field_id",
                "episode_id",
                "total_opportunity_rows",
                "entry_events",
                "non_entry_opportunities",
            },
            "cohort group fields",
        )
        group_id = _nonempty_string(row["group_id"], "cohort group ID")
        require(group_id not in seen, "duplicate cohort group")
        seen.add(group_id)
        total = _exact_int(row["total_opportunity_rows"], "cohort total", minimum=0)
        events = _exact_int(row["entry_events"], "cohort events", minimum=0)
        non_events = _exact_int(
            row["non_entry_opportunities"], "cohort non-events", minimum=0
        )
        require(total == events + non_events, "cohort total mismatch")
        failures = []
        if total < int(rule["minimum_total_opportunity_rows"]):
            failures.append("TOTAL_BELOW_500")
        if events < int(rule["minimum_entry_events"]):
            failures.append("ENTRY_EVENTS_BELOW_100")
        if non_events < int(rule["minimum_non_entry_opportunities"]):
            failures.append("NON_ENTRY_OPPORTUNITIES_BELOW_100")
        output.append(
            {
                **row,
                "support_pass": not failures,
                "failure_reasons": failures,
            }
        )
    return {
        "rule_id": rule["rule_id"],
        "group_count": len(output),
        "pass_count": sum(item["support_pass"] for item in output),
        "fail_count": sum(not item["support_pass"] for item in output),
        "groups": output,
    }


def _apply_suppression(values: Any, rule: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(values, list) and values, "report cells")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    upper = int(rule["suppressed_inclusive_upper"])
    for value in values:
        row = _mapping(value, "report cell")
        require(set(row) == {"cell_id", "count"}, "report cell fields")
        cell_id = _nonempty_string(row["cell_id"], "cell ID")
        require(cell_id not in seen, "duplicate report cell")
        seen.add(cell_id)
        count = _exact_int(row["count"], "cell count", minimum=0)
        suppressed = 1 <= count <= upper
        if count == 0:
            status = "ZERO_REPORTABLE_IF_NOT_DERIVABLE"
        elif suppressed:
            status = "SUPPRESSED_ONE_THROUGH_TEN"
        else:
            status = "DIRECTLY_REPORTABLE"
        output.append(
            {
                "cell_id": cell_id,
                "input_count": count,
                "reported_count": None if suppressed else count,
                "suppressed": suppressed,
                "status": status,
            }
        )
    return {
        "rule_id": rule["rule_id"],
        "cell_count": len(output),
        "suppressed_count": sum(item["suppressed"] for item in output),
        "cells": output,
    }


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    size = len(vector)
    augmented = [matrix[index][:] + [vector[index]] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * source
                for current, source in zip(augmented[row], augmented[column])
            ]
    return [augmented[row][-1] for row in range(size)]


def _weighted_vif(rows: Sequence[tuple[list[float], float]]) -> list[float | None]:
    vifs: list[float | None] = []
    for target_index in range(3):
        predictors = [index for index in range(3) if index != target_index]
        design = [[1.0, values[predictors[0]], values[predictors[1]]] for values, _ in rows]
        target = [values[target_index] for values, _ in rows]
        weights = [weight for _, weight in rows]
        matrix = [[0.0] * 3 for _ in range(3)]
        vector = [0.0] * 3
        for z, y, weight in zip(design, target, weights):
            for left in range(3):
                vector[left] += weight * z[left] * y
                for right in range(3):
                    matrix[left][right] += weight * z[left] * z[right]
        coefficients = _solve(matrix, vector)
        total_weight = sum(weights)
        mean = sum(weight * y for weight, y in zip(weights, target)) / total_weight
        sst = sum(weight * (y - mean) ** 2 for weight, y in zip(weights, target))
        if coefficients is None or sst <= 1e-12:
            vifs.append(None)
            continue
        sse = sum(
            weight
            * (y - sum(coef * item for coef, item in zip(coefficients, z))) ** 2
            for z, y, weight in zip(design, target, weights)
        )
        tolerance = max(0.0, min(1.0, sse / sst))
        vifs.append(None if tolerance <= 1e-12 else 1 / tolerance)
    return vifs


def _apply_capacity(
    values: Any,
    support_rule: Mapping[str, Any],
    vif_rule: Mapping[str, Any],
) -> dict[str, Any]:
    require(isinstance(values, list) and values, "capacity tables")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    labels = {
        (0, 0): "NEITHER",
        (1, 0): "PUBLIC_SUCCESS_ONLY",
        (0, 1): "CAPACITY_BRIDGE_ONLY",
        (1, 1): "BOTH",
    }
    columns = list(vif_rule["design_columns"])
    for value in values:
        row = _mapping(value, "capacity table")
        require(
            set(row)
            == {
                "table_id",
                "analysis_partition_id",
                "field_id",
                "episode_id",
                "cells",
            },
            "capacity table fields",
        )
        table_id = _nonempty_string(row["table_id"], "capacity table ID")
        require(table_id not in seen, "duplicate capacity table")
        seen.add(table_id)
        cells = row["cells"]
        require(isinstance(cells, list) and len(cells) == 4, "capacity cells")
        parsed: dict[tuple[int, int], dict[str, Any]] = {}
        vif_rows: list[tuple[list[float], float]] = []
        support_failures: list[str] = []
        output_cells: list[dict[str, Any]] = []
        for item in cells:
            cell = _mapping(item, "capacity cell")
            require(
                set(cell)
                == {
                    "signal",
                    "capacity",
                    "entry_events",
                    "non_entry_opportunities",
                    "analytic_weight_sum",
                },
                "capacity cell fields",
            )
            signal = _exact_int(cell["signal"], "signal")
            capacity = _exact_int(cell["capacity"], "capacity")
            require(signal in (0, 1) and capacity in (0, 1), "binary capacity cell")
            key = (signal, capacity)
            require(key not in parsed, "duplicate capacity cell")
            events = _exact_int(cell["entry_events"], "cell events", minimum=0)
            non_events = _exact_int(
                cell["non_entry_opportunities"], "cell non-events", minimum=0
            )
            total = events + non_events
            weight = _finite_positive(cell["analytic_weight_sum"], "cell weight")
            failures = []
            if total < int(support_rule["minimum_total_opportunities_per_cell"]):
                failures.append("TOTAL_BELOW_50")
            if events < int(support_rule["minimum_entry_events_per_cell"]):
                failures.append("ENTRY_EVENTS_BELOW_11")
            if non_events < int(
                support_rule["minimum_non_entry_opportunities_per_cell"]
            ):
                failures.append("NON_ENTRY_OPPORTUNITIES_BELOW_11")
            support_failures.extend(f"{labels[key]}:{reason}" for reason in failures)
            parsed[key] = cell
            output_cells.append(
                {
                    "cell_id": labels[key],
                    "signal": signal,
                    "capacity": capacity,
                    "entry_events": events,
                    "non_entry_opportunities": non_events,
                    "total_opportunities": total,
                    "analytic_weight_sum": _round(weight),
                    "support_pass": not failures,
                    "failure_reasons": failures,
                }
            )
            vif_rows.append(([float(signal), float(capacity), float(signal * capacity)], weight))
        require(set(parsed) == set(labels), "missing capacity cell")
        output_cells.sort(key=lambda item: (item["signal"], item["capacity"]))
        vifs = _weighted_vif(vif_rows)
        vif_values = [
            {"column": column, "vif": None if value is None else _round(value)}
            for column, value in zip(columns, vifs)
        ]
        finite_vifs = [value for value in vifs if value is not None]
        vif_pass = len(finite_vifs) == 3 and all(
            value < float(vif_rule["threshold"]) for value in finite_vifs
        )
        support_pass = not support_failures
        output.append(
            {
                "table_id": table_id,
                "analysis_partition_id": _nonempty_string(
                    row["analysis_partition_id"], "partition"
                ),
                "field_id": _exact_int(row["field_id"], "field ID", minimum=1),
                "episode_id": _exact_int(
                    row["episode_id"], "episode ID", minimum=1
                ),
                "support_pass": support_pass,
                "support_failure_reasons": support_failures,
                "vif_full_rank": len(finite_vifs) == 3,
                "vif_values": vif_values,
                "maximum_vif": (
                    None if not finite_vifs else _round(max(finite_vifs))
                ),
                "vif_pass": vif_pass,
                "separate_channel_pass": support_pass and vif_pass,
                "failure_action": (
                    None
                    if support_pass and vif_pass
                    else "COLLAPSE_TO_ANY_PEER_SIGNAL_OR_CAPACITY_AND_FORFEIT_SEPARATE_CHANNEL_CLAIM"
                ),
                "cells": output_cells,
            }
        )
    return {
        "support_rule_id": support_rule["rule_id"],
        "vif_rule_id": vif_rule["rule_id"],
        "table_count": len(output),
        "support_fail_count": sum(not item["support_pass"] for item in output),
        "vif_fail_count": sum(not item["vif_pass"] for item in output),
        "separate_channel_pass_count": sum(
            item["separate_channel_pass"] for item in output
        ),
        "tables": output,
    }


def run(contract_path: Path, input_path: Path) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    input_path = input_path.resolve(strict=True)
    contract = _read_contract(contract_path)
    input_document = _read_json(input_path, "input")
    require(input_document.get("schema_version") == INPUT_SCHEMA, "input schema")
    require(input_document.get("status") in PERMITTED_INPUT_STATUSES, "input status")
    require(
        set(input_document)
        == {
            "schema_version",
            "status",
            "sampling_strata",
            "weight_groups",
            "cohort_groups",
            "report_cells",
            "capacity_tables",
        },
        "input top-level fields",
    )
    rules = _choice_rules(contract)
    return {
        "schema_version": OUTPUT_SCHEMA,
        "status": OUTPUT_STATUS,
        "input_status": input_document["status"],
        "input_sha256": _sha256(input_path),
        "contract_sha256": _sha256(contract_path),
        "sampling_application": _apply_sampling(
            input_document["sampling_strata"], rules["negative_sampling_ratio"]
        ),
        "weight_application": _apply_weights(
            input_document["weight_groups"],
            rules["inverse_probability_weight_truncation"],
        ),
        "cohort_support_application": _apply_cohort_support(
            input_document["cohort_groups"], rules["minimum_cohort_size"]
        ),
        "small_cell_application": _apply_suppression(
            input_document["report_cells"],
            rules["small_cell_suppression_threshold"],
        ),
        "capacity_collinearity_application": _apply_capacity(
            input_document["capacity_tables"],
            rules["capacity_cell_support_threshold"],
            rules["collinearity_threshold"],
        ),
        "claim_boundary": contract["claim_boundary"].strip(),
    }


def _write_create_only(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--input", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.contract, args.input)
    payload = json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    _write_create_only(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
