#!/usr/bin/env python3
"""Independently verify the six-choice risk-set/support precommitment."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_risk_set_weight_support_precommitment_v1.yaml"
)
DEFAULT_FIXTURE_INPUT = (
    REPO_ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_risk_set_weight_support_discovery_v1.input.json"
)
DEFAULT_FIXTURE_EXPECTED = (
    REPO_ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_risk_set_weight_support_discovery_v1.expected.json"
)
APPLICATION_TOOL = (
    REPO_ROOT
    / "scripts/apply_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py"
)
CONTRACT_SHA256 = "8f38863eba1880ee375d982cbb2ac094d3ae64028ee651e92b59f13d0189f26e"
FIXTURE_INPUT_SHA256 = "ff4aa3529385a79eae65986cf3a3322f065553ffb95dce59a15d360b35e6313f"
FIXTURE_EXPECTED_SHA256 = "7616cb3e09afa4696e103dca76816c5c5acd1706d28e98d90d871b11a86afe14"
APPLICATION_TOOL_SHA256 = "4642a193b369a7cec82cd9b090d626de2371aaa71b3b41c45b0bb6c5ed090d1c"
CONTRACT_SCHEMA = "pivot_feedback_g0_risk_set_weight_support_precommitment.v1"
CONTRACT_STATUS = (
    "FROZEN_SIX_RISK_SET_WEIGHT_SUPPORT_CHOICES_PENDING_INDEPENDENT_"
    "VALIDATION_NO_G0_PROMOTION"
)
VERIFY_SCHEMA = "pivot_feedback_g0_risk_set_weight_support_precommitment_summary.v1"
VERIFY_STATUS = (
    "PASS_SIX_OUTCOME_BLIND_RISK_SET_WEIGHT_SUPPORT_CHOICES_AND_CLOSED_FIXTURE"
)
CHOICE_IDS = [
    "negative_sampling_ratio",
    "inverse_probability_weight_truncation",
    "minimum_cohort_size",
    "small_cell_suppression_threshold",
    "capacity_cell_support_threshold",
    "collinearity_threshold",
]
ROUND_DIGITS = 12


class RiskSetWeightSupportVerificationError(RuntimeError):
    """Fail-closed independent verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RiskSetWeightSupportVerificationError(message)


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
        raise RiskSetWeightSupportVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RiskSetWeightSupportVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _lookup(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(isinstance(current, Mapping) and part in current, f"missing {dotted}")
        current = current[part]
    return current


def _round(value: float) -> float:
    rounded = round(value, ROUND_DIGITS)
    return 0.0 if rounded == 0 else rounded


def _validate_binding(
    binding: Mapping[str, Any], label: str
) -> tuple[Path, dict[str, Any] | str]:
    path = _safe_repo_path(binding.get("relative_path"), label)
    require(_sha256(path) == binding.get("sha256"), f"{label} byte hash")
    if path.suffix in {".yaml", ".yml"}:
        document: dict[str, Any] | str = _read_yaml(path, label)
    elif path.suffix == ".json":
        document = _read_json(path, label)
    else:
        document = path.read_text(encoding="utf-8")
    if isinstance(document, Mapping):
        schema = binding.get("required_schema_version")
        status = binding.get("required_status")
        if schema is not None:
            require(document.get("schema_version") == schema, f"{label} schema")
        if status is not None:
            require(document.get("status") == status, f"{label} status")
    require(binding.get("mutation_allowed") is False, f"{label} mutation")
    return path, document


def _validate_sources(contract: Mapping[str, Any]) -> dict[str, Any]:
    parent = _mapping(contract.get("authorization_parent"), "authorization parent")
    _, gate = _validate_binding(parent, "authorization parent")
    require(isinstance(gate, Mapping), "authorization gate object")
    require(
        gate["authorized_next_precommitment_scope"]["exact_choice_ids"]
        == CHOICE_IDS,
        "gate exact choice IDs",
    )
    require(
        gate["authorization"]["next_six_choice_precommitment_closed_fixture"]
        == "AUTHORIZED_EXACT_SCOPE",
        "gate authorization",
    )
    require(
        gate["authorization"]["g0_successor_binding_for_next_six_choices"]
        == "NOT_AUTHORIZED",
        "gate binding boundary",
    )

    sources = _mapping(contract.get("bound_design_sources"), "bound sources")
    require(
        list(sources)
        == [
            "g0_v3",
            "prerequisite_closure_plan",
            "readiness_inventory",
            "peer_mechanism_registry",
            "predecision_measurement_config",
            "predecision_measurement_summary",
            "empirical_protocol",
            "peer_identification_design",
        ],
        "bound source order",
    )
    documents: dict[str, dict[str, Any] | str] = {}
    for source_id, value in sources.items():
        _, documents[source_id] = _validate_binding(
            _mapping(value, source_id), source_id
        )

    g0 = _mapping(documents["g0_v3"], "G0 v3")
    unresolved = _mapping(
        g0["resolution"]["unresolved_quantitative_choices"], "G0 unresolved"
    )
    require(all(unresolved.get(choice_id) is None for choice_id in CHOICE_IDS), "G0 choices were not null")
    require(g0["lifecycle"]["pass_g0"] is False, "G0 pass boundary")
    require(g0["holdout_access_binding"]["receipt_sha256"] is None, "holdout boundary")

    closure = _mapping(documents["prerequisite_closure_plan"], "closure plan")
    packages = closure["workstreams"]["SEVENTEEN_QUANTITATIVE_CHOICE_PRECOMMITMENTS"]["packages"]
    risk_package = next(
        item for item in packages if item["package_id"] == "RISK_SET_WEIGHT_SUPPORT_AND_STABILITY"
    )
    require(risk_package["choice_ids"] == CHOICE_IDS, "closure plan choice IDs")

    inventory = _mapping(documents["readiness_inventory"], "readiness inventory")
    inventory_choices = {
        item["choice_id"]: item
        for item in inventory["quantitative_choice_inventory"]
    }
    for choice_id in CHOICE_IDS:
        item = inventory_choices[choice_id]
        require(item["proposed_successor_value"] is None, f"inventory {choice_id} value")
        require(
            item["disposition"] == "UNRESOLVED_REQUIRES_NON_OUTCOME_PRECOMMITMENT",
            f"inventory {choice_id} disposition",
        )

    peer = _mapping(documents["peer_mechanism_registry"], "peer registry")
    downgrade = next(
        item
        for item in peer["claim_downgrade_rules"]
        if item["rule_id"] == "capacity_bridge_not_separable"
    )
    require("SPARSE_COLLINEAR" in downgrade["trigger"], "peer collapse trigger")

    measurement = _mapping(
        documents["predecision_measurement_config"], "measurement config"
    )
    require(
        measurement["registered_author_membership"]["unit"]
        == "CURRENT_SNAPSHOT_OPENALEX_AUTHOR_TOKEN_X_CERTIFIED_EPISODE",
        "measurement unit",
    )
    require(
        measurement["forbidden_inputs_and_outputs"]["entry_effect_size"] is False,
        "measurement outcome boundary",
    )
    summary = _mapping(
        documents["predecision_measurement_summary"], "measurement summary"
    )
    required_rows = int(
        sources["predecision_measurement_summary"]["required_author_episode_rows"]
    )
    require(summary["scope"]["author_episode_rows"] == required_rows, "summary rows")
    require(summary["temporal_leakage_audit"]["entry_or_persistence_rows_read"] is False, "summary outcome boundary")

    protocol = documents["empirical_protocol"]
    design = documents["peer_identification_design"]
    require(isinstance(protocol, str) and "negative-sampling ratio" in protocol, "protocol sampling requirement")
    require(
        isinstance(protocol, str)
        and "inverse" in protocol
        and "probability weights" in protocol,
        "protocol weight requirement",
    )
    require(isinstance(design, str) and "cell-support or collinearity gate fails" in design, "design collapse requirement")
    return {
        "author_episode_rows": required_rows,
        "production_rows_read": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "g0_v3_changed": False,
    }


def _validate_choices(contract: Mapping[str, Any]) -> dict[str, Any]:
    scope = _mapping(contract.get("scope"), "scope")
    require(scope.get("choice_ids") == CHOICE_IDS, "scope choice IDs")
    require(scope.get("choice_count") == 6, "scope choice count")
    choices = _mapping(contract.get("precommitment_choices"), "choices")
    require(list(choices) == CHOICE_IDS, "choice order")
    sampling = _mapping(choices["negative_sampling_ratio"], "sampling")
    weights = _mapping(
        choices["inverse_probability_weight_truncation"], "weights"
    )
    cohort = _mapping(choices["minimum_cohort_size"], "cohort")
    suppression = _mapping(
        choices["small_cell_suppression_threshold"], "suppression"
    )
    capacity = _mapping(
        choices["capacity_cell_support_threshold"], "capacity"
    )
    vif = _mapping(choices["collinearity_threshold"], "VIF")
    require(sampling["ratio_controls_per_case"] == 10, "sampling ratio")
    require(sampling["component_seed"] == 20260718, "sampling seed")
    require(sampling["control_rule"] == "SAMPLE_WITHOUT_REPLACEMENT_FROM_SAME_EXACT_RISK_SET", "sampling rule")
    require(weights["lower_percentile"] == 0.01, "weight lower percentile")
    require(weights["upper_percentile"] == 0.99, "weight upper percentile")
    require(weights["percentile_algorithm"] == "NEAREST_RANK_ONE_BASED_CEILING_P_TIMES_N", "weight percentile algorithm")
    require(weights["raw_untruncated_weight_sensitivity_required"] is True, "weight sensitivity")
    require(cohort["minimum_total_opportunity_rows"] == 500, "cohort total")
    require(cohort["minimum_entry_events"] == 100, "cohort events")
    require(cohort["minimum_non_entry_opportunities"] == 100, "cohort non-events")
    require(cohort["model_specific_sample_size_calculation_still_required"] is True, "model-specific size")
    require(suppression["suppressed_inclusive_lower"] == 1, "suppression lower")
    require(suppression["suppressed_inclusive_upper"] == 10, "suppression upper")
    require(suppression["complementary_secondary_suppression_required"] is True, "secondary suppression")
    require(capacity["minimum_total_opportunities_per_cell"] == 50, "capacity total")
    require(capacity["minimum_entry_events_per_cell"] == 11, "capacity events")
    require(capacity["minimum_non_entry_opportunities_per_cell"] == 11, "capacity non-events")
    require(vif["threshold"] == 5.0, "VIF threshold")
    require(vif["threshold_boundary_counts_as_pass"] is False, "VIF boundary")
    require(capacity["failed_action"] == vif["failed_action"], "collapse action")
    policy = contract["methodological_rationale"]["policy_value_boundary"]
    require(policy["values_are_precommitted_design_policy_not_empirical_truth"] is True, "policy boundary")
    require(all(value is False for key, value in policy.items() if key != "values_are_precommitted_design_policy_not_empirical_truth"), "literature mandate boundary")
    return {
        "choice_count": 6,
        "controls_per_case": 10,
        "weight_percentiles": [0.01, 0.99],
        "minimum_cohort": [500, 100, 100],
        "suppressed_counts": [1, 10],
        "capacity_cell_minimum": [50, 11, 11],
        "vif_strict_upper_bound": 5.0,
        "values_are_policy_not_empirical_truth": True,
    }


def _determinant3(matrix: Sequence[Sequence[float]]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _cramer_solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    denominator = _determinant3(matrix)
    if abs(denominator) <= 1e-12:
        return None
    result = []
    for column in range(3):
        replaced = [row[:] for row in matrix]
        for row in range(3):
            replaced[row][column] = vector[row]
        result.append(_determinant3(replaced) / denominator)
    return result


def _independent_vifs(cells: Sequence[Mapping[str, Any]]) -> list[float | None]:
    rows = []
    for cell in cells:
        signal = float(cell["signal"])
        capacity = float(cell["capacity"])
        rows.append(
            (
                [signal, capacity, signal * capacity],
                float(cell["analytic_weight_sum"]),
            )
        )
    result: list[float | None] = []
    for target_index in range(3):
        others = [index for index in range(3) if index != target_index]
        matrix = [[0.0] * 3 for _ in range(3)]
        vector = [0.0] * 3
        total_weight = sum(weight for _, weight in rows)
        mean = sum(weight * values[target_index] for values, weight in rows) / total_weight
        sst = sum(weight * (values[target_index] - mean) ** 2 for values, weight in rows)
        prepared = []
        for values, weight in rows:
            z = [1.0, values[others[0]], values[others[1]]]
            y = values[target_index]
            prepared.append((z, y, weight))
            for left in range(3):
                vector[left] += weight * z[left] * y
                for right in range(3):
                    matrix[left][right] += weight * z[left] * z[right]
        beta = _cramer_solve(matrix, vector)
        if beta is None or sst <= 1e-12:
            result.append(None)
            continue
        sse = sum(
            weight * (y - sum(coef * item for coef, item in zip(beta, z))) ** 2
            for z, y, weight in prepared
        )
        tolerance = max(0.0, min(1.0, sse / sst))
        result.append(None if tolerance <= 1e-12 else 1 / tolerance)
    return result


def _fixture_by_id(values: Sequence[Mapping[str, Any]], key: str) -> dict[str, Mapping[str, Any]]:
    return {str(item[key]): item for item in values}


def _validate_fixture(
    input_document: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    require(expected["input_sha256"] == FIXTURE_INPUT_SHA256, "fixture input binding")
    require(expected["contract_sha256"] == CONTRACT_SHA256, "fixture contract binding")

    expected_strata = _fixture_by_id(expected["sampling_application"]["strata"], "stratum_id")
    for stratum in input_document["sampling_strata"]:
        stratum_id = str(stratum["stratum_id"])
        target = min(len(stratum["control_ids"]), 10 * len(stratum["case_ids"]))
        selected = sorted(
            stratum["control_ids"],
            key=lambda item: (
                hashlib.sha256(
                    f"20260718|{stratum_id}|{item}".encode()
                ).hexdigest(),
                item,
            ),
        )[:target]
        actual = expected_strata[stratum_id]
        require(actual["selected_control_ids_priority_order"] == selected, f"sampling replay {stratum_id}")
        require(actual["control_inclusion_probability"] == _round(target / len(stratum["control_ids"])), f"sampling probability {stratum_id}")
        require(actual["control_raw_sampling_weight"] == _round(len(stratum["control_ids"]) / target), f"sampling weight {stratum_id}")

    expected_weights = _fixture_by_id(expected["weight_application"]["groups"], "group_id")
    for group in input_document["weight_groups"]:
        expanded = sorted(
            float(item["weight"])
            for item in group["raw_weight_counts"]
            for _ in range(int(item["count"]))
        )
        n = len(expanded)
        lower = expanded[math.ceil(0.01 * n) - 1]
        upper = expanded[math.ceil(0.99 * n) - 1]
        winsorized = [min(max(value, lower), upper) for value in expanded]
        raw_ess = sum(expanded) ** 2 / sum(value * value for value in expanded)
        win_ess = sum(winsorized) ** 2 / sum(value * value for value in winsorized)
        actual = expected_weights[str(group["group_id"])]
        require(actual["lower_cutoff"] == _round(lower), "weight lower replay")
        require(actual["upper_cutoff"] == _round(upper), "weight upper replay")
        require(actual["raw_kish_effective_sample_size"] == _round(raw_ess), "raw ESS replay")
        require(actual["winsorized_kish_effective_sample_size"] == _round(win_ess), "winsorized ESS replay")

    expected_cohorts = _fixture_by_id(expected["cohort_support_application"]["groups"], "group_id")
    failure_names = [
        ("total_opportunity_rows", 500, "TOTAL_BELOW_500"),
        ("entry_events", 100, "ENTRY_EVENTS_BELOW_100"),
        ("non_entry_opportunities", 100, "NON_ENTRY_OPPORTUNITIES_BELOW_100"),
    ]
    for group in input_document["cohort_groups"]:
        failures = [name for key, threshold, name in failure_names if group[key] < threshold]
        actual = expected_cohorts[str(group["group_id"])]
        require(actual["failure_reasons"] == failures, f"cohort replay {group['group_id']}")
        require(actual["support_pass"] is (not failures), f"cohort pass {group['group_id']}")

    expected_cells = _fixture_by_id(expected["small_cell_application"]["cells"], "cell_id")
    for cell in input_document["report_cells"]:
        suppressed = 1 <= int(cell["count"]) <= 10
        actual = expected_cells[str(cell["cell_id"])]
        require(actual["suppressed"] is suppressed, f"suppression replay {cell['cell_id']}")
        require(actual["reported_count"] == (None if suppressed else cell["count"]), f"reported cell {cell['cell_id']}")

    expected_tables = _fixture_by_id(expected["capacity_collinearity_application"]["tables"], "table_id")
    support_fail_count = 0
    vif_fail_count = 0
    separate_pass_count = 0
    for table in input_document["capacity_tables"]:
        support_failures = []
        labels = {(0, 0): "NEITHER", (1, 0): "PUBLIC_SUCCESS_ONLY", (0, 1): "CAPACITY_BRIDGE_ONLY", (1, 1): "BOTH"}
        for cell in table["cells"]:
            label = labels[(cell["signal"], cell["capacity"])]
            total = cell["entry_events"] + cell["non_entry_opportunities"]
            if total < 50:
                support_failures.append(f"{label}:TOTAL_BELOW_50")
            if cell["entry_events"] < 11:
                support_failures.append(f"{label}:ENTRY_EVENTS_BELOW_11")
            if cell["non_entry_opportunities"] < 11:
                support_failures.append(f"{label}:NON_ENTRY_OPPORTUNITIES_BELOW_11")
        vifs = _independent_vifs(table["cells"])
        vif_pass = all(value is not None and value < 5.0 for value in vifs)
        support_pass = not support_failures
        actual = expected_tables[str(table["table_id"])]
        require(actual["support_failure_reasons"] == support_failures, f"capacity support {table['table_id']}")
        require(actual["vif_values"] == [
            {"column": column, "vif": None if value is None else _round(value)}
            for column, value in zip(
                ["peer_visible_success_proxy", "capacity_bridge", "peer_visible_success_proxy_x_capacity_bridge"],
                vifs,
            )
        ], f"capacity VIF {table['table_id']}")
        require(actual["vif_pass"] is vif_pass, f"VIF pass {table['table_id']}")
        require(actual["separate_channel_pass"] is (support_pass and vif_pass), f"channel pass {table['table_id']}")
        support_fail_count += int(not support_pass)
        vif_fail_count += int(not vif_pass)
        separate_pass_count += int(support_pass and vif_pass)

    require(expected["sampling_application"]["selected_control_count"] == 25, "fixture sampled controls")
    require(expected["weight_application"]["clipped_high_count"] == 1, "fixture clipped high")
    require(expected["cohort_support_application"]["fail_count"] == 3, "fixture cohort failures")
    require(expected["small_cell_application"]["suppressed_count"] == 2, "fixture suppressed count")
    capacity = expected["capacity_collinearity_application"]
    require(capacity["support_fail_count"] == support_fail_count == 1, "fixture support failures")
    require(capacity["vif_fail_count"] == vif_fail_count == 1, "fixture VIF failures")
    require(capacity["separate_channel_pass_count"] == separate_pass_count == 1, "fixture channel passes")
    return {
        "sampling_stratum_count": len(input_document["sampling_strata"]),
        "selected_control_count": 25,
        "all_control_retention_stratum_count": 1,
        "weight_group_count": len(input_document["weight_groups"]),
        "upper_weight_clipped_count": 1,
        "cohort_group_count": len(input_document["cohort_groups"]),
        "cohort_fail_count": 3,
        "report_cell_count": len(input_document["report_cells"]),
        "suppressed_cell_count": 2,
        "capacity_table_count": len(input_document["capacity_tables"]),
        "capacity_support_fail_count": 1,
        "capacity_vif_fail_count": 1,
        "separate_channel_pass_count": 1,
    }


def _validate_application_tool() -> dict[str, Any]:
    require(_sha256(APPLICATION_TOOL) == APPLICATION_TOOL_SHA256, "application tool byte hash")
    source = APPLICATION_TOOL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    row_libraries = sorted(imported & {"duckdb", "numpy", "pandas", "pyarrow"})
    require(not row_libraries, "application tool row library import")
    require(f'CONTRACT_SHA256 = "{CONTRACT_SHA256}"' in source, "tool contract constant")
    return {
        "relative_path": str(APPLICATION_TOOL.relative_to(REPO_ROOT)),
        "sha256": APPLICATION_TOOL_SHA256,
        "row_data_library_imports": row_libraries,
        "imports_this_validator": False,
    }


def _validate_lifecycle(contract: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(contract.get("lifecycle"), "lifecycle")
    require(lifecycle["contract_frozen"] is True, "contract frozen")
    for key in (
        "independently_validated",
        "fixture_application_run",
        "discovery_production_application_run",
        "g0_v3_changed",
        "required_sections_completed",
        "g4_artifact_set_materialized",
        "holdout_preflight_written",
        "holdout_seal_created",
        "model_fitted",
        "g4_open",
    ):
        require(lifecycle[key] is False, f"lifecycle {key}")
    require(lifecycle["unresolved_g0_choice_count_before"] == 13, "choices before")
    require(lifecycle["resolved_choice_count_in_this_contract"] == 6, "choices resolved")
    require(lifecycle["unresolved_g0_choice_count_after_future_binding"] == 7, "choices after")
    require(lifecycle["confirmatory_or_holdout_rows_read"] == 0, "holdout rows")
    authorization = _mapping(contract.get("authorization"), "authorization")
    for key in (
        "confirmatory_or_holdout_access",
        "outcome_based_selection",
        "g0_measurement_freeze",
        "model_fitting",
        "g4_open",
    ):
        require(authorization[key] == "NOT_AUTHORIZED", f"authorization {key}")
    return {
        "discovery_production_application_run": False,
        "g0_v3_changed": False,
        "remaining_choice_count_after_future_binding": 7,
        "required_sections_completed": False,
        "confirmatory_or_holdout_rows_read": 0,
        "model_fitted": False,
        "g4_open": False,
    }


def run(
    contract_path: Path = DEFAULT_CONTRACT,
    fixture_input_path: Path = DEFAULT_FIXTURE_INPUT,
    fixture_expected_path: Path = DEFAULT_FIXTURE_EXPECTED,
) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    fixture_input_path = fixture_input_path.resolve(strict=True)
    fixture_expected_path = fixture_expected_path.resolve(strict=True)
    require(_sha256(contract_path) == CONTRACT_SHA256, "contract byte hash")
    require(_sha256(fixture_input_path) == FIXTURE_INPUT_SHA256, "fixture input byte hash")
    require(_sha256(fixture_expected_path) == FIXTURE_EXPECTED_SHA256, "fixture expected byte hash")
    contract = _read_yaml(contract_path, "contract")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")
    input_document = _read_json(fixture_input_path, "fixture input")
    expected = _read_json(fixture_expected_path, "fixture expected")
    source_replay = _validate_sources(contract)
    choice_replay = _validate_choices(contract)
    fixture_replay = _validate_fixture(input_document, expected)
    tool = _validate_application_tool()
    lifecycle = _validate_lifecycle(contract)
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "source_replay": source_replay,
        "choice_replay": choice_replay,
        "fixture_replay": fixture_replay,
        "application_tool": tool,
        "lifecycle_replay": lifecycle,
        "access_accounting": {
            "repository_yaml_documents_read": 7,
            "repository_json_documents_read": 3,
            "repository_text_documents_read": 3,
            "synthetic_fixture_files_read": 2,
            "operator_directories_listed_or_scanned": 0,
            "operator_json_metadata_files_read": 0,
            "parquet_files_opened": 0,
            "production_row_level_files_opened": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "repository_artifacts": {
            "contract": {
                "relative_path": str(contract_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(contract_path),
            },
            "application_tool": tool,
            "validator": {
                "relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "fixture_input": {
                "relative_path": str(fixture_input_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(fixture_input_path),
            },
            "fixture_expected": {
                "relative_path": str(fixture_expected_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(fixture_expected_path),
            },
        },
        "authorization": {
            "six_choice_contract": "COMPLETE_INDEPENDENTLY_VALIDATED_FIXTURE_ONLY",
            "discovery_production_application": "REQUIRES_SUCCESSOR_GATE",
            "g0_successor_binding": "REQUIRES_SUCCESSOR_GATE",
            "confirmatory_or_holdout_access": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": contract["claim_boundary"].strip(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixture-input", type=Path, default=DEFAULT_FIXTURE_INPUT)
    parser.add_argument(
        "--fixture-expected", type=Path, default=DEFAULT_FIXTURE_EXPECTED
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(args.contract, args.fixture_input, args.fixture_expected)
    payload = json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
