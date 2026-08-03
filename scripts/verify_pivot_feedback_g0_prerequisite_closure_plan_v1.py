#!/usr/bin/env python3
"""Independently validate the metadata-only G0 prerequisite closure plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_g0_prerequisite_closure_plan_v1.yaml"
)
PLAN_SCHEMA = "pivot_feedback_g0_prerequisite_closure_plan.v1"
PLAN_STATUS = "PENDING_INDEPENDENT_VALIDATION_FOUR_WORKSTREAM_PLAN_NO_EXECUTION"
VERIFY_SCHEMA = "pivot_feedback_g0_prerequisite_closure_plan_verification.v1"
VERIFY_STATUS = (
    "PASS_METADATA_ONLY_G0_PREREQUISITE_CLOSURE_PLAN_"
    "FOUR_WORKSTREAMS_NO_EXECUTION"
)
EXPECTED_BINDINGS = {
    "authorization_parent": {
        "path": "experiments/configs/pivot_feedback_gate_state_v50.yaml",
        "sha256": (
            "bfbba47a5fd2b76c84941bfa741a16e40f5bb8ce89b25b334685760e1e45f375"
        ),
    },
    "immutable_g0_v1_skeleton": {
        "path": (
            "experiments/configs/pivot_feedback_g0_measurement_freeze_v1.yaml"
        ),
        "sha256": (
            "dc947df2f87220379401c43ea1914b48c71ed04ae474e5695003723d2c591590"
        ),
    },
    "readiness_inventory": {
        "path": (
            "experiments/configs/pivot_feedback_g0_readiness_inventory_v1.yaml"
        ),
        "sha256": (
            "85e9f5c1ffa25a516261d5756074a3ab0ef7dd893c6a00adb081449d5b72419d"
        ),
    },
    "gate_sequence": {
        "path": "experiments/configs/pivot_feedback_gate_sequence_v1.yaml",
        "sha256": (
            "3a3a655234c4efb74a4900e65624c903176615cee19da760cd06b1416e3fd365"
        ),
    },
    "legacy_g_minus_1_registry": {
        "path": (
            "experiments/configs/pivot_feedback_g_minus_1_readiness_v0.yaml"
        ),
        "sha256": (
            "e5048af2d3cca3b1ff04bbecc1a6cbf5dcc6756bf29db995d3cefbe3a017f2be"
        ),
    },
    "g_minus_1a_summary": {
        "path": (
            "experiments/evidence/results/"
            "pivot_feedback_g_minus_1a_receipt_20260723.summary.json"
        ),
        "sha256": (
            "8383e1884c3679df225b743f775832f02e4d767ac5a6b047dbde8aacf9e12802"
        ),
    },
    "g_minus_1a_config": {
        "path": (
            "experiments/configs/pivot_feedback_g_minus_1a_receipt_v1.yaml"
        ),
        "sha256": (
            "be240ec0e06778973194fda772c398cb560b0ac7ff886ee583d34140d889e7bb"
        ),
    },
    "g_minus_1a_verifier": {
        "path": "scripts/verify_pivot_feedback_g_minus_1a_receipt_v1.py",
        "sha256": (
            "b1971e35eb3a33dbb385073bd02536c87cb99f304114b33d619448dede9a0237"
        ),
    },
}
EXPECTED_WORKSTREAMS = (
    "G_MINUS_1_RECEIPT_CLOSURE",
    "HOLDOUT_ACCESS_SEAL_RECEIPT_CLOSURE",
    "SEVENTEEN_QUANTITATIVE_CHOICE_PRECOMMITMENTS",
    "FIFTEEN_REQUIRED_SECTION_COMPLETION_CONTRACTS",
)
EXPECTED_PACKAGES = {
    "MEASUREMENT_AND_PROXY_RULES": (
        "attention_surge_threshold",
        "public_success_threshold",
        "citation_maturation_lag_years",
        "negative_control_equivalence_margins",
    ),
    "RISK_SET_WEIGHT_SUPPORT_AND_STABILITY": (
        "negative_sampling_ratio",
        "inverse_probability_weight_truncation",
        "minimum_cohort_size",
        "small_cell_suppression_threshold",
        "capacity_cell_support_threshold",
        "collinearity_threshold",
    ),
    "G1_REPLICATION": ("g1_replication_tolerance",),
    "G2_PREDICTIVE_GATE": (
        "g2_score_margin",
        "g2_maximum_calibration_degradation",
        "g2_macro_trajectory_tolerance",
    ),
    "G3_SIMULATION_GATE": (
        "simulation_draws_or_mcse_stopping_rule",
        "parameter_budget",
        "tuning_budget",
    ),
}
EXPECTED_CHOICES = tuple(
    choice for choices in EXPECTED_PACKAGES.values() for choice in choices
)
EXPECTED_SECTIONS = (
    "population_and_author_eligibility",
    "pivot_and_robustness_definitions",
    "topic_attention_and_surge",
    "temporal_availability_and_maturation",
    "peer_exposure_and_source_assignment",
    "capacity_bridge_and_entry_mode",
    "risk_set_sampling_and_weights",
    "outcome_horizons_and_h3_timing",
    "negative_control_decision_gates",
    "minimum_support_and_suppression",
    "partition_and_atomic_fold_resolution",
    "estimators_and_uncertainty",
    "hierarchical_multiplicity",
    "g2_predictive_gate",
    "g3_simulation_gate",
)


class G0PrerequisiteClosurePlanError(RuntimeError):
    """Fail-closed G0 prerequisite closure-plan validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G0PrerequisiteClosurePlanError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    require(
        isinstance(value, Sequence) and not isinstance(value, str),
        f"{label} must be a list",
    )
    return list(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G0PrerequisiteClosurePlanError(
            f"invalid {label}: {error}"
        ) from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G0PrerequisiteClosurePlanError(
            f"invalid {label}: {error}"
        ) from error


def _lookup(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing field {dotted}",
        )
        current = current[part]
    return current


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _validate_document_binding(
    spec_value: Any,
    *,
    label: str,
    expected_key: str,
    json_document: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _mapping(spec_value, label)
    expected = EXPECTED_BINDINGS[expected_key]
    require(spec.get("relative_path") == expected["path"], f"{label} path")
    require(spec.get("sha256") == expected["sha256"], f"{label} declared hash")
    path = _safe_repo_path(spec.get("relative_path"), label)
    require(_sha256(path) == expected["sha256"], f"{label} byte hash")
    document = (
        _read_json(path, label) if json_document else _read_yaml(path, label)
    )
    require(
        document.get("schema_version") == spec.get("required_schema_version"),
        f"{label} schema",
    )
    if "required_status" in spec:
        status_key = str(spec.get("required_status_key", "status"))
        require(
            _lookup(document, status_key) == spec.get("required_status"),
            f"{label} status",
        )
    for dotted, expected_value in _mapping(
        spec.get("required_fields", {}),
        f"{label} required fields",
    ).items():
        require(
            _lookup(document, dotted) == expected_value,
            f"{label} field {dotted}",
        )
    return document, {
        "relative_path": expected["path"],
        "sha256": expected["sha256"],
        "schema_version": document["schema_version"],
    }


def _validate_verifier_binding(spec_value: Any) -> dict[str, Any]:
    spec = _mapping(spec_value, "G-1A verifier")
    expected = EXPECTED_BINDINGS["g_minus_1a_verifier"]
    require(spec.get("relative_path") == expected["path"], "G-1A verifier path")
    require(spec.get("sha256") == expected["sha256"], "G-1A verifier declared hash")
    path = _safe_repo_path(spec.get("relative_path"), "G-1A verifier")
    require(_sha256(path) == expected["sha256"], "G-1A verifier byte hash")
    return {"relative_path": expected["path"], "sha256": expected["sha256"]}


def _validate_workstream_dag(plan: Mapping[str, Any]) -> dict[str, Any]:
    dag = _mapping(plan.get("workstream_dag"), "workstream DAG")
    order = tuple(_sequence(dag.get("order"), "workstream order"))
    require(order == EXPECTED_WORKSTREAMS, "workstream order or membership")
    require(dag.get("workstream_count") == 4, "workstream count")
    dependencies = _mapping(dag.get("dependencies"), "workstream dependencies")
    require(tuple(dependencies) == EXPECTED_WORKSTREAMS, "dependency membership")
    for workstream in EXPECTED_WORKSTREAMS[:3]:
        require(dependencies[workstream] == [], f"{workstream} dependency")
    require(
        tuple(dependencies[EXPECTED_WORKSTREAMS[3]]) == EXPECTED_WORKSTREAMS[:3],
        "section workstream dependencies",
    )
    require(
        dag.get("all_workstreams_currently_complete") is False,
        "workstreams cannot be complete",
    )
    require(
        dag.get("plan_does_not_authorize_execution") is True,
        "plan execution boundary",
    )
    return {
        "workstream_count": 4,
        "ordered_workstreams": list(order),
        "acyclic": True,
        "all_workstreams_currently_complete": False,
    }


def _validate_g_minus_1_route(
    plan: Mapping[str, Any],
    *,
    skeleton: Mapping[str, Any],
    sequence: Mapping[str, Any],
    legacy: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_config: Mapping[str, Any],
) -> dict[str, Any]:
    workstream = _mapping(
        _lookup(plan, "workstreams.G_MINUS_1_RECEIPT_CLOSURE"),
        "G-1 closure workstream",
    )
    diagnostic = _mapping(workstream.get("diagnostic"), "G-1 diagnostic")
    require(
        workstream.get("current_state")
        == "PASSED_G_MINUS_1A_RECEIPT_EXISTS_LEGACY_G_MINUS_1_BINDING_INCOMPATIBLE",
        "G-1 closure state",
    )
    require(
        diagnostic.get("g0_v1_expected_schema_version")
        == _lookup(skeleton, "g_minus_1_binding.required_schema_version"),
        "G0 v1 expected G-1 schema",
    )
    require(
        diagnostic.get("g0_v1_expected_status")
        == _lookup(skeleton, "g_minus_1_binding.required_status"),
        "G0 v1 expected G-1 status",
    )
    require(
        diagnostic.get("available_receipt_schema_version")
        == receipt.get("schema_version"),
        "available G-1A schema",
    )
    require(
        diagnostic.get("available_receipt_status") == receipt.get("status"),
        "available G-1A status",
    )
    require(
        diagnostic.get("available_receipt_sha256")
        == EXPECTED_BINDINGS["g_minus_1a_summary"]["sha256"],
        "available G-1A hash",
    )
    require(legacy.get("status") == "G_MINUS_1_PENDING", "legacy G-1 status")
    require(
        _lookup(legacy, "gate_state.pass_g_minus_1") is False,
        "legacy G-1 pass",
    )
    require(
        _lookup(receipt, "authorization.legacy_g_minus_1") == "NOT_PASSED",
        "receipt legacy boundary",
    )
    require(
        _lookup(sequence, "separation_rules.final_cutoff_panel_owner") == "G0_5",
        "final panel owner",
    )
    require(
        _lookup(
            sequence,
            "separation_rules.final_cutoff_panel_is_g_minus_1a_prerequisite",
        )
        is False,
        "final panel G-1A prerequisite",
    )
    require(
        _lookup(receipt, "provenance.config_sha256")
        == EXPECTED_BINDINGS["g_minus_1a_config"]["sha256"],
        "receipt config provenance",
    )
    require(
        _lookup(receipt, "provenance.verifier_sha256")
        == EXPECTED_BINDINGS["g_minus_1a_verifier"]["sha256"],
        "receipt verifier provenance",
    )
    require(
        receipt_config.get("label") == "pivot_feedback_g_minus_1a_receipt_v1",
        "receipt config label",
    )
    require(
        diagnostic.get("mismatch_class")
        == "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE",
        "G-1 mismatch class",
    )
    deliverable = _mapping(workstream.get("next_deliverable"), "G-1 deliverable")
    require(
        deliverable.get("kind")
        == "VERSIONED_METADATA_ONLY_BINDING_RECONCILIATION_CONTRACT",
        "G-1 deliverable kind",
    )
    for key in (
        "may_mutate_g0_v1",
        "may_pass_legacy_g_minus_1",
        "may_reissue_g_minus_1a_receipt",
        "may_read_operator_or_row_data",
    ):
        require(deliverable.get(key) is False, f"G-1 boundary {key}")
    require(
        workstream.get("execution_status") == "NOT_AUTHORIZED_PLAN_ONLY",
        "G-1 execution boundary",
    )
    return {
        "finding": "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE",
        "passing_g_minus_1a_receipt_exists": True,
        "legacy_g_minus_1_remains_pending": True,
        "next_deliverable": deliverable["kind"],
        "execution_authorized": False,
    }


def _validate_holdout_route(
    plan: Mapping[str, Any],
    *,
    skeleton: Mapping[str, Any],
) -> dict[str, Any]:
    workstream = _mapping(
        _lookup(plan, "workstreams.HOLDOUT_ACCESS_SEAL_RECEIPT_CLOSURE"),
        "holdout closure workstream",
    )
    diagnostic = _mapping(workstream.get("diagnostic"), "holdout diagnostic")
    require(
        workstream.get("current_state")
        == "NULL_BINDING_NO_VERSIONED_CONTRACT_OR_RECEIPT_BOUND",
        "holdout closure state",
    )
    require(
        diagnostic.get("expected_schema_version")
        == _lookup(skeleton, "holdout_access_binding.required_schema_version"),
        "holdout expected schema",
    )
    require(
        diagnostic.get("expected_status")
        == _lookup(skeleton, "holdout_access_binding.required_status"),
        "holdout expected status",
    )
    require(diagnostic.get("receipt_relative_path") is None, "holdout path")
    require(diagnostic.get("receipt_sha256") is None, "holdout hash")
    partitions = _sequence(
        _lookup(skeleton, "fixed_design_choices.temporal_partitions"),
        "temporal partitions",
    )
    g4 = next(
        item
        for item in partitions
        if isinstance(item, Mapping)
        and item.get("partition_id") == "g4_temporal_holdout"
    )
    require(
        diagnostic.get("g4_partition_entry_years") == [2020, 2021, 2022],
        "holdout entry years",
    )
    require(g4.get("entry_start_year") == 2020, "G4 start year")
    require(g4.get("entry_end_year") == 2022, "G4 end year")
    require(
        diagnostic.get("outcome_followup_through_year")
        == g4.get("outcome_followup_through_year"),
        "holdout followup year",
    )
    require(g4.get("sealed") is True, "G4 partition sealed declaration")
    require(
        diagnostic.get("g4_domain_and_author_cold")
        == _lookup(skeleton, "fixed_design_choices.full_panel.g4_domain_and_author_cold"),
        "holdout domain-author cold rule",
    )
    deliverable = _mapping(
        workstream.get("next_deliverable"),
        "holdout deliverable",
    )
    for key in (
        "may_issue_or_open_seal",
        "may_enumerate_or_hash_holdout_artifacts",
        "may_read_confirmatory_or_holdout_rows",
    ):
        require(deliverable.get(key) is False, f"holdout boundary {key}")
    require(
        workstream.get("execution_status") == "NOT_AUTHORIZED_PLAN_ONLY",
        "holdout execution boundary",
    )
    return {
        "binding_state": "PENDING_NULL_BINDING",
        "contract_or_receipt_bound": False,
        "entry_years": [2020, 2021, 2022],
        "execution_authorized": False,
    }


def _validate_choice_route(
    plan: Mapping[str, Any],
    *,
    skeleton: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    workstream = _mapping(
        _lookup(
            plan,
            "workstreams.SEVENTEEN_QUANTITATIVE_CHOICE_PRECOMMITMENTS",
        ),
        "choice workstream",
    )
    packages = [
        _mapping(item, "choice package")
        for item in _sequence(workstream.get("packages"), "choice packages")
    ]
    require(
        tuple(item.get("package_id") for item in packages)
        == tuple(EXPECTED_PACKAGES),
        "choice package order or membership",
    )
    flattened: list[str] = []
    for item in packages:
        package_id = str(item["package_id"])
        choices = tuple(_sequence(item.get("choice_ids"), f"{package_id} choices"))
        require(choices == EXPECTED_PACKAGES[package_id], f"{package_id} choices")
        current_values = _sequence(
            item.get("current_values"),
            f"{package_id} current values",
        )
        require(len(current_values) == len(choices), f"{package_id} value count")
        require(
            all(value is None for value in current_values),
            f"{package_id} values must remain null",
        )
        flattened.extend(choices)
    require(tuple(flattened) == EXPECTED_CHOICES, "flattened choice membership")
    skeleton_choices = _mapping(
        _lookup(skeleton, "resolution.unresolved_quantitative_choices"),
        "skeleton choices",
    )
    require(tuple(skeleton_choices) == EXPECTED_CHOICES, "skeleton choice membership")
    require(
        all(value is None for value in skeleton_choices.values()),
        "skeleton choices remain null",
    )
    inventory_items = [
        _mapping(item, "inventory choice")
        for item in _sequence(
            inventory.get("quantitative_choice_inventory"),
            "inventory choices",
        )
    ]
    unresolved_inventory = tuple(
        str(item["choice_id"])
        for item in inventory_items
        if item.get("proposed_successor_value") is None
    )
    require(unresolved_inventory == EXPECTED_CHOICES, "inventory choice closure")
    require(workstream.get("package_count") == 5, "package count")
    require(workstream.get("choice_count") == 17, "choice count")
    require(workstream.get("resolved_value_count") == 0, "new resolved value count")
    boundary = _mapping(
        workstream.get("selection_boundary"),
        "choice selection boundary",
    )
    for key in (
        "may_use_confirmatory_or_holdout_outcomes",
        "may_change_confirmatory_periods",
        "may_rank_or_select_fields_episodes_or_thresholds_by_effect",
    ):
        require(boundary.get(key) is False, f"choice selection boundary {key}")
    deliverable = _mapping(
        workstream.get("next_deliverable"),
        "choice deliverable",
    )
    for key in (
        "may_write_values_now",
        "may_call_external_api_now",
        "may_read_confirmatory_or_holdout_rows",
    ):
        require(deliverable.get(key) is False, f"choice deliverable boundary {key}")
    return {
        "package_count": 5,
        "choice_count": 17,
        "newly_resolved_value_count": 0,
        "all_values_null": True,
        "execution_authorized": False,
    }


def _validate_section_route(
    plan: Mapping[str, Any],
    *,
    skeleton: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    workstream = _mapping(
        _lookup(
            plan,
            "workstreams.FIFTEEN_REQUIRED_SECTION_COMPLETION_CONTRACTS",
        ),
        "section workstream",
    )
    sections = [
        _mapping(item, "section item")
        for item in _sequence(workstream.get("sections"), "sections")
    ]
    require(
        tuple(item.get("section_id") for item in sections) == EXPECTED_SECTIONS,
        "section order or membership",
    )
    require(
        all(item.get("state") == "NOT_FREEZE_READY" for item in sections),
        "section state",
    )
    require(
        all(item.get("required_closure") for item in sections),
        "section closure rule",
    )
    require(
        tuple(
            _sequence(
                _lookup(skeleton, "resolution.unresolved_required_sections"),
                "skeleton sections",
            )
        )
        == EXPECTED_SECTIONS,
        "skeleton section closure",
    )
    inventory_sections = [
        _mapping(item, "inventory section")
        for item in _sequence(
            inventory.get("required_section_inventory"),
            "inventory sections",
        )
    ]
    require(
        tuple(item.get("section_id") for item in inventory_sections)
        == EXPECTED_SECTIONS,
        "inventory section closure",
    )
    require(
        all(
            item.get("disposition")
            in {"PARTIAL_NOT_FREEZE_READY", "UNRESOLVED_NOT_FREEZE_READY"}
            for item in inventory_sections
        ),
        "inventory sections must remain not freeze-ready",
    )
    require(workstream.get("section_count") == 15, "section count")
    require(
        workstream.get("freeze_ready_section_count") == 0,
        "freeze-ready section count",
    )
    deliverable = _mapping(
        workstream.get("next_deliverable"),
        "section deliverable",
    )
    require(
        deliverable.get("may_mark_any_section_complete_now") is False,
        "section completion boundary",
    )
    require(deliverable.get("may_pass_g0_now") is False, "section G0 boundary")
    return {
        "section_count": 15,
        "freeze_ready_section_count": 0,
        "all_sections_have_machine_checkable_future_closure": True,
        "execution_authorized": False,
    }


def _validate_fail_closed_boundary(plan: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(plan.get("lifecycle"), "lifecycle")
    expected_lifecycle = {
        "g0_v1_skeleton_mutated": False,
        "legacy_g_minus_1_promoted": False,
        "g_minus_1a_receipt_reissued": False,
        "holdout_seal_contract_written": False,
        "holdout_seal_receipt_issued": False,
        "unresolved_values_changed": False,
        "required_sections_completed": False,
        "confirmatory_or_holdout_rows_read": False,
        "api_requests_created_or_submitted": False,
        "pass_g0": False,
        "model_fitted": False,
        "g4_open": False,
    }
    for key, expected in expected_lifecycle.items():
        require(lifecycle.get(key) == expected, f"lifecycle boundary {key}")
    access = _mapping(plan.get("access_accounting_contract"), "access accounting")
    expected_access = {
        "repository_documents_read": 9,
        "operator_json_metadata_files_read": 0,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    require(access == expected_access, "access accounting")
    authorization = _mapping(plan.get("authorization"), "authorization")
    require(
        authorization.get("plan_validation") == "AUTHORIZED_METADATA_ONLY",
        "plan validation authorization",
    )
    require(
        authorization.get("g_minus_1a_binding_reconciliation_contract")
        == "REQUIRES_SUCCESSOR_GATE",
        "binding reconciliation gate requirement",
    )
    for key in (
        "actual_g0_v2_binding_change",
        "legacy_g_minus_1_promotion",
        "holdout_seal_receipt_issuance_or_opening",
        "quantitative_choice_resolution",
        "required_section_completion",
        "confirmatory_or_holdout_access",
        "outcome_based_selection",
        "g0_measurement_freeze",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"authorization {key}")
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect",
    )
    next_step = _mapping(plan.get("next_atomic_step"), "next atomic step")
    require(
        next_step.get("workstream_id") == "G_MINUS_1_RECEIPT_CLOSURE",
        "next workstream",
    )
    require(
        next_step.get("deliverable_kind")
        == "VERSIONED_METADATA_ONLY_BINDING_RECONCILIATION_CONTRACT",
        "next deliverable",
    )
    require(
        next_step.get("execution_requires_successor_gate") is True,
        "next successor gate",
    )
    require(
        next_step.get("actual_g0_v2_binding_change_authorized") is False,
        "G0 v2 boundary",
    )
    return {
        "access_accounting": expected_access,
        "next_atomic_step": next_step["deliverable_kind"],
        "actual_g0_v2_binding_change_authorized": False,
        "pass_g0": False,
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def run(plan_path: Path) -> dict[str, Any]:
    plan = _read_yaml(plan_path, "G0 prerequisite closure plan")
    require(plan.get("schema_version") == PLAN_SCHEMA, "plan schema")
    require(plan.get("status") == PLAN_STATUS, "plan status")
    require(
        plan.get("mode")
        == "VERSIONED_METADATA_ONLY_FOUR_WORKSTREAM_CLOSURE_PLAN_NO_EXECUTION",
        "plan mode",
    )

    documents: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for field, expected_key in (
        ("authorization_parent", "authorization_parent"),
        ("immutable_g0_v1_skeleton", "immutable_g0_v1_skeleton"),
        ("readiness_inventory", "readiness_inventory"),
        ("gate_sequence", "gate_sequence"),
        ("legacy_g_minus_1_registry", "legacy_g_minus_1_registry"),
    ):
        documents[field], bindings[field] = _validate_document_binding(
            plan.get(field),
            label=field.replace("_", " "),
            expected_key=expected_key,
        )
    receipt_specs = _mapping(
        plan.get("passed_g_minus_1a_receipt"),
        "passed G-1A receipt",
    )
    documents["g_minus_1a_summary"], bindings["g_minus_1a_summary"] = (
        _validate_document_binding(
            receipt_specs.get("summary"),
            label="G-1A summary",
            expected_key="g_minus_1a_summary",
            json_document=True,
        )
    )
    documents["g_minus_1a_config"], bindings["g_minus_1a_config"] = (
        _validate_document_binding(
            receipt_specs.get("config"),
            label="G-1A config",
            expected_key="g_minus_1a_config",
        )
    )
    bindings["g_minus_1a_verifier"] = _validate_verifier_binding(
        receipt_specs.get("verifier")
    )

    dag_replay = _validate_workstream_dag(plan)
    g_minus_1_replay = _validate_g_minus_1_route(
        plan,
        skeleton=documents["immutable_g0_v1_skeleton"],
        sequence=documents["gate_sequence"],
        legacy=documents["legacy_g_minus_1_registry"],
        receipt=documents["g_minus_1a_summary"],
        receipt_config=documents["g_minus_1a_config"],
    )
    holdout_replay = _validate_holdout_route(
        plan,
        skeleton=documents["immutable_g0_v1_skeleton"],
    )
    choice_replay = _validate_choice_route(
        plan,
        skeleton=documents["immutable_g0_v1_skeleton"],
        inventory=documents["readiness_inventory"],
    )
    section_replay = _validate_section_route(
        plan,
        skeleton=documents["immutable_g0_v1_skeleton"],
        inventory=documents["readiness_inventory"],
    )
    boundary_replay = _validate_fail_closed_boundary(plan)

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "plan_sha256": _sha256(plan_path),
        "repository_bindings": bindings,
        "workstream_dag_replay": dag_replay,
        "g_minus_1_route_replay": g_minus_1_replay,
        "holdout_route_replay": holdout_replay,
        "quantitative_choice_route_replay": choice_replay,
        "required_section_route_replay": section_replay,
        "fail_closed_boundary_replay": boundary_replay,
        "authorization": {
            "closure_plan": "COMPLETE_INDEPENDENTLY_VALIDATED",
            "g_minus_1a_binding_reconciliation_contract": (
                "REQUIRES_SUCCESSOR_GATE"
            ),
            "actual_g0_v2_binding_change": "NOT_AUTHORIZED",
            "holdout_seal_contract": "NOT_AUTHORIZED",
            "quantitative_choice_resolution": "NOT_AUTHORIZED",
            "required_section_completion": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The metadata-only four-workstream closure plan is independently "
            "valid and executes no closure. The first blocker is a contract-"
            "generation mismatch, not missing evidence: an exact passing "
            "G-1A receipt exists while immutable G0 v1 expects a legacy G-1 "
            "schema that remains pending and cannot be promoted here. All "
            "seventeen values remain null, all fifteen sections remain not "
            "freeze-ready, and the holdout receipt remains unbound. No "
            "operator row data, confirmatory or holdout data, API request, "
            "G0 v2 change, G0 seal, model, G4 result, causality, or "
            "scientific effect is established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    result = run(args.plan.resolve(strict=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
