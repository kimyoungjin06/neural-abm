#!/usr/bin/env python3
"""Independently validate the metadata-only G0 v1 successor skeleton."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SKELETON = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_g0_measurement_freeze_v1.yaml"
)
SKELETON_SCHEMA = "pivot_feedback_g0_measurement_freeze.v2"
SKELETON_STATUS = (
    "PENDING_G0_SUCCESSOR_SKELETON_TWO_D0_VALUES_SERIALIZED_NO_SEAL"
)
VERIFY_SCHEMA = "pivot_feedback_g0_measurement_freeze_v1_verification.v1"
VERIFY_STATUS = (
    "PASS_METADATA_ONLY_G0_V1_SUCCESSOR_SKELETON_"
    "TWO_RESOLVED_SEVENTEEN_PENDING_NO_SEAL"
)
PARENT_PATH = "experiments/configs/pivot_feedback_gate_state_v49.yaml"
PARENT_SHA256 = "3d6c4f60634f2673f63597bb5f8a76a50e3329989551095416cbbb4ad3fdfdc0"
PREDECESSOR_PATH = (
    "experiments/configs/pivot_feedback_g0_measurement_freeze_v0.yaml"
)
PREDECESSOR_SHA256 = (
    "2c2912506b96e1c7a09ba008bd15fb824a5b8e3545cf9c387cd3e0cb81044606"
)
INVENTORY_PATH = (
    "experiments/configs/pivot_feedback_g0_readiness_inventory_v1.yaml"
)
INVENTORY_SHA256 = (
    "85e9f5c1ffa25a516261d5756074a3ab0ef7dd893c6a00adb081449d5b72419d"
)
EXPECTED_RESOLVED = {
    "author_minimum_history_works": {
        "value": "NONE",
        "selected_from_observed_outcomes": False,
        "source": "D0_THRESHOLD_FREE_OBSERVABLE_HISTORY_RULE",
    },
    "primary_pivot_thresholds": {
        "value": (
            "FOCAL_GE_1_AND_PRE_ENTRY_EQ_0_WITH_FOUR_STATE_PRECEDENCE"
        ),
        "selected_from_observed_outcomes": False,
        "source": "D0_EXACT_THRESHOLD_FREE_FOUR_STATE_CLASSIFIER",
    },
}
EXPECTED_UNRESOLVED_CHOICES = (
    "attention_surge_threshold",
    "public_success_threshold",
    "citation_maturation_lag_years",
    "negative_control_equivalence_margins",
    "negative_sampling_ratio",
    "inverse_probability_weight_truncation",
    "minimum_cohort_size",
    "small_cell_suppression_threshold",
    "capacity_cell_support_threshold",
    "collinearity_threshold",
    "g1_replication_tolerance",
    "g2_score_margin",
    "g2_maximum_calibration_degradation",
    "g2_macro_trajectory_tolerance",
    "simulation_draws_or_mcse_stopping_rule",
    "parameter_budget",
    "tuning_budget",
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
EXPECTED_BLOCKERS = (
    "G_MINUS_1_RECEIPT_ABSENT",
    "HOLDOUT_ACCESS_SEAL_RECEIPT_ABSENT",
    "SEVENTEEN_QUANTITATIVE_CHOICES_UNRESOLVED",
    "FIFTEEN_REQUIRED_SECTIONS_NOT_FREEZE_READY",
)
COPIED_SUBTREES = (
    "foundation_parents",
    "g_minus_1_binding",
    "holdout_access_binding",
    "fixed_design_choices",
    "inherited_subtree_digests",
    "failure_policy",
)


class G0MeasurementFreezeV1Error(RuntimeError):
    """Fail-closed G0 v1 successor-skeleton validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G0MeasurementFreezeV1Error(message)


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
        raise G0MeasurementFreezeV1Error(f"invalid {label}: {error}") from error


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


def _validate_binding(
    spec_value: Any,
    *,
    label: str,
    expected_path: str,
    expected_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _mapping(spec_value, label)
    require(spec.get("relative_path") == expected_path, f"{label} relative path")
    require(spec.get("sha256") == expected_hash, f"{label} declared hash")
    path = _safe_repo_path(spec.get("relative_path"), label)
    require(_sha256(path) == expected_hash, f"{label} byte hash")
    document = _read_yaml(path, label)
    require(
        document.get("schema_version") == spec.get("required_schema_version"),
        f"{label} schema",
    )
    if "required_label" in spec:
        require(document.get("label") == spec.get("required_label"), f"{label} label")
    if "required_mode" in spec:
        require(document.get("mode") == spec.get("required_mode"), f"{label} mode")
    if "required_status" in spec:
        status_key = str(spec.get("required_status_key", "status"))
        require(
            _lookup(document, status_key) == spec.get("required_status"),
            f"{label} status",
        )
    fields = _mapping(spec.get("required_fields", {}), f"{label} required fields")
    for dotted, expected in fields.items():
        require(_lookup(document, dotted) == expected, f"{label} field {dotted}")
    return document, {
        "relative_path": expected_path,
        "sha256": expected_hash,
        "schema_version": document["schema_version"],
    }


def _validate_predecessor_copy(
    skeleton: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    for subtree in COPIED_SUBTREES:
        require(
            skeleton.get(subtree) == predecessor.get(subtree),
            f"predecessor copied subtree {subtree}",
        )

    old_lifecycle = _mapping(predecessor.get("lifecycle"), "predecessor lifecycle")
    lifecycle = _mapping(skeleton.get("lifecycle"), "lifecycle")
    for key, expected in old_lifecycle.items():
        require(lifecycle.get(key) == expected, f"lifecycle preservation {key}")
    require(
        lifecycle.get("successor_skeleton_status")
        == "TWO_D0_VALUES_SERIALIZED_SEVENTEEN_PENDING",
        "successor skeleton lifecycle status",
    )

    old_completion = _mapping(
        predecessor.get("completion_contract"),
        "predecessor completion contract",
    )
    completion = _mapping(skeleton.get("completion_contract"), "completion contract")
    for key, expected in old_completion.items():
        if key == "validation_status":
            continue
        require(completion.get(key) == expected, f"completion preservation {key}")
    require(
        completion.get("validation_status")
        == "VALID_PENDING_G0_SUCCESSOR_SKELETON",
        "successor completion validation status",
    )
    require(
        completion.get("pending_contract_sealable") is False,
        "pending contract sealable",
    )
    return {
        "immutable_predecessor_sha256": PREDECESSOR_SHA256,
        "copied_subtrees": list(COPIED_SUBTREES),
        "copied_subtree_count": len(COPIED_SUBTREES),
        "predecessor_mutated": False,
    }


def _validate_resolution(
    skeleton: Mapping[str, Any],
    *,
    parent: Mapping[str, Any],
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    resolution = _mapping(skeleton.get("resolution"), "resolution")
    require(
        resolution.get("state")
        == "PENDING_TWO_D0_VALUES_SERIALIZED_SEVENTEEN_PENDING",
        "resolution state",
    )
    resolved = _mapping(
        resolution.get("resolved_measurements"),
        "resolved measurements",
    )
    require(resolved == EXPECTED_RESOLVED, "exact two resolved measurements")
    require(
        all(
            item["selected_from_observed_outcomes"] is False
            for item in resolved.values()
        ),
        "resolved value outcome independence",
    )
    parent_values = _mapping(
        _lookup(parent, "g0_readiness_decision.resolved_successor_values"),
        "parent resolved values",
    )
    require(
        {key: item["value"] for key, item in resolved.items()} == parent_values,
        "parent resolved-value closure",
    )

    inventory_items = [
        _mapping(item, "inventory choice")
        for item in _sequence(
            inventory.get("quantitative_choice_inventory"),
            "inventory choices",
        )
    ]
    inventory_values = {
        str(item["choice_id"]): item.get("proposed_successor_value")
        for item in inventory_items
        if item.get("proposed_successor_value") is not None
    }
    require(
        {key: item["value"] for key, item in resolved.items()} == inventory_values,
        "inventory resolved-value closure",
    )

    unresolved = _mapping(
        resolution.get("unresolved_quantitative_choices"),
        "unresolved quantitative choices",
    )
    require(
        tuple(unresolved) == EXPECTED_UNRESOLVED_CHOICES,
        "unresolved choice order or membership",
    )
    require(
        all(value is None for value in unresolved.values()),
        "unresolved choice must remain null",
    )
    sections = _sequence(
        resolution.get("unresolved_required_sections"),
        "unresolved required sections",
    )
    require(
        tuple(sections) == EXPECTED_SECTIONS,
        "required section order or membership",
    )
    expected_counts = {
        "resolved_measurement_count": 2,
        "unresolved_quantitative_choice_count": 17,
        "unresolved_required_section_count": 15,
        "freeze_ready_required_section_count": 0,
    }
    for key, expected in expected_counts.items():
        require(resolution.get(key) == expected, f"resolution count {key}")
    require(resolution.get("successor_required") is True, "successor required")
    require(
        tuple(skeleton.get("readiness_blockers", [])) == EXPECTED_BLOCKERS,
        "readiness blockers",
    )
    return {
        "resolved_measurement_count": 2,
        "resolved_measurement_values": {
            key: item["value"] for key, item in resolved.items()
        },
        "resolved_values_selected_from_observed_outcomes": False,
        "unresolved_quantitative_choice_count": len(unresolved),
        "unresolved_required_section_count": len(sections),
        "freeze_ready_required_section_count": 0,
        "readiness_blockers": list(EXPECTED_BLOCKERS),
    }


def _validate_fail_closed_boundary(skeleton: Mapping[str, Any]) -> dict[str, Any]:
    require(
        _lookup(skeleton, "g_minus_1_binding.state") == "PENDING",
        "G-1 binding state",
    )
    require(
        _lookup(skeleton, "g_minus_1_binding.receipt_relative_path") is None,
        "G-1 receipt path",
    )
    require(
        _lookup(skeleton, "g_minus_1_binding.receipt_sha256") is None,
        "G-1 receipt hash",
    )
    require(
        _lookup(skeleton, "holdout_access_binding.state") == "PENDING",
        "holdout binding state",
    )
    require(
        _lookup(skeleton, "holdout_access_binding.receipt_relative_path") is None,
        "holdout receipt path",
    )
    require(
        _lookup(skeleton, "holdout_access_binding.receipt_sha256") is None,
        "holdout receipt hash",
    )
    lifecycle = _mapping(skeleton.get("lifecycle"), "lifecycle")
    expected_lifecycle = {
        "confirmatory_outcomes_inspected": False,
        "g4_holdout_opened": False,
        "pass_g0": False,
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "execution_authorized": False,
        "config_alone_can_authorize_execution": False,
        "successor_required_for_freeze": True,
    }
    for key, expected in expected_lifecycle.items():
        require(lifecycle.get(key) == expected, f"fail-closed lifecycle {key}")

    authorization = _mapping(skeleton.get("authorization"), "authorization")
    require(
        authorization.get("repository_metadata_validation") == "AUTHORIZED",
        "metadata validation authorization",
    )
    expected_not_authorized = (
        "additional_quantitative_choice_resolution",
        "required_section_completion",
        "operator_parquet_or_row_level_read",
        "confirmatory_or_holdout_access",
        "outcome_based_selection",
        "final_candidate_eligibility",
        "g0_measurement_freeze_seal",
        "model_fitting",
        "g4_open",
    )
    for key in expected_not_authorized:
        require(authorization.get(key) == "NOT_AUTHORIZED", f"authorization {key}")
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect boundary",
    )
    return {
        "g_minus_1_receipt_state": "PENDING_NULL_BINDING",
        "holdout_access_seal_state": "PENDING_NULL_BINDING",
        "pending_contract_sealable": False,
        "pass_g0": False,
        "confirmatory_or_holdout_opened": False,
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def run(skeleton_path: Path) -> dict[str, Any]:
    skeleton = _read_yaml(skeleton_path, "G0 v1 successor skeleton")
    require(skeleton.get("schema_version") == SKELETON_SCHEMA, "skeleton schema")
    require(skeleton.get("status") == SKELETON_STATUS, "skeleton status")
    require(
        skeleton.get("mode")
        == (
            "VERSIONED_METADATA_ONLY_G0_SUCCESSOR_SKELETON_"
            "TWO_RESOLVED_SEVENTEEN_PENDING"
        ),
        "skeleton mode",
    )

    parent, parent_binding = _validate_binding(
        skeleton.get("authorization_parent"),
        label="authorization parent",
        expected_path=PARENT_PATH,
        expected_hash=PARENT_SHA256,
    )
    predecessor, predecessor_binding = _validate_binding(
        skeleton.get("immutable_predecessor"),
        label="immutable predecessor",
        expected_path=PREDECESSOR_PATH,
        expected_hash=PREDECESSOR_SHA256,
    )
    inventory, inventory_binding = _validate_binding(
        skeleton.get("readiness_inventory_binding"),
        label="readiness inventory",
        expected_path=INVENTORY_PATH,
        expected_hash=INVENTORY_SHA256,
    )
    require(
        _lookup(parent, "authorized_g0_successor_skeleton_scope.independent_validation_required")
        is True,
        "parent independent validation requirement",
    )
    require(
        _lookup(parent, "g0_readiness_decision.status")
        == "NOT_READY_FOR_G0_FREEZE",
        "parent G0 readiness decision",
    )
    require(
        _lookup(inventory, "readiness_decision.g0_freeze_ready") is False,
        "inventory G0 readiness decision",
    )

    copy_replay = _validate_predecessor_copy(skeleton, predecessor)
    resolution_replay = _validate_resolution(
        skeleton,
        parent=parent,
        inventory=inventory,
    )
    boundary_replay = _validate_fail_closed_boundary(skeleton)

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "skeleton_sha256": _sha256(skeleton_path),
        "repository_bindings": {
            "authorization_parent": parent_binding,
            "immutable_predecessor": predecessor_binding,
            "readiness_inventory": inventory_binding,
        },
        "access_accounting": {
            "repository_documents_read": 4,
            "operator_json_metadata_files_read": 0,
            "operator_parquet_files_opened": 0,
            "production_row_level_files_opened": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "api_requests_created_or_submitted": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "predecessor_copy_replay": copy_replay,
        "resolution_replay": resolution_replay,
        "fail_closed_boundary_replay": boundary_replay,
        "authorization": {
            "g0_successor_skeleton": "COMPLETE_INDEPENDENTLY_VALIDATED",
            "g0_quantitative_choice_resolution_beyond_d0": "NOT_AUTHORIZED",
            "g0_required_section_completion": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "confirmatory_or_holdout_opening": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The versioned metadata-only G0 v1 skeleton is independently "
            "valid. It binds the original pending overlay byte-for-byte, "
            "copies six prospective subtrees without change, and serializes "
            "only two D0-resolved outcome-independent values. Seventeen "
            "quantitative choices and all fifteen required sections remain "
            "not freeze-ready; G-1 and holdout-seal receipts remain pending "
            "and null. No operator row data, confirmatory or holdout data, "
            "API request, G0 seal, model, G4 result, causality, or scientific "
            "effect is established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", type=Path, default=DEFAULT_SKELETON)
    args = parser.parse_args()
    result = run(args.skeleton.resolve(strict=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
