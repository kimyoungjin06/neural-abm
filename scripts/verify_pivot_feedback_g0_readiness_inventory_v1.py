#!/usr/bin/env python3
"""Independently validate the metadata-only G0 readiness inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_g0_readiness_inventory_v1.yaml"
)
INVENTORY_SCHEMA = "pivot_feedback_g0_readiness_inventory.v1"
INVENTORY_STATUS = "G0_NOT_READY_VERSIONED_SUCCESSOR_CONTRACT_REQUIRED"
VERIFY_SCHEMA = "pivot_feedback_g0_readiness_inventory_verification.v1"
VERIFY_STATUS = "PASS_METADATA_ONLY_G0_READINESS_INVENTORY_G0_NOT_READY"
RESOLVED_DISPOSITION = "RESOLVED_BY_D0_REQUIRES_G0_SUCCESSOR_SERIALIZATION"
PARTIAL_SECTION = "PARTIAL_NOT_FREEZE_READY"
UNRESOLVED_SECTION = "UNRESOLVED_NOT_FREEZE_READY"
EXPECTED_CHOICES = (
    "author_minimum_history_works",
    "primary_pivot_thresholds",
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
EXPECTED_HARD_BLOCKERS = {
    "G_MINUS_1_RECEIPT_ABSENT",
    "HOLDOUT_ACCESS_SEAL_RECEIPT_ABSENT",
    "SEVENTEEN_QUANTITATIVE_CHOICES_UNRESOLVED",
    "FIFTEEN_REQUIRED_SECTIONS_NOT_FREEZE_READY",
}


class G0ReadinessInventoryError(RuntimeError):
    """Fail-closed G0 readiness inventory error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G0ReadinessInventoryError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sequence(value: Any, label: str) -> list[Any]:
    require(isinstance(value, Sequence) and not isinstance(value, str), f"{label} list")
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
        raise G0ReadinessInventoryError(f"invalid {label}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G0ReadinessInventoryError(f"invalid {label}: {error}") from error


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


def _validate_repository_binding(
    spec_value: Any,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _mapping(spec_value, label)
    path = _safe_repo_path(spec.get("relative_path"), label)
    require(_sha256(path) == spec.get("sha256"), f"{label} hash")
    document = _read_yaml(path, label)
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
    for dotted, expected in _mapping(
        spec.get("required_fields", {}),
        f"{label} fields",
    ).items():
        require(_lookup(document, dotted) == expected, f"{label} field {dotted}")
    return document, {
        "relative_path": str(path.relative_to(REPO_ROOT)),
        "sha256": spec["sha256"],
        "schema_version": document["schema_version"],
    }


def _safe_basename(value: Any, label: str) -> str:
    require(isinstance(value, str) and value, f"{label} basename")
    require(Path(value).name == value and value not in {".", ".."}, f"unsafe {label}")
    return value


def _validate_operator_metadata(
    spec_value: Any,
    *,
    operator_base: Path,
    label: str,
) -> dict[str, Any]:
    spec = _mapping(spec_value, label)
    root_name = _safe_basename(spec.get("root_basename"), f"{label} root")
    receipt_name = _safe_basename(
        spec.get("receipt_basename"),
        f"{label} receipt",
    )
    verification_name = _safe_basename(
        spec.get("verification_basename"),
        f"{label} verification",
    )
    root = operator_base / root_name
    require(root.is_dir() and not root.is_symlink(), f"{label} root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, f"{label} root mode")
    receipt_path = root / receipt_name
    verification_path = operator_base / verification_name
    for path, expected_hash, path_label in (
        (receipt_path, spec.get("receipt_sha256"), "receipt"),
        (
            verification_path,
            spec.get("verification_sha256"),
            "verification",
        ),
    ):
        require(path.is_file() and not path.is_symlink(), f"{label} {path_label}")
        require(path.suffix == ".json", f"{label} {path_label} JSON only")
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"{label} {path_label} mode")
        require(_sha256(path) == expected_hash, f"{label} {path_label} hash")
    receipt = _read_json(receipt_path, f"{label} receipt")
    verification = _read_json(verification_path, f"{label} verification")
    require(
        receipt.get("schema_version") == spec.get("receipt_schema_version"),
        f"{label} receipt schema",
    )
    require(receipt.get("status") == spec.get("receipt_status"), f"{label} receipt status")
    require(receipt.get("receipt_written_last") is True, f"{label} receipt last")
    require(
        verification.get("schema_version")
        == spec.get("verification_schema_version"),
        f"{label} verification schema",
    )
    require(
        verification.get("status") == spec.get("verification_status"),
        f"{label} verification status",
    )
    receipt_hash_field = str(spec.get("verification_receipt_hash_field"))
    require(
        _lookup(verification, receipt_hash_field) == spec.get("receipt_sha256"),
        f"{label} verification receipt closure",
    )
    return {
        "root_basename": root_name,
        "receipt_sha256": spec["receipt_sha256"],
        "verification_sha256": spec["verification_sha256"],
        "receipt_status": receipt["status"],
        "verification_status": verification["status"],
        "json_metadata_files_read": 2,
        "parquet_files_opened": 0,
    }


def _validate_overlay(
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    choices = _mapping(
        _lookup(overlay, "resolution.unresolved_quantitative_choices"),
        "overlay choices",
    )
    sections = _sequence(
        _lookup(overlay, "resolution.unresolved_required_sections"),
        "overlay sections",
    )
    require(tuple(choices) == EXPECTED_CHOICES, "overlay choice order or membership")
    require(all(value is None for value in choices.values()), "overlay choice must be null")
    require(tuple(sections) == EXPECTED_SECTIONS, "overlay section order or membership")
    require(_lookup(overlay, "g_minus_1_binding.state") == "PENDING", "G-1 state")
    require(_lookup(overlay, "g_minus_1_binding.receipt_relative_path") is None, "G-1 path")
    require(_lookup(overlay, "g_minus_1_binding.receipt_sha256") is None, "G-1 hash")
    require(_lookup(overlay, "holdout_access_binding.state") == "PENDING", "holdout state")
    require(
        _lookup(overlay, "holdout_access_binding.receipt_relative_path") is None,
        "holdout path",
    )
    require(
        _lookup(overlay, "holdout_access_binding.receipt_sha256") is None,
        "holdout hash",
    )
    require(
        _lookup(overlay, "fixed_design_choices.temporal_clocks.persistence_outcome")
        == {"start_offset": 1, "end_offset": 3, "inclusive": True},
        "persistence clock",
    )
    require(
        _lookup(overlay, "fixed_design_choices.multiplicity.first_test") == "H3",
        "first multiplicity test",
    )
    require(
        _lookup(overlay, "fixed_design_choices.multiplicity.conditional_method")
        == "HOLM",
        "multiplicity method",
    )
    return {
        "unresolved_quantitative_choices": list(choices),
        "unresolved_quantitative_choice_count": len(choices),
        "unresolved_required_sections": sections,
        "unresolved_required_section_count": len(sections),
        "g_minus_1_receipt_state": "PENDING_NULL_BINDING",
        "holdout_access_seal_state": "PENDING_NULL_BINDING",
        "pending_contract_sealable": False,
    }


def _validate_inventory_mappings(
    inventory: Mapping[str, Any],
    *,
    overlay_replay: Mapping[str, Any],
    support_documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    choices = [
        _mapping(item, "choice inventory item")
        for item in _sequence(
            inventory.get("quantitative_choice_inventory"),
            "choice inventory",
        )
    ]
    require(
        tuple(item.get("choice_id") for item in choices) == EXPECTED_CHOICES,
        "inventory choice order or membership",
    )
    resolved = [item for item in choices if item.get("disposition") == RESOLVED_DISPOSITION]
    require(
        [item["choice_id"] for item in resolved]
        == ["author_minimum_history_works", "primary_pivot_thresholds"],
        "D0-resolved choices",
    )
    for item in choices:
        require(item.get("outcome_based_choice", False) is False, "outcome-based choice")
        if item in resolved:
            require(item.get("proposed_successor_value") is not None, "resolved value")
        else:
            require(item.get("proposed_successor_value") is None, "unresolved value")

    pivot_definition = support_documents["pivot_definition"]
    require(
        _lookup(
            pivot_definition,
            "classification_input.minimum_prior_eligible_work_count",
        )
        == "NONE",
        "threshold-free history choice",
    )
    require(
        _lookup(pivot_definition, "classification.threshold_free") is True,
        "threshold-free classifier",
    )
    require(
        resolved[0].get("proposed_successor_value") == "NONE",
        "history successor value",
    )

    sections = [
        _mapping(item, "section inventory item")
        for item in _sequence(
            inventory.get("required_section_inventory"),
            "section inventory",
        )
    ]
    require(
        tuple(item.get("section_id") for item in sections) == EXPECTED_SECTIONS,
        "inventory section order or membership",
    )
    section_dispositions = [item.get("disposition") for item in sections]
    require(
        set(section_dispositions) <= {PARTIAL_SECTION, UNRESOLVED_SECTION},
        "section disposition vocabulary",
    )
    partial_count = section_dispositions.count(PARTIAL_SECTION)
    unresolved_count = section_dispositions.count(UNRESOLVED_SECTION)
    require(partial_count == 10, "partial section count")
    require(unresolved_count == 5, "unresolved section count")

    totals = _mapping(inventory.get("inventory_totals"), "inventory totals")
    expected_totals = {
        "quantitative_choices_total": 19,
        "quantitative_choices_resolved_by_d0": 2,
        "quantitative_choices_still_unresolved": 17,
        "required_sections_total": 15,
        "required_sections_freeze_ready": 0,
        "required_sections_partial": 10,
        "required_sections_unresolved": 5,
    }
    require(totals == expected_totals, "inventory totals")
    require(
        overlay_replay["unresolved_quantitative_choice_count"] == 19,
        "overlay choice count",
    )
    require(
        overlay_replay["unresolved_required_section_count"] == 15,
        "overlay section count",
    )
    return {
        **expected_totals,
        "resolved_choice_ids": [item["choice_id"] for item in resolved],
        "all_sections_not_freeze_ready": True,
    }


def _validate_decision(
    inventory: Mapping[str, Any],
    *,
    support_documents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    decision = _mapping(inventory.get("readiness_decision"), "readiness decision")
    require(decision.get("status") == "NOT_READY_FOR_G0_FREEZE", "readiness status")
    require(decision.get("d0_core_complete") is True, "D0 core complete")
    require(decision.get("g0_freeze_ready") is False, "G0 freeze ready")
    require(decision.get("pending_overlay_sealable") is False, "overlay sealable")
    require(set(decision.get("hard_blockers", [])) == EXPECTED_HARD_BLOCKERS, "blockers")
    sequence = support_documents["gate_sequence"]
    require(
        _lookup(sequence, "separation_rules.final_cutoff_panel_owner") == "G0_5",
        "final panel owner",
    )
    require(
        _lookup(sequence, "separation_rules.final_cutoff_panel_is_g0_prerequisite")
        is False,
        "final panel prerequisite",
    )
    require(decision.get("final_cutoff_panel_owner") == "G0_5", "decision panel owner")
    require(
        decision.get("final_cutoff_panel_is_g0_prerequisite") is False,
        "decision panel prerequisite",
    )
    scope = _mapping(inventory.get("authorized_successor_scope"), "successor scope")
    require(
        scope.get("mode") == "VERSIONED_G0_SUCCESSOR_SKELETON_METADATA_ONLY_NO_SEAL",
        "successor mode",
    )
    require(scope.get("may_serialize_two_d0_resolved_quantitative_choices") is True, "serialize")
    require(scope.get("must_keep_seventeen_choices_explicitly_pending") is True, "pending choices")
    require(scope.get("must_keep_all_fifteen_sections_not_freeze_ready") is True, "pending sections")
    for key in (
        "may_open_operator_parquet_or_row_level_data",
        "may_read_confirmatory_or_holdout_rows",
        "may_create_or_submit_api_requests",
        "may_choose_values_from_observed_outcomes",
        "may_change_confirmatory_periods",
        "may_rank_or_select_fields_episodes_or_thresholds_by_outcome",
        "may_establish_final_candidate_eligibility",
        "may_seal_or_pass_g0",
        "may_fit_or_score_a_model",
        "may_open_g4",
    ):
        require(scope.get(key) is False, f"successor boundary {key}")
    return {
        "status": decision["status"],
        "hard_blockers": sorted(EXPECTED_HARD_BLOCKERS),
        "g0_freeze_ready": False,
        "successor_mode": scope["mode"],
    }


def run(
    inventory_path: Path,
    *,
    operator_base: Path,
) -> dict[str, Any]:
    inventory = _read_yaml(inventory_path, "inventory")
    require(inventory.get("schema_version") == INVENTORY_SCHEMA, "inventory schema")
    require(inventory.get("status") == INVENTORY_STATUS, "inventory status")

    repo_results: dict[str, Any] = {}
    support_documents: dict[str, Mapping[str, Any]] = {}
    parent_document, repo_results["authorization_parent"] = (
        _validate_repository_binding(
            inventory.get("authorization_parent"),
            "authorization parent",
        )
    )
    overlay, repo_results["pending_g0_overlay"] = _validate_repository_binding(
        inventory.get("pending_g0_overlay"),
        "pending G0 overlay",
    )
    supporting = _mapping(
        inventory.get("supporting_repository_bindings"),
        "supporting repository bindings",
    )
    for label, spec in supporting.items():
        document, result = _validate_repository_binding(spec, label)
        support_documents[label] = document
        repo_results[label] = result
    require(len(repo_results) == 9, "repository document count")
    require(
        _lookup(parent_document, "gate_state.D0.complete") is True,
        "parent D0 completion",
    )
    require(
        _lookup(parent_document, "gate_state.G0.complete") is False,
        "parent G0 pending",
    )

    overlay_replay = _validate_overlay(overlay)
    mapping_replay = _validate_inventory_mappings(
        inventory,
        overlay_replay=overlay_replay,
        support_documents=support_documents,
    )
    decision_replay = _validate_decision(
        inventory,
        support_documents=support_documents,
    )

    operator_results: dict[str, Any] = {}
    operator_specs = _mapping(
        inventory.get("operator_metadata_bindings"),
        "operator metadata bindings",
    )
    for label, spec in operator_specs.items():
        operator_results[label] = _validate_operator_metadata(
            spec,
            operator_base=operator_base,
            label=label,
        )
    require(len(operator_results) == 4, "operator binding count")
    metadata_files = sum(
        result["json_metadata_files_read"] for result in operator_results.values()
    )
    parquet_files = sum(
        result["parquet_files_opened"] for result in operator_results.values()
    )
    access = _mapping(
        inventory.get("access_accounting_contract"),
        "access accounting contract",
    )
    expected_access = {
        "repository_documents_read": 9,
        "operator_json_metadata_files_read": 8,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    require(access == expected_access, "access accounting")
    require(metadata_files == 8 and parquet_files == 0, "operator access replay")

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "inventory_sha256": _sha256(inventory_path),
        "repository_bindings": repo_results,
        "operator_metadata_bindings": operator_results,
        "access_accounting": expected_access,
        "pending_overlay_replay": overlay_replay,
        "inventory_mapping_replay": mapping_replay,
        "readiness_decision_replay": decision_replay,
        "authorization": {
            "d0_core_entry_and_persistence_calibration": (
                "COMPLETE_INDEPENDENTLY_VERIFIED"
            ),
            "g0_successor_skeleton": (
                "AUTHORIZED_METADATA_ONLY_TWO_RESOLVED_SEVENTEEN_PENDING_NO_SEAL"
            ),
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "confirmatory_or_holdout_opening": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "This independent validation reads nine repository documents and "
            "eight JSON metadata files, opens zero Parquet files, and confirms "
            "that G0 is not ready: two of nineteen quantitative choices have "
            "D0-resolved successor values, seventeen remain unresolved, all "
            "fifteen required sections remain not freeze-ready, and both G-1 "
            "and holdout-seal receipts are absent. It authorizes no G0 seal, "
            "confirmatory access, model, G4 opening, or scientific effect."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--operator-base", type=Path, default=Path("/ssd"))
    args = parser.parse_args()
    result = run(
        args.inventory.resolve(strict=True),
        operator_base=args.operator_base.resolve(strict=True),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
