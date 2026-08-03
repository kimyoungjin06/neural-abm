#!/usr/bin/env python3
"""Verify why a production G4 holdout seal cannot yet be truthfully created."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESOLUTION = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_holdout_production_input_resolution_v1.yaml"
)
RESOLUTION_SCHEMA = "pivot_feedback_holdout_production_input_resolution.v1"
RESOLUTION_STATUS = "BLOCKED_NO_MATERIALIZED_G4_ARTIFACT_SET_NO_SEAL_ACTION"
VERIFY_SCHEMA = "pivot_feedback_holdout_production_input_resolution_summary.v1"
VERIFY_STATUS = (
    "PASS_METADATA_ONLY_INPUT_RESOLUTION_NO_MATERIALIZED_G4_ARTIFACT_SET"
)
EXPECTED_BINDINGS = {
    "authorization_parent": {
        "path": "experiments/configs/pivot_feedback_gate_state_v56.yaml",
        "sha256": (
            "138bd1b22830ab5ac9f5940fcbd802058cd41be0f5f79853708dd1e2f6ce6696"
        ),
        "schema": "pivot_feedback_gate_state.v56",
        "status": (
            "VALID_EXPLICIT_PREFLIGHT_TO_SEAL_CHAIN_CLOSED_FIXTURE_"
            "PRODUCTION_INPUTS_REQUIRED"
        ),
        "format": "yaml",
    },
    "cross_field_matrix": {
        "path": "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml",
        "sha256": (
            "6fd3e2c7aae7a7aac085ed73ba4bf3d64a91c02d60dce8a9a6a262591a0c934a"
        ),
        "schema": "pivot_feedback_cross_field_matrix.v1",
        "status": None,
        "format": "yaml",
    },
    "g0_v2": {
        "path": "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml",
        "sha256": (
            "a2cb0cbf444a7c128eecc40a815b80d5fe86f617a36849330612491b03934eb3"
        ),
        "schema": "pivot_feedback_g0_measurement_freeze.v3",
        "status": "PENDING_G0_V2_G_MINUS_1A_BOUND_THREE_BLOCKERS_NO_SEAL",
        "format": "yaml",
    },
    "entry_source_contract_summary": {
        "path": (
            "experiments/evidence/results/"
            "pivot_feedback_d0_entry_source_classifier_contract_"
            "20260731.summary.json"
        ),
        "sha256": (
            "9fee3d875205c154491bb5ba212d69eda81249413d498746670147a7cfb0ef99"
        ),
        "schema": (
            "pivot_feedback_d0_entry_source_classifier_contract_summary.v1"
        ),
        "status": (
            "PASS_EXACT_FOUR_ENTRY_SOURCE_CLASSIFIER_CONTRACT_RECEIPTS_"
            "AND_FIXTURES"
        ),
        "format": "json",
    },
    "entry_source_production_summary": {
        "path": (
            "experiments/evidence/results/"
            "pivot_feedback_d0_entry_source_production_20260731.summary.json"
        ),
        "sha256": (
            "29c27909ac2f9cf73a071e627fbb0b3e2d3e684339b254c66e409ad0e60fd0ca"
        ),
        "schema": "pivot_feedback_d0_entry_source_production_summary.v1",
        "status": (
            "PASS_INDEPENDENT_EXACT_FOUR_THREE_COUNT_SOURCE_REPLAY_"
            "NO_STATE_LABELS"
        ),
        "format": "json",
    },
}
EXPECTED_SUCCESSOR_ORDER = [
    "COMPLETE_OUTCOME_BLIND_G0_PRECOMMITMENTS_AND_REQUIRED_SECTIONS",
    "FREEZE_OUTCOME_BLIND_FIELD_EPISODE_AND_ATOMIC_PARTITION_ASSIGNMENT",
    "MATERIALIZE_PRIVATE_G4_ARTIFACT_SET_WITHOUT_CONFIRMATORY_ANALYSIS",
    "BIND_PRIVATE_SOURCE_ROOT_AND_COMPLETE_EXPLICIT_ARTIFACT_LIST",
    "AUTHOR_OPERATOR_PREFLIGHT_FROM_EXPLICIT_LIST_ONLY",
    "CREATE_AND_INDEPENDENTLY_VERIFY_BYTE_SEAL",
    "BIND_VERIFIED_RECEIPT_IN_FINAL_G0_SUCCESSOR",
]


class HoldoutProductionInputResolutionError(RuntimeError):
    """Fail-closed metadata-only production-input resolution error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutProductionInputResolutionError(message)


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


def _read_document(path: Path, document_format: str, label: str) -> dict[str, Any]:
    try:
        if document_format == "yaml":
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise HoldoutProductionInputResolutionError(
            f"invalid {label}: {error}"
        ) from error
    return _mapping(value, label)


def _validate_binding(
    resolution: Mapping[str, Any], key: str
) -> dict[str, Any]:
    if key == "authorization_parent":
        declared = _mapping(resolution.get(key), key)
    else:
        sources = _mapping(resolution.get("bound_design_sources"), "sources")
        declared = _mapping(sources.get(key), key)
    expected = EXPECTED_BINDINGS[key]
    require(declared.get("relative_path") == expected["path"], f"{key} path")
    require(declared.get("sha256") == expected["sha256"], f"{key} hash")
    require(
        declared.get("required_schema_version") == expected["schema"],
        f"{key} required schema",
    )
    if expected["status"] is not None:
        require(
            declared.get("required_status") == expected["status"],
            f"{key} required status",
        )
    require(declared.get("mutation_allowed") is False, f"{key} mutation")
    path = _safe_repo_path(declared["relative_path"], key)
    require(_sha256(path) == expected["sha256"], f"{key} byte hash")
    document = _read_document(path, expected["format"], key)
    require(document.get("schema_version") == expected["schema"], f"{key} schema")
    if expected["status"] is not None:
        require(document.get("status") == expected["status"], f"{key} status")
    return document


def _validate_declared_inputs(resolution: Mapping[str, Any]) -> dict[str, Any]:
    declared = _mapping(
        resolution.get("declared_input_resolution"), "declared input resolution"
    )
    source = _mapping(declared.get("private_source_root_absolute_path"), "source")
    artifacts = _mapping(
        declared.get("complete_explicit_artifact_relative_path_list"),
        "artifact list",
    )
    require(source == {"state": "UNBOUND", "value": None}, "source root state")
    require(
        artifacts == {"state": "UNBOUND", "value": None, "artifact_count": 0},
        "artifact list state",
    )
    require(declared.get("production_preflight") == {"state": "NOT_RUN"}, "preflight")
    require(declared.get("production_seal") == {"state": "NOT_RUN"}, "seal")
    require(
        declared.get("production_receipt") == {"state": "NOT_WRITTEN"},
        "receipt",
    )
    require(
        declared.get("independent_receipt_verification")
        == {"state": "NOT_RUN"},
        "receipt verification",
    )
    return {
        "source_root_bound": False,
        "complete_artifact_list_bound": False,
        "artifact_count": 0,
        "production_preflight_written": False,
        "production_seal_created": False,
        "production_receipt_written": False,
    }


def _validate_materialization(
    resolution: Mapping[str, Any],
    *,
    matrix: Mapping[str, Any],
    g0: Mapping[str, Any],
    entry_contract: Mapping[str, Any],
    entry_production: Mapping[str, Any],
) -> dict[str, Any]:
    replay = _mapping(resolution.get("materialization_replay"), "materialization")
    logical = _mapping(replay.get("logical_partition"), "logical partition")
    require(
        logical
        == {
            "partition_id": "g4_temporal_holdout",
            "entry_years": [2020, 2021, 2022],
            "outcome_followup_through_year": 2025,
            "domain_cold": True,
            "author_cold": True,
            "state": "LOGICALLY_RESERVED",
        },
        "logical partition replay",
    )

    slots = matrix.get("engineering_slots")
    require(isinstance(slots, list), "engineering slots")
    matching = [item for item in slots if item.get("slot_id") == "g4_domain_author_cold_slot"]
    require(len(matching) == 1, "G4 slot uniqueness")
    slot = _mapping(matching[0], "G4 slot")
    require(
        slot.get("scientific_selection_status")
        == "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT",
        "G4 selection status",
    )
    require(slot.get("selected_domain_id") is None, "G4 domain must be unresolved")
    require(slot.get("selected_field_id") is None, "G4 field must be unresolved")
    declared_slot = _mapping(
        replay.get("g4_domain_author_cold_slot"), "declared G4 slot"
    )
    require(
        declared_slot
        == {
            "scientific_selection_status": (
                "PENDING_OUTCOME_BLIND_STRUCTURAL_AUDIT"
            ),
            "selected_domain_id": None,
            "selected_field_id": None,
            "state": "NOT_MATERIALIZED",
        },
        "declared G4 slot replay",
    )

    resolution_state = _mapping(g0.get("resolution"), "G0 resolution")
    readiness = _mapping(g0.get("readiness"), "G0 readiness")
    require(
        resolution_state.get("unresolved_quantitative_choice_count") == 17,
        "unresolved choice count",
    )
    require(
        resolution_state.get("unresolved_required_section_count") == 15,
        "unresolved section count",
    )
    require(
        resolution_state.get("freeze_ready_required_section_count") == 0,
        "freeze-ready section count",
    )
    require(readiness.get("g0_freeze_ready") is False, "G0 readiness")
    declared_g0 = _mapping(
        replay.get("quantitative_and_section_state"), "declared G0 state"
    )
    require(
        declared_g0
        == {
            "unresolved_quantitative_choice_count": 17,
            "unresolved_required_section_count": 15,
            "freeze_ready_required_section_count": 0,
            "g0_freeze_ready": False,
        },
        "declared G0 state replay",
    )

    scope = _mapping(entry_contract.get("scope"), "entry source scope")
    require(scope.get("entry_year_minimum") == 1953, "entry year minimum")
    require(scope.get("entry_year_maximum") == 2012, "entry year maximum")
    operator = _mapping(entry_production.get("operator_artifacts"), "operator artifacts")
    require(
        operator.get("root_basename")
        == "pivot-feedback-d0-entry-source-v1-full-20260731",
        "entry source root",
    )
    require(
        operator.get("feature_basename") == "pivot_entry_source_features.parquet",
        "entry source feature",
    )
    require(
        "No production classifier state" in entry_production.get("claim_boundary", ""),
        "entry source claim boundary",
    )
    declared_d0 = _mapping(
        replay.get("available_d0_entry_source"), "declared D0 source"
    )
    require(declared_d0.get("entry_year_maximum") == 2012, "declared D0 end")
    require(
        declared_d0.get("may_be_relabelled_as_g4_holdout_artifact") is False,
        "D0 substitution boundary",
    )
    return {
        "logical_partition_reserved": True,
        "g4_domain_selected": False,
        "g4_field_selected": False,
        "g4_artifact_set_materialized": False,
        "unresolved_quantitative_choice_count": 17,
        "unresolved_required_section_count": 15,
        "available_d0_entry_year_maximum": 2012,
        "requested_g4_entry_years": [2020, 2021, 2022],
        "existing_d0_artifact_substitution_valid": False,
    }


def _validate_decision_and_boundary(resolution: Mapping[str, Any]) -> dict[str, Any]:
    decision = _mapping(resolution.get("resolution_decision"), "decision")
    require(
        decision.get("blocker_class")
        == "PREREQUISITE_ARTIFACT_NOT_MATERIALIZED_NOT_MISSING_USER_PATH",
        "blocker class",
    )
    for key in (
        "source_root_can_be_safely_inferred",
        "complete_artifact_list_can_be_safely_inferred",
        "existing_d0_artifact_can_be_substituted",
        "production_preflight_may_run_now",
        "production_seal_may_run_now",
        "logical_partition_reservation_equals_byte_seal",
    ):
        require(decision.get(key) is False, f"decision boundary {key}")
    require(
        decision.get("production_status")
        == "NOT_RUN_NO_MATERIALIZED_G4_ARTIFACT_SET",
        "production status",
    )
    require(
        resolution.get("required_successor_order") == EXPECTED_SUCCESSOR_ORDER,
        "successor order",
    )

    access = _mapping(resolution.get("access_accounting_contract"), "access")
    expected_access = {
        "repository_yaml_documents_read": 4,
        "repository_json_documents_read": 2,
        "operator_directories_listed": 0,
        "operator_paths_searched_or_scanned": 0,
        "operator_json_metadata_files_read": 0,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "holdout_artifact_paths_enumerated": 0,
        "holdout_artifact_bytes_hashed": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "production_preflight_written": False,
        "production_seal_created": False,
        "production_receipt_written": False,
        "model_fitted": False,
        "g4_open": False,
    }
    require(access == expected_access, "access accounting")
    authorization = _mapping(resolution.get("authorization"), "authorization")
    require(
        authorization.get("repository_metadata_resolution")
        == "AUTHORIZED_AND_COMPLETE",
        "metadata authorization",
    )
    require(
        authorization.get("production_preflight")
        == "NOT_AUTHORIZED_PREREQUISITE_ARTIFACT_NOT_MATERIALIZED",
        "preflight authorization",
    )
    for key in (
        "operator_path_search_or_scan",
        "implicit_artifact_discovery",
        "holdout_seal_opening",
        "confirmatory_or_holdout_row_read",
        "outcome_based_selection",
        "g0_measurement_freeze",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"closed authorization {key}",
        )
    return {
        "blocker_class": decision["blocker_class"],
        "production_status": decision["production_status"],
        "logical_partition_reservation_equals_byte_seal": False,
        "next_workstream": decision["next_workstream"],
        "required_successor_order": EXPECTED_SUCCESSOR_ORDER,
        "access_accounting": expected_access,
    }


def run(resolution_path: Path = DEFAULT_RESOLUTION) -> dict[str, Any]:
    resolution_path = resolution_path.resolve(strict=True)
    resolution = _read_document(resolution_path, "yaml", "resolution contract")
    require(resolution.get("schema_version") == RESOLUTION_SCHEMA, "resolution schema")
    require(resolution.get("status") == RESOLUTION_STATUS, "resolution status")

    documents = {
        key: _validate_binding(resolution, key) for key in EXPECTED_BINDINGS
    }
    inputs = _validate_declared_inputs(resolution)
    materialization = _validate_materialization(
        resolution,
        matrix=documents["cross_field_matrix"],
        g0=documents["g0_v2"],
        entry_contract=documents["entry_source_contract_summary"],
        entry_production=documents["entry_source_production_summary"],
    )
    decision = _validate_decision_and_boundary(resolution)
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "declared_inputs": inputs,
        "materialization_replay": materialization,
        "resolution_decision": {
            key: value for key, value in decision.items() if key != "access_accounting"
        },
        "access_accounting": decision["access_accounting"],
        "repository_artifacts": {
            "resolution_contract": {
                "relative_path": str(resolution_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(resolution_path),
            },
            "validator": {
                "relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "authorization": {
            "repository_metadata_resolution": "COMPLETE_INDEPENDENTLY_VERIFIED",
            "operator_path_search_or_scan": "NOT_AUTHORIZED",
            "production_preflight": (
                "NOT_AUTHORIZED_PREREQUISITE_ARTIFACT_NOT_MATERIALIZED"
            ),
            "production_seal_creation_or_receipt_issuance": (
                "NOT_AUTHORIZED_NO_VALID_PREFLIGHT"
            ),
            "holdout_seal_opening": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": resolution["claim_boundary"].strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=Path, default=DEFAULT_RESOLUTION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.resolution)
    payload = json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
