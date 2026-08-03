#!/usr/bin/env python3
"""Validate the metadata-only future holdout-access seal contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_holdout_access_seal_v1.yaml"
)
CONTRACT_SCHEMA = "pivot_feedback_holdout_access_seal_contract.v1"
CONTRACT_STATUS = "PENDING_INDEPENDENT_VALIDATION_FUTURE_SEAL_CREATION_ONLY"
VERIFY_SCHEMA = "pivot_feedback_holdout_access_seal_contract_verification.v1"
VERIFY_STATUS = (
    "PASS_METADATA_ONLY_HOLDOUT_SEAL_CONTRACT_"
    "PARTITION_FROZEN_CREATION_AND_OPENING_SEPARATE"
)
EXPECTED_RECEIPT_SCHEMA = "pivot_feedback_holdout_access_seal_outputs.v1"
EXPECTED_RECEIPT_STATUS = "PASS_CONFIRMATORY_PARTITIONS_SEALED_AND_UNOPENED"
EXPECTED_BINDINGS = {
    "authorization_parent": {
        "path": "experiments/configs/pivot_feedback_gate_state_v53.yaml",
        "sha256": (
            "e658bf3e8d92eb7cf8f901315d3584de7cb00202d14810fb3de847e056b6f7ae"
        ),
        "schema": "pivot_feedback_gate_state.v53",
        "status": (
            "VALID_G0_V2_G_MINUS_1A_BOUND_THREE_BLOCKERS_"
            "HOLDOUT_SEAL_CONTRACT_AUTHORIZED"
        ),
    },
    "bound_g0_v2": {
        "path": "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml",
        "sha256": (
            "a2cb0cbf444a7c128eecc40a815b80d5fe86f617a36849330612491b03934eb3"
        ),
        "schema": "pivot_feedback_g0_measurement_freeze.v3",
        "status": "PENDING_G0_V2_G_MINUS_1A_BOUND_THREE_BLOCKERS_NO_SEAL",
    },
    "validated_prerequisite_closure_plan": {
        "path": (
            "experiments/configs/pivot_feedback_g0_prerequisite_"
            "closure_plan_v1.yaml"
        ),
        "sha256": (
            "45db08ea9e51d861270c1be4b8ac675e56e30df7b57268f515fd665708b2c4fc"
        ),
        "schema": "pivot_feedback_g0_prerequisite_closure_plan.v1",
        "status": (
            "PENDING_INDEPENDENT_VALIDATION_FOUR_WORKSTREAM_PLAN_NO_EXECUTION"
        ),
    },
}
EXPECTED_PHASES = (
    "CONTRACT_VALIDATION",
    "SEAL_CREATION",
    "SEAL_INDEPENDENT_VERIFICATION",
    "SEAL_OPENING",
)
EXPECTED_PROTOCOL_STEPS = (
    "VERIFY_SEPARATE_SUCCESSOR_GATE_AUTHORIZES_SEAL_CREATION",
    "VERIFY_EXACT_CONTRACT_G0_V2_AND_CLOSURE_PLAN_HASHES",
    "RESOLVE_EXPLICIT_AUTHORIZED_ARTIFACT_SET_WITHOUT_ROW_PARSING",
    "REJECT_MISSING_NONREGULAR_SYMLINKED_DUPLICATE_OR_PREVIOUSLY_OPENED_ARTIFACT",
    "HASH_AUTHORIZED_ARTIFACT_BYTES_IN_PRIVATE_STAGING_AREA",
    "WRITE_AND_FSYNC_STAGING_MANIFEST",
    "REHASH_FINAL_STAGED_ARTIFACTS_AND_MANIFEST",
    "ATOMICALLY_RENAME_STAGING_DIRECTORY_TO_FINAL_SEALED_DIRECTORY",
    "REVALIDATE_FINAL_DIRECTORY_AND_ZERO_PRIOR_OPEN_EVENTS",
    "WRITE_COMPLETION_RECEIPT_LAST",
    "RUN_INDEPENDENT_READ_ONLY_SEAL_VERIFIER",
)
EXPECTED_MANIFEST_FIELDS = (
    "schema_version",
    "status",
    "contract_relative_path",
    "contract_sha256",
    "g0_v2_relative_path",
    "g0_v2_sha256",
    "prerequisite_closure_plan_relative_path",
    "prerequisite_closure_plan_sha256",
    "partition_contract",
    "artifact_count",
    "artifact_entries",
    "manifest_fingerprint_sha256",
    "created_at_utc",
)
EXPECTED_ARTIFACT_ENTRY_FIELDS = (
    "canonical_relative_path",
    "byte_size",
    "sha256",
    "regular_file",
    "symlink",
)
EXPECTED_RECEIPT_FIELDS = (
    "schema_version",
    "status",
    "contract_relative_path",
    "contract_sha256",
    "manifest_operator_relative_path",
    "manifest_sha256",
    "partition_contract",
    "artifact_count",
    "artifact_bytes_total",
    "manifest_fingerprint_sha256",
    "seal_created_at_utc",
    "receipt_written_at_utc",
    "receipt_written_last",
    "seal_opened",
    "prior_open_event_count",
    "confirmatory_or_holdout_rows_read",
    "independent_verification_required",
)


class HoldoutAccessSealContractError(RuntimeError):
    """Fail-closed metadata-only holdout-seal contract error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HoldoutAccessSealContractError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


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
        raise HoldoutAccessSealContractError(f"invalid {label}: {error}") from error


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
    contract: Mapping[str, Any], key: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = _mapping(contract.get(key), key)
    expected = EXPECTED_BINDINGS[key]
    require(spec.get("relative_path") == expected["path"], f"{key} path")
    require(spec.get("sha256") == expected["sha256"], f"{key} declared hash")
    path = _safe_repo_path(spec.get("relative_path"), key)
    require(_sha256(path) == expected["sha256"], f"{key} byte hash")
    document = _read_yaml(path, key)
    require(
        document.get("schema_version") == expected["schema"],
        f"{key} schema",
    )
    require(document.get("status") == expected["status"], f"{key} status")
    require(
        spec.get("required_schema_version") == expected["schema"],
        f"{key} required schema",
    )
    require(
        spec.get("required_status") == expected["status"],
        f"{key} required status",
    )
    for dotted, expected_value in _mapping(
        spec.get("required_fields", {}), f"{key} required fields"
    ).items():
        require(
            _lookup(document, dotted) == expected_value,
            f"{key} field {dotted}",
        )
    require(spec.get("mutation_allowed") is False, f"{key} mutation boundary")
    return document, {
        "relative_path": expected["path"],
        "sha256": expected["sha256"],
        "schema_version": expected["schema"],
    }


def _validate_partition(
    contract: Mapping[str, Any], *, g0_v2: Mapping[str, Any]
) -> dict[str, Any]:
    partition = _mapping(contract.get("partition_contract"), "partition contract")
    expected = {
        "partition_id": "g4_temporal_holdout",
        "entry_start_year": 2020,
        "entry_end_year": 2022,
        "entry_years": [2020, 2021, 2022],
        "outcome_followup_through_year": 2025,
        "temporal_holdout": True,
        "domain_cold": True,
        "author_cold": True,
        "outcome_blind_definition": True,
        "partition_membership_immutable_after_contract_validation": True,
        "future_seal_must_bind_this_partition_exactly": True,
    }
    require(partition == expected, "partition contract exact freeze")

    partitions = _lookup(g0_v2, "fixed_design_choices.temporal_partitions")
    require(isinstance(partitions, list), "G0 v2 temporal partitions")
    matches = [item for item in partitions if item.get("partition_id") == expected["partition_id"]]
    require(len(matches) == 1, "G0 v2 holdout partition uniqueness")
    source = _mapping(matches[0], "G0 v2 holdout partition")
    require(source.get("entry_start_year") == 2020, "G0 v2 holdout start year")
    require(source.get("entry_end_year") == 2022, "G0 v2 holdout end year")
    require(
        source.get("outcome_followup_through_year") == 2025,
        "G0 v2 holdout follow-up",
    )
    require(source.get("sealed") is True, "G0 v2 holdout declared sealed")
    require(
        _lookup(g0_v2, "fixed_design_choices.full_panel.g4_domain_and_author_cold")
        is True,
        "G0 v2 domain and author cold",
    )
    return expected


def _validate_plan_route(plan: Mapping[str, Any]) -> dict[str, Any]:
    route = _mapping(
        _lookup(plan, "workstreams.HOLDOUT_ACCESS_SEAL_RECEIPT_CLOSURE"),
        "holdout closure route",
    )
    require(
        route.get("current_state")
        == "NULL_BINDING_NO_VERSIONED_CONTRACT_OR_RECEIPT_BOUND",
        "holdout route current state",
    )
    diagnostic = _mapping(route.get("diagnostic"), "holdout diagnostic")
    require(
        diagnostic.get("expected_schema_version") == EXPECTED_RECEIPT_SCHEMA,
        "future receipt schema",
    )
    require(
        diagnostic.get("expected_status") == EXPECTED_RECEIPT_STATUS,
        "future receipt status",
    )
    require(diagnostic.get("receipt_relative_path") is None, "receipt path pending")
    require(diagnostic.get("receipt_sha256") is None, "receipt hash pending")
    expected_acceptance = [
        "SEPARATE_SEAL_CREATION_FROM_SEAL_OPENING",
        "FREEZE_PARTITION_YEARS_AND_DOMAIN_AUTHOR_COLD_RULE",
        "REQUIRE_ATOMIC_MANIFEST_AND_HASH_CLOSURE_BEFORE_ANY_CONFIRMATORY_READ",
        "REQUIRE_RECEIPT_WRITTEN_LAST_AND_INDEPENDENTLY_VERIFIED",
        "FAIL_CLOSED_ON_MISSING_CHANGED_OR_PREVIOUSLY_OPENED_ARTIFACT",
        "MAKE_SEAL_RECEIPT_NONAUTHORIZING_FOR_MODEL_FITTING_OR_G4_OPEN",
    ]
    require(
        route.get("acceptance_criteria") == expected_acceptance,
        "holdout route acceptance criteria",
    )
    return {
        "future_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
        "future_receipt_status": EXPECTED_RECEIPT_STATUS,
        "receipt_currently_bound": False,
        "acceptance_criterion_count": len(expected_acceptance),
    }


def _validate_phase_separation(contract: Mapping[str, Any]) -> dict[str, Any]:
    phases = _mapping(
        contract.get("phase_separation_contract"), "phase separation contract"
    )
    require(tuple(phases.get("ordered_phases", ())) == EXPECTED_PHASES, "phase order")
    validation = _mapping(phases.get("contract_validation"), "contract validation")
    require(validation.get("state") == "AUTHORIZED_METADATA_ONLY", "validation state")
    require(
        validation.get("may_define_future_manifest_and_receipt_schemas") is True,
        "future schema definition",
    )
    for key in (
        "may_enumerate_or_hash_holdout_artifacts",
        "may_create_or_issue_seal",
        "may_open_seal",
    ):
        require(validation.get(key) is False, f"contract validation boundary {key}")

    creation = _mapping(phases.get("future_seal_creation"), "future seal creation")
    require(
        creation.get("state") == "REQUIRES_SEPARATE_SUCCESSOR_GATE",
        "future creation gate",
    )
    for key in (
        "must_bind_this_contract_byte_for_byte",
        "must_bind_g0_v2_byte_for_byte",
        "must_bind_prerequisite_closure_plan_byte_for_byte",
        "may_enumerate_and_hash_only_explicitly_authorized_holdout_artifacts",
        "may_issue_receipt_only_after_atomic_finalization",
    ):
        require(creation.get(key) is True, f"future creation requirement {key}")
    for key in ("may_parse_or_analyze_holdout_rows", "may_open_seal"):
        require(creation.get(key) is False, f"future creation boundary {key}")

    verification = _mapping(
        phases.get("future_seal_independent_verification"),
        "future seal independent verification",
    )
    require(verification.get("state") == "REQUIRED_AFTER_RECEIPT", "verification state")
    for key in (
        "must_rehash_final_artifacts_and_manifest",
        "must_verify_receipt_was_written_last",
        "must_verify_zero_prior_open_events",
    ):
        require(verification.get(key) is True, f"verification requirement {key}")
    for key in ("may_parse_or_analyze_holdout_rows", "may_open_seal"):
        require(verification.get(key) is False, f"verification boundary {key}")

    opening = _mapping(phases.get("future_seal_opening"), "future seal opening")
    require(
        opening.get("state") == "REQUIRES_DISTINCT_G4_OPENING_GATE",
        "opening gate",
    )
    for key in (
        "creation_receipt_alone_authorizes_opening",
        "independent_verification_alone_authorizes_opening",
        "may_be_combined_with_creation_phase",
    ):
        require(opening.get(key) is False, f"opening separation {key}")
    return {
        "ordered_phases": list(EXPECTED_PHASES),
        "creation_requires_successor_gate": True,
        "opening_requires_distinct_g4_gate": True,
        "creation_and_opening_separate": True,
    }


def _validate_future_artifact_contracts(contract: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _mapping(contract.get("future_manifest_contract"), "manifest contract")
    require(
        manifest.get("required_schema_version")
        == "pivot_feedback_holdout_access_seal_manifest.v1",
        "manifest schema",
    )
    require(
        manifest.get("required_status")
        == "SEALED_ARTIFACT_MANIFEST_FINALIZED_UNOPENED",
        "manifest status",
    )
    require(tuple(manifest.get("required_top_level_fields", ())) == EXPECTED_MANIFEST_FIELDS, "manifest fields")
    require(tuple(manifest.get("artifact_entry_required_fields", ())) == EXPECTED_ARTIFACT_ENTRY_FIELDS, "artifact entry fields")
    for key in (
        "canonical_paths_unique",
        "regular_files_only",
        "symlinks_forbidden",
        "duplicate_paths_forbidden",
        "artifact_set_must_be_explicit_before_finalization",
        "staging_manifest_must_not_be_treated_as_final",
    ):
        require(manifest.get(key) is True, f"manifest safety {key}")

    receipt = _mapping(contract.get("future_receipt_contract"), "receipt contract")
    require(receipt.get("required_schema_version") == EXPECTED_RECEIPT_SCHEMA, "receipt schema")
    require(receipt.get("required_status") == EXPECTED_RECEIPT_STATUS, "receipt status")
    require(receipt.get("receipt_written_last") is True, "receipt written last")
    require(tuple(receipt.get("required_top_level_fields", ())) == EXPECTED_RECEIPT_FIELDS, "receipt fields")
    require(
        _mapping(receipt.get("required_assertions"), "receipt assertions")
        == {
            "receipt_written_last": True,
            "seal_opened": False,
            "prior_open_event_count": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "independent_verification_required": True,
        },
        "receipt assertions",
    )
    for key in (
        "receipt_does_not_authorize_model_fitting",
        "receipt_does_not_authorize_g4_open",
        "receipt_does_not_establish_scientific_effect",
    ):
        require(receipt.get(key) is True, f"receipt nonauthorization {key}")
    return {
        "manifest_schema": manifest["required_schema_version"],
        "receipt_schema": receipt["required_schema_version"],
        "receipt_status": receipt["required_status"],
        "receipt_written_last": True,
        "receipt_nonauthorizing_for_model_or_g4": True,
    }


def _validate_protocol_and_failure_policy(contract: Mapping[str, Any]) -> dict[str, Any]:
    protocol = _mapping(contract.get("atomic_creation_protocol"), "atomic protocol")
    require(tuple(protocol.get("ordered_steps", ())) == EXPECTED_PROTOCOL_STEPS, "atomic protocol order")
    for key in (
        "receipt_must_not_exist_before_final_directory_revalidation",
        "final_manifest_and_artifact_hash_closure_required_before_receipt",
        "receipt_write_must_be_atomic",
        "receipt_overwrite_forbidden",
    ):
        require(protocol.get(key) is True, f"atomic protocol requirement {key}")
    require(
        protocol.get("seal_creation_failure_writes_pass_receipt") is False,
        "failure receipt boundary",
    )

    policy = _mapping(contract.get("fail_closed_policy"), "fail-closed policy")
    expected_actions = {
        "missing_artifact_action": "FAIL_NO_RECEIPT",
        "changed_artifact_action": "FAIL_NO_RECEIPT",
        "nonregular_or_symlink_artifact_action": "FAIL_NO_RECEIPT",
        "duplicate_or_noncanonical_path_action": "FAIL_NO_RECEIPT",
        "previously_opened_artifact_action": "FAIL_NO_RECEIPT",
        "prior_receipt_or_manifest_mismatch_action": "FAIL_NO_RECEIPT",
        "partial_staging_action": "QUARANTINE_NO_RECEIPT",
        "failed_independent_verification_action": "INVALIDATE_SEAL_NO_OPENING",
        "any_confirmatory_read_before_verified_receipt_action": "BLOCK_AND_INVALIDATE",
    }
    for key, value in expected_actions.items():
        require(policy.get(key) == value, f"fail-closed action {key}")
    for key in (
        "missing_artifact_may_be_ignored",
        "changed_artifact_may_be_rebased_silently",
        "previously_opened_artifact_may_be_resealed",
    ):
        require(policy.get(key) is False, f"fail-closed boundary {key}")
    return {
        "ordered_step_count": len(EXPECTED_PROTOCOL_STEPS),
        "receipt_written_after_final_revalidation": True,
        "missing_changed_or_previously_opened_fail_closed": True,
        "failed_verification_blocks_opening": True,
    }


def _validate_current_boundary(contract: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(contract.get("lifecycle"), "lifecycle")
    require(
        lifecycle.get("contract_status") == "PENDING_INDEPENDENT_VALIDATION",
        "lifecycle contract status",
    )
    false_fields = (
        "holdout_artifact_paths_enumerated",
        "holdout_artifact_bytes_hashed",
        "staging_manifest_written",
        "final_manifest_written",
        "seal_created_or_issued",
        "seal_receipt_written",
        "seal_independently_verified",
        "seal_opened",
        "operator_parquet_or_row_level_data_read",
        "confirmatory_or_holdout_rows_read",
        "api_requests_created_or_submitted",
        "g0_v2_changed",
        "quantitative_choices_resolved",
        "required_sections_completed",
        "pass_g0",
        "model_fitted",
        "g4_open",
    )
    for key in false_fields:
        require(lifecycle.get(key) is False, f"lifecycle boundary {key}")

    access = _mapping(contract.get("access_accounting_contract"), "access accounting")
    expected_access = {
        "repository_documents_read": 4,
        "operator_json_metadata_files_read": 0,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "holdout_artifact_paths_enumerated": 0,
        "holdout_artifact_bytes_hashed": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    require(access == expected_access, "access accounting exact boundary")

    authorization = _mapping(contract.get("authorization"), "authorization")
    require(
        authorization.get("contract_validation") == "AUTHORIZED_METADATA_ONLY",
        "contract validation authorization",
    )
    require(
        authorization.get("future_seal_creation_or_receipt_issuance")
        == "REQUIRES_SEPARATE_SUCCESSOR_GATE",
        "future seal creation authorization",
    )
    expected_closed = {
        "seal_opening": "NOT_AUTHORIZED",
        "holdout_artifact_enumeration_or_hashing": "NOT_AUTHORIZED",
        "operator_parquet_or_row_level_read": "NOT_AUTHORIZED",
        "confirmatory_or_holdout_access": "NOT_AUTHORIZED",
        "outcome_based_selection": "NOT_AUTHORIZED",
        "g0_quantitative_choice_resolution": "NOT_AUTHORIZED",
        "g0_required_section_completion": "NOT_AUTHORIZED",
        "g0_measurement_freeze": "NOT_AUTHORIZED",
        "model_fitting": "NOT_AUTHORIZED",
        "g4_open": "NOT_AUTHORIZED",
        "scientific_effect": "NOT_ESTABLISHED",
    }
    for key, value in expected_closed.items():
        require(authorization.get(key) == value, f"authorization boundary {key}")
    return {
        "access_accounting": expected_access,
        "seal_created": False,
        "receipt_written": False,
        "holdout_opened": False,
        "model_fitted": False,
        "g4_open": False,
    }


def run(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    contract = _read_yaml(contract_path, "holdout seal contract")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")

    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for key in EXPECTED_BINDINGS:
        documents[key], bindings[key] = _validate_binding(contract, key)

    partition_replay = _validate_partition(
        contract, g0_v2=documents["bound_g0_v2"]
    )
    plan_route_replay = _validate_plan_route(
        documents["validated_prerequisite_closure_plan"]
    )
    phase_replay = _validate_phase_separation(contract)
    future_artifact_replay = _validate_future_artifact_contracts(contract)
    atomic_replay = _validate_protocol_and_failure_policy(contract)
    current_boundary_replay = _validate_current_boundary(contract)

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "contract_sha256": _sha256(contract_path),
        "repository_bindings": bindings,
        "partition_replay": partition_replay,
        "plan_route_replay": plan_route_replay,
        "phase_separation_replay": phase_replay,
        "future_artifact_contract_replay": future_artifact_replay,
        "atomic_and_fail_closed_replay": atomic_replay,
        "current_boundary_replay": current_boundary_replay,
        "authorization": {
            "holdout_seal_contract": "COMPLETE_INDEPENDENTLY_VALIDATED",
            "future_seal_creation_or_receipt_issuance": "REQUIRES_SUCCESSOR_GATE",
            "holdout_artifact_enumeration_or_hashing": "NOT_AUTHORIZED",
            "seal_opening": "NOT_AUTHORIZED",
            "g0_quantitative_choice_resolution": "NOT_AUTHORIZED",
            "g0_required_section_completion": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The metadata-only holdout-seal contract is independently valid. "
            "It fixes the 2020-2022 entry partition, follow-up through 2025, "
            "domain- and author-cold rules, and future manifest, receipt, "
            "atomic-finalization, independent-verification, and fail-closed "
            "requirements. It keeps seal creation distinct from seal opening. "
            "No holdout artifact path or byte was enumerated or hashed, no "
            "seal or receipt was created, no confirmatory or holdout row was "
            "read, and no G0 pass, model, G4 result, causality, or scientific "
            "effect is established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    print(json.dumps(run(args.contract), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
