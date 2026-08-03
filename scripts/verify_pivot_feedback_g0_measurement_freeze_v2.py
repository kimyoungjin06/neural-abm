#!/usr/bin/env python3
"""Validate the G0 v2 G-1A binding-only successor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_g0_measurement_freeze.v3"
CONFIG_STATUS = "PENDING_G0_V2_G_MINUS_1A_BOUND_THREE_BLOCKERS_NO_SEAL"
VERIFY_SCHEMA = "pivot_feedback_g0_measurement_freeze_v2_verification.v1"
VERIFY_STATUS = (
    "PASS_G0_V2_BINDING_ONLY_G_MINUS_1A_SATISFIED_"
    "THREE_BLOCKERS_REMAIN_NO_SEAL"
)
G_MINUS_1A_STATUS = (
    "PASS_G_MINUS_1A_BOUNDED_PRIMITIVE_AND_SOURCE_FIDELITY_"
    "WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS"
)
EXPECTED_BINDINGS = {
    "authorization_parent": {
        "path": "experiments/configs/pivot_feedback_gate_state_v52.yaml",
        "sha256": (
            "420dce85a01884dba1d54604f4caea6f0debefcd0aed246265e67598f635a889"
        ),
    },
    "immutable_predecessor": {
        "path": (
            "experiments/configs/pivot_feedback_g0_measurement_freeze_v1.yaml"
        ),
        "sha256": (
            "dc947df2f87220379401c43ea1914b48c71ed04ae474e5695003723d2c591590"
        ),
    },
    "binding_reconciliation_contract": {
        "path": (
            "experiments/configs/"
            "pivot_feedback_g0_g_minus_1a_binding_reconciliation_v1.yaml"
        ),
        "sha256": (
            "cb6c733fdd793590fa8c3cd57776348d2303e03351a33c94c66e269fb6989905"
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
PRESERVED_SUBTREES = (
    "foundation_parents",
    "holdout_access_binding",
    "fixed_design_choices",
    "resolution",
    "inherited_subtree_digests",
    "completion_contract",
    "failure_policy",
    "authorization",
)
EXPECTED_REMAINING_BLOCKERS = (
    "HOLDOUT_ACCESS_SEAL_RECEIPT_ABSENT",
    "SEVENTEEN_QUANTITATIVE_CHOICES_UNRESOLVED",
    "FIFTEEN_REQUIRED_SECTIONS_NOT_FREEZE_READY",
)


class G0MeasurementFreezeV2Error(RuntimeError):
    """Fail-closed G0 v2 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G0MeasurementFreezeV2Error(message)


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
        raise G0MeasurementFreezeV2Error(f"invalid {label}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise G0MeasurementFreezeV2Error(f"invalid {label}: {error}") from error


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


def _validate_predecessor_preservation(
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    for subtree in PRESERVED_SUBTREES:
        require(
            config.get(subtree) == predecessor.get(subtree),
            f"predecessor preserved subtree {subtree}",
        )
    lifecycle = _mapping(config.get("lifecycle"), "lifecycle")
    predecessor_lifecycle = _mapping(
        predecessor.get("lifecycle"),
        "predecessor lifecycle",
    )
    for key, expected in predecessor_lifecycle.items():
        if key == "successor_skeleton_status":
            continue
        require(lifecycle.get(key) == expected, f"lifecycle preservation {key}")
    require(
        lifecycle.get("successor_skeleton_status")
        == "G_MINUS_1A_BOUND_THREE_BLOCKERS_PENDING",
        "G0 v2 lifecycle status",
    )
    return {
        "immutable_predecessor_sha256": EXPECTED_BINDINGS[
            "immutable_predecessor"
        ]["sha256"],
        "preserved_subtrees": list(PRESERVED_SUBTREES),
        "preserved_subtree_count": len(PRESERVED_SUBTREES),
        "predecessor_mutated": False,
    }


def _validate_legacy_binding(
    config: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> dict[str, Any]:
    legacy = _mapping(
        config.get("legacy_g_minus_1_binding"),
        "legacy G-1 binding",
    )
    source = _mapping(
        predecessor.get("g_minus_1_binding"),
        "predecessor G-1 binding",
    )
    require(legacy.get("source_field_name") == "g_minus_1_binding", "legacy source")
    require(
        legacy.get("state") == "PENDING_NOT_PASSED_IMMUTABLE_LINEAGE",
        "legacy state",
    )
    for key in (
        "receipt_relative_path",
        "receipt_sha256",
        "required_schema_version",
        "required_status",
        "result_fingerprint_sha256",
    ):
        require(legacy.get(key) == source.get(key), f"legacy preservation {key}")
    require(legacy.get("lineage_only") is True, "legacy lineage-only state")
    require(
        legacy.get("prerequisite_role_active") is False,
        "legacy prerequisite role",
    )
    require(legacy.get("legacy_pass_inferred") is False, "legacy pass inference")
    return {
        "state": legacy["state"],
        "lineage_only": True,
        "prerequisite_role_active": False,
        "legacy_pass_inferred": False,
    }


def _validate_g_minus_1a_binding(
    config: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _mapping(config.get("g_minus_1a_binding"), "G-1A binding")
    contract_binding = _mapping(
        contract.get("future_g0_v2_binding_contract"),
        "reconciliation future binding",
    )
    expected = {
        "state": "SATISFIED_G_MINUS_1A_BOUNDED_EVIDENCE_ONLY",
        "receipt_relative_path": EXPECTED_BINDINGS["g_minus_1a_summary"]["path"],
        "receipt_sha256": EXPECTED_BINDINGS["g_minus_1a_summary"]["sha256"],
        "required_schema_version": receipt["schema_version"],
        "required_status": receipt["status"],
        "required_gate_id": receipt["gate_id"],
        "config_relative_path": EXPECTED_BINDINGS["g_minus_1a_config"]["path"],
        "config_sha256": EXPECTED_BINDINGS["g_minus_1a_config"]["sha256"],
        "verifier_relative_path": EXPECTED_BINDINGS["g_minus_1a_verifier"]["path"],
        "verifier_sha256": EXPECTED_BINDINGS["g_minus_1a_verifier"]["sha256"],
        "semantic_role": (
            "BOUNDED_PRIMITIVE_AND_SOURCE_FIDELITY_PREREQUISITE_ONLY"
        ),
        "receipt_unresolved_boundary_count": 14,
    }
    for key, expected_value in expected.items():
        require(binding.get(key) == expected_value, f"G-1A binding {key}")
    for key in (
        "receipt_relative_path",
        "receipt_sha256",
        "required_schema_version",
        "required_status",
        "required_gate_id",
        "config_relative_path",
        "config_sha256",
        "verifier_relative_path",
        "verifier_sha256",
        "semantic_role",
    ):
        require(
            binding.get(key) == contract_binding.get(key),
            f"G-1A contract closure {key}",
        )
    require(
        binding.get("current_snapshot_proxy_and_author_token_caveats_preserved")
        is True,
        "G-1A caveat preservation",
    )
    for key in (
        "may_pass_or_relabel_legacy_g_minus_1",
        "may_by_itself_pass_g0",
        "may_by_itself_authorize_model_fitting",
        "may_by_itself_open_g4",
        "may_by_itself_establish_scientific_effect",
    ):
        require(binding.get(key) is False, f"G-1A boundary {key}")
    require(
        _mapping(
            _lookup(contract, "preserved_receipt_boundaries.unresolved_boundaries"),
            "contract receipt boundaries",
        )
        == _mapping(
            receipt.get("unresolved_boundaries"),
            "receipt unresolved boundaries",
        ),
        "G-1A unresolved boundary closure",
    )
    require(
        str(_lookup(contract, "preserved_receipt_boundaries.exact_claim_boundary")).strip()
        == str(receipt.get("claim_boundary")).strip(),
        "G-1A claim boundary closure",
    )
    return {
        "state": binding["state"],
        "receipt_sha256": binding["receipt_sha256"],
        "receipt_unresolved_boundary_count": 14,
        "claim_boundary_preserved_exactly": True,
        "legacy_g_minus_1_passed": False,
        "g0_passed": False,
    }


def _validate_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    readiness = _mapping(config.get("readiness"), "readiness")
    require(
        readiness.get("status") == "G0_NOT_READY_THREE_BLOCKERS_REMAIN",
        "readiness status",
    )
    require(
        readiness.get("resolved_blocker") == "G_MINUS_1_RECEIPT_ABSENT",
        "resolved blocker",
    )
    require(readiness.get("resolved_blocker_count") == 1, "resolved blocker count")
    require(
        readiness.get("remaining_blocker_count") == 3,
        "remaining blocker count",
    )
    require(
        tuple(readiness.get("remaining_blockers", []))
        == EXPECTED_REMAINING_BLOCKERS,
        "remaining blockers",
    )
    require(
        readiness.get("g_minus_1a_bounded_prerequisite_satisfied") is True,
        "G-1A prerequisite",
    )
    require(
        readiness.get("legacy_g_minus_1_passed") is False,
        "legacy G-1 readiness",
    )
    require(readiness.get("g0_freeze_ready") is False, "G0 freeze readiness")
    require(
        _lookup(config, "holdout_access_binding.state") == "PENDING",
        "holdout state",
    )
    require(
        _lookup(config, "holdout_access_binding.receipt_sha256") is None,
        "holdout receipt hash",
    )
    require(
        _lookup(config, "resolution.unresolved_quantitative_choice_count") == 17,
        "unresolved choice count",
    )
    require(
        _lookup(config, "resolution.unresolved_required_section_count") == 15,
        "unresolved section count",
    )
    return {
        "resolved_blocker": readiness["resolved_blocker"],
        "remaining_blockers": list(EXPECTED_REMAINING_BLOCKERS),
        "remaining_blocker_count": 3,
        "g_minus_1a_bounded_prerequisite_satisfied": True,
        "g0_freeze_ready": False,
    }


def _validate_fail_closed_boundary(config: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(config.get("lifecycle"), "lifecycle")
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
        require(lifecycle.get(key) == expected, f"lifecycle boundary {key}")
    access = _mapping(
        config.get("access_accounting_contract"),
        "access accounting",
    )
    expected_access = {
        "repository_documents_read": 7,
        "operator_json_metadata_files_read": 0,
        "operator_parquet_files_opened": 0,
        "production_row_level_files_opened": 0,
        "confirmatory_or_holdout_rows_read": 0,
        "api_requests_created_or_submitted": 0,
        "model_fitted": False,
        "g4_open": False,
    }
    require(access == expected_access, "access accounting")
    require(
        _lookup(config, "completion_contract.pending_contract_sealable") is False,
        "pending contract sealability",
    )
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "additional_quantitative_choice_resolution",
        "required_section_completion",
        "operator_parquet_or_row_level_read",
        "confirmatory_or_holdout_access",
        "outcome_based_selection",
        "final_candidate_eligibility",
        "g0_measurement_freeze_seal",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"authorization {key}")
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect",
    )
    return {
        "access_accounting": expected_access,
        "pending_contract_sealable": False,
        "pass_g0": False,
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _read_yaml(config_path, "G0 v2 config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(config.get("status") == CONFIG_STATUS, "config status")
    require(
        config.get("mode")
        == "VERSIONED_G_MINUS_1A_BINDING_ONLY_SUCCESSOR_THREE_BLOCKERS_REMAIN",
        "config mode",
    )

    parent, parent_binding = _validate_document_binding(
        config.get("authorization_parent"),
        label="authorization parent",
        expected_key="authorization_parent",
    )
    predecessor, predecessor_binding = _validate_document_binding(
        config.get("immutable_predecessor"),
        label="immutable predecessor",
        expected_key="immutable_predecessor",
    )
    contract, contract_binding = _validate_document_binding(
        config.get("binding_reconciliation_contract"),
        label="binding reconciliation contract",
        expected_key="binding_reconciliation_contract",
    )
    require(
        _lookup(parent, "authorized_g0_v2_binding_only_successor_scope.may_add_exact_g_minus_1a_binding")
        is True,
        "parent exact G-1A binding authorization",
    )
    require(
        _lookup(parent, "authorized_g0_v2_binding_only_successor_scope.may_seal_or_pass_g0")
        is False,
        "parent G0 pass boundary",
    )

    g_minus_1a_spec = _mapping(config.get("g_minus_1a_binding"), "G-1A binding")
    summary_path = _safe_repo_path(
        g_minus_1a_spec.get("receipt_relative_path"),
        "G-1A receipt",
    )
    require(
        g_minus_1a_spec.get("receipt_relative_path")
        == EXPECTED_BINDINGS["g_minus_1a_summary"]["path"],
        "G-1A receipt path",
    )
    require(
        _sha256(summary_path) == EXPECTED_BINDINGS["g_minus_1a_summary"]["sha256"],
        "G-1A receipt byte hash",
    )
    receipt = _read_json(summary_path, "G-1A receipt")
    require(receipt.get("schema_version") == "pivot_feedback_g_minus_1a_receipt.v1", "receipt schema")
    require(receipt.get("status") == G_MINUS_1A_STATUS, "receipt status")

    config_path_bound = _safe_repo_path(
        g_minus_1a_spec.get("config_relative_path"),
        "G-1A receipt config",
    )
    require(
        _sha256(config_path_bound)
        == EXPECTED_BINDINGS["g_minus_1a_config"]["sha256"],
        "G-1A receipt config hash",
    )
    receipt_config = _read_yaml(config_path_bound, "G-1A receipt config")
    require(
        receipt_config.get("schema_version")
        == "pivot_feedback_g_minus_1a_receipt_config.v1",
        "G-1A receipt config schema",
    )
    verifier_path = _safe_repo_path(
        g_minus_1a_spec.get("verifier_relative_path"),
        "G-1A receipt verifier",
    )
    require(
        _sha256(verifier_path)
        == EXPECTED_BINDINGS["g_minus_1a_verifier"]["sha256"],
        "G-1A receipt verifier hash",
    )
    require(
        _lookup(receipt, "provenance.config_sha256")
        == g_minus_1a_spec.get("config_sha256"),
        "G-1A config provenance",
    )
    require(
        _lookup(receipt, "provenance.verifier_sha256")
        == g_minus_1a_spec.get("verifier_sha256"),
        "G-1A verifier provenance",
    )

    predecessor_replay = _validate_predecessor_preservation(config, predecessor)
    legacy_replay = _validate_legacy_binding(config, predecessor)
    g_minus_1a_replay = _validate_g_minus_1a_binding(
        config,
        contract=contract,
        receipt=receipt,
    )
    readiness_replay = _validate_readiness(config)
    boundary_replay = _validate_fail_closed_boundary(config)

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "repository_bindings": {
            "authorization_parent": parent_binding,
            "immutable_predecessor": predecessor_binding,
            "binding_reconciliation_contract": contract_binding,
            "g_minus_1a_receipt": {
                "relative_path": EXPECTED_BINDINGS["g_minus_1a_summary"]["path"],
                "sha256": EXPECTED_BINDINGS["g_minus_1a_summary"]["sha256"],
                "schema_version": receipt["schema_version"],
            },
            "g_minus_1a_config": {
                "relative_path": EXPECTED_BINDINGS["g_minus_1a_config"]["path"],
                "sha256": EXPECTED_BINDINGS["g_minus_1a_config"]["sha256"],
                "schema_version": receipt_config["schema_version"],
            },
            "g_minus_1a_verifier": {
                "relative_path": EXPECTED_BINDINGS["g_minus_1a_verifier"]["path"],
                "sha256": EXPECTED_BINDINGS["g_minus_1a_verifier"]["sha256"],
            },
        },
        "predecessor_preservation_replay": predecessor_replay,
        "legacy_binding_replay": legacy_replay,
        "g_minus_1a_binding_replay": g_minus_1a_replay,
        "readiness_replay": readiness_replay,
        "fail_closed_boundary_replay": boundary_replay,
        "authorization": {
            "g0_v2_binding_only_successor": "COMPLETE_INDEPENDENTLY_VALIDATED",
            "holdout_seal_contract": "REQUIRES_SUCCESSOR_GATE",
            "quantitative_choice_resolution": "NOT_AUTHORIZED",
            "required_section_completion": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "G0 v2 is a valid binding-only successor. It preserves eight G0 "
            "v1 subtrees exactly, retains legacy G-1 as pending immutable "
            "lineage, and binds the exact G-1A receipt without relabeling. "
            "Only the absent-G-1-receipt blocker is resolved; the holdout "
            "receipt, seventeen quantitative choices, and fifteen required "
            "sections remain blockers, so G0 remains unsealed. No operator "
            "row data, confirmatory or holdout data, API request, model, G4 "
            "result, causality, or scientific effect is established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config.resolve(strict=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
