#!/usr/bin/env python3
"""Validate the metadata-only future G0 v2 G-1A binding contract."""

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
    / "experiments/configs/"
    "pivot_feedback_g0_g_minus_1a_binding_reconciliation_v1.yaml"
)
CONTRACT_SCHEMA = "pivot_feedback_g0_g_minus_1a_binding_reconciliation.v1"
CONTRACT_STATUS = "PENDING_INDEPENDENT_VALIDATION_FUTURE_G0_V2_BINDING_ONLY"
VERIFY_SCHEMA = (
    "pivot_feedback_g0_g_minus_1a_binding_reconciliation_verification.v1"
)
VERIFY_STATUS = (
    "PASS_METADATA_ONLY_G_MINUS_1A_BINDING_RECONCILIATION_"
    "FUTURE_G0_V2_ONLY_NO_RELABEL"
)
G_MINUS_1A_STATUS = (
    "PASS_G_MINUS_1A_BOUNDED_PRIMITIVE_AND_SOURCE_FIDELITY_"
    "WITH_CURRENT_SNAPSHOT_PROXY_LIMITATIONS"
)
EXPECTED_BINDINGS = {
    "authorization_parent": {
        "path": "experiments/configs/pivot_feedback_gate_state_v51.yaml",
        "sha256": (
            "b14b1ed090b2a5fc4f0d77101c03905f1242bb8ba8a36552e47f2587b18eb599"
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
    "validated_closure_plan": {
        "path": (
            "experiments/configs/pivot_feedback_g0_prerequisite_"
            "closure_plan_v1.yaml"
        ),
        "sha256": (
            "45db08ea9e51d861270c1be4b8ac675e56e30df7b57268f515fd665708b2c4fc"
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
EXPECTED_PRESERVED_SUBTREES = (
    "foundation_parents",
    "holdout_access_binding",
    "fixed_design_choices",
    "resolution",
    "inherited_subtree_digests",
    "completion_contract",
    "failure_policy",
    "authorization",
)
EXPECTED_DELTAS = (
    "RETAIN_LEGACY_G_MINUS_1_AS_PENDING_IMMUTABLE_LINEAGE",
    "ADD_EXACT_G_MINUS_1A_BOUNDED_EVIDENCE_BINDING",
)


class GMinus1ABindingReconciliationError(RuntimeError):
    """Fail-closed G-1A binding-reconciliation validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GMinus1ABindingReconciliationError(message)


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
        raise GMinus1ABindingReconciliationError(
            f"invalid {label}: {error}"
        ) from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GMinus1ABindingReconciliationError(
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


def _validate_source_diagnosis(
    contract: Mapping[str, Any],
    *,
    skeleton: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    diagnosis = _mapping(
        contract.get("source_binding_diagnosis"),
        "source binding diagnosis",
    )
    legacy_binding = _mapping(
        skeleton.get("g_minus_1_binding"),
        "G0 v1 G-1 binding",
    )
    expected_pairs = {
        "immutable_field_name": "g_minus_1_binding",
        "immutable_state": legacy_binding["state"],
        "immutable_receipt_relative_path": legacy_binding["receipt_relative_path"],
        "immutable_receipt_sha256": legacy_binding["receipt_sha256"],
        "immutable_required_schema_version": legacy_binding[
            "required_schema_version"
        ],
        "immutable_required_status": legacy_binding["required_status"],
        "available_gate_id": receipt["gate_id"],
        "available_schema_version": receipt["schema_version"],
        "available_status": receipt["status"],
        "available_receipt_sha256": EXPECTED_BINDINGS["g_minus_1a_summary"][
            "sha256"
        ],
        "mismatch_class": "CONTRACT_GENERATION_MISMATCH_NOT_EVIDENCE_ABSENCE",
    }
    for key, expected in expected_pairs.items():
        require(diagnosis.get(key) == expected, f"diagnosis {key}")
    for key in (
        "schema_aliasing_allowed",
        "status_relabeling_allowed",
        "legacy_pass_inference_allowed",
    ):
        require(diagnosis.get(key) is False, f"diagnosis boundary {key}")
    return {
        "mismatch_class": diagnosis["mismatch_class"],
        "schema_aliasing_allowed": False,
        "status_relabeling_allowed": False,
        "legacy_pass_inference_allowed": False,
    }


def _validate_future_binding(
    contract: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    binding = _mapping(
        contract.get("future_g0_v2_binding_contract"),
        "future G0 v2 binding",
    )
    expected = {
        "future_field_name": "g_minus_1a_binding",
        "legacy_field_name_if_retained": "legacy_g_minus_1_binding",
        "legacy_field_state_if_retained": (
            "PENDING_NOT_PASSED_IMMUTABLE_LINEAGE"
        ),
        "binding_state_if_exact": (
            "SATISFIED_G_MINUS_1A_BOUNDED_EVIDENCE_ONLY"
        ),
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
    }
    for key, expected_value in expected.items():
        require(binding.get(key) == expected_value, f"future binding {key}")
    require(
        binding.get("may_satisfy_future_g0_v2_g_minus_1a_prerequisite") is True,
        "future G-1A prerequisite satisfaction",
    )
    for key in (
        "may_pass_or_relabel_legacy_g_minus_1",
        "may_by_itself_pass_g0",
        "may_by_itself_authorize_model_fitting",
        "may_by_itself_open_g4",
        "may_by_itself_establish_scientific_effect",
    ):
        require(binding.get(key) is False, f"future binding boundary {key}")
    return {
        "future_field_name": binding["future_field_name"],
        "binding_state_if_exact": binding["binding_state_if_exact"],
        "exact_receipt_sha256": binding["receipt_sha256"],
        "legacy_g_minus_1_passed": False,
        "g0_passed": False,
    }


def _validate_preserved_boundaries(
    contract: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    preserved = _mapping(
        contract.get("preserved_receipt_boundaries"),
        "preserved receipt boundaries",
    )
    require(
        _mapping(
            preserved.get("unresolved_boundaries"),
            "preserved unresolved boundaries",
        )
        == _mapping(
            receipt.get("unresolved_boundaries"),
            "receipt unresolved boundaries",
        ),
        "receipt unresolved-boundary preservation",
    )
    require(
        str(preserved.get("exact_claim_boundary")).strip()
        == str(receipt.get("claim_boundary")).strip(),
        "receipt claim-boundary preservation",
    )
    require(
        preserved.get("natural_person_identity_established") is False,
        "natural-person identity boundary",
    )
    require(
        preserved.get("historical_openalex_visibility_established") is False,
        "historical visibility boundary",
    )
    require(
        preserved.get("current_snapshot_proxy_only") is True,
        "current-snapshot proxy boundary",
    )
    return {
        "unresolved_boundary_count": len(receipt["unresolved_boundaries"]),
        "claim_boundary_preserved_exactly": True,
        "natural_person_identity_established": False,
        "historical_openalex_visibility_established": False,
        "current_snapshot_proxy_only": True,
    }


def _validate_future_preservation(
    contract: Mapping[str, Any],
    *,
    skeleton: Mapping[str, Any],
) -> dict[str, Any]:
    future = _mapping(
        contract.get("future_g0_v2_preservation_contract"),
        "future G0 v2 preservation contract",
    )
    require(
        future.get("predecessor_relative_path")
        == EXPECTED_BINDINGS["immutable_g0_v1_skeleton"]["path"],
        "future predecessor path",
    )
    require(
        future.get("predecessor_sha256")
        == EXPECTED_BINDINGS["immutable_g0_v1_skeleton"]["sha256"],
        "future predecessor hash",
    )
    require(
        tuple(future.get("must_preserve_from_g0_v1", []))
        == EXPECTED_PRESERVED_SUBTREES,
        "future preserved subtree membership",
    )
    for subtree in EXPECTED_PRESERVED_SUBTREES:
        require(subtree in skeleton, f"missing predecessor subtree {subtree}")
    require(
        tuple(future.get("only_permitted_semantic_delta", []))
        == EXPECTED_DELTAS,
        "future permitted semantic delta",
    )
    expected_values = {
        "must_keep_holdout_binding_pending_and_null": True,
        "must_keep_unresolved_quantitative_choice_count": 17,
        "must_keep_not_freeze_ready_section_count": 15,
        "must_keep_pass_g0_false": True,
        "must_keep_model_fitting_unauthorized": True,
        "must_keep_g4_unopened": True,
        "future_successor_not_authored_by_this_contract": True,
        "future_successor_requires_separate_authorization_gate": True,
    }
    for key, expected in expected_values.items():
        require(future.get(key) == expected, f"future preservation {key}")
    require(
        _lookup(skeleton, "holdout_access_binding.state") == "PENDING",
        "predecessor holdout state",
    )
    require(
        _lookup(skeleton, "holdout_access_binding.receipt_sha256") is None,
        "predecessor holdout hash",
    )
    require(
        _lookup(skeleton, "resolution.unresolved_quantitative_choice_count")
        == 17,
        "predecessor unresolved choice count",
    )
    require(
        _lookup(skeleton, "resolution.unresolved_required_section_count") == 15,
        "predecessor unresolved section count",
    )
    require(_lookup(skeleton, "lifecycle.pass_g0") is False, "predecessor G0")
    return {
        "preserved_subtree_count": len(EXPECTED_PRESERVED_SUBTREES),
        "permitted_semantic_delta_count": len(EXPECTED_DELTAS),
        "holdout_binding_remains_pending": True,
        "unresolved_quantitative_choice_count": 17,
        "not_freeze_ready_section_count": 15,
        "future_successor_authored": False,
    }


def _validate_fail_closed_boundary(contract: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(contract.get("lifecycle"), "lifecycle")
    expected_lifecycle = {
        "g0_v1_mutated": False,
        "g0_v2_authored_or_modified": False,
        "g_minus_1a_receipt_reissued_or_modified": False,
        "legacy_g_minus_1_promoted": False,
        "holdout_binding_changed": False,
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
    access = _mapping(
        contract.get("access_accounting_contract"),
        "access accounting",
    )
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
    authorization = _mapping(contract.get("authorization"), "authorization")
    require(
        authorization.get("contract_validation") == "AUTHORIZED_METADATA_ONLY",
        "contract validation authorization",
    )
    require(
        authorization.get("future_g0_v2_binding_only_successor")
        == "REQUIRES_SUCCESSOR_GATE",
        "future successor gate",
    )
    for key in (
        "g0_v1_mutation",
        "actual_g0_v2_authoring_or_modification",
        "g_minus_1a_receipt_reissue_or_modification",
        "legacy_g_minus_1_promotion",
        "holdout_binding_change",
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
    return {
        "access_accounting": expected_access,
        "g0_v1_mutated": False,
        "g0_v2_authored_or_modified": False,
        "legacy_g_minus_1_promoted": False,
        "pass_g0": False,
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def run(contract_path: Path) -> dict[str, Any]:
    contract = _read_yaml(contract_path, "G-1A binding reconciliation contract")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")
    require(
        contract.get("mode")
        == (
            "VERSIONED_METADATA_ONLY_FUTURE_G0_V2_BINDING_CONTRACT_"
            "NO_RELABELED_LEGACY_PASS"
        ),
        "contract mode",
    )

    documents: dict[str, Mapping[str, Any]] = {}
    bindings: dict[str, Any] = {}
    for field, expected_key in (
        ("authorization_parent", "authorization_parent"),
        ("immutable_g0_v1_skeleton", "immutable_g0_v1_skeleton"),
        ("validated_closure_plan", "validated_closure_plan"),
        ("gate_sequence", "gate_sequence"),
        ("legacy_g_minus_1_registry", "legacy_g_minus_1_registry"),
    ):
        documents[field], bindings[field] = _validate_document_binding(
            contract.get(field),
            label=field.replace("_", " "),
            expected_key=expected_key,
        )
    receipt_specs = _mapping(
        contract.get("exact_g_minus_1a_receipt"),
        "exact G-1A receipt",
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

    require(
        _lookup(documents["legacy_g_minus_1_registry"], "gate_state.pass_g_minus_1")
        is False,
        "legacy G-1 must remain not passed",
    )
    require(
        _lookup(
            documents["gate_sequence"],
            "separation_rules.final_cutoff_panel_owner",
        )
        == "G0_5",
        "final cutoff panel owner",
    )
    receipt = documents["g_minus_1a_summary"]
    require(receipt.get("status") == G_MINUS_1A_STATUS, "G-1A exact status")
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

    diagnosis_replay = _validate_source_diagnosis(
        contract,
        skeleton=documents["immutable_g0_v1_skeleton"],
        receipt=receipt,
    )
    future_binding_replay = _validate_future_binding(
        contract,
        receipt=receipt,
    )
    boundary_preservation_replay = _validate_preserved_boundaries(
        contract,
        receipt=receipt,
    )
    future_preservation_replay = _validate_future_preservation(
        contract,
        skeleton=documents["immutable_g0_v1_skeleton"],
    )
    fail_closed_replay = _validate_fail_closed_boundary(contract)

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "contract_sha256": _sha256(contract_path),
        "repository_bindings": bindings,
        "diagnosis_replay": diagnosis_replay,
        "future_binding_replay": future_binding_replay,
        "receipt_boundary_preservation_replay": boundary_preservation_replay,
        "future_g0_v2_preservation_replay": future_preservation_replay,
        "fail_closed_boundary_replay": fail_closed_replay,
        "authorization": {
            "binding_reconciliation_contract": (
                "COMPLETE_INDEPENDENTLY_VALIDATED"
            ),
            "future_g0_v2_binding_only_successor": "REQUIRES_SUCCESSOR_GATE",
            "actual_g0_v2_authoring_or_modification": "NOT_AUTHORIZED",
            "legacy_g_minus_1_promotion": "NOT_AUTHORIZED",
            "holdout_binding_change": "NOT_AUTHORIZED",
            "quantitative_choice_resolution": "NOT_AUTHORIZED",
            "required_section_completion": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": (
            "The metadata-only G-1A binding-reconciliation contract is "
            "independently valid. It defines a distinct future "
            "g_minus_1a_binding over the exact existing receipt while "
            "forbidding schema aliasing, status relabeling, or a legacy G-1 "
            "pass inference. It preserves all receipt caveats and leaves G0 "
            "v1, G0 v2, holdout state, seventeen unresolved values, and "
            "fifteen not-freeze-ready sections unchanged. No operator row "
            "data, confirmatory or holdout data, API request, G0 seal, model, "
            "G4 result, causality, or scientific effect is established."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    args = parser.parse_args()
    result = run(args.contract.resolve(strict=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
