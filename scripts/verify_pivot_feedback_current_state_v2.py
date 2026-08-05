#!/usr/bin/env python3
"""Independently verify the additive atomic F-closure milestone authority."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v2.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_current_state_v2_20260805.summary.json"
)
PARENT_PATH = ROOT / "experiments/configs/pivot_feedback_current_state_v1.yaml"
BUILDER_PATH = ROOT / "scripts/build_pivot_feedback_current_state_v2.py"
PARENT_SHA256 = "52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd"
PARENT_SCHEMA = "pivot_feedback_current_state.v1"
PARENT_STATUS = (
    "ACTIVE_F_PHASE_PRE_G1_REGISTRY_BOUND_THIRTEEN_SECTIONS_PANEL_AND_SEAL_PENDING"
)
AUTHORITY_SCHEMA = "pivot_feedback_current_state.v2"
AUTHORITY_STATUS = (
    "ACTIVE_E_PHASE_F_ATOMICALLY_CLOSED_G0_COMPATIBILITY_PASS_"
    "HOLDOUT_SEALED_UNOPENED"
)
SUMMARY_SCHEMA = "pivot_feedback_current_state_v2_summary.v1"
SUMMARY_STATUS = (
    "PASS_ATOMIC_F_CLOSURE_G0_COMPATIBILITY_PASS_E_ENTRY_READY_FAIL_CLOSED"
)

RECEIPTS = {
    "pre_g1_decision_registry": {
        "relative_path": (
            "experiments/evidence/results/"
            "pivot_feedback_current_state_v1_20260803.summary.json"
        ),
        "sha256": "e58cd56086bfc919d76241c194d1b9d96051b969c5c00ced0e73f925757d32ce",
        "schema_version": "pivot_feedback_current_state_summary.v1",
        "status": "PASS_ACTIVE_SIX_PHASE_AUTHORITY_F_IN_PROGRESS_FAIL_CLOSED",
    },
    "thirteen_pre_g1_required_sections": {
        "relative_path": (
            "experiments/evidence/results/"
            "pivot_feedback_f_pre_g1_scientific_sections_v1_20260803.summary.json"
        ),
        "sha256": "c0c7da0d7eb26a510b02a20fd4c85b371cc05aad0603d893dc9d7e45c1a8b2ee",
        "schema_version": "pivot_feedback_f_pre_g1_scientific_sections_summary.v1",
        "status": "PASS_THIRTEEN_PRE_G1_SCIENTIFIC_SECTIONS_FROZEN_NO_F_PASS",
    },
    "outcome_blind_final_panel_conformance": {
        "relative_path": (
            "experiments/evidence/results/"
            "pivot_feedback_f_aligned_final_panel_conformance_v1_20260804.summary.json"
        ),
        "sha256": "e18f0e954e109baf953b161eb1321d8be4a584ee6f4fc383989b8aa3fb2bc35e",
        "schema_version": "pivot_feedback_f_aligned_final_panel_conformance_summary.v1",
        "status": (
            "PASS_OUTCOME_BLIND_ALIGNED_FINAL_PANEL_LEAVEOUT_AND_SUPPORT_"
            "CONFORMANCE_ATOMIC_PARTITION_AUTHORIZED_NO_F_PASS"
        ),
    },
    "atomic_partition_assignment": {
        "relative_path": (
            "experiments/evidence/results/"
            "pivot_feedback_f_atomic_partition_assignment_v1_20260804.summary.json"
        ),
        "sha256": "a1c57c541dd7f7a6042e524d6340da2bcd1246eb48021958bb60ddd72e313b0d",
        "schema_version": "pivot_feedback_f_atomic_partition_assignment_summary.v1",
        "status": (
            "PASS_OUTCOME_BLIND_ATOMIC_EPISODE_FAMILY_PARTITION_ASSIGNMENT_"
            "PRIVATE_HOLDOUT_MATERIALIZATION_AUTHORIZED_NO_F_PASS"
        ),
    },
    "holdout_byte_seal": {
        "relative_path": (
            "experiments/evidence/results/"
            "pivot_feedback_holdout_access_seal_receipt_verification_v1_"
            "20260805.summary.json"
        ),
        "sha256": "b881c113b9c6f07b7a9e143beecbaa693b94ddb5e82a032186bbffa7ff3c239d",
        "schema_version": (
            "pivot_feedback_holdout_access_seal_receipt_verification_summary.v1"
        ),
        "status": (
            "PASS_PRODUCTION_PRIVATE_G4_BYTE_SEAL_RECEIPT_INDEPENDENTLY_"
            "VERIFIED_UNOPENED_NO_F_PASS"
        ),
    },
}

F_RECEIPT_IDS = [
    "PRE_G1_DECISION_REGISTRY_COMPLETE",
    "THIRTEEN_PRE_G1_REQUIRED_SECTIONS_COMPLETE",
    "OUTCOME_BLIND_FINAL_PANEL_CONFORMANCE",
    "ATOMIC_PARTITION_AND_HOLDOUT_BYTE_SEAL",
]


class CurrentStateV2VerificationError(RuntimeError):
    """Fail-closed atomic F-closure verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentStateV2VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _document(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, label)


def _load_receipts(authority: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    bindings = _mapping(
        authority.get("f_phase_transition_receipt_bindings"),
        "transition receipt bindings",
    )
    require(set(bindings) == set(RECEIPTS), "transition receipt binding set")
    documents: dict[str, dict[str, Any]] = {}
    for name, expected in RECEIPTS.items():
        binding = _mapping(bindings.get(name), name)
        require(
            binding
            == {
                **expected,
                "mutation_allowed": False,
            },
            f"{name} binding",
        )
        path = ROOT / expected["relative_path"]
        require(_sha256(path) == expected["sha256"], f"{name} hash")
        document = _document(path, name)
        require(document.get("schema_version") == expected["schema_version"], f"{name} schema")
        require(document.get("status") == expected["status"], f"{name} status")
        documents[name] = document
    return documents


def _validate_receipt_semantics(documents: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    registry = documents["pre_g1_decision_registry"]
    require(registry["decision_registry_replay"]["pre_g1_registry_complete"] is True, "registry completion")
    require(registry["decision_registry_replay"]["pre_g1_choice_count"] == 13, "registry choices")

    sections = documents["thirteen_pre_g1_required_sections"]
    inventory = sections["section_inventory_replay"]
    require(inventory["required_section_count"] == 13, "required sections")
    require(inventory["frozen_section_count"] == 13, "frozen sections")
    require(inventory["later_stage_sections_included"] == 0, "later sections")
    require(inventory["outcome_blind"] is True, "section outcome blindness")

    panel = documents["outcome_blind_final_panel_conformance"]
    require(panel["fail_closed_state"]["final_panel_conformance_complete"] is True, "panel completion")
    require(panel["independent_verification"]["episode_rows_verified"] == 65, "panel episodes")
    require(panel["independent_verification"]["raw_id_columns_absent"] is True, "panel raw IDs")

    partition = documents["atomic_partition_assignment"]
    require(partition["fail_closed_state"]["atomic_episode_family_partition_assignment_complete"] is True, "partition completion")
    partition_ids = [row["partition_id"] for row in partition["partition_assignment"]]
    require(partition_ids == ["discovery_calibration", "g1_empirical_replication", "model_selection", "g4_temporal_holdout"], "partition order")
    g4 = partition["partition_assignment"][-1]
    require(g4["sealed"] is True and g4["author_cold"] is True, "G4 assignment")

    seal = documents["holdout_byte_seal"]
    transition = seal["f_phase_transition_state"]
    require(transition["all_four_transition_receipts_available_for_atomic_reconciliation"] is True, "transition conjunction")
    require(transition["f_closure_reconciliation_complete"] is False, "pre-closure seal state")
    fail_closed = seal["fail_closed_state"]
    require(fail_closed["holdout_seal_independently_verified"] is True, "seal verification")
    require(fail_closed["seal_opened"] is False, "seal opening")

    row_read_values = [
        registry["fail_closed_replay"]["confirmatory_or_holdout_rows_read"],
        sections["access_accounting"]["confirmatory_or_holdout_rows_read"],
        panel["fail_closed_state"]["confirmatory_entry_or_persistence_labels_read"],
        partition["fail_closed_state"]["confirmatory_entry_or_persistence_labels_read"],
        fail_closed["confirmatory_or_holdout_analytic_rows_read"],
    ]
    require(row_read_values == [0, 0, 0, 0, 0], "receipt row-read boundary")
    return {
        "receipt_count": 4,
        "receipt_ids": F_RECEIPT_IDS,
        "all_hashes_exact": True,
        "all_statuses_pass": True,
        "pre_g1_choice_count": 13,
        "pre_g1_section_count": 13,
        "final_panel_episode_count": 65,
        "atomic_partition_count": 4,
        "holdout_seal_independently_verified": True,
        "seal_opened": False,
        "confirmatory_or_holdout_rows_read": 0,
    }


def _validate_parent_and_tooling(authority: Mapping[str, Any]) -> dict[str, Any]:
    require(_sha256(PARENT_PATH) == PARENT_SHA256, "parent hash")
    parent = _document(PARENT_PATH, "parent authority")
    require(parent.get("schema_version") == PARENT_SCHEMA, "parent schema")
    require(parent.get("status") == PARENT_STATUS, "parent status")
    additive = _mapping(authority.get("additive_parent"), "additive parent")
    require(
        additive
        == {
            "relative_path": PARENT_PATH.relative_to(ROOT).as_posix(),
            "sha256": PARENT_SHA256,
            "required_schema_version": PARENT_SCHEMA,
            "required_status": PARENT_STATUS,
            "mutation_allowed": False,
        },
        "additive parent binding",
    )
    tooling = _mapping(authority.get("closure_tooling"), "closure tooling")
    expected_tooling = {
        "builder": {
            "relative_path": BUILDER_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256(BUILDER_PATH),
        },
        "independent_verifier": {
            "relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    require(tooling == expected_tooling, "closure tooling binding")
    return parent


def _validate_preserved_and_routing(
    authority: Mapping[str, Any], parent: Mapping[str, Any]
) -> dict[str, Any]:
    for key in (
        "authority_transition",
        "bound_freeze_inputs",
        "staged_decision_registry",
        "irreversible_boundaries",
    ):
        require(authority.get(key) == parent.get(key), f"preserved subtree {key}")

    routing = _mapping(authority.get("active_routing"), "active routing")
    parent_routing = _mapping(parent.get("active_routing"), "parent routing")
    require(routing.get("phase_order") == ["D", "C", "F", "E", "M", "H"], "phase order")
    require(routing.get("active_phase") == "E", "active phase")
    phases = routing.get("phases")
    parent_phases = parent_routing.get("phases")
    require(isinstance(phases, list) and isinstance(parent_phases, list), "phase lists")
    require(len(phases) == len(parent_phases) == 6, "phase count")
    for phase, predecessor in zip(phases, parent_phases, strict=True):
        require(phase["phase_id"] == predecessor["phase_id"], "phase identity")
        for key, value in predecessor.items():
            if key not in {"state", "complete"}:
                require(phase.get(key) == value, f"phase {phase['phase_id']} {key}")
    by_id = {phase["phase_id"]: phase for phase in phases}
    require(all(by_id[item]["complete"] is True for item in ("D", "C", "F")), "completed phases")
    require(all(by_id[item]["complete"] is False for item in ("E", "M", "H")), "future phases")
    require(by_id["F"]["state"] == "COMPLETE_FREEZE_PANEL_CONFORMANCE_AND_BYTE_SEAL", "F state")
    require(by_id["E"]["state"] == "READY_NOT_RUN", "E state")
    require(by_id["H"]["state"] == "BYTE_SEALED_UNOPENED", "H state")

    sections = deepcopy(parent["staged_section_registry"])
    before_e = sections["before_e_confirmatory_access"]
    before_e["state"] = "COMPLETE_OUTCOME_BLIND_HASH_BOUND"
    before_e["freeze_ready_section_count"] = 13
    require(authority.get("staged_section_registry") == sections, "section registry transition")
    return {
        "phase_order": ["D", "C", "F", "E", "M", "H"],
        "active_phase": "E",
        "complete_phases": ["D", "C", "F"],
        "incomplete_phases": ["E", "M", "H"],
        "F_state": "COMPLETE_FREEZE_PANEL_CONFORMANCE_AND_BYTE_SEAL",
        "E_state": "READY_NOT_RUN",
        "H_state": "BYTE_SEALED_UNOPENED",
    }


def _validate_closure_and_fail_closed(authority: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    closure = _mapping(authority.get("f_phase_closure"), "F closure")
    require(closure.get("state") == "COMPLETE_ATOMIC_FOUR_RECEIPT_CONJUNCTION", "closure state")
    require(closure.get("completed_receipts") == F_RECEIPT_IDS, "completed receipts")
    require(closure.get("pending_receipts") == [], "pending receipts")
    require(closure.get("completion_is_one_atomic_milestone") is True, "atomic milestone")
    require(closure.get("intermediate_receipts_may_record_provenance_but_are_not_new_gates") is True, "micro-gate boundary")
    require(closure.get("all_receipt_hashes_bound") is True, "receipt hash binding")
    require(closure.get("closure_is_new_scientific_gate") is False, "new-gate boundary")
    require(closure.get("confirmatory_or_holdout_rows_read") == 0, "closure row reads")
    require(closure.get("pass_f_now") is True, "F pass")
    require(closure.get("pass_g0_now") is True, "G0 compatibility pass")

    lifecycle = _mapping(authority.get("lifecycle"), "lifecycle")
    expected = {
        "pre_g1_pending_choice_count": 0,
        "pre_g1_pending_section_count": 0,
        "final_panel_conformance_run": True,
        "atomic_partition_assignment_complete": True,
        "holdout_receipt_bound": True,
        "holdout_seal_independently_verified": True,
        "holdout_seal_opened": False,
        "pass_f": True,
        "pass_g0": True,
        "confirmatory_or_holdout_rows_read": 0,
        "empirical_g1_status": "NOT_RUN",
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    for key, value in expected.items():
        require(lifecycle.get(key) == value, f"lifecycle {key}")

    authorization = _mapping(authority.get("authorization"), "authorization")
    expected_authorization = {
        "six_phase_active_routing": "ACTIVE",
        "f_phase_closure": "COMPLETE_ATOMIC_FOUR_RECEIPT_CONJUNCTION",
        "e_g1_execution": (
            "AUTHORIZED_NEXT_EXACT_REGISTERED_SCOPE_REQUIRES_SEPARATE_EXECUTION_ARTIFACT"
        ),
        "confirmatory_or_holdout_analytic_access": "NOT_AUTHORIZED_BY_THIS_CLOSURE_ARTIFACT",
        "confirmatory_analytic_access": "AUTHORIZED_ONLY_IN_SEPARATE_E_EXECUTION_SUCCESSOR",
        "holdout_analytic_access": "NOT_AUTHORIZED_BEFORE_M_RESOLVED_AND_SCORING_CONTRACT_LOCKED",
        "g2_or_g3_choice_resolution": "NOT_AUTHORIZED_UNTIL_REGISTERED_ENTRY_BOUNDARY",
        "outcome_based_threshold_selection": "NOT_AUTHORIZED",
        "pass_f_without_all_receipts": "NOT_AUTHORIZED",
        "pass_g0_without_all_receipts": "NOT_AUTHORIZED",
        "model_fitting": "NOT_AUTHORIZED",
        "g4_open": "NOT_AUTHORIZED",
        "causal_claim": "NOT_AUTHORIZED",
        "scientific_effect": "NOT_ESTABLISHED",
    }
    require(authorization == expected_authorization, "authorization boundary")
    return (
        {
            "completed_receipt_count": 4,
            "pending_receipt_count": 0,
            "one_atomic_completion_milestone": True,
            "micro_gate_added": False,
            "pass_f": True,
            "pass_g0_compatibility": True,
        },
        {
            "confirmatory_or_holdout_rows_read": 0,
            "empirical_g1_status": "NOT_RUN",
            "seal_opened": False,
            "model_score_read": False,
            "model_fitted": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
    )


def verify(path: Path = DEFAULT_AUTHORITY) -> dict[str, Any]:
    authority = _document(path, "current state v2")
    require(authority.get("schema_version") == AUTHORITY_SCHEMA, "authority schema")
    require(authority.get("mode") == "ACTIVE_SIX_PHASE_ATOMIC_F_CLOSURE_E_ENTRY_FAIL_CLOSED", "authority mode")
    require(authority.get("status") == AUTHORITY_STATUS, "authority status")
    parent = _validate_parent_and_tooling(authority)
    documents = _load_receipts(authority)
    receipt_replay = _validate_receipt_semantics(documents)
    routing = _validate_preserved_and_routing(authority, parent)
    closure, fail_closed = _validate_closure_and_fail_closed(authority)
    authority_path = path.resolve()
    return {
        "schema_version": SUMMARY_SCHEMA,
        "status": SUMMARY_STATUS,
        "authority_transition_replay": {
            "predecessor_schema_version": PARENT_SCHEMA,
            "predecessor_sha256": PARENT_SHA256,
            "predecessor_immutable": True,
            "changed_scientific_stage": "F_TO_E_READY",
            "legacy_G0_G0_5_compatibility_pass": True,
        },
        "f_transition_receipt_replay": receipt_replay,
        "active_routing_replay": routing,
        "f_closure_replay": closure,
        "fail_closed_replay": fail_closed,
        "access_accounting": {
            "repository_yaml_documents_read": 2,
            "repository_json_documents_read": 5,
            "operator_or_row_level_files_read": 0,
            "discovery_rows_read": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "seal_open_events": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "repository_artifacts": {
            "active_authority": {
                "relative_path": (
                    authority_path.relative_to(ROOT).as_posix()
                    if authority_path.is_relative_to(ROOT)
                    else authority_path.name
                ),
                "sha256": _sha256(authority_path),
            },
            "builder": {
                "relative_path": BUILDER_PATH.relative_to(ROOT).as_posix(),
                "sha256": _sha256(BUILDER_PATH),
            },
            "independent_verifier": {
                "relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "authorization": authority["authorization"],
        "claim_boundary": (
            "All four exact F transition receipts independently close the freeze "
            "milestone and its legacy G0/G0.5 compatibility state. Routing is E-ready, "
            "but G1 has not run and this closure reads no confirmatory or holdout row, "
            "opens no seal, reads no model score, fits no model, opens no G4 holdout, "
            "and establishes no causality or scientific effect."
        ),
    }


def _write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "invalid summary parent")
    require(not path.exists() and not path.is_symlink(), "summary output exists")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        encoded = payload.encode("utf-8")
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            require(written > 0, "short summary write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(args.authority)
    if args.output is not None:
        output = args.output.resolve()
        require(output == DEFAULT_OUTPUT.resolve(), "unexpected summary output path")
        _write_json_create_only(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
