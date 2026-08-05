#!/usr/bin/env python3
"""Build the additive atomic F-closure milestone authority from exact receipts."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[1]
PARENT_PATH = ROOT / "experiments/configs/pivot_feedback_current_state_v1.yaml"
DEFAULT_OUTPUT = ROOT / "experiments/configs/pivot_feedback_current_state_v2.yaml"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_current_state_v2.py"
PARENT_SHA256 = "52c262455b87a1e89fd28b370408fc9c697b26c7ca9e2cb15de59fd27d3ac0bd"
PARENT_SCHEMA = "pivot_feedback_current_state.v1"
PARENT_STATUS = (
    "ACTIVE_F_PHASE_PRE_G1_REGISTRY_BOUND_THIRTEEN_SECTIONS_PANEL_AND_SEAL_PENDING"
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


class CurrentStateV2BuildError(RuntimeError):
    """Fail-closed F-closure authority build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentStateV2BuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), label)


def _yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)


def _validate_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise CurrentStateV2BuildError("invalid closure timestamp") from error
    require(parsed.tzinfo is not None, "closure timestamp timezone")
    require(parsed.utcoffset() == timezone.utc.utcoffset(parsed), "closure timestamp UTC")
    return parsed.isoformat().replace("+00:00", "Z")


def _load_receipts() -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for name, binding in RECEIPTS.items():
        path = ROOT / binding["relative_path"]
        require(_sha256(path) == binding["sha256"], f"{name} hash")
        document = _json(path, name)
        require(document.get("schema_version") == binding["schema_version"], f"{name} schema")
        require(document.get("status") == binding["status"], f"{name} status")
        documents[name] = document
    return documents


def _validate_receipt_conjunction(documents: Mapping[str, Mapping[str, Any]]) -> None:
    registry = documents["pre_g1_decision_registry"]
    require(registry["decision_registry_replay"]["pre_g1_registry_complete"] is True, "registry completion")
    require(registry["fail_closed_replay"]["pass_f"] is False, "registry premature F pass")

    sections = documents["thirteen_pre_g1_required_sections"]
    inventory = sections["section_inventory_replay"]
    require(inventory["required_section_count"] == 13, "section requirement count")
    require(inventory["frozen_section_count"] == 13, "frozen section count")
    require(inventory["later_stage_sections_included"] == 0, "later section boundary")
    require(inventory["outcome_blind"] is True, "section outcome blindness")
    require(sections["authorization"]["section_freeze"] == "COMPLETE_INDEPENDENTLY_VERIFIED", "section verification")
    require(sections["access_accounting"]["confirmatory_or_holdout_rows_read"] == 0, "section row reads")

    panel = documents["outcome_blind_final_panel_conformance"]
    require(panel["fail_closed_state"]["final_panel_conformance_complete"] is True, "panel completion")
    require(panel["independent_verification"]["episode_rows_verified"] == 65, "panel episode rows")
    require(panel["independent_verification"]["raw_id_columns_absent"] is True, "panel raw IDs")
    require(panel["fail_closed_state"]["confirmatory_entry_or_persistence_labels_read"] == 0, "panel label reads")

    partition = documents["atomic_partition_assignment"]
    require(partition["fail_closed_state"]["atomic_episode_family_partition_assignment_complete"] is True, "partition completion")
    g4 = next(row for row in partition["partition_assignment"] if row["partition_id"] == "g4_temporal_holdout")
    require(g4["assigned_episode_count"] == 37 and g4["assigned_family_count"] == 16, "G4 assignment")
    require(g4["sealed"] is True and g4["author_cold"] is True, "G4 partition boundary")
    require(partition["fail_closed_state"]["confirmatory_entry_or_persistence_labels_read"] == 0, "partition label reads")

    seal = documents["holdout_byte_seal"]
    transition = seal["f_phase_transition_state"]
    require(transition["all_four_transition_receipts_available_for_atomic_reconciliation"] is True, "seal transition readiness")
    require(transition["f_closure_reconciliation_complete"] is False, "seal closure predecessor")
    sealed_state = seal["fail_closed_state"]
    require(sealed_state["holdout_byte_seal_complete"] is True, "byte seal completion")
    require(sealed_state["holdout_seal_independently_verified"] is True, "byte seal verification")
    require(sealed_state["seal_opened"] is False, "seal opening boundary")
    require(sealed_state["confirmatory_or_holdout_analytic_rows_read"] == 0, "seal row reads")


def build_document(*, closed_at_utc: str) -> dict[str, Any]:
    closed_at_utc = _validate_timestamp(closed_at_utc)
    require(_sha256(PARENT_PATH) == PARENT_SHA256, "parent hash")
    parent = _yaml(PARENT_PATH, "parent authority")
    require(parent.get("schema_version") == PARENT_SCHEMA, "parent schema")
    require(parent.get("status") == PARENT_STATUS, "parent status")
    require(parent["authorization"]["f_phase_closure"] == "AUTHORIZED_EXACT_SCOPE", "F closure authorization")
    documents = _load_receipts()
    _validate_receipt_conjunction(documents)
    require(VERIFIER_PATH.is_file() and not VERIFIER_PATH.is_symlink(), "missing independent verifier")

    output = deepcopy(parent)
    output["schema_version"] = "pivot_feedback_current_state.v2"
    output["label"] = "pivot_feedback_current_state_v2"
    output["mode"] = "ACTIVE_SIX_PHASE_ATOMIC_F_CLOSURE_E_ENTRY_FAIL_CLOSED"
    output["status"] = (
        "ACTIVE_E_PHASE_F_ATOMICALLY_CLOSED_G0_COMPATIBILITY_PASS_"
        "HOLDOUT_SEALED_UNOPENED"
    )
    output["additive_parent"] = {
        "relative_path": PARENT_PATH.relative_to(ROOT).as_posix(),
        "sha256": PARENT_SHA256,
        "required_schema_version": PARENT_SCHEMA,
        "required_status": PARENT_STATUS,
        "mutation_allowed": False,
    }
    output["closure_tooling"] = {
        "builder": {
            "relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "independent_verifier": {
            "relative_path": VERIFIER_PATH.relative_to(ROOT).as_posix(),
            "sha256": _sha256(VERIFIER_PATH),
        },
    }
    output["f_phase_transition_receipt_bindings"] = {
        name: {
            **binding,
            "mutation_allowed": False,
        }
        for name, binding in RECEIPTS.items()
    }

    routing = output["active_routing"]
    routing["active_phase"] = "E"
    phases = {phase["phase_id"]: phase for phase in routing["phases"]}
    phases["F"]["state"] = "COMPLETE_FREEZE_PANEL_CONFORMANCE_AND_BYTE_SEAL"
    phases["F"]["complete"] = True
    phases["E"]["state"] = "READY_NOT_RUN"
    phases["H"]["state"] = "BYTE_SEALED_UNOPENED"

    before_e_sections = output["staged_section_registry"]["before_e_confirmatory_access"]
    before_e_sections["state"] = "COMPLETE_OUTCOME_BLIND_HASH_BOUND"
    before_e_sections["freeze_ready_section_count"] = 13

    closure = output["f_phase_closure"]
    closure["state"] = "COMPLETE_ATOMIC_FOUR_RECEIPT_CONJUNCTION"
    closure["completed_receipts"] = F_RECEIPT_IDS
    closure["pending_receipts"] = []
    closure["closed_at_utc"] = closed_at_utc
    closure["all_receipt_hashes_bound"] = True
    closure["closure_is_new_scientific_gate"] = False
    closure["confirmatory_or_holdout_rows_read"] = 0
    closure["pass_f_now"] = True
    closure["pass_g0_now"] = True

    lifecycle = output["lifecycle"]
    lifecycle["pre_g1_pending_section_count"] = 0
    lifecycle["final_panel_conformance_run"] = True
    lifecycle["atomic_partition_assignment_complete"] = True
    lifecycle["holdout_receipt_bound"] = True
    lifecycle["holdout_seal_independently_verified"] = True
    lifecycle["holdout_seal_opened"] = False
    lifecycle["f_phase_closed_at_utc"] = closed_at_utc
    lifecycle["pass_f"] = True
    lifecycle["pass_g0"] = True

    output["authorization"] = {
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
    output["validation_contract"] = {
        "verifier_relative_path": VERIFIER_PATH.relative_to(ROOT).as_posix(),
        "summary_json_relative_path": (
            "experiments/evidence/results/"
            "pivot_feedback_current_state_v2_20260805.summary.json"
        ),
        "summary_markdown_relative_path": (
            "experiments/evidence/results/"
            "pivot_feedback_current_state_v2_20260805.summary.md"
        ),
        "repository_metadata_only": True,
        "operator_or_row_level_files_may_be_read": False,
        "confirmatory_or_holdout_rows_may_be_read": False,
        "seal_may_be_opened": False,
        "model_may_be_fitted": False,
    }
    output["claim_boundary"] = (
        "The additive authority closes F and its legacy G0/G0.5 compatibility state "
        "only from the exact conjunction of all four frozen transition receipts. It "
        "moves routing to E-ready but does not execute G1, read a confirmatory or "
        "holdout row, open the private seal, read a model-selection score, fit a "
        "model, open G4, establish causality, or establish a scientific effect."
    )
    return output


def _write_yaml_create_only(path: Path, value: Mapping[str, Any]) -> None:
    require(path.parent.is_dir() and not path.parent.is_symlink(), "invalid output parent")
    require(not path.exists() and not path.is_symlink(), "output exists")
    payload = yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True)
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
            require(written > 0, "short output write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--closed-at-utc", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_document(closed_at_utc=args.closed_at_utc or _utc_now())
    output = args.output.resolve()
    require(output == DEFAULT_OUTPUT.resolve(), "unexpected production output path")
    _write_yaml_create_only(output, document)
    print(
        json.dumps(
            {
                "schema_version": document["schema_version"],
                "status": document["status"],
                "active_phase": document["active_routing"]["active_phase"],
                "completed_receipt_count": len(
                    document["f_phase_closure"]["completed_receipts"]
                ),
                "pending_receipt_count": len(
                    document["f_phase_closure"]["pending_receipts"]
                ),
                "pass_f": document["lifecycle"]["pass_f"],
                "pass_g0": document["lifecycle"]["pass_g0"],
                "confirmatory_or_holdout_rows_read": document["lifecycle"][
                    "confirmatory_or_holdout_rows_read"
                ],
                "model_fitted": document["lifecycle"]["model_fitted"],
                "g4_open": document["lifecycle"]["g4_open"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
