#!/usr/bin/env python3
"""Independently verify the additive terminal-BLOCKED E/G1 authority."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v3.yaml"
PARENT = ROOT / "experiments/configs/pivot_feedback_current_state_v2.yaml"
EXECUTION_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_e_g1_execution_v1_20260805.summary.json"
)
BUILDER = ROOT / "scripts/build_pivot_feedback_current_state_v3.py"
VERIFIER = Path(__file__).resolve()
PARENT_SHA = "ca431afb63da4cf2e09d0563c0509e3685efac29dc8679df0f9cc7f1ae7851ab"
EXECUTION_SUMMARY_SHA = "75855d8bc688c1c5001f2c19c25cd6c8a99c62355beb975e6cf1c3df425e6e5c"
AUTHORITY_STATUS = "ACTIVE_E_PHASE_BLOCKED_INPUT_AUTHORIZATION_CONFLICT_G1_NOT_RUN_HOLDOUT_SEALED_UNOPENED"
SUMMARY_STATUS = "PASS_ACTIVE_E_TERMINAL_BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ"


class CurrentStateV3VerificationError(RuntimeError):
    """Raised when the E/G1 terminal transition does not replay exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentStateV3VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def verify(authority_path: Path) -> dict[str, Any]:
    authority_path = authority_path.resolve(strict=True)
    require(_sha256(PARENT) == PARENT_SHA, "parent hash drift")
    require(_sha256(EXECUTION_SUMMARY) == EXECUTION_SUMMARY_SHA, "execution summary hash drift")
    parent = yaml.safe_load(PARENT.read_text(encoding="utf-8"))
    authority = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    result = json.loads(EXECUTION_SUMMARY.read_text(encoding="utf-8"))
    require(parent.get("schema_version") == "pivot_feedback_current_state.v2", "parent schema drift")
    require(parent.get("status") == "ACTIVE_E_PHASE_F_ATOMICALLY_CLOSED_G0_COMPATIBILITY_PASS_HOLDOUT_SEALED_UNOPENED", "parent status drift")
    require(result.get("schema_version") == "pivot_feedback_e_g1_execution_summary.v1", "execution summary schema drift")
    require(result.get("status") == "PASS_INDEPENDENT_VERIFICATION_OF_TERMINAL_BLOCKED_E_G1_NO_CONFIRMATORY_READ", "execution summary status drift")
    require(authority.get("schema_version") == "pivot_feedback_current_state.v3", "authority schema drift")
    require(authority.get("status") == AUTHORITY_STATUS, "authority status drift")
    require(authority.get("mode") == "ACTIVE_SIX_PHASE_E_G1_TERMINAL_BLOCKED_FAIL_CLOSED_NO_CONFIRMATORY_READ", "authority mode drift")

    additive = authority["additive_parent"]
    require(additive == {
        "relative_path": "experiments/configs/pivot_feedback_current_state_v2.yaml",
        "sha256": PARENT_SHA,
        "required_schema_version": "pivot_feedback_current_state.v2",
        "required_status": "ACTIVE_E_PHASE_F_ATOMICALLY_CLOSED_G0_COMPATIBILITY_PASS_HOLDOUT_SEALED_UNOPENED",
        "mutation_allowed": False,
    }, "additive parent binding drift")

    preserved = set(parent) - {
        "schema_version", "label", "mode", "status", "active_routing", "lifecycle",
        "authorization", "additive_parent", "validation_contract", "claim_boundary",
    }
    for key in preserved:
        require(authority.get(key) == parent.get(key), f"preserved subtree drift: {key}")

    expected_routing = deepcopy(parent["active_routing"])
    for phase in expected_routing["phases"]:
        if phase["phase_id"] == "E":
            phase["state"] = "BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ"
            phase["complete"] = False
            phase["may_read_confirmatory_or_holdout_rows"] = False
    expected_routing["active_phase"] = "E"
    require(authority["active_routing"] == expected_routing, "active routing transition drift")

    expected_lifecycle = deepcopy(parent["lifecycle"])
    expected_lifecycle.update({
        "confirmatory_or_holdout_rows_read": 0,
        "empirical_g1_status": "BLOCKED",
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    })
    require(authority["lifecycle"] == expected_lifecycle, "lifecycle transition drift")

    expected_authorization = deepcopy(parent["authorization"])
    expected_authorization.update({
        "e_g1_execution": "BLOCKED_REQUIRES_NEW_EXPLICIT_CONFLICT_RESOLUTION_AUTHORITY",
        "confirmatory_or_holdout_analytic_access": "NOT_AUTHORIZED_WHILE_E_INPUT_AUTHORIZATION_CONFLICT_PERSISTS",
        "confirmatory_analytic_access": "NOT_AUTHORIZED_UNTIL_NEW_EXPLICIT_CONFLICT_RESOLUTION_AUTHORITY",
        "g2_or_g3_choice_resolution": "NOT_AUTHORIZED_E_NOT_PASSED",
        "model_fitting": "NOT_AUTHORIZED",
        "g4_open": "NOT_AUTHORIZED",
        "causal_claim": "NOT_AUTHORIZED",
        "scientific_effect": "NOT_ESTABLISHED",
    })
    require(authority["authorization"] == expected_authorization, "authorization transition drift")

    attempt = authority["e_phase_execution_attempt"]
    require(attempt["state"] == "TERMINAL_BLOCKED_NO_ROUTE_PROMOTION", "attempt state drift")
    require(attempt["attempt_count"] == 1, "attempt count drift")
    require(attempt["terminal_status"] == "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION", "terminal status drift")
    require(attempt["block_id"] == "G1_AUTHOR_COLD_GUARD_REQUIRED_BUT_PRIVATE_G4_ACCESS_FORBIDDEN", "block id drift")
    require(attempt["execution_summary"] == {
        "relative_path": EXECUTION_SUMMARY.relative_to(ROOT).as_posix(),
        "sha256": EXECUTION_SUMMARY_SHA,
        "required_schema_version": "pivot_feedback_e_g1_execution_summary.v1",
        "required_status": "PASS_INDEPENDENT_VERIFICATION_OF_TERMINAL_BLOCKED_E_G1_NO_CONFIRMATORY_READ",
        "mutation_allowed": False,
    }, "execution summary binding drift")
    require(attempt["registered_scope"] == result["registered_scope_replay"], "registered scope replay drift")
    expected_zero = {
        "confirmatory_rows_read": 0,
        "holdout_rows_read": 0,
        "labels_materialized": 0,
        "estimators_run": 0,
        "private_g4_artifacts_read": 0,
    }
    for key, value in expected_zero.items():
        require(attempt[key] == value, f"attempt access drift: {key}")
    require(attempt["holdout_seal_opened"] is False and attempt["g4_open"] is False, "holdout opened")
    require(attempt["scope_or_threshold_changed"] is False, "scope or threshold changed")
    require(attempt["resolution_requires_new_explicit_authority"] is True, "resolution authority boundary drift")

    tooling = authority["state_tooling"]
    require(tooling["builder"] == {
        "relative_path": BUILDER.relative_to(ROOT).as_posix(),
        "sha256": _sha256(BUILDER),
    }, "builder binding drift")
    require(tooling["independent_verifier"] == {
        "relative_path": VERIFIER.relative_to(ROOT).as_posix(),
        "sha256_required_in_independent_summary": True,
    }, "verifier binding drift")
    validation = authority["validation_contract"]
    require(validation["repository_metadata_only"] is True, "validation may read rows")
    require(validation["operator_or_row_level_files_may_be_read"] is False, "operator read authorized")
    require(validation["confirmatory_or_holdout_rows_may_be_read"] is False, "outcome read authorized")
    require(validation["seal_may_be_opened"] is False and validation["model_may_be_fitted"] is False, "unsafe validation")

    return {
        "schema_version": "pivot_feedback_current_state_v3_summary.v1",
        "status": SUMMARY_STATUS,
        "authority": {
            "relative_path": _relative(authority_path),
            "sha256": _sha256(authority_path),
            "parent_sha256": PARENT_SHA,
        },
        "bound_tooling": {
            "builder_relative_path": BUILDER.relative_to(ROOT).as_posix(),
            "builder_sha256": _sha256(BUILDER),
            "independent_verifier_relative_path": VERIFIER.relative_to(ROOT).as_posix(),
            "independent_verifier_sha256": _sha256(VERIFIER),
        },
        "execution_result_replay": {
            "execution_summary_sha256": EXECUTION_SUMMARY_SHA,
            "terminal_status": attempt["terminal_status"],
            "block_id": attempt["block_id"],
            "independent_execution_verification_pass": True,
            "confirmatory_rows_read": 0,
            "holdout_rows_read": 0,
            "private_g4_artifacts_read": 0,
            "seal_opened": False,
        },
        "routing_replay": {
            "active_phase": "E",
            "E_state": "BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ",
            "E_complete": False,
            "M_state": "NOT_RUN",
            "H_state": "BYTE_SEALED_UNOPENED",
            "pass_f": True,
            "pass_g0_compatibility": True,
            "pass_e": False,
            "pass_g1": False,
        },
        "lifecycle_replay": {
            "empirical_g1_status": "BLOCKED",
            "confirmatory_or_holdout_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
            "holdout_seal_opened": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": authority["authorization"],
        "claim_boundary": authority["claim_boundary"],
    }


def _write_create_only(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise CurrentStateV3VerificationError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, default=DEFAULT_AUTHORITY)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = verify(args.authority)
    if args.output is not None:
        _write_create_only(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
