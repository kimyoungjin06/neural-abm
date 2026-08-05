#!/usr/bin/env python3
"""Build the additive E/G1 terminal-BLOCKED active authority."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / "experiments/configs/pivot_feedback_current_state_v2.yaml"
EXECUTION_SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_e_g1_execution_v1_20260805.summary.json"
)
DEFAULT_OUTPUT = ROOT / "experiments/configs/pivot_feedback_current_state_v3.yaml"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_current_state_v3.py"
PARENT_SHA = "ca431afb63da4cf2e09d0563c0509e3685efac29dc8679df0f9cc7f1ae7851ab"
EXECUTION_SUMMARY_SHA = "75855d8bc688c1c5001f2c19c25cd6c8a99c62355beb975e6cf1c3df425e6e5c"


class CurrentStateV3BuildError(RuntimeError):
    """Raised when the blocked E transition cannot be authored exactly."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentStateV3BuildError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_document() -> dict[str, Any]:
    require(_sha256(PARENT) == PARENT_SHA, "parent hash drift")
    require(_sha256(EXECUTION_SUMMARY) == EXECUTION_SUMMARY_SHA, "execution summary hash drift")
    parent = yaml.safe_load(PARENT.read_text(encoding="utf-8"))
    result = json.loads(EXECUTION_SUMMARY.read_text(encoding="utf-8"))
    require(parent.get("schema_version") == "pivot_feedback_current_state.v2", "parent schema drift")
    require(parent.get("status") == "ACTIVE_E_PHASE_F_ATOMICALLY_CLOSED_G0_COMPATIBILITY_PASS_HOLDOUT_SEALED_UNOPENED", "parent status drift")
    require(result.get("schema_version") == "pivot_feedback_e_g1_execution_summary.v1", "execution summary schema drift")
    require(result.get("status") == "PASS_INDEPENDENT_VERIFICATION_OF_TERMINAL_BLOCKED_E_G1_NO_CONFIRMATORY_READ", "execution verification status drift")
    terminal = result["terminal_state"]
    require(terminal["empirical_g1_status"] == "BLOCKED", "execution not terminal blocked")
    require(terminal["confirmatory_rows_read"] == 0 and terminal["holdout_rows_read"] == 0, "execution rows were read")
    require(terminal["holdout_seal_opened"] is False and terminal["g4_open"] is False, "holdout opened")

    output = deepcopy(parent)
    output["schema_version"] = "pivot_feedback_current_state.v3"
    output["label"] = "pivot_feedback_current_state_v3"
    output["mode"] = "ACTIVE_SIX_PHASE_E_G1_TERMINAL_BLOCKED_FAIL_CLOSED_NO_CONFIRMATORY_READ"
    output["status"] = "ACTIVE_E_PHASE_BLOCKED_INPUT_AUTHORIZATION_CONFLICT_G1_NOT_RUN_HOLDOUT_SEALED_UNOPENED"
    for phase in output["active_routing"]["phases"]:
        if phase["phase_id"] == "E":
            phase["state"] = "BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ"
            phase["complete"] = False
            phase["may_read_confirmatory_or_holdout_rows"] = False
    output["active_routing"]["active_phase"] = "E"
    output["e_phase_execution_attempt"] = {
        "state": "TERMINAL_BLOCKED_NO_ROUTE_PROMOTION",
        "attempt_count": 1,
        "terminal_status": "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION",
        "block_id": "G1_AUTHOR_COLD_GUARD_REQUIRED_BUT_PRIVATE_G4_ACCESS_FORBIDDEN",
        "execution_summary": {
            "relative_path": EXECUTION_SUMMARY.relative_to(ROOT).as_posix(),
            "sha256": EXECUTION_SUMMARY_SHA,
            "required_schema_version": "pivot_feedback_e_g1_execution_summary.v1",
            "required_status": "PASS_INDEPENDENT_VERIFICATION_OF_TERMINAL_BLOCKED_E_G1_NO_CONFIRMATORY_READ",
            "mutation_allowed": False,
        },
        "registered_scope": result["registered_scope_replay"],
        "confirmatory_rows_read": 0,
        "holdout_rows_read": 0,
        "labels_materialized": 0,
        "estimators_run": 0,
        "private_g4_artifacts_read": 0,
        "holdout_seal_opened": False,
        "g4_open": False,
        "scope_or_threshold_changed": False,
        "resolution_requires_new_explicit_authority": True,
    }
    lifecycle = output["lifecycle"]
    lifecycle["confirmatory_or_holdout_rows_read"] = 0
    lifecycle["empirical_g1_status"] = "BLOCKED"
    lifecycle["model_fitted"] = False
    lifecycle["g4_open"] = False
    lifecycle["scientific_effect"] = "NOT_ESTABLISHED"
    authorization = output["authorization"]
    authorization["e_g1_execution"] = "BLOCKED_REQUIRES_NEW_EXPLICIT_CONFLICT_RESOLUTION_AUTHORITY"
    authorization["confirmatory_or_holdout_analytic_access"] = "NOT_AUTHORIZED_WHILE_E_INPUT_AUTHORIZATION_CONFLICT_PERSISTS"
    authorization["confirmatory_analytic_access"] = "NOT_AUTHORIZED_UNTIL_NEW_EXPLICIT_CONFLICT_RESOLUTION_AUTHORITY"
    authorization["g2_or_g3_choice_resolution"] = "NOT_AUTHORIZED_E_NOT_PASSED"
    authorization["model_fitting"] = "NOT_AUTHORIZED"
    authorization["g4_open"] = "NOT_AUTHORIZED"
    authorization["causal_claim"] = "NOT_AUTHORIZED"
    authorization["scientific_effect"] = "NOT_ESTABLISHED"
    output["additive_parent"] = {
        "relative_path": PARENT.relative_to(ROOT).as_posix(),
        "sha256": PARENT_SHA,
        "required_schema_version": "pivot_feedback_current_state.v2",
        "required_status": "ACTIVE_E_PHASE_F_ATOMICALLY_CLOSED_G0_COMPATIBILITY_PASS_HOLDOUT_SEALED_UNOPENED",
        "mutation_allowed": False,
    }
    output["state_tooling"] = {
        "builder": {
            "relative_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "independent_verifier": {
            "relative_path": VERIFIER.relative_to(ROOT).as_posix(),
            "sha256_required_in_independent_summary": True,
        },
    }
    output["validation_contract"] = {
        "verifier_relative_path": VERIFIER.relative_to(ROOT).as_posix(),
        "summary_json_relative_path": "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.json",
        "summary_markdown_relative_path": "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.md",
        "repository_metadata_only": True,
        "operator_or_row_level_files_may_be_read": False,
        "confirmatory_or_holdout_rows_may_be_read": False,
        "seal_may_be_opened": False,
        "model_may_be_fitted": False,
    }
    output["claim_boundary"] = (
        "This additive authority records the independently verified terminal BLOCKED "
        "E/G1 execution attempt caused by the frozen author-cold requirement and the "
        "simultaneous prohibition on private G4 access. It preserves F and G0 "
        "compatibility, reads no confirmatory or holdout row, opens no seal, fits no "
        "model, does not pass E or G1, and establishes no scientific effect."
    )
    return output


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CurrentStateV3BuildError(f"output exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    document = build_document()
    _write_create_only(args.output, document)
    print(json.dumps({
        "schema_version": document["schema_version"],
        "status": document["status"],
        "active_phase": document["active_routing"]["active_phase"],
        "empirical_g1_status": document["lifecycle"]["empirical_g1_status"],
        "confirmatory_rows_read": document["lifecycle"]["confirmatory_or_holdout_rows_read"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
