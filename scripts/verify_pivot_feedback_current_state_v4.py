#!/usr/bin/env python3
"""Independently verify the terminal support-blocked E/G1 active authority."""

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
DEFAULT_AUTHORITY = ROOT / "experiments/configs/pivot_feedback_current_state_v4.yaml"
PARENT = ROOT / "experiments/configs/pivot_feedback_current_state_v3.yaml"
PARENT_SHA = "0e16e844a8a7ce8b3321eff47c3763f45582b4eef8f039dc6e18dc092bf94380"
PARENT_SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_current_state_v3_20260805.summary.json"
PARENT_SUMMARY_SHA = "10c950d2516b3f1e9ea7e61af4c9018f8e4e27a2eeffb82371387e1b3359ba21"
EXCEPTION = ROOT / "experiments/configs/pivot_feedback_e_g1_guard_only_exception_v1.yaml"
EXCEPTION_SHA = "c9827e04af658b5004e86524aec964e620dd7216848d95d736df20cee0841842"
EXCEPTION_SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_e_g1_guard_only_exception_v1_20260805.summary.json"
EXCEPTION_SUMMARY_SHA = "7d284482b21fe6f0919ac677db5198611cdbdc3840e157d1c9933de5ad4f86d6"
EXECUTION_SUMMARY = ROOT / "experiments/evidence/results/pivot_feedback_e_g1_guarded_risk_label_source_v1_20260805.summary.json"
EXECUTION_SUMMARY_SHA = "f6d55daac53506db8e5cd198dbb280868a51e56af52e85f100a766f447552178"
BUILDER = ROOT / "scripts/build_pivot_feedback_current_state_v4.py"
VERIFIER = Path(__file__).resolve()

AUTHORITY_STATUS = (
    "ACTIVE_E_PHASE_TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_"
    "CONFIRMATORY_LABEL_MATERIALIZATION_NO_ESTIMATION_HOLDOUT_SEALED_UNOPENED"
)
SUMMARY_STATUS = (
    "PASS_ACTIVE_E_TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_"
    "CONFIRMATORY_MATERIALIZATION_NO_ESTIMATION"
)
TERMINAL = "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION"
BLOCK_ID = "G1_FROZEN_EPISODE_COHORT_SUPPORT_FLOOR_FAILED_AFTER_GUARD_FIRST_MATERIALIZATION"


class CurrentStateV4VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentStateV4VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _binding(path: Path, sha: str, schema: str, status: str) -> dict[str, Any]:
    return {
        "relative_path": path.relative_to(ROOT).as_posix(),
        "sha256": sha,
        "required_schema_version": schema,
        "required_status": status,
        "mutation_allowed": False,
    }


def verify(authority_path: Path) -> dict[str, Any]:
    for path, sha, name in (
        (PARENT, PARENT_SHA, "parent"),
        (PARENT_SUMMARY, PARENT_SUMMARY_SHA, "parent summary"),
        (EXCEPTION, EXCEPTION_SHA, "exception"),
        (EXCEPTION_SUMMARY, EXCEPTION_SUMMARY_SHA, "exception summary"),
        (EXECUTION_SUMMARY, EXECUTION_SUMMARY_SHA, "execution summary"),
    ):
        require(_sha256(path) == sha, f"{name} hash drift")
    authority_path = authority_path.resolve(strict=True)
    parent = yaml.safe_load(PARENT.read_text(encoding="utf-8"))
    authority = yaml.safe_load(authority_path.read_text(encoding="utf-8"))
    exception = yaml.safe_load(EXCEPTION.read_text(encoding="utf-8"))
    exception_summary = _json(EXCEPTION_SUMMARY)
    execution = _json(EXECUTION_SUMMARY)
    require(authority.get("schema_version") == "pivot_feedback_current_state.v4", "authority schema drift")
    require(authority.get("status") == AUTHORITY_STATUS, "authority status drift")
    require(authority.get("mode") == "ACTIVE_SIX_PHASE_E_G1_TERMINAL_SUPPORT_BLOCKED_AFTER_CONFIRMATORY_LABELS_NO_ESTIMATION", "authority mode drift")
    require(exception.get("authorization_basis") == "EXPLICIT_USER_APPROVAL_GUARD_ONLY_EXCEPTION", "exception approval drift")
    require(exception_summary.get("status") == "PASS_INDEPENDENT_EXACT_GUARD_ONLY_EXCEPTION_G1_RESUME_AUTHORIZED", "exception summary drift")
    require(execution.get("status") == "PASS_INDEPENDENT_G1_GUARD_FIRST_RISK_AND_LABEL_SOURCE_ROW_REPLAY_ESTIMATION_NOT_RUN", "execution summary drift")
    disposition = execution["registered_execution_disposition"]
    rows = execution["independent_row_replay"]
    access = execution["access_boundary"]
    guard = execution["guard_replay"]

    require(authority["additive_parent"] == _binding(PARENT, PARENT_SHA, "pivot_feedback_current_state.v3", parent["status"]), "parent binding drift")
    preserved = set(parent) - {
        "schema_version", "label", "mode", "status", "active_routing", "lifecycle",
        "authorization", "additive_parent", "validation_contract", "claim_boundary",
        "e_phase_execution_attempt", "state_tooling",
    }
    for key in preserved:
        require(authority.get(key) == parent.get(key), f"preserved subtree drift: {key}")

    expected_routing = deepcopy(parent["active_routing"])
    for phase in expected_routing["phases"]:
        if phase["phase_id"] == "E":
            phase["state"] = "TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_LABEL_MATERIALIZATION"
            phase["complete"] = False
            phase["may_read_confirmatory_or_holdout_rows"] = False
            phase["may_fit_model"] = False
    expected_routing["active_phase"] = "E"
    require(authority["active_routing"] == expected_routing, "routing drift")

    expected_lifecycle = deepcopy(parent["lifecycle"])
    expected_lifecycle.update({
        "confirmatory_or_holdout_rows_read": rows["guarded_risk_rows"],
        "empirical_g1_status": "BLOCKED",
        "g1_guard_only_exception_independently_verified": True,
        "g1_guarded_label_rows_materialized": rows["guarded_risk_rows"],
        "g1_entry_rows_materialized": rows["entry_rows"],
        "g1_support_failed_unit_count": disposition["failed_unit_count"],
        "g1_estimators_run": 0,
        "model_fitted": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    })
    require(authority["lifecycle"] == expected_lifecycle, "lifecycle drift")

    expected_authorization = deepcopy(parent["authorization"])
    expected_authorization.update({
        "e_g1_execution": "TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_NO_ROUTE_PROMOTION",
        "confirmatory_or_holdout_analytic_access": "NOT_AUTHORIZED_AFTER_TERMINAL_E_SUPPORT_FAILURE",
        "confirmatory_analytic_access": "NOT_AUTHORIZED_FOR_CURRENT_CONFIRMATORY_COHORT_AFTER_TERMINAL_SUPPORT_FAILURE",
        "g2_or_g3_choice_resolution": "NOT_AUTHORIZED_E_NOT_PASSED",
        "outcome_based_threshold_selection": "NOT_AUTHORIZED",
        "scope_or_threshold_relaxation_on_read_cohort": "NOT_AUTHORIZED",
        "same_cohort_g1_retry": "NOT_AUTHORIZED",
        "disjoint_g1_successor": "REQUIRES_NEW_OUTCOME_BLIND_PREAUTHORIZATION_AND_FRESH_FREEZE",
        "model_fitting": "NOT_AUTHORIZED",
        "g4_open": "NOT_AUTHORIZED",
        "causal_claim": "NOT_AUTHORIZED",
        "scientific_effect": "NOT_ESTABLISHED",
    })
    require(authority["authorization"] == expected_authorization, "authorization drift")

    transition = authority["guard_only_exception_transition"]
    require(transition["authority"] == _binding(EXCEPTION, EXCEPTION_SHA, "pivot_feedback_e_g1_guard_only_exception.v1", "AUTHORIZED_EXACT_GUARD_ONLY_EXCEPTION_PENDING_INDEPENDENT_VERIFICATION"), "exception binding drift")
    require(transition["independent_verification"] == _binding(EXCEPTION_SUMMARY, EXCEPTION_SUMMARY_SHA, "pivot_feedback_e_g1_guard_only_exception_summary.v1", "PASS_INDEPENDENT_EXACT_GUARD_ONLY_EXCEPTION_G1_RESUME_AUTHORIZED"), "exception verification binding drift")
    require(transition["resolved_parent_block_id"] == parent["e_phase_execution_attempt"]["block_id"], "resolved block drift")
    require(transition["resolution_scope"] == "EXACT_AUTHOR_COLD_GUARD_PROJECTION_AND_PRE_LABEL_ANTI_JOIN_ONLY", "exception scope drift")
    require(transition["other_private_g4_access_authorized"] is False and transition["seal_open_authorized"] is False, "exception expansion")

    attempt = authority["e_phase_execution_attempt"]
    require(attempt["state"] == "TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_NO_ROUTE_PROMOTION", "attempt state drift")
    require(attempt["attempt_count"] == 2 and attempt["terminal_status"] == TERMINAL and attempt["block_id"] == BLOCK_ID, "attempt terminal drift")
    require(attempt["previous_attempt_parent"] == _binding(PARENT, PARENT_SHA, "pivot_feedback_current_state.v3", parent["status"]), "previous attempt binding drift")
    require(attempt["execution_summary"] == _binding(EXECUTION_SUMMARY, EXECUTION_SUMMARY_SHA, "pivot_feedback_e_g1_guarded_risk_label_source_summary.v1", "PASS_INDEPENDENT_G1_GUARD_FIRST_RISK_AND_LABEL_SOURCE_ROW_REPLAY_ESTIMATION_NOT_RUN"), "execution binding drift")
    require(attempt["registered_scope"] == {
        "entry_years": [2013, 2016], "episode_count": 28, "family_count": 13,
        "field_ids_in_registered_slot_order": [16, 32, 11],
        "partition_id": "g1_empirical_replication", "response_cohort_count": 84,
        "scope_changed": False,
    }, "registered scope drift")
    require(attempt["materialization"]["guarded_risk_rows"] == rows["guarded_risk_rows"], "materialized row drift")
    require(attempt["materialization"]["entry_rows"] == rows["entry_rows"], "entry row drift")
    require(attempt["support"] == {
        "frozen_policy_rule_id": disposition["frozen_policy_rule_id"],
        "failed_unit_action": disposition["failed_unit_action"],
        "failed_unit_count": 2,
        "failed_units": disposition["failed_units"],
        "pooling_or_threshold_relaxation_authorized": False,
    }, "support replay drift")
    require(attempt["access_accounting"] == {
        "confirmatory_rows_materialized": rows["guarded_risk_rows"],
        "guard_rows_read": guard["guard_rows"],
        "holdout_outcome_rows_read": 0,
        "other_private_g4_files_read": 0,
        "seal_open_events": 0,
        "g4_open": False,
        "feature_control_rows_materialized": 0,
        "sampling_weighting_runs": 0,
        "estimators_run": 0,
        "model_selection_scores_read": 0,
    }, "attempt access drift")
    require(attempt["scope_or_threshold_changed"] is False and attempt["same_cohort_retry_authorized"] is False, "unsafe retry drift")
    require(attempt["disjoint_successor_requires_new_outcome_blind_precommitment"] is True, "successor boundary drift")
    require(access["holdout_outcome_rows_read"] == 0 and access["estimators_run"] == 0, "execution access drift")

    require(authority["state_tooling"] == {
        "builder": {"relative_path": BUILDER.relative_to(ROOT).as_posix(), "sha256": _sha256(BUILDER)},
        "independent_verifier": {"relative_path": VERIFIER.relative_to(ROOT).as_posix(), "sha256_required_in_independent_summary": True},
    }, "state tooling drift")
    validation = authority["validation_contract"]
    require(validation["repository_metadata_only"] is True, "validation row access drift")
    require(validation["operator_or_row_level_files_may_be_read"] is False, "operator access drift")
    require(validation["confirmatory_or_holdout_rows_may_be_read"] is False, "validation confirmatory access drift")
    require(validation["seal_may_be_opened"] is False and validation["model_may_be_fitted"] is False, "unsafe validation")

    return {
        "schema_version": "pivot_feedback_current_state_v4_summary.v1",
        "status": SUMMARY_STATUS,
        "authority": {
            "relative_path": authority_path.relative_to(ROOT).as_posix(),
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
            "terminal_status": TERMINAL,
            "block_id": BLOCK_ID,
            "guard_only_exception_verified": True,
            "guarded_risk_rows": rows["guarded_risk_rows"],
            "entry_rows": rows["entry_rows"],
            "support_failed_episode_ids": [row["episode_id"] for row in disposition["failed_units"]],
            "support_failed_unit_count": disposition["failed_unit_count"],
            "estimators_run": 0,
            "holdout_outcome_rows_read": 0,
            "other_private_g4_files_read": 0,
            "seal_opened": False,
        },
        "routing_replay": {
            "active_phase": "E",
            "E_state": "TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_LABEL_MATERIALIZATION",
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
            "confirmatory_or_holdout_rows_read": rows["guarded_risk_rows"],
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
        raise CurrentStateV4VerificationError(f"output exists: {path}")
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
