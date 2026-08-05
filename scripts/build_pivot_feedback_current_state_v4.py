#!/usr/bin/env python3
"""Build the additive E/G1 terminal support-blocked active authority."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
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
DEFAULT_OUTPUT = ROOT / "experiments/configs/pivot_feedback_current_state_v4.yaml"
VERIFIER = ROOT / "scripts/verify_pivot_feedback_current_state_v4.py"

AUTHORITY_STATUS = (
    "ACTIVE_E_PHASE_TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_"
    "CONFIRMATORY_LABEL_MATERIALIZATION_NO_ESTIMATION_HOLDOUT_SEALED_UNOPENED"
)
TERMINAL = "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION"
BLOCK_ID = "G1_FROZEN_EPISODE_COHORT_SUPPORT_FLOOR_FAILED_AFTER_GUARD_FIRST_MATERIALIZATION"


class CurrentStateV4BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentStateV4BuildError(message)


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


def build_document() -> dict[str, Any]:
    for path, sha, name in (
        (PARENT, PARENT_SHA, "parent"),
        (PARENT_SUMMARY, PARENT_SUMMARY_SHA, "parent summary"),
        (EXCEPTION, EXCEPTION_SHA, "exception"),
        (EXCEPTION_SUMMARY, EXCEPTION_SUMMARY_SHA, "exception summary"),
        (EXECUTION_SUMMARY, EXECUTION_SUMMARY_SHA, "execution summary"),
    ):
        require(_sha256(path) == sha, f"{name} hash drift")
    parent = yaml.safe_load(PARENT.read_text(encoding="utf-8"))
    parent_summary = _json(PARENT_SUMMARY)
    exception = yaml.safe_load(EXCEPTION.read_text(encoding="utf-8"))
    exception_summary = _json(EXCEPTION_SUMMARY)
    execution = _json(EXECUTION_SUMMARY)
    require(parent.get("schema_version") == "pivot_feedback_current_state.v3", "parent schema drift")
    require(parent_summary.get("status") == "PASS_ACTIVE_E_TERMINAL_BLOCKED_INPUT_AUTHORIZATION_CONFLICT_NO_CONFIRMATORY_READ", "parent summary status drift")
    require(exception.get("authorization_basis") == "EXPLICIT_USER_APPROVAL_GUARD_ONLY_EXCEPTION", "exception approval drift")
    require(exception_summary.get("status") == "PASS_INDEPENDENT_EXACT_GUARD_ONLY_EXCEPTION_G1_RESUME_AUTHORIZED", "exception verification drift")
    require(execution.get("status") == "PASS_INDEPENDENT_G1_GUARD_FIRST_RISK_AND_LABEL_SOURCE_ROW_REPLAY_ESTIMATION_NOT_RUN", "execution verification drift")
    disposition = execution["registered_execution_disposition"]
    access = execution["access_boundary"]
    rows = execution["independent_row_replay"]
    guard = execution["guard_replay"]
    require(disposition["terminal_status"] == TERMINAL, "execution is not terminal blocked")
    require(disposition["blocking_condition"] == "FROZEN_EPISODE_COHORT_SUPPORT_FLOOR_FAILED", "support blocker drift")
    require(disposition["failed_unit_count"] == 2, "failed unit count drift")
    require([row["episode_id"] for row in disposition["failed_units"]] == [11, 25], "failed episode drift")
    require(access["confirmatory_rows_materialized"] == rows["guarded_risk_rows"] == 72_715_221, "confirmatory row closure")
    require(access["estimators_run"] == 0 and access["holdout_outcome_rows_read"] == 0, "unsafe downstream execution")
    require(guard["materialized_guard_overlap_rows"] == 0, "guard overlap")

    output = deepcopy(parent)
    output["schema_version"] = "pivot_feedback_current_state.v4"
    output["label"] = "pivot_feedback_current_state_v4"
    output["mode"] = "ACTIVE_SIX_PHASE_E_G1_TERMINAL_SUPPORT_BLOCKED_AFTER_CONFIRMATORY_LABELS_NO_ESTIMATION"
    output["status"] = AUTHORITY_STATUS
    for phase in output["active_routing"]["phases"]:
        if phase["phase_id"] == "E":
            phase["state"] = "TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_AFTER_GUARD_FIRST_LABEL_MATERIALIZATION"
            phase["complete"] = False
            phase["may_read_confirmatory_or_holdout_rows"] = False
            phase["may_fit_model"] = False
    output["active_routing"]["active_phase"] = "E"
    output["lifecycle"].update({
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
    output["authorization"].update({
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
    output["additive_parent"] = _binding(PARENT, PARENT_SHA, "pivot_feedback_current_state.v3", parent["status"])
    output["guard_only_exception_transition"] = {
        "authority": _binding(EXCEPTION, EXCEPTION_SHA, "pivot_feedback_e_g1_guard_only_exception.v1", "AUTHORIZED_EXACT_GUARD_ONLY_EXCEPTION_PENDING_INDEPENDENT_VERIFICATION"),
        "independent_verification": _binding(EXCEPTION_SUMMARY, EXCEPTION_SUMMARY_SHA, "pivot_feedback_e_g1_guard_only_exception_summary.v1", "PASS_INDEPENDENT_EXACT_GUARD_ONLY_EXCEPTION_G1_RESUME_AUTHORIZED"),
        "resolved_parent_block_id": parent["e_phase_execution_attempt"]["block_id"],
        "resolution_scope": "EXACT_AUTHOR_COLD_GUARD_PROJECTION_AND_PRE_LABEL_ANTI_JOIN_ONLY",
        "other_private_g4_access_authorized": False,
        "seal_open_authorized": False,
    }
    output["e_phase_execution_attempt"] = {
        "state": "TERMINAL_BLOCKED_FROZEN_COHORT_SUPPORT_NO_ROUTE_PROMOTION",
        "attempt_count": 2,
        "terminal_status": TERMINAL,
        "block_id": BLOCK_ID,
        "previous_attempt_parent": _binding(PARENT, PARENT_SHA, "pivot_feedback_current_state.v3", parent["status"]),
        "execution_summary": _binding(EXECUTION_SUMMARY, EXECUTION_SUMMARY_SHA, "pivot_feedback_e_g1_guarded_risk_label_source_summary.v1", "PASS_INDEPENDENT_G1_GUARD_FIRST_RISK_AND_LABEL_SOURCE_ROW_REPLAY_ESTIMATION_NOT_RUN"),
        "registered_scope": {
            "entry_years": [2013, 2016],
            "episode_count": 28,
            "family_count": 13,
            "field_ids_in_registered_slot_order": [16, 32, 11],
            "partition_id": "g1_empirical_replication",
            "response_cohort_count": 84,
            "scope_changed": False,
        },
        "materialization": {
            "production_root": execution["production_artifact"]["root"],
            "receipt_sha256": execution["production_artifact"]["receipt_sha256"],
            "report_sha256": execution["production_artifact"]["report_sha256"],
            "manifest_sha256": execution["production_artifact"]["manifest_sha256"],
            "cohort_file_count": execution["production_artifact"]["cohort_file_count"],
            "pre_guard_risk_rows": rows["pre_guard_risk_rows"],
            "guard_excluded_rows": rows["guard_excluded_rows"],
            "guarded_risk_rows": rows["guarded_risk_rows"],
            "entry_rows": rows["entry_rows"],
            "non_entry_rows": rows["non_entry_rows"],
            "persistence_rows": rows["persistence_rows"],
            "one_off_entry_rows": rows["one_off_entry_rows"],
            "alternative_pivot_entry_rows": rows["alternative_pivot_entry_rows"],
        },
        "support": {
            "frozen_policy_rule_id": disposition["frozen_policy_rule_id"],
            "failed_unit_action": disposition["failed_unit_action"],
            "failed_unit_count": disposition["failed_unit_count"],
            "failed_units": disposition["failed_units"],
            "pooling_or_threshold_relaxation_authorized": False,
        },
        "access_accounting": {
            "confirmatory_rows_materialized": access["confirmatory_rows_materialized"],
            "guard_rows_read": guard["guard_rows"],
            "holdout_outcome_rows_read": 0,
            "other_private_g4_files_read": 0,
            "seal_open_events": 0,
            "g4_open": False,
            "feature_control_rows_materialized": 0,
            "sampling_weighting_runs": 0,
            "estimators_run": 0,
            "model_selection_scores_read": 0,
        },
        "scope_or_threshold_changed": False,
        "same_cohort_retry_authorized": False,
        "disjoint_successor_requires_new_outcome_blind_precommitment": True,
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
        "summary_json_relative_path": "experiments/evidence/results/pivot_feedback_current_state_v4_20260805.summary.json",
        "summary_markdown_relative_path": "experiments/evidence/results/pivot_feedback_current_state_v4_20260805.summary.md",
        "repository_metadata_only": True,
        "operator_or_row_level_files_may_be_read": False,
        "confirmatory_or_holdout_rows_may_be_read": False,
        "seal_may_be_opened": False,
        "model_may_be_fitted": False,
    }
    output["claim_boundary"] = (
        "This additive authority records the exact guard-only exception, the independently "
        "verified guard-first materialization of 72,715,221 G1 confirmatory label rows, and "
        "the terminal failure of two frozen episode support units before sampling or "
        "estimation. It changes no scope or threshold, reads no holdout outcome, opens no "
        "seal, fits no estimator or model, does not pass E or G1, and establishes no "
        "association, null, prediction, mechanism, transport, causal effect, or scientific effect."
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output
    require(not output.exists(), f"output exists: {output}")
    document = build_document()
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(json.dumps({"status": document["status"], "sha256": _sha256(output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
