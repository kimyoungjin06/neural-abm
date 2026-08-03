#!/usr/bin/env python3
"""Verify the six-choice-only G0 v4 binding successor."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_G0_V4 = (
    REPO_ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v4.yaml"
)
G0_V3_PATH = (
    REPO_ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v3.yaml"
)
GATE_V60_PATH = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v60.yaml"
PRECOMMITMENT_PATH = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_risk_set_weight_support_precommitment_v1.yaml"
)
G0_V3_SHA256 = "0d15a1a7df535dbbc7e50ca0c3bb386da9a704cd051fb829d275b6d5a776a8fe"
GATE_V60_SHA256 = "592459943ef48da8e6f88af54b18dfcc06aa76db254deee60731b7a1bee35f99"
PRECOMMITMENT_SHA256 = (
    "8f38863eba1880ee375d982cbb2ac094d3ae64028ee651e92b59f13d0189f26e"
)
G0_V4_SCHEMA = "pivot_feedback_g0_measurement_freeze.v5"
G0_V4_STATUS = (
    "PENDING_G0_V4_TWELVE_RESOLVED_SEVEN_CHOICES_AND_FIFTEEN_SECTIONS_NO_SEAL"
)
VERIFY_SCHEMA = "pivot_feedback_g0_measurement_freeze_v4_binding_summary.v1"
VERIFY_STATUS = "PASS_G0_V4_EXACT_SIX_CHOICE_BINDING_SEVEN_CHOICES_REMAIN"
CHOICE_IDS = [
    "negative_sampling_ratio",
    "inverse_probability_weight_truncation",
    "minimum_cohort_size",
    "small_cell_suppression_threshold",
    "capacity_cell_support_threshold",
    "collinearity_threshold",
]
REMAINING_CHOICE_IDS = [
    "g1_replication_tolerance",
    "g2_score_margin",
    "g2_maximum_calibration_degradation",
    "g2_macro_trajectory_tolerance",
    "simulation_draws_or_mcse_stopping_rule",
    "parameter_budget",
    "tuning_budget",
]
ALLOWED_CHANGED_TOP_LEVEL = {
    "schema_version",
    "label",
    "mode",
    "status",
    "authorization_parent",
    "immutable_predecessor",
    "risk_set_weight_support_precommitment_binding",
    "lifecycle",
    "resolution",
    "readiness",
    "authorization",
    "access_accounting_contract",
    "claim_boundary",
}


class G0MeasurementFreezeV4Error(RuntimeError):
    """Fail-closed G0 v4 binding verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G0MeasurementFreezeV4Error(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise G0MeasurementFreezeV4Error(f"invalid {label}: {error}") from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _lookup(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(isinstance(current, Mapping) and part in current, f"missing {dotted}")
        current = current[part]
    return current


def _validate_authorization(g0: Mapping[str, Any]) -> dict[str, Any]:
    parent = _mapping(g0.get("authorization_parent"), "authorization parent")
    require(
        parent.get("relative_path")
        == "experiments/configs/pivot_feedback_gate_state_v60.yaml",
        "authorization parent path",
    )
    require(parent.get("sha256") == GATE_V60_SHA256, "authorization parent hash")
    require(
        parent.get("required_schema_version") == "pivot_feedback_gate_state.v60",
        "authorization schema",
    )
    require(
        parent.get("required_status")
        == "VALID_SIX_RISK_SET_WEIGHT_SUPPORT_CHOICES_FIXTURE_ONLY_G0_V4_BINDING_AUTHORIZED",
        "authorization status",
    )
    gate_path = _safe_repo_path(parent["relative_path"], "authorization parent")
    require(_sha256(gate_path) == GATE_V60_SHA256, "authorization parent byte hash")
    gate = _read_yaml(gate_path, "authorization gate")
    for dotted, expected in _mapping(
        parent.get("required_fields"), "required fields"
    ).items():
        require(_lookup(gate, dotted) == expected, f"authorization field {dotted}")
    require(parent.get("mutation_allowed") is False, "authorization mutation")

    binding = _mapping(
        g0.get("risk_set_weight_support_precommitment_binding"),
        "precommitment binding",
    )
    require(
        binding.get("relative_path")
        == "experiments/configs/pivot_feedback_g0_risk_set_weight_support_precommitment_v1.yaml",
        "precommitment path",
    )
    require(binding.get("sha256") == PRECOMMITMENT_SHA256, "precommitment hash")
    require(
        binding.get("required_schema_version")
        == "pivot_feedback_g0_risk_set_weight_support_precommitment.v1",
        "precommitment schema",
    )
    require(
        binding.get("required_status")
        == "FROZEN_SIX_RISK_SET_WEIGHT_SUPPORT_CHOICES_PENDING_INDEPENDENT_VALIDATION_NO_G0_PROMOTION",
        "precommitment status",
    )
    require(
        binding.get("authorization_gate_sha256") == GATE_V60_SHA256,
        "binding gate hash",
    )
    require(binding.get("exact_choice_ids") == CHOICE_IDS, "binding choice IDs")
    require(binding.get("mutation_allowed") is False, "binding mutation")
    contract_path = _safe_repo_path(binding["relative_path"], "precommitment")
    require(_sha256(contract_path) == PRECOMMITMENT_SHA256, "precommitment byte hash")
    return _read_yaml(contract_path, "precommitment contract")


def _validate_predecessor(g0: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = _mapping(g0.get("immutable_predecessor"), "predecessor")
    require(
        predecessor.get("relative_path")
        == "experiments/configs/pivot_feedback_g0_measurement_freeze_v3.yaml",
        "predecessor path",
    )
    require(predecessor.get("sha256") == G0_V3_SHA256, "predecessor hash")
    require(
        predecessor.get("required_schema_version")
        == "pivot_feedback_g0_measurement_freeze.v4",
        "predecessor schema",
    )
    require(
        predecessor.get("required_status")
        == "PENDING_G0_V3_SIX_RESOLVED_THIRTEEN_CHOICES_AND_FIFTEEN_SECTIONS_NO_SEAL",
        "predecessor status",
    )
    require(
        predecessor.get("predecessor_mutation_allowed") is False,
        "predecessor mutation",
    )
    path = _safe_repo_path(predecessor["relative_path"], "predecessor")
    require(_sha256(path) == G0_V3_SHA256, "predecessor byte hash")
    document = _read_yaml(path, "G0 v3")
    for dotted, expected in _mapping(
        predecessor.get("required_fields"), "predecessor fields"
    ).items():
        require(_lookup(document, dotted) == expected, f"predecessor field {dotted}")
    return document


def _validate_unchanged_subtrees(
    g0_v3: Mapping[str, Any], g0_v4: Mapping[str, Any]
) -> list[str]:
    unchanged = sorted(set(g0_v3) - ALLOWED_CHANGED_TOP_LEVEL)
    require(
        set(g0_v4) - ALLOWED_CHANGED_TOP_LEVEL == set(unchanged),
        "unexpected top-level successor keys",
    )
    for key in unchanged:
        require(g0_v4.get(key) == g0_v3.get(key), f"unauthorized subtree drift {key}")
    require(
        "risk_set_weight_support_precommitment_binding" not in g0_v3,
        "predecessor binding collision",
    )
    return unchanged


def _validate_allowed_metadata_changes(
    g0_v3: Mapping[str, Any], g0_v4: Mapping[str, Any]
) -> None:
    lifecycle_old = deepcopy(_mapping(g0_v3.get("lifecycle"), "v3 lifecycle"))
    lifecycle_new = deepcopy(_mapping(g0_v4.get("lifecycle"), "v4 lifecycle"))
    require(
        lifecycle_new.pop("successor_skeleton_status")
        == "TWELVE_CHOICES_SERIALIZED_SEVEN_PENDING_THREE_BLOCKERS",
        "successor lifecycle status",
    )
    lifecycle_old.pop("successor_skeleton_status")
    require(lifecycle_new == lifecycle_old, "lifecycle drift")

    readiness_old = deepcopy(_mapping(g0_v3.get("readiness"), "v3 readiness"))
    readiness_new = deepcopy(_mapping(g0_v4.get("readiness"), "v4 readiness"))
    require(
        readiness_new.pop("status")
        == "G0_NOT_READY_THREE_BLOCKERS_REMAIN_SEVEN_CHOICES_PENDING",
        "readiness status",
    )
    readiness_old.pop("status")
    require(
        readiness_new["remaining_blockers"]
        == [
            "HOLDOUT_ACCESS_SEAL_RECEIPT_ABSENT",
            "SEVEN_QUANTITATIVE_CHOICES_UNRESOLVED",
            "FIFTEEN_REQUIRED_SECTIONS_NOT_FREEZE_READY",
        ],
        "readiness blockers",
    )
    readiness_new["remaining_blockers"] = readiness_old["remaining_blockers"]
    require(readiness_new == readiness_old, "readiness drift")

    auth_old = deepcopy(_mapping(g0_v3.get("authorization"), "v3 authorization"))
    auth_new = deepcopy(_mapping(g0_v4.get("authorization"), "v4 authorization"))
    require(
        auth_new.pop("additional_quantitative_choice_resolution")
        == "SIX_NEW_CHOICES_BOUND_TWELVE_TOTAL_NO_OTHERS_AUTHORIZED",
        "choice authorization",
    )
    auth_old.pop("additional_quantitative_choice_resolution")
    require(auth_new == auth_old, "authorization drift")

    access_old = deepcopy(
        _mapping(g0_v3.get("access_accounting_contract"), "v3 access")
    )
    access_new = deepcopy(
        _mapping(g0_v4.get("access_accounting_contract"), "v4 access")
    )
    require(access_new.pop("repository_documents_read") == 12, "repository reads")
    access_old.pop("repository_documents_read")
    require(access_new == access_old, "access drift")


def _validate_resolution(
    g0_v3: Mapping[str, Any],
    g0_v4: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    old = _mapping(g0_v3.get("resolution"), "v3 resolution")
    new = _mapping(g0_v4.get("resolution"), "v4 resolution")
    require(
        new.get("state") == "PENDING_TWELVE_VALUES_SERIALIZED_SEVEN_PENDING",
        "resolution state",
    )
    require(new.get("resolved_measurement_count") == 12, "resolved count")
    require(new.get("unresolved_quantitative_choice_count") == 7, "unresolved count")
    require(new.get("unresolved_required_section_count") == 15, "section count")
    require(new.get("freeze_ready_required_section_count") == 0, "freeze-ready count")
    require(new.get("successor_required") is True, "successor required")
    require(
        new.get("unresolved_required_sections")
        == old.get("unresolved_required_sections"),
        "required-section drift",
    )
    unresolved = _mapping(new.get("unresolved_quantitative_choices"), "unresolved choices")
    require(list(unresolved) == REMAINING_CHOICE_IDS, "remaining choice order")
    require(all(value is None for value in unresolved.values()), "remaining choice values")

    old_resolved = _mapping(old.get("resolved_measurements"), "old resolved")
    resolved = _mapping(new.get("resolved_measurements"), "new resolved")
    require(list(resolved)[:6] == list(old_resolved), "inherited resolved order")
    for key, value in old_resolved.items():
        require(resolved.get(key) == value, f"inherited resolved drift {key}")
    require(list(resolved)[6:] == CHOICE_IDS, "new resolved order")

    choices = _mapping(contract.get("precommitment_choices"), "precommitment choices")
    sampling = _mapping(resolved["negative_sampling_ratio"], "resolved sampling")
    require(sampling["selected_from_observed_outcomes"] is False, "sampling outcome selection")
    require(
        sampling["value"]
        == {
            "rule_id": choices["negative_sampling_ratio"]["rule_id"],
            "ratio_controls_per_case": 10,
            "sampling": "WITHOUT_REPLACEMENT_WITHIN_EXACT_RISK_SET",
            "deterministic_component_seed": 20260718,
            "control_inclusion_probability": "SELECTED_DIVIDED_BY_ELIGIBLE_CONTROLS",
            "case_inclusion_probability": 1.0,
        },
        "sampling binding",
    )
    weights = _mapping(
        resolved["inverse_probability_weight_truncation"], "resolved weights"
    )
    require(weights["selected_from_observed_outcomes"] is False, "weight outcome selection")
    require(weights["value"]["percentile_bounds"] == [0.01, 0.99], "weight binding")
    require(weights["value"]["raw_untruncated_sensitivity_required"] is True, "weight sensitivity binding")
    cohort = _mapping(resolved["minimum_cohort_size"], "resolved cohort")
    require(cohort["selected_from_observed_outcomes"] is False, "cohort outcome selection")
    require(
        [
            cohort["value"]["minimum_total_opportunity_rows"],
            cohort["value"]["minimum_entry_events"],
            cohort["value"]["minimum_non_entry_opportunities"],
        ]
        == [500, 100, 100],
        "cohort binding",
    )
    suppression = _mapping(
        resolved["small_cell_suppression_threshold"], "resolved suppression"
    )
    require(suppression["value"]["suppressed_inclusive_range"] == [1, 10], "suppression binding")
    require(suppression["value"]["cms_policy_applies_to_openalex"] is False, "CMS boundary")
    capacity = _mapping(
        resolved["capacity_cell_support_threshold"], "resolved capacity"
    )
    require(
        [
            capacity["value"]["minimum_total_opportunities_per_cell"],
            capacity["value"]["minimum_entry_events_per_cell"],
            capacity["value"]["minimum_non_entry_opportunities_per_cell"],
        ]
        == [50, 11, 11],
        "capacity binding",
    )
    collinearity = _mapping(
        resolved["collinearity_threshold"], "resolved collinearity"
    )
    require(collinearity["value"]["strict_upper_bound"] == 5.0, "VIF binding")
    require(collinearity["value"]["threshold_boundary_counts_as_pass"] is False, "VIF boundary")
    require(
        capacity["value"]["failure_action"]
        == collinearity["value"]["failure_action"]
        == "COLLAPSE_CHANNELS_AND_FORFEIT_SEPARATE_CLAIM",
        "collapse binding",
    )
    require(
        all(
            resolved[choice_id]["selected_from_observed_outcomes"] is False
            for choice_id in CHOICE_IDS
        ),
        "new choice outcome selection",
    )
    return {
        "resolved_measurement_count": 12,
        "newly_bound_choice_ids": CHOICE_IDS,
        "remaining_choice_ids": REMAINING_CHOICE_IDS,
        "remaining_choice_count": 7,
        "required_section_count_pending": 15,
    }


def _validate_current_boundary(g0: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(g0.get("lifecycle"), "lifecycle")
    for key in (
        "confirmatory_outcomes_inspected",
        "g4_holdout_opened",
        "pass_g0",
        "pass_g1",
        "execution_authorized",
    ):
        require(lifecycle.get(key) is False, f"lifecycle boundary {key}")
    holdout = _mapping(g0.get("holdout_access_binding"), "holdout binding")
    require(holdout.get("state") == "PENDING", "holdout state")
    require(holdout.get("receipt_relative_path") is None, "holdout receipt path")
    require(holdout.get("receipt_sha256") is None, "holdout receipt hash")
    require(g0["readiness"]["g0_freeze_ready"] is False, "G0 readiness")
    access = _mapping(g0.get("access_accounting_contract"), "access")
    for key in (
        "operator_json_metadata_files_read",
        "operator_parquet_files_opened",
        "production_row_level_files_opened",
        "confirmatory_or_holdout_rows_read",
        "api_requests_created_or_submitted",
    ):
        require(access.get(key) == 0, f"access boundary {key}")
    require(access.get("model_fitted") is False, "model boundary")
    require(access.get("g4_open") is False, "G4 boundary")
    return {
        "holdout_receipt_bound": False,
        "g0_freeze_ready": False,
        "confirmatory_or_holdout_rows_read": 0,
        "model_fitted": False,
        "g4_open": False,
    }


def run(g0_v4_path: Path = DEFAULT_G0_V4) -> dict[str, Any]:
    g0_v4_path = g0_v4_path.resolve(strict=True)
    g0 = _read_yaml(g0_v4_path, "G0 v4")
    require(g0.get("schema_version") == G0_V4_SCHEMA, "G0 v4 schema")
    require(g0.get("label") == "pivot_feedback_g0_measurement_freeze_v4", "G0 v4 label")
    require(
        g0.get("mode")
        == "VERSIONED_SIX_CHOICE_BINDING_ONLY_SUCCESSOR_THREE_BLOCKERS_REMAIN",
        "G0 v4 mode",
    )
    require(g0.get("status") == G0_V4_STATUS, "G0 v4 status")
    precommitment = _validate_authorization(g0)
    g0_v3 = _validate_predecessor(g0)
    unchanged = _validate_unchanged_subtrees(g0_v3, g0)
    _validate_allowed_metadata_changes(g0_v3, g0)
    resolution = _validate_resolution(g0_v3, g0, precommitment)
    boundary = _validate_current_boundary(g0)
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "preservation_replay": {
            "immutable_predecessor_sha256": G0_V3_SHA256,
            "unchanged_top_level_subtrees": unchanged,
            "unchanged_top_level_subtree_count": len(unchanged),
            "only_authorized_six_choice_binding_changed": True,
        },
        "resolution_replay": resolution,
        "current_boundary_replay": boundary,
        "access_accounting": {
            "repository_yaml_documents_read": 4,
            "operator_json_metadata_files_read": 0,
            "operator_parquet_files_opened": 0,
            "production_row_level_files_opened": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "repository_artifacts": {
            "g0_v4": {
                "relative_path": str(g0_v4_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(g0_v4_path),
            },
            "validator": {
                "relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "authorization": {
            "g0_v4_six_choice_binding": "COMPLETE_INDEPENDENTLY_VERIFIED",
            "remaining_choice_resolution": "REQUIRES_SUCCESSOR_GATE",
            "required_section_completion": "NOT_AUTHORIZED",
            "confirmatory_or_holdout_access": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": g0["claim_boundary"].strip(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-v4", type=Path, default=DEFAULT_G0_V4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(args.g0_v4)
    payload = json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
