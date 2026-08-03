#!/usr/bin/env python3
"""Verify the four-choice-only G0 v3 binding successor."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_G0_V3 = (
    REPO_ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v3.yaml"
)
G0_V2_PATH = (
    REPO_ROOT / "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml"
)
GATE_V58_PATH = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v58.yaml"
PRECOMMITMENT_PATH = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_measurement_proxy_precommitment_v1.yaml"
)
G0_V2_SHA256 = "a2cb0cbf444a7c128eecc40a815b80d5fe86f617a36849330612491b03934eb3"
GATE_V58_SHA256 = "cd88a20b6bab1da474617818e8e6207c881a24487d8c2c7bed42e803822a9753"
PRECOMMITMENT_SHA256 = (
    "2e4d90a9541cd9973b8a69bf5fb34f6f35c3d1570a12a4c46b37cac77cca6b5f"
)
G0_V3_SCHEMA = "pivot_feedback_g0_measurement_freeze.v4"
G0_V3_STATUS = (
    "PENDING_G0_V3_SIX_RESOLVED_THIRTEEN_CHOICES_AND_FIFTEEN_SECTIONS_NO_SEAL"
)
VERIFY_SCHEMA = "pivot_feedback_g0_measurement_freeze_v3_binding_summary.v1"
VERIFY_STATUS = "PASS_G0_V3_EXACT_FOUR_CHOICE_BINDING_THIRTEEN_CHOICES_REMAIN"
CHOICE_IDS = [
    "attention_surge_threshold",
    "public_success_threshold",
    "citation_maturation_lag_years",
    "negative_control_equivalence_margins",
]
REMAINING_CHOICE_IDS = [
    "negative_sampling_ratio",
    "inverse_probability_weight_truncation",
    "minimum_cohort_size",
    "small_cell_suppression_threshold",
    "capacity_cell_support_threshold",
    "collinearity_threshold",
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
    "measurement_proxy_precommitment_binding",
    "lifecycle",
    "resolution",
    "readiness",
    "authorization",
    "access_accounting_contract",
    "claim_boundary",
}


class G0MeasurementFreezeV3Error(RuntimeError):
    """Fail-closed G0 v3 binding verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise G0MeasurementFreezeV3Error(message)


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
        raise G0MeasurementFreezeV3Error(f"invalid {label}: {error}") from error


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


def _validate_authorization(g0: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = _mapping(g0.get("authorization_parent"), "authorization parent")
    require(
        parent.get("relative_path")
        == "experiments/configs/pivot_feedback_gate_state_v58.yaml",
        "authorization parent path",
    )
    require(parent.get("sha256") == GATE_V58_SHA256, "authorization parent hash")
    require(parent.get("required_schema_version") == "pivot_feedback_gate_state.v58", "authorization schema")
    require(
        parent.get("required_status")
        == "VALID_FOUR_MEASUREMENT_PROXY_CHOICES_FIXTURE_ONLY_G0_V3_BINDING_AUTHORIZED",
        "authorization status",
    )
    gate_path = _safe_repo_path(parent["relative_path"], "authorization parent")
    require(_sha256(gate_path) == GATE_V58_SHA256, "authorization parent byte hash")
    gate = _read_yaml(gate_path, "authorization gate")
    for dotted, expected in _mapping(parent.get("required_fields"), "required fields").items():
        require(_lookup(gate, dotted) == expected, f"authorization field {dotted}")
    require(parent.get("mutation_allowed") is False, "authorization mutation")

    binding = _mapping(
        g0.get("measurement_proxy_precommitment_binding"), "precommitment binding"
    )
    require(
        binding.get("relative_path")
        == "experiments/configs/pivot_feedback_g0_measurement_proxy_precommitment_v1.yaml",
        "precommitment path",
    )
    require(binding.get("sha256") == PRECOMMITMENT_SHA256, "precommitment hash")
    require(binding.get("required_schema_version") == "pivot_feedback_g0_measurement_proxy_precommitment.v1", "precommitment schema")
    require(
        binding.get("required_status")
        == "FROZEN_FOUR_MEASUREMENT_PROXY_CHOICES_PENDING_INDEPENDENT_VALIDATION_NO_G0_PROMOTION",
        "precommitment status",
    )
    require(binding.get("authorization_gate_sha256") == GATE_V58_SHA256, "binding gate hash")
    require(binding.get("exact_choice_ids") == CHOICE_IDS, "binding choice IDs")
    require(binding.get("mutation_allowed") is False, "binding mutation")
    contract_path = _safe_repo_path(binding["relative_path"], "precommitment")
    require(_sha256(contract_path) == PRECOMMITMENT_SHA256, "precommitment byte hash")
    contract = _read_yaml(contract_path, "precommitment contract")
    return gate, contract


def _validate_predecessor(g0: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = _mapping(g0.get("immutable_predecessor"), "predecessor")
    require(
        predecessor.get("relative_path")
        == "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml",
        "predecessor path",
    )
    require(predecessor.get("sha256") == G0_V2_SHA256, "predecessor hash")
    require(predecessor.get("required_schema_version") == "pivot_feedback_g0_measurement_freeze.v3", "predecessor schema")
    require(
        predecessor.get("required_status")
        == "PENDING_G0_V2_G_MINUS_1A_BOUND_THREE_BLOCKERS_NO_SEAL",
        "predecessor status",
    )
    require(predecessor.get("predecessor_mutation_allowed") is False, "predecessor mutation")
    path = _safe_repo_path(predecessor["relative_path"], "predecessor")
    require(_sha256(path) == G0_V2_SHA256, "predecessor byte hash")
    document = _read_yaml(path, "G0 v2")
    for dotted, expected in _mapping(predecessor.get("required_fields"), "predecessor fields").items():
        require(_lookup(document, dotted) == expected, f"predecessor field {dotted}")
    return document


def _validate_unchanged_subtrees(g0_v2: Mapping[str, Any], g0_v3: Mapping[str, Any]) -> list[str]:
    unchanged = sorted(set(g0_v2) - ALLOWED_CHANGED_TOP_LEVEL)
    require(
        set(g0_v3) - ALLOWED_CHANGED_TOP_LEVEL == set(unchanged),
        "unexpected top-level successor keys",
    )
    for key in unchanged:
        require(g0_v3.get(key) == g0_v2.get(key), f"unauthorized subtree drift {key}")
    require("measurement_proxy_precommitment_binding" not in g0_v2, "predecessor binding collision")
    return unchanged


def _validate_allowed_metadata_changes(g0_v2: Mapping[str, Any], g0_v3: Mapping[str, Any]) -> None:
    lifecycle_v2 = deepcopy(_mapping(g0_v2.get("lifecycle"), "v2 lifecycle"))
    lifecycle_v3 = deepcopy(_mapping(g0_v3.get("lifecycle"), "v3 lifecycle"))
    require(
        lifecycle_v3.pop("successor_skeleton_status")
        == "SIX_CHOICES_SERIALIZED_THIRTEEN_PENDING_THREE_BLOCKERS",
        "successor lifecycle status",
    )
    lifecycle_v2.pop("successor_skeleton_status")
    require(lifecycle_v3 == lifecycle_v2, "lifecycle drift")

    readiness_v2 = deepcopy(_mapping(g0_v2.get("readiness"), "v2 readiness"))
    readiness_v3 = deepcopy(_mapping(g0_v3.get("readiness"), "v3 readiness"))
    require(
        readiness_v3.pop("status")
        == "G0_NOT_READY_THREE_BLOCKERS_REMAIN_THIRTEEN_CHOICES_PENDING",
        "readiness status",
    )
    readiness_v2.pop("status")
    require(
        readiness_v3["remaining_blockers"]
        == [
            "HOLDOUT_ACCESS_SEAL_RECEIPT_ABSENT",
            "THIRTEEN_QUANTITATIVE_CHOICES_UNRESOLVED",
            "FIFTEEN_REQUIRED_SECTIONS_NOT_FREEZE_READY",
        ],
        "readiness blockers",
    )
    readiness_v3["remaining_blockers"] = readiness_v2["remaining_blockers"]
    require(readiness_v3 == readiness_v2, "readiness drift")

    auth_v2 = deepcopy(_mapping(g0_v2.get("authorization"), "v2 authorization"))
    auth_v3 = deepcopy(_mapping(g0_v3.get("authorization"), "v3 authorization"))
    require(
        auth_v3.pop("additional_quantitative_choice_resolution")
        == "FOUR_CHOICES_BOUND_NO_OTHERS_AUTHORIZED",
        "choice authorization",
    )
    auth_v2.pop("additional_quantitative_choice_resolution")
    require(auth_v3 == auth_v2, "authorization drift")

    access_v2 = deepcopy(
        _mapping(g0_v2.get("access_accounting_contract"), "v2 access")
    )
    access_v3 = deepcopy(
        _mapping(g0_v3.get("access_accounting_contract"), "v3 access")
    )
    require(access_v3.pop("repository_documents_read") == 9, "repository reads")
    access_v2.pop("repository_documents_read")
    require(access_v3 == access_v2, "access drift")


def _validate_resolution(
    g0_v2: Mapping[str, Any],
    g0_v3: Mapping[str, Any],
    precommitment: Mapping[str, Any],
) -> dict[str, Any]:
    old = _mapping(g0_v2.get("resolution"), "v2 resolution")
    new = _mapping(g0_v3.get("resolution"), "v3 resolution")
    require(new.get("state") == "PENDING_SIX_VALUES_SERIALIZED_THIRTEEN_PENDING", "resolution state")
    require(new.get("resolved_measurement_count") == 6, "resolved count")
    require(new.get("unresolved_quantitative_choice_count") == 13, "unresolved count")
    require(new.get("unresolved_required_section_count") == 15, "section count")
    require(new.get("freeze_ready_required_section_count") == 0, "freeze-ready count")
    require(new.get("successor_required") is True, "successor required")
    require(new.get("unresolved_required_sections") == old.get("unresolved_required_sections"), "required-section drift")
    unresolved = _mapping(new.get("unresolved_quantitative_choices"), "unresolved choices")
    require(list(unresolved) == REMAINING_CHOICE_IDS, "remaining choice order")
    require(all(value is None for value in unresolved.values()), "remaining choice values")

    old_resolved = _mapping(old.get("resolved_measurements"), "old resolved")
    resolved = _mapping(new.get("resolved_measurements"), "new resolved")
    require(list(resolved)[:2] == list(old_resolved), "inherited resolved order")
    for key, value in old_resolved.items():
        require(resolved.get(key) == value, f"inherited resolved drift {key}")
    require(list(resolved)[2:] == CHOICE_IDS, "new resolved order")

    choices = _mapping(precommitment.get("precommitment_choices"), "precommitment choices")
    attention = _mapping(resolved["attention_surge_threshold"], "resolved attention")
    require(attention.get("selected_from_observed_outcomes") is False, "attention outcome selection")
    require(
        attention.get("value")
        == {
            "rule_id": choices["attention_surge_threshold"]["rule_id"],
            "primary_modified_z_threshold_inclusive": 3.0,
            "recent_window_offsets": [-3, -2, -1],
            "reference_window_offsets": [-6, -5, -4],
            "mad_zero_action": "NO_SURGE_FAIL_CLOSED",
        },
        "attention binding",
    )
    public = _mapping(resolved["public_success_threshold"], "resolved public")
    require(public.get("selected_from_observed_outcomes") is False, "public outcome selection")
    require(
        public.get("value")
        == {
            "rule_id": choices["public_success_threshold"]["rule_id"],
            "target_fraction": 0.10,
            "boundary_tie_rule": "INCLUDE_ALL_WORKS_AT_OR_ABOVE_CUTOFF",
            "minimum_reference_cohort_for_decile_calculation": 10,
            "exact_ten_percent_membership_claim_allowed": False,
        },
        "public binding",
    )
    maturation = _mapping(resolved["citation_maturation_lag_years"], "resolved maturation")
    require(maturation.get("value") == 3, "maturation binding")
    require(maturation.get("current_snapshot_bibliographic_time_proxy_only") is True, "maturation proxy")
    require(maturation.get("selected_from_observed_outcomes") is False, "maturation outcome selection")
    margins = _mapping(resolved["negative_control_equivalence_margins"], "resolved margins")
    require(margins.get("selected_from_observed_outcomes") is False, "margin outcome selection")
    require(
        margins.get("value")
        == {
            "estimand_scale": "AVERAGE_MARGINAL_BINARY_OUTCOME_PROBABILITY_DIFFERENCE",
            "symmetric_practical_null": [-0.01, 0.01],
            "equivalence_test": "TWO_ONE_SIDED_TESTS",
            "one_sided_alpha": 0.05,
            "equivalent_confidence_interval": "TWO_SIDED_90_PERCENT",
        },
        "margin binding",
    )
    return {
        "resolved_measurement_count": 6,
        "newly_bound_choice_ids": CHOICE_IDS,
        "remaining_choice_ids": REMAINING_CHOICE_IDS,
        "remaining_choice_count": 13,
        "required_section_count_pending": 15,
    }


def _validate_current_boundary(g0: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(g0.get("lifecycle"), "lifecycle")
    for key in ("confirmatory_outcomes_inspected", "g4_holdout_opened", "pass_g0", "pass_g1", "execution_authorized"):
        require(lifecycle.get(key) is False, f"lifecycle boundary {key}")
    holdout = _mapping(g0.get("holdout_access_binding"), "holdout binding")
    require(holdout.get("state") == "PENDING", "holdout state")
    require(holdout.get("receipt_relative_path") is None, "holdout receipt path")
    require(holdout.get("receipt_sha256") is None, "holdout receipt hash")
    readiness = _mapping(g0.get("readiness"), "readiness")
    require(readiness.get("g0_freeze_ready") is False, "G0 readiness")
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


def run(g0_v3_path: Path = DEFAULT_G0_V3) -> dict[str, Any]:
    g0_v3_path = g0_v3_path.resolve(strict=True)
    g0 = _read_yaml(g0_v3_path, "G0 v3")
    require(g0.get("schema_version") == G0_V3_SCHEMA, "G0 v3 schema")
    require(g0.get("label") == "pivot_feedback_g0_measurement_freeze_v3", "G0 v3 label")
    require(
        g0.get("mode")
        == "VERSIONED_FOUR_CHOICE_BINDING_ONLY_SUCCESSOR_THREE_BLOCKERS_REMAIN",
        "G0 v3 mode",
    )
    require(g0.get("status") == G0_V3_STATUS, "G0 v3 status")
    _gate, precommitment = _validate_authorization(g0)
    g0_v2 = _validate_predecessor(g0)
    unchanged = _validate_unchanged_subtrees(g0_v2, g0)
    _validate_allowed_metadata_changes(g0_v2, g0)
    resolution = _validate_resolution(g0_v2, g0, precommitment)
    boundary = _validate_current_boundary(g0)
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "preservation_replay": {
            "immutable_predecessor_sha256": G0_V2_SHA256,
            "unchanged_top_level_subtrees": unchanged,
            "unchanged_top_level_subtree_count": len(unchanged),
            "only_authorized_four_choice_binding_changed": True,
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
            "g0_v3": {
                "relative_path": str(g0_v3_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(g0_v3_path),
            },
            "validator": {
                "relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "authorization": {
            "g0_v3_four_choice_binding": "COMPLETE_INDEPENDENTLY_VERIFIED",
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
    parser.add_argument("--g0-v3", type=Path, default=DEFAULT_G0_V3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(args.g0_v3)
    payload = json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
