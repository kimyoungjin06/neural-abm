#!/usr/bin/env python3
"""Validate the outcome-blind D0 measurement ABI before data execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ABI = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_measurement_abi_v1.yaml"
)
SCHEMA = "pivot_feedback_d0_measurement_abi.v1"
STATUS = "FROZEN_OUTCOME_BLIND_MEASUREMENT_ABI_EPISODE_RESELECTION_AUTHORIZED"
VERIFY_STATUS = (
    "PASS_D0_OUTCOME_BLIND_AUTHOR_LINKED_SCHOLARLY_COMMUNICATION_ABI_FROZEN"
)
EXPECTED_TYPES = {
    1: "article",
    2: "book",
    3: "book-chapter",
    4: "book-section",
    5: "database",
    6: "dataset",
    7: "dissertation",
    8: "editorial",
    9: "erratum",
    10: "grant",
    11: "letter",
    12: "libguides",
    13: "other",
    14: "paratext",
    15: "peer-review",
    16: "preprint",
    17: "reference-entry",
    18: "report",
    19: "report-component",
    20: "retraction",
    21: "review",
    22: "software",
    23: "standard",
    24: "supplementary-materials",
}
EXPECTED_INCLUDED = {
    "article",
    "book",
    "book-chapter",
    "book-section",
    "dissertation",
    "preprint",
    "report",
    "review",
}
EXPECTED_OFFSETS = {
    "baseline_offsets": [-6, -5, -4],
    "recent_offsets": [-3, -2, -1],
    "response_offsets_reserved_for_independence_only": [0, 1, 2],
}


class MeasurementABIError(RuntimeError):
    """Fail-closed D0 measurement-ABI validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementABIError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing {dotted_path}",
        )
        current = current[part]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, *, yaml_document: bool) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid document: {path}")
    try:
        value = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_document
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise MeasurementABIError(f"invalid document {path.name}: {error}") from error
    return _mapping(value, path.name)


def _safe_repo_path(repo_root: Path, value: Any) -> Path:
    require(isinstance(value, str) and value, "bound path must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe bound path")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), "bound path escapes repository")
    require(path.is_file() and not path.is_symlink(), "invalid bound path")
    return path


def _validate_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    yaml_document: bool,
) -> dict[str, Any]:
    binding = dict(raw)
    path = _safe_repo_path(repo_root, binding.get("relative_path"))
    digest = _sha256(path)
    require(digest == binding.get("sha256"), f"bound SHA mismatch: {path.name}")
    document = _read(path, yaml_document=yaml_document)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        f"bound schema mismatch: {path.name}",
    )
    require(
        document.get("status") == binding.get("required_status"),
        f"bound status mismatch: {path.name}",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        "required fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"bound field mismatch: {dotted_path}",
        )
    return {
        "relative_path": binding["relative_path"],
        "sha256": digest,
        "status": document["status"],
    }


def validate(
    path: Path = DEFAULT_ABI,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    path = path.resolve(strict=True)
    abi = _read(path, yaml_document=True)
    require(abi.get("schema_version") == SCHEMA, "ABI schema mismatch")
    require(abi.get("label") == "pivot_feedback_d0_measurement_abi_v1", "ABI label mismatch")
    require(
        abi.get("mode")
        == "OUTCOME_BLIND_AUTHOR_LINKED_SCHOLARLY_COMMUNICATION_ATTENTION_ABI",
        "ABI mode mismatch",
    )
    require(abi.get("status") == STATUS, "ABI status mismatch")

    bindings = _mapping(abi.get("repository_bindings"), "repository bindings")
    require(set(bindings) == {"gate_state", "linkage_audit"}, "binding set mismatch")
    bound = {
        "gate_state": _validate_binding(
            _mapping(bindings["gate_state"], "gate binding"),
            repo_root=repo_root,
            yaml_document=True,
        ),
        "linkage_audit": _validate_binding(
            _mapping(bindings["linkage_audit"], "linkage binding"),
            repo_root=repo_root,
            yaml_document=False,
        ),
    }

    semantic = _mapping(abi.get("semantic_adjudication"), "semantic adjudication")
    require(semantic.get("decision_date") == "2026-07-24", "decision date mismatch")
    require(
        semantic.get("reference_role")
        == "SEMANTIC_GUIDANCE_ONLY_FROZEN_RULE_BELOW_IS_THE_EXECUTABLE_CONTRACT",
        "external reference improperly executable",
    )
    for key in (
        "source_snapshot_vocabulary_is_older_than_current_reference",
        "not_a_quality_filter",
        "not_selected_from_entry_or_persistence_outcomes",
        "not_selected_to_preserve_historical_episode_ids",
    ):
        require(semantic.get(key) is True, f"semantic boundary missing: {key}")

    raw_dictionary = _mapping(abi.get("work_type_dictionary"), "work type dictionary")
    dictionary = {int(code): str(name) for code, name in raw_dictionary.items()}
    require(dictionary == EXPECTED_TYPES, "work type dictionary mismatch")
    eligibility = _mapping(abi.get("work_type_eligibility"), "work type eligibility")
    included = _mapping(eligibility.get("included"), "included work types")
    excluded = _mapping(eligibility.get("excluded"), "excluded work types")
    require(set(included) == EXPECTED_INCLUDED, "included work type set mismatch")
    require(
        set(excluded) == set(EXPECTED_TYPES.values()) - EXPECTED_INCLUDED,
        "excluded work type set mismatch",
    )
    require(not (set(included) & set(excluded)), "work type appears in both dispositions")
    require(
        eligibility.get("every_snapshot_type_must_have_exactly_one_disposition")
        is True,
        "work type closure not required",
    )

    contract = _mapping(abi.get("measurement_abi"), "measurement ABI")
    require(contract.get("selected_field_ids") == [17, 27], "selected fields differ")
    require(contract.get("field_reselection_under_work_abi") is False, "field scope reopened")
    require(contract.get("allowed_temporal_dispositions") == [2, 3], "temporal scope differs")
    require(contract.get("latest_attention_year") == 2011, "attention cutoff differs")
    expected_rules = {
        "null_year": "EXCLUDE",
        "null_primary_topic_or_field": "EXCLUDE",
        "work_type_rule": "INCLUDE_ONLY_WORK_TYPE_ELIGIBILITY_INCLUDED_KEYS",
        "paratext_rule": "EXCLUDE_IF_WORK_TYPE_PARATEXT_OR_IS_PARATEXT_TRUE",
        "paratext_type_boolean_mismatch": (
            "LEGACY_SNAPSHOT_EXPECTED_REPORT_EXACT_CROSSWALK"
        ),
        "retraction_rule": "EXCLUDE_IF_WORK_TYPE_RETRACTION_OR_IS_RETRACTED_TRUE",
        "is_retracted_null_rule": "RETAIN_IF_OTHERWISE_ELIGIBLE_AND_REPORT_EXPLICITLY",
        "valid_author_linkage_rule": (
            "AT_LEAST_ONE_DISTINCT_CURRENT_SNAPSHOT_OPENALEX_AUTHOR_TOKEN_WORK_EVENT"
        ),
        "zero_valid_author_token_rule": (
            "EXCLUDE_FROM_PRIMARY_ATTENTION_RETAIN_IN_MISSINGNESS_ACCOUNTING"
        ),
        "imputation": "NONE",
        "inverse_probability_weighting": "NONE",
        "global_linkage_coverage_factor": "FORBIDDEN",
        "team_cap_for_attention": "NONE",
    }
    for key, expected in expected_rules.items():
        require(contract.get(key) == expected, f"measurement rule mismatch: {key}")
    crosswalk = _mapping(
        contract.get("paratext_type_boolean_crosswalk_audit"),
        "paratext crosswalk audit",
    )
    require(
        crosswalk
        == {
            "full_cutoff_rows_mismatched": 2_307_929,
            "selected_field_predecision_topic_rows_mismatched": 23_342,
            "selected_scope_type_paratext_boolean_false_rows": 20_220,
            "selected_scope_non_paratext_type_boolean_true_rows": 3_122,
            "source_inputs_only": True,
            "author_linkage_or_outcomes_used": False,
        },
        "paratext crosswalk audit mismatch",
    )
    require(
        contract.get("author_event_year_field_and_disposition_must_match_work_metadata")
        is True,
        "author-work identity matching weakened",
    )
    require(
        contract.get("primary_topic_field_must_match_taxonomy") is True,
        "taxonomy matching weakened",
    )

    selection = _mapping(abi.get("episode_reselection"), "episode reselection")
    require(selection.get("entry_year_end_inclusive") == 2012, "entry cutoff differs")
    for key, expected in EXPECTED_OFFSETS.items():
        require(selection.get(key) == expected, f"selection offsets differ: {key}")
    pair = _mapping(selection.get("pair_selection"), "pair selection")
    require(pair.get("episodes_per_field") == 2, "episode count differs")
    require(pair.get("require_distinct_subfield_ids") is True, "subfield independence weakened")
    require(
        pair.get("require_disjoint_baseline_signal_and_reserved_response_windows")
        is True,
        "calendar independence weakened",
    )
    require(pair.get("minimum_entry_year_separation") == 9, "calendar separation differs")
    require(
        selection.get("historical_episode_ids_are_not_selection_inputs") is True,
        "historical episodes used as selection input",
    )
    require(
        selection.get("fail_if_two_field_four_episode_coverage_not_met") is True,
        "coverage failure not fail-closed",
    )

    forbidden = _mapping(
        abi.get("forbidden_inputs_and_outputs"),
        "forbidden inputs and outputs",
    )
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden state opened")
    output = _mapping(abi.get("output_contract"), "output contract")
    require(output.get("abi_validation_status") == VERIFY_STATUS, "ABI output status differs")
    for key in (
        "d0_complete_after_reselection",
        "leaveout_network_bridge_authorized_after_reselection_alone",
        "discovery_outcome_opening_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(output.get(key) is False, f"unsafe output authorization: {key}")
    require(
        output.get("content_and_indexing_audit_required_after_reselection") is True,
        "content/indexing audit not required",
    )

    code_by_name = {name: code for code, name in dictionary.items()}
    included_codes = sorted(code_by_name[name] for name in included)
    return {
        "schema_version": "pivot_feedback_d0_measurement_abi_verification.v1",
        "status": VERIFY_STATUS,
        "abi_relative_path": path.relative_to(repo_root).as_posix(),
        "abi_sha256": _sha256(path),
        "repository_bindings": bound,
        "included_work_type_codes": included_codes,
        "included_work_type_names": [dictionary[code] for code in included_codes],
        "excluded_work_type_codes": [
            code for code in sorted(dictionary) if code not in included_codes
        ],
        "selected_field_ids": contract["selected_field_ids"],
        "latest_attention_year": contract["latest_attention_year"],
        "valid_author_linkage_required": True,
        "outcomes_read": False,
        "episode_reselection_authorized": True,
        "leaveout_network_bridge_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abi", type=Path, default=DEFAULT_ABI)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(validate(args.abi), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
