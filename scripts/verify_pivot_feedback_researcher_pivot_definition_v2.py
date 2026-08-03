#!/usr/bin/env python3
"""Validate the threshold-free researcher-pivot definition v2 and fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_researcher_pivot_definition_v2.yaml"
)
SCHEMA = "pivot_feedback_researcher_pivot_definition.v2"
STATUS = (
    "FROZEN_THRESHOLD_FREE_OBSERVABLE_HISTORY_PIVOT_DEFINITION_V2_"
    "FIXTURES_ONLY"
)
VERIFY_SCHEMA = "pivot_feedback_researcher_pivot_definition_verification.v2"
VERIFY_STATUS = (
    "PASS_THRESHOLD_FREE_PAST_ONLY_OBSERVABLE_HISTORY_PIVOT_DEFINITION_V2"
)
STATES = [
    "NO_FOCAL_ENTRY",
    "NEW_TOPIC_ENTRY",
    "RETURN",
    "CONTINUATION",
]
REQUIRED_FEATURES = [
    "pre_entry_target_topic_work_count",
    "trailing_3y_target_topic_work_count",
    "focal_confirmed_eligible_target_topic_work_count",
]
SENSITIVITY_FIELDS = [
    "prior_observable_span_years",
    "prior_eligible_work_count",
]
FORBIDDEN_FRAGMENTS = [
    "future",
    "persistence",
    "post_entry_impact",
    "realized_future_citation",
    "peer_visible_success",
    "chaperoned_entry",
    "model_score",
    "minimum_history",
    "selected_threshold",
]


class PivotDefinitionV2Error(RuntimeError):
    """Fail-closed pivot-definition v2 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PivotDefinitionV2Error(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid contract: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise PivotDefinitionV2Error(f"invalid contract: {error}") from error
    return _mapping(value, "contract")


def _lookup_dotted(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"bound field missing: {dotted}",
        )
        current = current[part]
    return current


def _nonnegative_int(features: Mapping[str, Any], key: str) -> int:
    value = features.get(key)
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{key} must be a nonnegative integer",
    )
    return value


def classify_opportunity(features: Mapping[str, Any]) -> str:
    """Classify one author-token, primary-topic, focal-year opportunity."""

    values = _mapping(features, "features")
    lowered_keys = [str(key).lower() for key in values]
    for fragment in FORBIDDEN_FRAGMENTS:
        require(
            not any(fragment in key for key in lowered_keys),
            f"forbidden feature fragment: {fragment}",
        )
    require(
        set(values) == set(REQUIRED_FEATURES),
        "features must have the exact feature set",
    )
    prior = _nonnegative_int(values, "pre_entry_target_topic_work_count")
    trailing = _nonnegative_int(values, "trailing_3y_target_topic_work_count")
    focal = _nonnegative_int(
        values,
        "focal_confirmed_eligible_target_topic_work_count",
    )
    require(trailing <= prior, "trailing topic count exceeds pre-entry topic count")

    if focal == 0:
        return "NO_FOCAL_ENTRY"
    if prior == 0:
        return "NEW_TOPIC_ENTRY"
    if trailing == 0:
        return "RETURN"
    return "CONTINUATION"


def _validate_sensitivity(value: Any, label: str) -> dict[str, int]:
    sensitivity = _mapping(value, label)
    require(
        set(sensitivity) == set(SENSITIVITY_FIELDS),
        f"{label} must have the exact sensitivity field set",
    )
    return {
        key: _nonnegative_int(sensitivity, key)
        for key in SENSITIVITY_FIELDS
    }


def _validate_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    gate: bool,
) -> dict[str, Any]:
    binding = dict(raw)
    path = _safe_repo_path(repo_root, binding.get("relative_path"), "binding")
    require(_sha256(path) == binding.get("sha256"), f"bound SHA mismatch: {path.name}")
    document = _read_yaml(path)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        "bound schema mismatch",
    )
    require(
        document.get("status") == binding.get("required_status"),
        "bound status mismatch",
    )
    require(binding.get("mutation_allowed") is False, "bound mutation allowed")
    if gate:
        required_fields = _mapping(binding.get("required_fields"), "required fields")
        for dotted, expected in required_fields.items():
            require(
                _lookup_dotted(document, str(dotted)) == expected,
                f"bound field mismatch: {dotted}",
            )
    return {
        "relative_path": binding["relative_path"],
        "sha256": binding["sha256"],
        "status": document["status"],
    }


def _validate_contract_shape(contract: Mapping[str, Any]) -> None:
    require(contract.get("schema_version") == SCHEMA, "contract schema")
    require(contract.get("label") == "pivot_feedback_researcher_pivot_definition_v2", "contract label")
    require(
        contract.get("mode")
        == "THRESHOLD_FREE_PAST_ONLY_OBSERVABLE_HISTORY_NEW_TOPIC_ENTRY",
        "contract mode",
    )
    require(contract.get("status") == STATUS, "contract status")

    event = _mapping(contract.get("primary_event"), "primary event")
    require(event.get("unit") == "AUTHOR_TOKEN_X_PRIMARY_TOPIC_X_ENTRY_YEAR", "event unit")
    require(event.get("event_label") == "OBSERVABLE_HISTORY_NEW_TOPIC_ENTRY", "event label")
    require(event.get("pivot_state") == "NEW_TOPIC_ENTRY", "pivot state")
    require(event.get("true_lifetime_first_claim_allowed") is False, "lifetime claim")
    require(event.get("information_cutoff_for_prior_history") == "END_OF_T_MINUS_1", "history cutoff")
    require(event.get("post_focal_information_allowed") is False, "post-focal input")
    require(event.get("one_state_per_author_topic_year") is True, "event aggregation")
    require(
        event.get("repeated_new_topic_entry_for_same_author_topic_allowed") is False,
        "repeated entry",
    )

    inputs = _mapping(contract.get("classification_input"), "classification input")
    require(inputs.get("exact_required_features") == REQUIRED_FEATURES, "feature order")
    require(inputs.get("feature_types") == "NONNEGATIVE_INTEGER", "feature types")
    require(inputs.get("trailing_topic_count_must_not_exceed_pre_entry_topic_count") is True, "count relation")
    require(inputs.get("focal_content_confirmation_minimum_for_entry") == 1, "focal minimum")
    require(inputs.get("no_focal_entry_count") == 0, "no-entry count")
    require(inputs.get("minimum_prior_observable_span_years") == "NONE", "span threshold")
    require(inputs.get("minimum_prior_eligible_work_count") == "NONE", "work threshold")
    require(inputs.get("forbidden_feature_name_fragments") == FORBIDDEN_FRAGMENTS, "forbidden set")
    require(inputs.get("unknown_event_features_rejected") is True, "unknown features")

    sensitivity = _mapping(contract.get("sensitivity_only"), "sensitivity")
    require(sensitivity.get("fields") == SENSITIVITY_FIELDS, "sensitivity fields")
    require(sensitivity.get("may_change_primary_state") is False, "sensitivity gates state")
    require(sensitivity.get("threshold_specific_flags_materialized") is False, "threshold flags")
    require(sensitivity.get("primary_minimum_history_threshold") == "NONE", "history threshold")

    classification = _mapping(contract.get("classification"), "classification")
    require(classification.get("precedence") == STATES, "state precedence")
    require(set(_mapping(classification.get("rules"), "rules")) == set(STATES), "state rules")
    require(classification.get("exhaustive_for_valid_input") is True, "not exhaustive")
    require(classification.get("mutually_exclusive_for_valid_input") is True, "not exclusive")
    require(classification.get("threshold_free") is True, "not threshold free")


def _validate_valid_fixtures(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixtures = contract.get("valid_fixtures")
    require(isinstance(fixtures, list) and len(fixtures) == 7, "valid fixture count")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in fixtures:
        fixture = _mapping(raw, "valid fixture")
        fixture_id = fixture.get("fixture_id")
        require(isinstance(fixture_id, str) and fixture_id not in seen, "fixture ID")
        seen.add(fixture_id)
        features = _mapping(fixture.get("features"), f"{fixture_id} features")
        sensitivity = _validate_sensitivity(
            fixture.get("sensitivity"),
            f"{fixture_id} sensitivity",
        )
        state = classify_opportunity(features)
        require(state == fixture.get("expected_state"), f"fixture state: {fixture_id}")
        results.append(
            {
                "fixture_id": fixture_id,
                "state": state,
                "sensitivity": sensitivity,
            }
        )
    require({result["state"] for result in results} == set(STATES), "fixture state coverage")
    return results


def _validate_sensitivity_invariance(contract: Mapping[str, Any]) -> dict[str, Any]:
    fixture = _mapping(
        contract.get("sensitivity_invariance_fixture"),
        "sensitivity invariance fixture",
    )
    features = _mapping(fixture.get("event_features"), "invariance event features")
    profiles = fixture.get("sensitivity_profiles")
    require(isinstance(profiles, list) and len(profiles) == 3, "sensitivity profiles")
    state = classify_opportunity(features)
    expected = fixture.get("expected_state_for_every_profile")
    require(state == expected, "sensitivity invariance expected state")
    validated = [
        _validate_sensitivity(profile, f"sensitivity profile {index}")
        for index, profile in enumerate(profiles)
    ]
    require(
        [profile["prior_observable_span_years"] for profile in validated]
        == [0, 2, 40],
        "sensitivity span coverage",
    )
    require(
        [profile["prior_eligible_work_count"] for profile in validated]
        == [0, 2, 500],
        "sensitivity work coverage",
    )
    return {
        "profiles": len(validated),
        "state": state,
        "state_changes": 0,
    }


def _validate_invalid_fixtures(contract: Mapping[str, Any]) -> list[dict[str, str]]:
    fixtures = contract.get("invalid_fixtures")
    require(isinstance(fixtures, list) and len(fixtures) == 6, "invalid fixture count")
    results: list[dict[str, str]] = []
    for raw in fixtures:
        fixture = _mapping(raw, "invalid fixture")
        fixture_id = str(fixture.get("fixture_id"))
        fragment = fixture.get("expected_error_fragment")
        require(isinstance(fragment, str) and fragment, f"{fixture_id} error fragment")
        try:
            classify_opportunity(_mapping(fixture.get("features"), f"{fixture_id} features"))
        except PivotDefinitionV2Error as error:
            require(fragment in str(error), f"invalid fixture error: {fixture_id}")
            results.append({"fixture_id": fixture_id, "error": str(error)})
        else:
            raise PivotDefinitionV2Error(
                f"invalid fixture unexpectedly classified: {fixture_id}"
            )
    return results


def _bounded_exhaustive_replay() -> dict[str, Any]:
    counts = {state: 0 for state in STATES}
    rows = 0
    for prior in range(5):
        for trailing in range(prior + 1):
            for focal in range(3):
                features = {
                    "pre_entry_target_topic_work_count": prior,
                    "trailing_3y_target_topic_work_count": trailing,
                    "focal_confirmed_eligible_target_topic_work_count": focal,
                }
                observed = classify_opportunity(features)
                if focal == 0:
                    expected = "NO_FOCAL_ENTRY"
                elif prior == 0:
                    expected = "NEW_TOPIC_ENTRY"
                elif trailing == 0:
                    expected = "RETURN"
                else:
                    expected = "CONTINUATION"
                require(observed == expected, "bounded exhaustive replay mismatch")
                counts[observed] += 1
                rows += 1
    require(rows == 45, "bounded exhaustive row count")
    require(
        counts
        == {
            "NO_FOCAL_ENTRY": 15,
            "NEW_TOPIC_ENTRY": 2,
            "RETURN": 8,
            "CONTINUATION": 20,
        },
        "bounded exhaustive state counts",
    )
    return {"valid_count_tuples": rows, "state_counts": counts, "mismatches": 0}


def validate(
    path: Path = DEFAULT_CONTRACT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    path = path.resolve(strict=True)
    contract = _read_yaml(path)
    _validate_contract_shape(contract)
    bound = {
        "authorization_parent": _validate_binding(
            _mapping(contract.get("authorization_parent"), "authorization parent"),
            repo_root=repo_root,
            gate=True,
        ),
        "superseded_definition": _validate_binding(
            _mapping(contract.get("superseded_definition"), "superseded definition"),
            repo_root=repo_root,
            gate=False,
        ),
    }
    superseded = _mapping(contract.get("superseded_definition"), "superseded")
    require(
        superseded.get("removed_primary_gate")
        == {
            "minimum_prior_observable_span_years": 3,
            "minimum_prior_eligible_work_count": 3,
        },
        "superseded threshold",
    )
    valid_results = _validate_valid_fixtures(contract)
    invariance = _validate_sensitivity_invariance(contract)
    invalid_results = _validate_invalid_fixtures(contract)
    exhaustive = _bounded_exhaustive_replay()

    source = _mapping(contract.get("future_source_boundary"), "source boundary")
    require(source.get("pre_entry_target_topic_work_count_source") == "VERIFIED_PREDECISION_AUTHOR_MEASUREMENT", "pre-entry source")
    require(source.get("trailing_3y_target_topic_work_count_source") == "NOT_YET_CONTRACTED", "trailing source")
    require(source.get("focal_confirmed_eligible_target_topic_work_count_source") == "NOT_YET_CONTRACTED", "focal source")
    for key in (
        "definition_freeze_reads_source_rows",
        "definition_freeze_reads_focal_year_rows",
        "definition_freeze_reads_future_rows",
        "definition_freeze_creates_api_requests",
    ):
        require(source.get(key) is False, f"source boundary {key}")

    authorization = _mapping(contract.get("authorization"), "authorization")
    require(
        authorization.get("classification_fixture_execution") == "AUTHORIZED",
        "fixture execution",
    )
    for key, value in authorization.items():
        if key != "classification_fixture_execution":
            require(value == "NOT_AUTHORIZED" or value == "NOT_AUTHORIZED_BY_DEFINITION_ALONE", f"unsafe authorization: {key}")

    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "contract_sha256": _sha256(path),
        "bound": bound,
        "states": STATES,
        "required_features": REQUIRED_FEATURES,
        "valid_fixture_results": valid_results,
        "invalid_fixture_results": invalid_results,
        "sensitivity_invariance": invariance,
        "bounded_exhaustive_replay": exhaustive,
        "threshold_free": True,
        "primary_minimum_history_threshold": "NONE",
        "past_only": True,
        "observable_history_claim_only": True,
        "production_sources_read": False,
        "focal_year_rows_read": False,
        "future_rows_read": False,
        "outcomes_read": False,
        "api_requests_created": 0,
        "production_execution_authorized": False,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "claim_boundary": contract["claim_boundary"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.contract)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (PivotDefinitionV2Error, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_RESEARCHER_PIVOT_DEFINITION_V2",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
