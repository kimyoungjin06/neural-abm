#!/usr/bin/env python3
"""Validate and execute the simple past-only researcher-pivot definition v1."""

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
    / "experiments/configs/pivot_feedback_researcher_pivot_definition_v1.yaml"
)
SCHEMA = "pivot_feedback_researcher_pivot_definition.v1"
STATUS = "FROZEN_SIMPLE_PAST_ONLY_PIVOT_DEFINITION_EXECUTION_NOT_AUTHORIZED"
VERIFY_STATUS = "PASS_SIMPLE_PAST_ONLY_RESEARCHER_PIVOT_DEFINITION_FROZEN"
STATES = [
    "INSUFFICIENT_HISTORY",
    "NEW_TOPIC_PIVOT",
    "RETURN",
    "CONTINUATION",
]
REQUIRED_FEATURES = [
    "prior_observable_span_years",
    "prior_eligible_work_count",
    "prior_topic_work_count",
    "trailing_3y_topic_work_count",
    "focal_confirmed_on_topic_work_count",
]
FORBIDDEN_FRAGMENTS = [
    "future",
    "persistence",
    "post_pivot",
    "realized_citation",
    "impact_outcome",
    "model_score",
]


class PivotDefinitionError(RuntimeError):
    """Fail-closed pivot-definition validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PivotDefinitionError(message)


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
        raise PivotDefinitionError(f"invalid contract: {error}") from error
    return _mapping(value, "contract")


def _nonnegative_int(features: Mapping[str, Any], key: str) -> int:
    value = features.get(key)
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{key} must be a nonnegative integer",
    )
    return value


def classify_opportunity(features: Mapping[str, Any]) -> str:
    """Classify one validated researcher-primary-topic-year opportunity."""

    lowered_keys = [str(key).lower() for key in features]
    for fragment in FORBIDDEN_FRAGMENTS:
        require(
            not any(fragment in key for key in lowered_keys),
            f"future or outcome feature forbidden: {fragment}",
        )
    missing = [key for key in REQUIRED_FEATURES if key not in features]
    require(not missing, f"missing required features: {missing}")

    span = _nonnegative_int(features, "prior_observable_span_years")
    prior_works = _nonnegative_int(features, "prior_eligible_work_count")
    prior_topic = _nonnegative_int(features, "prior_topic_work_count")
    trailing_topic = _nonnegative_int(features, "trailing_3y_topic_work_count")
    focal_topic = _nonnegative_int(
        features,
        "focal_confirmed_on_topic_work_count",
    )

    require(focal_topic >= 1, "focal content confirmation missing")
    require(
        trailing_topic <= prior_topic,
        "trailing topic count exceeds prior topic count",
    )
    require(
        prior_topic > 0 or trailing_topic == 0,
        "zero prior topic count conflicts with trailing topic count",
    )

    if span < 3 or prior_works < 3:
        return "INSUFFICIENT_HISTORY"
    if prior_topic == 0:
        return "NEW_TOPIC_PIVOT"
    if trailing_topic == 0:
        return "RETURN"
    return "CONTINUATION"


def _validate_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    markdown: bool,
) -> dict[str, Any]:
    binding = dict(raw)
    path = _safe_repo_path(
        repo_root,
        binding.get("relative_path"),
        "bound repository path",
    )
    digest = _sha256(path)
    require(digest == binding.get("sha256"), f"bound SHA mismatch: {path.name}")
    if markdown:
        text = path.read_text(encoding="utf-8")
        required = binding.get("required_text")
        require(isinstance(required, list) and required, "required text missing")
        for fragment in required:
            require(
                isinstance(fragment, str) and fragment in text,
                f"bound protocol text missing: {fragment}",
            )
        observed_status = "TEXT_BOUND"
    else:
        document = _read_yaml(path)
        require(
            document.get("schema_version") == binding.get("required_schema_version"),
            "bound schema mismatch",
        )
        require(
            document.get("status") == binding.get("required_status"),
            "bound status mismatch",
        )
        require(binding.get("mutation_allowed") is False, "parent mutation allowed")
        observed_status = str(document["status"])
    return {
        "relative_path": binding["relative_path"],
        "sha256": digest,
        "status": observed_status,
    }


def validate(
    path: Path = DEFAULT_CONTRACT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    path = path.resolve(strict=True)
    contract = _read_yaml(path)
    require(contract.get("schema_version") == SCHEMA, "contract schema mismatch")
    require(
        contract.get("label") == "pivot_feedback_researcher_pivot_definition_v1",
        "contract label mismatch",
    )
    require(
        contract.get("mode") == "SIMPLE_PAST_ONLY_OBSERVABLE_HISTORY_NEW_TOPIC_ENTRY",
        "contract mode mismatch",
    )
    require(contract.get("status") == STATUS, "contract status mismatch")

    bindings = _mapping(contract.get("repository_bindings"), "bindings")
    require(
        set(bindings) == {"empirical_protocol", "gate_state_parent"},
        "binding set mismatch",
    )
    bound = {
        "empirical_protocol": _validate_binding(
            _mapping(bindings["empirical_protocol"], "protocol binding"),
            repo_root=repo_root,
            markdown=True,
        ),
        "gate_state_parent": _validate_binding(
            _mapping(bindings["gate_state_parent"], "gate binding"),
            repo_root=repo_root,
            markdown=False,
        ),
    }

    event = _mapping(contract.get("primary_event"), "primary event")
    require(
        event.get("unit") == "RESEARCHER_PRIMARY_TOPIC_ENTRY_YEAR",
        "event unit mismatch",
    )
    require(
        event.get("claim_label") == "OBSERVABLE_HISTORY_NEW_TOPIC_PIVOT",
        "claim label mismatch",
    )
    require(event.get("true_lifetime_first_claim_allowed") is False, "lifetime claim open")
    require(
        event.get("information_cutoff_for_history") == "END_OF_T_MINUS_1",
        "history cutoff mismatch",
    )
    require(event.get("post_focal_information_allowed") is False, "future opened")
    require(
        event.get("repeated_pivot_for_same_researcher_topic_allowed") is False,
        "repeated pivots allowed",
    )

    inputs = _mapping(contract.get("input_contract"), "input contract")
    require(inputs.get("required_features") == REQUIRED_FEATURES, "feature set mismatch")
    require(inputs.get("feature_types") == "NONNEGATIVE_INTEGER", "feature type mismatch")
    require(inputs.get("focal_content_confirmation_required") is True, "content unconfirmed")
    require(
        inputs.get("focal_confirmed_on_topic_work_count_minimum") == 1,
        "focal minimum mismatch",
    )
    require(
        inputs.get("forbidden_feature_name_fragments") == FORBIDDEN_FRAGMENTS,
        "forbidden feature set mismatch",
    )

    history = _mapping(contract.get("history_eligibility"), "history eligibility")
    require(
        history.get("minimum_prior_observable_span_years") == 3,
        "history span mismatch",
    )
    require(
        history.get("minimum_prior_eligible_work_count") == 3,
        "prior work minimum mismatch",
    )
    require(
        history.get("eligibility_is_not_part_of_pivot_novelty") is True,
        "eligibility and novelty conflated",
    )

    classification = _mapping(contract.get("classification"), "classification")
    require(classification.get("precedence") == STATES, "state precedence mismatch")
    require(
        set(_mapping(classification.get("rules"), "rules")) == set(STATES),
        "state rule set mismatch",
    )
    require(classification.get("exhaustive_for_valid_input") is True, "not exhaustive")
    require(
        classification.get("mutually_exclusive_for_valid_input") is True,
        "states overlap",
    )

    truth_table = {
        "insufficient_span": classify_opportunity(
            {
                "prior_observable_span_years": 2,
                "prior_eligible_work_count": 10,
                "prior_topic_work_count": 0,
                "trailing_3y_topic_work_count": 0,
                "focal_confirmed_on_topic_work_count": 1,
            }
        ),
        "insufficient_works": classify_opportunity(
            {
                "prior_observable_span_years": 9,
                "prior_eligible_work_count": 2,
                "prior_topic_work_count": 0,
                "trailing_3y_topic_work_count": 0,
                "focal_confirmed_on_topic_work_count": 1,
            }
        ),
        "new_topic": classify_opportunity(
            {
                "prior_observable_span_years": 9,
                "prior_eligible_work_count": 10,
                "prior_topic_work_count": 0,
                "trailing_3y_topic_work_count": 0,
                "focal_confirmed_on_topic_work_count": 1,
            }
        ),
        "return": classify_opportunity(
            {
                "prior_observable_span_years": 9,
                "prior_eligible_work_count": 10,
                "prior_topic_work_count": 2,
                "trailing_3y_topic_work_count": 0,
                "focal_confirmed_on_topic_work_count": 1,
            }
        ),
        "continuation": classify_opportunity(
            {
                "prior_observable_span_years": 9,
                "prior_eligible_work_count": 10,
                "prior_topic_work_count": 2,
                "trailing_3y_topic_work_count": 1,
                "focal_confirmed_on_topic_work_count": 1,
            }
        ),
    }
    require(
        truth_table
        == {
            "insufficient_span": "INSUFFICIENT_HISTORY",
            "insufficient_works": "INSUFFICIENT_HISTORY",
            "new_topic": "NEW_TOPIC_PIVOT",
            "return": "RETURN",
            "continuation": "CONTINUATION",
        },
        "decision table execution mismatch",
    )

    magnitude = _mapping(contract.get("pivot_magnitude"), "pivot magnitude")
    require(
        magnitude.get("role") == "SEPARATE_CONTINUOUS_ATTRIBUTE_NOT_EVENT_GATE",
        "distance role mismatch",
    )
    require(
        magnitude.get("primary_measure")
        == "ONE_MINUS_COSINE_REFERENCE_SOURCE_DISTRIBUTION",
        "distance measure mismatch",
    )
    require(magnitude.get("researcher_baseline_window_years") == 3, "distance window")
    require(magnitude.get("valid_range_inclusive") == [0.0, 1.0], "distance range")
    require(magnitude.get("threshold_for_primary_event") == "NONE", "distance gates event")

    downstream = _mapping(contract.get("downstream_outcomes"), "downstream outcomes")
    require(downstream.get("persistence_window_offsets") == [1, 2, 3], "persistence window")
    require(downstream.get("persistence_is_future_outcome") is True, "persistence not future")
    require(
        downstream.get("outcome_opening_authorized_by_definition_freeze") is False,
        "outcomes opened",
    )

    api = _mapping(contract.get("api_only_content_boundary"), "API-only boundary")
    require(api.get("human_review") == "NOT_RUN_BY_DESIGN", "human state mismatch")
    require(api.get("human_accuracy") == "NOT_AVAILABLE", "human accuracy invented")
    require(api.get("human_cohen_kappa") == "NOT_AVAILABLE", "human kappa invented")
    require(api.get("api_consensus_is_ground_truth") is False, "API called ground truth")
    require(api.get("current_frozen_title_sample_rows") == 384, "sample size mismatch")
    require(api.get("current_strict_dual_high_on_topic_rows") == 197, "strict rows mismatch")
    require(api.get("current_unresolved_quarantined_rows") == 187, "quarantine mismatch")
    require(api.get("current_sample_is_candidate_universe") is False, "sample promoted")
    require(api.get("complete_candidate_universe_screen_run") is False, "full screen claimed")
    require(api.get("unresolved_rows_must_be_quarantined") is True, "quarantine disabled")
    require(
        api.get("definition_freeze_authorizes_paid_api_execution") is False,
        "paid API execution opened",
    )

    authorization = _mapping(contract.get("authorization"), "authorization")
    require(
        authorization.get("classification_fixture_execution") == "AUTHORIZED",
        "fixture execution closed",
    )
    for key in (
        "production_pivot_classification",
        "complete_candidate_universe_api_screen",
        "episode_reselection",
        "author_opportunity_panel_rebuild",
        "persistence_outcome_read",
        "realized_citation_outcome_read",
        "model_fitting",
        "g4_open",
    ):
        require(
            authorization.get(key) == "NOT_AUTHORIZED",
            f"unsafe authorization: {key}",
        )

    return {
        "schema_version": "pivot_feedback_researcher_pivot_definition_verification.v1",
        "status": VERIFY_STATUS,
        "contract_sha256": _sha256(path),
        "bound": bound,
        "states": STATES,
        "truth_table": truth_table,
        "past_only": True,
        "observable_history_claim_only": True,
        "pivot_distance_gates_event": False,
        "persistence_is_future_outcome": True,
        "human_review": "NOT_RUN_BY_DESIGN",
        "human_accuracy": "NOT_AVAILABLE",
        "candidate_universe_screen_complete": False,
        "production_execution_authorized": False,
        "outcomes_read": False,
        "g4_open": False,
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
    except (PivotDefinitionError, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_RESEARCHER_PIVOT_DEFINITION_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
