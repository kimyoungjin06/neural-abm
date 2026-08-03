#!/usr/bin/env python3
"""Independently verify the frozen G0 measurement/proxy precommitment."""

from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_measurement_proxy_precommitment_v1.yaml"
)
DEFAULT_FIXTURE_INPUT = (
    REPO_ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_measurement_proxy_discovery_v1.input.json"
)
DEFAULT_FIXTURE_EXPECTED = (
    REPO_ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_measurement_proxy_discovery_v1.expected.json"
)
APPLICATION_TOOL = (
    REPO_ROOT
    / "scripts/apply_pivot_feedback_g0_measurement_proxy_precommitment_v1.py"
)
CONTRACT_SCHEMA = "pivot_feedback_g0_measurement_proxy_precommitment.v1"
CONTRACT_STATUS = (
    "FROZEN_FOUR_MEASUREMENT_PROXY_CHOICES_PENDING_INDEPENDENT_VALIDATION_"
    "NO_G0_PROMOTION"
)
VERIFY_SCHEMA = "pivot_feedback_g0_measurement_proxy_precommitment_summary.v1"
VERIFY_STATUS = (
    "PASS_FOUR_OUTCOME_BLIND_MEASUREMENT_PROXY_CHOICES_AND_CLOSED_FIXTURE"
)
EXPECTED_BINDINGS = {
    "authorization_parent": {
        "path": "experiments/configs/pivot_feedback_gate_state_v57.yaml",
        "sha256": (
            "44b35596624a13e58e5928355aea94221fc684d63b135ce81fa4482da452f3ce"
        ),
        "schema": "pivot_feedback_gate_state.v57",
        "status": (
            "VALID_PRODUCTION_INPUT_RESOLUTION_NO_MATERIALIZED_G4_"
            "ARTIFACT_SET_NO_SEAL"
        ),
        "format": "yaml",
    },
    "peer_mechanism_registry": {
        "path": "experiments/configs/pivot_feedback_peer_mechanism_registry_v0.yaml",
        "sha256": (
            "99eeb68b632f9f9a250f50320cfc7a293e3595de81984419d1528e2ee12aa6f8"
        ),
        "schema": "pivot_feedback_peer_mechanism_registry.v1",
        "status": "PENDING_G0_MEASUREMENT_FREEZE",
        "format": "yaml",
    },
    "cross_field_matrix": {
        "path": "experiments/configs/pivot_feedback_cross_field_matrix_v0.yaml",
        "sha256": (
            "6fd3e2c7aae7a7aac085ed73ba4bf3d64a91c02d60dce8a9a6a262591a0c934a"
        ),
        "schema": "pivot_feedback_cross_field_matrix.v1",
        "status": None,
        "format": "yaml",
    },
    "g0_v2": {
        "path": "experiments/configs/pivot_feedback_g0_measurement_freeze_v2.yaml",
        "sha256": (
            "a2cb0cbf444a7c128eecc40a815b80d5fe86f617a36849330612491b03934eb3"
        ),
        "schema": "pivot_feedback_g0_measurement_freeze.v3",
        "status": "PENDING_G0_V2_G_MINUS_1A_BOUND_THREE_BLOCKERS_NO_SEAL",
        "format": "yaml",
    },
    "persistence_outcome_contract": {
        "path": (
            "experiments/configs/"
            "pivot_feedback_d0_persistence_outcome_contract_v1.yaml"
        ),
        "sha256": (
            "177fd11366c36143f601743a27afb952b1482c2f90bc536b95787e114c482b6c"
        ),
        "schema": "pivot_feedback_d0_persistence_outcome_contract.v1",
        "status": "FROZEN_D0_PERSISTENCE_OUTCOME_SOURCE_CONTRACT_FIXTURES_ONLY",
        "format": "yaml",
    },
    "citation_chronology_summary": {
        "path": (
            "experiments/evidence/results/"
            "openalex_g1_citation_chronology_production_full_20260722.summary.json"
        ),
        "sha256": (
            "79cfca7ce217a25e682ef617c26d42d234fe84f5f4a0fafdae19d7d16c99893a"
        ),
        "schema": "openalex_g1_citation_chronology_production_full_summary.v1",
        "status": (
            "PASS_CURRENT_SNAPSHOT_CITATION_EDGE_CHRONOLOGY_ACCOUNTING_WITH_"
            "CANONICAL_TEMPORAL_V3_PARENT_AND_BIBLIOGRAPHIC_EVENT_TIME_"
            "CONVENTION_LIMITATION"
        ),
        "format": "json",
    },
    "empirical_protocol": {
        "path": (
            "docs/case-studies/researcher-pivot/"
            "pivot-feedback-empirical-nabm-protocol.md"
        ),
        "sha256": (
            "731e39dec4c7be5c29a0f0404b48992ad0caef152062d1d1d7876acc07064746"
        ),
        "schema": None,
        "status": None,
        "format": "text",
    },
}
CHOICE_IDS = [
    "attention_surge_threshold",
    "public_success_threshold",
    "citation_maturation_lag_years",
    "negative_control_equivalence_margins",
]
EQUIVALENCE_CONTROLS = [
    "lead_attention",
    "same_peer_unrelated_topic_success",
    "post_decision_future_citation_success",
    "future_tie",
    "matched_non_surge_topic",
]
CONTRAST_CONTROLS = [
    "degree_preserving_shuffled_network",
    "matched_non_neighbor",
]


class MeasurementProxyPrecommitmentError(RuntimeError):
    """Fail-closed precommitment verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementProxyPrecommitmentError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} path")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise MeasurementProxyPrecommitmentError(f"invalid {label}: {error}") from error


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MeasurementProxyPrecommitmentError(f"invalid {label}: {error}") from error


def _binding_spec(contract: Mapping[str, Any], key: str) -> dict[str, Any]:
    if key == "authorization_parent":
        return _mapping(contract.get(key), key)
    sources = _mapping(contract.get("bound_design_sources"), "bound sources")
    return _mapping(sources.get(key), key)


def _validate_binding(contract: Mapping[str, Any], key: str) -> Any:
    declared = _binding_spec(contract, key)
    expected = EXPECTED_BINDINGS[key]
    require(declared.get("relative_path") == expected["path"], f"{key} path")
    require(declared.get("sha256") == expected["sha256"], f"{key} hash")
    require(declared.get("mutation_allowed") is False, f"{key} mutation")
    path = _safe_repo_path(declared["relative_path"], key)
    require(_sha256(path) == expected["sha256"], f"{key} byte hash")
    if expected["format"] == "text":
        return path.read_text(encoding="utf-8")
    document = (
        _read_yaml(path, key)
        if expected["format"] == "yaml"
        else _read_json(path, key)
    )
    require(
        declared.get("required_schema_version") == expected["schema"],
        f"{key} required schema",
    )
    require(document.get("schema_version") == expected["schema"], f"{key} schema")
    if expected["status"] is not None:
        require(
            declared.get("required_status") == expected["status"],
            f"{key} required status",
        )
        require(document.get("status") == expected["status"], f"{key} status")
    return document


def _validate_source_semantics(documents: Mapping[str, Any]) -> dict[str, Any]:
    g0 = _mapping(documents["g0_v2"], "G0 v2")
    unresolved = _mapping(
        _mapping(g0.get("resolution"), "G0 resolution").get(
            "unresolved_quantitative_choices"
        ),
        "G0 unresolved choices",
    )
    for choice_id in CHOICE_IDS:
        require(unresolved.get(choice_id) is None, f"predecessor choice {choice_id}")

    peer = _mapping(documents["peer_mechanism_registry"], "peer registry")
    controls = peer.get("negative_controls")
    require(isinstance(controls, list), "peer negative controls")
    observed_ids = [item.get("control_id") for item in controls]
    require(set(observed_ids) == set(EQUIVALENCE_CONTROLS + CONTRAST_CONTROLS), "control IDs")
    by_id = {item["control_id"]: item for item in controls}
    for control_id in EQUIVALENCE_CONTROLS:
        require(
            by_id[control_id].get("required_gate")
            == "EQUIVALENCE_TO_FROZEN_NULL_REGION",
            f"equivalence gate {control_id}",
        )
    for control_id in CONTRAST_CONTROLS:
        require(
            by_id[control_id].get("required_gate")
            == "TRUE_MINUS_PLACEBO_CONTRAST_PASSES_FROZEN_G0_RULE",
            f"contrast gate {control_id}",
        )

    persistence = _mapping(documents["persistence_outcome_contract"], "persistence")
    maturation = _mapping(persistence.get("maturation_and_censoring"), "maturation")
    definition = _mapping(persistence.get("persistence_definition"), "persistence definition")
    require(maturation.get("matured_rule") == "entry_year + 3 <= 2025", "matured rule")
    require(definition.get("outcome_window_offsets") == [1, 2, 3], "persistence offsets")

    citation = _mapping(documents["citation_chronology_summary"], "citation summary")
    boundary = str(citation.get("claim_boundary", ""))
    require("current-snapshot proxy" in boundary, "citation proxy boundary")
    require("not evidence of when OpenAlex first observed" in boundary, "observation boundary")

    protocol = documents["empirical_protocol"]
    require("incumbent-only attention acceleration" in protocol, "attention protocol")
    require("G0 fixes a smallest effect of interest" in protocol, "equivalence protocol")
    require("registered equivalence" in protocol, "registered equivalence boundary")
    require("not merely a non-significant null" in protocol, "null-test boundary")
    return {
        "predecessor_choices_null": CHOICE_IDS,
        "negative_control_ids": observed_ids,
        "persistence_horizon_years": 3,
        "citation_current_snapshot_proxy_only": True,
    }


def _validate_choices(contract: Mapping[str, Any]) -> dict[str, Any]:
    scope = _mapping(contract.get("scope"), "scope")
    require(scope.get("choice_ids") == CHOICE_IDS, "choice order")
    require(scope.get("choice_count") == 4, "choice count")
    for key in (
        "selection_uses_confirmatory_or_holdout_outcomes",
        "selection_uses_g4_field_identity",
        "selection_uses_model_performance",
    ):
        require(scope.get(key) is False, f"selection boundary {key}")
    choices = _mapping(contract.get("precommitment_choices"), "choices")
    require(set(choices) == set(CHOICE_IDS), "choice keys")

    attention = _mapping(choices["attention_surge_threshold"], "attention")
    require(
        attention.get("rule_id")
        == "ROBUST_INCUMBENT_ATTENTION_ACCELERATION_MAD_V1",
        "attention rule",
    )
    require(attention.get("recent_window_offsets") == [-3, -2, -1], "recent offsets")
    require(attention.get("reference_window_offsets") == [-6, -5, -4], "reference offsets")
    require(attention.get("primary_modified_z_threshold_inclusive") == 3.0, "z threshold")
    require(attention.get("mad_zero_action") == "NO_SURGE_FAIL_CLOSED", "MAD zero")
    require(attention.get("primary_threshold_may_be_relaxed_for_coverage") is False, "coverage relaxation")

    public = _mapping(choices["public_success_threshold"], "public success")
    require(public.get("proxy_included") is True, "public proxy inclusion")
    require(public.get("target_fraction") == 0.10, "top-decile fraction")
    require(public.get("boundary_tie_rule") == "INCLUDE_ALL_WORKS_AT_OR_ABOVE_CUTOFF", "tie rule")
    require(public.get("exact_ten_percent_membership_claim_allowed") is False, "exact-decile claim")
    require(public.get("historical_platform_observation_time_established") is False, "platform time")
    require(public.get("researcher_visibility_established") is False, "visibility")

    maturation = _mapping(choices["citation_maturation_lag_years"], "maturation")
    require(maturation.get("value") == 3, "maturation value")
    require(maturation.get("current_snapshot_bibliographic_time_proxy_only") is True, "maturation proxy")
    require(maturation.get("historical_first_observation_time_claim_allowed") is False, "historical claim")

    margins = _mapping(choices["negative_control_equivalence_margins"], "margins")
    require(margins.get("symmetric_practical_null_lower") == -0.01, "lower margin")
    require(margins.get("symmetric_practical_null_upper") == 0.01, "upper margin")
    require(margins.get("equivalence_test") == "TWO_ONE_SIDED_TESTS", "TOST")
    require(margins.get("one_sided_alpha") == 0.05, "alpha")
    require(margins.get("equivalent_confidence_interval") == "TWO_SIDED_90_PERCENT", "CI")
    require(margins.get("equivalence_control_ids") == EQUIVALENCE_CONTROLS, "equivalence controls")
    require(margins.get("true_minus_placebo_control_ids") == CONTRAST_CONTROLS, "contrast controls")
    require(margins.get("nonsignificant_zero_test_counts_as_equivalence") is False, "null test")

    rationale = _mapping(contract.get("methodological_rationale"), "rationale")
    policy = _mapping(rationale.get("policy_value_boundary"), "policy boundary")
    require(all(value is False for key, value in policy.items() if key.startswith("literature_mandates")), "literature mandate boundary")
    require(policy.get("values_are_precommitted_design_policy_not_empirical_truth") is True, "policy truth boundary")
    return {
        "choice_count": 4,
        "attention_modified_z_threshold": 3.0,
        "public_success_target_fraction": 0.10,
        "citation_maturation_lag_years": 3,
        "negative_control_practical_null": [-0.01, 0.01],
        "tost_one_sided_alpha": 0.05,
    }


def _independent_fixture_replay(
    input_document: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    require(
        input_document.get("schema_version")
        == "pivot_feedback_g0_measurement_proxy_discovery_input.v1",
        "fixture input schema",
    )
    require(input_document.get("status") == "SYNTHETIC_CLOSED_FIXTURE", "fixture status")
    require(
        expected.get("schema_version")
        == "pivot_feedback_g0_measurement_proxy_discovery_output.v1",
        "fixture output schema",
    )
    require(
        expected.get("status")
        == "PASS_FROZEN_MEASUREMENT_PROXY_RULE_APPLICATION_DISCOVERY_ONLY",
        "fixture output status",
    )

    attention_groups: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for row in input_document["attention_topic_year_rows"]:
        require(type(row["decision_year"]) is int and row["decision_year"] <= 2012, "fixture discovery year")
        attention_groups[(row["field_id"], row["decision_year"])].append(
            (
                row["topic_id"],
                row["recent_incumbent_attention_count"],
                row["reference_incumbent_attention_count"],
            )
        )
    independent_surge_count = 0
    independent_mad_zero_count = 0
    for values in attention_groups.values():
        scores = [math.log1p(recent / 3) - math.log1p(reference / 3) for _, recent, reference in values]
        median = statistics.median(scores)
        mad = statistics.median(abs(score - median) for score in scores)
        independent_mad_zero_count += int(mad == 0)
        if mad != 0:
            for (_, recent, reference), score in zip(values, scores, strict=True):
                z_value = 0.6744897501960817 * (score - median) / mad
                independent_surge_count += int(z_value >= 3.0 and score > 0 and recent > reference)
    observed_attention = _mapping(expected.get("attention_surge_application"), "expected attention")
    require(observed_attention.get("primary_surge_count") == independent_surge_count, "fixture surge replay")
    require(observed_attention.get("mad_zero_group_count") == independent_mad_zero_count, "fixture MAD-zero replay")

    citation_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for row in input_document["citation_work_rows"]:
        citation_groups[(row["field_id"], row["publication_year"])].append(
            row["citation_count_age_3"]
        )
    replayed: dict[tuple[int, int], tuple[int | None, int | None, int | None]] = {}
    tie_inflation = 0
    unresolved = 0
    for key, counts in citation_groups.items():
        if len(counts) < 10:
            replayed[key] = (None, None, None)
            unresolved += 1
            continue
        target = math.ceil(len(counts) * 0.10)
        cutoff = sorted(counts, reverse=True)[target - 1]
        success = sum(count >= cutoff for count in counts)
        inflation = success - target
        tie_inflation += inflation
        replayed[key] = (cutoff, success, inflation)
    observed_citations = _mapping(expected.get("public_success_application"), "expected citations")
    for group in observed_citations["groups"]:
        key = (group["field_id"], group["publication_year"])
        require(
            replayed[key]
            == (
                group["citation_cutoff_age_3"],
                group["public_success_count"],
                group["inclusive_tie_inflation_count"],
            ),
            f"citation fixture replay {key}",
        )
    require(observed_citations.get("inclusive_tie_inflation_total") == tie_inflation, "tie inflation replay")
    require(observed_citations.get("unresolved_group_count") == unresolved, "small cohort replay")
    require(expected.get("citation_maturation_lag_years") == 3, "fixture maturation")
    margin = _mapping(expected.get("negative_control_equivalence_policy"), "fixture margin")
    require(margin.get("symmetric_practical_null") == [-0.01, 0.01], "fixture margin replay")
    return {
        "attention_group_count": len(attention_groups),
        "attention_primary_surge_count": independent_surge_count,
        "attention_mad_zero_group_count": independent_mad_zero_count,
        "citation_group_count": len(citation_groups),
        "citation_unresolved_group_count": unresolved,
        "citation_inclusive_tie_inflation_total": tie_inflation,
        "citation_maturation_lag_years": 3,
        "negative_control_practical_null": [-0.01, 0.01],
    }


def _validate_application_tool() -> dict[str, Any]:
    require(APPLICATION_TOOL.is_file() and not APPLICATION_TOOL.is_symlink(), "application tool")
    tree = ast.parse(APPLICATION_TOOL.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    forbidden = sorted(imported & {"duckdb", "numpy", "pandas", "polars", "pyarrow"})
    require(not forbidden, f"row-data library imports {forbidden}")
    return {
        "relative_path": str(APPLICATION_TOOL.relative_to(REPO_ROOT)),
        "sha256": _sha256(APPLICATION_TOOL),
        "row_data_library_imports": forbidden,
    }


def _validate_lifecycle(contract: Mapping[str, Any]) -> dict[str, Any]:
    lifecycle = _mapping(contract.get("lifecycle"), "lifecycle")
    require(lifecycle.get("contract_frozen") is True, "contract freeze")
    require(lifecycle.get("independently_validated") is False, "pre-validation state")
    require(lifecycle.get("discovery_production_application_run") is False, "production application")
    require(lifecycle.get("g0_v2_changed") is False, "G0 v2 mutation")
    require(lifecycle.get("unresolved_g0_choice_count_before") == 17, "choice count before")
    require(lifecycle.get("resolved_choice_count_in_this_contract") == 4, "resolved count")
    require(lifecycle.get("unresolved_g0_choice_count_after_future_binding") == 13, "choice count after")
    for key in (
        "required_sections_completed",
        "g4_artifact_set_materialized",
        "holdout_preflight_written",
        "holdout_seal_created",
        "model_fitted",
        "g4_open",
    ):
        require(lifecycle.get(key) is False, f"lifecycle boundary {key}")
    require(lifecycle.get("confirmatory_or_holdout_rows_read") == 0, "holdout reads")
    authorization = _mapping(contract.get("authorization"), "authorization")
    require(
        authorization.get("discovery_production_application")
        == "REQUIRES_SUCCESSOR_GATE",
        "production application gate",
    )
    require(
        authorization.get("g0_successor_binding")
        == "REQUIRES_INDEPENDENT_VALIDATION_AND_SUCCESSOR_GATE",
        "G0 successor gate",
    )
    for key in (
        "confirmatory_or_holdout_access",
        "outcome_based_selection",
        "g0_measurement_freeze",
        "model_fitting",
        "g4_open",
    ):
        require(
            authorization.get(key) == "NOT_AUTHORIZED",
            f"authorization boundary {key}",
        )
    return {
        "discovery_production_application_run": False,
        "g0_v2_changed": False,
        "resolved_choice_count_in_contract": 4,
        "remaining_choice_count_after_future_binding": 13,
        "confirmatory_or_holdout_rows_read": 0,
        "model_fitted": False,
        "g4_open": False,
    }


def run(
    contract_path: Path = DEFAULT_CONTRACT,
    fixture_input_path: Path = DEFAULT_FIXTURE_INPUT,
    fixture_expected_path: Path = DEFAULT_FIXTURE_EXPECTED,
) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=True)
    fixture_input_path = fixture_input_path.resolve(strict=True)
    fixture_expected_path = fixture_expected_path.resolve(strict=True)
    contract = _read_yaml(contract_path, "precommitment contract")
    require(contract.get("schema_version") == CONTRACT_SCHEMA, "contract schema")
    require(contract.get("status") == CONTRACT_STATUS, "contract status")
    documents = {
        key: _validate_binding(contract, key) for key in EXPECTED_BINDINGS
    }
    source_replay = _validate_source_semantics(documents)
    choice_replay = _validate_choices(contract)
    fixture_input = _read_json(fixture_input_path, "fixture input")
    fixture_expected = _read_json(fixture_expected_path, "fixture expected")
    fixture_replay = _independent_fixture_replay(fixture_input, fixture_expected)
    tool = _validate_application_tool()
    lifecycle = _validate_lifecycle(contract)
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "source_replay": source_replay,
        "choice_replay": choice_replay,
        "fixture_replay": fixture_replay,
        "application_tool": tool,
        "lifecycle_replay": lifecycle,
        "access_accounting": {
            "repository_yaml_documents_read": 5,
            "repository_json_documents_read": 2,
            "repository_text_documents_read": 1,
            "synthetic_fixture_files_read": 2,
            "operator_directories_listed_or_scanned": 0,
            "operator_json_metadata_files_read": 0,
            "parquet_files_opened": 0,
            "production_row_level_files_opened": 0,
            "confirmatory_or_holdout_rows_read": 0,
            "model_fitted": False,
            "g4_open": False,
        },
        "repository_artifacts": {
            "contract": {
                "relative_path": str(contract_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(contract_path),
            },
            "application_tool": tool,
            "validator": {
                "relative_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "fixture_input": {
                "relative_path": str(fixture_input_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(fixture_input_path),
            },
            "fixture_expected": {
                "relative_path": str(fixture_expected_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(fixture_expected_path),
            },
        },
        "authorization": {
            "four_choice_contract": "COMPLETE_INDEPENDENTLY_VALIDATED_FIXTURE_ONLY",
            "discovery_production_application": "REQUIRES_SUCCESSOR_GATE",
            "g0_successor_binding": "REQUIRES_SUCCESSOR_GATE",
            "confirmatory_or_holdout_access": "NOT_AUTHORIZED",
            "g0_measurement_freeze": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "claim_boundary": contract["claim_boundary"].strip(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--fixture-input", type=Path, default=DEFAULT_FIXTURE_INPUT)
    parser.add_argument(
        "--fixture-expected", type=Path, default=DEFAULT_FIXTURE_EXPECTED
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run(args.contract, args.fixture_input, args.fixture_expected)
    payload = json.dumps(result, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
