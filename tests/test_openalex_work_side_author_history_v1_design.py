from __future__ import annotations

import copy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from yaml.constructor import ConstructorError
from yaml.events import AliasEvent
from yaml.nodes import Node
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_work_side_author_history_v1.yaml"
)
DOC_PATH = (
    REPO_ROOT / "docs/research/openalex-work-side-author-history-v1-audit-design.md"
)
LEGACY_SURFACES = {
    REPO_ROOT / "experiments/configs/openalex_g1_work_side_author_history_v0.yaml": (
        "d909c597b04e45b92da664e7589c8abe4e29aae738401a4f961394c84f23111c"
    ),
    REPO_ROOT / "tests/test_openalex_work_side_author_history_design.py": (
        "cea9697e76825270e13253424dfcfb5c2f4fbeab7d08853e10564e1c16f33f73"
    ),
    REPO_ROOT / "docs/case-studies/researcher-pivot/"
    "openalex-work-side-author-history-audit-design.md": (
        "8614244ecdc33f7060505c99d86604b0972eeb3218f7f034f65d81cf93624213"
    ),
}

LIFECYCLE = {
    "design_status": "DESIGN_ONLY",
    "production_status": "PRODUCTION_NOT_RUN",
    "executable": False,
    "execution_script": None,
    "synthetic_executable": False,
    "production_executable": False,
    "synthetic_result_exists": False,
    "production_result_exists": False,
    "completion_receipt_exists": False,
    "pass_result_allowed_in_design_artifact": False,
    "production_execution_authorized": False,
    "implementation_requires_new_version_and_review": True,
}

TEMPORAL_SCHEMA_AND_PATHS = {
    "parent_version_policy": "TEMPORAL_V3_PARENT_ONLY_NO_V1_OR_V2_FALLBACK",
    "receipt_schema_version": "openalex_g1_main_temporal_validity_outputs.v3",
    "required_plan_schema_version": "openalex_g1_main_temporal_validity_plan.v3",
    "required_checkpoint_schema_version": (
        "openalex_g1_main_temporal_validity_checkpoint.v3"
    ),
    "required_report_schema_version": "openalex_g1_main_temporal_validity_report.v3",
    "required_state_schema_version": "openalex_g1_main_temporal_validity_state.v3",
    "required_status": (
        "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_"
        "EXACT_UTC_MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE"
    ),
    "required_config_path": (
        "experiments/configs/openalex_g1_main_temporal_validity_v2.yaml"
    ),
    "required_tool_path": "scripts/audit_openalex_main_temporal_validity_v3.py",
}

TEMPORAL_ANCHORS = {
    "required_config_sha256": (
        "672940908090fedd0b562dd938ad4583f5e8238830fc1ff5b613be9618f033f7"
    ),
    "required_tool_sha256": (
        "76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d"
    ),
    "required_receipt_sha256": (
        "2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe"
    ),
    "required_report_sha256": (
        "de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200"
    ),
    "required_exact_task_fingerprint_sha256": (
        "5d86a797481a14e95d08ecbe333bb207ff476bf7dc49886aec1db7b5b366001d"
    ),
    "required_exact_result_fingerprint_sha256": (
        "a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5"
    ),
    "required_exact_checkpoint_tree_sha256": (
        "725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392"
    ),
    "required_exact_source_tree_sha256": (
        "da3fa570a5fb49a36668dffdee008c2fea4d4b6fd5a3f7f17873df78b603b2a0"
    ),
}

TEMPORAL_METRIC_KEYS = (
    "raw_rows",
    "quarantined_w_minus_1_rows",
    "retained_rows",
    "null_id_rows",
    "grammar_valid_id_rows",
    "unexpected_malformed_id_rows",
    "w_zero_rows",
    "publication_both_null_rows",
    "publication_exactly_one_null_rows",
    "publication_both_nonnull_rows",
    "publication_date_parse_error_rows",
    "publication_year_invalid_rows",
    "publication_concordance_checked_rows",
    "publication_date_year_match_rows",
    "publication_date_year_mismatch_rows",
    "publication_pre_1900_rows",
    "publication_1900_before_snapshot_year_rows",
    "publication_snapshot_year_on_or_before_snapshot_rows",
    "publication_snapshot_year_after_snapshot_rows",
    "publication_post_snapshot_year_rows",
    "publication_year_null_rows",
    "publication_year_pre_1900_rows",
    "publication_year_snapshot_year_rows",
    "publication_year_post_snapshot_year_rows",
    "created_updated_both_null_rows",
    "created_updated_exactly_one_null_rows",
    "created_updated_both_nonnull_rows",
    "created_date_parse_error_rows",
    "updated_date_parse_error_rows",
    "created_updated_parsed_pair_rows",
    "created_lte_updated_rows",
    "created_gt_updated_rows",
    "created_after_snapshot_rows",
    "updated_after_snapshot_rows",
    "future_metadata_field_values",
)

FORBIDDEN_TEMPORAL_STATUSES = (
    "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_COMPONENT_WITH_FROZEN_EXCLUSIONS",
    "PILOT_MAIN_TEMPORAL_VALIDITY_ONLY",
    "BLOCKED_MAIN_TEMPORAL_VALIDITY_COMPONENT",
    "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_V2_WITH_"
    "EXACT_UTC_MILLISECONDS_AND_FROZEN_EXCLUSIONS",
    "PILOT_MAIN_TEMPORAL_VALIDITY_V2_EXACT_UTC_MILLISECOND_ONLY",
    "BLOCKED_MAIN_TEMPORAL_VALIDITY_V2_EXACT_UTC_MILLISECOND_COMPONENT",
)

CANONICAL_BINDING = {
    "design_status": "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN",
    "production_status": "PRODUCTION_NOT_RUN",
    "synthetic_executable": True,
    "production_executable": False,
    "synthetic_execution_script": (
        "scripts/audit_openalex_canonical_work_time_spine_v2.py"
    ),
    "synthetic_execution_script_sha256": (
        "6a2d1d87dd4a3e7d3c97056fabbe951a1528cfd62d8448acc74ae337f344b7b1"
    ),
    "synthetic_status": (
        "PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_TEMPORAL_V3_PARENT_CONTRACT_ONLY"
    ),
    "synthetic_contract_result_is_production_parent": False,
    "production_result_exists": False,
    "production_receipt_exists": False,
    "production_receipt_binding_state": "NOT_AVAILABLE_PRODUCTION_NOT_RUN",
    "production_parent_eligible": False,
    "reuse_mode_status": "BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE",
    "author_production_execution_state": (
        "BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE"
    ),
    "design_config_path": (
        "experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml"
    ),
    "design_config_sha256": (
        "2caef77c8b3dc2ffcb0a35920c9e3527096ec3a9f31d20166eec3a3486f4196b"
    ),
    "design_schema_version": "openalex_g1_canonical_work_time_spine_design.v2",
    "future_production_receipt_schema_state": "NOT_FROZEN_REQUIRES_NEW_VERSION",
    "future_production_receipt_sha256_state": ("NOT_AVAILABLE_MUST_NOT_BE_INVENTED"),
    "future_canonical_and_author_versions_required_before_binding": True,
    "future_schema_or_hash_may_be_invented_by_v1": False,
}

CITATION_BINDING = {
    "hard_parent_of_author_token_history": False,
    "relation_to_author_component": ("DOWNSTREAM_SIBLING_BLOCKER_FOR_G_MINUS_1_AND_G0"),
    "design_schema_version": "openalex_g1_citation_available_at_design.v2",
    "design_config_path": "experiments/configs/openalex_g1_citation_available_at_v1.yaml",
    "design_config_sha256": (
        "e281f7bca4ac801073a8d0cbd59cf8b9e7eca2d924f19545925a5cbff38fe75b"
    ),
    "synthetic_execution_script": "scripts/audit_openalex_citation_available_at_v2.py",
    "synthetic_execution_script_sha256": (
        "71de51ac081cb17b342395d3b9e1435d752340457e233367b84907e17188907a"
    ),
    "design_status": "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN",
    "synthetic_status": (
        "PASS_SYNTHETIC_CITATION_EDGE_CHRONOLOGY_TEMPORAL_V3_DIRECT_CONTRACT_ONLY"
    ),
    "production_status": "PRODUCTION_NOT_RUN",
    "production_receipt_exists": False,
    "g_minus_1_dependency_state": ("BLOCKED_CITATION_PRODUCTION_RECEIPT_NOT_AVAILABLE"),
    "historical_edge_observation_time_established": False,
    "outcome_maturation_state": "PENDING_NOT_DEFINED",
    "author_parent_injection_allowed": False,
}


class _UniqueKeyLoader(yaml.SafeLoader):
    def compose_node(self, parent: Any, index: Any) -> Node:
        if self.check_event(AliasEvent):
            event = self.peek_event()
            raise ConstructorError(
                "while composing a node",
                None,
                "YAML aliases are forbidden",
                event.start_mark,
            )
        return super().compose_node(parent, index)


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    if any(key_node.value == "<<" for key_node, _ in node.value):
        raise ConstructorError(
            "while constructing a mapping",
            node.start_mark,
            "YAML merge keys are forbidden",
            node.start_mark,
        )
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config() -> dict[str, Any]:
    value = yaml.load(CONFIG_PATH.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    assert isinstance(value, dict)
    return value


def _set_nested(value: dict[str, Any], path: str, replacement: Any) -> None:
    cursor = value
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor[part]
        assert isinstance(child, dict)
        cursor = child
    cursor[parts[-1]] = replacement


def _assert_mapping_subset(value: dict[str, Any], expected: dict[str, Any]) -> None:
    assert {key: value[key] for key in expected} == expected


def _validate_temporal(parent: dict[str, Any]) -> None:
    _assert_mapping_subset(parent, TEMPORAL_SCHEMA_AND_PATHS)
    _assert_mapping_subset(parent, TEMPORAL_ANCHORS)
    assert parent["role"] == "AGGREGATE_AUDIT_PARENT_NOT_ROW_LEVEL_AUTHOR_INPUT"
    assert parent["required_run_state"] == "COMPLETE_PENDING_RECEIPT"
    assert parent["required_full_scope"] is True
    assert parent["required_completion_contract"] == (
        "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
    )
    assert parent["inherited_policy"] == "FROZEN_MAIN_TEMPORAL_POLICY_V3"
    assert tuple(parent["exact_checkpoint_metric_keyset"]) == TEMPORAL_METRIC_KEYS
    assert len(TEMPORAL_METRIC_KEYS[:24]) == 24
    assert len(TEMPORAL_METRIC_KEYS[24:]) == 11
    assert parent["publication_side_metric_keyset_count"] == 24
    assert parent["created_updated_inherited_only_metric_keyset_count"] == 11
    assert (
        parent["created_updated_metrics_may_be_recomputed_by_future_author_runner"]
        is False
    )
    assert parent["row_level_dispositions_materialized"] is False
    assert parent["row_level_parent_product_exists"] is False
    assert parent["all_declared_rows_audited_by_temporal_v3"] is True
    assert parent["reusable_row_level_parent_product_persisted"] is False
    assert parent["direct_row_identity_claim_allowed"] is False
    assert parent["exact_receipt_keyset"] == [
        "schema_version",
        "status",
        "result_fingerprint_sha256",
        "task_fingerprint_sha256",
        "artifacts",
        "completion_contract",
        "operator_paths_redacted",
    ]
    assert parent["exact_fixed_receipt_artifact_paths"] == [
        "audit_plan.json",
        "run_state.json",
        "main_temporal_validity_audit.json",
        "main_temporal_validity_audit.md",
    ]
    assert parent["checkpoint_artifact_directory"] == "checkpoints/main"
    assert parent["checkpoint_artifact_filename_pattern"] == "^[0-9a-f]{64}[.]json$"
    assert parent["exact_checkpoint_artifact_count"] == 493
    assert parent["exact_total_receipt_artifact_count"] == 497
    assert parent["require_exact_receipt_artifact_set"] is True
    assert parent["require_every_receipt_artifact_relative_path_key_equality"] is True
    assert parent["require_every_receipt_artifact_bytes_and_sha256"] is True
    assert parent["require_same_content_baseline_parent"] is True
    forbidden = parent["forbidden_prior_parent"]
    assert tuple(forbidden["statuses"]) == FORBIDDEN_TEMPORAL_STATUSES
    assert set(forbidden["receipt_schema_versions"]) == {
        "openalex_g1_main_temporal_validity_outputs.v1",
        "openalex_g1_main_temporal_validity_outputs.v2",
    }
    assert set(forbidden["config_paths"]) == {
        "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml",
        "experiments/configs/openalex_g1_main_temporal_validity_v1.yaml",
    }
    assert set(forbidden["tool_paths"]) == {
        "scripts/audit_openalex_main_temporal_validity.py",
        "scripts/audit_openalex_main_temporal_validity_v2.py",
    }


def _validate_canonical(parent: dict[str, Any]) -> None:
    _assert_mapping_subset(parent, CANONICAL_BINDING)
    assert parent["role"] == (
        "ONLY_ALLOWED_ROW_LEVEL_TIME_SOURCE_CURRENTLY_UNAVAILABLE_FOR_PRODUCTION"
    )
    assert parent["forbidden_unavailable_keys"] == [
        "production_receipt_sha256",
        "required_future_receipt_schema_version",
    ]
    assert all(key not in parent for key in parent["forbidden_unavailable_keys"])
    assert parent["forbidden_legacy_design"] == {
        "design_config_path": (
            "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml"
        ),
        "design_config_sha256": (
            "192d78e3df094121e63922a6eaf20f57a57af41bc174c0fdbbcc347e3eaed385"
        ),
        "synthetic_execution_script": (
            "scripts/audit_openalex_canonical_work_time_spine.py"
        ),
        "synthetic_execution_script_sha256": (
            "827bab5a6edcd9a574149203c68c77ae5178a8da210499b81f05a388ace5c0dc"
        ),
        "design_schema_version": "openalex_g1_canonical_work_time_spine_design.v1",
        "synthetic_status": "PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_CONTRACT_ONLY",
    }


def _validate_claim_boundaries(config: dict[str, Any]) -> None:
    assert set(config["parents"]) == {
        "current_content_baseline",
        "main_temporal_validity",
        "canonical_work_time_spine",
    }
    assert config["required_parent_execution_order"] == [
        "current_content_baseline",
        "main_temporal_validity",
        "canonical_work_time_spine",
        "work_side_author_history",
    ]
    temporal = config["temporal_consumption"]
    assert temporal["temporal_parent_policy"] == "FROZEN_MAIN_TEMPORAL_POLICY_V3"
    assert temporal["temporal_v3_role"] == "AGGREGATE_VALIDITY_EVIDENCE_ONLY"
    assert temporal["only_allowed_row_level_time_source"] == (
        "canonical_work_time_spine"
    )
    assert temporal["canonical_row_level_source_available_for_production_now"] is False
    assert temporal["source_publication_date_reparse_allowed"] is False
    assert (
        temporal["direct_reconstruction_from_main_source_allowed_by_this_design"]
        is False
    )
    assert temporal["row_level_event_materialized_by_this_design"] is False

    identity = config["author_identity"]
    assert identity["token_not_person"] is True
    assert identity["natural_person_entity_claim_allowed"] is False
    assert identity["author_master_referential_integrity_claim_allowed"] is False
    assert identity["token_reassignment_allowed"] is False
    assert identity["name_or_orcid_used_to_reassign_tokens"] is False
    assert identity["historical_identity_availability_established"] is False
    assert identity["past_observable_token_linkage_claim_allowed"] is False

    master = config["incomplete_author_master"]
    assert master["actual_coverage"] == "NOT_MEASURED"
    assert master["theoretical_upper_bound_percent"] == 38.76968115
    assert master["theoretical_upper_bound_semantics"] == (
        "UNVERIFIED_ROW_COUNT_UPPER_BOUND"
    )
    assert master["actual_coverage_established"] is False
    assert master["full_parent_role_allowed"] is False
    assert master["author_entity_validation_allowed"] is False

    method = config["author_work_event_method"]
    assert method["representative_row_selection"] == "FORBIDDEN"
    assert method["full_snapshot_token_summaries_role"] == (
        "AUDIT_ONLY_NOT_PAST_CUTOFF_FEATURES"
    )
    assert method["final_field_selection_allowed"] is False
    assert method["candidate_selection_allowed"] is False
    assert method["threshold_driven_candidate_exclusion_allowed"] is False
    assert method["team_cap_predicates"] == {
        "cap_10": "declared_authors_count <= 10",
        "cap_50": "declared_authors_count <= 50",
        "cap_100": "declared_authors_count <= 100",
        "cap_ALL": "include_every_history_usable_unique_author_work_pair",
    }
    assert method["threshold_cell_predicate"] == (
        "maximum_annual_unique_work_count > threshold"
    )
    assert method["each_cell_reduces_same_view_same_cap_hidden_records"] is True
    assert method["cross_cap_source_substitution_allowed"] is False

    observability = config["historical_observability"]
    assert observability["required_cutoff"] == (
        "publication_day <= days_since_epoch(t_minus_1_december_31)"
    )
    assert observability["cutoff_uses_canonical_spine_publication_day_only"] is True
    assert observability["token_or_field_past_observability_established"] is False
    assert observability["citation_available_at_established"] is False
    assert observability["outcome_maturation_established"] is False

    diagnostics = config["diagnostic_semantics"]
    assert diagnostics["orcid_checksum_validation_role"] == (
        "LEXICAL_AND_CHECKSUM_DIAGNOSTIC_NOT_GROUND_TRUTH"
    )
    assert diagnostics["orcid_split_merge_semantics"] == (
        "ANOMALY_PROXY_NOT_GROUND_TRUTH"
    )
    assert diagnostics["exact_and_normalized_name_variation_role"] == (
        "ANOMALY_PROXY_NOT_ENTITY_RESOLUTION"
    )
    assert diagnostics["orcid_or_name_can_establish_author_entity"] is False
    assert diagnostics["orcid_or_name_can_establish_split_merge_truth"] is False
    assert diagnostics["past_observable_model_feature_allowed"] is False

    obligations = config["future_runner_obligations"]
    assert obligations["status"] == "OBLIGATIONS_ONLY_NO_RUNNER_OR_OUTPUT_ABI_IN_V1"
    assert obligations["production_runner_path"] == "NOT_FROZEN"
    assert obligations["production_output_abi"] == (
        "NOT_FROZEN_REQUIRES_SEPARATE_VERSION_AND_REVIEW"
    )
    assert obligations["author_work_event_artifact_schema"] == "NOT_FROZEN"
    assert obligations["author_work_event_artifact_receipt_schema"] == "NOT_FROZEN"
    assert (
        obligations["current_v1_freezes_any_production_output_or_event_artifact"]
        is False
    )
    strategy = obligations["future_closure_evidence_strategy"]
    assert strategy["role"] == (
        "FINALIZATION_AND_TERMINAL_REENTRY_PROOF_NOT_A_ROW_LEVEL_TIME_PARENT"
    )
    assert strategy["decision_status"] == "UNCHOSEN_REQUIRES_NEW_VERSION"
    assert strategy["exactly_one_strategy_must_be_frozen_before_production"] == [
        "RECEIPT_BOUND_AUTHOR_WORK_EVENT_ARTIFACT",
        "SAME_PARENT_FULL_RESCAN",
    ]
    assert strategy["strategy_may_change_on_resume"] is False
    assert (
        strategy["canonical_spine_remains_the_author_runner_row_level_time_input"]
        is True
    )
    artifact = strategy["receipt_bound_author_work_event_artifact_future_requirements"]
    assert artifact == {
        "retained_main_to_canonical_spine_join_closure": (
            "EXACTLY_ONE_SPINE_ROW_PER_RETAINED_MAIN_WORK"
        ),
        "event_row_count_and_schema_digest": "BIND_EXACT_VALUES_IN_NEW_VERSION",
        "event_content_tree_or_equivalent_digest": "BIND_EXACT_VALUE_IN_NEW_VERSION",
        "canonical_result_fingerprint": (
            "BIND_EXACT_VALUE_FROM_REAL_PRODUCTION_RECEIPT"
        ),
        "canonical_spine_data_tree_sha256": (
            "BIND_EXACT_VALUE_FROM_REAL_PRODUCTION_RECEIPT"
        ),
        "terminal_revalidation_method": "NOT_FROZEN_REQUIRES_NEW_VERSION",
    }
    rescan = strategy["same_parent_full_rescan_future_requirements"]
    assert rescan["revalidate_same_parent_receipts_and_source_identity"] is True
    assert (
        rescan["reproduce_exact_retained_main_to_canonical_spine_join_closure"] is True
    )
    assert rescan["recompute_event_row_and_schema_digests"] is True
    assert rescan["terminal_revalidation_method"] == ("NOT_FROZEN_REQUIRES_NEW_VERSION")
    split = obligations["temporal_metric_partition"]
    assert split == {
        "exact_parent_metric_key_count": 35,
        "publication_side_recomputed_key_count": 24,
        "created_updated_inherited_only_key_count": 11,
        "future_author_or_canonical_publication_scan_may_recompute_only_first_24_parent_keys": True,
        "future_author_or_canonical_scan_may_recompute_created_updated_11_keys": False,
        "created_updated_11_keys_must_be_inherited_from_exact_temporal_v3_receipt": True,
    }
    closures = obligations["hidden_cross_record_closure_obligations"]
    assert closures["status"] == "FUTURE_RUNNER_OBLIGATION_NOT_EXECUTED_BY_V1"
    assert len(closures["required_closures"]) == 7
    assert closures["hidden_arrays_published_by_v1"] is False
    assert closures["hidden_arrays_reopened_on_terminal_reentry_by_v1"] is False
    assert closures["v1_claims_hidden_closures_ran"] is False
    resume = obligations["terminal_and_resume_revalidation"]
    assert resume["required_in_future_production_version"] is True
    assert (
        resume["valid_terminal_reentry_must_rehash_receipt_and_every_bound_artifact"]
        is True
    )
    assert (
        resume[
            "valid_terminal_reentry_must_revalidate_all_parent_receipts_and_cross_bindings"
        ]
        is True
    )
    assert (
        resume[
            "failed_state_resume_must_revalidate_plan_parents_sources_and_completed_checkpoints"
        ]
        is True
    )
    assert (
        resume[
            "selected_closure_evidence_strategy_and_commitments_must_match_original_task"
        ]
        is True
    )
    assert resume["requirements_implemented_by_v1"] is False

    topic = config["parallel_siblings"]["work_topic_relation"]
    assert topic["hard_parent_of_author_token_history"] is False
    assert topic["relation_to_author_component"] == "PARALLEL_SIBLING"
    citation = config["downstream_sibling_blockers"]["citation_available_at"]
    _assert_mapping_subset(citation, CITATION_BINDING)
    assert citation["forbidden_legacy_design"] == {
        "design_schema_version": "openalex_g1_citation_available_at_design.v1",
        "design_config_path": "experiments/configs/openalex_g1_citation_available_at_v0.yaml",
        "design_config_sha256": (
            "51f5044459514b5150a8e24b2e4daf05575e06fc366d412bc3d046d87ba3c798"
        ),
        "synthetic_execution_script": "scripts/audit_openalex_citation_available_at.py",
        "synthetic_execution_script_sha256": (
            "0be5444c9b72aea95251e609543a0da44d1ff95a432c839b674921e639409bde"
        ),
        "report_status": "SYNTHETIC_CITATION_EDGE_CHRONOLOGY_ONLY",
    }

    assert config["current_gate_state"] == {
        "work_side_author_history": "DESIGN_ONLY",
        "current_content_baseline": "REVALIDATE_AT_FUTURE_EXECUTION",
        "main_temporal_validity": "PASS_EVIDENCE_BOUND_AGGREGATE_ONLY",
        "canonical_work_time_spine": (
            "BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE"
        ),
        "author_entity_validity": "UNRESOLVED_WORK_SIDE_TOKEN_ONLY",
        "author_master_coverage": "UNVERIFIED_ROW_COUNT_UPPER_BOUND",
        "historical_content_identity": "UNVERIFIED",
        "token_and_field_past_observability": (
            "UNVERIFIED_CURRENT_SNAPSHOT_RETROSPECTIVE_PROXY"
        ),
        "orcid_validation": "DIAGNOSTIC_NOT_GROUND_TRUTH",
        "a_zero_entity_validity": "UNRESOLVED",
        "split_merge_validity": "UNRESOLVED_ANOMALY_PROXY_ONLY",
        "candidate_eligibility": "PENDING",
        "citation_available_at": ("BLOCKED_CITATION_PRODUCTION_RECEIPT_NOT_AVAILABLE"),
        "outcome_maturation": "PENDING",
        "field_selection": "PENDING",
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }
    future = config["future_component_pass_preconditions"]
    assert future == {
        "hypothetical_status": (
            "PASS_CURRENT_SNAPSHOT_WORK_SIDE_AUTHOR_TOKEN_LINKED_RETROSPECTIVE_"
            "HISTORY_ACCOUNTING_AND_SENSITIVITY_COMPONENT_AUTHOR_ENTITY_AND_"
            "PAST_OBSERVABILITY_UNVERIFIED"
        ),
        "status_is_current_result": False,
        "require_new_versioned_production_runner_and_output_abi": True,
        "require_real_separately_versioned_canonical_production_receipt": True,
        "require_canonical_parent_state": "PASS_FROM_VALIDATED_PRODUCTION_RECEIPT",
        "require_one_frozen_closure_evidence_strategy": True,
        "require_hidden_closure_execution_and_receipt_binding": True,
        "require_terminal_and_resume_revalidation": True,
        "may_promote_only_bounded_author_token_accounting_scope": True,
        "may_establish_author_entities": False,
        "may_establish_candidate_eligibility": False,
        "may_establish_g_minus_1_or_g0": False,
        "may_establish_empirical_g1": False,
        "may_establish_scientific_effect": False,
    }

    claim = config["claim_boundary"]
    for phrase in (
        "DESIGN_ONLY",
        "aggregate/checkpoint evidence",
        "currently unavailable",
        "reparsing is forbidden",
        "tokens, not verified persons",
        "UNVERIFIED_ROW_COUNT_UPPER_BOUND",
        "anomaly proxies only",
        "parallel sibling",
        "downstream sibling",
        "did not run here",
        "G-1",
        "G0",
        "empirical G1",
        "unestablished",
    ):
        assert phrase in claim


def _validate_config(config: dict[str, Any]) -> None:
    assert config["schema_version"] == "openalex_g1_work_side_author_history_design.v2"
    assert config["label"] == (
        "openalex_g1_work_side_author_history_v1_temporal_v3_parent_only"
    )
    assert config["lifecycle"] == LIFECYCLE
    assert config["implementation_readiness"] == {
        "production_output_abi": "NOT_IMPLEMENTED_REQUIRES_SEPARATE_REVIEW",
        "executable_created_by_this_tranche": False,
        "result_created_by_this_tranche": False,
        "production_execution_authorized": False,
    }
    _validate_temporal(config["parents"]["main_temporal_validity"])
    _validate_canonical(config["parents"]["canonical_work_time_spine"])
    _validate_claim_boundaries(config)


def test_legacy_v0_surfaces_remain_byte_exact() -> None:
    assert {path: _sha256(path) for path in LEGACY_SURFACES} == LEGACY_SURFACES
    config = _load_config()
    assert config["legacy_v0_surfaces"]["paths_and_sha256"] == {
        str(path.relative_to(REPO_ROOT)): digest
        for path, digest in LEGACY_SURFACES.items()
    }


def test_unique_yaml_and_whole_design_validator_accept_successor() -> None:
    _validate_config(_load_config())
    with pytest.raises(ConstructorError, match="duplicate key"):
        yaml.load("status: A\nstatus: B\n", Loader=_UniqueKeyLoader)
    with pytest.raises(ConstructorError, match="aliases are forbidden"):
        yaml.load("a: &x 1\nb: *x\n", Loader=_UniqueKeyLoader)
    with pytest.raises(ConstructorError):
        yaml.load("a: &x {v: 1}\nb: {<<: *x}\n", Loader=_UniqueKeyLoader)
    with pytest.raises(ConstructorError):
        yaml.load("value: !custom tagged\n", Loader=_UniqueKeyLoader)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("design_status", "PASS"),
        ("production_status", "PASS"),
        ("executable", True),
        ("execution_script", "scripts/audit_openalex_work_side_author_history_v1.py"),
        ("synthetic_executable", True),
        ("production_executable", True),
        ("synthetic_result_exists", True),
        ("production_result_exists", True),
        ("completion_receipt_exists", True),
        ("pass_result_allowed_in_design_artifact", True),
        ("production_execution_authorized", True),
    ],
)
def test_lifecycle_or_result_promotion_is_rejected(
    field: str, unsafe_value: Any
) -> None:
    config = copy.deepcopy(_load_config())
    config["lifecycle"][field] = unsafe_value
    with pytest.raises(AssertionError):
        _validate_config(config)


def test_current_content_and_live_parent_design_hashes_are_exact() -> None:
    config = _load_config()
    content = config["parents"]["current_content_baseline"]
    assert content["required_plan_sha256"] == (
        "97294fe1c1fa026f664ca239ebfa9fc6dce593c847a20eb62157e38312b6725e"
    )
    assert content["required_task_fingerprint_sha256"] == (
        "f915cade4ce6581832c46957a2ad0a37ceabe2296099f0faeaac56ffa035693e"
    )
    assert content["required_receipt_basename"] == "content_freeze_audit.outputs.json"
    assert content["required_completion_contract"] == (
        "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
    )
    assert content["exact_receipt_keyset"] == [
        "schema_version",
        "status",
        "task_fingerprint_sha256",
        "plan_sha256",
        "result_fingerprint_sha256",
        "content_tree_sha256",
        "checkpoint_tree",
        "artifacts",
        "completion_contract",
        "run_state_bound_separately",
        "path_redacted",
    ]
    assert content["exact_receipt_artifact_paths"] == [
        "audit_plan.json",
        "content_freeze_audit.json",
        "content_freeze_audit.md",
    ]
    assert content["exact_receipt_artifact_count"] == 3
    for parent, path_key, sha_key in (
        (content, "required_config_path", "required_config_sha256"),
        (content, "required_tool_path", "required_tool_sha256"),
        (
            config["parents"]["main_temporal_validity"],
            "required_config_path",
            "required_config_sha256",
        ),
        (
            config["parents"]["main_temporal_validity"],
            "required_tool_path",
            "required_tool_sha256",
        ),
        (
            config["parents"]["canonical_work_time_spine"],
            "design_config_path",
            "design_config_sha256",
        ),
        (
            config["parents"]["canonical_work_time_spine"],
            "synthetic_execution_script",
            "synthetic_execution_script_sha256",
        ),
    ):
        assert _sha256(REPO_ROOT / parent[path_key]) == parent[sha_key]


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    [
        ("receipt_schema_version", "openalex_g1_main_temporal_validity_outputs.v1"),
        ("receipt_schema_version", "openalex_g1_main_temporal_validity_outputs.v2"),
        ("required_plan_schema_version", "openalex_g1_main_temporal_validity_plan.v1"),
        ("required_plan_schema_version", "openalex_g1_main_temporal_validity_plan.v2"),
        (
            "required_checkpoint_schema_version",
            "openalex_g1_main_temporal_validity_checkpoint.v1",
        ),
        (
            "required_checkpoint_schema_version",
            "openalex_g1_main_temporal_validity_checkpoint.v2",
        ),
        (
            "required_report_schema_version",
            "openalex_g1_main_temporal_validity_report.v1",
        ),
        (
            "required_report_schema_version",
            "openalex_g1_main_temporal_validity_report.v2",
        ),
        (
            "required_state_schema_version",
            "openalex_g1_main_temporal_validity_state.v1",
        ),
        (
            "required_state_schema_version",
            "openalex_g1_main_temporal_validity_state.v2",
        ),
        (
            "required_config_path",
            "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml",
        ),
        (
            "required_config_path",
            "experiments/configs/openalex_g1_main_temporal_validity_v1.yaml",
        ),
        ("required_tool_path", "scripts/audit_openalex_main_temporal_validity.py"),
        ("required_tool_path", "scripts/audit_openalex_main_temporal_validity_v2.py"),
    ],
)
def test_temporal_v1_v2_schema_and_path_substitution_is_rejected(
    field: str, legacy_value: str
) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["main_temporal_validity"][field] = legacy_value
    with pytest.raises(AssertionError):
        _validate_config(config)


@pytest.mark.parametrize("legacy_status", FORBIDDEN_TEMPORAL_STATUSES)
def test_every_legacy_temporal_status_is_rejected(legacy_status: str) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["main_temporal_validity"]["required_status"] = legacy_status
    with pytest.raises(AssertionError):
        _validate_config(config)


@pytest.mark.parametrize("field", tuple(TEMPORAL_ANCHORS))
def test_each_of_eight_temporal_v3_anchor_drifts_is_rejected(field: str) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["main_temporal_validity"][field] = "1" * 64
    with pytest.raises(AssertionError):
        _validate_config(config)


@pytest.mark.parametrize(
    ("field", "legacy_sha256"),
    [
        (
            "required_receipt_sha256",
            "f6eec88337fb8482400ad3a25c7b7b7099b1613d575e0c29c6c26575e8612780",
        ),
        (
            "required_config_sha256",
            "d208b45cc87de577726a8310a899eb66882a519bae29df5f7dbbf7833562b172",
        ),
        (
            "required_tool_sha256",
            "b4b39c016ba4c6867a3e08366f70f6125230c2a0a18c1e8c45d40826f373f563",
        ),
        (
            "required_receipt_sha256",
            "ec59f3c649baa4a6a5333daace92a5e46f6492fad62e4c1ed42a1702394425f4",
        ),
        (
            "required_config_sha256",
            "11a668b8f2b10c83e2d36b3f5d2cd16f1e5f86b41ef64f58c79f10cabebc30e8",
        ),
        (
            "required_tool_sha256",
            "b86f2496bbf7258c4e981f9a7ad8b1618fdcdc34a7385de628b133e0082a89dc",
        ),
    ],
)
def test_explicit_v1_v2_temporal_hash_substitution_is_rejected(
    field: str, legacy_sha256: str
) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["main_temporal_validity"][field] = legacy_sha256
    with pytest.raises(AssertionError):
        _validate_config(config)


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    [
        (
            "design_config_path",
            "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml",
        ),
        (
            "design_config_sha256",
            "192d78e3df094121e63922a6eaf20f57a57af41bc174c0fdbbcc347e3eaed385",
        ),
        (
            "synthetic_execution_script",
            "scripts/audit_openalex_canonical_work_time_spine.py",
        ),
        (
            "synthetic_execution_script_sha256",
            "827bab5a6edcd9a574149203c68c77ae5178a8da210499b81f05a388ace5c0dc",
        ),
        ("design_schema_version", "openalex_g1_canonical_work_time_spine_design.v1"),
        (
            "synthetic_status",
            "PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_CONTRACT_ONLY",
        ),
    ],
)
def test_legacy_canonical_design_substitution_is_rejected(
    field: str, legacy_value: Any
) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["canonical_work_time_spine"][field] = legacy_value
    with pytest.raises(AssertionError):
        _validate_config(config)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("production_status", "PASS"),
        ("production_executable", True),
        ("production_result_exists", True),
        ("production_receipt_exists", True),
        ("production_parent_eligible", True),
        ("reuse_mode_status", "PASS"),
        ("author_production_execution_state", "PASS"),
    ],
)
def test_canonical_production_promotion_is_rejected(
    field: str, unsafe_value: Any
) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["canonical_work_time_spine"][field] = unsafe_value
    with pytest.raises(AssertionError):
        _validate_config(config)


@pytest.mark.parametrize(
    "invented_key",
    ["production_receipt_sha256", "required_future_receipt_schema_version"],
)
def test_invented_canonical_receipt_schema_or_hash_is_rejected(
    invented_key: str,
) -> None:
    config = copy.deepcopy(_load_config())
    config["parents"]["canonical_work_time_spine"][invented_key] = "1" * 64
    with pytest.raises(AssertionError):
        _validate_config(config)


def test_incomplete_master_value_is_an_unverified_row_count_upper_bound() -> None:
    master = _load_config()["incomplete_author_master"]
    observed = (
        Decimal(master["observed_delta_rows"])
        * Decimal(100)
        / Decimal(master["work_side_distinct_author_tokens"])
    ).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
    assert observed == Decimal(str(master["theoretical_upper_bound_percent"]))
    assert master["actual_coverage"] == "NOT_MEASURED"
    assert master["theoretical_upper_bound_semantics"] == (
        "UNVERIFIED_ROW_COUNT_UPPER_BOUND"
    )
    assert master["actual_coverage_established"] is False


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        ("author_identity.token_not_person", False),
        ("author_identity.natural_person_entity_claim_allowed", True),
        ("author_identity.author_master_referential_integrity_claim_allowed", True),
        ("author_identity.token_reassignment_allowed", True),
        ("author_identity.name_or_orcid_used_to_reassign_tokens", True),
        ("author_identity.past_observable_token_linkage_claim_allowed", True),
        ("incomplete_author_master.actual_coverage", "MEASURED"),
        (
            "incomplete_author_master.theoretical_upper_bound_semantics",
            "ACTUAL_COVERAGE",
        ),
        ("incomplete_author_master.actual_coverage_established", True),
        ("incomplete_author_master.full_parent_role_allowed", True),
        ("incomplete_author_master.author_entity_validation_allowed", True),
        (
            "diagnostic_semantics.orcid_split_merge_semantics",
            "GROUND_TRUTH",
        ),
        ("diagnostic_semantics.orcid_or_name_can_establish_author_entity", True),
        ("diagnostic_semantics.orcid_or_name_can_establish_split_merge_truth", True),
        ("author_work_event_method.final_field_selection_allowed", True),
        ("author_work_event_method.candidate_selection_allowed", True),
        ("author_work_event_method.threshold_driven_candidate_exclusion_allowed", True),
        (
            "author_work_event_method.team_cap_predicates.cap_10",
            "declared_authors_count < 10",
        ),
        (
            "author_work_event_method.threshold_cell_predicate",
            "maximum_annual_unique_work_count >= threshold",
        ),
        ("author_work_event_method.cross_cap_source_substitution_allowed", True),
        (
            "historical_observability.required_cutoff",
            "source_publication_date <= cutoff",
        ),
        (
            "historical_observability.cutoff_uses_canonical_spine_publication_day_only",
            False,
        ),
        (
            "historical_observability.token_or_field_past_observability_established",
            True,
        ),
        ("current_gate_state.author_entity_validity", "PASS"),
        ("current_gate_state.historical_content_identity", "VERIFIED"),
        ("current_gate_state.orcid_validation", "GROUND_TRUTH"),
        ("current_gate_state.candidate_eligibility", "PASS"),
        ("current_gate_state.g_minus_1_data_readiness", "PASS"),
        ("current_gate_state.g0_measurement_freeze", "PASS"),
        ("current_gate_state.empirical_g1_signature", "PASS"),
        ("current_gate_state.pass_g1", True),
        ("current_gate_state.scientific_effect", "ESTABLISHED"),
    ],
)
def test_author_entity_candidate_gates_and_effect_promotions_are_rejected(
    path: str, unsafe_value: Any
) -> None:
    config = copy.deepcopy(_load_config())
    _set_nested(config, path, unsafe_value)
    with pytest.raises(AssertionError):
        _validate_config(config)


def test_work_topic_and_citation_are_nonparent_siblings_with_exact_live_hashes() -> (
    None
):
    config = _load_config()
    topic = config["parallel_siblings"]["work_topic_relation"]
    citation = config["downstream_sibling_blockers"]["citation_available_at"]
    for relation, path_key, sha_key in (
        (topic, "config_path", "config_sha256"),
        (topic, "tool_path", "tool_sha256"),
        (citation, "design_config_path", "design_config_sha256"),
        (
            citation,
            "synthetic_execution_script",
            "synthetic_execution_script_sha256",
        ),
    ):
        assert _sha256(REPO_ROOT / relation[path_key]) == relation[sha_key]
    assert "citation_available_at" not in config["parents"]
    assert "citation_available_at" not in config["required_parent_execution_order"]
    assert citation["production_receipt_exists"] is False
    assert citation["g_minus_1_dependency_state"] == (
        "BLOCKED_CITATION_PRODUCTION_RECEIPT_NOT_AVAILABLE"
    )
    assert citation["historical_edge_observation_time_established"] is False
    assert citation["outcome_maturation_state"] == "PENDING_NOT_DEFINED"


@pytest.mark.parametrize(
    ("field", "legacy_value"),
    [
        ("design_schema_version", "openalex_g1_citation_available_at_design.v1"),
        (
            "design_config_path",
            "experiments/configs/openalex_g1_citation_available_at_v0.yaml",
        ),
        (
            "design_config_sha256",
            "51f5044459514b5150a8e24b2e4daf05575e06fc366d412bc3d046d87ba3c798",
        ),
        (
            "synthetic_execution_script",
            "scripts/audit_openalex_citation_available_at.py",
        ),
        (
            "synthetic_execution_script_sha256",
            "0be5444c9b72aea95251e609543a0da44d1ff95a432c839b674921e639409bde",
        ),
        ("design_status", "SYNTHETIC_CITATION_EDGE_CHRONOLOGY_ONLY"),
    ],
)
def test_legacy_citation_surface_substitution_is_rejected(
    field: str, legacy_value: Any
) -> None:
    config = copy.deepcopy(_load_config())
    config["downstream_sibling_blockers"]["citation_available_at"][field] = legacy_value
    with pytest.raises(AssertionError):
        _validate_config(config)


def test_citation_hard_parent_or_execution_order_injection_is_rejected() -> None:
    config = copy.deepcopy(_load_config())
    citation = copy.deepcopy(
        config["downstream_sibling_blockers"]["citation_available_at"]
    )
    config["parents"]["citation_available_at"] = citation
    config["required_parent_execution_order"].insert(3, "citation_available_at")
    with pytest.raises(AssertionError):
        _validate_config(config)

    config = copy.deepcopy(_load_config())
    config["downstream_sibling_blockers"]["citation_available_at"][
        "hard_parent_of_author_token_history"
    ] = True
    with pytest.raises(AssertionError):
        _validate_config(config)


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        ("future_runner_obligations.production_runner_path", "scripts/audit.py"),
        ("future_runner_obligations.production_output_abi", "FROZEN_V1"),
        ("future_runner_obligations.author_work_event_artifact_schema", "events.v1"),
        (
            "future_runner_obligations.current_v1_freezes_any_production_output_or_event_artifact",
            True,
        ),
        (
            "future_runner_obligations.hidden_cross_record_closure_obligations.v1_claims_hidden_closures_ran",
            True,
        ),
        (
            "future_runner_obligations.temporal_metric_partition.future_author_or_canonical_scan_may_recompute_created_updated_11_keys",
            True,
        ),
        (
            "future_runner_obligations.terminal_and_resume_revalidation.requirements_implemented_by_v1",
            True,
        ),
        (
            "future_runner_obligations.future_closure_evidence_strategy.role",
            "NEW_ROW_LEVEL_TIME_PARENT",
        ),
        (
            "future_component_pass_preconditions."
            "require_one_frozen_closure_evidence_strategy",
            False,
        ),
        (
            "future_component_pass_preconditions."
            "may_promote_only_bounded_author_token_accounting_scope",
            False,
        ),
        (
            "future_component_pass_preconditions.hypothetical_status",
            "PASS_UNBOUNDED_AUTHOR_ENTITY_AND_EFFECT",
        ),
        (
            "future_component_pass_preconditions.require_canonical_parent_state",
            "PASS_WITHOUT_RECEIPT",
        ),
    ],
)
def test_future_only_output_event_closure_and_resume_claims_cannot_be_promoted(
    path: str, unsafe_value: Any
) -> None:
    config = copy.deepcopy(_load_config())
    _set_nested(config, path, unsafe_value)
    with pytest.raises(AssertionError):
        _validate_config(config)


def test_document_matches_the_design_only_claim_boundary_and_no_results_exist() -> None:
    config = _load_config()
    text = DOC_PATH.read_text(encoding="utf-8")
    for phrase in (
        "Status: `DESIGN_ONLY`",
        "Production status: `PRODUCTION_NOT_RUN`",
        "temporal-v3-only",
        "497",
        "38.76968115%",
        "UNVERIFIED_ROW_COUNT_UPPER_BOUND",
        "tokens, not persons",
        "BLOCKED_CANONICAL_PRODUCTION_RECEIPT_NOT_AVAILABLE",
        "receipt-bound author-work event artifact",
        "same-parent full rescan",
        "24",
        "11",
        "Hidden cross-record closures",
        "terminal resume revalidation",
        "citation `available_at`",
        "downstream sibling blocker",
        "G-1 and G0 remain `PENDING`",
        "empirical G1 remains `NOT_RUN`",
        "No executable, synthetic result, production result, or receipt exists",
    ):
        assert phrase in text
    assert config["lifecycle"] == LIFECYCLE
    assert not (
        REPO_ROOT / "scripts/audit_openalex_work_side_author_history_v1.py"
    ).exists()
    assert (
        list(
            (REPO_ROOT / "experiments/evidence/results").glob(
                "openalex_g1_work_side_author_history_v1*"
            )
        )
        == []
    )
