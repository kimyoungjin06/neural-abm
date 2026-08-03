from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import pytest
import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml"
)
DOC_PATH = (
    REPO_ROOT / "docs/case-studies/researcher-pivot/"
    "openalex-canonical-work-time-spine-v1-audit-design.md"
)
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v2.py"
RESULT_GLOB = "openalex_g1_canonical_work_time_spine_v1*"
LEGACY_SURFACES = {
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml": (
        "192d78e3df094121e63922a6eaf20f57a57af41bc174c0fdbbcc347e3eaed385"
    ),
    REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine.py": (
        "827bab5a6edcd9a574149203c68c77ae5178a8da210499b81f05a388ace5c0dc"
    ),
    REPO_ROOT / "tests/test_openalex_canonical_work_time_spine.py": (
        "cf2e091136b448a2fa3705664c1164bb2c3d3ab1cf971d663e03beab1cbb42f1"
    ),
    REPO_ROOT / "tests/test_openalex_canonical_work_time_spine_design.py": (
        "16a3965a09aa31b7160ca96e26eb954dbd2d792bd5e4b18a8c56901d5cee7df7"
    ),
    REPO_ROOT / "docs/case-studies/researcher-pivot/"
    "openalex-canonical-work-time-spine-audit-design.md": (
        "c75d9186ef9ffe473b50dc21343226d3deba7e626be468f21d8ce8b5b6097dab"
    ),
}


class UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that fails closed on duplicate mapping keys."""


def _construct_unique_mapping(
    loader: UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_unique(text: str) -> Any:
    return yaml.load(text, Loader=UniqueKeyLoader)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config() -> dict[str, Any]:
    value = _load_yaml_unique(CONFIG_PATH.read_text(encoding="utf-8"))
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


def _validate_claim_boundary(config: dict[str, Any]) -> None:
    assert config["lifecycle"] == {
        "design_status": "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN",
        "production_status": "PRODUCTION_NOT_RUN",
        "production_executable": False,
        "synthetic_executable": True,
        "execution_script": "scripts/audit_openalex_canonical_work_time_spine_v2.py",
        "production_result_exists": False,
        "synthetic_contract_result_repository_exists": False,
        "pass_result_allowed_in_design_artifact": False,
        "production_implementation_requires_new_review": True,
    }
    assert config["synthetic_implementation"] == {
        "scope": "SYNTHETIC_FIXTURE_ONLY",
        "status": (
            "PASS_SYNTHETIC_CANONICAL_WORK_TIME_SPINE_TEMPORAL_V3_PARENT_CONTRACT_ONLY"
        ),
        "production_status": "PRODUCTION_NOT_RUN",
        "production_authorized": False,
        "repository_production_config_refused_by_runner": True,
        "production_root_id_refused_by_runner": True,
        "explicit_synthetic_fixture_flag_required": True,
        "cli_source_argument": "--synthetic-source-root",
        "generic_source_root_argument_rejected": True,
        "source_root_id_prefix_required": "synthetic-",
        "resolved_source_path_must_not_contain_production_root_id": True,
        "hard_max_input_tasks": 8,
        "hard_max_footer_rows": 10000,
        "hard_max_source_bytes": 64 * 1024 * 1024,
        "hard_max_retained_rows": 10000,
        "hard_caps_validated_from_live_parquet_footers_before_any_column_payload_scan": (
            True
        ),
        "temporal_parent_anchor_override_allowed_only_for_closed_synthetic_fixture": (
            True
        ),
        "exact_partition_surface_exercised": 4096,
        "partition_count_override_allowed": False,
        "receipt_artifact_count_formula": "6 + 2*N + 2*4096",
        "production_artifact_invariant": {
            "input_tasks": 493,
            "artifacts": 9184,
        },
        "synthetic_schema_and_status_must_not_equal_production": True,
    }
    assert config["component_decision"] == {
        "architecture": "STANDALONE_COMPONENT_AFTER_TEMPORAL_V3_PASS",
        "modify_temporal_v3": False,
        "future_scan_fusion_deferred": True,
        "reason": (
            "preserve_narrow_temporal_audit_provenance_and_separate_large_"
            "local_data_product"
        ),
    }
    assert config["required_parent_execution_order"] == [
        "current_content_baseline",
        "main_temporal_validity",
        "canonical_work_time_spine",
    ]

    parents = config["parents"]
    content = parents["current_content_baseline"]
    temporal = parents["main_temporal_validity"]
    assert content["required_status"] == "PASS_CURRENT_CONTENT_BASELINE_COMPONENT"
    assert content["required_run_state"] == "COMPLETE"
    assert content["required_full_scope"] is True
    assert temporal["required_status"] == (
        "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_"
        "MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE"
    )
    assert temporal["required_state_status"] == "COMPLETE_PENDING_RECEIPT"
    assert temporal["required_completion_receipt_exists"] is True
    assert temporal["required_completion_contract"] == (
        "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
    )
    assert temporal["required_full_scope"] is True
    assert temporal["current_state"] == (
        "FULL_TEMPORAL_V3_PASS_EVIDENCE_BOUND_SYNTHETICALLY_PRODUCTION_NOT_RUN"
    )
    assert temporal["required_tool_sha256"] == (
        "76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d"
    )
    assert temporal["require_exact_receipt_artifact_set"] is True
    assert temporal["exact_total_receipt_artifact_count"] == 497

    binding = config["parent_cross_binding"]
    assert binding["mode_normalization_equation"] == (
        "content_mode == stat_imode(temporal_mode) == (temporal_mode & 0o7777)"
    )
    assert binding["require_content_and_temporal_relative_path_sets_exactly_equal"]

    classifier = config["disposition_classifier"]
    assert classifier["invalid_rows_receive_no_disposition_code"] is True
    assert classifier[
        "require_no_staging_or_partition_publication_after_any_rejected_row"
    ]

    staging = config["staging_record_abi"]
    assert staging["resolution"] == "OPENALEX_WORK_TIME_STAGING_RECORD_LE16_V1"
    assert staging["record_bytes"] == 16
    assert staging["reserved_padding_required_value"] == 0

    partition = config["partition_contract"]
    assert partition["canonical_sorted_record_bridge"][
        "require_pre_post_digest_equality"
    ]
    assert partition["writer_abi"]["store_arrow_schema"] is False
    assert partition["writer_abi"]["resume_requires_exact_runtime_fingerprint"]

    completion = config["completion_contract"]
    assert completion["receipt_exact_artifact_count_full"] == 9184
    assert completion["failure_state_status"] == (
        "FINALIZATION_FAILED_NO_VALID_RECEIPT"
    )
    assert completion["failure_must_not_leave_promotable_terminal_surface"] is True

    resources = config["resource_envelope"]
    assert resources["disk_safety_multiplier_milli_minimum"] == 1500
    assert resources["resume_may_subtract_only_rehashed_valid_completed_artifacts"]

    limitation = config["row_level_parent_artifact_limitation"]
    assert limitation["temporal_v3_materializes_row_level_dispositions"] is False
    assert limitation["temporal_v3_materializes_row_level_disposition_digest"] is (
        False
    )
    assert limitation[
        "direct_row_by_row_digest_equality_to_temporal_parent_possible"
    ] is (False)
    assert limitation["spine_creates_first_row_level_disposition_commitment"] is True
    assert limitation["forbidden_claim"] == (
        "row_level_artifact_identity_to_temporal_v3_output"
    )

    privacy = config["release_privacy"]
    assert privacy["individual_work_ids_repository_allowed"] is False
    assert privacy["individual_work_ids_release_allowed"] is False
    assert privacy["spine_partitions_repository_allowed"] is False
    assert privacy["spine_partitions_release_allowed"] is False

    consumer = config["consumer_contract"]["citation_chronology_spine_reuse"]
    assert consumer["main_source_files_opened_by_citation_component"] == 0
    assert consumer["main_source_bytes_scanned_by_citation_component"] == 0
    assert consumer["synthetic_consumer_test_must_fail_on_any_main_source_open"] is (
        True
    )

    gates = config["future_success_contract"]["required_full_gate_states"]
    assert gates["canonical_work_time_spine_component"] == "PASS"
    assert gates["row_level_spine_commitment"] == "CREATED_CURRENT_SNAPSHOT"
    assert gates["row_level_temporal_parent_artifact_identity"] == (
        "NOT_AVAILABLE_TEMPORAL_V3"
    )
    assert gates["historical_content_identity"] == "UNVERIFIED"
    assert gates["raw_source_id_validity"] == "BLOCKED"
    assert gates["w_zero_entity_validity"] == "UNRESOLVED"
    assert gates["citation_edge_chronology"] == "PENDING_NOT_IN_THIS_COMPONENT"
    assert gates["author_entity_validity"] == "PENDING"
    assert gates["candidate_eligibility"] == "PENDING"
    assert gates["g_minus_1_data_readiness"] == "PENDING"
    assert gates["g0_measurement_freeze"] == "PENDING"
    assert gates["empirical_g1_signature"] == "NOT_RUN"
    assert gates["pass_g1"] is False
    assert gates["scientific_effect"] == "NOT_ESTABLISHED"

    claim = config["claim_boundary"]
    required_phrases = (
        "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN",
        "standalone current-snapshot canonical work-time spine",
        "synthetic executable validates",
        "does not publish a row-level disposition artifact or digest",
        "cannot claim direct row-by-row artifact identity",
        "No production run or result exists",
    )
    assert all(phrase in claim for phrase in required_phrases)


def test_yaml_loader_rejects_duplicate_keys_and_repository_config_is_unique() -> None:
    assert _load_config()["schema_version"] == (
        "openalex_g1_canonical_work_time_spine_design.v2"
    )
    with pytest.raises(ConstructorError, match="duplicate key 'status'"):
        _load_yaml_unique("status: DESIGN_ONLY\nstatus: PASS\n")
    with pytest.raises(ConstructorError, match="duplicate key 'nested'"):
        _load_yaml_unique("outer:\n  nested: 1\n  nested: 2\n")


def test_legacy_v0_surfaces_are_preserved_byte_exact() -> None:
    assert {path: _sha256(path) for path in LEGACY_SURFACES} == LEGACY_SURFACES


def test_lifecycle_parents_completion_abi_and_claim_boundary_are_frozen() -> None:
    _validate_claim_boundary(_load_config())


def test_parent_cross_binding_has_exact_six_and_nine_field_abis() -> None:
    config = _load_config()
    binding = config["parent_cross_binding"]
    assert binding["content_source_stat_keyset"] == [
        "device",
        "inode",
        "mode",
        "bytes",
        "mtime_ns",
        "ctime_ns",
    ]
    assert binding["temporal_source_rich_identity_keyset"] == [
        "device",
        "inode",
        "mode",
        "uid",
        "gid",
        "links",
        "bytes",
        "mtime_ns",
        "ctime_ns",
    ]
    assert len(binding["content_source_stat_keyset"]) == 6
    assert len(binding["temporal_source_rich_identity_keyset"]) == 9
    assert binding["content_mode_representation"] == "permission_bits"
    assert binding["temporal_mode_representation"] == "full_stat_mode"
    assert binding["mode_normalization_equation"] == (
        "content_mode == stat_imode(temporal_mode) == (temporal_mode & 0o7777)"
    )
    assert binding["temporal_mode_permission_mask_octal"] == "07777"
    assert binding["require_content_task_paths_unique"] is True
    assert binding["require_temporal_task_paths_unique"] is True
    assert binding["require_content_and_temporal_relative_path_sets_exactly_equal"]
    assert binding["require_content_six_fields_equal_normalized_temporal_projection"]
    assert binding[
        "require_current_nine_field_stat_equal_temporal_identity_pre_and_post_scan"
    ]
    assert binding["require_all_493_tasks_bound_to_both_parents"] is True
    assert binding["verify_source_identity_before_and_after_each_file"] is True
    assert binding["verify_all_parents_again_at_finalization"] is True
    assert binding["current_baseline_only_not_historical_identity"] is True

    temporal = config["parents"]["main_temporal_validity"]
    assert temporal["required_full_task_count"] == 493
    assert temporal["required_raw_rows"] == 492293207
    assert temporal["required_source_bytes"] == 150062449219
    assert temporal["required_config_sha256"] == (
        "672940908090fedd0b562dd938ad4583f5e8238830fc1ff5b613be9618f033f7"
    )


def test_temporal_parent_exact_schema_tool_receipt_and_artifact_abi_is_frozen() -> None:
    temporal = _load_config()["parents"]["main_temporal_validity"]
    assert temporal["required_tool_path"] == (
        "scripts/audit_openalex_main_temporal_validity_v3.py"
    )
    assert temporal["required_tool_sha256"] == (
        "76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d"
    )
    assert {
        key: temporal[key]
        for key in (
            "required_plan_schema_version",
            "required_checkpoint_schema_version",
            "required_report_schema_version",
            "schema_version",
            "required_state_schema_version",
        )
    } == {
        "required_plan_schema_version": ("openalex_g1_main_temporal_validity_plan.v3"),
        "required_checkpoint_schema_version": (
            "openalex_g1_main_temporal_validity_checkpoint.v3"
        ),
        "required_report_schema_version": (
            "openalex_g1_main_temporal_validity_report.v3"
        ),
        "schema_version": "openalex_g1_main_temporal_validity_outputs.v3",
        "required_state_schema_version": (
            "openalex_g1_main_temporal_validity_state.v3"
        ),
    }
    assert temporal["exact_receipt_keyset"] == [
        "schema_version",
        "status",
        "result_fingerprint_sha256",
        "task_fingerprint_sha256",
        "artifacts",
        "completion_contract",
        "operator_paths_redacted",
    ]
    assert temporal["exact_fixed_receipt_artifact_paths"] == [
        "audit_plan.json",
        "run_state.json",
        "main_temporal_validity_audit.json",
        "main_temporal_validity_audit.md",
    ]
    assert temporal["checkpoint_artifact_directory"] == "checkpoints/main"
    assert temporal["checkpoint_artifact_filename_derivation"] == (
        "sha256_utf8_relative_source_path"
    )
    assert temporal["checkpoint_artifact_filename_pattern"] == ("^[0-9a-f]{64}[.]json$")
    assert temporal["exact_checkpoint_artifact_count"] == 493
    assert temporal["exact_total_receipt_artifact_count"] == 497
    assert temporal["report_result_fingerprint_payload_keyset"] == [
        "status",
        "pilot_local_contract_clean",
        "gate_status",
        "scope",
        "metrics",
        "publication_year_extrema",
        "aggregate_regression",
        "violations",
        "accounting_contract",
        "created_updated_relation_diagnostic",
        "checkpoint_evidence",
        "source_identity_evidence",
        "content_baseline_parent",
        "task_fingerprint_sha256",
        "claim_boundary",
    ]
    for key in (
        "require_exact_receipt_artifact_set",
        "require_every_receipt_artifact_relative_path_key_equality",
        "require_every_receipt_artifact_bytes_and_sha256",
        "require_plan_task_fingerprint_recomputed_from_canonical_plan_without_fingerprint",
        "require_report_result_fingerprint_recomputed_from_frozen_report_payload",
        "require_checkpoint_self_hash_metrics_hash_task_and_source_binding",
        "require_checkpoint_tree_recomputed_from_exact_checkpoint_set",
        "require_source_tree_recomputed_from_temporal_plan",
    ):
        assert temporal[key] is True


def test_temporal_v3_full_pass_anchors_and_nonblocking_order_profile_are_exact() -> (
    None
):
    temporal = _load_config()["parents"]["main_temporal_validity"]
    assert temporal["parent_version_policy"] == (
        "TEMPORAL_V3_PARENT_ONLY_NO_V1_OR_V2_FALLBACK"
    )
    assert temporal["required_config_path"] == (
        "experiments/configs/openalex_g1_main_temporal_validity_v2.yaml"
    )
    assert temporal["required_receipt_sha256"] == (
        "2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe"
    )
    assert temporal["required_report_sha256"] == (
        "de98be73fb0365672cec7ee770c40de090ae012d45a0422faebd34c0b9d19200"
    )
    assert temporal["required_exact_task_fingerprint_sha256"] == (
        "5d86a797481a14e95d08ecbe333bb207ff476bf7dc49886aec1db7b5b366001d"
    )
    assert temporal["required_exact_result_fingerprint_sha256"] == (
        "a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5"
    )
    assert temporal["required_exact_checkpoint_tree_sha256"] == (
        "725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392"
    )
    assert temporal["required_exact_source_tree_sha256"] == (
        "da3fa570a5fb49a36668dffdee008c2fea4d4b6fd5a3f7f17873df78b603b2a0"
    )
    assert {
        key: temporal[key]
        for key in (
            "required_raw_rows",
            "required_quarantined_rows",
            "required_retained_rows",
            "required_metadata_parsed_pair_rows",
            "required_created_lte_updated_rows",
            "required_created_gt_updated_rows",
            "required_future_metadata_field_values",
            "required_created_updated_null_rows",
            "required_created_date_parse_error_rows",
            "required_updated_date_parse_error_rows",
            "required_snapshot_year_on_or_before_rows",
            "required_snapshot_year_after_snapshot_rows",
        )
    } == {
        "required_raw_rows": 492293207,
        "required_quarantined_rows": 1,
        "required_retained_rows": 492293206,
        "required_metadata_parsed_pair_rows": 492293206,
        "required_created_lte_updated_rows": 492293203,
        "required_created_gt_updated_rows": 3,
        "required_future_metadata_field_values": 0,
        "required_created_updated_null_rows": 0,
        "required_created_date_parse_error_rows": 0,
        "required_updated_date_parse_error_rows": 0,
        "required_snapshot_year_on_or_before_rows": 11509999,
        "required_snapshot_year_after_snapshot_rows": 6736,
    }
    assert temporal["created_updated_order_profile_is_descriptive_nonblocking"]
    assert set(temporal["forbidden_prior_parent_sha256"].values()) == {
        "f6eec88337fb8482400ad3a25c7b7b7099b1613d575e0c29c6c26575e8612780",
        "d208b45cc87de577726a8310a899eb66882a519bae29df5f7dbbf7833562b172",
        "b4b39c016ba4c6867a3e08366f70f6125230c2a0a18c1e8c45d40826f373f563",
        "ec59f3c649baa4a6a5333daace92a5e46f6492fad62e4c1ed42a1702394425f4",
        "11a668b8f2b10c83e2d36b3f5d2cd16f1e5f86b41ef64f58c79f10cabebc30e8",
        "b86f2496bbf7258c4e981f9a7ad8b1618fdcdc34a7385de628b133e0082a89dc",
    }


def test_four_column_schema_and_six_value_enum_are_exact() -> None:
    schema = _load_config()["spine_schema"]
    assert schema["field_order"] == [
        "work_id_numeric",
        "publication_day",
        "publication_year",
        "temporal_disposition",
    ]
    fields = schema["fields"]
    assert fields["work_id_numeric"]["physical_type"] == "uint64"
    assert fields["work_id_numeric"]["nullable"] is False
    assert fields["publication_day"]["physical_type"] == "int32"
    assert fields["publication_day"]["nullable"] is True
    assert fields["publication_year"]["physical_type"] == "uint16"
    assert fields["publication_year"]["nullable"] is True
    assert fields["temporal_disposition"]["physical_type"] == "uint8"
    assert fields["temporal_disposition"]["nullable"] is False
    assert schema["disposition_enum"] == {
        0: "BOTH_NULL_EXCLUDED",
        1: "VALID_PRE_1900_EXCLUDED",
        2: "USABLE_1900_BEFORE_SNAPSHOT_YEAR",
        3: "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT",
        4: "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED",
        5: "VALID_POST_SNAPSHOT_YEAR_EXCLUDED",
    }
    assert len(schema["disposition_enum"]) == 6
    assert schema["consumer_must_use_disposition_not_non_null_date_as_eligibility"]
    assert schema["raw_work_id_string_materialized"] is False
    assert schema["created_date_materialized"] is False
    assert schema["updated_date_materialized"] is False


def test_classifier_predicates_fail_before_staging_and_parent_metrics_are_exact() -> (
    None
):
    classifier = _load_config()["disposition_classifier"]
    assert classifier["resolution"] == (
        "FROZEN_WORK_TIME_SPINE_DISPOSITION_CLASSIFIER_V1"
    )
    assert classifier["snapshot_date"] == "2026-03-30"
    assert classifier["pre_1900_boundary_date_inclusive"] == "1900-01-01"
    assert classifier["evaluation_order"] == [
        "exclude_frozen_w_minus_1_before_validation_and_classification",
        "reject_invalid_retained_rows_before_staging",
        "BOTH_NULL_EXCLUDED",
        "VALID_PRE_1900_EXCLUDED",
        "USABLE_1900_BEFORE_SNAPSHOT_YEAR",
        "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT",
        "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED",
        "VALID_POST_SNAPSHOT_YEAR_EXCLUDED",
    ]
    assert classifier["predicates"] == {
        "BOTH_NULL_EXCLUDED": ("publication_date_is_null_and_publication_year_is_null"),
        "VALID_PRE_1900_EXCLUDED": (
            "publication_pair_valid_and_publication_day_before_1900_01_01"
        ),
        "USABLE_1900_BEFORE_SNAPSHOT_YEAR": (
            "publication_pair_valid_and_day_on_or_after_1900_01_01_and_"
            "derived_year_before_2026"
        ),
        "USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT": (
            "publication_pair_valid_and_derived_year_equals_2026_and_"
            "day_on_or_before_2026_03_30"
        ),
        "VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED": (
            "publication_pair_valid_and_derived_year_equals_2026_and_"
            "day_after_2026_03_30"
        ),
        "VALID_POST_SNAPSHOT_YEAR_EXCLUDED": (
            "publication_pair_valid_and_derived_year_after_2026"
        ),
    }
    assert classifier["reject_before_staging_metrics"] == [
        "publication_exactly_one_null_rows",
        "publication_date_parse_error_rows",
        "publication_year_invalid_rows",
        "publication_date_year_mismatch_rows",
    ]
    assert classifier["retained_identifier_reject_before_staging_metrics"] == [
        "null_id_rows",
        "unexpected_malformed_id_rows",
    ]
    assert classifier["invalid_rows_receive_no_disposition_code"] is True
    assert classifier["require_exactly_one_disposition_for_every_retained_valid_row"]
    assert classifier[
        "require_no_staging_or_partition_publication_after_any_rejected_row"
    ]
    assert classifier["temporal_parent_per_file_metric_keyset"] == [
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
    ]
    assert len(classifier["temporal_parent_per_file_metric_keyset"]) == 35
    assert (
        classifier["spine_recomputed_per_file_metric_keyset"]
        == classifier["temporal_parent_per_file_metric_keyset"][:24]
    )
    assert classifier["require_temporal_checkpoint_metric_keyset_exactly_v3"] is True
    assert (
        classifier[
            "created_updated_metrics_are_parent_validity_and_descriptive_evidence_not_"
            "recomputed_by_three_column_spine_scan"
        ]
        is True
    )
    assert classifier["require_per_file_publication_year_extrema_equality"] is True
    assert classifier["disposition_metrics_hash_payload"] == (
        "canonical_json_utf8_sorted_keys_compact_no_nan"
    )


def test_staging_record_offsets_sentinels_names_and_hash_framing_are_exact() -> None:
    staging = _load_config()["staging_record_abi"]
    assert staging["resolution"] == "OPENALEX_WORK_TIME_STAGING_RECORD_LE16_V1"
    assert staging["record_bytes"] == 16
    assert staging["endianness"] == "little"
    assert staging["field_layout"] == [
        {
            "name": "work_id_numeric",
            "offset": 0,
            "bytes": 8,
            "encoding": "uint64_le",
        },
        {
            "name": "publication_day",
            "offset": 8,
            "bytes": 4,
            "encoding": "int32_le",
        },
        {
            "name": "publication_year",
            "offset": 12,
            "bytes": 2,
            "encoding": "uint16_le",
        },
        {
            "name": "temporal_disposition",
            "offset": 14,
            "bytes": 1,
            "encoding": "uint8",
        },
        {
            "name": "reserved_padding",
            "offset": 15,
            "bytes": 1,
            "encoding": "zero_uint8",
        },
    ]
    assert staging["both_null_publication_day_sentinel"] == -(2**31)
    assert staging["both_null_publication_year_sentinel"] == 0
    assert staging["both_null_requires_disposition_code"] == 0
    assert staging["nonnull_rows_forbid_null_sentinels"] is True
    assert staging["reserved_padding_required_value"] == 0
    assert staging["shard_directory"] == "staging/main"
    assert staging["shard_filename_derivation"] == ("sha256_utf8_relative_source_path")
    assert staging["shard_filename_pattern"] == "^[0-9a-f]{64}[.]bin$"
    assert staging["exact_shard_count_full"] == 493
    digest = staging["canonical_record_hash"]
    assert digest["resolution"] == "LENGTH_FRAMED_SHA256_V1"
    assert digest["domain_utf8"] == "openalex-work-time-records-v1"
    assert digest["framing"] == (
        "u64le(domain_bytes) || domain_utf8 || u64le(record_count) || "
        "repeat(u64le(16) || record16)"
    )
    assert digest["chunking_independent"] is True
    projection = staging["source_projection_hash"]
    assert projection["framing"] == (
        "repeat(u64le(serialized_batch_byte_length) || serialized_arrow_record_batch)"
    )
    assert projection["projected_columns_exactly"] == [
        "id",
        "publication_date",
        "publication_year",
    ]
    assert projection["batch_size_bound_in_plan_and_task_fingerprint"] is True
    assert projection["pyarrow_runtime_fingerprint_bound_in_plan"] is True


def test_identifier_and_publication_accounting_close_exactly() -> None:
    config = _load_config()
    identifiers = config["identifier_accounting"]
    assert identifiers["raw_rows"] == 492293207
    assert identifiers["frozen_w_minus_1_quarantine_rows"] == 1
    assert identifiers["retained_rows"] == 492293206
    assert identifiers["raw_rows"] == (
        identifiers["frozen_w_minus_1_quarantine_rows"] + identifiers["retained_rows"]
    )
    assert identifiers["retained_distinct_numeric_ids"] == identifiers["retained_rows"]
    assert identifiers["duplicate_numeric_id_groups"] == 0
    assert identifiers["duplicate_numeric_row_excess"] == 0
    assert identifiers["minimum_numeric_id"] == 0
    assert identifiers["maximum_numeric_id"] == 7143276261
    assert identifiers["w_zero_rows_retained"] == 1
    assert identifiers["output_w_minus_1_rows"] == 0
    assert identifiers["w_zero_entity_validity"] == "UNRESOLVED"

    publication = config["publication_accounting"]
    fixed = publication["fixed_disposition_anchors"]
    assert fixed == {
        "BOTH_NULL_EXCLUDED": 27565532,
        "VALID_PRE_1900_EXCLUDED": 6692869,
        "USABLE_1900_BEFORE_SNAPSHOT_YEAR": 446514060,
        "VALID_POST_SNAPSHOT_YEAR_EXCLUDED": 4010,
    }
    assert publication["snapshot_year_total_rows"] == 11516735
    split = publication["snapshot_year_split"]
    assert split["USABLE_SNAPSHOT_YEAR_ON_OR_BEFORE_SNAPSHOT"] == 11509999
    assert split["VALID_SNAPSHOT_YEAR_AFTER_SNAPSHOT_EXCLUDED"] == 6736
    assert split["required_sum"] == publication["snapshot_year_total_rows"]
    assert split["frozen_from_exact_full_temporal_v3_pass_only"] is True
    assert (
        sum(fixed.values()) + publication["snapshot_year_total_rows"]
        == (identifiers["retained_rows"])
    )
    assert publication["required_publication_year_minimum"] == 1000
    assert publication["required_publication_year_maximum"] == 2050
    assert len(publication["required_zero_failure_metrics"]) == 4
    assert publication["w_zero_disposition"]["expected_category_precommitted"] is (
        False
    )


def test_partition_writer_checkpoint_resume_and_receipt_last_are_exact() -> None:
    config = _load_config()
    partition = config["partition_contract"]
    assert partition["algorithm"] == "exact_numeric_modulo"
    assert partition["partition_count"] == 4096
    assert partition["formula"] == "work_id_numeric_mod_4096"
    assert partition["assignment_total_and_single_valued"] is True
    assert partition["different_ids_may_share_partition"] is True
    assert partition["hash_collision_semantics_applicable"] is False
    assert partition["exact_file_count"] == 4096
    assert partition["every_partition_file_required"] is True
    assert partition["require_strictly_increasing_work_id_within_partition"] is True
    assert partition["require_global_exactly_once_membership"] is True
    assert partition["empty_partition_checkpoint_minimum_and_maximum"] is None
    assert partition["empty_partition_strict_sortedness_vacuously_true"] is True
    bridge = partition["canonical_sorted_record_bridge"]
    assert bridge == {
        "record_abi": "OPENALEX_WORK_TIME_STAGING_RECORD_LE16_V1",
        "ordering": "strictly_increasing_work_id_numeric",
        "pre_parquet_digest_source": (
            "verified_staging_records_assigned_by_exact_numeric_modulo"
        ),
        "post_parquet_digest_source": (
            "decoded_parquet_rows_reencoded_to_staging_record_abi"
        ),
        "digest_abi": "LENGTH_FRAMED_SHA256_V1",
        "require_pre_post_digest_equality": True,
        "require_pre_post_row_count_equality": True,
    }
    writer = partition["writer_abi"]
    assert writer == {
        "parquet_format_version": "2.6",
        "compression": "NONE",
        "dictionary_encoding": False,
        "write_statistics": False,
        "row_group_rows": 262144,
        "write_batch_size": 262144,
        "data_page_version": "1.0",
        "write_page_index": False,
        "write_page_checksum": False,
        "use_byte_stream_split": False,
        "store_arrow_schema": False,
        "sorting_columns": [],
        "combine_chunks_before_write": True,
        "input_arrow_schema_metadata": None,
        "bind_pyarrow_and_parquet_runtime_versions": True,
        "runtime_fingerprint_fields": [
            "pyarrow_version",
            "parquet_cpp_version",
            "python_implementation",
            "python_version",
            "platform_machine",
        ],
        "bind_runtime_fingerprint_to_plan_task_and_receipt": True,
        "resume_requires_exact_runtime_fingerprint": True,
    }
    assert partition["arrow_schema"] == [
        {"name": "work_id_numeric", "arrow_type": "uint64", "nullable": False},
        {"name": "publication_day", "arrow_type": "int32", "nullable": True},
        {"name": "publication_year", "arrow_type": "uint16", "nullable": True},
        {
            "name": "temporal_disposition",
            "arrow_type": "uint8",
            "nullable": False,
        },
    ]
    assert partition["parquet_schema"] == [
        {
            "name": "work_id_numeric",
            "physical_type": "INT64",
            "logical_type": "UINT_64",
            "repetition": "REQUIRED",
        },
        {
            "name": "publication_day",
            "physical_type": "INT32",
            "logical_type": "NONE",
            "repetition": "OPTIONAL",
        },
        {
            "name": "publication_year",
            "physical_type": "INT32",
            "logical_type": "UINT_16",
            "repetition": "OPTIONAL",
        },
        {
            "name": "temporal_disposition",
            "physical_type": "INT32",
            "logical_type": "UINT_8",
            "repetition": "REQUIRED",
        },
    ]
    assert partition["atomic_partition_publication_required"] is True

    execution = config["execution_plan"]
    assert execution["scan_stage"]["checkpoint_unit"] == "input_file"
    assert (
        "temporal_parent_checkpoint_sha256"
        in execution["scan_stage"]["checkpoint_binds"]
    )
    assert (
        "row_level_disposition_payload_sha256"
        in execution["scan_stage"]["checkpoint_binds"]
    )
    assert execution["partition_stage"]["checkpoint_unit"] == "output_partition"
    assert (
        "staging_shard_tree_sha256" in execution["partition_stage"]["checkpoint_binds"]
    )
    resume = execution["resume_policy"]
    assert resume["require_exact_existing_plan"] is True
    assert resume[
        "rehash_current_three_column_projection_before_reusing_scan_checkpoint"
    ]
    assert resume["verify_staging_shard_before_reuse"] is True
    assert resume["verify_partition_output_before_reuse"] is True
    assert resume["partition_count_or_mode_change_allowed"] is False
    assert resume["valid_completed_receipt_is_terminal"] is True
    assert resume["terminal_validation_occurs_before_any_source_payload_open"] is (True)
    assert resume[
        "terminal_validation_requires_exact_artifact_set_and_all_bytes_sha256"
    ]

    completion = config["completion_contract"]
    assert completion["pre_receipt_state"] == "COMPLETE_PENDING_RECEIPT"
    assert completion["receipt_written_last"] is True
    assert completion["receipt_completion_contract"] == (
        "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
    )
    assert completion["final_validation_rehashes_all_control_and_data_artifacts"]
    assert completion["receipt_exact_artifact_count_full"] == 9184
    assert completion["exact_scan_checkpoint_count_full"] == 493
    assert completion["exact_partition_checkpoint_count_full"] == 4096
    assert completion["exact_staging_shard_count_full"] == 493
    assert completion["exact_spine_partition_count_full"] == 4096
    assert completion["failure_invalidation_order"] == [
        "remove_authoritative_completion_receipt_first",
        "write_finalization_failed_state",
        "remove_final_report_and_markdown",
    ]
    assert completion["failure_state_status"] == (
        "FINALIZATION_FAILED_NO_VALID_RECEIPT"
    )
    assert completion["failure_state_omits_result_fingerprint_sha256"] is True
    assert completion["cleanup_errors_aggregated_and_raised_not_swallowed"] is True
    assert completion["failure_must_not_leave_promotable_terminal_surface"] is True
    assert completion["failure_removes_receipt_and_final_report_surfaces"] is True
    assert completion["resumable_checkpoints_preserved_on_failure"] is True
    assert completion["production_outputs_created_by_this_design"] is False


def test_cross_stage_conservation_and_exact_receipt_surface_are_frozen() -> None:
    config = _load_config()
    conservation = config["cross_stage_conservation"]
    assert conservation["required_equal_row_totals"] == {
        "raw_rows": 492293207,
        "quarantined_w_minus_1_rows": 1,
        "scan_retained_rows": 492293206,
        "staging_manifest_rows": 492293206,
        "partition_input_rows": 492293206,
        "parquet_manifest_rows": 492293206,
        "decoded_final_rows": 492293206,
    }
    for key in (
        "require_sum_scan_checkpoint_rows_equals_staging_manifest_rows",
        "require_sum_staging_shard_rows_equals_partition_input_rows",
        "require_each_verified_staging_record_assigned_to_exactly_one_partition",
        "require_partition_assignment_matches_work_id_numeric_mod_4096",
        "require_sum_all_4096_partition_rows_equals_decoded_final_rows",
        "require_partition_record_digest_pre_post_parquet_equality",
        "require_global_partition_digest_tree_bound_to_result_fingerprint",
        "require_decoded_disposition_counts_equal_scan_and_temporal_parent",
        "require_decoded_nullability_and_year_day_invariants",
        "require_decoded_w0_cross_tab_and_id_extrema",
        "require_no_duplicate_numeric_id_within_or_across_partitions",
    ):
        assert conservation[key] is True

    partition_binds = config["execution_plan"]["partition_stage"]["checkpoint_binds"]
    for key in (
        "pre_parquet_sorted_record_sha256",
        "decoded_parquet_record_sha256",
        "disposition_counts",
        "nullability_counts",
    ):
        assert key in partition_binds

    completion = config["completion_contract"]
    assert completion["exact_top_level_entries"] == [
        "audit.lock",
        "audit_plan.json",
        "run_state.json",
        "checkpoints",
        "staging",
        "spine",
        "staging_manifest.json",
        "work_time_spine_manifest.json",
        "work_time_spine_audit.json",
        "work_time_spine_audit.md",
        "work_time_spine_audit.outputs.json",
    ]
    assert completion["checkpoint_directories"] == {
        "scan": "checkpoints/scan/main",
        "partition": "checkpoints/partition",
    }
    assert completion["scan_checkpoint_filename_pattern"] == ("^[0-9a-f]{64}[.]json$")
    assert completion["partition_checkpoint_filename_pattern"] == (
        "^part-[0-9]{4}[.]json$"
    )
    assert completion["receipt_excludes"] == [
        "audit.lock",
        "work_time_spine_audit.outputs.json",
    ]
    assert completion["receipt_exact_keyset"] == [
        "schema_version",
        "status",
        "task_fingerprint_sha256",
        "result_fingerprint_sha256",
        "scan_checkpoint_tree_sha256",
        "staging_shard_tree_sha256",
        "partition_checkpoint_tree_sha256",
        "spine_data_tree_sha256",
        "artifacts",
        "completion_contract",
        "operator_paths_redacted",
    ]
    assert completion[
        "require_exact_published_file_set_equals_receipt_artifact_set_plus_exclusions"
    ]
    assert completion["require_every_receipt_artifact_path_bytes_sha256"] is True
    assert completion["receipt_exact_artifact_count_full"] == (
        6 + 493 + 4096 + 493 + 4096
    )


def test_resources_pilots_consumers_and_privacy_remain_bounded() -> None:
    config = _load_config()
    resources = config["resource_envelope"]
    assert resources["declared_input_files"] == 493
    assert resources["declared_input_rows"] == 492293207
    assert resources["declared_source_bytes"] == 150062449219
    assert resources["fixed_staging_record_bytes"] == 16
    assert resources["staging_payload_bytes_upper_anchor"] == 492293206 * 16
    assert resources["staging_payload_gib_rounded_3dp"] == 7.336
    assert resources["ram_gib_minimum_planning"] == 16
    assert resources["ram_gib_preferred_planning"] == 32
    assert resources["temporary_ssd_gib_minimum"] == 64
    assert resources["temporary_ssd_gib_preferred"] == 96
    assert resources["require_live_preflight_before_full"] is True
    assert resources["disk_safety_multiplier_milli_minimum"] == 1500
    assert resources["parquet_bytes_per_row_upper_bound"] == 32
    assert resources["parquet_fixed_bytes_per_partition_upper_bound"] == 131072
    assert resources["partition_spool_bytes_upper_bound"] == 492293206 * 16
    assert resources["control_artifacts_bytes_upper_bound"] == 2 * 1024**3
    assert resources["disk_reserve_bytes"] == 8 * 1024**3
    assert resources["atomic_publish_headroom"] == ("largest_next_artifact_upper_bound")
    assert resources["full_required_free_disk_formula"] == (
        "ceil(1500 * (remaining_staging_bytes + remaining_partition_spool_bytes "
        "+ remaining_parquet_bytes_upper_bound + remaining_control_bytes + "
        "atomic_publish_headroom_bytes) / 1000) + disk_reserve_bytes"
    )
    assert resources["resume_may_subtract_only_rehashed_valid_completed_artifacts"]
    assert resources["ram_preflight_formula"] == (
        "max(projected_batch_bytes_upper_bound, "
        "partition_sort_chunk_bytes_upper_bound) + ram_reserve_bytes"
    )
    assert resources["require_partition_sort_chunk_bound_in_plan"] is True
    assert resources["require_projected_batch_bound_in_plan"] is True
    assert resources[
        "forbid_source_disk_competition_with_content_freeze_or_other_heavy_scan"
    ]
    assert resources["gpu_used"] is False

    rules = config["pilot_and_full_rules"]
    assert rules["max_files"]["promotion_allowed"] is False
    assert rules["deterministic_id_bucket"]["require_all_493_input_files"] is True
    assert rules["deterministic_id_bucket"]["promotion_allowed"] is False
    assert rules["full"]["explicit_authorization_required"] is True
    assert rules["full"]["require_all_493_input_files"] is True
    assert rules["full"]["require_all_492293206_retained_rows_materialized"]
    assert rules["full"]["require_all_4096_partitions"] is True
    assert rules["full"]["require_all_parent_full_receipts"] is True

    consumers = config["consumer_contract"]
    citation = consumers["citation_chronology_spine_reuse"]
    assert citation["full_spine_receipt_required"] is True
    assert citation["required_spine_rows"] == 492293206
    assert citation["main_source_files_opened_by_citation_component"] == 0
    assert citation["main_source_bytes_scanned_by_citation_component"] == 0
    assert citation["synthetic_consumer_test_must_fail_on_any_main_source_open"]
    author = consumers["work_side_author_history"]
    assert author["publication_date_reparse_can_be_eliminated"] is True
    assert author["complete_main_source_scan_eliminated"] is False
    assert consumers["work_topic_relation"]["main_scan_eliminated"] is False
    assert consumers["universal_work_metadata_spine_expansion_allowed_in_v0"] is (False)

    privacy = config["release_privacy"]
    assert privacy["public_summary_allows_aggregates_and_checksums_only"] is True
    assert privacy["individual_work_ids_repository_allowed"] is False
    assert privacy["individual_work_ids_release_allowed"] is False
    assert privacy["spine_partitions_repository_allowed"] is False
    assert privacy["spine_partitions_release_allowed"] is False
    assert privacy["operator_local_detailed_outputs_mode"] == "0600"
    assert privacy["operator_local_directory_mode"] == "0700"
    assert set(privacy["release_forbidden_operator_local_directory_basenames"]) == {
        "staging",
        "spine",
    }
    assert set(privacy["release_forbidden_operator_local_path_fragments"]) == {
        "checkpoints/scan/main",
        "checkpoints/partition",
    }
    assert set(privacy["release_forbidden_operator_local_basenames"]) == {
        "staging_manifest.json",
        "work_time_spine_manifest.json",
    }


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        ("lifecycle.production_status", "PASS"),
        ("lifecycle.production_executable", True),
        ("synthetic_implementation.production_authorized", True),
        ("synthetic_implementation.hard_max_input_tasks", 493),
        ("synthetic_implementation.generic_source_root_argument_rejected", False),
        ("synthetic_implementation.partition_count_override_allowed", True),
        ("component_decision.modify_temporal_v3", True),
        ("parents.main_temporal_validity.required_state_status", "COMPLETE"),
        ("parents.main_temporal_validity.required_tool_sha256", "0" * 64),
        ("parents.main_temporal_validity.require_exact_receipt_artifact_set", False),
        ("parent_cross_binding.mode_normalization_equation", "modes_equal"),
        (
            "parent_cross_binding."
            "require_content_and_temporal_relative_path_sets_exactly_equal",
            False,
        ),
        ("disposition_classifier.invalid_rows_receive_no_disposition_code", False),
        (
            "disposition_classifier."
            "require_no_staging_or_partition_publication_after_any_rejected_row",
            False,
        ),
        ("staging_record_abi.record_bytes", 15),
        ("staging_record_abi.reserved_padding_required_value", 1),
        (
            "partition_contract.canonical_sorted_record_bridge."
            "require_pre_post_digest_equality",
            False,
        ),
        ("partition_contract.writer_abi.store_arrow_schema", True),
        (
            "partition_contract.writer_abi.resume_requires_exact_runtime_fingerprint",
            False,
        ),
        ("completion_contract.receipt_exact_artifact_count_full", 9183),
        ("completion_contract.failure_state_status", "COMPLETE"),
        (
            "completion_contract.failure_must_not_leave_promotable_terminal_surface",
            False,
        ),
        ("resource_envelope.disk_safety_multiplier_milli_minimum", 1000),
        (
            "resource_envelope."
            "resume_may_subtract_only_rehashed_valid_completed_artifacts",
            False,
        ),
        (
            "row_level_parent_artifact_limitation."
            "temporal_v3_materializes_row_level_dispositions",
            True,
        ),
        (
            "row_level_parent_artifact_limitation."
            "direct_row_by_row_digest_equality_to_temporal_parent_possible",
            True,
        ),
        (
            "consumer_contract.citation_chronology_spine_reuse."
            "main_source_files_opened_by_citation_component",
            493,
        ),
        (
            "future_success_contract.required_full_gate_states."
            "g_minus_1_data_readiness",
            "PASS",
        ),
        ("release_privacy.individual_work_ids_release_allowed", True),
    ],
)
def test_claim_validator_rejects_unsafe_promotions(
    path: str,
    unsafe_value: Any,
) -> None:
    config = copy.deepcopy(_load_config())
    _set_nested(config, path, unsafe_value)
    with pytest.raises(AssertionError):
        _validate_claim_boundary(config)


def test_document_config_and_synthetic_runtime_surfaces_are_synchronized() -> None:
    config = _load_config()
    text = DOC_PATH.read_text(encoding="utf-8")
    required_text = (
        "Status: `IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN`",
        "Production status: `PRODUCTION_NOT_RUN`",
        "Synthetic executable boundary",
        "6 + 2*N + 2*4096",
        "COMPLETE_PENDING_RECEIPT",
        "492,293,206",
        "Six-field and nine-field source cross-binding",
        "Minimal four-column schema",
        "4,096-partition",
        "LENGTH_FRAMED_SHA256_V1",
        "9,184 regular artifacts",
        "FINALIZATION_FAILED_NO_VALID_RECEIPT",
        "same-row-count",
        "does not publish a row-level disposition artifact",
        config["future_success_contract"]["full_component_status"],
        "main_source_files_opened_by_citation_component",
        "It uses no GPU",
    )
    assert all(item in text for item in required_text)
    assert SCRIPT_PATH.is_file()
    assert (
        config["lifecycle"]["execution_script"]
        == SCRIPT_PATH.relative_to(REPO_ROOT).as_posix()
    )
    assert config["completion_contract"][
        "production_outputs_created_by_this_design"
    ] is (False)
    result_dir = REPO_ROOT / "experiments/evidence/results"
    assert list(result_dir.glob(RESULT_GLOB)) == []
