from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml"
)
DOC_PATH = (
    REPO_ROOT
    / "docs/research/openalex-canonical-work-time-spine-v2-production-design.md"
)
RUNNER_PATH = REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v3.py"
RUNNER_SHA256_SENTINEL = "0" * 64
EXPECTED_RUNNER_SHA256 = (
    "bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280"
)
EXPECTED_SEMANTIC_CONFIG_SHA256 = (
    "3e21393832cc7c2624177fd7f73f0c1ddb80ece21dfd2b4d84e173a7f32f741a"
)
PRESERVED_SURFACES = {
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
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml": (
        "2caef77c8b3dc2ffcb0a35920c9e3527096ec3a9f31d20166eec3a3486f4196b"
    ),
    REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v2.py": (
        "6a2d1d87dd4a3e7d3c97056fabbe951a1528cfd62d8448acc74ae337f344b7b1"
    ),
    REPO_ROOT / "tests/test_openalex_canonical_work_time_spine_v2.py": (
        "c793169343d53abc425905ad137862744e6f6ccb8dfc681b6ce4d6d0f3d4eaf4"
    ),
    REPO_ROOT / "tests/test_openalex_canonical_work_time_spine_v2_design.py": (
        "f8d78a9c137ff6042aaccef3985971b801ba170e37e6775c3d2a748d556d7705"
    ),
    REPO_ROOT / "docs/case-studies/researcher-pivot/"
    "openalex-canonical-work-time-spine-v1-audit-design.md": (
        "5994ab36a7efb8fbab3dfd8ba36800c2364bec9943c17fc60c1d1d673acebd8d"
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


def _load_config() -> dict[str, Any]:
    value = _load_yaml_unique(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_config_sha256(config: dict[str, Any]) -> str:
    normalized = copy.deepcopy(config)
    runner = normalized["runner_contract"]
    assert isinstance(runner, dict)
    runner["required_tool_sha256"] = RUNNER_SHA256_SENTINEL
    payload = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _set_nested(value: dict[str, Any], path: str, replacement: Any) -> None:
    cursor = value
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor[part]
        assert isinstance(child, dict)
        cursor = child
    cursor[parts[-1]] = replacement


def _validate_nonpromotion_boundary(config: dict[str, Any]) -> None:
    lifecycle = config["lifecycle"]
    assert lifecycle == {
        "design_status": "IMPLEMENTED_PRODUCTION_CAPABLE_PRODUCTION_NOT_RUN",
        "production_status": "PRODUCTION_NOT_RUN",
        "production_executable": True,
        "execution_script": "scripts/audit_openalex_canonical_work_time_spine_v3.py",
        "production_result_exists": False,
        "production_receipt_exists": False,
        "production_output_created_by_this_contract": False,
        "full_execution_authorized_by_config_alone": False,
        "synthetic_v2_result_promotable": False,
        "pilot_or_test_result_promotable": False,
    }

    runner = config["runner_contract"]
    assert runner == {
        "required_tool_path": "scripts/audit_openalex_canonical_work_time_spine_v3.py",
        "required_tool_sha256": EXPECTED_RUNNER_SHA256,
        "semantic_config_fingerprint_normalization": {
            "serialization": "CANONICAL_JSON_SORTED_KEYS_COMPACT_UTF8",
            "normalized_field": "runner_contract.required_tool_sha256",
            "normalized_value": RUNNER_SHA256_SENTINEL,
            "all_other_config_fields_included": True,
        },
        "semantic_config_sha256_stored_in_config": False,
        "production_and_pilot_require_repository_default_config_path": True,
    }
    assert _sha256(RUNNER_PATH) == EXPECTED_RUNNER_SHA256
    assert _semantic_config_sha256(config) == EXPECTED_SEMANTIC_CONFIG_SHA256

    modes = config["execution_modes"]
    full = modes["full_production"]
    assert full["required_cli_flags"] == [
        "--execute",
        "--authorize-full",
        "--acknowledge-exclusive-source-scan",
    ]
    assert full["all_three_flags_required_together"] is True
    assert full["config_alone_never_authorizes_execution"] is True
    assert full["forbidden_cli_flags"] == ["--max-files", "--test-fixture"]
    assert full["source_batch_rows_exact"] == 262144
    assert full["partition_sort_chunk_rows_exact"] == 1048576
    assert full["batch_or_sort_bound_override_allowed"] is False

    pilot = modes["production_pilot"]
    assert pilot["status"] == (
        "PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE"
    )
    assert pilot["production_component_pass"] is False
    assert pilot["promotion_allowed"] is False
    assert pilot["full_authorization_flag_forbidden"] is True
    assert pilot["full_receipt_basename_forbidden"] is True
    assert pilot["maximum_source_files"] == 1
    assert pilot["selected_source_files_must_equal_maximum"] is True
    assert pilot["max_files_cli_value_exact"] == 1
    assert pilot["source_batch_rows_exact"] == 262144
    assert pilot["partition_sort_chunk_rows_exact"] == 1048576
    assert pilot["batch_or_sort_bound_override_allowed"] is False

    fixture = modes["closed_test_fixture"]
    assert fixture["exposure"] == "PYTHON_API_ONLY_NOT_CLI"
    assert fixture["cli_flag_accepted"] is False
    assert fixture["status"] == (
        "PASS_TEST_FIXTURE_CANONICAL_WORK_TIME_SPINE_EXTERNAL_MEMORY_CONTRACT_ONLY"
    )
    assert fixture["production_component_pass"] is False
    assert fixture["production_root_id_forbidden"] is True
    assert fixture["promotion_allowed"] is False
    assert fixture["source_batch_rows_hard_maximum"] == 262144
    assert fixture["partition_sort_chunk_rows_hard_maximum"] == 1048576
    assert fixture["positive_values_required"] is True
    assert modes["schema_and_status_sets_pairwise_disjoint"] is True

    production = config["production_output_abi"]
    assert production["full_component_status"] == (
        "PASS_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_COMPONENT_WITH_"
        "TEMPORAL_V3_PARENT"
    )
    assert production["production_component_pass_true_only_for_exact_full_status"]
    assert production["production_schemas_forbidden_in_pilot_and_test_modes"]
    production_schemas = set(production["schemas"].values())
    assert set(production["schemas"]) == {
        "plan",
        "scan_checkpoint",
        "partition_checkpoint",
        "staging_manifest",
        "spine_manifest",
        "preflight_evidence",
        "report",
        "state",
        "outputs",
    }
    assert set(pilot["schemas"]) == {
        "plan",
        "scan_checkpoint",
        "staging_manifest",
        "preflight_evidence",
        "report",
        "state",
        "outputs",
    }
    assert set(fixture["schemas"]) == {
        "plan",
        "scan_checkpoint",
        "partition_checkpoint",
        "staging_manifest",
        "spine_manifest",
        "preflight_evidence",
        "report",
        "state",
        "outputs",
    }
    pilot_schemas = set(pilot["schemas"].values())
    fixture_schemas = set(fixture["schemas"].values())
    assert production_schemas.isdisjoint(pilot_schemas)
    assert production_schemas.isdisjoint(fixture_schemas)
    assert pilot_schemas.isdisjoint(fixture_schemas)

    io_boundary = config["parent_validation_io_boundary"]
    assert io_boundary[
        "parent_control_json_hash_and_parse_from_same_regular_file_snapshot"
    ]
    assert io_boundary["canonical_spine_own_json_serialization_abi"] == (
        "JSON_SORTED_KEYS_INDENT_2_UTF8_PLUS_LF"
    )
    assert io_boundary["family_specific_json_serialization_abi_exact_match_required"]
    assert io_boundary["generic_cross_family_json_serializer_acceptance_forbidden"]
    assert io_boundary["parent_validator_source_payload_columns_opened"] == 0
    assert io_boundary["parent_validator_source_payload_bytes_scanned"] == 0
    assert io_boundary["forbidden_old_v2_five_column_parent_payload_rescan"]
    assert io_boundary["own_source_projection_full_success_passes"] == 1
    assert io_boundary["own_source_payload_opened_only_by_scan_stage"] is True
    assert (
        io_boundary["resume_valid_scan_checkpoint_additional_source_payload_passes"]
        == 0
    )

    scan = config["streaming_scan_contract"]
    assert scan["source_batch_rows"] == 262144
    assert scan["projected_arrow_batch_nbytes_upper_bound"] == 268435456
    assert scan["projected_ipc_serialized_bytes_upper_bound"] == 268435456
    assert scan["encoded_record_batch_bytes_upper_bound"] == 4194304
    assert scan["simultaneous_retained_payload_bytes_upper_bound"] == 541065216
    assert scan["memory_bound_scope"] == (
        "RETAINED_PAYLOAD_BUFFERS_NOT_TOTAL_RSS_OR_NATIVE_ALLOCATOR_SCRATCH"
    )
    assert scan["staging_record_bytes"] == 16
    assert scan["successful_full_source_payload_scan_count"] == 1
    assert scan["all_records_collection_materialized_in_memory"] is False
    assert scan["all_numeric_ids_collection_materialized_in_memory"] is False
    assert scan["all_partition_row_payloads_materialized_in_memory"] is False
    assert (
        scan["bounded_scan_and_partition_checkpoint_metadata_collections_allowed"]
        is True
    )
    assert (
        scan["maximum_scan_and_partition_checkpoint_metadata_entries_full"]
        == 493 + 4096
    )
    assert scan[
        "checkpoint_metadata_maximum_is_not_a_bound_on_other_separately_bounded_control_objects"
    ]

    bounded = config["bounded_partition_and_sort_contract"]
    assert bounded["partition_count"] == 4096
    assert bounded["spool"]["exact_payload_bytes_full"] == 7876691296
    assert bounded["spool"]["maximum_open_files"] == 4096
    assert bounded["spool"]["maximum_buffer_bytes"] == 536870912
    assert bounded["spool"]["open_handle_eviction_or_reopen_forbidden"]
    assert bounded["spool"]["cross_invocation_reuse_forbidden"]
    assert bounded["spool"]["rebuilt_from_verified_staging_each_nonterminal_invocation"]
    assert bounded["external_runs"]["sort_chunk_rows"] == 1048576
    assert bounded["external_runs"]["exact_payload_bytes_upper_bound_full"] == (
        7876691296
    )
    assert bounded["external_runs"]["sort_representation"] == (
        "NUMPY_STRUCTURED_FIXED_WIDTH_RECORDS"
    )
    assert bounded["external_runs"]["numpy_structured_dtype"] == {
        "itemsize": 16,
        "fields": [
            {"name": "work_id_numeric", "dtype": "<u8", "offset": 0},
            {"name": "publication_day", "dtype": "<i4", "offset": 8},
            {"name": "publication_year", "dtype": "<u2", "offset": 12},
            {"name": "temporal_disposition", "dtype": "u1", "offset": 14},
            {"name": "reserved_padding", "dtype": "u1", "offset": 15},
        ],
    }
    assert bounded["external_runs"]["sort_algorithm"] == (
        "numpy_argsort_work_id_numeric_stable"
    )
    assert (
        bounded["external_runs"]["python_record_object_or_tuple_per_row_forbidden"]
        is True
    )
    assert bounded["external_runs"]["sort_working_bytes_preflight_estimate"] == (
        134217728
    )
    assert bounded["external_runs"]["intermediate_merged_run_files_forbidden"]
    assert bounded["external_runs"]["cross_invocation_reuse_forbidden"]
    assert (
        bounded["partition_publication"][
            "parquet_python_record_object_or_tuple_per_row_forbidden"
        ]
        is True
    )
    conservation = bounded["decoded_partition_conservation"]
    assert conservation["partition_checkpoint_binds_publication_year_extrema"]
    assert conservation["partition_checkpoint_binds_w_zero_disposition_counts"]
    assert conservation[
        "reusable_partition_checkpoint_revalidates_decoded_conservation"
    ]
    assert conservation["full_global_publication_year_minimum"] == 1000
    assert conservation["full_global_publication_year_maximum"] == 2050
    assert conservation["full_global_w_zero_rows"] == 1
    assert conservation["w_zero_expected_disposition_category_precommitted"] is False
    assert conservation["w_zero_entity_validity"] == "UNRESOLVED"

    runtime = config["runtime_binding"]
    assert runtime["exact_fields"] == [
        "pyarrow_version",
        "parquet_cpp_version",
        "numpy_version",
        "python_implementation",
        "python_version",
        "platform_machine",
    ]
    assert runtime["bind_to_plan_task_fingerprint_and_receipt"] is True
    assert runtime["resume_requires_exact_equality"] is True

    resources = config["resource_envelope"]
    assert resources["staging_payload_bytes_exact"] == 7876691296
    assert resources["external_run_payload_bytes_upper_bound"] == 7876691296
    assert resources["base_required_free_disk_bytes_full"] == 95321921648
    assert resources["required_free_disk_bytes_full_fresh"] == 153143246352
    assert resources["required_free_disk_bytes_full_nonterminal_resume"] == 95321921648
    assert resources["each_partition_output_bytes_must_not_exceed_formula"] == (
        "partition_rows * 32 + 131072"
    )
    assert resources["reused_and_final_partition_bytes_must_recheck_same_formula"]
    assert resources["aggregate_parquet_output_bytes_must_not_exceed_plan_bound"]
    assert resources["aggregate_control_artifact_bytes_must_not_exceed_plan_bound"]
    assert (
        resources["control_aggregate_includes_audit_lock_and_canonical_receipt_bytes"]
        is True
    )
    assert (
        resources["canonical_receipt_bytes_precomputed_and_checked_before_publication"]
        is True
    )
    assert resources["each_atomic_candidate_bytes_must_not_exceed_plan_bound"]
    assert resources["minimum_available_ram_bytes_full"] == 17179869184
    assert resources["spool_buffer_bytes_upper_bound"] == 536870912
    assert resources["spool_demultiplex_working_bytes_preflight_estimate"] == (
        134217728
    )
    assert resources["calculated_ram_preflight_sizing_bytes"] == 9685696512
    assert resources[
        "calculated_ram_preflight_sizing_bytes_is_not_overwritten_by_minimum_gate"
    ]
    assert resources["minimum_nofile_soft_limit"] == 8192
    assert resources["projected_batch_bytes_upper_bound"] == 541065216
    assert (
        resources["projected_batch_bound_is_simultaneous_retained_payload_buffers"]
        is True
    )
    assert resources["sort_working_representation_must_be_compact_numpy"] is True
    assert resources["python_object_per_sort_or_parquet_row_forbidden"] is True
    assert resources["require_source_and_output_distinct_device"] is True
    assert resources["closed_test_fixture_distinct_device_exempt"] is True
    assert resources["nofile_soft_limit_probe_is_runtime_gate"] is True
    assert resources["atomic_rename_fsync_readback_probe_is_runtime_gate"] is True
    assert resources[
        "nonterminal_resume_threshold_requires_validated_existing_nonterminal_plan"
    ]
    assert resources[
        "nonterminal_resume_threshold_requires_state_completed_prefix_pairs_exist"
    ]
    assert resources["resume_cli_flag_alone_never_selects_nonterminal_resume_threshold"]
    assert resources["missing_empty_or_lock_only_output_selects_fresh_threshold"]
    assert (
        resources[
            "production_and_pilot_runtime_batch_and_sort_values_must_equal_contract"
        ]
        is True
    )
    assert (
        resources[
            "closed_fixture_runtime_batch_and_sort_values_must_not_exceed_contract"
        ]
        is True
    )

    preflight = config["preflight_evidence_contract"]
    assert preflight["receipt_bound"] is True
    assert preflight["excluded_from_plan_task_and_result_fingerprints"] is True
    assert preflight["allowed_invocation_kind_values"] == [
        "FRESH",
        "NONTERMINAL_RESUME",
    ]
    assert preflight[
        "invocation_kind_derived_from_validated_output_state_not_resume_cli_flag"
    ]
    assert preflight["selected_required_available_disk_bytes_is_invocation_threshold"]
    assert preflight["nonterminal_resume_refreshes_after_new_live_preflight"] is True
    assert preflight["terminal_resume_must_not_rewrite"] is True
    assert preflight["terminal_resume_validates_receipt_bound_bytes_and_sha256"]

    operator_gates = config["pre_full_operational_gates"]
    assert operator_gates["full_execution_current_readiness"] == (
        "BLOCKED_PENDING_OPERATOR_GATES"
    )
    assert operator_gates["one_file_nonpromotable_production_pilot"]["status"] == (
        "NOT_RUN"
    )
    benchmark = operator_gates["one_file_nonpromotable_production_pilot"][
        "benchmark_capture_contract"
    ]
    assert benchmark["implementation_status"] == "PENDING_IMPLEMENTATION_AND_REVIEW"
    assert benchmark["artifact_receipt_bound"] is False
    assert benchmark["artifact_programmatic_parent_of_full_receipt"] is False
    assert operator_gates["source_drive_health_review"]["status"] == "NOT_RUN"
    assert operator_gates["source_quiet_window_review"]["status"] == "NOT_RUN"
    assert operator_gates["downstream_pipeline_capacity_review"]["status"] == (
        "NOT_RUN"
    )
    assert (
        operator_gates["downstream_pipeline_capacity_review"][
            "one_file_pilot_extrapolation_alone_sufficient_for_full_authorization"
        ]
        is False
    )
    assert operator_gates["independent_read_only_verifier"]["status"] == (
        "PENDING_IMPLEMENTATION_AND_REVIEW"
    )
    assert operator_gates["explicit_new_full_authorization"]["status"] == (
        "NOT_GRANTED"
    )
    assert (
        operator_gates["no_pilot_or_full_execution_performed_by_this_contract_update"]
        is True
    )

    semantics = config["resource_bound_semantics"]
    assert semantics["total_process_rss_upper_bound_claimed"] is False
    assert semantics["native_allocator_scratch_exactly_measured"] is False
    assert semantics["estimates_still_bound_to_plan_and_resume_runtime_fingerprint"]

    resume = config["resume_and_corruption_contract"]
    assert resume["exact_existing_plan_bytes_required"] is True
    assert resume["valid_scan_checkpoint_reuse_opens_source_payload"] is False
    assert resume["mismatched_reusable_artifact_policy"] == (
        "FAIL_CLOSED_NO_AUTOMATIC_OVERWRITE"
    )
    assert resume["temporary_spool_or_runs_reusable_across_invocations"] is False
    assert resume["temporary_spool_or_runs_policy"] == (
        "DELETE_AND_REBUILD_FROM_VERIFIED_STAGING_EACH_NONTERMINAL_INVOCATION"
    )
    assert resume["fresh_only_writes_initial_running_scan_zero_state"]
    assert resume[
        "nonterminal_resume_preserves_existing_monotonic_state_before_reuse_validation"
    ]
    assert resume["completed_receipt_is_terminal"] is True
    assert resume["current_expected_plan_runtime_or_config_mismatch_policy"] == (
        "CURRENT_INVOCATION_INCOMPATIBLE_NO_WRITE"
    )
    assert resume["current_source_stat_or_parent_control_drift_policy"] == (
        "CURRENT_INVOCATION_INCOMPATIBLE_NO_WRITE"
    )
    assert resume[
        "current_invocation_incompatibility_preserves_receipt_report_state_and_artifacts"
    ]
    assert resume["terminal_validation_keyboard_interrupt_or_system_exit_policy"] == (
        "PRESERVE_ALL_BYTES_AND_RERAISE"
    )
    assert resume["terminal_validation_interruption_is_not_intrinsic_corruption"]
    assert resume["terminal_resume_validation_is_entirely_read_only"]
    assert resume["terminal_receipt_hash_tree_or_reconstruction_mismatch_policy"] == (
        "TERMINAL_VALIDATION_FAILED_NO_WRITE"
    )
    assert resume["terminal_resume_automatic_invalidation_or_repair_forbidden"]

    security = config["output_security_contract"]
    assert (
        security[
            "read_only_output_path_policy_validation_before_any_probe_mkdir_lock_or_write"
        ]
        is True
    )
    assert (
        security["terminal_resume_runs_same_read_only_output_path_policy_validation"]
        is True
    )
    assert security[
        "receipt_and_invocation_classification_revalidated_immediately_after_lock"
    ]
    assert security[
        "postlock_invocation_kind_change_invalidates_prelock_selected_threshold"
    ]
    assert security[
        "receipt_appearing_before_lock_routes_terminal_before_nonterminal_mutation"
    ]
    assert security["terminal_lock_must_preexist_regular_operator_owned_mode_0600"]
    assert security["terminal_lock_open_forbids_create_and_chmod"]
    assert security[
        "valid_terminal_resume_preserves_nine_field_stat_for_every_file_including_lock"
    ]
    assert security["receipt_with_missing_lock_fails_no_write_and_does_not_create_lock"]
    assert security["atomic_probe_allowed_only_after_output_path_policy_passes"] is True
    assert security["source_output_and_parent_ancestor_symlink_rejection_required"]
    assert security["source_payload_open_uses_o_nofollow"] is True
    assert security["temporary_record_create_uses_o_nofollow_and_o_excl"] is True
    assert security["control_json_atomic_writes_use_dirfd_and_o_nofollow"] is True
    assert security["all_data_io_descriptor_relative_claimed"] is False
    assert security["atomic_probe_target_hardlink_count_must_equal_one"] is True
    assert security["all_published_artifact_hardlink_count_one_claimed"] is False
    assert security["fsync_parent_directory_after_atomic_rename"] is True
    assert security["exact_allowed_directory_set_frozen_for_all_scopes"]
    assert security["exact_allowed_directories_all_scopes"] == [
        "checkpoints",
        "checkpoints/scan",
        "checkpoints/scan/main",
        "checkpoints/partition",
        "staging",
        "staging/main",
        "spine",
    ]
    assert security["undeclared_empty_directories_forbidden"]
    assert (
        security["output_root_mode_revalidated_at_phase_final_and_terminal_boundaries"]
        is True
    )
    assert security["output_root_owner_must_equal_effective_operator_uid"] is True
    assert security["individual_work_ids_repository_or_release_allowed"] is False

    completion = config["completion_contract"]
    assert completion["exact_receipt_artifact_count_full"] == 9185
    assert completion["receipt_written_last"] is True
    assert completion["final_validation_streaming"] is True
    assert completion["final_validation_may_materialize_all_records_or_ids"] is False
    assert (
        completion["substantive_exact_set_and_reconstruction_validation_before_receipt"]
        is True
    )
    assert (
        completion[
            "post_receipt_operations_are_read_only_and_limited_to_"
            "canonical_receipt_readback_size_and_exact_surface_recheck"
        ]
        is True
    )
    assert completion[
        "canonical_receipt_serialization_bytes_revalidated_on_terminal_resume"
    ]
    assert completion[
        "post_receipt_keyboard_interrupt_or_system_exit_preserves_bytes_and_reraises"
    ]
    assert completion[
        "production_and_pilot_pre_receipt_own_frozen_input_revalidation_required"
    ]
    assert completion[
        "pre_receipt_default_config_hash_and_nine_field_identity_revalidated"
    ]
    assert completion[
        "pre_receipt_current_tool_hash_and_nine_field_identity_revalidated"
    ]
    assert completion["pre_receipt_foundation_config_sha256_revalidated"]
    assert completion["pre_receipt_pinned_predecessor_tool_sha256_revalidated"]
    assert completion["closed_fixture_exempt_from_repository_frozen_input_revalidation"]
    assert (
        completion["pre_receipt_file_snapshot_hash_is_loaded_code_attestation"] is False
    )
    assert completion["running_process_code_object_byte_identity_attested"] is False
    assert completion["pre_receipt_validation_failure_leaves_receipt_absent"] is True
    assert completion[
        "same_invocation_finalization_failure_must_not_leave_promotable_receipt_or_report"
    ]

    consumers = config["consumer_rebinding_boundary"]
    assert (
        consumers["production_spine_pass_automatically_rebinds_existing_consumers"]
        is False
    )
    assert consumers[
        "each_consumer_requires_new_versioned_config_runner_test_and_receipt_binding"
    ]

    claim = config["claim_boundary"]
    required_claim_phrases = (
        "IMPLEMENTED_PRODUCTION_CAPABLE_PRODUCTION_NOT_RUN",
        "records no production result",
        "old five-column temporal payload scan",
        "three-column source projection is scanned exactly once",
        "Pilot and closed-test schemas and statuses are distinct and nonpromotable",
        "Existing author-v1 and citation-v1 consumers require explicit versioned",
        "scientific effect",
        "PRODUCTION_NOT_RUN",
    )
    assert all(phrase in claim for phrase in required_claim_phrases)


def test_yaml_is_unique_and_uses_the_production_overlay_schema() -> None:
    assert _load_config()["schema_version"] == (
        "openalex_g1_canonical_work_time_spine_design.v3"
    )
    with pytest.raises(ConstructorError, match="duplicate key 'status'"):
        _load_yaml_unique("status: DESIGN\nstatus: PASS\n")


def test_all_v0_and_v1_surfaces_are_preserved_byte_exact() -> None:
    assert {path: _sha256(path) for path in PRESERVED_SURFACES} == PRESERVED_SURFACES


def test_runner_and_semantic_config_are_non_circularly_frozen() -> None:
    config = _load_config()
    runner = config["runner_contract"]
    assert runner["required_tool_path"] == RUNNER_PATH.relative_to(REPO_ROOT).as_posix()
    assert runner["required_tool_sha256"] == EXPECTED_RUNNER_SHA256
    assert _sha256(RUNNER_PATH) == EXPECTED_RUNNER_SHA256
    assert runner["semantic_config_sha256_stored_in_config"] is False
    assert "semantic_config_sha256" not in runner
    assert runner["semantic_config_fingerprint_normalization"] == {
        "serialization": "CANONICAL_JSON_SORTED_KEYS_COMPACT_UTF8",
        "normalized_field": "runner_contract.required_tool_sha256",
        "normalized_value": RUNNER_SHA256_SENTINEL,
        "all_other_config_fields_included": True,
    }
    assert _semantic_config_sha256(config) == EXPECTED_SEMANTIC_CONFIG_SHA256


def test_foundation_is_exact_and_only_operational_sections_are_superseded() -> None:
    foundation = _load_config()["foundation_contract"]
    assert foundation["policy"] == (
        "EXACT_V1_SCIENTIFIC_CONTRACT_PLUS_PRODUCTION_EXTERNAL_MEMORY_OVERLAY"
    )
    assert (
        foundation["config_sha256"]
        == PRESERVED_SURFACES[
            REPO_ROOT
            / "experiments/configs/openalex_g1_canonical_work_time_spine_v1.yaml"
        ]
    )
    assert (
        foundation["synthetic_runner_sha256"]
        == PRESERVED_SURFACES[
            REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v2.py"
        ]
    )
    assert foundation["load_and_validate_foundation_before_overlay"] is True
    assert foundation["foundation_mutation_allowed"] is False
    assert set(foundation["superseded_operational_sections"]) == {
        "lifecycle",
        "synthetic_implementation",
        "execution_plan",
        "completion_contract",
        "resource_envelope",
        "pilot_and_full_rules",
    }
    assert {
        "spine_schema",
        "disposition_classifier",
        "identifier_accounting",
        "publication_accounting",
        "staging_record_abi",
        "partition_contract",
        "cross_stage_conservation",
        "release_privacy",
    }.issubset(foundation["inherited_without_semantic_change"])


def test_exact_live_content_and_temporal_parent_anchors_are_frozen() -> None:
    config = _load_config()
    source = config["source"]
    assert source["task_file_index_must_be_contiguous_zero_based"] is True
    assert source["task_relative_paths_must_be_unique"] is True
    assert source["task_relative_paths_must_be_strictly_lexically_increasing"] is True
    parents = config["production_parent_anchors"]
    assert parents["parent_code_repository_root_policy"] == (
        "CURRENT_REPOSITORY_ROOT_ONLY"
    )
    assert parents["temporal_repo_root_override_allowed"] is False
    assert (
        parents["reject_nondefault_temporal_repo_root_before_parent_control_access"]
        is True
    )
    assert parents["imported_temporal_module_globals_mutation_forbidden"] is True
    assert (
        parents["imported_temporal_module_repo_root_must_equal_current_repository_root"]
        is True
    )
    assert (
        parents[
            "temporal_module_production_and_frozen_config_constants_must_equal_"
            "anchored_config_sha256"
        ]
        is True
    )
    content = parents["current_content_baseline"]
    assert content["required_json_serialization_abi"] == (
        "JSON_SORTED_KEYS_INDENT_2_UTF8_PLUS_LF"
    )
    assert content["required_receipt_sha256"] == (
        "082161a29bcd8fd79fd85e0bf6814a5c7950b97b847ea08056e17eebcd369a12"
    )
    assert content["required_status"] == "PASS_CURRENT_CONTENT_BASELINE_COMPONENT"
    assert content["required_run_state_status"] == "COMPLETE"
    assert content["required_result_fingerprint_sha256"] == (
        "12ebd47d88973f37f4ed5b6f8cdd5901eb5c0b7005a41a6ab3b96423a40bbdf1"
    )
    assert content["required_content_tree_sha256"] == (
        "73bb8b0409c782ce5964d480ac6c6cecb04bab8c09df31999e581000b79ccff6"
    )
    assert content["required_checkpoint_tree_sha256"] == (
        "e509867989cec9dfd398221ee5d7abd69dc73fa7330acf9302413184d786bfef"
    )
    assert content["required_file_count"] == 21019
    assert content["required_source_bytes"] == 988404537807

    temporal = parents["main_temporal_validity"]
    assert temporal["required_json_serialization_abi"] == (
        "CANONICAL_JSON_SORTED_KEYS_COMPACT_UTF8_PLUS_LF"
    )
    assert temporal["parent_version_policy"] == (
        "TEMPORAL_V3_ONLY_NO_V1_OR_V2_FALLBACK"
    )
    assert temporal["required_receipt_sha256"] == (
        "2908087fc6c96f2aadfdfb08774ed8c3cd54dd38bd54d4e12b1831e7d863ddfe"
    )
    assert temporal["required_status"] == (
        "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_FIELD_VALIDITY_V3_WITH_EXACT_UTC_"
        "MILLISECONDS_AND_NONBLOCKING_ORDER_PROFILE"
    )
    assert temporal["required_state_status"] == "COMPLETE_PENDING_RECEIPT"
    assert temporal["required_tool_sha256"] == (
        "76e25ffb2bb45977f70ac089db1011b4219d6215e546f16cbe11ce1efdb2cc1d"
    )
    assert temporal["required_result_fingerprint_sha256"] == (
        "a844c624e81567bf89536218b0c488144fe2730f93c53e72193e6cfe4931bfb5"
    )
    assert temporal["required_checkpoint_tree_sha256"] == (
        "725cf22ce96eca153fa1c9542bd09bd628a4ea1687a2fee3e5610a8cded95392"
    )
    assert temporal["required_source_tree_sha256"] == (
        "da3fa570a5fb49a36668dffdee008c2fea4d4b6fd5a3f7f17873df78b603b2a0"
    )
    assert temporal["required_exact_total_receipt_artifact_count"] == 497
    assert temporal["required_raw_rows"] == 492293207
    assert temporal["required_retained_rows"] == 492293206

    binding = parents["cross_binding"]
    assert binding["require_content_and_temporal_task_paths_exactly_equal"]
    assert binding["require_all_493_tasks_bound_to_both_parents"]
    assert binding["content_mode_normalization"] == (
        "content_mode == stat_imode(temporal_mode) == (temporal_mode & 0o7777)"
    )


def test_parent_validation_never_repeats_the_temporal_five_column_scan() -> None:
    boundary = _load_config()["parent_validation_io_boundary"]
    assert boundary[
        "parent_control_json_hash_and_parse_from_same_regular_file_snapshot"
    ]
    assert boundary["canonical_spine_own_json_serialization_abi"] == (
        "JSON_SORTED_KEYS_INDENT_2_UTF8_PLUS_LF"
    )
    assert boundary["family_specific_json_serialization_abi_exact_match_required"]
    assert boundary["generic_cross_family_json_serializer_acceptance_forbidden"]
    assert boundary["parent_control_artifacts_may_be_opened"] is True
    assert boundary["current_source_stat_may_be_read"] is True
    assert boundary["parent_validator_source_payload_columns_opened"] == 0
    assert boundary["parent_validator_source_payload_bytes_scanned"] == 0
    assert boundary["forbidden_old_v2_five_column_parent_payload_rescan"] is True
    assert boundary["forbidden_parent_payload_projection"] == [
        "id",
        "publication_date",
        "publication_year",
        "created_date",
        "updated_date",
    ]
    assert boundary["own_source_projection_columns_exactly"] == [
        "id",
        "publication_date",
        "publication_year",
    ]
    assert boundary["own_source_projection_full_success_passes"] == 1
    assert boundary["own_source_file_open_count_full_success"] == 493
    assert boundary["own_source_payload_opened_only_by_scan_stage"] is True
    assert boundary["finalization_identity_check"] == "stat_only_no_source_payload"
    assert (
        boundary["resume_valid_scan_checkpoint_additional_source_payload_passes"] == 0
    )


def test_streaming_scan_and_external_sort_are_memory_bounded() -> None:
    config = _load_config()
    scan = config["streaming_scan_contract"]
    assert scan["algorithm"] == (
        "ONE_SOURCE_FILE_AT_A_TIME_ONE_PROJECTED_RECORD_BATCH_AT_A_TIME"
    )
    assert scan["source_batch_rows"] == 262144
    assert scan["projected_arrow_batch_nbytes_upper_bound"] == 256 * 1024**2
    assert scan["projected_ipc_serialized_bytes_upper_bound"] == 256 * 1024**2
    assert scan["encoded_record_batch_bytes_upper_bound"] == 262144 * 16
    assert scan["simultaneous_retained_payload_bytes_upper_bound"] == (
        2 * 256 * 1024**2 + 262144 * 16
    )
    assert scan["simultaneous_retained_payload_formula"] == (
        "268435456 + 268435456 + 262144 * 16"
    )
    assert scan["memory_bound_scope"] == (
        "RETAINED_PAYLOAD_BUFFERS_NOT_TOTAL_RSS_OR_NATIVE_ALLOCATOR_SCRATCH"
    )
    assert (
        scan["each_object_and_combined_bound_must_pass_before_next_buffer_growth"]
        is True
    )
    assert scan["successful_full_source_payload_scan_count"] == 1
    assert scan["staging_record_bytes"] == 16
    assert scan["staging_shard_count_full"] == 493
    assert scan["write_uncommitted_task_temp_then_fsync_and_atomic_rename"]
    for forbidden_collection in (
        "all_task_row_payloads_materialized_in_memory",
        "all_records_collection_materialized_in_memory",
        "all_numeric_ids_collection_materialized_in_memory",
        "all_partition_row_payloads_materialized_in_memory",
    ):
        assert scan[forbidden_collection] is False
    assert (
        scan["bounded_scan_and_partition_checkpoint_metadata_collections_allowed"]
        is True
    )
    assert scan["maximum_scan_and_partition_checkpoint_metadata_entries_full"] == 4589
    assert (
        scan["maximum_scan_and_partition_checkpoint_metadata_entries_formula"]
        == "493 + 4096"
    )
    assert scan[
        "checkpoint_metadata_maximum_is_not_a_bound_on_other_separately_bounded_control_objects"
    ]

    bounded = config["bounded_partition_and_sort_contract"]
    assert bounded["partition_count"] == 4096
    assert bounded["partition_formula"] == "work_id_numeric_mod_4096"
    spool = bounded["spool"]
    assert spool["record_bytes"] == 16
    assert spool["exact_payload_bytes_full"] == 492293206 * 16
    assert spool["exact_file_count"] == 4096
    assert spool["staging_read_passes"] == 1
    assert spool["demultiplex_representation"] == (
        "NUMPY_STABLE_BATCH_PARTITION_GROUPING"
    )
    assert spool["demultiplex_chunk_rows"] == 1048576
    assert spool["maximum_open_files"] == 4096
    assert spool["buffer_bytes_per_open_file"] == 8192 * 16
    assert spool["maximum_buffered_records"] == 4096 * 8192
    assert spool["maximum_buffer_bytes"] == 4096 * 8192 * 16
    assert spool["demultiplex_working_bytes_preflight_estimate"] == 128 * 1024**2
    assert spool["open_handle_eviction_or_reopen_forbidden"]
    assert spool["maximum_close_and_fsync_operations_full"] == 4096
    assert spool["cross_invocation_reuse_forbidden"]
    assert spool["rebuilt_from_verified_staging_each_nonterminal_invocation"]

    runs = bounded["external_runs"]
    assert runs["sort_chunk_rows"] == 1048576
    assert runs["sort_chunk_encoded_bytes"] == 1048576 * 16
    assert runs["sort_representation"] == "NUMPY_STRUCTURED_FIXED_WIDTH_RECORDS"
    assert runs["numpy_structured_dtype"]["itemsize"] == 16
    assert runs["sort_algorithm"] == "numpy_argsort_work_id_numeric_stable"
    assert runs["python_record_object_or_tuple_per_row_forbidden"] is True
    assert (
        runs[
            "sort_working_bound_includes_input_keys_permutation_output_and_native_scratch"
        ]
        is True
    )
    assert runs["sort_working_bytes_preflight_estimate"] == 128 * 1024**2
    assert runs["theoretical_maximum_runs_in_one_partition"] == 470
    assert runs["maximum_open_run_files"] == 512
    assert runs["one_pass_k_way_merge_directly_to_parquet"] is True
    assert runs["intermediate_merged_run_files_forbidden"] is True
    assert runs["cross_invocation_reuse_forbidden"]
    assert runs["exact_payload_bytes_upper_bound_full"] == 492293206 * 16
    assert runs["duplicate_detection_rule"] == (
        "EQUAL_NUMERIC_IDS_SHARE_MODULO_PARTITION_AND_FAIL_STRICT_ORDER"
    )

    conservation = bounded["decoded_partition_conservation"]
    assert conservation["partition_checkpoint_binds_publication_year_extrema"]
    assert conservation["partition_checkpoint_binds_w_zero_disposition_counts"]
    assert conservation[
        "reusable_partition_checkpoint_revalidates_decoded_conservation"
    ]
    assert conservation["full_global_publication_year_minimum"] == 1000
    assert conservation["full_global_publication_year_maximum"] == 2050
    assert conservation["full_global_w_zero_rows"] == 1
    assert conservation["w_zero_expected_disposition_category_precommitted"] is False
    assert conservation["full_report_publication_accounting_includes_year_extrema"]
    assert conservation[
        "full_report_identifier_accounting_includes_w_zero_disposition_cross_tab"
    ]
    assert conservation["w_zero_entity_validity"] == "UNRESOLVED"


def test_resource_formulas_include_staging_spool_runs_parquet_and_headroom() -> None:
    resources = _load_config()["resource_envelope"]
    rows = 492293206
    staging = rows * 16
    spool = rows * 16
    runs = rows * 16
    parquet = rows * 32 + 4096 * 131072
    largest_next = rows * 32 + 131072
    control = 2 * 1024**3
    reserve = 8 * 1024**3
    worst_case = staging + spool + runs + parquet + control + largest_next
    required = (1500 * worst_case + 999) // 1000 + reserve

    assert resources["gpu_used"] is False
    assert resources["staging_payload_bytes_exact"] == staging == 7876691296
    assert resources["partition_spool_payload_bytes_upper_bound"] == spool
    assert resources["external_run_payload_bytes_upper_bound"] == runs
    assert resources["parquet_payload_bytes_upper_bound"] == parquet == 16290253504
    assert resources["each_partition_output_bytes_must_not_exceed_formula"] == (
        "partition_rows * 32 + 131072"
    )
    assert resources["reused_and_final_partition_bytes_must_recheck_same_formula"]
    assert resources["aggregate_parquet_output_bytes_must_not_exceed_plan_bound"]
    assert resources["largest_next_atomic_artifact_bytes_upper_bound"] == (largest_next)
    assert resources["aggregate_control_artifact_bytes_must_not_exceed_plan_bound"]
    assert (
        resources["control_aggregate_includes_audit_lock_and_canonical_receipt_bytes"]
        is True
    )
    assert (
        resources["canonical_receipt_bytes_precomputed_and_checked_before_publication"]
        is True
    )
    assert resources["each_atomic_candidate_bytes_must_not_exceed_plan_bound"]
    assert resources["worst_case_remaining_bytes_before_multiplier"] == (worst_case)
    assert worst_case == 57821324704
    assert resources["disk_safety_multiplier_milli"] == 1500
    assert resources["disk_reserve_bytes"] == reserve
    assert resources["base_required_free_disk_bytes_full"] == required == 95321921648
    assert resources["base_required_free_disk_gib_rounded_3dp"] == 88.775
    assert resources["full_fresh_restartability_additional_resident_bytes"] == (
        worst_case
    )
    assert (
        resources["required_free_disk_bytes_full_fresh"]
        == (required + worst_case)
        == 153143246352
    )
    assert resources["required_free_disk_gib_full_fresh_rounded_3dp"] == 142.626
    assert resources["required_free_disk_bytes_full_nonterminal_resume"] == required
    assert (
        resources["required_free_disk_gib_full_nonterminal_resume_rounded_3dp"]
        == 88.775
    )
    assert resources["preferred_free_disk_bytes_full"] == 160 * 1024**3
    assert resources["preferred_free_disk_gib_full"] == 160
    assert resources["plan_resource_requirements_bind_both_full_disk_thresholds"]
    assert resources["invocation_selects_threshold_without_mutating_plan"]
    assert resources[
        "nonterminal_resume_threshold_requires_validated_existing_nonterminal_plan"
    ]
    assert resources[
        "nonterminal_resume_threshold_requires_state_completed_prefix_pairs_exist"
    ]
    assert resources["resume_cli_flag_alone_never_selects_nonterminal_resume_threshold"]
    assert resources["missing_empty_or_lock_only_output_selects_fresh_threshold"]
    assert resources["completed_artifact_subtraction_from_preflight_forbidden"]

    calculated_ram = (
        max(
            resources["projected_batch_bytes_upper_bound"],
            resources["sort_working_bytes_preflight_estimate"],
            resources["spool_demultiplex_working_bytes_preflight_estimate"],
        )
        + resources["spool_buffer_bytes_upper_bound"]
        + resources["parquet_writer_buffer_bytes_preflight_estimate"]
        + resources["merge_heap_bytes_preflight_estimate"]
        + resources["ram_reserve_bytes"]
    )
    assert calculated_ram == resources["calculated_ram_preflight_sizing_bytes"]
    assert calculated_ram == 9685696512
    assert resources["ram_preflight_sizing_formula"] == (
        "max(541065216, 134217728, 134217728) + 536870912 + 16777216 + 1048576 + 8589934592"
    )
    assert resources[
        "calculated_ram_preflight_sizing_bytes_is_not_overwritten_by_minimum_gate"
    ]
    assert resources["minimum_available_ram_bytes_full"] == 16 * 1024**3
    assert resources["preferred_available_ram_bytes_full"] == 32 * 1024**3
    assert resources["projected_arrow_batch_bytes_upper_bound"] == 256 * 1024**2
    assert resources["projected_ipc_serialized_bytes_upper_bound"] == 256 * 1024**2
    assert resources["encoded_record_batch_bytes_upper_bound"] == 262144 * 16
    assert resources["projected_batch_bytes_upper_bound"] == 541065216
    assert (
        resources["projected_batch_bound_is_simultaneous_retained_payload_buffers"]
        is True
    )
    assert resources["sort_working_representation_must_be_compact_numpy"] is True
    assert resources["python_object_per_sort_or_parquet_row_forbidden"] is True
    assert resources["minimum_nofile_soft_limit"] == 8192
    assert resources[
        "require_live_disk_ram_nofile_preflight_before_source_payload_open"
    ]
    assert resources["require_source_and_output_distinct_device"] is True
    assert resources["distinct_device_gate_applies_to_production_full_and_pilot"]
    assert resources["closed_test_fixture_distinct_device_exempt"] is True
    assert resources["nofile_soft_limit_probe_is_runtime_gate"] is True
    assert resources["atomic_rename_fsync_readback_probe_is_runtime_gate"] is True
    assert (
        resources[
            "production_and_pilot_runtime_batch_and_sort_values_must_equal_contract"
        ]
        is True
    )
    assert (
        resources[
            "closed_fixture_runtime_batch_and_sort_values_must_not_exceed_contract"
        ]
        is True
    )
    assert resources["source_scan_exclusivity_acknowledgement_required"] is True

    semantics = _load_config()["resource_bound_semantics"]
    assert {
        "spool_buffer_bytes_and_open_handle_count",
        "parquet_bytes_per_partition_and_aggregate",
        "control_artifact_aggregate_bytes",
        "largest_atomic_candidate_bytes",
    }.issubset(semantics["executable_observed_or_reconstructed_bounds"])
    assert set(
        semantics["declared_engineering_preflight_estimates_not_directly_measured"]
    ) == {
        "native_sort_working_bytes",
        "spool_demultiplex_working_bytes",
        "parquet_writer_buffer_bytes",
        "merge_heap_bytes",
        "interpreter_and_allocator_ram_reserve",
        "calculated_ram_preflight_sizing_bytes",
    }
    assert semantics["total_process_rss_upper_bound_claimed"] is False
    assert semantics["native_allocator_scratch_exactly_measured"] is False


def test_redundant_numeric_contract_surfaces_remain_synchronized() -> None:
    config = _load_config()
    modes = config["execution_modes"]
    scan = config["streaming_scan_contract"]
    bounded = config["bounded_partition_and_sort_contract"]
    resources = config["resource_envelope"]
    preflight = config["preflight_evidence_contract"]

    assert (
        modes["full_production"]["source_batch_rows_exact"]
        == modes["production_pilot"]["source_batch_rows_exact"]
        == scan["source_batch_rows"]
        == resources["source_batch_rows"]
    )
    assert (
        modes["full_production"]["partition_sort_chunk_rows_exact"]
        == modes["production_pilot"]["partition_sort_chunk_rows_exact"]
        == bounded["external_runs"]["sort_chunk_rows"]
        == resources["sort_chunk_rows"]
    )
    assert (
        scan["projected_arrow_batch_nbytes_upper_bound"]
        == resources["projected_arrow_batch_bytes_upper_bound"]
    )
    assert (
        scan["projected_ipc_serialized_bytes_upper_bound"]
        == resources["projected_ipc_serialized_bytes_upper_bound"]
    )
    assert (
        scan["encoded_record_batch_bytes_upper_bound"]
        == resources["encoded_record_batch_bytes_upper_bound"]
    )
    assert (
        scan["simultaneous_retained_payload_bytes_upper_bound"]
        == resources["projected_batch_bytes_upper_bound"]
    )
    assert (
        bounded["external_runs"]["sort_working_bytes_preflight_estimate"]
        == resources["sort_working_bytes_preflight_estimate"]
    )
    assert (
        bounded["spool"]["maximum_buffer_bytes"]
        == resources["spool_buffer_bytes_upper_bound"]
    )
    assert (
        bounded["spool"]["demultiplex_working_bytes_preflight_estimate"]
        == resources["spool_demultiplex_working_bytes_preflight_estimate"]
    )
    assert (
        bounded["spool"]["exact_payload_bytes_full"]
        == resources["partition_spool_payload_bytes_upper_bound"]
    )
    assert (
        bounded["external_runs"]["exact_payload_bytes_upper_bound_full"]
        == resources["external_run_payload_bytes_upper_bound"]
    )
    assert preflight["full_selected_required_available_disk_bytes"] == {
        "FRESH": resources["required_free_disk_bytes_full_fresh"],
        "NONTERMINAL_RESUME": resources[
            "required_free_disk_bytes_full_nonterminal_resume"
        ],
    }


def test_preflight_evidence_is_receipt_bound_but_fingerprint_excluded() -> None:
    contract = _load_config()["preflight_evidence_contract"]
    assert contract["basename"] == "preflight_evidence.json"
    assert contract["receipt_bound"] is True
    assert contract["excluded_from_plan_task_and_result_fingerprints"] is True
    assert contract["exact_keyset"] == [
        "schema_version",
        "scope",
        "task_fingerprint_sha256",
        "invocation_kind",
        "observed_free_disk_bytes",
        "selected_required_available_disk_bytes",
        "observed_available_ram_bytes",
        "required_available_ram_bytes",
        "observed_nofile_soft_limit",
        "required_nofile_soft_limit",
        "atomic_rename_fsync_readback_probe",
        "source_st_dev",
        "output_st_dev",
        "source_output_device_gate",
        "latest_nonterminal_invocation_only",
        "operator_paths_redacted",
    ]
    assert contract["allowed_invocation_kind_values"] == [
        "FRESH",
        "NONTERMINAL_RESUME",
    ]
    assert contract[
        "invocation_kind_derived_from_validated_output_state_not_resume_cli_flag"
    ]
    assert contract["selected_required_available_disk_bytes_is_invocation_threshold"]
    assert contract["full_selected_required_available_disk_bytes"] == {
        "FRESH": 153143246352,
        "NONTERMINAL_RESUME": 95321921648,
    }
    assert contract["atomic_rename_fsync_readback_probe_required_value"] == "PASS"
    assert contract["production_source_device_separation_required_value"] == (
        "PASS_DISTINCT_DEVICES"
    )
    assert contract["closed_fixture_device_separation_required_value"] == (
        "EXEMPT_CLOSED_TEST_FIXTURE"
    )
    assert contract["observed_values_must_satisfy_required_values"] is True
    assert contract["absolute_paths_forbidden"] is True
    assert contract["operator_paths_redacted_required_value"] is True
    assert contract["initial_execution_written_before_first_source_payload_open"]
    assert contract["nonterminal_resume_refreshes_after_new_live_preflight"] is True
    assert contract["terminal_resume_must_not_rewrite"] is True
    assert contract["terminal_resume_validates_receipt_bound_bytes_and_sha256"]


def test_pre_full_operational_gates_remain_unsatisfied_and_nonpromotable() -> None:
    gates = _load_config()["pre_full_operational_gates"]
    assert gates["full_execution_current_readiness"] == (
        "BLOCKED_PENDING_OPERATOR_GATES"
    )
    assert (
        gates["code_is_production_capable_but_real_hdd_throughput_is_unbenchmarked"]
        is True
    )
    pilot = gates["one_file_nonpromotable_production_pilot"]
    assert pilot["status"] == "NOT_RUN"
    assert pilot["exact_source_files"] == 1
    assert pilot["required_recorded_metrics"] == [
        "selected_source_rows",
        "selected_source_bytes",
        "pilot_wall_seconds_external",
        "rows_per_second",
        "source_bytes_per_second",
        "projected_full_scan_wall_seconds",
    ]
    assert pilot["pilot_report_metric_sources"] == {
        "selected_source_rows": "scope.raw_rows",
        "selected_source_bytes": "scope.source_bytes",
    }
    assert pilot["rows_per_second_formula"] == (
        "selected_source_rows / pilot_wall_seconds_external"
    )
    assert pilot["source_bytes_per_second_formula"] == (
        "selected_source_bytes / pilot_wall_seconds_external"
    )
    assert pilot["projected_full_scan_wall_seconds_formula"] == (
        "492293207 / rows_per_second"
    )
    assert (
        pilot["extrapolation_excludes_partition_sort_final_validation_and_contention"]
        is True
    )
    benchmark = pilot["benchmark_capture_contract"]
    assert benchmark["implementation_status"] == "PENDING_IMPLEMENTATION_AND_REVIEW"
    assert benchmark["external_capture_required"]
    assert "/usr/bin/time" in benchmark["command_template"]
    assert benchmark["required_benchmark_artifact_basename"] == (
        "openalex_g1_canonical_work_time_spine_pilot_benchmark.json"
    )
    assert benchmark["required_benchmark_artifact_schema"] == (
        "openalex_g1_canonical_work_time_spine_pilot_benchmark.v1"
    )
    assert benchmark["required_benchmark_artifact_status"] == (
        "PILOT_BENCHMARK_CAPTURED_OPERATOR_LOCAL_NONPROMOTABLE"
    )
    assert benchmark["artifact_location_policy"] == (
        "OPERATOR_LOCAL_OUTSIDE_REPOSITORY_RELEASE_AND_PILOT_OUTPUT_ROOT"
    )
    assert benchmark["artifact_receipt_bound"] is False
    assert benchmark["artifact_programmatic_parent_of_full_receipt"] is False
    assert benchmark["required_artifact_fields"] == [
        "schema_version",
        "status",
        "pilot_report_sha256",
        "selected_source_rows",
        "selected_source_bytes",
        "pilot_wall_seconds_external",
        "rows_per_second",
        "source_bytes_per_second",
        "projected_full_scan_wall_seconds",
        "raw_time_capture_sha256",
        "operator_paths_redacted",
    ]
    assert benchmark[
        "implementation_and_independent_review_required_before_pilot_gate_can_pass"
    ]
    assert pilot["operator_review_required_before_issuing_full_authorization"]
    assert pilot["programmatic_parent_of_full_receipt"] is False
    health = gates["source_drive_health_review"]
    assert health["status"] == "NOT_RUN"
    assert health["fresh_smart_or_equivalent_read_only_health_check_required"]
    assert health["reason"] == "PRIOR_DEV_SDA_READ_RESET_EVENT_REQUIRES_FRESH_REVIEW"
    quiet = gates["source_quiet_window_review"]
    assert quiet["status"] == "NOT_RUN"
    assert quiet["process_and_io_activity_check_required"] is True
    assert quiet["concurrent_heavy_source_scan_allowed"] is False
    downstream = gates["downstream_pipeline_capacity_review"]
    assert downstream["status"] == "NOT_RUN"
    assert downstream["one_file_pilot_covers_scan_and_staging_only"]
    assert downstream["one_file_pilot_does_not_measure"] == [
        "4096_handle_spool_and_numpy_demultiplex",
        "external_sort_and_parquet_publication",
        "9185_artifact_final_rehash",
    ]
    assert downstream["acceptable_review_routes"] == [
        "BOUNDED_NONPROMOTABLE_STAGING_DERIVED_DOWNSTREAM_BENCHMARK",
        "CONSERVATIVE_ANALYTIC_TIME_AND_IO_ESTIMATE_WITH_OPERATOR_ACCEPTANCE",
    ]
    assert (
        downstream[
            "one_file_pilot_extrapolation_alone_sufficient_for_full_authorization"
        ]
        is False
    )
    assert downstream["required_before_issuing_full_authorization"]
    assert downstream["programmatic_parent_of_full_receipt"] is False
    verifier = gates["independent_read_only_verifier"]
    assert verifier["status"] == "PENDING_IMPLEMENTATION_AND_REVIEW"
    assert verifier["self_validation_inside_production_runner_is_not_independent"]
    authorization = gates["explicit_new_full_authorization"]
    assert authorization["status"] == "NOT_GRANTED"
    assert authorization["current_implementation_work_does_not_grant_full_execution"]
    assert authorization[
        "required_after_pilot_health_quiet_window_and_downstream_capacity_review"
    ]
    assert gates["no_pilot_or_full_execution_performed_by_this_contract_update"]


def test_resume_corruption_and_output_security_fail_closed() -> None:
    config = _load_config()
    resume = config["resume_and_corruption_contract"]
    assert resume["exact_existing_plan_bytes_required"] is True
    assert "foundation_synthetic_runner_sha256" in resume["plan_binds"]
    assert resume["runtime_fingerprint_must_match_for_resume"] is True
    assert resume["valid_scan_checkpoint_reuse_opens_source_payload"] is False
    assert resume["checkpoint_without_artifact_reusable"] is False
    assert resume["artifact_without_checkpoint_reusable"] is False
    assert resume["state_declared_completed_prefix_missing_pair_policy"] == (
        "FAIL_CLOSED_BEFORE_RESUME_THRESHOLD_SELECTION"
    )
    assert resume["complete_pairs_ahead_of_state_may_enter_normal_reuse_validation"]
    assert resume["fresh_only_writes_initial_running_scan_zero_state"]
    assert resume[
        "nonterminal_resume_preserves_existing_monotonic_state_before_reuse_validation"
    ]
    assert resume["mismatched_reusable_artifact_policy"] == (
        "FAIL_CLOSED_NO_AUTOMATIC_OVERWRITE"
    )
    assert resume[
        "corrupt_nonterminal_reusable_output_requires_fresh_empty_output_root"
    ]
    assert resume["temporary_spool_or_runs_reusable_across_invocations"] is False
    assert resume["temporary_spool_or_runs_policy"] == (
        "DELETE_AND_REBUILD_FROM_VERIFIED_STAGING_EACH_NONTERMINAL_INVOCATION"
    )
    assert resume["completed_receipt_is_terminal"] is True
    assert resume[
        "terminal_resume_validates_every_receipt_artifact_before_any_source_payload_open"
    ]
    assert resume[
        "terminal_self_consistent_disk_surface_validated_before_current_invocation_compatibility"
    ]
    assert resume["current_expected_plan_runtime_or_config_mismatch_policy"] == (
        "CURRENT_INVOCATION_INCOMPATIBLE_NO_WRITE"
    )
    assert resume["current_source_stat_or_parent_control_drift_policy"] == (
        "CURRENT_INVOCATION_INCOMPATIBLE_NO_WRITE"
    )
    assert resume[
        "current_invocation_incompatibility_preserves_receipt_report_state_and_artifacts"
    ]
    assert resume["terminal_validation_keyboard_interrupt_or_system_exit_policy"] == (
        "PRESERVE_ALL_BYTES_AND_RERAISE"
    )
    assert resume["terminal_validation_interruption_is_not_intrinsic_corruption"]
    assert resume["terminal_resume_validation_is_entirely_read_only"]
    assert resume["terminal_receipt_hash_tree_or_reconstruction_mismatch_policy"] == (
        "TERMINAL_VALIDATION_FAILED_NO_WRITE"
    )
    assert resume["terminal_resume_automatic_invalidation_or_repair_forbidden"]
    assert resume[
        "corrupt_terminal_surface_requires_explicit_future_operator_quarantine_or_repair"
    ]
    assert resume["same_invocation_finalization_failure_invalidation_order"] == [
        "remove_outputs_receipt_first",
        "write_finalization_failed_state_without_result_fingerprint",
        "remove_json_and_markdown_reports",
    ]

    security = config["output_security_contract"]
    assert security["output_must_be_outside_repository_source_and_parent_output_roots"]
    assert (
        security[
            "read_only_output_path_policy_validation_before_any_probe_mkdir_lock_or_write"
        ]
        is True
    )
    assert (
        security["terminal_resume_runs_same_read_only_output_path_policy_validation"]
        is True
    )
    assert security["atomic_probe_allowed_only_after_output_path_policy_passes"] is True
    assert security["source_output_and_parent_ancestor_symlink_rejection_required"]
    assert security["source_payload_open_uses_o_nofollow"] is True
    assert security["temporary_record_create_uses_o_nofollow_and_o_excl"] is True
    assert security["control_json_atomic_writes_use_dirfd_and_o_nofollow"] is True
    assert security["phase_boundary_source_identity_revalidation_required"] is True
    assert security["published_artifacts_revalidated_as_regular_non_symlinks"]
    assert security["all_data_io_descriptor_relative_claimed"] is False
    assert security["atomic_probe_target_hardlink_count_must_equal_one"] is True
    assert security["all_published_artifact_hardlink_count_one_claimed"] is False
    assert security["exclusive_output_lock_required"] is True
    assert security["stale_lock_never_ignored_implicitly"] is True
    assert security["terminal_lock_must_preexist_regular_operator_owned_mode_0600"]
    assert security["terminal_lock_open_forbids_create_and_chmod"]
    assert security[
        "valid_terminal_resume_preserves_nine_field_stat_for_every_file_including_lock"
    ]
    assert security["receipt_with_missing_lock_fails_no_write_and_does_not_create_lock"]
    assert security[
        "prelock_receipt_or_invocation_classification_not_authoritative_after_lock"
    ]
    assert security[
        "receipt_and_invocation_classification_revalidated_immediately_after_lock"
    ]
    assert security[
        "postlock_invocation_kind_change_invalidates_prelock_selected_threshold"
    ]
    assert security[
        "receipt_appearing_before_lock_routes_terminal_before_nonterminal_mutation"
    ]
    assert security["output_directory_mode"] == "0700"
    assert security["output_file_mode"] == "0600"
    assert (
        security["output_root_mode_revalidated_at_phase_final_and_terminal_boundaries"]
        is True
    )
    assert security["output_root_owner_must_equal_effective_operator_uid"] is True
    assert security["exact_allowed_directory_set_frozen_for_all_scopes"]
    assert len(security["exact_allowed_directories_all_scopes"]) == 7
    assert security["undeclared_empty_directories_forbidden"]
    assert security["fsync_file_before_atomic_rename"] is True
    assert security["fsync_parent_directory_after_atomic_rename"] is True
    assert security["reject_undeclared_entries"] is True
    assert security[
        "operator_absolute_paths_redacted_from_plan_report_state_and_receipt"
    ]
    assert security["individual_work_ids_repository_or_release_allowed"] is False


def test_exact_full_receipt_surface_is_9185_and_contains_no_temporary_data() -> None:
    completion = _load_config()["completion_contract"]
    assert len(completion["exact_fixed_receipt_artifacts"]) == 7
    assert "preflight_evidence.json" in completion["exact_fixed_receipt_artifacts"]
    expected = 7 + 2 * 493 + 2 * 4096
    assert expected == 9185
    assert completion["exact_receipt_artifact_count_full"] == expected
    assert completion["exact_receipt_artifact_count_formula"] == ("7 + 2*493 + 2*4096")
    assert completion["receipt_excludes"] == [
        "audit.lock",
        "work_time_spine_audit.outputs.json",
    ]
    assert completion["temporary_entries_forbidden_at_completion"] == [
        ".work_time_spine_tmp",
    ]
    assert completion["receipt_written_last"] is True
    assert completion["completion_literal"] == (
        "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
    )
    assert completion["final_validation_streaming"] is True
    assert completion["final_validation_may_materialize_all_records_or_ids"] is False
    assert completion["final_validation_rehashes_all_9185_artifacts"] is True
    assert (
        completion["substantive_exact_set_and_reconstruction_validation_before_receipt"]
        is True
    )
    assert (
        completion[
            "post_receipt_operations_are_read_only_and_limited_to_"
            "canonical_receipt_readback_size_and_exact_surface_recheck"
        ]
        is True
    )
    assert completion[
        "canonical_receipt_serialization_bytes_revalidated_on_terminal_resume"
    ]
    assert completion[
        "post_receipt_keyboard_interrupt_or_system_exit_preserves_bytes_and_reraises"
    ]
    assert completion[
        "production_and_pilot_pre_receipt_own_frozen_input_revalidation_required"
    ]
    assert completion[
        "pre_receipt_default_config_hash_and_nine_field_identity_revalidated"
    ]
    assert completion[
        "pre_receipt_current_tool_hash_and_nine_field_identity_revalidated"
    ]
    assert completion["pre_receipt_foundation_config_sha256_revalidated"]
    assert completion["pre_receipt_pinned_predecessor_tool_sha256_revalidated"]
    assert completion["closed_fixture_exempt_from_repository_frozen_input_revalidation"]
    assert (
        completion["pre_receipt_file_snapshot_hash_is_loaded_code_attestation"] is False
    )
    assert completion["running_process_code_object_byte_identity_attested"] is False
    assert completion["pre_receipt_validation_failure_leaves_receipt_absent"] is True
    assert completion[
        "same_invocation_finalization_failure_must_not_leave_promotable_receipt_or_report"
    ]


def test_existing_consumers_are_not_implicitly_promoted() -> None:
    config = _load_config()
    boundary = config["consumer_rebinding_boundary"]
    assert boundary[
        "existing_author_v1_and_citation_v1_are_bound_to_synthetic_spine_v1"
    ]
    assert boundary[
        "production_spine_pass_automatically_rebinds_existing_consumers"
    ] is (False)
    assert boundary[
        "each_consumer_requires_new_versioned_config_runner_test_and_receipt_binding"
    ]
    gates = config["future_success_contract"]["required_full_gate_states"]
    assert gates["canonical_work_time_spine_component"] == "PASS"
    assert gates["historical_content_identity"] == "UNVERIFIED"
    assert gates["raw_source_id_validity"] == "BLOCKED"
    assert gates["w_zero_entity_validity"] == "UNRESOLVED"
    assert gates["citation_edge_chronology"] == "PENDING_NOT_IN_THIS_COMPONENT"
    assert gates["author_entity_validity"] == "PENDING"
    assert gates["g_minus_1_data_readiness"] == "PENDING"
    assert gates["g0_measurement_freeze"] == "PENDING"
    assert gates["empirical_g1_signature"] == "NOT_RUN"
    assert gates["pass_g1"] is False
    assert gates["scientific_effect"] == "NOT_ESTABLISHED"


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        ("lifecycle.production_status", "PASS"),
        ("lifecycle.production_result_exists", True),
        ("lifecycle.full_execution_authorized_by_config_alone", True),
        ("runner_contract.required_tool_sha256", "f" * 64),
        ("execution_modes.full_production.all_three_flags_required_together", False),
        (
            "execution_modes.full_production.batch_or_sort_bound_override_allowed",
            True,
        ),
        ("execution_modes.production_pilot.production_component_pass", True),
        ("execution_modes.production_pilot.promotion_allowed", True),
        ("execution_modes.closed_test_fixture.cli_flag_accepted", True),
        ("execution_modes.closed_test_fixture.production_component_pass", True),
        ("execution_modes.schema_and_status_sets_pairwise_disjoint", False),
        (
            "production_output_abi.production_schemas_forbidden_in_pilot_and_test_modes",
            False,
        ),
        (
            "parent_validation_io_boundary.parent_validator_source_payload_columns_opened",
            5,
        ),
        (
            "parent_validation_io_boundary."
            "parent_control_json_hash_and_parse_from_same_regular_file_snapshot",
            False,
        ),
        (
            "production_parent_anchors.main_temporal_validity."
            "required_json_serialization_abi",
            "JSON_SORTED_KEYS_INDENT_2_UTF8_PLUS_LF",
        ),
        (
            "production_parent_anchors.current_content_baseline."
            "required_json_serialization_abi",
            "CANONICAL_JSON_SORTED_KEYS_COMPACT_UTF8_PLUS_LF",
        ),
        (
            "production_parent_anchors."
            "imported_temporal_module_globals_mutation_forbidden",
            False,
        ),
        (
            "production_parent_anchors."
            "imported_temporal_module_repo_root_must_equal_current_repository_root",
            False,
        ),
        (
            "production_parent_anchors."
            "temporal_module_production_and_frozen_config_constants_must_equal_"
            "anchored_config_sha256",
            False,
        ),
        (
            "parent_validation_io_boundary.canonical_spine_own_json_serialization_abi",
            "CANONICAL_JSON_SORTED_KEYS_COMPACT_UTF8_PLUS_LF",
        ),
        ("parent_validation_io_boundary.own_source_projection_full_success_passes", 2),
        ("streaming_scan_contract.all_records_collection_materialized_in_memory", True),
        (
            "bounded_partition_and_sort_contract.decoded_partition_conservation."
            "full_global_publication_year_minimum",
            1001,
        ),
        (
            "bounded_partition_and_sort_contract.external_runs.sort_chunk_rows",
            492293206,
        ),
        (
            "bounded_partition_and_sort_contract.spool.maximum_open_files",
            64,
        ),
        (
            "bounded_partition_and_sort_contract.spool."
            "open_handle_eviction_or_reopen_forbidden",
            False,
        ),
        (
            "bounded_partition_and_sort_contract.external_runs."
            "python_record_object_or_tuple_per_row_forbidden",
            False,
        ),
        ("resource_envelope.external_run_payload_bytes_upper_bound", 0),
        ("resource_envelope.minimum_nofile_soft_limit", 1024),
        (
            "resource_envelope."
            "calculated_ram_preflight_sizing_bytes_is_not_overwritten_by_minimum_gate",
            False,
        ),
        ("resource_envelope.required_free_disk_bytes_full_fresh", 64 * 1024**3),
        (
            "resource_envelope.required_free_disk_bytes_full_nonterminal_resume",
            64 * 1024**3,
        ),
        (
            "resource_envelope."
            "resume_cli_flag_alone_never_selects_nonterminal_resume_threshold",
            False,
        ),
        (
            "resource_envelope."
            "nonterminal_resume_threshold_requires_state_completed_prefix_pairs_exist",
            False,
        ),
        ("preflight_evidence_contract.terminal_resume_must_not_rewrite", False),
        (
            "pre_full_operational_gates.explicit_new_full_authorization.status",
            "GRANTED",
        ),
        (
            "pre_full_operational_gates.one_file_nonpromotable_production_pilot."
            "benchmark_capture_contract.implementation_status",
            "PASS",
        ),
        (
            "pre_full_operational_gates.downstream_pipeline_capacity_review.status",
            "PASS",
        ),
        (
            "resume_and_corruption_contract.valid_scan_checkpoint_reuse_opens_source_payload",
            True,
        ),
        (
            "resume_and_corruption_contract.mismatched_reusable_artifact_policy",
            "OVERWRITE",
        ),
        (
            "resume_and_corruption_contract."
            "temporary_spool_or_runs_reusable_across_invocations",
            True,
        ),
        (
            "resume_and_corruption_contract."
            "nonterminal_resume_preserves_existing_monotonic_state_before_reuse_validation",
            False,
        ),
        (
            "resume_and_corruption_contract."
            "current_expected_plan_runtime_or_config_mismatch_policy",
            "INVALIDATE",
        ),
        (
            "resume_and_corruption_contract."
            "current_source_stat_or_parent_control_drift_policy",
            "INVALIDATE",
        ),
        (
            "resume_and_corruption_contract."
            "terminal_validation_interruption_is_not_intrinsic_corruption",
            False,
        ),
        (
            "resume_and_corruption_contract."
            "terminal_resume_validation_is_entirely_read_only",
            False,
        ),
        (
            "resume_and_corruption_contract."
            "terminal_resume_automatic_invalidation_or_repair_forbidden",
            False,
        ),
        (
            "output_security_contract.individual_work_ids_repository_or_release_allowed",
            True,
        ),
        (
            "output_security_contract."
            "read_only_output_path_policy_validation_before_any_probe_mkdir_lock_or_write",
            False,
        ),
        (
            "output_security_contract."
            "output_root_mode_revalidated_at_phase_final_and_terminal_boundaries",
            False,
        ),
        (
            "output_security_contract."
            "receipt_and_invocation_classification_revalidated_immediately_after_lock",
            False,
        ),
        (
            "output_security_contract.terminal_lock_open_forbids_create_and_chmod",
            False,
        ),
        ("output_security_contract.undeclared_empty_directories_forbidden", False),
        ("completion_contract.exact_receipt_artifact_count_full", 9184),
        (
            "completion_contract.final_validation_may_materialize_all_records_or_ids",
            True,
        ),
        (
            "completion_contract."
            "canonical_receipt_serialization_bytes_revalidated_on_terminal_resume",
            False,
        ),
        (
            "completion_contract."
            "production_and_pilot_pre_receipt_own_frozen_input_revalidation_required",
            False,
        ),
        (
            "completion_contract."
            "closed_fixture_exempt_from_repository_frozen_input_revalidation",
            False,
        ),
        (
            "completion_contract."
            "pre_receipt_file_snapshot_hash_is_loaded_code_attestation",
            True,
        ),
        (
            "completion_contract.running_process_code_object_byte_identity_attested",
            True,
        ),
        (
            "completion_contract."
            "post_receipt_keyboard_interrupt_or_system_exit_preserves_bytes_and_reraises",
            False,
        ),
        (
            "consumer_rebinding_boundary.production_spine_pass_automatically_rebinds_existing_consumers",
            True,
        ),
    ],
)
def test_nonpromotion_validator_rejects_unsafe_mutations(
    path: str,
    unsafe_value: Any,
) -> None:
    config = copy.deepcopy(_load_config())
    _set_nested(config, path, unsafe_value)
    with pytest.raises(AssertionError):
        _validate_nonpromotion_boundary(config)


def test_nonpromotion_boundary_is_frozen() -> None:
    _validate_nonpromotion_boundary(_load_config())


def test_design_document_and_config_are_synchronized() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    config = _load_config()
    required_text = (
        "IMPLEMENTED_PRODUCTION_CAPABLE_PRODUCTION_NOT_RUN",
        "PRODUCTION_NOT_RUN",
        "--execute --authorize-full --acknowledge-exclusive-source-scan",
        "control artifacts and live stat",
        "five-column payload rescan",
        "three-column source scan",
        "7,876,691,296",
        "16,290,253,504",
        "95,321,921,648",
        "153,143,246,352",
        "142.626 GiB",
        "160 GiB",
        "9,185",
        "one-pass k-way merge",
        EXPECTED_RUNNER_SHA256,
        EXPECTED_SEMANTIC_CONFIG_SHA256,
        "FAIL_CLOSED_NO_AUTOMATIC_OVERWRITE",
        "PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE",
        "PASS_TEST_FIXTURE_CANONICAL_WORK_TIME_SPINE_EXTERNAL_MEMORY_CONTRACT_ONLY",
        "author-v1 and citation-v1",
        config["production_output_abi"]["full_component_status"],
    )
    assert all(item in text for item in required_text)
