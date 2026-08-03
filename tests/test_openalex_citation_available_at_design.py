from __future__ import annotations

import copy
from collections import Counter
import hashlib
from pathlib import Path
import struct
from typing import Any

import pytest
import yaml
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_citation_available_at_v0.yaml"
)
DOC_PATH = (
    REPO_ROOT / "docs/case-studies/researcher-pivot/"
    "openalex-citation-available-at-audit-design.md"
)
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_citation_available_at.py"
SPINE_CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml"
)
TEMPORAL_CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_main_temporal_validity_v0.yaml"
)
MEMBERSHIP_RESULT_PATH = (
    REPO_ROOT
    / "experiments/evidence/results/openalex_g1_work_source_membership_20260717.summary.json"
)
RESULT_GLOBS = (
    "openalex_g1_citation_available_at*",
    "citation_edge_chronology_audit*",
    "deduplicated_edge_event_partitions*",
    "chronology_safe_edge_partitions_by_cited_work*",
    "ephemeral_noncanonical_direct_mode_time_index*",
    "duplicate_multiplicity_aggregates.parquet*",
    "chronology_lag_aggregates.parquet*",
    "open_world_target_coverage.parquet*",
    "cutoff_and_maturation_coverage.parquet*",
)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        assert key not in mapping, f"duplicate YAML key: {key!r}"
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.load(
        path.read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    assert isinstance(value, dict)
    return value


def _load_config() -> dict[str, Any]:
    return _load_yaml(CONFIG_PATH)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_nested(value: dict[str, Any], path: str, replacement: Any) -> None:
    cursor = value
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor[part]
        assert isinstance(child, dict)
        cursor = child
    cursor[parts[-1]] = replacement


def _classify_pair(
    source_id: int,
    target_id: int,
    work_times: dict[int, tuple[int | None, int]],
) -> str:
    source_day, source_disposition = work_times[source_id]
    if source_id == target_id:
        return "self_loop"
    if source_disposition not in {2, 3}:
        return "source_temporal_unusable"
    if target_id not in work_times:
        return "target_external_open_world"
    target_day, target_disposition = work_times[target_id]
    if target_disposition not in {2, 3}:
        return "target_temporal_unusable"
    assert source_day is not None and target_day is not None
    lag_days = source_day - target_day
    if lag_days < 0:
        return "negative_lag_target_after_source"
    if lag_days == 0:
        return "same_day"
    return "positive_lag"


def _synthetic_accounting(
    mode: str,
    pairs: list[tuple[int, int]],
    *,
    direct_time_loader: Any,
    spine_times: dict[int, tuple[int | None, int]],
) -> dict[str, Any]:
    if mode == "exact_temporal_reconstruction_from_current_source":
        work_times = direct_time_loader()
    else:
        assert mode == "canonical_work_time_spine_reuse"
        work_times = dict(spine_times)
    multiplicities = Counter(pairs)
    buckets = Counter(
        _classify_pair(source, target, work_times) for source, target in multiplicities
    )
    raw_self_loops = sum(
        count for (source, target), count in multiplicities.items() if source == target
    )
    unique_self_loops = sum(source == target for source, target in multiplicities)
    raw_external = sum(
        count
        for (_, target), count in multiplicities.items()
        if target not in work_times
    )
    unique_external = sum(target not in work_times for _, target in multiplicities)
    raw_rows = len(pairs)
    unique_rows = len(multiplicities)
    return {
        "raw_pair_rows": raw_rows,
        "unique_pair_rows": unique_rows,
        "duplicate_pair_row_excess": raw_rows - unique_rows,
        "raw_self_loop_rows": raw_self_loops,
        "unique_self_loop_pairs": unique_self_loops,
        "duplicate_self_loop_row_excess": raw_self_loops - unique_self_loops,
        "raw_external_target_rows": raw_external,
        "raw_in_snapshot_target_rows": raw_rows - raw_external,
        "unique_external_target_pairs": unique_external,
        "unique_in_snapshot_target_pairs": unique_rows - unique_external,
        "buckets": dict(buckets),
        "chronology_safe_primary_pairs": buckets["same_day"] + buckets["positive_lag"],
    }


def _pair_bucket(source_id: int, target_id: int, bucket_count: int) -> int:
    assert bucket_count > 1
    payload = struct.pack(">QQ", source_id, target_id)
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % bucket_count


def _validate_mode_parent_fixture(
    config: dict[str, Any], mode: str, fixture: dict[str, dict[str, Any]]
) -> None:
    required = config["mode_conditioned_parent_sets"][mode]["required_in_order"]
    assert list(fixture) == required
    content = fixture["current_content_baseline"]
    assert content["state"] == "COMPLETE"
    assert content["completion_contract"] == (
        "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
    )
    assert content["artifact_tree_valid"] is True
    temporal = fixture["main_temporal_validity"]
    assert temporal["state"] == "COMPLETE_PENDING_RECEIPT"
    assert temporal["completion_contract"] == (
        "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
    )
    assert temporal["artifact_tree_valid"] is True
    assert temporal["embedded_content_matches"] is True
    assert fixture["work_source_membership_regression"]["hash_valid"] is True
    if mode == "canonical_work_time_spine_reuse":
        spine = fixture["canonical_work_time_spine"]
        assert spine["state"] == "COMPLETE_PENDING_RECEIPT"
        assert spine["completion_contract"] == (
            "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
        )
        assert spine["artifact_tree_valid"] is True
        assert spine["partition_files"] == 4096
        assert spine["rows"] == 492293206
        assert spine["embedded_parents_match"] is True


def _validate_claim_boundary(config: dict[str, Any]) -> None:
    assert config["lifecycle"] == {
        "design_status": "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN",
        "implementation_status": "IMPLEMENTED_SYNTHETICALLY",
        "production_status": "PRODUCTION_NOT_RUN",
        "executable": True,
        "execution_script": "scripts/audit_openalex_citation_available_at.py",
        "synthetic_fixture_only": True,
        "production_execution_supported": False,
        "production_source_access_supported": False,
        "production_result_exists": False,
        "pass_result_allowed_in_design_artifact": False,
        "implementation_requires_new_review": False,
        "production_authorization_requires_separate_implementation_review": True,
    }
    assert config["required_parent_execution_order"] == [
        "current_content_baseline",
        "main_temporal_validity",
        "precommit_temporal_input_mode",
        "validate_mode_conditioned_temporal_input",
    ]

    parents = config["parents"]
    content = parents["current_content_baseline"]
    temporal = parents["main_temporal_validity"]
    membership = parents["work_source_membership_regression"]
    spine_parent = parents["canonical_work_time_spine"]
    assert content["required_status"] == "PASS_CURRENT_CONTENT_BASELINE_COMPONENT"
    assert content["required_run_state"] == "COMPLETE"
    assert content["required_completion_contract"] == (
        "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
    )
    assert content["required_full_scope"] is True
    assert temporal["required_status"] == (
        "PASS_CURRENT_SNAPSHOT_MAIN_TEMPORAL_VALIDITY_COMPONENT_WITH_FROZEN_EXCLUSIONS"
    )
    assert temporal["required_full_scope"] is True
    assert temporal["current_state"] == ("IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN")
    assert temporal["required_run_state"] == "COMPLETE_PENDING_RECEIPT"
    assert temporal["required_completion_contract"] == (
        "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
    )
    assert temporal["require_completion_receipt_written_last"] is True
    assert temporal["require_final_disk_validation"] is True
    assert temporal["require_same_content_baseline_parent"] is True
    assert membership["historical_byte_identity_to_current_baseline_established"] is (
        False
    )
    assert membership["require_reproduce_anchors_on_current_bytes"] is True
    assert spine_parent["design_status"] == (
        "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN"
    )
    assert spine_parent["production_status"] == "PRODUCTION_NOT_RUN"
    assert spine_parent["synthetic_executable"] is True
    assert spine_parent["production_executable"] is False
    assert spine_parent["synthetic_execution_script"] == (
        "scripts/audit_openalex_canonical_work_time_spine.py"
    )
    assert spine_parent["synthetic_contract_result_is_production_parent"] is False
    assert spine_parent["production_result_exists"] is False
    assert spine_parent["design_config_path"] == (
        "experiments/configs/openalex_g1_canonical_work_time_spine_v0.yaml"
    )
    assert spine_parent["design_config_sha256"] == (
        "192d78e3df094121e63922a6eaf20f57a57af41bc174c0fdbbcc347e3eaed385"
    )
    assert spine_parent["required_future_rows"] == 492293206
    assert spine_parent["required_future_partition_files"] == 4096
    assert spine_parent["required_future_run_state"] == "COMPLETE_PENDING_RECEIPT"
    assert spine_parent["required_future_completion_contract"] == (
        "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
    )

    mode_parents = config["mode_conditioned_parent_sets"]
    assert mode_parents["exact_temporal_reconstruction_from_current_source"][
        "required_in_order"
    ] == [
        "current_content_baseline",
        "main_temporal_validity",
        "work_source_membership_regression",
    ]
    assert mode_parents["canonical_work_time_spine_reuse"]["required_in_order"] == [
        "current_content_baseline",
        "main_temporal_validity",
        "work_source_membership_regression",
        "canonical_work_time_spine",
    ]
    assert mode_parents["parent_set_is_part_of_task_fingerprint"] is True

    availability = config["available_at_policy"]
    assert availability["resolution"] == "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION"
    assert availability["edge_available_at_field"] == "citing_work_publication_date"
    assert availability["edge_first_observed_timestamp_present"] is False
    assert availability["openalex_observation_time_established"] is False
    assert availability["historical_openalex_graph_established"] is False
    assert availability["created_date_or_updated_date_used_as_edge_timestamp"] is False

    reuse_mode = config["temporal_input_modes"]["canonical_work_time_spine_reuse"]
    assert reuse_mode["citation_component_may_build_or_repair_canonical_spine"] is (
        False
    )
    assert reuse_mode["citation_component_validates_existing_spine_only"] is True

    open_world = config["open_world_policy"]
    assert open_world["external_target_is_orphan"] is False
    assert open_world["reconstructed_inbound_counts_are_total_citations"] is False

    maturation = config["maturation_contract"]
    assert maturation["total_citation_claim_allowed"] is False
    assert maturation["historically_observed_citation_count_claim_allowed"] is False
    assert maturation["historical_edge_observation_available_by_decision_year_t"] == (
        "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT"
    )
    assert maturation[
        "temporal_maturity_alone_establishes_historical_availability"
    ] is (False)

    leakage = config["leakage_policy"]
    assert leakage["current_cited_by_count_used"] is False
    assert leakage["retrospective_backfill_may_be_called_historically_observed"] is (
        False
    )

    privacy = config["release_privacy"]
    assert privacy["detailed_work_ids_repository_allowed"] is False
    assert privacy["detailed_work_ids_release_allowed"] is False

    gates = config["future_success_contract"]["required_full_gate_states"]
    assert gates["citation_edge_accounting"] == "PASS"
    assert gates["bibliographic_available_at_convention"] == "PASS_WITH_LIMITATION"
    assert gates["openalex_first_observed_time"] == (
        "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT"
    )
    assert gates["historical_graph_reconstruction"] == "RETROSPECTIVE_PROXY_ONLY"
    assert gates["historical_content_identity"] == "UNVERIFIED"
    assert gates["raw_source_id_validity"] == "BLOCKED"
    assert gates["w_zero_entity_validity"] == "UNRESOLVED"
    assert gates["external_target_chronology"] == "UNRESOLVED_OPEN_WORLD"
    assert gates["fixed_horizon_outcome_maturation"] == ("PENDING_SEPARATE_G0_POLICY")
    assert gates["cutoff_aware_t_minus_1_researcher_graph"] == "PENDING"
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
        "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION",
        "not an observed OpenAlex indexing timestamp",
        "cannot establish when OpenAlex first observed an edge",
        "restricted to closed synthetic fixtures",
        "No production run or result exists",
    )
    assert all(phrase in claim for phrase in required_phrases)


def test_design_lifecycle_parents_and_claim_boundary_are_frozen() -> None:
    _validate_claim_boundary(_load_config())


def test_parent_config_and_regression_hash_cross_links_are_live() -> None:
    parents = _load_config()["parents"]
    assert (
        _sha256(SPINE_CONFIG_PATH)
        == parents["canonical_work_time_spine"]["design_config_sha256"]
    )
    assert (
        _sha256(TEMPORAL_CONFIG_PATH)
        == parents["main_temporal_validity"]["required_config_sha256"]
    )
    assert (
        _sha256(MEMBERSHIP_RESULT_PATH)
        == parents["work_source_membership_regression"]["sha256"]
    )


def test_exact_scope_cross_binding_and_regression_anchors() -> None:
    config = _load_config()
    tables = config["tables"]
    scopes = config["mode_scope_contracts"]
    assert tables["main"] == {
        "name": "openalex_works_20260330",
        "expected_files": 493,
        "expected_raw_rows": 492293207,
        "expected_source_bytes": 150062449219,
        "projected_columns": [
            "id",
            "publication_date",
            "publication_year",
            "created_date",
            "updated_date",
        ],
    }
    assert tables["referenced_works"] == {
        "name": "openalex_works_20260330__referenced_works",
        "expected_files": 3044,
        "expected_raw_rows": 3042536160,
        "expected_source_bytes": 37119669321,
        "projected_columns": ["id", "referenced_works"],
    }
    direct_scope = scopes["exact_temporal_reconstruction_from_current_source"]
    reuse_scope = scopes["canonical_work_time_spine_reuse"]
    assert direct_scope["execution_abi"] == "CITATION_DIRECT_RECONSTRUCTION_V1"
    assert direct_scope["scanned_source_files"] == sum(
        table["expected_files"] for table in tables.values()
    )
    assert direct_scope["scanned_source_rows"] == sum(
        table["expected_raw_rows"] for table in tables.values()
    )
    assert direct_scope["scanned_source_bytes"] == sum(
        table["expected_source_bytes"] for table in tables.values()
    )
    assert direct_scope["scanned_main_source_files"] == 493
    assert direct_scope["canonical_spine_partition_files_consumed"] == 0
    assert reuse_scope["execution_abi"] == "CITATION_CANONICAL_SPINE_REUSE_V1"
    assert reuse_scope["scanned_source_files"] == 3044
    assert reuse_scope["scanned_main_source_files"] == 0
    assert reuse_scope["logical_bound_source_tasks"] == 3537
    assert reuse_scope["canonical_spine_partition_files_consumed"] == 4096
    assert reuse_scope["canonical_spine_rows_consumed"] == 492293206
    assert reuse_scope["minimum_physical_data_files_consumed"] == 3044 + 4096
    assert reuse_scope["canonical_spine_partition_bytes_consumed"] == (
        "BIND_EXACT_VALUE_FROM_VALIDATED_SPINE_RECEIPT"
    )
    assert scopes["abi_specific_checkpoints_interchangeable"] is False
    assert scopes["gpu_used"] is False

    binding = config["cross_binding"]
    assert (
        binding["direct_reconstruction_source_task_count"]
        == direct_scope["scanned_source_files"]
    )
    assert (
        binding["direct_reconstruction_source_bytes"]
        == direct_scope["scanned_source_bytes"]
    )
    assert binding["spine_reuse_source_task_count"] == 3044
    assert binding["spine_reuse_source_bytes"] == 37119669321
    assert binding["required_source_stat_keyset"] == [
        "device",
        "inode",
        "mode",
        "bytes",
        "mtime_ns",
        "ctime_ns",
    ]
    assert binding["bind_every_scanned_source_task_to_content_plan"] is True
    assert binding["bind_main_task_set_directly_or_transitively_through_spine"]
    assert binding["verify_source_identity_before_and_after_each_file"] is True
    assert binding["require_exact_parent_receipt_artifact_sets_and_hashes"] is True
    assert binding["current_baseline_only_not_historical_identity"] is True

    modes = config["temporal_input_modes"]
    assert modes["required_precommit_before_execution"] is True
    assert modes["mode_is_part_of_task_fingerprint"] is True
    assert modes["mode_change_on_resume_allowed"] is False
    assert modes["allowed_modes"] == [
        "canonical_work_time_spine_reuse",
        "exact_temporal_reconstruction_from_current_source",
    ]
    reuse = modes["canonical_work_time_spine_reuse"]
    direct = modes["exact_temporal_reconstruction_from_current_source"]
    assert reuse["scanned_source_files"] == 3044
    assert reuse["scanned_source_bytes"] == 37119669321
    assert reuse["required_spine_rows"] == 492293206
    assert reuse["required_spine_partition_files"] == 4096
    assert reuse["citation_component_may_build_or_repair_canonical_spine"] is False
    assert reuse["citation_component_validates_existing_spine_only"] is True
    assert reuse["bind_493_main_tasks_transitively_through_spine_receipt"]
    assert direct["scanned_source_files"] == 3537
    assert direct["scanned_source_bytes"] == 187182118540
    assert direct["main_source_files_scanned_in_this_component"] == 493
    assert direct["temporal_lookup_kind"] == (
        "EPHEMERAL_NONCANONICAL_DIRECT_MODE_INDEX"
    )
    assert direct["temporal_lookup_may_be_published_or_reused_as_canonical_spine"] is (
        False
    )
    assert modes["checkpoints_interchangeable_between_modes"] is False

    anchors = config["accounting_anchors"]
    rows = anchors["referenced_rows"]
    targets = anchors["distinct_target_ids"]
    assert rows == {
        "raw_rows": 3042536160,
        "in_snapshot_target_rows": 2904520313,
        "external_target_rows": 138015847,
        "self_loop_rows": 1072993,
    }
    assert targets == {
        "all_valid_targets": 135996024,
        "in_snapshot_targets": 112406279,
        "external_targets": 23589745,
    }
    assert rows["raw_rows"] == (
        rows["in_snapshot_target_rows"] + rows["external_target_rows"]
    )
    assert targets["all_valid_targets"] == (
        targets["in_snapshot_targets"] + targets["external_targets"]
    )
    assert set(anchors["invalid_target_rows"].values()) == {0}
    assert anchors["duplicate_source_target_edges"] == (
        "NOT_MEASURED_REQUIRES_THIS_AUDIT"
    )
    assert anchors["anchors_are_required_current_byte_regressions_not_new_result"]


def test_canonical_spine_consumer_abi_matches_live_spine_design() -> None:
    citation = _load_config()["canonical_spine_consumer_abi"]
    spine = _load_yaml(SPINE_CONFIG_PATH)
    assert (
        citation["field_order"]
        == spine["spine_schema"]["field_order"]
        == [
            "work_id_numeric",
            "publication_day",
            "publication_year",
            "temporal_disposition",
        ]
    )
    for field in citation["field_order"]:
        expected = citation["fields"][field]
        actual = spine["spine_schema"]["fields"][field]
        assert expected["physical_type"] == actual["physical_type"]
        assert expected["nullable"] is actual["nullable"]
        if "null_only_for_disposition" in expected:
            assert (
                expected["null_only_for_disposition"]
                == actual["null_only_for_disposition"]
            )
    assert citation["disposition_enum"] == spine["spine_schema"]["disposition_enum"]
    assert citation["usable_disposition_codes"] == [2, 3]
    partition = citation["partition_contract"]
    live_partition = spine["partition_contract"]
    for key in (
        "algorithm",
        "partition_count",
        "formula",
        "filename_pattern",
        "exact_file_count",
        "every_partition_file_required",
        "within_partition_sort_key",
        "require_strictly_increasing_work_id_within_partition",
        "require_global_exactly_once_membership",
    ):
        assert partition[key] == live_partition[key]


def test_direction_dedup_and_seven_exclusive_buckets_are_exact() -> None:
    config = _load_config()
    direction = config["edge_direction"]
    assert direction == {
        "source_column": "id",
        "source_role": "citing_work",
        "target_column": "referenced_works",
        "target_role": "cited_work",
        "directed_edge_notation": "citing_work_to_cited_work",
        "symmetrization_allowed": False,
        "reversal_allowed": False,
    }

    dedup = config["deduplication"]
    assert dedup["raw_pair_key"] == ["citing_work_id", "cited_work_id"]
    assert dedup["graph_and_outcome_unit"] == "exact_distinct_directed_pair"
    assert dedup["preserve_raw_multiplicity"] is True
    assert dedup["duplicate_pair_counts_once_in_graph_and_outcomes"] is True
    assert len(dedup["required_metrics"]) == 10
    assert {
        "raw_self_loop_rows = sum_pair_multiplicity_over_unique_self_loop_pairs",
        "raw_self_loop_rows = unique_self_loop_pairs + duplicate_self_loop_row_excess",
        "unique_in_snapshot_target_pairs + unique_external_target_pairs = unique_pair_rows",
    } <= set(dedup["exact_invariants"])
    assert dedup["raw_to_unique_bridge_required"] is True

    buckets = config["exclusive_pair_bucket_policy"]
    assert buckets["precedence"] == [
        "self_loop",
        "source_temporal_unusable",
        "target_external_open_world",
        "target_temporal_unusable",
        "negative_lag_target_after_source",
        "same_day",
        "positive_lag",
    ]
    assert buckets["bucket_count"] == len(buckets["precedence"]) == 7
    assert buckets["classifier_resolution"] == (
        "TOTAL_FIRST_TRUE_PRECEDENCE_CLASSIFIER_V1"
    )
    assert buckets["require_exactly_one_bucket_per_unique_pair"] is True
    assert buckets["classifier_input_domain"]["temporal_usable_disposition_codes"] == [
        2,
        3,
    ]
    assert buckets["signed_lag_days_definition"] == (
        "citing_publication_day - cited_publication_day"
    )
    assert buckets["require_full_attribute_cross_tabs_in_addition_to_precedence_bucket"]
    assert buckets["chronology_safe_primary_buckets"] == ["same_day", "positive_lag"]
    assert buckets["primary_graph_excludes_self_loops"] is True
    assert buckets["external_target_is_not_source_orphan"] is True
    assert (
        "unique_pair_rows = sum_all_seven_exclusive_pair_buckets"
        in buckets["exact_invariants"]
    )
    assert (
        "chronology_safe_primary_pairs = same_day_pairs + positive_lag_pairs"
        in (buckets["exact_invariants"])
    )
    assert len(buckets["required_cross_tab_axes"]) == 6

    sensitivity = config["chronology_sensitivity"]
    assert sensitivity["primary_negative_lag_tolerance_days"] == 0
    assert sensitivity["supplemental_tolerance_days"] == [30, 365]
    assert sensitivity["supplemental_eligibility_predicate_template"] == (
        "signed_lag_days >= -tolerance_days"
    )
    assert sensitivity[
        "annual_only_eligibility_predicate_after_common_prerequisites"
    ] == ("cited_publication_year <= citing_publication_year")
    assert sensitivity["annual_only_chronology_sensitivity"] is True
    assert sensitivity["sensitivity_may_replace_primary"] is False
    assert sensitivity["negative_lag_is_automatically_proven_impossible_citation"] is (
        False
    )


def test_two_mode_synthetic_accounting_is_equivalent_and_reuse_opens_no_main() -> None:
    work_times = {
        1: (100, 2),
        3: (None, 0),
        5: (None, 0),
        6: (110, 2),
        7: (100, 3),
        8: (80, 2),
    }
    pairs = [
        (1, 1),
        (1, 1),
        (3, 99),
        (1, 99),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 8),
        (1, 8),
    ]
    main_opens = 0

    def load_direct_times() -> dict[int, tuple[int | None, int]]:
        nonlocal main_opens
        main_opens += 1
        return dict(work_times)

    direct = _synthetic_accounting(
        "exact_temporal_reconstruction_from_current_source",
        pairs,
        direct_time_loader=load_direct_times,
        spine_times={},
    )
    assert main_opens == 1

    def forbidden_main_open() -> dict[int, tuple[int | None, int]]:
        raise AssertionError("spine-reuse mode opened the main source")

    reuse = _synthetic_accounting(
        "canonical_work_time_spine_reuse",
        pairs,
        direct_time_loader=forbidden_main_open,
        spine_times=work_times,
    )
    assert reuse == direct
    assert reuse["buckets"] == {
        "self_loop": 1,
        "source_temporal_unusable": 1,
        "target_external_open_world": 1,
        "target_temporal_unusable": 1,
        "negative_lag_target_after_source": 1,
        "same_day": 1,
        "positive_lag": 1,
    }
    assert sum(reuse["buckets"].values()) == reuse["unique_pair_rows"]
    assert reuse["raw_self_loop_rows"] == (
        reuse["unique_self_loop_pairs"] + reuse["duplicate_self_loop_row_excess"]
    )
    assert reuse["raw_pair_rows"] == (
        reuse["raw_in_snapshot_target_rows"] + reuse["raw_external_target_rows"]
    )
    assert reuse["unique_pair_rows"] == (
        reuse["unique_in_snapshot_target_pairs"] + reuse["unique_external_target_pairs"]
    )


def test_hash_bucket_abi_is_pair_stable_and_order_sensitive() -> None:
    first = _pair_bucket(1, 8, 17)
    assert first == _pair_bucket(1, 8, 17)
    assert first == (
        int.from_bytes(hashlib.sha256(struct.pack(">QQ", 1, 8)).digest()[:8], "big")
        % 17
    )
    assert struct.pack(">QQ", 1, 8) != struct.pack(">QQ", 8, 1)


@pytest.mark.parametrize(
    ("mode", "mutation"),
    [
        ("exact_temporal_reconstruction_from_current_source", "extra_spine_parent"),
        ("canonical_work_time_spine_reuse", "temporal_state"),
        ("canonical_work_time_spine_reuse", "spine_tree"),
        ("canonical_work_time_spine_reuse", "spine_parent_cross_binding"),
    ],
)
def test_mode_parent_contract_rejects_parent_substitution_or_corruption(
    mode: str, mutation: str
) -> None:
    fixture = {
        "current_content_baseline": {
            "state": "COMPLETE",
            "completion_contract": (
                "REQUIRE_COMPLETE_STATE_AND_VALID_OUTPUTS_MANIFEST"
            ),
            "artifact_tree_valid": True,
        },
        "main_temporal_validity": {
            "state": "COMPLETE_PENDING_RECEIPT",
            "completion_contract": ("RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"),
            "artifact_tree_valid": True,
            "embedded_content_matches": True,
        },
        "work_source_membership_regression": {"hash_valid": True},
    }
    if mode == "canonical_work_time_spine_reuse":
        fixture["canonical_work_time_spine"] = {
            "state": "COMPLETE_PENDING_RECEIPT",
            "completion_contract": ("RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"),
            "artifact_tree_valid": True,
            "partition_files": 4096,
            "rows": 492293206,
            "embedded_parents_match": True,
        }
    _validate_mode_parent_fixture(_load_config(), mode, fixture)
    if mutation == "extra_spine_parent":
        fixture["canonical_work_time_spine"] = {"state": "COMPLETE_PENDING_RECEIPT"}
    elif mutation == "temporal_state":
        fixture["main_temporal_validity"]["state"] = "COMPLETE"
    elif mutation == "spine_tree":
        fixture["canonical_work_time_spine"]["artifact_tree_valid"] = False
    elif mutation == "spine_parent_cross_binding":
        fixture["canonical_work_time_spine"]["embedded_parents_match"] = False
    with pytest.raises(AssertionError):
        _validate_mode_parent_fixture(_load_config(), mode, fixture)


def test_open_world_maturation_cutoff_and_leakage_boundaries_are_frozen() -> None:
    config = _load_config()
    open_world = config["open_world_policy"]
    assert open_world["external_target_is_orphan"] is False
    assert open_world["external_target_publication_date_available"] is False
    assert open_world["external_target_allowed_in_primary_chronology_safe_graph"] is (
        False
    )
    assert open_world["external_target_allowed_in_fixed_horizon_outcome"] is False
    assert open_world[
        "missing_external_citing_works_measurable_from_this_snapshot"
    ] is (False)
    assert open_world["reconstructed_inbound_counts_are_total_citations"] is False

    maturation = config["maturation_contract"]
    assert maturation["status"] == "PENDING_SEPARATE_HASHED_G0_POLICY"
    assert maturation["primary_horizon_years"] is None
    assert maturation["horizon_may_be_selected_from_this_audit_result"] is False
    assert set(maturation["required_future_freeze_fields"]) == {
        "primary_horizon_years",
        "allowed_work_types",
        "same_day_inclusion",
        "leap_day_policy",
        "negative_lag_sensitivity",
        "field_normalization",
        "missing_target_policy",
    }
    assert maturation["february_29_rule"] == (
        "clip_to_february_28_in_non_leap_target_year"
    )
    assert maturation["target_is_temporally_mature_for_decision_year_t_if"] == (
        "maturity_date_lte_december_31_of_t_minus_1"
    )
    assert maturation["historical_edge_observation_available_by_decision_year_t"] == (
        "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT"
    )
    assert maturation[
        "temporal_maturity_alone_establishes_historical_availability"
    ] is (False)
    assert maturation[
        "g0_same_day_choice_may_change_primary_audit_bucket_definition"
    ] is (False)
    assert maturation["total_citation_claim_allowed"] is False
    assert maturation["historically_observed_citation_count_claim_allowed"] is False

    graph = config["t_minus_1_parent_contract"]
    assert graph["cutoff_rule"] == (
        "citing_publication_date_lte_december_31_of_t_minus_1"
    )
    assert graph["graph_is_historical_openalex_snapshot"] is False
    assert graph["edge_observation_by_cutoff_is_established"] is False
    assert graph["edge_observation_by_cutoff_status"] == (
        "UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT"
    )
    assert graph["researcher_peer_exposure_requires_author_history"] is True
    assert graph["topic_or_field_stratification_requires_work_topic_relation"] is (True)
    assert graph["candidate_panel_materialization_allowed_by_this_component"] is False

    leakage = config["leakage_policy"]
    assert "cited_by_count" in leakage["prohibited_projection_or_model_inputs"]
    assert "fwci" in leakage["prohibited_projection_or_model_inputs"]
    assert leakage["current_cited_by_count_used"] is False
    assert leakage[
        "current_snapshot_edges_with_pre_cutoff_publication_dates_are_retrospective_backfill"
    ]
    assert leakage["retrospective_backfill_may_be_called_historically_observed"] is (
        False
    )
    counts = leakage["counts_by_year"]
    assert counts == {
        "core_parent_allowed": False,
        "optional_role": "DIAGNOSTIC_LABEL_ONLY_SEPARATE_CONTRACT",
        "represented_year_start": 2012,
        "represented_year_end": 2026,
        "model_feature_allowed": False,
        "missing_edge_imputation_allowed": False,
        "horizon_selection_allowed": False,
        "primary_inclusion_rule_change_allowed": False,
    }


def test_dependency_parallelism_spine_resource_and_privacy_are_bounded() -> None:
    config = _load_config()
    dependencies = config["dependency_parallelism"]
    assert dependencies["generic_citation_chronology_requires_work_topic_relation"] is (
        False
    )
    assert dependencies["generic_citation_chronology_requires_author_history"] is False
    assert dependencies[
        "topic_or_field_stratified_outcome_requires_work_topic_relation"
    ]
    assert dependencies["researcher_level_peer_exposure_requires_author_history"]
    assert dependencies["researcher_level_peer_exposure_requires_work_topic_relation"]
    assert dependencies[
        "relation_and_citation_logically_parallel_after_required_parents"
    ]
    assert dependencies["heavy_source_scans_should_run_concurrently"] is False

    spine = config["work_time_spine_recommendation"]
    assert spine["status"] == "RECOMMENDED_SEPARATE_COMPONENT_AFTER_TEMPORAL_PASS"
    assert spine["modify_existing_temporal_v0"] is False
    assert spine["minimum_columns"] == [
        "work_numeric_id",
        "publication_day",
        "publication_year",
        "temporal_disposition",
    ]
    assert spine["required_rows"] == 492293206
    assert spine["require_one_to_one_retained_main_ids"] is True
    assert spine["eliminate_citation_main_rescan"] is True
    assert spine["eliminate_author_main_rescan_with_time_columns_only"] is False
    assert spine["universal_work_spine_recommended_now"] is False

    partition_plan = config["partition_plan"]
    assert partition_plan["mode_specific_temporal_input_stage"] == {
        "canonical_work_time_spine_reuse": (
            "validate_preexisting_full_canonical_spine_receipt_and_tree_only"
        ),
        "exact_temporal_reconstruction_from_current_source": (
            "build_ephemeral_noncanonical_direct_mode_time_index"
        ),
    }
    assert partition_plan["citation_component_may_publish_canonical_spine"] is False

    resources = config["resource_envelope"]
    assert resources["direct_reconstruction_input_files"] == 3537
    assert resources["direct_reconstruction_projected_rows"] == 3534829367
    assert resources["direct_reconstruction_source_bytes"] == 187182118540
    assert resources["spine_reuse_source_input_files"] == 3044
    assert resources["spine_reuse_projected_source_rows"] == 3042536160
    assert resources["spine_reuse_source_bytes"] == 37119669321
    assert resources["spine_reuse_spine_partition_input_files"] == 4096
    assert resources["spine_reuse_spine_rows"] == 492293206
    assert resources["spine_reuse_minimum_physical_data_input_files"] == 7140
    assert resources["spine_reuse_spine_partition_bytes"] == (
        "BIND_EXACT_VALUE_FROM_VALIDATED_SPINE_RECEIPT"
    )
    assert resources["spine_reuse_removed_main_scan_bytes"] == 150062449219
    assert resources["pair_ids_only_uncompressed_bytes"] == 3042536160 * 16
    assert resources["ram_gib_minimum_planning"] == 64
    assert resources["ram_gib_preferred_planning"] == 128
    assert resources["temporary_ssd_gib_minimum"] == 300
    assert resources["temporary_ssd_gib_preferred"] == 500
    assert resources["forbid_hdd_competition_with_content_freeze"] is True
    assert resources["gpu_used"] is False

    privacy = config["release_privacy"]
    assert privacy["public_summary_allows_aggregates_and_checksums_only"] is True
    assert privacy["detailed_work_ids_repository_allowed"] is False
    assert privacy["detailed_work_ids_release_allowed"] is False
    assert privacy["operator_local_detailed_outputs_mode"] == "0600"
    assert privacy["operator_local_directory_mode"] == "0700"
    assert privacy["local_artifact_tree_hash_required"] is True
    assert set(privacy["release_forbidden_operator_local_basenames"]) == {
        "ephemeral_noncanonical_direct_mode_time_index",
        "deduplicated_edge_event_partitions",
        "chronology_safe_edge_partitions_by_cited_work",
        "duplicate_multiplicity_aggregates.parquet",
        "chronology_lag_aggregates.parquet",
        "open_world_target_coverage.parquet",
        "cutoff_and_maturation_coverage.parquet",
    }

    outputs = config["planned_outputs"]
    assert "canonical_work_time_spine" not in outputs["operator_local_data"]
    assert outputs[
        "canonical_work_time_spine_created_or_published_by_citation_component"
    ] is (False)
    assert outputs["temporary_mode_specific_data_are_not_completion_outputs"] is True
    completion = config["completion_contract"]
    assert completion["pre_receipt_state"] == "COMPLETE_PENDING_RECEIPT"
    assert completion["receipt_written_last"] is True
    assert completion["exact_receipt_artifact_set_required"] is True
    assert completion[
        "final_validation_rehashes_all_control_checkpoint_and_data_artifacts"
    ]


def test_every_non_full_mode_is_explicitly_unpromotable() -> None:
    rules = _load_config()["pilot_and_full_rules"]
    assert rules["max_files"]["promotion_allowed"] is False
    assert rules["isolated_part_file"]["representative_scientific_pilot_allowed"] is (
        False
    )
    assert rules["isolated_part_file"]["promotion_allowed"] is False
    assert rules["deterministic_hash_bucket"][
        "require_all_inputs_for_precommitted_mode"
    ]
    bucket = rules["deterministic_hash_bucket"]
    assert bucket["canonical_key_bytes"] == (
        "uint64_citing_id_big_endian_then_uint64_cited_id_big_endian"
    )
    assert bucket["hash_function"] == "SHA256"
    assert bucket["bucket_integer"] == "unsigned_big_endian_first_8_digest_bytes"
    assert bucket["bucket_formula"] == "bucket_integer_mod_bucket_count"
    assert bucket["bucket_count_and_index_are_part_of_task_fingerprint"] is True
    assert bucket["identical_pair_rows_always_select_together"] is True
    assert rules["deterministic_hash_bucket"]["global_coverage_claim_allowed"] is False
    assert rules["deterministic_hash_bucket"]["promotion_allowed"] is False
    assert rules["full"]["explicit_authorization_required"] is True
    assert rules["full"]["require_all_inputs_for_precommitted_mode"] is True
    assert rules["full"]["direct_mode_required_source_files"] == 3537
    assert rules["full"]["spine_reuse_required_source_files"] == 3044
    assert rules["full"]["spine_reuse_requires_complete_spine_receipt"] is True
    assert rules["full"]["require_exact_mode_conditioned_parent_set"] is True
    assert rules["full"]["require_exact_accounting_closure"] is True
    assert rules["full"]["required_pre_receipt_state"] == "COMPLETE_PENDING_RECEIPT"
    assert rules["full"]["required_receipt_completion_contract"] == (
        "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
    )
    assert rules["full"]["may_promote_only_component_scope"] is True


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        ("lifecycle.production_status", "PASS"),
        ("lifecycle.production_execution_supported", True),
        ("lifecycle.synthetic_fixture_only", False),
        ("available_at_policy.openalex_observation_time_established", True),
        ("available_at_policy.historical_openalex_graph_established", True),
        ("open_world_policy.external_target_is_orphan", True),
        ("maturation_contract.total_citation_claim_allowed", True),
        (
            "maturation_contract."
            "temporal_maturity_alone_establishes_historical_availability",
            True,
        ),
        ("leakage_policy.current_cited_by_count_used", True),
        (
            "leakage_policy.retrospective_backfill_may_be_called_historically_observed",
            True,
        ),
        (
            "temporal_input_modes.canonical_work_time_spine_reuse."
            "citation_component_may_build_or_repair_canonical_spine",
            True,
        ),
        (
            "future_success_contract.required_full_gate_states."
            "openalex_first_observed_time",
            "PASS",
        ),
        (
            "future_success_contract.required_full_gate_states."
            "g_minus_1_data_readiness",
            "PASS",
        ),
        ("release_privacy.detailed_work_ids_release_allowed", True),
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


def test_document_and_config_are_synchronized_design_artifacts() -> None:
    config = _load_config()
    text = DOC_PATH.read_text(encoding="utf-8")
    required_text = (
        "Status: `IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN`",
        "Production status: `PRODUCTION_NOT_RUN`",
        "BIBLIOGRAPHIC_EVENT_TIME_CONVENTION",
        "3,042,536,160",
        "138,015,847",
        "Seven exact exclusive pair buckets",
        "CITATION_DIRECT_RECONSTRUCTION_V1",
        "CITATION_CANONICAL_SPINE_REUSE_V1",
        "temporally mature",
        "ephemeral_noncanonical_direct_mode_time_index",
        config["future_success_contract"]["full_component_status"],
        "openalex_first_observed_time=UNIDENTIFIABLE_FROM_CURRENT_SNAPSHOT",
        "It uses no GPU",
    )
    assert all(item in text for item in required_text)
    assert SCRIPT_PATH.is_file()
    assert config["planned_outputs"]["production_outputs_created_by_this_design"] is (
        False
    )
    result_dir = REPO_ROOT / "experiments/evidence/results"
    assert {
        path for pattern in RESULT_GLOBS for path in result_dir.glob(pattern)
    } == set()
