from __future__ import annotations

import copy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
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
    REPO_ROOT / "experiments/configs/openalex_g1_work_side_author_history_v0.yaml"
)
DOC_PATH = (
    REPO_ROOT / "docs/case-studies/researcher-pivot/"
    "openalex-work-side-author-history-audit-design.md"
)
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_work_side_author_history.py"
RESULT_GLOB = "openalex_g1_work_side_author_history*"


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that fails closed on duplicate mapping keys."""

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


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _load_yaml_unique(text: str) -> Any:
    return yaml.load(text, Loader=_UniqueKeyLoader)


def _load_config() -> dict[str, Any]:
    value = _load_yaml_unique(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _domain_hash(domain: str, value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain.encode("utf-8") + b"\0" + payload).hexdigest()


def _reduce_year_records_for_author(
    year_records: list[dict[str, int]], author_numeric_id: int
) -> dict[str, int]:
    same_author = [
        record
        for record in year_records
        if record["author_numeric_id"] == author_numeric_id
    ]
    years = [record["publication_year"] for record in same_author]
    annual_counts = [record["unique_work_count"] for record in same_author]
    first_year = min(years)
    last_year = max(years)
    inclusive_span = last_year - first_year + 1
    return {
        "author_numeric_id": author_numeric_id,
        "first_usable_publication_year": first_year,
        "last_usable_publication_year": last_year,
        "active_year_count": len(same_author),
        "inclusive_span_years": inclusive_span,
        "inactive_gap_years": inclusive_span - len(same_author),
        "total_unique_work_count": sum(annual_counts),
        "maximum_annual_unique_work_count": max(annual_counts),
    }


def _same_author_year_token_reductions_are_exact(
    year_records: list[dict[str, int]], token_records: list[dict[str, int]]
) -> bool:
    year_keys = [
        (record["author_numeric_id"], record["publication_year"])
        for record in year_records
    ]
    year_authors = {record["author_numeric_id"] for record in year_records}
    token_authors = {record["author_numeric_id"] for record in token_records}
    if (
        len(year_keys) != len(set(year_keys))
        or year_authors != token_authors
        or len(token_records) != len(token_authors)
    ):
        return False
    expected = [
        _reduce_year_records_for_author(year_records, author_numeric_id)
        for author_numeric_id in sorted(year_authors)
    ]
    return token_records == expected


def _is_exact_positive_author_projection(
    base_records: list[dict[str, int]], positive_records: list[dict[str, int]]
) -> bool:
    return positive_records == [
        record for record in base_records if record["author_numeric_id"] > 0
    ]


def _reduce_hidden_cap_records_to_grid(
    *,
    view: str,
    caps: list[int | str],
    thresholds: list[int],
    capped_records: dict[str, list[dict[str, int]]],
) -> dict[str, Any]:
    pair_counts: dict[str, int] = {}
    cells: dict[str, int] = {}
    for cap in caps:
        cap_key = str(cap)
        records = capped_records[cap_key]
        pair_keys = [
            (record["author_numeric_id"], record["work_numeric_id"])
            for record in records
        ]
        assert len(pair_keys) == len(set(pair_keys))
        pair_counts[cap_key] = len(records)
        author_year_works: dict[tuple[int, int], set[int]] = {}
        for record in records:
            key = (record["author_numeric_id"], record["publication_year"])
            author_year_works.setdefault(key, set()).add(record["work_numeric_id"])
        maximum_annual: dict[int, int] = {}
        for (author_numeric_id, _), work_ids in author_year_works.items():
            maximum_annual[author_numeric_id] = max(
                maximum_annual.get(author_numeric_id, 0), len(work_ids)
            )
        for threshold in thresholds:
            cells[f"cap_{cap_key}_gt_{threshold}"] = sum(
                annual_count > threshold for annual_count in maximum_annual.values()
            )
    return {
        "view": view,
        "cap_pair_counts": pair_counts,
        "cap_threshold_token_grid": cells,
    }


def _set_nested(value: dict[str, Any], path: str, replacement: Any) -> None:
    cursor = value
    parts = path.split(".")
    for part in parts[:-1]:
        child = cursor[part]
        assert isinstance(child, dict)
        cursor = child
    cursor[parts[-1]] = replacement


def _validate_claim_boundary(config: dict[str, Any]) -> None:
    lifecycle = config["lifecycle"]
    assert lifecycle == {
        "design_status": "DESIGN_ONLY",
        "production_status": "PRODUCTION_NOT_RUN",
        "executable": False,
        "execution_script": None,
        "synthetic_executable": False,
        "production_executable": False,
        "production_result_exists": False,
        "synthetic_result_exists": False,
        "pass_result_allowed_in_design_artifact": False,
        "implementation_requires_new_review": True,
    }

    readiness = config["implementation_readiness"]
    assert readiness == {
        "synthetic_contract_status": (
            "READY_FOR_SEPARATE_SYNTHETIC_ONLY_IMPLEMENTATION"
        ),
        "executable_created_by_this_tranche": False,
        "result_created_by_this_tranche": False,
        "production_output_abi": "NOT_IMPLEMENTED_REQUIRES_SEPARATE_REVIEW",
        "production_execution_authorized": False,
    }

    assert config["required_parent_execution_order"] == [
        "current_content_baseline",
        "main_temporal_validity",
        "canonical_work_time_spine",
        "work_side_author_history",
    ]
    parents = config["parents"]
    assert parents["current_content_baseline"]["required_run_state"] == "COMPLETE"
    assert parents["current_content_baseline"]["required_full_scope"] is True
    assert parents["main_temporal_validity"]["required_full_scope"] is True
    assert parents["canonical_work_time_spine"]["required_full_scope"] is True
    assert (
        parents["main_temporal_validity"]["current_state"]
        == "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN"
    )
    assert parents["canonical_work_time_spine"]["current_state"] == (
        "IMPLEMENTED_SYNTHETICALLY_PRODUCTION_NOT_RUN"
    )
    assert parents["main_temporal_validity"]["require_same_content_baseline_parent"]
    assert (
        parents["main_temporal_validity"]["row_level_dispositions_materialized"]
        is False
    )
    sibling = config["parallel_siblings"]["work_topic_relation"]
    assert sibling["hard_parent_of_author_token_history"] is False
    assert sibling["relation_to_core"] == "parallel_sibling"

    identity = config["identity_policy"]
    assert identity["unit_name"] == "current_snapshot_openalex_author_id_token"
    assert identity["token_not_person"] is True
    assert identity["natural_person_entity_claim_allowed"] is False
    assert identity["author_master_referential_integrity_claim_allowed"] is False
    assert identity["token_reassignment_in_component"] == "forbidden"
    assert identity["name_or_orcid_used_to_reassign_tokens"] is False
    assert identity["historical_identity_availability_established"] is False
    assert identity["past_observable_token_linkage_claim_allowed"] is False

    gates = config["future_success_contract"]["required_full_gate_states"]
    assert gates["author_entity_validity"] == "UNRESOLVED_WORK_SIDE_ONLY"
    assert (
        gates["author_master_coverage"] == "BLOCKED_INCOMPLETE_MAX_38_76968115_PERCENT"
    )
    assert gates["split_merge_validity"] == ("UNRESOLVED_PROXY_DIAGNOSTICS_ONLY")
    assert gates["orcid_validation"] == "DIAGNOSTIC_NOT_GROUND_TRUTH"
    assert gates["candidate_eligibility"] == "PENDING"
    assert gates["canonical_work_time_spine_component"] == "PASS"
    assert "work_topic_relation_component" not in gates
    assert gates["token_and_field_past_observability"] == (
        "UNVERIFIED_CURRENT_SNAPSHOT_RETROSPECTIVE_PROXY"
    )
    assert gates["g_minus_1_data_readiness"] == "PENDING"
    assert gates["g0_measurement_freeze"] == "PENDING"
    assert gates["empirical_g1_signature"] == "NOT_RUN"
    assert gates["pass_g1"] is False
    assert gates["scientific_effect"] == "NOT_ESTABLISHED"

    incomplete_master = config["incomplete_author_master"]
    assert incomplete_master["full_parent_role_allowed"] is False
    assert incomplete_master["author_entity_validation_allowed"] is False
    assert (
        config["sensitivity_policy"]["threshold_driven_candidate_exclusion_allowed"]
        is False
    )
    observability = config["historical_observability_and_cutoff"]
    assert observability["token_or_field_past_observability_established"] is False
    assert observability["full_snapshot_summaries_role"] == (
        "AUDIT_ONLY_NOT_PAST_CUTOFF_FEATURES"
    )
    assert (
        config["release_privacy"]["detailed_identity_outputs_release_allowed"] is False
    )

    claim = config["claim_boundary"]
    required_phrases = (
        "DESIGN_ONLY",
        "tokens, not verified persons",
        "historically observable",
        "parallel sibling",
        "No executable or result exists",
        "candidate eligibility",
    )
    assert all(phrase in claim for phrase in required_phrases)


def test_yaml_trust_boundary_rejects_duplicates_aliases_merges_and_tags() -> None:
    assert _load_config()["schema_version"] == (
        "openalex_g1_work_side_author_history_design.v1"
    )
    with pytest.raises(ConstructorError, match="duplicate key 'status'"):
        _load_yaml_unique("status: DESIGN_ONLY\nstatus: PASS\n")
    with pytest.raises(ConstructorError, match="aliases are forbidden"):
        _load_yaml_unique("left: &value 1\nright: *value\n")
    with pytest.raises(ConstructorError, match="merge keys are forbidden"):
        _load_yaml_unique("copy:\n  <<: {x: 1}\n")
    with pytest.raises(ConstructorError):
        _load_yaml_unique("value: !unsafe payload\n")


def test_design_lifecycle_parents_and_claim_boundary_are_frozen() -> None:
    _validate_claim_boundary(_load_config())


def test_projection_and_accounting_anchors_are_exact() -> None:
    config = _load_config()
    tables = config["tables"]
    assert tables["main"]["projected_columns"] == [
        "id",
        "authors_count",
        "primary_topic__field__id",
    ]
    assert tables["authorships"]["projected_columns"] == [
        "id",
        "authorships__author__id",
        "authorships__author__orcid",
        "authorships__raw_author_name",
        "authorships__author__display_name",
        "authorships__author_position",
    ]
    assert tables["main"]["projected_physical_nullable"] == {
        "id": True,
        "authors_count": True,
        "primary_topic__field__id": True,
    }
    assert all(tables["authorships"]["projected_physical_nullable"].values())

    totals = config["scope_totals"]
    assert totals["expected_files"] == sum(
        table["expected_files"] for table in tables.values()
    )
    assert totals["expected_raw_rows"] == sum(
        table["expected_raw_rows"] for table in tables.values()
    )
    assert totals["expected_source_bytes"] == sum(
        table["expected_source_bytes"] for table in tables.values()
    )
    assert totals == {
        "expected_files": 1856,
        "expected_raw_rows": 1853844245,
        "expected_source_bytes": 253474946263,
        "expected_source_gib_rounded_3dp": 236.067,
        "gpu_used": False,
    }

    anchors = config["accounting_anchors"]
    main = anchors["main"]
    authorships = anchors["authorships"]
    work_classes = anchors["work_classes"]
    assert main["retained_rows"] == (
        main["raw_rows"] - main["frozen_w_minus_1_quarantine_rows"]
    )
    assert authorships["retained_rows"] == (
        authorships["raw_rows"] - authorships["frozen_w_minus_1_source_rows"]
    )
    assert authorships["retained_rows"] == sum(
        authorships[key]
        for key in (
            "null_author_id_rows",
            "grammar_valid_non_null_author_id_rows",
            "malformed_non_null_author_id_rows",
        )
    )
    assert authorships["retained_rows"] == (
        authorships["non_null_orcid_rows"] + authorships["null_orcid_rows"]
    )
    assert main["retained_rows"] == sum(work_classes.values())
    assert config["accounting_anchors"]["declared_authors_count"] == {
        "exact_matches_observed_raw_slots": 492293206,
        "mismatches": 0,
    }
    assert len(config["exact_invariants"]) == 11


def test_incomplete_master_ceiling_is_exact_and_non_promotable() -> None:
    master = _load_config()["incomplete_author_master"]
    numerator = Decimal(master["observed_delta_rows"])
    denominator = Decimal(master["work_side_distinct_author_tokens"])
    places = int(master["reported_decimal_places"])
    quantum = Decimal(1).scaleb(-places)
    observed = (numerator * Decimal(100) / denominator).quantize(
        quantum,
        rounding=ROUND_HALF_UP,
    )
    assert observed == Decimal(str(master["maximum_theoretical_coverage_percent"]))
    assert master["maximum_theoretical_coverage_percent"] == 38.76968115
    assert master["full_parent_role_allowed"] is False
    assert master["author_entity_validation_allowed"] is False
    assert master["allowed_role"] == (
        "LINKED_SUBSET_DIAGNOSTIC_ONLY_AFTER_SEPARATE_LINEAGE_CONTRACT"
    )


def test_parent_abis_hashes_and_transitive_commitments_are_frozen() -> None:
    config = _load_config()
    parents = config["parents"]

    content = parents["current_content_baseline"]
    temporal = parents["main_temporal_validity"]
    spine = parents["canonical_work_time_spine"]
    for parent, config_key, hash_key in (
        (content, "required_config_path", "required_config_sha256"),
        (temporal, "required_config_path", "required_config_sha256"),
        (spine, "design_config_path", "design_config_sha256"),
    ):
        assert _sha256(REPO_ROOT / parent[config_key]) == parent[hash_key]
    for parent, tool_key, hash_key in (
        (content, "required_tool_path", "required_tool_sha256"),
        (temporal, "required_tool_path", "required_tool_sha256"),
        (spine, "synthetic_tool_path", "synthetic_tool_sha256"),
    ):
        assert _sha256(REPO_ROOT / parent[tool_key]) == parent[hash_key]

    relation = config["parallel_siblings"]["work_topic_relation"]
    assert _sha256(REPO_ROOT / relation["config_path"]) == relation["config_sha256"]
    assert _sha256(REPO_ROOT / relation["tool_path"]) == relation["tool_sha256"]
    assert relation["hard_parent_of_author_token_history"] is False
    assert relation["relation_to_core"] == "parallel_sibling"
    for prerequisite in config["coverage_prerequisites"].values():
        assert (
            _sha256(REPO_ROOT / prerequisite["relative_path"]) == prerequisite["sha256"]
        )

    assert content["receipt_schema_version"] == "openalex_g1_content_freeze_outputs.v1"
    assert content["exact_receipt_artifact_count"] == 3
    assert len(content["exact_receipt_keyset"]) == 11
    assert temporal["receipt_schema_version"] == (
        "openalex_g1_main_temporal_validity_outputs.v1"
    )
    assert temporal["exact_checkpoint_artifact_count"] == 493
    assert temporal["exact_receipt_artifact_count"] == 497
    assert spine["required_future_receipt_schema_version"] == (
        "openalex_g1_canonical_work_time_spine_outputs.v1"
    )
    assert spine["required_rows"] == 492293206
    assert spine["required_partition_count"] == 4096
    assert spine["required_receipt_artifact_count"] == 9184
    assert [field["name"] for field in spine["required_row_schema"]] == [
        "work_id_numeric",
        "publication_day",
        "publication_year",
        "temporal_disposition",
    ]

    cross = config["parent_cross_binding"]
    assert all(cross.values()) is False  # final field records no verified receipts
    assert all(
        cross[key]
        for key in cross
        if key != "production_parent_receipts_verified_by_this_design"
    )


def test_spine_temporal_dispositions_are_unique_disjoint_and_exhaustive() -> None:
    temporal = _load_config()["temporal_reuse"]
    usable = temporal["usable_disposition_codes"]
    excluded = temporal["excluded_disposition_codes"]
    assert usable == [2, 3]
    assert excluded == [0, 1, 4, 5]
    assert len(usable) == len(set(usable))
    assert len(excluded) == len(set(excluded))
    assert set(usable).isdisjoint(excluded)
    assert set(usable) | set(excluded) == set(range(6))
    assert temporal["only_row_level_source"] == "canonical_work_time_spine"
    assert temporal["source_publication_date_reparse_allowed"] is False


def test_exact_dedup_order_position_and_cutoff_contracts_are_frozen() -> None:
    config = _load_config()
    pair = config["pair_policy"]
    assert pair["exact_key"] == ["author_numeric_id", "work_numeric_id"]
    assert pair["representative_row_selection"] == "forbidden"
    assert set(pair["preserve_across_duplicate_rows"]) == {
        "raw_multiplicity",
        "exact_raw_orcid_set",
        "canonical_valid_orcid_set",
        "exact_raw_author_name_set",
        "exact_display_name_set",
        "normalized_raw_author_name_set",
        "normalized_display_name_set",
        "author_position_category_set",
        "exact_raw_position_set",
        "conflict_flags",
    }
    assert set(pair["conflict_flag_predicates"]) == {
        "raw_orcid_conflict",
        "canonical_orcid_conflict",
        "raw_author_name_conflict",
        "display_name_conflict",
        "normalized_raw_name_conflict",
        "normalized_display_name_conflict",
        "position_category_conflict",
        "raw_position_conflict",
        "position_conflict",
    }
    assert pair["tagged_raw_set_encoding"] == {
        "null": {"kind": "null"},
        "string": {"kind": "string", "value": "exact_utf8_string"},
    }
    assert set(pair["null_and_empty_set_semantics"]) == {
        "exact_raw_orcid_set",
        "exact_raw_author_name_set",
        "exact_display_name_set",
        "normalized_raw_author_name_set",
        "normalized_display_name_set",
        "exact_raw_position_set",
        "canonical_valid_orcid_set",
    }

    position = config["authorship_order_and_position"]
    assert position["categories"] == [
        "first",
        "middle",
        "last",
        "null",
        "unexpected_non_null",
    ]
    assert position["physical_row_order_semantic"] is False
    assert position["physical_row_order_may_break_ties"] is False
    assert position["middle_author_total_order_recoverable"] is False
    assert position["category_set_sort_comparator"] == "declared_category_order"
    assert position["category_set_sort_order"] == [
        "first",
        "middle",
        "last",
        "null",
        "unexpected_non_null",
    ]
    assert position["category_set_duplicates_allowed"] is False
    assert (
        position[
            "trimming_casefold_or_unicode_normalization_before_classification_allowed"
        ]
        is False
    )
    assert position["lexeme_classification"]["unexpected_non_null"] == (
        "every_other_string_including_empty_case_variants_and_whitespace_variants"
    )

    populations = config["diagnostic_population_contract"]
    assert populations["primary_population"] == (
        "all_retained_grammar_valid_distinct_author_work_pairs"
    )
    assert populations["secondary_population"] == (
        "history_usable_distinct_author_work_pairs_with_temporal_disposition_2_or_3"
    )
    assert (
        populations[
            "duplicate_rows_contribute_once_per_distinct_pair_after_unioning_tagged_sets"
        ]
        is True
    )
    assert populations["compute_and_digest_both_populations_for"] == [
        "orcid_categories_and_cross_cardinalities",
        "exact_and_normalized_name_variation",
        "position_categories_and_raw_lexeme_conflicts",
    ]
    assert populations["synthetic_public_report_exposes_for_each_population"] == [
        "distinct_pair_count",
        "opaque_canonical_aggregate_sha256",
    ]
    assert (
        populations[
            "readable_category_or_cardinality_tables_in_synthetic_public_report"
        ]
        is False
    )

    cutoff = config["historical_observability_and_cutoff"]
    assert cutoff["required_cutoff"] == (
        "publication_date_lte_december_31_of_t_minus_1"
    )
    assert cutoff["downstream_recompute_from_cutoff_eligible_unique_author_work_events"]
    assert cutoff["token_assignment_is_current_snapshot_retrospective_proxy"]
    assert cutoff["primary_field_is_current_snapshot_retrospective_annotation"]
    assert cutoff["token_or_field_past_observability_established"] is False


def test_orcid_and_unicode_normalization_abis_are_exact() -> None:
    config = _load_config()
    orcid = config["orcid_policy"]
    assert orcid["outer_whitespace_trimming_allowed"] is False
    assert orcid["percent_decoding_allowed"] is False
    assert len(orcid["recoverable_alternate_lexical_forms_in_order"]) == 5
    assert orcid["checksum"] == {
        "resolution": "ISO_7064_MOD_11_2",
        "input": "first_15_ascii_digits",
        "recurrence": "total = (total + digit) * 2",
        "remainder": "total % 11",
        "check_value": "(12 - remainder) % 11",
        "check_character": "decimal_digit_unless_10_then_uppercase_X",
    }
    assert orcid["split_merge_semantics"] == "anomaly_proxy_not_ground_truth"

    names = config["name_policy"]
    assert names["diagnostic_normalization_steps"] == [
        "unicode_nfkc",
        "unicode_casefold",
        "map_each_python_str_isspace_character_to_ascii_space",
        "collapse_ascii_space_runs",
        "strip_outer_ascii_space",
    ]
    assert names["normalization_runtime_fingerprint_fields"] == [
        "python_implementation",
        "python_version",
        "unicodedata_unidata_version",
    ]
    assert names["runtime_unicode_version_must_match_plan_and_resume"] is True


def test_synthetic_only_abi_is_exact_but_not_implemented() -> None:
    config = _load_config()
    contract = config["synthetic_execution_contract"]
    assert contract["readiness"] == "READY_FOR_SEPARATE_IMPLEMENTATION"
    assert contract["executable_exists_now"] is False
    assert contract["production_or_source_root_cli_allowed"] is False
    assert contract["exact_future_cli"] == [
        "--config",
        "--synthetic-fixture",
        "--out-dir",
        "--execute-synthetic",
        "--resume",
    ]
    assert contract["fixture_exact_top_level_keys"] == [
        "schema_version",
        "surface_kind",
        "production_input_access_allowed",
        "root_id",
        "snapshot_date",
        "synthetic_parents",
        "main_rows",
        "spine_rows",
        "authorship_rows",
    ]
    assert contract["fixture_exact_top_level_types"] == {
        "schema_version": "string",
        "surface_kind": "string",
        "production_input_access_allowed": "boolean",
        "root_id": "string",
        "snapshot_date": "string",
        "synthetic_parents": "object",
        "main_rows": "array_of_objects",
        "spine_rows": "array_of_objects",
        "authorship_rows": "array_of_objects",
    }
    assert contract["fixture_snapshot_date_exact"] == "2026-03-30"
    assert contract["fixture_main_row_exact_keyset"] == [
        "id",
        "authors_count",
        "primary_topic__field__id",
    ]
    assert contract["fixture_spine_row_exact_keyset"] == [
        "work_id_numeric",
        "publication_day",
        "publication_year",
        "temporal_disposition",
    ]
    spine_semantics = contract["fixture_spine_semantic_validation"]
    assert spine_semantics["temporal_disposition_exact_allowed_values"] == list(
        range(6)
    )
    assert spine_semantics["code_0_requires_both_publication_day_and_year_null"]
    assert spine_semantics[
        "codes_1_through_5_require_both_publication_day_and_year_non_null"
    ]
    assert spine_semantics["non_null_year_must_equal_year_derived_from_publication_day"]
    assert spine_semantics["code_6_through_255_rejected_before_accounting"]
    assert spine_semantics[
        "fixture_spine_numeric_id_set_must_exactly_equal_retained_main_numeric_id_set"
    ]
    assert spine_semantics[
        "missing_duplicate_or_extra_spine_rows_rejected_before_accounting"
    ]
    assert contract["fixture_authorship_row_exact_keyset"] == [
        "id",
        "authorships__author__id",
        "authorships__author__orcid",
        "authorships__raw_author_name",
        "authorships__author__display_name",
        "authorships__author_position",
    ]
    assert contract["strict_json_duplicate_keys_rejected"] is True
    assert contract["strict_json_nan_infinity_rejected"] is True
    assert contract["strict_json_lone_unicode_surrogates_rejected"] is True
    assert contract["hard_max_fixture_bytes"] == 1048576
    assert contract["hard_max_main_rows"] == 10000
    assert contract["hard_max_spine_rows"] == 10000
    assert contract["hard_max_authorship_rows"] == 20000
    trust = contract["config_trust_boundary"]
    assert trust["duplicate_mapping_keys_rejected"] is True
    assert trust["yaml_aliases_merge_keys_and_custom_tags_rejected"] is True
    assert trust["future_tool_embeds_audited_config_sha256"] is True
    assert trust["supplied_config_sha256_must_equal_tool_embedded_sha256"] is True
    assert (
        trust["security_critical_values_are_code_constants_and_must_equal_config"]
        is True
    )
    assert trust["security_critical_exact_values"] == {
        "production_or_source_root_cli_allowed": False,
        "production_input_access_allowed": False,
        "hard_max_fixture_bytes": 1048576,
        "hard_max_main_rows": 10000,
        "hard_max_spine_rows": 10000,
        "hard_max_authorship_rows": 20000,
        "output_directory_mode": "0700",
        "output_file_mode": "0600",
    }
    assert contract["production_input_access_allowed"] is False
    assert contract["fixture_lstat_and_size_checked_before_payload_open"] is True
    assert contract["fixture_root_id_is_not_available_before_payload_parse"] is True
    assert (
        contract[
            "fixture_root_id_equal_to_or_containing_production_root_id_refused_after_strict_parse_before_row_validation"
        ]
        is True
    )
    assert contract["production_parity_established"] is False
    assert contract["cutoff_contract_is_downstream_consumer_only"] is True
    assert contract["synthetic_cutoff_materialization_or_result_claim_allowed"] is (
        False
    )
    assert contract["exact_receipt_artifacts"] == [
        "audit_plan.json",
        "run_state.json",
        "checkpoints/synthetic_accounting.json",
        "author_history_audit.json",
        "author_history_audit.md",
    ]
    assert contract["exact_receipt_artifact_count"] == 5
    assert contract["receipt_artifact_record_exact_keyset"] == [
        "relative_path",
        "bytes",
        "sha256",
    ]
    assert (
        contract["receipt_artifact_mapping_exact_keyset"]
        == (contract["exact_receipt_artifacts"])
    )
    assert contract["pre_receipt_state"] == "COMPLETE_PENDING_RECEIPT"
    assert contract["receipt_written_last"] is True
    assert contract["terminal_reentry_opens_fixture_payload"] is False
    assert contract["exact_final_file_paths"] == [
        *contract["exact_receipt_artifacts"],
        "author_history_audit.outputs.json",
    ]
    assert contract["exact_final_directory_paths"] == ["checkpoints"]
    assert contract["resume_contract"] == {
        "nonempty_output_requires_resume_flag": True,
        "exact_existing_plan_required": True,
        "verify_config_tool_runtime_fixture_and_parent_commitments": True,
        "verify_checkpoint_schema_self_hash_task_and_metrics_hash": True,
        "valid_terminal_receipt_is_terminal": True,
        "terminal_validation_precedes_fixture_payload_open": True,
        "valid_terminal_reentry_zero_mutation": True,
        "zero_mutation_fields_for_every_existing_entry": [
            "bytes",
            "sha256",
            "inode",
            "mode",
            "mtime_ns",
            "ctime_ns",
        ],
        "atime_is_not_part_of_zero_mutation_assertion": True,
        "execution_mode_or_cap_override_allowed": False,
        "failed_state_resume_requires_successful_full_revalidation_before_running": True,
        "failed_state_resume_revalidation_exact_checks": [
            "exact_existing_plan",
            "config_tool_runtime_fixture_and_parent_commitments",
            "checkpoint_schema_self_hash_task_and_metrics_hash",
            "failure_state_tuple_and_receipt_absence",
            "output_types_modes_link_counts_and_exact_failed_entry_set",
        ],
        "failed_state_exact_existing_file_paths": [
            "audit_plan.json",
            "run_state.json",
            "checkpoints/synthetic_accounting.json",
        ],
        "failed_state_exact_existing_directory_paths": ["checkpoints"],
        "failed_state_forbidden_file_paths": [
            "author_history_audit.json",
            "author_history_audit.md",
            "author_history_audit.outputs.json",
        ],
        "failed_state_temporary_files_must_be_absent": True,
        "finalization_failed_state_requires_valid_resumable_checkpoint": True,
        "successful_failed_state_resume_next_status": (
            "RUNNING_SYNTHETIC_AUTHOR_HISTORY"
        ),
        "failed_resume_revalidation_status_remains": (
            "FINALIZATION_FAILED_NO_VALID_RECEIPT"
        ),
        "failed_resume_revalidation_receipt_must_be_absent": True,
        "direct_failed_state_to_complete_allowed": False,
        "same_task_clean_and_successfully_resumed_result_fingerprint_must_equal": True,
    }
    safety = contract["source_and_output_safety"]
    assert safety["published_lock_file_created"] is False
    assert safety["fixture_lstat_type_must_be_regular_file"] is True
    assert safety["fixture_link_count_must_equal_one"] is True
    assert safety["fixture_open_flags"] == "O_RDONLY|O_NOFOLLOW"
    assert safety["reject_fixture_output_same_resolved_path"] is True
    assert safety["reject_fixture_output_ancestor_or_descendant_overlap"] is True
    assert (
        safety["every_existing_and_published_output_file_link_count_must_equal_one"]
        is True
    )
    assert (
        safety["final_validation_rechecks_root_0700_and_all_directories_0700"] is True
    )
    assert (
        safety[
            "final_validation_rechecks_all_files_0600_regular_owner_and_link_count_one"
        ]
        is True
    )
    assert contract["synthetic_report_raw_author_or_work_identifiers_allowed"] is (
        False
    )
    assert contract["failure_invalidation_order"][0] == (
        "remove_authoritative_completion_receipt_first"
    )
    assert config["production_output_contract"]["status"] == (
        "NOT_IMPLEMENTED_REQUIRES_SEPARATE_REVIEW"
    )
    assert SCRIPT_PATH.exists() is False


def test_synthetic_output_schemas_and_hash_domains_are_exact() -> None:
    contract = _load_config()["synthetic_execution_contract"]
    schemas = contract["output_json_schemas"]
    assert set(schemas) == {
        "audit_plan",
        "run_state",
        "synthetic_accounting_checkpoint",
        "author_history_report",
    }
    for schema in schemas.values():
        assert set(schema["exact_keyset"]) == set(schema["exact_field_types"])

    nested = contract["output_nested_object_schemas"]
    assert nested["row_counts"]["exact_keyset"] == [
        "main_rows",
        "spine_rows",
        "authorship_rows",
    ]
    assert nested["checkpoint_metrics_exact_keyset"] == [
        "source_accounting",
        "work_and_slot_accounting",
        "author_id_accounting",
        "pair_accounting",
        "temporal_accounting",
        "history_accounting",
        "team_and_hyper_sensitivity",
        "orcid_name_and_position_diagnostics",
        "gate_status",
    ]
    assert nested["pair_accounting"]["exact_keyset"][-3:] == [
        "position_category_conflict_pairs",
        "raw_position_conflict_pairs",
        "combined_position_conflict_pairs",
    ]
    assert nested["team_cap_threshold_grid"]["views"] == [
        "base",
        "positive_author_id",
    ]
    assert nested["team_cap_threshold_grid"]["caps"] == [
        "10",
        "50",
        "100",
        "ALL",
    ]
    assert nested["team_cap_threshold_grid"]["thresholds"] == [
        "gt_50",
        "gt_100",
        "gt_250",
        "gt_500",
        "gt_1000",
    ]
    assert len(nested["team_cap_threshold_grid"]["exact_keyset"]) == 20
    assert nested["team_cap_threshold_grid"]["per_threshold_cap_closure"] == (
        "cap_10_lte_cap_50_lte_cap_100_lte_cap_ALL"
    )
    assert (
        nested["team_cap_threshold_grid"][
            "positive_author_id_cell_lte_corresponding_base_cell"
        ]
        is True
    )
    assert set(nested["team_and_hyper_sensitivity"]["closure"]) == {
        "positive_author_id_cap_pair_count_lte_corresponding_base_pair_count_for_each_cap",
        "base_ALL_cap_pair_count_equals_base_view_unique_pairs",
        "positive_ALL_cap_pair_count_equals_positive_view_unique_pairs",
        "base_view_grid_sha256_equals_domain_hash_of_visible_base_view_grid_payload",
        "positive_view_grid_sha256_equals_domain_hash_of_visible_positive_view_grid_payload",
    }
    for name in (
        "runtime_fingerprint",
        "row_counts",
        "scope",
        "source_accounting",
        "work_and_slot_accounting",
        "author_id_accounting",
        "pair_accounting",
        "temporal_accounting",
        "history_accounting",
        "team_and_hyper_sensitivity",
        "orcid_name_and_position_diagnostics",
        "gate_status",
    ):
        assert set(nested[name]["exact_keyset"]) == set(
            nested[name]["exact_field_types"]
        )
    assert nested["gate_status"]["exact_values"] == {
        "synthetic_contract": (
            "PASS_SYNTHETIC_WORK_SIDE_AUTHOR_TOKEN_HISTORY_CONTRACT_ONLY"
        ),
        "production_status": "PRODUCTION_NOT_RUN",
        "production_parity_established": False,
        "author_entity_validity": "UNRESOLVED_WORK_SIDE_ONLY",
        "token_and_field_past_observability": (
            "UNVERIFIED_CURRENT_SNAPSHOT_RETROSPECTIVE_PROXY"
        ),
        "g_minus_1_data_readiness": "PENDING",
        "g0_measurement_freeze": "PENDING",
        "empirical_g1_signature": "NOT_RUN",
        "pass_g1": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }

    statuses = contract["output_status_and_phase_literals"]
    assert statuses["terminal_success_state_status"] == "COMPLETE_PENDING_RECEIPT"
    assert statuses["terminal_failure_state_status"] == (
        "FINALIZATION_FAILED_NO_VALID_RECEIPT"
    )
    assert contract["execution_abi_literal"].endswith("synthetic_execution.v1")
    assert contract["canonical_json_abi_literal"].endswith("canonical_json.v1")

    hashes = contract["canonical_hash_contract"]
    assert hashes["digest"] == "sha256"
    assert hashes["canonical_json_encoding"] == "utf8"
    assert hashes["canonical_json_ensure_ascii"] is False
    assert hashes["canonical_json_allow_nan"] is False
    assert hashes["floating_point_values_allowed_in_hashed_payloads"] is False
    assert hashes["domain_separator_encoding"] == (
        "utf8_domain_then_single_nul_byte_then_canonical_json_bytes"
    )
    assert (
        hashes["task_fingerprint"]["exact_payload_keyset"]
        == (contract["task_fingerprint_exact_payload"])
    )
    assert (
        hashes["result_fingerprint"]["exact_payload_keyset"]
        == (contract["result_fingerprint_direct_payload"])
    )
    assert (
        len(
            {
                hashes["task_fingerprint"]["domain"],
                hashes["metrics_sha256"]["domain"],
                hashes["checkpoint_sha256"]["domain"],
                hashes["result_fingerprint"]["domain"],
            }
        )
        == 4
    )
    aggregates = contract["aggregate_record_contracts"]
    pair_record = aggregates["deduplicated_author_work_diagnostic_pair"]
    assert set(pair_record["exact_keyset"]) == set(pair_record["exact_field_types"])
    assert pair_record["record_sort_key"] == [
        "author_numeric_id",
        "work_numeric_id",
    ]
    assert pair_record["author_position_category_set_sort_comparator"] == (
        "declared_category_order"
    )
    assert pair_record["author_position_category_set_sort_order"] == [
        "first",
        "middle",
        "last",
        "null",
        "unexpected_non_null",
    ]
    assert len(aggregates["conflict_flags"]["exact_keyset"]) == 9
    aggregate_digests = hashes["aggregate_digests"]
    assert set(aggregate_digests) == {
        "primary_pair_diagnostics",
        "history_usable_pair_diagnostics",
        "base_view_per_year",
        "positive_view_per_year",
        "base_view_per_token",
        "positive_view_per_token",
        "base_view_team_hyper_grid",
        "positive_view_team_hyper_grid",
    }
    assert len({entry["domain"] for entry in aggregate_digests.values()}) == 8


def test_synthetic_parent_payload_and_identifier_value_contracts_are_exact() -> None:
    config = _load_config()
    contract = config["synthetic_execution_contract"]
    payloads = contract["fixture_synthetic_parent_payloads"]
    assert set(payloads) == {
        "current_content_baseline",
        "main_temporal_validity",
        "canonical_work_time_spine",
    }
    assert payloads["current_content_baseline"]["allowed_status"] == (
        "SYNTHETIC_CONTENT_BASELINE_PARENT_ONLY"
    )
    assert payloads["main_temporal_validity"]["allowed_status"] == (
        "SYNTHETIC_MAIN_TEMPORAL_PARENT_ONLY"
    )
    assert payloads["canonical_work_time_spine"]["allowed_status"] == (
        "SYNTHETIC_CANONICAL_SPINE_PARENT_ONLY"
    )
    for name, payload in payloads.items():
        assert set(payload["exact_keyset"]) == set(payload["exact_field_types"])
        for field, field_type in payload["exact_field_types"].items():
            if field.endswith("sha256"):
                assert field_type == "lowercase_64_hex_string"
        assert payload["allowed_status"].startswith("SYNTHETIC_")
        assert name in contract["fixture_synthetic_parent_keyset"]
    assert all(contract["fixture_parent_cross_binding"].values())
    assert contract["production_parent_status_literals_in_fixture_allowed"] is False
    assert contract["fixture_scalar_validation"] == {
        "json_boolean_never_accepted_as_integer": True,
        "integer_ranges_are_checked_before_cast": True,
        "all_strings_must_be_valid_utf8_without_lone_surrogates": True,
        "snapshot_date_must_equal_fixture_snapshot_date_exact": True,
        "digest_pattern": "^[0-9a-f]{64}$",
        "parent_rows_type": "non_boolean_integer",
        "parent_rows_minimum": 0,
        "parent_rows_maximum": 10000,
    }

    validation = config["identifier_and_value_validation"]
    assert validation["json_boolean_accepted_as_integer"] is False
    assert validation["w_minus_1_allowed_in_spine"] is False
    assert validation["w_zero_policy"] == (
        "retain_as_numeric_zero_with_entity_validity_unresolved"
    )
    assert validation["a_zero_policy"] == (
        "retain_in_base_view_exclude_in_positive_id_view"
    )
    assert validation["authors_count_minimum_inclusive"] == 0
    assert validation["authors_count_maximum_inclusive"] == 4294967295
    assert validation["primary_field_entity_or_past_observability_established"] is (
        False
    )

    coverage = config["coverage"]
    assert coverage["exact_count_units"] == [
        "distinct_work_count",
        "raw_authorship_slot_count",
        "grammar_valid_author_slot_count",
        "distinct_author_work_pair_count",
    ]


def test_state_receipt_quarantine_and_markdown_abis_are_exact() -> None:
    contract = _load_config()["synthetic_execution_contract"]
    state = contract["output_json_schemas"]["run_state"]
    assert state["completed_row_counts_schema_ref"] == (
        "output_nested_object_schemas.row_counts"
    )
    transitions = contract["run_state_transition_contract"]
    tuples = transitions["allowed_status_phase_error_result_tuples"]
    assert {entry["status"] for entry in tuples} == {
        "PLANNED_SYNTHETIC_AUTHOR_HISTORY",
        "RUNNING_SYNTHETIC_AUTHOR_HISTORY",
        "COMPLETE_PENDING_RECEIPT",
        "FINALIZATION_FAILED_NO_VALID_RECEIPT",
    }
    success = next(
        entry for entry in tuples if entry["status"] == "COMPLETE_PENDING_RECEIPT"
    )
    failure = next(
        entry
        for entry in tuples
        if entry["status"] == "FINALIZATION_FAILED_NO_VALID_RECEIPT"
    )
    assert success["error_redacted"] == "null"
    assert success["result_fingerprint_sha256"] == "lowercase_64_hex_string"
    assert failure["error_redacted"] == "nonempty_redacted_string"
    assert failure["result_fingerprint_sha256"] == "null"
    assert transitions["allowed_transitions"][-1] == (
        "FINALIZATION_FAILED_NO_VALID_RECEIPT_to_"
        "RUNNING_SYNTHETIC_AUTHOR_HISTORY_only_after_successful_resume_revalidation"
    )
    assert set(transitions["forbidden_transitions"]) == {
        "FINALIZATION_FAILED_NO_VALID_RECEIPT_to_"
        "RUNNING_SYNTHETIC_AUTHOR_HISTORY_on_failed_resume_revalidation",
        "FINALIZATION_FAILED_NO_VALID_RECEIPT_to_COMPLETE_PENDING_RECEIPT_direct",
        "any_terminal_success_to_running_after_valid_receipt",
    }

    resume = contract["resume_contract"]
    assert resume[
        "failed_state_resume_requires_successful_full_revalidation_before_running"
    ]
    assert resume["successful_failed_state_resume_next_status"] == (
        "RUNNING_SYNTHETIC_AUTHOR_HISTORY"
    )
    assert resume["failed_resume_revalidation_status_remains"] == (
        "FINALIZATION_FAILED_NO_VALID_RECEIPT"
    )
    assert resume["failed_resume_revalidation_receipt_must_be_absent"]
    assert resume["direct_failed_state_to_complete_allowed"] is False
    assert resume[
        "same_task_clean_and_successfully_resumed_result_fingerprint_must_equal"
    ]

    assert contract["receipt_artifact_record_exact_field_types"] == {
        "relative_path": "exact_declared_posix_relative_path_string",
        "bytes": "non_boolean_nonnegative_integer",
        "sha256": "lowercase_64_hex_string",
    }
    receipt_closure = contract["receipt_artifact_record_closure"]
    assert all(
        receipt_closure[key]
        for key in receipt_closure
        if key != "extra_artifact_mapping_keys_allowed"
    )
    assert receipt_closure["extra_artifact_mapping_keys_allowed"] is False
    nested = contract["output_nested_object_schemas"]
    assert nested["report_parents_schema_ref"] == (
        "output_nested_object_schemas.parent_commitments"
    )
    assert nested["report_parents_must_equal_plan_parent_commitments"] is True

    work = nested["work_and_slot_accounting"]
    assert {
        "main_w_minus_1_rows",
        "authorship_w_minus_1_rows",
        "total_w_minus_1_rows",
    } <= set(work["exact_keyset"])
    assert work["closure"][0] == (
        "total_w_minus_1_rows_equals_main_plus_authorship_w_minus_1_rows"
    )
    markdown = contract["markdown_render_contract"]
    assert markdown["authority"] == (
        "NONAUTHORITATIVE_DERIVED_PRESENTATION_JSON_REPORT_IS_SEMANTIC_AUTHORITY"
    )
    assert markdown["renderer_abi"].endswith("synthetic_markdown.v1")


def test_cross_artifact_semantic_bindings_and_numeric_closures_are_exact() -> None:
    config = _load_config()
    contract = config["synthetic_execution_contract"]
    nested = contract["output_nested_object_schemas"]
    binding = contract["cross_artifact_semantic_binding_contract"]

    assert (
        binding["checkpoint_metrics_report_exact_object_equality_sections"]
        == nested["checkpoint_metrics_exact_keyset"]
    )
    assert binding["completed_row_count_projection_exact_keyset"] == [
        "main_rows",
        "spine_rows",
        "authorship_rows",
    ]
    assert binding["task_fingerprint_exact_equality_targets"] == [
        "checkpoints/synthetic_accounting.json.task_fingerprint_sha256",
        "run_state.json.task_fingerprint_sha256",
        "author_history_audit.json.task_fingerprint_sha256",
        "author_history_audit.outputs.json.task_fingerprint_sha256",
    ]
    assert binding["successful_completion_row_count_projection_targets"] == {
        "run_state.json.completed_row_counts": [
            "main_rows",
            "spine_rows",
            "authorship_rows",
        ],
        "checkpoints/synthetic_accounting.json.metrics.source_accounting": [
            "main_rows",
            "spine_rows",
            "authorship_rows",
        ],
        "author_history_audit.json.scope": [
            "main_rows",
            "spine_rows",
            "authorship_rows",
        ],
    }
    assert binding["result_fingerprint_exact_source_paths"] == {
        "task_fingerprint_sha256": "audit_plan.json.task_fingerprint_sha256",
        "metrics_sha256": ("checkpoints/synthetic_accounting.json.metrics_sha256"),
        "parent_commitments": "audit_plan.json.parent_commitments",
        "row_counts": "audit_plan.json.row_counts",
        "gate_status": ("checkpoints/synthetic_accounting.json.metrics.gate_status"),
        "claim_boundary": "repository_config#/claim_boundary",
    }
    assert binding["result_fingerprint_exact_equality_targets"] == [
        "run_state.json.result_fingerprint_sha256",
        "author_history_audit.json.result_fingerprint_sha256",
        "author_history_audit.outputs.json.result_fingerprint_sha256",
    ]
    assert binding["status_and_literal_cross_bindings"] == {
        "report_status_equals_checkpoint_gate_synthetic_contract": True,
        "receipt_status_equals_report_status": True,
        "report_production_status_equals_checkpoint_gate_production_status": True,
        "report_surface_kind_equals_synthetic_execution_surface_kind": True,
        "report_operator_paths_redacted_is_true": True,
        "receipt_completion_contract_equals_declared_literal": True,
    }
    assert binding["config_pointer_namespace"] == {
        "repository_config": (
            "experiments/configs/openalex_g1_work_side_author_history_v0.yaml"
        ),
        "pointer_syntax": "RFC6901",
    }
    assert binding["claim_boundary_authoritative_source"] == (
        "repository_config#/claim_boundary"
    )
    assert (
        binding["future_runtime_mutation_tests_implemented_in_this_design_tranche"]
        is False
    )
    assert binding[
        "required_future_semantic_mutation_tests_recompute_local_self_hashes_and_receipt_artifact_hashes_before_expect_rejection"
    ]
    assert binding["pre_complete_pending_receipt_validation"][
        "excludes_not_yet_published_terminal_state_and_receipt_targets"
    ]
    assert binding["post_complete_pre_receipt_validation"][
        "includes_terminal_state_task_completed_rows_and_result_targets"
    ]
    assert binding["post_complete_pre_receipt_validation"][
        "excludes_not_yet_published_receipt_targets"
    ]
    assert binding["prospective_receipt_validation"] == {
        "validates_in_memory_task_result_status_completion_contract_and_exact_artifact_map": True,
        "rehashes_all_five_published_pre_receipt_artifacts": True,
        "requires_post_complete_pre_receipt_validation_success": True,
    }
    assert binding[
        "receipt_written_atomically_only_after_prospective_receipt_validation"
    ]
    assert binding[
        "final_disk_validation_after_receipt_write_rechecks_every_semantic_binding"
    ]
    assert binding["cross_binding_failure_routing"] == {
        "validation_order": [
            "repository_config_plan_task_parents_and_checkpoint_self_hashes_first",
            "dependent_report_state_receipt_and_artifact_bindings_second",
        ],
        "invalid_authoritative_config_plan_task_parent_or_checkpoint_route": (
            "pre_valid_checkpoint_failure_contract"
        ),
        "valid_authoritative_sources_but_invalid_dependent_report_state_or_receipt_route": (
            "failure_invalidation_order"
        ),
        "resumable_failed_state_may_be_written_only_when_authoritative_plan_and_checkpoint_remain_valid": True,
        "mutation_tests_must_cover_each_edge_endpoint_and_expected_route": True,
    }

    mutation_edges = set(binding["required_future_mutation_failure_test_edges"])
    assert mutation_edges == {
        "report_source_accounting_vs_checkpoint_metrics",
        "report_work_and_slot_accounting_vs_checkpoint_metrics",
        "report_author_id_accounting_vs_checkpoint_metrics",
        "report_pair_accounting_vs_checkpoint_metrics",
        "report_temporal_accounting_vs_checkpoint_metrics",
        "report_history_accounting_vs_checkpoint_metrics",
        "report_team_and_hyper_sensitivity_vs_checkpoint_metrics",
        "report_orcid_name_and_position_diagnostics_vs_checkpoint_metrics",
        "report_gate_status_vs_checkpoint_metrics",
        "plan_task_vs_checkpoint_task",
        "plan_task_vs_run_state_task",
        "plan_task_vs_report_task",
        "plan_task_vs_receipt_task",
        "plan_parents_vs_report_parents",
        "plan_rows_vs_checkpoint_source_projection",
        "plan_rows_vs_terminal_run_state_completed_rows",
        "plan_rows_vs_report_scope_projection",
        "recomputed_task_hash",
        "recomputed_metrics_hash",
        "recomputed_checkpoint_hash",
        "recomputed_result_hash",
        "result_vs_terminal_run_state",
        "result_vs_report",
        "result_vs_receipt",
        "report_claim_boundary_vs_config_claim_boundary",
        "report_status_vs_checkpoint_gate_status",
        "receipt_status_vs_report_status",
        "report_production_status_vs_checkpoint_gate_status",
        "each_of_five_receipt_artifact_byte_and_hash_records",
    }

    equations = set(
        contract["output_numeric_cross_section_closure_contract"]["exact_equations"]
    )
    assert {
        "source_main_rows_equals_work_retained_work_rows_plus_main_w_minus_1_rows",
        "source_spine_rows_equals_source_matched_main_spine_rows",
        "source_matched_main_spine_rows_equals_work_retained_work_rows",
        "source_authorship_rows_equals_source_matched_authorship_rows_plus_work_authorship_w_minus_1_rows",
        "temporal_history_usable_pairs_equals_history_base_view_unique_pairs",
        "history_base_view_tokens_lte_author_distinct_base_tokens",
        "history_positive_view_tokens_lte_author_distinct_positive_tokens",
        "diagnostics_primary_population_pairs_equals_pair_distinct_author_work_pairs",
        "diagnostics_history_usable_population_pairs_equals_temporal_history_usable_pairs",
        "team_base_ALL_pairs_equals_history_base_view_unique_pairs",
        "team_positive_ALL_pairs_equals_history_positive_view_unique_pairs",
    } <= equations
    numeric_closure = contract["output_numeric_cross_section_closure_contract"]
    assert numeric_closure["construction_time_hidden_record_closures"] == [
        "history_base_view_tokens_equals_base_per_token_aggregate_record_count",
        "history_positive_view_tokens_equals_positive_per_token_aggregate_record_count",
        "base_view_per_year_author_keyset_equals_base_view_per_token_author_keyset",
        "positive_view_per_year_author_keyset_equals_positive_view_per_token_author_keyset",
        "each_view_per_token_record_exactly_reduces_same_author_per_year_records",
        "positive_view_per_year_records_equal_base_view_per_year_records_filtered_by_author_numeric_id_gt_0",
        "positive_view_per_token_records_equal_base_view_per_token_records_filtered_by_author_numeric_id_gt_0",
        "each_positive_view_hidden_capped_record_array_equals_corresponding_base_view_hidden_capped_record_array_filtered_by_author_numeric_id_gt_0",
        "each_view_cap_pair_count_and_threshold_cell_exactly_reduces_corresponding_same_view_same_cap_hidden_capped_records",
    ]
    hidden_cross_record = numeric_closure[
        "construction_time_hidden_cross_record_reduction_contract"
    ]
    assert hidden_cross_record["scope"] == (
        "CONSTRUCTION_TIME_ONLY_BEFORE_AGGREGATE_DIGEST_AND_METRICS_HASH"
    )
    assert hidden_cross_record["views"] == ["base", "positive_author_id"]
    assert hidden_cross_record["author_key_field"] == "author_numeric_id"
    assert hidden_cross_record["per_view_hidden_record_key_uniqueness"] == {
        "per_year": ["author_numeric_id", "publication_year"],
        "per_token": ["author_numeric_id"],
    }
    assert hidden_cross_record["per_view_author_keyset_equality"] == {
        "base": (
            "base_view_per_year_author_keyset_equals_base_view_per_token_author_keyset"
        ),
        "positive_author_id": (
            "positive_view_per_year_author_keyset_equals_"
            "positive_view_per_token_author_keyset"
        ),
    }
    assert hidden_cross_record[
        "per_author_token_from_same_author_per_year_exact_reductions"
    ] == {
        "first_usable_publication_year": "minimum_publication_year",
        "last_usable_publication_year": "maximum_publication_year",
        "active_year_count": (
            "count_of_distinct_same_author_publication_years_equals_"
            "count_of_same_author_per_year_records"
        ),
        "inclusive_span_years": "maximum_year_minus_minimum_year_plus_one",
        "inactive_gap_years": "inclusive_span_years_minus_active_year_count",
        "total_unique_work_count": ("sum_of_same_author_per_year_unique_work_count"),
        "maximum_annual_unique_work_count": (
            "maximum_of_same_author_per_year_unique_work_count"
        ),
    }
    positive_projection = hidden_cross_record["positive_author_id_exact_projection"]
    assert positive_projection["predicate"] == "author_numeric_id > 0"
    assert positive_projection["preserves_declared_record_sort_order"]
    assert positive_projection["exact_array_equalities"] == [
        "positive_view_per_year_equals_base_view_per_year_filtered_by_predicate",
        "positive_view_per_token_equals_base_view_per_token_filtered_by_predicate",
        "for_each_cap_positive_hidden_capped_pair_records_equal_base_hidden_capped_pair_records_filtered_by_predicate",
        "for_each_cap_positive_hidden_capped_per_year_records_equal_base_hidden_capped_per_year_records_filtered_by_predicate",
        "for_each_cap_positive_hidden_capped_per_token_records_equal_base_hidden_capped_per_token_records_filtered_by_predicate",
    ]
    cap_reduction = hidden_cross_record[
        "team_cap_grid_from_corresponding_hidden_capped_records"
    ]
    assert cap_reduction == {
        "caps": [10, 50, 100, "ALL"],
        "thresholds": [50, 100, 250, 500, 1000],
        "numeric_cap_pair_filter": "declared_authors_count <= cap",
        "all_cap_pair_filter": (
            "include_every_same_view_history_usable_unique_author_work_pair"
        ),
        "hidden_capped_pair_record_key": ["author_numeric_id", "work_numeric_id"],
        "hidden_capped_pair_record_key_unique": True,
        "cap_pair_count_reduction": "exact_hidden_capped_pair_record_count",
        "hidden_capped_per_year_reduction": (
            "group_same_cap_pairs_by_author_numeric_id_and_publication_year_"
            "then_count_unique_work_numeric_id"
        ),
        "hidden_capped_per_token_reduction": (
            "apply_same_author_per_year_exact_reductions_within_same_cap"
        ),
        "threshold_cell_reduction": (
            "count_distinct_same_cap_author_numeric_id_with_maximum_annual_"
            "unique_work_count_strictly_greater_than_threshold"
        ),
        "no_cross_view_or_cross_cap_source_substitution_allowed": True,
    }
    assert hidden_cross_record["checked_while_hidden_arrays_are_in_memory"]
    assert hidden_cross_record["terminal_reentry_reopens_hidden_arrays"] is False
    assert (
        hidden_cross_record[
            "terminal_reentry_independent_cross_record_reduction_claim_allowed"
        ]
        is False
    )
    assert numeric_closure[
        "hidden_record_closures_checked_before_aggregate_digest_and_metrics_hash"
    ]
    assert numeric_closure[
        "hidden_record_arrays_not_published_or_reopened_on_terminal_reentry"
    ]
    assert numeric_closure[
        "terminal_reentry_does_not_claim_independent_hidden_record_cardinality_recomputation"
    ]

    order_contract = contract["processing_order_and_fixture_byte_identity_contract"]
    assert order_contract[
        "same_exact_fixture_bytes_internal_traversal_order_chunking_and_successful_resume_result_fingerprint_equal"
    ]
    assert order_contract[
        "byte_permuted_fixture_with_same_logical_rows_aggregate_objects_and_metrics_sha256_equal"
    ]
    assert order_contract[
        "byte_permuted_fixture_fixture_sha256_task_fingerprint_and_result_fingerprint_must_differ"
    ]

    schemas = contract["output_json_schemas"]
    artifact_schemas = {
        "audit_plan.json": set(schemas["audit_plan"]["exact_keyset"]),
        "run_state.json": set(schemas["run_state"]["exact_keyset"]),
        "checkpoints/synthetic_accounting.json": set(
            schemas["synthetic_accounting_checkpoint"]["exact_keyset"]
        ),
        "author_history_audit.json": set(
            schemas["author_history_report"]["exact_keyset"]
        ),
        "author_history_audit.outputs.json": set(contract["exact_receipt_keyset"]),
    }

    def assert_declared_path_exists(path: str) -> None:
        if path.startswith("repository_config#/"):
            pointer = path.partition("#/")[2]
            assert "/" not in pointer
            assert pointer in config
            return
        root = next(
            (
                candidate
                for candidate in sorted(artifact_schemas, key=len, reverse=True)
                if path == candidate or path.startswith(f"{candidate}.")
            ),
            None,
        )
        assert root is not None, path
        suffix = path[len(root) :].removeprefix(".")
        if not suffix:
            return
        parts = suffix.split(".")
        assert parts[0] in artifact_schemas[root], path
        if root == "checkpoints/synthetic_accounting.json" and parts[0] == "metrics":
            if len(parts) > 1:
                assert parts[1] in nested["checkpoint_metrics_exact_keyset"], path
        else:
            assert len(parts) == 1, path

    declared_paths = {
        binding["checkpoint_metrics_source"],
        binding["report_metrics_target"],
        binding["task_fingerprint_authoritative_source"],
        *binding["task_fingerprint_exact_equality_targets"],
        binding["parent_commitments_authoritative_source"],
        binding["parent_commitments_exact_equality_target"],
        binding["completed_row_counts_authoritative_source"],
        *binding["successful_completion_row_count_projection_targets"],
        binding["claim_boundary_authoritative_source"],
        *binding["result_fingerprint_exact_source_paths"].values(),
        *binding["result_fingerprint_exact_equality_targets"],
    }
    for path in declared_paths:
        assert_declared_path_exists(path)


def test_declared_aggregate_domains_bind_sensitive_values_and_ignore_order() -> None:
    contract = _load_config()["synthetic_execution_contract"]
    hashes = contract["canonical_hash_contract"]
    domains = hashes["aggregate_digests"]
    pair_domain = domains["primary_pair_diagnostics"]["domain"]
    aggregates = contract["aggregate_record_contracts"]
    pair_record = aggregates["deduplicated_author_work_diagnostic_pair"]

    flags = {
        name: False
        for name in contract["aggregate_record_contracts"]["conflict_flags"][
            "exact_keyset"
        ]
    }
    pair = {
        "author_numeric_id": 1,
        "work_numeric_id": 7,
        "raw_multiplicity": 1,
        "temporal_disposition": 2,
        "history_usable": True,
        "exact_raw_orcid_set": [{"kind": "null"}],
        "canonical_valid_orcid_set": [],
        "exact_raw_author_name_set": [{"kind": "string", "value": "Alpha"}],
        "exact_display_name_set": [{"kind": "string", "value": "Alpha"}],
        "normalized_raw_author_name_set": [{"kind": "string", "value": "alpha"}],
        "normalized_display_name_set": [{"kind": "string", "value": "alpha"}],
        "author_position_category_set": ["first"],
        "exact_raw_position_set": [{"kind": "string", "value": "first"}],
        "conflict_flags": flags,
    }
    second = copy.deepcopy(pair)
    second["author_numeric_id"] = 0
    second["work_numeric_id"] = 8

    def pair_digest(records: list[dict[str, Any]]) -> str:
        ordered = sorted(
            records,
            key=lambda row: (row["author_numeric_id"], row["work_numeric_id"]),
        )
        return _domain_hash(pair_domain, ordered)

    baseline = pair_digest([pair, second])
    assert baseline == pair_digest([second, pair])
    chunks = [[second], [pair]]
    assert baseline == pair_digest([row for chunk in chunks for row in chunk])

    variants: list[dict[str, Any]] = []
    multiplicity = copy.deepcopy(pair)
    multiplicity["raw_multiplicity"] = 2
    variants.append(multiplicity)
    tagged_orcid = copy.deepcopy(pair)
    tagged_orcid["exact_raw_orcid_set"] = [
        {"kind": "null"},
        {"kind": "string", "value": ""},
    ]
    variants.append(tagged_orcid)
    tagged_name = copy.deepcopy(pair)
    tagged_name["exact_raw_author_name_set"].append(
        {"kind": "string", "value": "ALPHA"}
    )
    variants.append(tagged_name)
    position = copy.deepcopy(pair)
    position["exact_raw_position_set"].append({"kind": "string", "value": " First "})
    position["conflict_flags"]["raw_position_conflict"] = True
    position["conflict_flags"]["position_conflict"] = True
    variants.append(position)
    variant_digests = {pair_digest([variant, second]) for variant in variants}
    assert len(variant_digests) == len(variants)
    assert baseline not in variant_digests
    assert pair_digest([pair]) != baseline  # A0 inclusion/exclusion is bound.

    category_order = {
        value: index
        for index, value in enumerate(
            pair_record["author_position_category_set_sort_order"]
        )
    }
    category_permutation_a = copy.deepcopy(pair)
    category_permutation_b = copy.deepcopy(pair)
    category_permutation_a["author_position_category_set"] = [
        "last",
        "first",
        "middle",
    ]
    category_permutation_b["author_position_category_set"] = [
        "middle",
        "last",
        "first",
    ]
    for candidate in (category_permutation_a, category_permutation_b):
        candidate["author_position_category_set"] = sorted(
            set(candidate["author_position_category_set"]),
            key=category_order.__getitem__,
        )
        candidate["conflict_flags"]["position_category_conflict"] = True
        candidate["conflict_flags"]["position_conflict"] = True
    assert pair_digest([category_permutation_a]) == pair_digest(
        [category_permutation_b]
    )

    year_schema = aggregates["per_author_year_aggregate"]
    team_counts = {key: 0 for key in year_schema["team_band_count_exact_keyset"]}
    field_counts = {key: 0 for key in year_schema["primary_field_count_exact_keyset"]}
    team_counts["1"] = 1
    field_counts["1"] = 1
    year_record = {
        "author_numeric_id": 1,
        "publication_year": 2020,
        "unique_work_count": 1,
        "team_band_unique_work_counts": team_counts,
        "primary_field_unique_work_counts": field_counts,
    }
    year_domain = domains["base_view_per_year"]["domain"]
    year_digest = _domain_hash(year_domain, [year_record])
    team_variant = copy.deepcopy(year_record)
    team_variant["team_band_unique_work_counts"]["1"] = 0
    team_variant["team_band_unique_work_counts"]["2"] = 1
    field_variant = copy.deepcopy(year_record)
    field_variant["primary_field_unique_work_counts"]["1"] = 0
    field_variant["primary_field_unique_work_counts"]["2"] = 1
    assert _domain_hash(year_domain, [team_variant]) != year_digest
    assert _domain_hash(year_domain, [field_variant]) != year_digest

    token_record = {
        "author_numeric_id": 1,
        "first_usable_publication_year": 2020,
        "last_usable_publication_year": 2021,
        "active_year_count": 2,
        "inclusive_span_years": 2,
        "inactive_gap_years": 0,
        "total_unique_work_count": 2,
        "maximum_annual_unique_work_count": 1,
    }
    token_domain = domains["base_view_per_token"]["domain"]
    token_digest = _domain_hash(token_domain, [token_record])
    span_variant = copy.deepcopy(token_record)
    span_variant["last_usable_publication_year"] = 2022
    span_variant["inclusive_span_years"] = 3
    span_variant["inactive_gap_years"] = 1
    count_variant = copy.deepcopy(token_record)
    count_variant["total_unique_work_count"] = 3
    count_variant["maximum_annual_unique_work_count"] = 2
    assert _domain_hash(token_domain, [span_variant]) != token_digest
    assert _domain_hash(token_domain, [count_variant]) != token_digest

    same_payload = {"domain_separation_canary": 1}
    separated = {
        _domain_hash(entry["domain"], same_payload) for entry in domains.values()
    }
    assert len(separated) == len(domains)

    grid_contract = contract["output_nested_object_schemas"]["team_cap_threshold_grid"]
    grid = {key: 0 for key in grid_contract["exact_keyset"]}
    grid_payload = {
        "view": "base",
        "cap_pair_counts": {"10": 0, "50": 0, "100": 0, "ALL": 0},
        "cap_threshold_token_grid": grid,
    }
    grid_domain = domains["base_view_team_hyper_grid"]["domain"]
    grid_digest = _domain_hash(grid_domain, grid_payload)
    threshold_variant = copy.deepcopy(grid_payload)
    threshold_variant["cap_threshold_token_grid"]["cap_10_gt_50"] = 1
    assert _domain_hash(grid_domain, threshold_variant) != grid_digest

    metrics_domain = hashes["metrics_sha256"]["domain"]
    result_domain = hashes["result_fingerprint"]["domain"]

    def result_for(pair_sha256: str) -> str:
        metrics_sha256 = _domain_hash(
            metrics_domain,
            {
                "primary_population_aggregate_sha256": pair_sha256,
                "base_view_grid_sha256": grid_digest,
            },
        )
        return _domain_hash(
            result_domain,
            {
                "task_fingerprint_sha256": "0" * 64,
                "metrics_sha256": metrics_sha256,
                "parent_commitments": {},
                "row_counts": {},
                "gate_status": {},
                "claim_boundary": "synthetic-only",
            },
        )

    assert result_for(baseline) != result_for(next(iter(variant_digests)))


def test_fixture_byte_permutation_keeps_metrics_but_changes_task_and_result() -> None:
    contract = _load_config()["synthetic_execution_contract"]
    hashes = contract["canonical_hash_contract"]
    task_payload = {
        "execution_abi": contract["execution_abi_literal"],
        "config_sha256": "1" * 64,
        "tool_sha256": "2" * 64,
        "fixture_sha256": "3" * 64,
        "runtime_fingerprint": {"unicode": "frozen"},
        "parent_commitments": {"synthetic": "parent"},
        "row_counts": {"main_rows": 2, "spine_rows": 2, "authorship_rows": 2},
    }
    permuted_payload = copy.deepcopy(task_payload)
    permuted_payload["fixture_sha256"] = "4" * 64
    task_a = _domain_hash(hashes["task_fingerprint"]["domain"], task_payload)
    task_b = _domain_hash(hashes["task_fingerprint"]["domain"], permuted_payload)
    assert task_a != task_b

    metrics = {"logically_order_independent_counts": 2}
    metrics_a = _domain_hash(hashes["metrics_sha256"]["domain"], metrics)
    metrics_b = _domain_hash(hashes["metrics_sha256"]["domain"], copy.deepcopy(metrics))
    assert metrics_a == metrics_b

    def result(task: str, metrics_sha256: str) -> str:
        return _domain_hash(
            hashes["result_fingerprint"]["domain"],
            {
                "task_fingerprint_sha256": task,
                "metrics_sha256": metrics_sha256,
                "parent_commitments": task_payload["parent_commitments"],
                "row_counts": task_payload["row_counts"],
                "gate_status": {"synthetic_contract": "PASS"},
                "claim_boundary": "synthetic-only",
            },
        )

    assert result(task_a, metrics_a) == result(task_a, metrics_b)
    assert result(task_a, metrics_a) != result(task_b, metrics_b)


def test_hidden_same_author_reduction_rejects_swapped_author_summaries() -> None:
    year_records = [
        {"author_numeric_id": 1, "publication_year": 2020, "unique_work_count": 2},
        {"author_numeric_id": 1, "publication_year": 2022, "unique_work_count": 3},
        {"author_numeric_id": 2, "publication_year": 2021, "unique_work_count": 7},
    ]
    valid_tokens = [
        _reduce_year_records_for_author(year_records, author_numeric_id)
        for author_numeric_id in (1, 2)
    ]
    assert _same_author_year_token_reductions_are_exact(year_records, valid_tokens)

    swapped_tokens = copy.deepcopy(valid_tokens)
    first_summary = {
        key: value
        for key, value in swapped_tokens[0].items()
        if key != "author_numeric_id"
    }
    second_summary = {
        key: value
        for key, value in swapped_tokens[1].items()
        if key != "author_numeric_id"
    }
    swapped_tokens[0].update(second_summary)
    swapped_tokens[1].update(first_summary)

    assert {record["author_numeric_id"] for record in swapped_tokens} == {1, 2}
    assert sum(record["total_unique_work_count"] for record in swapped_tokens) == sum(
        record["total_unique_work_count"] for record in valid_tokens
    )
    assert not _same_author_year_token_reductions_are_exact(
        year_records, swapped_tokens
    )


def test_hidden_positive_arrays_are_exact_stable_base_filters() -> None:
    base_year_records = [
        {"author_numeric_id": 0, "publication_year": 2020, "unique_work_count": 1},
        {"author_numeric_id": 1, "publication_year": 2020, "unique_work_count": 2},
        {"author_numeric_id": 2, "publication_year": 2021, "unique_work_count": 3},
    ]
    base_token_records = [
        _reduce_year_records_for_author(base_year_records, author_numeric_id)
        for author_numeric_id in (0, 1, 2)
    ]
    base_capped_pair_records = [
        {"author_numeric_id": 0, "publication_year": 2020, "work_numeric_id": 1},
        {"author_numeric_id": 1, "publication_year": 2020, "work_numeric_id": 2},
        {"author_numeric_id": 2, "publication_year": 2021, "work_numeric_id": 3},
    ]

    for base_records in (
        base_year_records,
        base_token_records,
        base_capped_pair_records,
    ):
        positive_records = [
            copy.deepcopy(record)
            for record in base_records
            if record["author_numeric_id"] > 0
        ]
        assert _is_exact_positive_author_projection(base_records, positive_records)

        mutated_projection = copy.deepcopy(positive_records)
        mutated_projection.reverse()
        assert not _is_exact_positive_author_projection(
            base_records, mutated_projection
        )


def test_hidden_cap_grid_must_exactly_reduce_corresponding_capped_records() -> None:
    config = _load_config()
    hidden_contract = config["synthetic_execution_contract"][
        "output_numeric_cross_section_closure_contract"
    ]["construction_time_hidden_cross_record_reduction_contract"]
    grid_contract = hidden_contract[
        "team_cap_grid_from_corresponding_hidden_capped_records"
    ]
    caps = grid_contract["caps"]
    thresholds = grid_contract["thresholds"]

    history_records = [
        {
            "author_numeric_id": author_numeric_id,
            "publication_year": 2020,
            "work_numeric_id": work_numeric_id,
            "declared_authors_count": declared_authors_count,
        }
        for author_numeric_id, declared_authors_count, first_work, count in (
            (1, 10, 1, 51),
            (2, 11, 1001, 101),
            (3, 51, 2001, 251),
            (4, 101, 3001, 501),
        )
        for work_numeric_id in range(first_work, first_work + count)
    ]
    capped_records = {
        str(cap): [
            record
            for record in history_records
            if cap == "ALL" or record["declared_authors_count"] <= cap
        ]
        for cap in caps
    }
    expected = _reduce_hidden_cap_records_to_grid(
        view="base",
        caps=caps,
        thresholds=thresholds,
        capped_records=capped_records,
    )
    assert expected["cap_pair_counts"]["10"] == 51
    assert expected["cap_threshold_token_grid"]["cap_10_gt_50"] == 1
    assert expected["cap_threshold_token_grid"]["cap_50_gt_100"] == 1
    assert expected["cap_threshold_token_grid"]["cap_100_gt_250"] == 1
    assert expected["cap_threshold_token_grid"]["cap_ALL_gt_500"] == 1

    mutated_cell = copy.deepcopy(expected)
    mutated_cell["cap_threshold_token_grid"]["cap_10_gt_50"] += 1
    assert mutated_cell != _reduce_hidden_cap_records_to_grid(
        view="base",
        caps=caps,
        thresholds=thresholds,
        capped_records=capped_records,
    )

    mutated_pair_count = copy.deepcopy(expected)
    mutated_pair_count["cap_pair_counts"]["50"] += 1
    assert mutated_pair_count != _reduce_hidden_cap_records_to_grid(
        view="base",
        caps=caps,
        thresholds=thresholds,
        capped_records=capped_records,
    )


def test_grid_closures_cover_cap_threshold_view_and_all_pair_axes() -> None:
    config = _load_config()
    contract = config["synthetic_execution_contract"]
    grid_contract = contract["output_nested_object_schemas"]["team_cap_threshold_grid"]
    caps = grid_contract["caps"]
    thresholds = grid_contract["thresholds"]

    def grid(offset: int) -> dict[str, int]:
        return {
            f"cap_{cap}_{threshold}": offset + cap_index + (len(thresholds) - index)
            for cap_index, cap in enumerate(caps)
            for index, threshold in enumerate(thresholds)
        }

    base_pairs = {"10": 10, "50": 20, "100": 30, "ALL": 40}
    positive_pairs = {"10": 8, "50": 15, "100": 25, "ALL": 35}
    base_grid = grid(4)
    positive_grid = grid(0)

    def valid() -> bool:
        pair_cap = all(
            base_pairs[left] <= base_pairs[right]
            and positive_pairs[left] <= positive_pairs[right]
            for left, right in zip(caps, caps[1:])
        )
        pair_view = all(positive_pairs[cap] <= base_pairs[cap] for cap in caps)
        threshold_axis = all(
            values[index + 1] <= values[index]
            for cap in caps
            for values in (
                [base_grid[f"cap_{cap}_{threshold}"] for threshold in thresholds],
                [positive_grid[f"cap_{cap}_{threshold}"] for threshold in thresholds],
            )
            for index in range(len(values) - 1)
        )
        cap_axis = all(
            values[index] <= values[index + 1]
            for threshold in thresholds
            for values in (
                [base_grid[f"cap_{cap}_{threshold}"] for cap in caps],
                [positive_grid[f"cap_{cap}_{threshold}"] for cap in caps],
            )
            for index in range(len(values) - 1)
        )
        cell_view = all(positive_grid[key] <= base_grid[key] for key in base_grid)
        all_pairs = base_pairs["ALL"] == 40 and positive_pairs["ALL"] == 35
        return (
            pair_cap
            and pair_view
            and threshold_axis
            and cap_axis
            and cell_view
            and all_pairs
        )

    assert valid()
    base_grid["cap_10_gt_50"] = base_grid["cap_50_gt_50"] + 1
    assert not valid()
    base_grid = grid(4)
    base_grid["cap_10_gt_1000"] = base_grid["cap_10_gt_500"] + 1
    assert not valid()
    base_grid = grid(4)
    positive_pairs["10"] = base_pairs["10"] + 1
    assert not valid()
    positive_pairs["10"] = 8
    positive_grid["cap_10_gt_50"] = base_grid["cap_10_gt_50"] + 1
    assert not valid()

    sensitivity = config["sensitivity_policy"]
    assert sensitivity["required_team_cap_boundary_fixture_values"] == [
        10,
        11,
        50,
        51,
        100,
        101,
    ]
    assert sensitivity["required_hyper_threshold_boundary_fixture_values"] == [
        50,
        51,
        100,
        101,
        250,
        251,
        500,
        501,
        1000,
        1001,
    ]


def test_sensitivity_partition_resource_and_privacy_contracts_are_exact() -> None:
    config = _load_config()
    sensitivity = config["sensitivity_policy"]
    assert sensitivity["exact_team_cap_views"] == [10, 50, 100, "ALL"]
    assert sensitivity["hyper_prolific_thresholds_strictly_greater_than"] == [
        50,
        100,
        250,
        500,
        1000,
    ]
    assert sensitivity["diagnostic_only_not_invalid_entity_label"] is True
    assert sensitivity["threshold_driven_candidate_exclusion_allowed"] is False

    partitions = config["partition_plan"]
    assert partitions["sampling"] is False
    assert partitions["collision_possible"] is False
    assert partitions["work_partition_count"] == 4096
    assert partitions["author_partition_count"] == 4096
    assert partitions["work_partition_formula"] == "work_id_numeric % 4096"
    assert partitions["author_partition_formula"] == "author_id_numeric % 4096"
    assert partitions["work_stage_sort_key"] == [
        "work_id_numeric",
        "author_id_numeric",
    ]
    assert partitions["author_stage_sort_key"] == [
        "author_id_numeric",
        "work_id_numeric",
    ]
    assert partitions["hash_only_identity_or_deduplication_allowed"] is False
    assert len(partitions["stages"]) == 6
    assert "projected_payload_sha256" in partitions["checkpoint_binds"]
    assert "spool_segment_sha256" in partitions["checkpoint_binds"]

    resources = config["resource_envelope"]
    assert resources["declared_source_bytes"] == 253474946263
    assert resources["ram_gib_minimum_planning"] == 64
    assert resources["ram_gib_preferred_planning"] == 128
    assert resources["temporary_ssd_gib_minimum"] == 250
    assert resources["temporary_ssd_gib_preferred"] == 400
    assert resources["runtime_hours_lower_planning"] == 4
    assert resources["runtime_hours_upper_planning"] == 12
    assert resources["forbid_hdd_competition_with_content_freeze"] is True
    assert resources["gpu_used"] is False

    privacy = config["release_privacy"]
    assert privacy["public_summary_allows_aggregates_and_checksums_only"] is True
    assert privacy["detailed_identity_outputs_repository_allowed"] is False
    assert privacy["detailed_identity_outputs_release_allowed"] is False
    assert set(privacy["prohibited_public_fields"]) == {
        "raw_author_id",
        "raw_author_name",
        "author_display_name",
        "raw_orcid",
        "top_author_ranking",
    }
    assert privacy["release_inspector_must_reject_prohibited_outputs"] is True
    assert set(privacy["forbidden_operator_local_basenames"]) == {
        "coverage_cube.parquet",
        "author_year_history.parquet",
        "author_token_history.parquet",
        "team_hyper_sensitivity_grid.parquet",
        "orcid_cardinality_aggregates.parquet",
        "name_variation_aggregates.parquet",
    }
    assert set(privacy["forbidden_operator_local_directory_basenames"]) == {
        "author_history_staging",
        "author_history_checkpoints",
        "author_history_data",
    }
    assert privacy["researcher_pivot_sdist_boundary"] == (
        "exact_path_and_content_sha256_allowlist"
    )
    public = privacy["researcher_pivot_public_exact_sha256"]
    assert len(public) == 9
    for relative_path, expected_sha256 in public.items():
        assert relative_path.startswith(
            privacy["researcher_pivot_public_subtree_prefix"]
        )
        assert _sha256(REPO_ROOT / relative_path) == expected_sha256
    assert privacy["unmatched_researcher_pivot_subtree_member_policy"] == "reject"
    assert privacy["matched_path_content_sha256_mismatch_policy"] == "reject"
    assert set(privacy["required_release_canary_rejections"]) == {
        "coverage_cube.parquet.gz",
        "author_token_history.csv",
        "author_history_data.tar.zst",
        "renamed-author-records.parquet",
        "modified_bytes_at_an_allowlisted_path",
        "dotdot_path_alias",
        "backslash_path_alias",
        "duplicate_member_with_forbidden_payload",
        "symlink_or_hardlink_member",
        "unexpected_directory_member",
        "multiple_top_level_roots",
    }
    assert privacy["raw_tar_member_contract"] == {
        "exactly_one_top_level_sdist_root": True,
        "absolute_paths_allowed": False,
        "empty_dot_dotdot_or_repeated_separator_segments_allowed": False,
        "backslash_allowed": False,
        "nul_or_control_characters_allowed": False,
        "canonical_posix_spelling_required": True,
        "duplicate_raw_member_names_allowed": False,
        "duplicate_root_stripped_member_names_allowed": False,
        "non_regular_members_allowed": False,
    }


def test_every_non_full_mode_is_explicitly_unpromotable() -> None:
    rules = _load_config()["pilot_and_full_rules"]
    assert rules["max_files"]["promotion_allowed"] is False
    assert rules["author_hash_bucket"]["promotion_allowed"] is False
    assert rules["author_hash_bucket"]["global_coverage_claim_allowed"] is False
    assert rules["incomplete_author_delta_join"]["promotion_allowed"] is False
    assert rules["full"]["explicit_authorization_required"] is True
    assert rules["full"]["require_all_declared_files"] is True
    assert rules["full"]["require_all_parent_full_receipts"] is True
    assert rules["full"]["may_promote_only_component_scope"] is True


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        ("lifecycle.production_status", "PASS"),
        ("lifecycle.executable", True),
        ("identity_policy.token_not_person", False),
        ("identity_policy.natural_person_entity_claim_allowed", True),
        ("identity_policy.past_observable_token_linkage_claim_allowed", True),
        (
            "future_success_contract.required_full_gate_states.author_entity_validity",
            "PASS",
        ),
        (
            "future_success_contract.required_full_gate_states.candidate_eligibility",
            "PASS",
        ),
        ("incomplete_author_master.full_parent_role_allowed", True),
        (
            "sensitivity_policy.threshold_driven_candidate_exclusion_allowed",
            True,
        ),
        (
            "historical_observability_and_cutoff."
            "token_or_field_past_observability_established",
            True,
        ),
        ("release_privacy.detailed_identity_outputs_release_allowed", True),
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
        "Status: `DESIGN_ONLY`",
        "Production status: `PRODUCTION_NOT_RUN`",
        "38.76968115%",
        "author-ID token",
        "not a verified person",
        config["future_success_contract"]["full_component_status"],
        "author_entity_validity=UNRESOLVED_WORK_SIDE_ONLY",
        "candidate-researcher eligibility",
        "It uses no GPU",
        "history usable = {2, 3}",
        "work-topic relation is a parallel sibling",
        "No executable or result exists in this tranche",
        "NOT_IMPLEMENTED_REQUIRES_SEPARATE_REVIEW",
    )
    assert all(item in text for item in required_text)
    assert SCRIPT_PATH.exists() is False
    assert (
        config["production_output_contract"][
            "production_outputs_created_by_this_design"
        ]
        is False
    )
    result_dir = REPO_ROOT / "experiments/evidence/results"
    assert list(result_dir.glob(RESULT_GLOB)) == []
