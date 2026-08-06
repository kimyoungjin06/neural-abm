from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_openalex_footer_preflight_v1_20260806.summary.json"
)
SCRIPT = ROOT / "scripts/inspect_pivot_feedback_lean_h3_openalex_footer_preflight_v1.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_representative_footer_schema_and_manifest_counts_are_bound() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["status"] == (
        "PASS_REPRESENTATIVE_REMOTE_FOOTER_SCHEMA_STABLE_WORKS_SUFFICIENT_FOR_"
        "LEAN_H3_AUTHORS_DEFERRED_NO_ROW_ACCESS_NO_DOWNLOAD"
    )
    works = summary["schema_replay"]["works"]
    authors = summary["schema_replay"]["authors"]
    assert works["sample_count"] == authors["sample_count"] == 3
    assert works["captured_manifest_file_count"] == 2_446
    assert authors["captured_manifest_file_count"] == 1_961
    assert works["schema_sha256"] == (
        "2dcf41ac6cecb07d812d0d9348a6344481e5a093186c95857ed46ace11a30781"
    )
    assert authors["schema_sha256"] == (
        "f6a4c5ac1f90eaa6684a6b69b569963c36323c73a4718387b4841972ddf7d8fa"
    )
    for entity in (works, authors):
        assert entity["all_sample_schema_fingerprints_equal"] is True
        assert {item["selection"] for item in entity["samples"]} == {
            "earliest_partition_first_url",
            "latest_partition_last_url",
            "smallest_file",
        }
        assert all(
            item["footer_row_count"] == item["manifest_record_count"]
            and item["remote_content_length_bytes"] == item["manifest_content_length_bytes"]
            for item in entity["samples"]
        )


def test_works_contains_lean_h3_primitives_and_authors_is_deferred() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    path_types = summary["schema_replay"]["works"]["required_path_types"]
    assert path_types == {
        "authorships.author.id": "string",
        "authorships.author.orcid": "string",
        "id": "string",
        "is_paratext": "bool",
        "is_retracted": "bool",
        "primary_topic.domain.id": "string",
        "primary_topic.field.id": "string",
        "primary_topic.id": "string",
        "primary_topic.subfield.id": "string",
        "publication_date": "date32[day]",
        "publication_year": "int32",
        "type": "string",
    }
    disposition = summary["dependency_disposition"]
    assert disposition["initial_download_entities"] == ["works", "topics", "subfields", "fields"]
    assert disposition["initial_download_file_count"] == 2_454
    assert disposition["initial_download_manifest_content_length_bytes"] == 724_973_106_144
    assert disposition["authors_exclusion_saves_bytes"] == 52_769_910_496
    assert disposition["authors_entity"] == (
        "DEFERRED_OPTIONAL_IDENTITY_ENRICHMENT_NOT_REQUIRED_FOR_LEAN_H3_CORE"
    )


def test_storage_budget_and_access_boundary_are_fail_closed() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    plan = summary["transfer_storage_plan"]
    assert plan["raw_source_cap_bytes"] == 724_973_106_144
    assert plan["working_transformed_and_replay_cap_bytes"] == 1_087_459_659_216
    assert plan["total_reserved_budget_bytes"] == 1_812_432_765_360
    assert plan["planned_margin_after_reserve_bytes"] == 1_211_321_306_704
    assert plan["download_started"] is False
    access = summary["access_accounting"]
    assert access["remote_parquet_files_opened_footer_only"] == 6
    assert access["remote_parquet_files_materialized_locally"] == 0
    assert access["remote_transport_bytes_metered"] is False
    assert access["remote_source_row_values_read"] == 0
    assert access["holdout_or_private_g4_rows_read"] == 0
    assert access["g4_open"] is False
    authorization = summary["authorization"]
    assert authorization["full_source_data_download"] == "NOT_AUTHORIZED_BY_THIS_PREFLIGHT"
    assert authorization["source_row_value_access"] == "NOT_AUTHORIZED_BY_THIS_PREFLIGHT"
    assert authorization["manifest_driven_downloader_implementation_and_dry_run"] == "AUTHORIZED_NEXT"


def test_preflight_binds_source_receipt_and_uses_footer_only_api() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    binding = summary["source_binding"]
    source_receipt = ROOT / binding["source_receipt_relative_path"]
    assert _sha256(source_receipt) == binding["source_receipt_sha256"]
    source = SCRIPT.read_text(encoding="utf-8")
    assert "pq.ParquetFile(" in source
    for forbidden in ("pq.read_table(", ".read_row_group(", ".iter_batches("):
        assert forbidden not in source
    unresolved = summary["remaining_pre_download_uncertainties"]
    assert unresolved["all_works_file_schema_unanimity"] == (
        "NOT_ESTABLISHED_BY_THREE_FILE_REPRESENTATIVE_PREFLIGHT"
    )
