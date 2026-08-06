#!/usr/bin/env python3
"""Inspect representative OpenAlex Parquet footers without reading row values."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pyarrow as pa
from pyarrow import fs
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RECEIPT = ROOT / (
    "experiments/evidence/results/"
    "pivot_feedback_lean_h3_openalex_source_intake_v1_20260806.summary.json"
)
SOURCE_RECEIPT_SHA256 = "9c6def04d23755b8d35ae6ce90be282c45f6938afd39df4287545151af8c7946"
MANIFEST_ROOT = Path("/ssd/pivot-feedback-lean-h3-openalex-source-intake-v1-20260806")
STATUS = (
    "PASS_REPRESENTATIVE_REMOTE_FOOTER_SCHEMA_STABLE_WORKS_SUFFICIENT_FOR_"
    "LEAN_H3_AUTHORS_DEFERRED_NO_ROW_ACCESS_NO_DOWNLOAD"
)
WORKS_REQUIRED_PATHS = (
    "id",
    "publication_date",
    "publication_year",
    "type",
    "is_paratext",
    "is_retracted",
    "authorships.author.id",
    "authorships.author.orcid",
    "primary_topic.id",
    "primary_topic.subfield.id",
    "primary_topic.field.id",
    "primary_topic.domain.id",
)


class FooterPreflightError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FooterPreflightError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _updated_date(url: str) -> str:
    marker = "updated_date="
    require(marker in url, f"missing updated_date partition: {url}")
    return url.split(marker, 1)[1].split("/", 1)[0]


def _representative_files(files: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    require(bool(files), "empty file list")
    by_time = sorted(files, key=lambda item: (_updated_date(str(item["url"])), str(item["url"])))
    by_size = sorted(files, key=lambda item: (int(item["meta"]["content_length"]), str(item["url"])))
    selected = {
        "earliest_partition_first_url": by_time[0],
        "latest_partition_last_url": by_time[-1],
        "smallest_file": by_size[0],
    }
    require(len({str(item["url"]) for item in selected.values()}) == 3, "sample URLs overlap")
    return selected


def _nested_type(schema: pa.Schema, dotted_path: str) -> str:
    parts = dotted_path.split(".")
    try:
        data_type = schema.field(parts[0]).type
        for part in parts[1:]:
            while pa.types.is_list(data_type) or pa.types.is_large_list(data_type):
                data_type = data_type.value_type
            require(pa.types.is_struct(data_type), f"non-struct before {part}: {dotted_path}")
            data_type = data_type.field(part).type
    except (KeyError, IndexError) as exc:
        raise FooterPreflightError(f"missing schema path: {dotted_path}") from exc
    return str(data_type)


def _inspect_entity(
    entity: str,
    manifest: Mapping[str, Any],
    s3: fs.S3FileSystem,
    required_paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    samples: list[dict[str, Any]] = []
    selections = _representative_files(list(manifest["files"]))
    for selection, item in selections.items():
        url = str(item["url"])
        require(url.startswith("s3://openalex/"), f"unexpected source URL: {url}")
        key = url.removeprefix("s3://")
        expected_bytes = int(item["meta"]["content_length"])
        expected_rows = int(item["meta"]["record_count"])
        info = s3.get_file_info(key)
        require(info.type == fs.FileType.File, f"remote file missing: {url}")
        require(info.size == expected_bytes, f"remote size drift: {url}")
        with s3.open_input_file(key) as source:
            parquet = pq.ParquetFile(source)
            metadata = parquet.metadata
            schema = parquet.schema_arrow.remove_metadata()
            schema_bytes = schema.serialize().to_pybytes()
            require(metadata.num_rows == expected_rows, f"footer row-count drift: {url}")
            path_types = {path: _nested_type(schema, path) for path in required_paths}
            samples.append(
                {
                    "selection": selection,
                    "url": url,
                    "partition_updated_date": _updated_date(url),
                    "manifest_content_length_bytes": expected_bytes,
                    "remote_content_length_bytes": info.size,
                    "manifest_record_count": expected_rows,
                    "footer_row_count": metadata.num_rows,
                    "footer_row_group_count": metadata.num_row_groups,
                    "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
                    "top_level_fields": schema.names,
                    "required_path_types": path_types,
                }
            )
    fingerprints = {item["schema_sha256"] for item in samples}
    top_level = {tuple(item["top_level_fields"]) for item in samples}
    required = {
        tuple(sorted(item["required_path_types"].items()))
        for item in samples
    }
    require(len(fingerprints) == 1, f"representative {entity} schema drift")
    require(len(top_level) == 1, f"representative {entity} top-level drift")
    require(len(required) == 1, f"representative {entity} required-path drift")
    return {
        "sampling_rule": "EARLIEST_FIRST_URL_LATEST_LAST_URL_AND_SMALLEST_FILE_BY_CAPTURED_MANIFEST",
        "sample_count": len(samples),
        "captured_manifest_file_count": len(manifest["files"]),
        "all_sample_schema_fingerprints_equal": True,
        "schema_sha256": next(iter(fingerprints)),
        "top_level_fields": list(next(iter(top_level))),
        "required_path_types": dict(next(iter(required))),
        "samples": samples,
    }


def inspect() -> dict[str, Any]:
    require(SOURCE_RECEIPT.is_file() and not SOURCE_RECEIPT.is_symlink(), "invalid source receipt")
    require(_sha256(SOURCE_RECEIPT) == SOURCE_RECEIPT_SHA256, "source receipt hash drift")
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    require(receipt["snapshot_candidate"]["manifest_date"] == "2026-06-26", "snapshot date drift")
    require(Path(receipt["captured_manifest_root"]) == MANIFEST_ROOT, "manifest root drift")

    manifests: dict[str, dict[str, Any]] = {}
    bindings = {item["label"]: item for item in receipt["captured_manifests"]}
    for entity in ("works", "authors", "topics", "subfields", "fields"):
        binding = bindings[entity]
        path = MANIFEST_ROOT / str(binding["relative_path"])
        require(path.is_file() and not path.is_symlink(), f"invalid manifest: {entity}")
        require(path.stat().st_size == binding["bytes"], f"manifest size drift: {entity}")
        require(_sha256(path) == binding["sha256"], f"manifest hash drift: {entity}")
        manifests[entity] = json.loads(path.read_text(encoding="utf-8"))

    s3 = fs.S3FileSystem(anonymous=True, region="us-east-1")
    works = _inspect_entity("works", manifests["works"], s3, WORKS_REQUIRED_PATHS)
    authors = _inspect_entity("authors", manifests["authors"], s3)

    core_entities = ("works", "topics", "subfields", "fields")
    raw_bytes = sum(int(manifests[name]["content_length"]) for name in core_entities)
    raw_files = sum(len(manifests[name]["files"]) for name in core_entities)
    authors_bytes = int(manifests["authors"]["content_length"])
    working_budget = raw_bytes * 3 // 2
    total_reserved = raw_bytes + working_budget
    available = int(receipt["download_capacity_preflight"]["ssd_available_bytes_at_capture"])
    require(total_reserved <= available, "planned storage reserve exceeds captured free space")

    return {
        "schema_version": "pivot_feedback_lean_h3_openalex_footer_preflight_summary.v1",
        "status": STATUS,
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "mode": "REPRESENTATIVE_REMOTE_PARQUET_FOOTER_ONLY_NO_ROW_VALUE_ACCESS_NOT_SCIENTIFIC_GATE_NOT_EXECUTION_AUTHORITY",
        "source_binding": {
            "source_receipt_relative_path": SOURCE_RECEIPT.relative_to(ROOT).as_posix(),
            "source_receipt_sha256": SOURCE_RECEIPT_SHA256,
            "captured_manifest_root": str(MANIFEST_ROOT),
            "snapshot_manifest_date": "2026-06-26",
        },
        "schema_replay": {"works": works, "authors": authors},
        "dependency_disposition": {
            "initial_download_entities": list(core_entities),
            "initial_download_file_count": raw_files,
            "initial_download_manifest_content_length_bytes": raw_bytes,
            "authors_entity": "DEFERRED_OPTIONAL_IDENTITY_ENRICHMENT_NOT_REQUIRED_FOR_LEAN_H3_CORE",
            "authors_exclusion_saves_bytes": authors_bytes,
            "reason": (
                "WORKS_AUTHORSHIPS_CONTAIN_AUTHOR_ID_AND_ORCID_AND_WORKS_PRIMARY_TOPIC_CONTAINS_"
                "TOPIC_SUBFIELD_FIELD_DOMAIN_HIERARCHY"
            ),
            "taxonomy_entities_retained": (
                "TINY_AUTHORITATIVE_TAXONOMY_BINDING_DESPITE_NESTED_WORKS_HIERARCHY"
            ),
        },
        "transfer_storage_plan": {
            "source_selection": "EXACT_URLS_FROM_CAPTURED_MANIFESTS_NOT_MUTABLE_PREFIX_SYNC",
            "download_method": "MANIFEST_DRIVEN_RESUMABLE_RANGE_DOWNLOADER_WITH_PART_FILES_AND_CHECKPOINT",
            "per_file_preconditions": ["EXACT_CAPTURED_URL", "CONTENT_LENGTH_EQUALS_CAPTURED_MANIFEST"],
            "post_download_inventory": "COMPUTE_LOCAL_SHA256_FOR_ALL_FILES_AND_BIND_EXACT_PATH_SIZE_HASH",
            "raw_source_cap_bytes": raw_bytes,
            "working_transformed_and_replay_cap_bytes": working_budget,
            "working_cap_multiplier_of_raw": 1.5,
            "total_reserved_budget_bytes": total_reserved,
            "total_budget_multiplier_of_raw": 2.5,
            "ssd_available_bytes_at_source_intake_capture": available,
            "planned_margin_after_reserve_bytes": available - total_reserved,
            "budget_semantics": "OPERATIONAL_CAP_NOT_OBSERVED_TRANSFORMATION_SIZE_NOT_SCIENTIFIC_THRESHOLD",
            "download_started": False,
        },
        "remaining_pre_download_uncertainties": {
            "all_works_file_schema_unanimity": "NOT_ESTABLISHED_BY_THREE_FILE_REPRESENTATIVE_PREFLIGHT",
            "historical_and_calendar_year_completeness": "NOT_ESTABLISHED_BY_SCHEMA",
            "h3_horizon_maturity": "NOT_ESTABLISHED_BY_SCHEMA",
            "disjoint_discovery_and_confirmatory_cohorts": "NOT_SELECTED",
            "estimand_specific_precision_rule": "NOT_FROZEN",
            "actual_transformed_storage_use": "NOT_OBSERVED_BUDGET_ONLY",
        },
        "authorization": {
            "manifest_driven_downloader_implementation_and_dry_run": "AUTHORIZED_NEXT",
            "full_source_data_download": "NOT_AUTHORIZED_BY_THIS_PREFLIGHT",
            "source_row_value_access": "NOT_AUTHORIZED_BY_THIS_PREFLIGHT",
            "current_g1_or_model_selection_row_access": "NOT_AUTHORIZED",
            "current_g4_or_holdout_access": "NOT_AUTHORIZED",
            "confirmatory_successor_execution": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "access_accounting": {
            "local_captured_manifest_files_read": 5,
            "remote_parquet_files_opened_footer_only": 6,
            "remote_parquet_files_materialized_locally": 0,
            "remote_transport_bytes_metered": False,
            "remote_source_row_values_read": 0,
            "current_g1_label_rows_read": 0,
            "model_selection_rows_read": 0,
            "holdout_or_private_g4_rows_read": 0,
            "seal_open_events": 0,
            "g4_open": False,
            "models_fitted": 0,
        },
        "claim_boundary": (
            "This representative footer-only preflight establishes that the sampled earliest, latest, "
            "and smallest Works files share one schema containing the lean-H3 author, date, and topic "
            "primitives, so the separate Authors entity is not required for the lean-H3 core download. "
            "It does not establish schema unanimity across all files, historical completeness, horizon "
            "maturity, eligible cohorts, precision, download completion, an association, null, "
            "prediction, mechanism, causal effect, or scientific effect."
        ),
    }


def _write_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FooterPreflightError(f"output exists: {path}")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = inspect()
    if args.output is not None:
        _write_create_only(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
