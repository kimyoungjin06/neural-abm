#!/usr/bin/env python3
"""Independently verify a production citation chronology output tree."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import audit_openalex_canonical_work_time_spine_v3 as canonical
import audit_openalex_work_side_author_history_v3 as helper


OUTPUTS_SCHEMA = "openalex_g1_citation_edge_chronology_outputs.v3"
REPORT_SCHEMA = "openalex_g1_citation_edge_chronology_report.v3"
VERIFICATION_SCHEMA = "openalex_g1_citation_edge_chronology_verification.v3"
FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_CITATION_EDGE_CHRONOLOGY_ACCOUNTING_WITH_CANONICAL_"
    "TEMPORAL_V3_PARENT_AND_BIBLIOGRAPHIC_EVENT_TIME_CONVENTION_LIMITATION"
)
VERIFICATION_STATUS = "PASS_INDEPENDENT_FULL_ARTIFACT_AND_ROW_LEVEL_CITATION_CHRONOLOGY_VERIFICATION"
COMPLETION_CONTRACT = "RECEIPT_WRITTEN_LAST_AND_FINAL_DISK_VALIDATED"
SOURCE_ROW_ACCOUNTING_KEYS = frozenset(
    {"raw_in_snapshot_target_rows", "raw_external_target_rows"}
)
PARTITION_COUNT = 4096
BATCH_ROWS = 262_144
USABLE = frozenset({2, 3})
BUCKET_NAMES = (
    "self_loop",
    "source_temporal_unusable",
    "target_external_open_world",
    "target_temporal_unusable",
    "negative_lag_target_after_source",
    "same_day",
    "positive_lag",
)
BUCKET_CODE = {name: index for index, name in enumerate(BUCKET_NAMES)}
AXIS_CATEGORIES = {
    "primary_bucket_x_source_temporal_disposition": tuple(f"disposition_{value}" for value in range(6)),
    "primary_bucket_x_target_membership": ("external_open_world", "in_snapshot"),
    "primary_bucket_x_target_temporal_disposition_or_external_na": (
        "disposition_0", "disposition_1", "disposition_2", "disposition_3",
        "disposition_4", "disposition_5", "external_NA",
    ),
    "primary_bucket_x_self_loop_flag": ("false", "true"),
    "primary_bucket_x_signed_lag_class_or_na": ("negative", "same_day", "positive", "NA"),
    "primary_bucket_x_pair_multiplicity_class": ("one", "two", "three_or_more"),
}

DEDUP_SCHEMA = pa.schema(
    [
        pa.field("citing_work_id", pa.uint64(), nullable=False),
        pa.field("cited_work_id", pa.uint64(), nullable=False),
        pa.field("raw_pair_multiplicity", pa.uint32(), nullable=False),
        pa.field("citing_temporal_disposition", pa.uint8(), nullable=False),
        pa.field("target_membership", pa.uint8(), nullable=False),
        pa.field("cited_temporal_disposition", pa.uint8(), nullable=True),
        pa.field("citing_publication_date", pa.date32(), nullable=True),
        pa.field("available_at", pa.date32(), nullable=True),
        pa.field("cited_publication_date", pa.date32(), nullable=True),
        pa.field("signed_lag_days", pa.int32(), nullable=True),
        pa.field("primary_bucket", pa.uint8(), nullable=False),
    ]
)
SAFE_SCHEMA = pa.schema(
    [
        pa.field("cited_work_id", pa.uint64(), nullable=False),
        pa.field("citing_work_id", pa.uint64(), nullable=False),
        pa.field("available_at", pa.date32(), nullable=False),
        pa.field("cited_publication_date", pa.date32(), nullable=False),
        pa.field("signed_lag_days", pa.int32(), nullable=False),
        pa.field("raw_pair_multiplicity", pa.uint32(), nullable=False),
    ]
)


class CitationVerificationError(RuntimeError):
    """Controlled verification failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CitationVerificationError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be an array")
    return list(value)


def _safe_relative(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts and path.as_posix() == value, f"unsafe {field}")
    return path


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _report_fingerprint(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    provenance = dict(payload["provenance"])
    provenance.pop("result_fingerprint_sha256", None)
    payload["provenance"] = provenance
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _numpy(array: pa.Array, dtype: str) -> np.ndarray:
    return array.to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def _nullable_int(array: pa.Array, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    nulls = pc.is_null(array).to_numpy(zero_copy_only=False)
    target_type = pa.int32() if pa.types.is_date32(array.type) else pa.int64()
    values = pc.fill_null(pc.cast(array, target_type), 0).to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    return values, nulls


def _expected_bucket(
    source: np.ndarray,
    target: np.ndarray,
    source_disposition: np.ndarray,
    internal: np.ndarray,
    target_disposition: np.ndarray,
    lag: np.ndarray,
) -> np.ndarray:
    source_usable = np.isin(source_disposition, np.array(sorted(USABLE), dtype="u1"))
    target_usable = internal & np.isin(target_disposition, np.array(sorted(USABLE), dtype="u1"))
    result = np.full(len(source), np.uint8(255), dtype="u1")
    remaining = np.ones(len(source), dtype=bool)
    for name, condition in (
        ("self_loop", source == target),
        ("source_temporal_unusable", ~source_usable),
        ("target_external_open_world", ~internal),
        ("target_temporal_unusable", ~target_usable),
        ("negative_lag_target_after_source", lag < 0),
        ("same_day", lag == 0),
        ("positive_lag", lag > 0),
    ):
        selected = remaining & condition
        result[selected] = np.uint8(BUCKET_CODE[name])
        remaining[selected] = False
    require(not bool(np.any(remaining)), "independent classifier is not total")
    return result


def _update_cross(
    totals: dict[str, np.ndarray], bucket: np.ndarray, categories: Mapping[str, np.ndarray]
) -> None:
    for axis, values in categories.items():
        count = len(AXIS_CATEGORIES[axis])
        observed = np.bincount(
            bucket.astype(np.int64) * count + values.astype(np.int64),
            minlength=len(BUCKET_NAMES) * count,
        ).reshape(len(BUCKET_NAMES), count)
        totals[axis] += observed.astype(np.uint64)


def _render_cross(totals: Mapping[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    cross: dict[str, Any] = {}
    margins: dict[str, Any] = {}
    for axis, categories in AXIS_CATEGORIES.items():
        values = totals[axis]
        cross[axis] = {
            bucket: {
                category: int(values[bucket_index, category_index])
                for category_index, category in enumerate(categories)
            }
            for bucket_index, bucket in enumerate(BUCKET_NAMES)
        }
        margins[axis] = {
            category: int(values[:, category_index].sum(dtype=np.uint64))
            for category_index, category in enumerate(categories)
        }
    return cross, margins


def _verify_artifacts(root: Path, receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    artifacts = [_mapping(item, "receipt artifact") for item in _list(receipt.get("artifacts"), "receipt artifacts")]
    require(receipt.get("artifact_count") == len(artifacts), "receipt artifact count mismatch")
    observed: list[dict[str, Any]] = []
    for index, item in enumerate(artifacts, start=1):
        if index % 1024 == 0:
            print(f"artifact hash verification {index}/{len(artifacts)}", file=sys.stderr, flush=True)
        relative = _safe_relative(item.get("relative_path"), "artifact path")
        actual = canonical._artifact_contract(root, relative.as_posix())
        require(actual == item, f"artifact differs: {relative.as_posix()}")
        observed.append(actual)
    require(receipt.get("artifact_tree_sha256") == helper._tree_digest(observed), "artifact tree mismatch")
    return observed


def _verify_manifest(root: Path, name: str, schema: pa.Schema) -> dict[str, Any]:
    manifest = helper.read_json(root / name, name)
    require(manifest.get("partition_count") == PARTITION_COUNT, f"{name} partition count mismatch")
    partitions = _list(manifest.get("partitions"), f"{name} partitions")
    require(len(partitions) == PARTITION_COUNT, f"{name} partition record count mismatch")
    records: list[dict[str, Any]] = []
    rows = 0
    for expected_partition, raw in enumerate(partitions):
        record = _mapping(raw, f"{name} partition")
        require(record.get("partition_id") == expected_partition, f"{name} partition order mismatch")
        path = root / _safe_relative(record.get("relative_path"), f"{name} path")
        parquet = pq.ParquetFile(path)
        require(parquet.schema_arrow == schema, f"{name} schema mismatch: {path.name}")
        require(parquet.metadata.num_rows == record.get("rows"), f"{name} footer row mismatch: {path.name}")
        require(canonical._artifact_contract(root, record["relative_path"]) == {
            "relative_path": record["relative_path"], "bytes": record["bytes"], "sha256": record["sha256"]
        }, f"{name} record artifact mismatch")
        records.append(record)
        rows += record["rows"]
    require(manifest.get("rows") == rows, f"{name} row total mismatch")
    require(manifest.get("partition_tree_sha256") == helper._tree_digest(records), f"{name} tree mismatch")
    return manifest


def _verify_deduplicated_rows(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    totals = {
        "raw_pair_rows": 0,
        "unique_pair_rows": 0,
        "duplicate_pair_groups": 0,
        "duplicate_pair_row_excess": 0,
        "maximum_pair_multiplicity": 0,
        "raw_self_loop_rows": 0,
        "unique_self_loop_pairs": 0,
        "unique_in_snapshot_target_pairs": 0,
        "unique_external_target_pairs": 0,
        "unique_pairs_with_available_at": 0,
        "chronology_safe_primary_pairs": 0,
        "raw_in_snapshot_target_rows": 0,
        "raw_external_target_rows": 0,
    }
    bucket_totals = np.zeros(len(BUCKET_NAMES), dtype=np.uint64)
    cross_totals = {
        axis: np.zeros((len(BUCKET_NAMES), len(categories)), dtype=np.uint64)
        for axis, categories in AXIS_CATEGORIES.items()
    }
    sensitivity = {"primary_zero_day": 0, "tolerance_30_days": 0, "tolerance_365_days": 0, "annual_only": 0}
    for partition, record in enumerate(_list(manifest["partitions"], "dedup partitions")):
        if partition % 64 == 0:
            print(f"deduplicated row verification {partition}/{PARTITION_COUNT}", file=sys.stderr, flush=True)
        path = root / _safe_relative(record["relative_path"], "dedup path")
        parquet = pq.ParquetFile(path)
        previous: tuple[int, int] | None = None
        partition_rows = 0
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
            source = _numpy(batch.column(0), "<u8")
            target = _numpy(batch.column(1), "<u8")
            multiplicity = _numpy(batch.column(2), "<u4")
            source_disposition = _numpy(batch.column(3), "u1")
            membership = _numpy(batch.column(4), "u1")
            target_disposition, target_disposition_null = _nullable_int(batch.column(5), "u1")
            source_day, source_day_null = _nullable_int(batch.column(6), "<i4")
            available_day, available_null = _nullable_int(batch.column(7), "<i4")
            target_day, target_day_null = _nullable_int(batch.column(8), "<i4")
            lag, lag_null = _nullable_int(batch.column(9), "<i4")
            bucket = _numpy(batch.column(10), "u1")
            require(bool(np.all(source % PARTITION_COUNT == partition)), "dedup citing partition mismatch")
            require(bool(np.all(multiplicity >= 1)), "zero pair multiplicity")
            require(bool(np.all(source_disposition <= 5)), "unknown source disposition")
            require(bool(np.all((membership == 0) | (membership == 1))), "unknown target membership")
            internal = membership == 1
            require(bool(np.array_equal(target_disposition_null, ~internal)), "target disposition nullability mismatch")
            require(bool(np.all(target_disposition[internal] <= 5)), "unknown target disposition")
            require(bool(np.array_equal(source_day_null, source_disposition == 0)), "citing publication-date nullability mismatch")
            source_usable = np.isin(source_disposition, np.array(sorted(USABLE), dtype="u1"))
            require(bool(np.array_equal(available_null, ~source_usable)), "available_at nullability mismatch")
            require(bool(np.all(available_day[source_usable] == source_day[source_usable])), "available_at differs from citing publication date")
            expected_target_null = (~internal) | (target_disposition == 0)
            require(bool(np.array_equal(target_day_null, expected_target_null)), "cited publication-date nullability mismatch")
            expected_lag_null = source_day_null | target_day_null
            require(bool(np.array_equal(lag_null, expected_lag_null)), "signed-lag nullability mismatch")
            lag_available = ~lag_null
            require(bool(np.all(lag[lag_available].astype(np.int64) == source_day[lag_available].astype(np.int64) - target_day[lag_available].astype(np.int64))), "signed lag arithmetic mismatch")
            expected_bucket = _expected_bucket(source, target, source_disposition, internal, target_disposition, lag)
            require(bool(np.array_equal(bucket, expected_bucket)), "stored primary bucket differs from independent classifier")
            if len(source):
                if previous is not None:
                    require((int(source[0]), int(target[0])) > previous, "dedup order differs across row groups")
                if len(source) > 1:
                    ordered = (source[1:] > source[:-1]) | ((source[1:] == source[:-1]) & (target[1:] > target[:-1]))
                    require(bool(np.all(ordered)), "dedup pairs are not strictly ordered")
                previous = (int(source[-1]), int(target[-1]))
            self_loop = source == target
            target_usable = internal & np.isin(target_disposition, np.array(sorted(USABLE), dtype="u1"))
            common = (~self_loop) & source_usable & target_usable
            safe = np.isin(bucket, np.array([BUCKET_CODE["same_day"], BUCKET_CODE["positive_lag"]], dtype="u1"))
            require(bool(np.array_equal(safe, common & (lag >= 0))), "safe predicate differs")
            totals["raw_pair_rows"] += int(multiplicity.sum(dtype=np.uint64))
            totals["unique_pair_rows"] += len(source)
            totals["duplicate_pair_groups"] += int(np.count_nonzero(multiplicity > 1))
            totals["duplicate_pair_row_excess"] += int((multiplicity.astype(np.uint64) - 1).sum(dtype=np.uint64))
            totals["maximum_pair_multiplicity"] = max(totals["maximum_pair_multiplicity"], int(multiplicity.max(initial=0)))
            totals["raw_self_loop_rows"] += int(multiplicity[self_loop].sum(dtype=np.uint64))
            totals["unique_self_loop_pairs"] += int(np.count_nonzero(self_loop))
            totals["unique_in_snapshot_target_pairs"] += int(np.count_nonzero(internal))
            totals["unique_external_target_pairs"] += int(np.count_nonzero(~internal))
            totals["unique_pairs_with_available_at"] += int(np.count_nonzero(source_usable))
            totals["chronology_safe_primary_pairs"] += int(np.count_nonzero(safe))
            totals["raw_in_snapshot_target_rows"] += int(multiplicity[internal].sum(dtype=np.uint64))
            totals["raw_external_target_rows"] += int(multiplicity[~internal].sum(dtype=np.uint64))
            bucket_totals += np.bincount(bucket, minlength=len(BUCKET_NAMES)).astype(np.uint64)
            lag_class = np.full(len(source), 3, dtype="u1")
            lag_class[lag_available & (lag < 0)] = 0
            lag_class[lag_available & (lag == 0)] = 1
            lag_class[lag_available & (lag > 0)] = 2
            multiplicity_class = np.full(len(source), 2, dtype="u1")
            multiplicity_class[multiplicity == 1] = 0
            multiplicity_class[multiplicity == 2] = 1
            target_category = target_disposition.copy()
            target_category[~internal] = 6
            _update_cross(
                cross_totals,
                bucket,
                {
                    "primary_bucket_x_source_temporal_disposition": source_disposition,
                    "primary_bucket_x_target_membership": internal.astype("u1"),
                    "primary_bucket_x_target_temporal_disposition_or_external_na": target_category,
                    "primary_bucket_x_self_loop_flag": self_loop.astype("u1"),
                    "primary_bucket_x_signed_lag_class_or_na": lag_class,
                    "primary_bucket_x_pair_multiplicity_class": multiplicity_class,
                },
            )
            sensitivity["primary_zero_day"] += int(np.count_nonzero(common & (lag >= 0)))
            sensitivity["tolerance_30_days"] += int(np.count_nonzero(common & (lag >= -30)))
            sensitivity["tolerance_365_days"] += int(np.count_nonzero(common & (lag >= -365)))
            if bool(np.any(common)):
                source_year = source_day[common].astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64)
                target_year = target_day[common].astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64)
                sensitivity["annual_only"] += int(np.count_nonzero(target_year <= source_year))
            partition_rows += len(source)
        require(partition_rows == record["rows"], "dedup decoded partition row mismatch")
    cross, margins = _render_cross(cross_totals)
    return {
        "totals": totals,
        "buckets": {name: int(bucket_totals[index]) for index, name in enumerate(BUCKET_NAMES)},
        "sensitivity": sensitivity,
        "cross_tabs": cross,
        "attribute_margins": margins,
    }


def _verify_safe_rows(root: Path, manifest: Mapping[str, Any]) -> dict[str, int]:
    rows = 0
    raw_multiplicity = 0
    for partition, record in enumerate(_list(manifest["partitions"], "safe partitions")):
        if partition % 64 == 0:
            print(f"safe row verification {partition}/{PARTITION_COUNT}", file=sys.stderr, flush=True)
        path = root / _safe_relative(record["relative_path"], "safe path")
        parquet = pq.ParquetFile(path)
        previous: tuple[int, int] | None = None
        partition_rows = 0
        for batch in parquet.iter_batches(batch_size=BATCH_ROWS):
            target = _numpy(batch.column(0), "<u8")
            source = _numpy(batch.column(1), "<u8")
            available = _numpy(pc.cast(batch.column(2), pa.int32()), "<i4")
            cited = _numpy(pc.cast(batch.column(3), pa.int32()), "<i4")
            lag = _numpy(batch.column(4), "<i4")
            multiplicity = _numpy(batch.column(5), "<u4")
            require(bool(np.all(target % PARTITION_COUNT == partition)), "safe cited partition mismatch")
            require(bool(np.all(lag >= 0)), "negative safe lag")
            require(bool(np.all(lag.astype(np.int64) == available.astype(np.int64) - cited.astype(np.int64))), "safe lag arithmetic mismatch")
            require(bool(np.all(multiplicity >= 1)), "zero safe multiplicity")
            if len(target):
                if previous is not None:
                    require((int(target[0]), int(source[0])) > previous, "safe order differs across row groups")
                if len(target) > 1:
                    ordered = (target[1:] > target[:-1]) | ((target[1:] == target[:-1]) & (source[1:] > source[:-1]))
                    require(bool(np.all(ordered)), "safe pairs are not strictly ordered")
                previous = (int(target[-1]), int(source[-1]))
            partition_rows += len(target)
            raw_multiplicity += int(multiplicity.sum(dtype=np.uint64))
        require(partition_rows == record["rows"], "safe decoded partition row mismatch")
        rows += partition_rows
    return {"rows": rows, "raw_pair_multiplicity": raw_multiplicity}


def _verify_report_accounting(
    report: Mapping[str, Any],
    dedup: Mapping[str, Any],
    safe: Mapping[str, int],
    safe_manifest: Mapping[str, Any],
) -> None:
    totals = _mapping(dedup["totals"], "independent pair totals")
    accounting = _mapping(report["pair_accounting"], "report pair accounting")
    for key, value in totals.items():
        if key in SOURCE_ROW_ACCOUNTING_KEYS:
            continue
        require(accounting.get(key) == value, f"independent pair accounting differs: {key}")
    source = _mapping(report["source_accounting"], "source accounting")
    require(source["raw_in_snapshot_target_rows"] == totals["raw_in_snapshot_target_rows"], "raw internal membership differs")
    require(source["raw_external_target_rows"] == totals["raw_external_target_rows"], "raw external membership differs")
    require(report["exclusive_pair_buckets"] == dedup["buckets"], "independent bucket totals differ")
    require(report["chronology_sensitivity_unique_pairs"] == dedup["sensitivity"], "independent sensitivity differs")
    require(report["cross_tabs"] == dedup["cross_tabs"], "independent cross-tabs differ")
    require(report["attribute_margins"] == dedup["attribute_margins"], "independent attribute margins differ")
    require(safe["rows"] == totals["chronology_safe_primary_pairs"], "safe/dedup row closure differs")
    require(safe["rows"] == safe_manifest["rows"], "safe manifest row closure differs")


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    receipt_path = root / "citation_edge_chronology_audit.outputs.json"
    receipt = helper.read_json(receipt_path, "citation receipt")
    require(receipt.get("schema_version") == OUTPUTS_SCHEMA, "receipt schema mismatch")
    require(receipt.get("status") == FULL_STATUS, "receipt full status mismatch")
    require(receipt.get("scope") == "FULL_CURRENT_SNAPSHOT_PRODUCTION", "receipt scope mismatch")
    require(receipt.get("production_component_pass") is True, "receipt is not production PASS")
    require(receipt.get("completion_contract") == COMPLETION_CONTRACT, "receipt completion contract mismatch")
    artifacts = _verify_artifacts(root, receipt)
    report = helper.read_json(root / "citation_edge_chronology_audit.json", "citation report")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == FULL_STATUS, "report status mismatch")
    require(_report_fingerprint(report) == receipt.get("result_fingerprint_sha256"), "report fingerprint mismatch")
    dedup_manifest = _verify_manifest(root, "deduplicated_edge_manifest.json", DEDUP_SCHEMA)
    safe_manifest = _verify_manifest(root, "chronology_safe_edge_manifest.json", SAFE_SCHEMA)
    dedup = _verify_deduplicated_rows(root, dedup_manifest)
    safe = _verify_safe_rows(root, safe_manifest)
    _verify_report_accounting(report, dedup, safe, safe_manifest)
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "verification_status": VERIFICATION_STATUS,
        "receipt_sha256": canonical._sha256_file(receipt_path),
        "receipt_artifact_count": len(artifacts),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "task_fingerprint_sha256": receipt["task_fingerprint_sha256"],
        "result_fingerprint_sha256": receipt["result_fingerprint_sha256"],
        "full_artifact_hash_verification": True,
        "full_deduplicated_row_scan": True,
        "full_chronology_safe_row_scan": True,
        "deduplicated_unique_pair_rows": dedup["totals"]["unique_pair_rows"],
        "chronology_safe_pair_rows": safe["rows"],
        "claim_boundary": (
            "Verification establishes artifact integrity and internal chronology mechanics only. "
            "It does not establish historical edge observation, maturation, G-1, G0, empirical G1, "
            "an NABM effect, or a scientific effect."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify(args.output_dir)
    except (CitationVerificationError, OSError, pa.ArrowException) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
