#!/usr/bin/env python3
"""Independent read-only verifier for normalized cutoff-primitives outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/pivot_feedback_cutoff_primitives_v0.yaml"
OUTPUTS_SCHEMA = "pivot_feedback_cutoff_primitives_outputs.v1"
REPORT_SCHEMA = "pivot_feedback_cutoff_primitives_report.v1"
MANIFEST_SCHEMA = "pivot_feedback_cutoff_primitives_manifest.v1"
PARTITION_COUNT = 4096

WORK_SCHEMA = pa.schema(
    [
        pa.field("work_id_numeric", pa.uint64(), nullable=False),
        pa.field("publication_day", pa.int32(), nullable=True),
        pa.field("publication_year", pa.uint16(), nullable=True),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=False),
        pa.field("is_paratext", pa.bool_(), nullable=False),
        pa.field("is_retracted", pa.bool_(), nullable=True),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("primary_topic_id", pa.uint32(), nullable=True),
        pa.field("primary_field_id", pa.uint8(), nullable=True),
        pa.field("primary_domain_id", pa.uint8(), nullable=True),
    ]
)


class VerificationError(RuntimeError):
    """Fail-closed verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            while chunk := handle.read(chunk_bytes):
                digest.update(chunk)
    finally:
        pass
    return digest.hexdigest()


def _tree_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        list(records),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path, field: str) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _safe_relative(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field}")
    require(path.as_posix() == value, f"noncanonical {field}")
    return path


def _tree_identity(root: Path) -> dict[str, tuple[int, int, int]]:
    result: dict[str, tuple[int, int, int]] = {}
    for path in sorted(root.rglob("*"), key=lambda value: value.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        require(not path.is_symlink(), f"symlink in output tree: {relative}")
        result[relative] = (info.st_mode, info.st_size, info.st_mtime_ns)
    return result


def _nullable_numpy(array: pa.ChunkedArray, dtype: str) -> tuple[np.ndarray, np.ndarray]:
    combined = array.combine_chunks()
    nulls = pc.is_null(combined).to_numpy(zero_copy_only=False).astype(bool, copy=False)
    values = pc.fill_null(combined, 0).to_numpy(zero_copy_only=False).astype(dtype, copy=False)
    return values, nulls


def _canonical_partition(root: Path, partition: int) -> pa.Table:
    path = root / "spine" / f"part-{partition:04d}.parquet"
    table = pq.read_table(path).combine_chunks()
    expected = pa.schema(
        [
            pa.field("work_id_numeric", pa.uint64(), nullable=False),
            pa.field("publication_day", pa.int32(), nullable=True),
            pa.field("publication_year", pa.uint16(), nullable=True),
            pa.field("temporal_disposition", pa.uint8(), nullable=False),
        ]
    )
    require(table.schema == expected, f"canonical schema mismatch in partition {partition}")
    return table


def _verify_artifacts(root: Path, receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = receipt.get("artifacts")
    require(isinstance(raw, list), "receipt artifacts must be an array")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        item = _mapping(value, f"artifact {index}")
        relative = _safe_relative(item.get("relative_path"), f"artifact {index} path")
        relative_text = relative.as_posix()
        require(relative_text not in seen, "duplicate receipt artifact")
        seen.add(relative_text)
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"missing artifact {relative_text}")
        require(path.stat().st_size == item.get("bytes"), f"artifact byte mismatch: {relative_text}")
        require(_sha256_file(path) == item.get("sha256"), f"artifact SHA mismatch: {relative_text}")
        verified.append(item)
    published = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(
        published == seen | {"pivot_feedback_cutoff_panel.outputs.json"},
        "published file set differs from receipt",
    )
    require(len(verified) == receipt.get("artifact_count"), "receipt artifact count mismatch")
    return verified


def _validate_parent_receipt(root: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = root / str(expected["receipt_basename"])
    require(path.is_file() and not path.is_symlink(), f"missing {label} receipt")
    require(_sha256_file(path) == expected["receipt_sha256"], f"{label} receipt SHA mismatch")
    value = _read_json(path, f"{label} receipt")
    require(value.get("schema_version") == expected["receipt_schema_version"], f"{label} schema mismatch")
    require(value.get("status") == expected["required_status"], f"{label} status mismatch")
    wanted = expected.get("result_fingerprint_sha256")
    if wanted is not None:
        require(value.get("result_fingerprint_sha256") == wanted, f"{label} result fingerprint mismatch")
    return value


def verify(
    root: Path,
    *,
    config_path: Path,
    canonical_root: Path,
    content_root: Path,
    author_root: Path,
    relation_root: Path,
    citation_root: Path,
) -> dict[str, Any]:
    root = root.resolve(strict=True)
    before = _tree_identity(root)
    config_bytes = config_path.resolve(strict=True).read_bytes()
    config = yaml.safe_load(config_bytes)
    require(isinstance(config, dict), "config must be an object")
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    receipt_path = root / "pivot_feedback_cutoff_panel.outputs.json"
    receipt_sha = _sha256_file(receipt_path)
    receipt = _read_json(receipt_path, "receipt")
    require(receipt.get("schema_version") == OUTPUTS_SCHEMA, "receipt schema mismatch")
    artifacts = _verify_artifacts(root, receipt)
    require(_tree_digest(artifacts) == receipt.get("artifact_tree_sha256"), "artifact tree mismatch")
    report = _read_json(root / "pivot_feedback_cutoff_panel.json", "report")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == receipt.get("status"), "report/receipt status mismatch")
    require(report.get("scope") == receipt.get("scope"), "report/receipt scope mismatch")
    require(report.get("provenance", {}).get("config_sha256") == config_sha, "config binding mismatch")
    manifest = _read_json(root / "work_metadata_manifest.json", "manifest")
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema mismatch")
    require(manifest.get("partition_count") == PARTITION_COUNT, "manifest partition count mismatch")
    parts = manifest.get("partitions")
    require(isinstance(parts, list) and len(parts) == PARTITION_COUNT, "manifest parts mismatch")

    type_dictionary = _mapping(config["input_profile"]["work_type_dictionary"], "type dictionary")
    type_by_code = {int(code): str(name) for code, name in type_dictionary.items()}
    type_counts: Counter[str] = Counter()
    rows = 0
    null_counts = {
        "type": 0,
        "is_paratext": 0,
        "is_retracted": 0,
        "authors_count": 0,
        "primary_topic__id": 0,
        "primary_topic__domain__id": 0,
        "primary_topic__field__id": 0,
    }
    authors_min: int | None = None
    authors_max: int | None = None
    full_scope = receipt.get("scope") == "FULL_CURRENT_SNAPSHOT_PRODUCTION"
    for partition, raw_item in enumerate(parts):
        if partition % 128 == 0:
            print(f"verify work partitions {partition}/{PARTITION_COUNT}", file=sys.stderr, flush=True)
        item = _mapping(raw_item, f"manifest part {partition}")
        require(item.get("partition_id") == partition, "manifest partition order mismatch")
        relative = _safe_relative(item.get("relative_path"), "manifest partition path")
        require(relative.as_posix() == f"work_metadata/part-{partition:04d}.parquet", "manifest partition path mismatch")
        table = pq.read_table(root / relative).combine_chunks()
        require(table.schema == WORK_SCHEMA, f"work schema mismatch in partition {partition}")
        require(len(table) == item.get("rows"), f"work row count mismatch in partition {partition}")
        work = table["work_id_numeric"].to_numpy(zero_copy_only=False)
        require(not len(work) or bool(np.all(work[1:] > work[:-1])), "work IDs not strictly increasing")
        require(not len(work) or bool(np.all(work % PARTITION_COUNT == partition)), "work partition formula mismatch")
        require((int(work[0]) if len(work) else None) == item.get("first_work_id"), "first work mismatch")
        require((int(work[-1]) if len(work) else None) == item.get("last_work_id"), "last work mismatch")

        canonical_table = _canonical_partition(canonical_root, partition)
        canonical_work = canonical_table["work_id_numeric"].to_numpy(zero_copy_only=False)
        if full_scope:
            require(np.array_equal(work, canonical_work), "full canonical work closure mismatch")
            selected = np.arange(len(work), dtype=np.int64)
        else:
            selected = np.searchsorted(canonical_work, work)
            require(bool(np.all(selected < len(canonical_work))), "pilot work outside canonical")
            require(np.array_equal(canonical_work[selected], work), "pilot work absent from canonical")
        for name in ("publication_day", "publication_year"):
            actual_values, actual_null = _nullable_numpy(table[name], "<i8")
            expected_values, expected_null = _nullable_numpy(
                canonical_table[name].take(pa.array(selected, type=pa.int64())), "<i8"
            )
            require(np.array_equal(actual_null, expected_null), f"{name} null-mask mismatch")
            require(np.array_equal(actual_values, expected_values), f"{name} value mismatch")
        require(
            np.array_equal(
                table["temporal_disposition"].to_numpy(zero_copy_only=False),
                canonical_table["temporal_disposition"].take(pa.array(selected, type=pa.int64())).to_numpy(zero_copy_only=False),
            ),
            "temporal disposition mismatch",
        )

        codes = table["work_type_code"].to_numpy(zero_copy_only=False)
        require(bool(np.all(np.isin(codes, np.array(sorted(type_by_code), dtype="u1")))), "unknown work type code")
        for code, count in zip(*np.unique(codes, return_counts=True), strict=True):
            type_counts[type_by_code[int(code)]] += int(count)
        require(table["is_paratext"].null_count == 0, "null is_paratext output")
        null_counts["is_retracted"] += table["is_retracted"].null_count
        authors = table["declared_authors_count"].to_numpy(zero_copy_only=False)
        if len(authors):
            local_min, local_max = int(authors.min()), int(authors.max())
            authors_min = local_min if authors_min is None else min(authors_min, local_min)
            authors_max = local_max if authors_max is None else max(authors_max, local_max)
        topic_null = pc.is_null(table["primary_topic_id"]).to_numpy(zero_copy_only=False)
        field_null = pc.is_null(table["primary_field_id"]).to_numpy(zero_copy_only=False)
        domain_null = pc.is_null(table["primary_domain_id"]).to_numpy(zero_copy_only=False)
        require(np.array_equal(topic_null, field_null), "topic/field output null-mask mismatch")
        require(np.array_equal(topic_null, domain_null), "topic/domain output null-mask mismatch")
        count_null = int(np.count_nonzero(topic_null))
        null_counts["primary_topic__id"] += count_null
        null_counts["primary_topic__field__id"] += count_null
        null_counts["primary_topic__domain__id"] += count_null
        rows += len(table)

    require(rows == manifest.get("rows") == receipt.get("work_metadata_rows"), "global row closure mismatch")
    require(
        _tree_digest([_mapping(value, "manifest part") for value in parts])
        == manifest.get("work_metadata_tree_sha256"),
        "independent work-metadata tree mismatch",
    )
    require(manifest.get("work_metadata_tree_sha256") == receipt.get("work_metadata_tree_sha256"), "work tree binding mismatch")
    if full_scope:
        profile = _mapping(config["input_profile"], "profile")
        require(rows == config["source"]["retained_rows"], "full retained-row mismatch")
        raw_type_counts = _mapping(profile["exact_work_type_counts"], "raw type counts")
        output_type_counts = dict(sorted(type_counts.items()))
        require(set(output_type_counts) == set(raw_type_counts), "full type domain mismatch")
        type_deltas = {
            name: int(raw_type_counts[name]) - int(output_type_counts[name])
            for name in raw_type_counts
        }
        require(all(delta >= 0 for delta in type_deltas.values()), "output type count exceeds raw input")
        require(sum(type_deltas.values()) == 1, "type counts do not remove exactly the W-1 row")
        raw_null_counts = _mapping(profile["null_counts"], "raw null counts")
        null_deltas = {
            name: int(raw_null_counts[name]) - int(null_counts[name])
            for name in raw_null_counts
        }
        require(all(delta in (0, 1) for delta in null_deltas.values()), "null counts are not a one-row quarantine projection")
        require(
            null_deltas["primary_topic__id"]
            == null_deltas["primary_topic__field__id"]
            == null_deltas["primary_topic__domain__id"],
            "topic hierarchy quarantine null deltas differ",
        )
        require(authors_min == profile["authors_count_minimum"], "full authors minimum mismatch")
        require(authors_max == profile["authors_count_maximum"], "full authors maximum mismatch")
        source_accounting = _mapping(report["source_accounting"], "report source accounting")
        require(source_accounting.get("work_type_counts") == raw_type_counts, "report raw type profile mismatch")
        require(source_accounting.get("null_counts") == raw_null_counts, "report raw null profile mismatch")

    parents_cfg = _mapping(config["parents"], "parents")
    roots = {
        "current_content_baseline": content_root.resolve(strict=True),
        "canonical_work_time_spine": canonical_root.resolve(strict=True),
        "author_history": author_root.resolve(strict=True),
        "work_topic_relation": relation_root.resolve(strict=True),
        "citation_chronology": citation_root.resolve(strict=True),
    }
    parent_receipts = {
        key: _validate_parent_receipt(root_value, _mapping(parents_cfg[key], key), key)
        for key, root_value in roots.items()
    }
    relation_cfg = _mapping(parents_cfg["work_topic_relation"], "relation")
    relation = roots["work_topic_relation"] / _safe_relative(relation_cfg["relation_relative_path"], "relation path")
    require(relation.stat().st_size == relation_cfg["relation_bytes"], "relation bytes mismatch")
    require(_sha256_file(relation) == relation_cfg["relation_sha256"], "relation SHA mismatch")
    citation_manifest = _read_json(roots["citation_chronology"] / "chronology_safe_edge_manifest.json", "citation manifest")
    require(
        citation_manifest.get("partition_tree_sha256")
        == parents_cfg["citation_chronology"]["chronology_safe_partition_tree_sha256"],
        "citation tree mismatch",
    )

    after = _tree_identity(root)
    require(before == after, "verification changed the output tree")
    return {
        "schema_version": "pivot_feedback_cutoff_primitives_independent_verification.v1",
        "status": "PASS_READ_ONLY_INDEPENDENT_NORMALIZED_CUTOFF_PRIMITIVES_VERIFICATION",
        "scope": receipt.get("scope"),
        "production_component_pass": bool(full_scope),
        "receipt_sha256": receipt_sha,
        "artifacts_rehashed": len(artifacts),
        "full_work_metadata_row_scan": True,
        "canonical_row_join_replayed": True,
        "work_metadata_rows": rows,
        "parent_receipt_sha256_verified": {
            key: parents_cfg[key]["receipt_sha256"] for key in parent_receipts
        },
        "relation_payload_rehashed": True,
        "citation_manifest_tree_verified": True,
        "output_tree_unchanged": True,
        "claim_boundary": config["claim_boundary"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--content-root", type=Path, required=True)
    parser.add_argument("--author-root", type=Path, required=True)
    parser.add_argument("--relation-root", type=Path, required=True)
    parser.add_argument("--citation-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = verify(
            args.root,
            config_path=args.config,
            canonical_root=args.canonical_root,
            content_root=args.content_root,
            author_root=args.author_root,
            relation_root=args.relation_root,
            citation_root=args.citation_root,
        )
        payload = json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
        if args.report is not None:
            report = args.report.resolve()
            require(args.root.resolve(strict=True) not in report.parents, "verification report must be outside output root")
            report.write_text(payload, encoding="utf-8")
        sys.stdout.write(payload)
        return 0
    except (VerificationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED_CUTOFF_PRIMITIVES_VERIFICATION", "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
