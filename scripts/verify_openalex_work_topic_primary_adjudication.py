#!/usr/bin/env python3
"""Independently verify the full work-topic primary-score successor output.

The verifier does not import the producer. It rehashes the receipt-bound output
tree and frozen parent control artifacts, reads the two frozen main source rows,
and streams every parent event group against every canonical output record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
STATUS = (
    "PASS_CURRENT_SNAPSHOT_WORK_TOPIC_RELATION_COMPONENT_"
    "WITH_PRIMARY_EXACT_SCORE_ADJUDICATION"
)
VERIFIER_STATUS = "PASS_FULL_READ_ONLY_INDEPENDENT_PRIMARY_ADJUDICATION_VERIFICATION"
RECEIPT_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_outputs.v1"
PLAN_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_plan.v1"
SIDECAR_SCHEMA = "openalex_g1_work_topic_score_conflict_sidecar.v1"
REPORT_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_report.v1"
STATE_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_state.v1"
VERIFIER_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_verifier.v1"

CONFIG_RELATIVE_PATH = (
    "experiments/configs/openalex_g1_work_topic_relation_primary_adjudication_v1.yaml"
)
CONFIG_SHA256 = "4bdc6caee203450b367f1e7621c4785373063edce93318e295aff5fd49752165"
PRODUCER_RELATIVE_PATH = "scripts/adjudicate_openalex_work_topic_primary_scores.py"
PRODUCER_SHA256 = "bf9f0af96d6af19b4732b40e59e070e9cc3b9232fd556b099cd50ea8b11b9d62"
PARENT_RECEIPT_SHA256 = (
    "7cf91e50a3934f783d278ae47b3764ec650b7d36c3625bfc4564da405a551507"
)
PARENT_RESULT_FINGERPRINT = (
    "ce13bc8aa2e07578c84c1f540d2f21f7e46b18189ddda12c963a2ccc154db6d5"
)
PARENT_ARTIFACTS = {
    "audit_plan.json": {
        "bytes": 2_180_730,
        "sha256": "b19de52306663a64a07404084f6e5b34e9d1ae44a2abb3f8937e41a3be0f47ae",
    },
    "work_topic_relation.events.sorted.bin": {
        "bytes": 34_100_538_048,
        "sha256": "b714008803b49dca0e6c4d5f273aa458a20549b415185721b255492f99768623",
    },
    "work_topic_relation_audit.json": {
        "bytes": 23_122,
        "sha256": "88aa8fa4d854e508c7e513b63f3240002a1ae19e25a1248ebf5e4fb2a08798f5",
    },
}
EXPECTED_OUTPUT_ARTIFACTS = {
    "adjudication_plan.json",
    "run_state.json",
    "work_topic_edges.primary_adjudicated.sorted.bin",
    "work_topic_relation_primary_adjudication.json",
    "work_topic_relation_primary_adjudication.md",
    "work_topic_score_conflicts.primary_adjudicated.json",
}
OUTPUT_RECEIPT = "work_topic_relation_primary_adjudication.outputs.json"

TOPIC_BITS = 14
EVENT_DTYPE = np.dtype(
    [
        ("key", "<u8"),
        ("score_bits", "<u8"),
        ("locator", "<u4"),
        ("kind", "u1"),
        ("padding", "V3"),
    ],
    align=False,
)
RELATION_DTYPE = np.dtype([("key", "<u8"), ("score_bits", "<u8")], align=False)
EXPECTED_COUNTS = {
    "event_rows": 1_420_855_752,
    "topic_raw_rows": 928_562_546,
    "unique_work_topic_edges": 928_561_783,
    "duplicate_edge_groups": 714,
    "duplicate_row_excess": 763,
    "score_conflict_edge_groups": 2,
}
EXPECTED_CONFLICTS = {
    (5_401_631 << TOPIC_BITS) | 13_112: {
        "work_numeric_id": 5_401_631,
        "topic_numeric_id": 13_112,
        "canonical_score": 0.0806,
        "canonical_score_bits_hex": "3fb4a2339c0ebee0",
        "main": {
            "relative_path": "openalex_works_20260330/part-487.parquet",
            "file_row": 49_810,
            "row_group": 1,
            "row_group_row": 17_042,
        },
        "topic_rows": [
            {
                "relative_path": "openalex_works_20260330__topics/delta-12.parquet",
                "file_row": 527_788,
                "score": 0.0805,
                "score_bits_hex": "3fb49ba5e353f7cf",
                "selected": False,
            },
            {
                "relative_path": "openalex_works_20260330__topics/delta-12.parquet",
                "file_row": 527_787,
                "score": 0.0806,
                "score_bits_hex": "3fb4a2339c0ebee0",
                "selected": True,
            },
        ],
    },
    (26_888_826 << TOPIC_BITS) | 10_455: {
        "work_numeric_id": 26_888_826,
        "topic_numeric_id": 10_455,
        "canonical_score": 0.7347,
        "canonical_score_bits_hex": "3fe782a9930be0df",
        "main": {
            "relative_path": "openalex_works_20260330/part-452.parquet",
            "file_row": 111_456,
            "row_group": 3,
            "row_group_row": 13_152,
        },
        "topic_rows": [
            {
                "relative_path": "openalex_works_20260330__topics/delta-41.parquet",
                "file_row": 256_212,
                "score": 0.7346,
                "score_bits_hex": "3fe781d7dbf487fd",
                "selected": False,
            },
            {
                "relative_path": "openalex_works_20260330__topics/delta-41.parquet",
                "file_row": 256_211,
                "score": 0.7347,
                "score_bits_hex": "3fe782a9930be0df",
                "selected": True,
            },
        ],
    },
}


class VerificationError(RuntimeError):
    """Raised when the independent verification contract fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    require(path.is_file() and not path.is_symlink(), "hash input is not regular")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while block := handle.read(chunk_bytes):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def read_json(path: Path, field: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{field} is not regular")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8", closefd=False) as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {field}") from exc
    finally:
        os.close(descriptor)
    require(isinstance(value, dict), f"{field} is not a mapping")
    return value


def safe_file(root: Path, relative: str, field: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field} path")
    candidate = root / path
    require(
        candidate.is_file() and not candidate.is_symlink(), f"{field} is not regular"
    )
    return candidate


def rich_identity(path: Path) -> dict[str, int]:
    info = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(info.st_mode), "source is not regular")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(info.st_mode),
        "links": int(info.st_nlink),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "bytes": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def _score_bits(value: float) -> int:
    require(math.isfinite(value), "score is not finite")
    normalized = 0.0 if value == 0.0 else value
    return int(np.asarray([normalized], dtype="<f8").view("<u8")[0])


def _verify_repository_bindings(repo_root: Path) -> None:
    require(
        sha256_file(safe_file(repo_root, CONFIG_RELATIVE_PATH, "config"))
        == CONFIG_SHA256,
        "config SHA mismatch",
    )
    require(
        sha256_file(safe_file(repo_root, PRODUCER_RELATIVE_PATH, "producer"))
        == PRODUCER_SHA256,
        "producer SHA mismatch",
    )


def _verify_output_controls(output: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require(output.is_dir() and not output.is_symlink(), "output is not a directory")
    require(stat.S_IMODE(output.stat().st_mode) == 0o700, "output mode mismatch")
    actual = {path.name for path in output.iterdir()}
    require(actual == EXPECTED_OUTPUT_ARTIFACTS | {OUTPUT_RECEIPT}, "output tree drift")
    for path in output.iterdir():
        require(path.is_file() and not path.is_symlink(), "output contains non-file")
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, "output file mode mismatch")

    receipt = read_json(output / OUTPUT_RECEIPT, "output receipt")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema mismatch")
    require(receipt.get("status") == STATUS, "receipt status mismatch")
    artifacts = receipt.get("artifacts")
    require(isinstance(artifacts, dict), "receipt artifacts missing")
    require(set(artifacts) == EXPECTED_OUTPUT_ARTIFACTS, "receipt artifact set drift")
    for relative, raw in artifacts.items():
        require(isinstance(raw, dict), "artifact contract is not a mapping")
        require(raw.get("relative_path") == relative, "artifact relative path drift")
        path = safe_file(output, relative, "output artifact")
        require(path.stat().st_size == raw.get("bytes"), "artifact byte count mismatch")
        require(
            re.fullmatch(r"[0-9a-f]{64}", str(raw.get("sha256"))) is not None,
            "artifact SHA is invalid",
        )
        require(sha256_file(path) == raw["sha256"], "artifact SHA mismatch")

    plan = read_json(output / "adjudication_plan.json", "plan")
    report = read_json(
        output / "work_topic_relation_primary_adjudication.json", "report"
    )
    state = read_json(output / "run_state.json", "state")
    sidecar = read_json(
        output / "work_topic_score_conflicts.primary_adjudicated.json", "sidecar"
    )
    require(plan.get("schema_version") == PLAN_SCHEMA, "plan schema mismatch")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(sidecar.get("schema_version") == SIDECAR_SCHEMA, "sidecar schema mismatch")
    require(state.get("schema_version") == STATE_SCHEMA, "state schema mismatch")
    require(
        report.get("status") == STATUS
        and state.get("status") == "COMPLETE_PENDING_RECEIPT",
        "terminal status mismatch",
    )
    require(plan.get("config_sha256") == CONFIG_SHA256, "plan config binding mismatch")
    require(
        plan.get("tool_sha256") == PRODUCER_SHA256, "plan producer binding mismatch"
    )
    require(
        plan.get("parent_receipt_sha256") == PARENT_RECEIPT_SHA256,
        "plan parent binding mismatch",
    )
    require(
        plan.get("parent_result_fingerprint_sha256") == PARENT_RESULT_FINGERPRINT,
        "plan parent fingerprint mismatch",
    )
    plan_copy = json.loads(json.dumps(plan))
    plan_fingerprint = plan_copy.pop("task_fingerprint_sha256", None)
    require(
        sha256_bytes(canonical_json_bytes(plan_copy)) == plan_fingerprint,
        "task fingerprint mismatch",
    )
    report_copy = json.loads(json.dumps(report))
    result_fingerprint = report_copy.get("provenance", {}).pop(
        "result_fingerprint_sha256", None
    )
    require(
        sha256_bytes(canonical_json_bytes(report_copy)) == result_fingerprint,
        "result fingerprint mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256") == result_fingerprint,
        "receipt fingerprint mismatch",
    )
    require(
        state.get("task_fingerprint_sha256") == plan_fingerprint,
        "state task fingerprint mismatch",
    )
    require(
        state.get("result_fingerprint_sha256") == result_fingerprint,
        "state result fingerprint mismatch",
    )
    require(
        report.get("provenance", {}).get("config_sha256") == CONFIG_SHA256,
        "report config binding mismatch",
    )
    require(
        report.get("provenance", {}).get("tool_sha256") == PRODUCER_SHA256,
        "report producer binding mismatch",
    )
    require(
        report.get("blocked_parent", {}).get("receipt_sha256") == PARENT_RECEIPT_SHA256,
        "report parent binding mismatch",
    )
    require(
        report.get("blocked_parent", {}).get("every_receipt_artifact_rehashed") is True,
        "parent rehash declaration missing",
    )
    require(
        all(
            prefix not in json.dumps(report) for prefix in ("/home/", "/hdd/", "/ssd/")
        ),
        "report leaks operator path",
    )
    relation = report.get("canonical_relation_artifact", {})
    require(
        relation.get("rows") == EXPECTED_COUNTS["unique_work_topic_edges"],
        "relation row count mismatch",
    )
    require(
        relation.get("bytes")
        == EXPECTED_COUNTS["unique_work_topic_edges"] * RELATION_DTYPE.itemsize,
        "relation byte count mismatch",
    )
    _verify_sidecar(sidecar, report)
    return report, receipt


def _verify_sidecar(sidecar: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    require(
        sidecar.get("status")
        == "PASS_ALL_PARENT_SCORE_CONFLICTS_PRIMARY_EXACT_ADJUDICATED",
        "sidecar status mismatch",
    )
    require(sidecar.get("conflict_group_count") == 2, "sidecar conflict count mismatch")
    require(
        sidecar.get("adjudicated_group_count") == 2,
        "sidecar adjudicated count mismatch",
    )
    require(sidecar.get("unresolved_group_count") == 0, "sidecar unresolved conflicts")
    conflicts = sidecar.get("conflicts")
    require(
        isinstance(conflicts, list) and len(conflicts) == 2, "sidecar conflicts drift"
    )
    for conflict in conflicts:
        require(isinstance(conflict, dict), "conflict is not a mapping")
        key = (int(conflict.get("work_numeric_id")) << TOPIC_BITS) | int(
            conflict.get("topic_numeric_id")
        )
        require(key in EXPECTED_CONFLICTS, "unexpected sidecar conflict")
        expected = EXPECTED_CONFLICTS[key]
        require(
            conflict.get("canonical_score") == expected["canonical_score"],
            "canonical score mismatch",
        )
        require(
            conflict.get("canonical_score_bits_hex")
            == expected["canonical_score_bits_hex"],
            "canonical bits mismatch",
        )
        main = conflict.get("main_primary_readback", {})
        for field, value in expected["main"].items():
            require(main.get(field) == value, f"main readback drift: {field}")
        require(
            conflict.get("raw_topic_rows") == expected["topic_rows"],
            "raw topic sidecar drift",
        )
    require(
        report.get("conflict_sidecar_summary") == conflicts,
        "report/sidecar conflict drift",
    )


def _verify_parent_controls(parent: Path) -> dict[str, Any]:
    require(parent.is_dir() and not parent.is_symlink(), "parent is not a directory")
    receipt_path = safe_file(
        parent, "work_topic_relation_audit.outputs.json", "parent receipt"
    )
    require(
        sha256_file(receipt_path) == PARENT_RECEIPT_SHA256,
        "parent receipt SHA mismatch",
    )
    receipt = read_json(receipt_path, "parent receipt")
    require(
        receipt.get("status") == "BLOCKED_WORK_TOPIC_RELATION_COMPONENT",
        "parent status mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256") == PARENT_RESULT_FINGERPRINT,
        "parent result fingerprint mismatch",
    )
    artifacts = receipt.get("artifacts", {})
    for relative, expected in PARENT_ARTIFACTS.items():
        require(
            artifacts.get(relative, {}).get("bytes") == expected["bytes"],
            "parent artifact bytes drift",
        )
        require(
            artifacts.get(relative, {}).get("sha256") == expected["sha256"],
            "parent artifact contract drift",
        )
        path = safe_file(parent, relative, "parent artifact")
        require(
            path.stat().st_size == expected["bytes"], "parent artifact size mismatch"
        )
        require(sha256_file(path) == expected["sha256"], "parent artifact SHA mismatch")
    return read_json(parent / "audit_plan.json", "parent plan")


def _read_source_row(path: Path, file_row: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    offset = 0
    for row_group in range(parquet.metadata.num_row_groups):
        rows = parquet.metadata.row_group(row_group).num_rows
        if offset <= file_row < offset + rows:
            table = parquet.read_row_group(
                row_group,
                columns=["id", "primary_topic__id", "primary_topic__score"],
            )
            row = table.slice(file_row - offset, 1).to_pylist()[0]
            row["row_group"] = row_group
            row["row_group_row"] = file_row - offset
            return row
        offset += rows
    raise VerificationError("source row exceeds footer")


def _verify_source_rows(source: Path, parent_plan: Mapping[str, Any]) -> None:
    tasks = {
        task["relative_path"]: task
        for task in parent_plan.get("tasks", [])
        if task.get("table_key") == "main"
    }
    for expected in EXPECTED_CONFLICTS.values():
        main = expected["main"]
        relative = str(main["relative_path"])
        require(relative in tasks, "main task absent from parent plan")
        path = safe_file(source, relative, "main source")
        require(
            rich_identity(path) == tasks[relative].get("source_rich_identity"),
            "main source identity mismatch",
        )
        row = _read_source_row(path, int(main["file_row"]))
        require(
            row["id"] == f"https://openalex.org/W{expected['work_numeric_id']}",
            "source work ID mismatch",
        )
        require(
            row["primary_topic__id"]
            == f"https://openalex.org/T{expected['topic_numeric_id']}",
            "source primary topic mismatch",
        )
        require(
            row["primary_topic__score"] == expected["canonical_score"],
            "source primary score mismatch",
        )
        require(row["row_group"] == main["row_group"], "source row group mismatch")
        require(
            row["row_group_row"] == main["row_group_row"],
            "source row-group row mismatch",
        )


def _event_blocks(events: np.memmap, chunk_rows: int) -> Iterator[np.ndarray]:
    carry = np.empty(0, dtype=EVENT_DTYPE)
    offset = 0
    while offset < len(events):
        end = min(len(events), offset + chunk_rows)
        block = np.array(events[offset:end], dtype=EVENT_DTYPE, copy=True)
        if len(carry):
            block = np.concatenate([carry, block])
        boundaries = np.flatnonzero(block["key"][1:] != block["key"][:-1]) + 1
        if end < len(events):
            require(
                len(boundaries) > 0, "verification chunk is smaller than one key group"
            )
            cut = int(boundaries[-1])
            complete, carry = block[:cut], block[cut:]
        else:
            complete, carry = block, np.empty(0, dtype=EVENT_DTYPE)
        if len(complete):
            yield complete
        offset = end
    require(len(carry) == 0, "verification carry remains")


def verify_payload_equivalence(
    *,
    parent_events_path: Path,
    relation_path: Path,
    expected_counts: Mapping[str, int],
    selected_scores: Mapping[int, int],
    chunk_rows: int,
) -> dict[str, int]:
    event_rows = int(expected_counts["event_rows"])
    relation_rows = int(expected_counts["unique_work_topic_edges"])
    require(
        parent_events_path.stat().st_size == event_rows * EVENT_DTYPE.itemsize,
        "parent event byte count mismatch",
    )
    require(
        relation_path.stat().st_size == relation_rows * RELATION_DTYPE.itemsize,
        "relation byte count mismatch",
    )
    events = np.memmap(
        parent_events_path, dtype=EVENT_DTYPE, mode="r", shape=(event_rows,)
    )
    relation = np.memmap(
        relation_path, dtype=RELATION_DTYPE, mode="r", shape=(relation_rows,)
    )
    relation_offset = 0
    topic_raw_rows = 0
    duplicate_groups = 0
    duplicate_excess = 0
    conflict_keys: set[int] = set()
    try:
        for block in _event_blocks(events, chunk_rows):
            require(
                np.all((block["kind"] == 0) | (block["kind"] == 1)),
                "invalid event kind",
            )
            keys = block["key"]
            starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
            ends = np.r_[starts[1:], len(block)]
            lengths = ends - starts
            main_counts = np.add.reduceat((block["kind"] == 0).astype(np.int64), starts)
            require(np.all(main_counts <= 1), "multiple main markers in key group")
            topic_counts = lengths - main_counts
            mask = topic_counts > 0
            if not np.any(mask):
                continue
            group_keys = keys[starts[mask]]
            first = block["score_bits"][starts[mask] + main_counts[mask]]
            last = block["score_bits"][ends[mask] - 1]
            expected_scores = np.array(first, copy=True)
            for index in np.flatnonzero(first != last):
                key = int(group_keys[index])
                require(key in selected_scores, "unexpected payload score conflict")
                expected_scores[index] = np.uint64(selected_scores[key])
                conflict_keys.add(key)
            observed = relation[relation_offset : relation_offset + len(group_keys)]
            require(len(observed) == len(group_keys), "relation ended early")
            require(
                np.array_equal(observed["key"], group_keys),
                "relation key differs from parent set view",
            )
            require(
                np.array_equal(observed["score_bits"], expected_scores),
                "relation score differs from adjudicated parent view",
            )
            selected_counts = topic_counts[mask]
            relation_offset += len(group_keys)
            topic_raw_rows += int(np.sum(selected_counts, dtype=np.int64))
            duplicate_groups += int(np.count_nonzero(selected_counts > 1))
            duplicate_excess += int(
                np.sum(np.maximum(selected_counts - 1, 0), dtype=np.int64)
            )
    finally:
        del relation
        del events
    observed_counts = {
        "topic_raw_rows": topic_raw_rows,
        "unique_work_topic_edges": relation_offset,
        "duplicate_edge_groups": duplicate_groups,
        "duplicate_row_excess": duplicate_excess,
        "score_conflict_edge_groups": len(conflict_keys),
    }
    for field, observed in observed_counts.items():
        require(observed == expected_counts[field], f"payload count mismatch: {field}")
    require(conflict_keys == set(selected_scores), "payload conflict key set mismatch")
    return observed_counts


def verify(
    *, repo_root: Path, parent: Path, source: Path, output: Path, chunk_rows: int
) -> dict[str, Any]:
    _verify_repository_bindings(repo_root)
    report, receipt = _verify_output_controls(output)
    parent_plan = _verify_parent_controls(parent)
    _verify_source_rows(source, parent_plan)
    selected = {
        key: int(str(value["canonical_score_bits_hex"]), 16)
        for key, value in EXPECTED_CONFLICTS.items()
    }
    counts = verify_payload_equivalence(
        parent_events_path=parent / "work_topic_relation.events.sorted.bin",
        relation_path=output / "work_topic_edges.primary_adjudicated.sorted.bin",
        expected_counts=EXPECTED_COUNTS,
        selected_scores=selected,
        chunk_rows=chunk_rows,
    )
    return {
        "schema_version": VERIFIER_SCHEMA,
        "status": VERIFIER_STATUS,
        "output_receipt_sha256": sha256_file(output / OUTPUT_RECEIPT),
        "result_fingerprint_sha256": receipt["result_fingerprint_sha256"],
        "canonical_relation_sha256": report["canonical_relation_artifact"]["sha256"],
        "parent_receipt_sha256": PARENT_RECEIPT_SHA256,
        "source_primary_rows_replayed": len(EXPECTED_CONFLICTS),
        "payload_equivalence": counts,
        "gpu_used": False,
        "claim_boundary": (
            "Read-only implementation-level verification of the frozen parent, "
            "two exact primary-score decisions, and the complete canonical relation "
            "payload. This does not pass G-1, G0, empirical G1, or a scientific claim."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--parent-output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=4_194_304)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require(args.chunk_rows > 1, "chunk rows must exceed one")
    result = verify(
        repo_root=args.repo_root.resolve(),
        parent=args.parent_output_dir.resolve(),
        source=args.source_root.resolve(),
        output=args.output_dir.resolve(),
        chunk_rows=args.chunk_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, OSError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
