#!/usr/bin/env python3
"""Resolve the bounded OpenAlex work-topic score conflicts from a frozen parent.

This successor never reruns topic classification.  It validates the exact
blocked full-scope relation receipt, reads the frozen main row located by each
conflicting edge's main marker, and selects a score only when the main
``primary_topic`` ID and exact canonical score match one of that edge's child
rows.  Every conflicting raw row is retained in a receipt-bound JSON sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/openalex_g1_work_topic_relation_primary_adjudication_v1.yaml"
)

CONFIG_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_config.v1"
PLAN_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_plan.v1"
SIDECAR_SCHEMA = "openalex_g1_work_topic_score_conflict_sidecar.v1"
REPORT_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_report.v1"
OUTPUTS_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_outputs.v1"
STATE_SCHEMA = "openalex_g1_work_topic_relation_primary_adjudication_state.v1"

# Patched only after the reviewed YAML bytes are final. Tests replace this
# binding with their closed synthetic config digest.
PRODUCTION_CONFIG_SHA256 = (
    "4bdc6caee203450b367f1e7621c4785373063edce93318e295aff5fd49752165"
)
FROZEN_CONFIG_SHA256 = PRODUCTION_CONFIG_SHA256

FULL_STATUS = (
    "PASS_CURRENT_SNAPSHOT_WORK_TOPIC_RELATION_COMPONENT_"
    "WITH_PRIMARY_EXACT_SCORE_ADJUDICATION"
)

MAIN_KIND = np.uint8(0)
TOPIC_KIND = np.uint8(1)
TOPIC_BITS = 14
LOCATOR_ROW_BITS = 20
LOCATOR_ROW_MASK = (1 << LOCATOR_ROW_BITS) - 1

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
DEDUP_DTYPE = np.dtype([("key", "<u8"), ("score_bits", "<u8")], align=False)
assert EVENT_DTYPE.itemsize == 24
assert DEDUP_DTYPE.itemsize == 16


class AdjudicationError(RuntimeError):
    """Raised when a frozen evidence or adjudication contract fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AdjudicationError(message)


def require_mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be a mapping")
    return dict(value)


def require_list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be a list")
    return list(value)


def require_int(value: Any, field: str) -> int:
    require(type(value) is int, f"{field} must be an integer")
    return int(value)


def require_sha256(value: Any, field: str) -> str:
    text = str(value)
    require(re.fullmatch(r"[0-9a-f]{64}", text) is not None, f"{field} is not SHA-256")
    return text


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def published_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, *, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    require(
        path.is_file() and not path.is_symlink(), "hash input is not a regular file"
    )
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        require(stat.S_ISREG(info.st_mode), "hash input is not regular")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            while block := handle.read(chunk_bytes):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def read_json(path: Path, field: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{field} is not a regular file")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            return require_mapping(json.load(handle), field)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdjudicationError(f"cannot read {field}") from exc


def safe_relative_path(value: Any, field: str) -> Path:
    path = Path(str(value))
    require(not path.is_absolute(), f"{field} must be relative")
    require(path.parts and path != Path("."), f"{field} may not be empty")
    require(".." not in path.parts, f"{field} may not contain '..'")
    return path


def rich_identity(path: Path, field: str) -> dict[str, int]:
    require(path.is_file() and not path.is_symlink(), f"{field} is not regular")
    info = path.stat(follow_symlinks=False)
    require(stat.S_ISREG(info.st_mode), f"{field} is not regular")
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


def _assert_directory_chain(path: Path, *, field: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if not current.exists() and not current.is_symlink():
            break
        info = current.lstat()
        require(not stat.S_ISLNK(info.st_mode), f"{field} contains a symlink")


def secure_join(root: Path, relative: Path, *, field: str) -> Path:
    _assert_directory_chain(root, field=field)
    candidate = root / relative
    _assert_directory_chain(candidate, field=field)
    require(
        candidate.is_file() and not candidate.is_symlink(), f"{field} is not regular"
    )
    return candidate


def atomic_write(path: Path, payload: bytes) -> None:
    require(
        path.parent.is_dir() and not path.parent.is_symlink(), "output parent invalid"
    )
    temporary = path.parent / f".{path.name}.tmp-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    os.chmod(path, 0o600)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write(path, published_json_bytes(value))


def atomic_write_text(path: Path, value: str) -> None:
    atomic_write(path, value.encode("utf-8"))


def _validate_config(config: Mapping[str, Any]) -> None:
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    parent = require_mapping(config.get("blocked_parent"), "blocked parent")
    require(
        parent.get("required_status") == "BLOCKED_WORK_TOPIC_RELATION_COMPONENT",
        "parent status drift",
    )
    require(
        parent.get("verify_every_receipt_artifact") is True,
        "parent rehash policy drift",
    )
    counts = require_mapping(parent.get("required_full_counts"), "parent counts")
    require(counts.get("score_conflict_edge_groups") == 2, "conflict count drift")
    expected = require_list(parent.get("expected_conflict_keys"), "conflict keys")
    require(
        len(expected) == counts["score_conflict_edge_groups"],
        "conflict key count drift",
    )
    abi = require_mapping(config.get("event_abi"), "event ABI")
    require(abi.get("bytes_per_event") == EVENT_DTYPE.itemsize, "event size drift")
    require(
        abi.get("sort_order") == ["key", "kind", "score_bits", "locator"],
        "sort order drift",
    )
    require(abi.get("topic_bits") == TOPIC_BITS, "topic bits drift")
    require(abi.get("locator_row_bits") == LOCATOR_ROW_BITS, "locator bits drift")
    policy = require_mapping(config.get("adjudication_policy"), "adjudication policy")
    require(
        policy.get("forbid_unconditional_first_max_mean_or_min") is True,
        "unsafe score policy",
    )
    require(
        policy.get("require_every_parent_conflict_observed") is True,
        "conflict completeness drift",
    )
    require(
        policy.get("require_no_unexpected_conflict") is True,
        "unexpected conflict policy drift",
    )
    require(
        policy.get("preserve_every_conflict_row_in_sidecar") is True,
        "sidecar policy drift",
    )
    require(
        policy.get("unresolved_conflict_policy")
        == "block_without_publishing_deduplicated_relation",
        "unresolved conflict policy drift",
    )
    success = require_mapping(config.get("success_contract"), "success contract")
    require(success.get("full_status") == FULL_STATUS, "success status drift")


def read_config(path: Path) -> tuple[dict[str, Any], str, int]:
    require(path.is_file() and not path.is_symlink(), "config is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
    finally:
        os.close(descriptor)
    observed = sha256_bytes(payload)
    require(
        observed == FROZEN_CONFIG_SHA256, "config SHA-256 differs from frozen contract"
    )
    try:
        config = require_mapping(yaml.safe_load(payload.decode("utf-8")), "config")
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise AdjudicationError("cannot parse config") from exc
    _validate_config(config)
    return config, observed, len(payload)


def resolve_path(explicit: Path | None, env_name: str, *, field: str) -> Path:
    raw = explicit if explicit is not None else os.environ.get(env_name)
    require(raw not in {None, ""}, f"{field} path is missing")
    path = Path(str(raw)).expanduser().resolve()
    require(path.is_dir() and not path.is_symlink(), f"{field} is not a directory")
    _assert_directory_chain(path, field=field)
    return path


def _expected_conflict_keys(config: Mapping[str, Any]) -> list[int]:
    parent = require_mapping(config.get("blocked_parent"), "blocked parent")
    values = []
    for raw in require_list(parent.get("expected_conflict_keys"), "conflict keys"):
        item = require_mapping(raw, "conflict key")
        work = require_int(item.get("work_numeric_id"), "work numeric ID")
        topic = require_int(item.get("topic_numeric_id"), "topic numeric ID")
        require(
            work >= 0 and 0 < topic < (1 << TOPIC_BITS), "conflict key out of range"
        )
        values.append((work << TOPIC_BITS) | topic)
    require(values == sorted(set(values)), "conflict keys must be unique and sorted")
    return values


def _validate_parent(
    config: Mapping[str, Any], parent_dir: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = require_mapping(config.get("blocked_parent"), "blocked parent")
    receipt_path = secure_join(
        parent_dir, Path(str(contract.get("receipt_basename"))), field="parent receipt"
    )
    require(
        sha256_file(receipt_path) == contract.get("receipt_sha256"),
        "parent receipt SHA mismatch",
    )
    receipt = read_json(receipt_path, "parent receipt")
    require(
        receipt.get("schema_version") == contract.get("receipt_schema_version"),
        "parent receipt schema mismatch",
    )
    require(
        receipt.get("status") == contract.get("required_status"),
        "parent receipt status mismatch",
    )
    require(
        receipt.get("result_fingerprint_sha256")
        == contract.get("result_fingerprint_sha256"),
        "parent result fingerprint mismatch",
    )
    checkpoint = require_mapping(
        receipt.get("checkpoint_evidence"), "parent checkpoint evidence"
    )
    require(
        checkpoint.get("checkpoint_count") == contract.get("checkpoint_count"),
        "parent checkpoint count mismatch",
    )
    require(
        checkpoint.get("checkpoint_tree_sha256")
        == contract.get("checkpoint_tree_sha256"),
        "parent checkpoint tree mismatch",
    )

    artifacts = require_mapping(receipt.get("artifacts"), "parent artifacts")
    require(
        len(artifacts)
        == require_int(
            contract.get("receipt_artifact_count"), "receipt artifact count"
        ),
        "parent artifact count mismatch",
    )
    for key in sorted(artifacts):
        item = require_mapping(artifacts[key], f"parent artifact {key}")
        relative = safe_relative_path(item.get("relative_path"), "parent artifact path")
        require(relative.as_posix() == key, "parent artifact key/path mismatch")
        path = secure_join(parent_dir, relative, field="parent artifact")
        identity = rich_identity(path, "parent artifact")
        require(
            identity["bytes"] == require_int(item.get("bytes"), "artifact bytes"),
            "parent artifact size mismatch",
        )
        require(
            sha256_file(path) == require_sha256(item.get("sha256"), "artifact SHA"),
            "parent artifact SHA mismatch",
        )

    for relative_name, raw in require_mapping(
        contract.get("required_artifacts"), "required parent artifacts"
    ).items():
        expected = require_mapping(raw, "required parent artifact")
        actual = require_mapping(
            artifacts.get(relative_name), "parent artifact contract"
        )
        require(
            actual.get("bytes") == expected.get("bytes"),
            "required parent artifact bytes drift",
        )
        require(
            actual.get("sha256") == expected.get("sha256"),
            "required parent artifact SHA drift",
        )

    plan = read_json(
        secure_join(parent_dir, Path("audit_plan.json"), field="parent plan"),
        "parent plan",
    )
    report = read_json(
        secure_join(
            parent_dir,
            Path("work_topic_relation_audit.json"),
            field="parent report",
        ),
        "parent report",
    )
    sort_manifest = read_json(
        secure_join(
            parent_dir, Path("sort_manifest.json"), field="parent sort manifest"
        ),
        "parent sort manifest",
    )
    require(
        plan.get("schema_version") == "openalex_g1_work_topic_relation_plan.v1",
        "parent plan schema mismatch",
    )
    require(plan.get("full_scope") is True, "parent is not full scope")
    require(
        plan.get("task_fingerprint_sha256") == contract.get("task_fingerprint_sha256"),
        "parent task fingerprint mismatch",
    )
    require(
        plan.get("config_sha256") == contract.get("config_sha256"),
        "parent config binding mismatch",
    )
    require(
        plan.get("tool_sha256") == contract.get("tool_sha256"),
        "parent tool binding mismatch",
    )
    require(
        report.get("status") == contract.get("required_status"),
        "parent report status mismatch",
    )
    require(
        report.get("provenance", {}).get("result_fingerprint_sha256")
        == contract.get("result_fingerprint_sha256"),
        "parent report fingerprint mismatch",
    )
    require(
        sort_manifest.get("event_rows")
        == contract["required_full_counts"]["event_rows"],
        "parent event rows mismatch",
    )
    require(
        sort_manifest.get("sort_order") == ["key", "kind", "score_bits", "locator"],
        "parent sort order mismatch",
    )

    edge = require_mapping(
        report.get("accounting", {})
        .get("edge_groups", {})
        .get("grammar_valid_including_w0"),
        "parent edge accounting",
    )
    for key, expected in require_mapping(
        contract.get("required_full_counts"), "required full counts"
    ).items():
        if key == "event_rows":
            continue
        require(edge.get(key) == expected, f"parent count mismatch: {key}")
    require(
        report.get("gate_status", {}).get("primary_topic_child_inclusion") == "PASS",
        "parent primary-child gate mismatch",
    )
    require(
        report.get("gate_status", {}).get("frozen_exact_score_dedup_view")
        == "BLOCKED_SCORE_CONFLICT",
        "parent conflict gate mismatch",
    )
    require(
        report.get("deduplicated_relation_artifact", {}).get("status")
        == "NOT_MATERIALIZED_SCORE_CONFLICT",
        "parent dedup state mismatch",
    )
    samples = require_list(
        report.get("accounting", {})
        .get("edge_groups", {})
        .get("score_conflict_samples"),
        "parent conflict samples",
    )
    sample_keys = sorted(
        (require_int(x["work_numeric_id"], "sample work") << TOPIC_BITS)
        | require_int(x["topic_numeric_id"], "sample topic")
        for x in samples
    )
    require(
        sample_keys == _expected_conflict_keys(config),
        "parent conflict samples are not exhaustive expected keys",
    )
    return receipt, plan, report, sort_manifest


def _score_bits(value: float) -> int:
    require(math.isfinite(value), "primary score is not finite")
    normalized = 0.0 if value == 0.0 else value
    return int(np.asarray([normalized], dtype="<f8").view("<u8")[0])


def _score_from_bits(value: int) -> float:
    return float(np.asarray([value], dtype="<u8").view("<f8")[0])


def _task_paths(plan: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {"main": [], "topics": []}
    for raw in require_list(plan.get("tasks"), "parent tasks"):
        task = require_mapping(raw, "parent task")
        key = str(task.get("table_key"))
        require(key in grouped, "unknown parent task table")
        require(
            task.get("file_index") == len(grouped[key]), "parent task file index drift"
        )
        grouped[key].append(task)
    return grouped


def _decode_locator(
    locator: int, *, table_key: str, tasks: Mapping[str, list[dict[str, Any]]]
) -> tuple[dict[str, Any], int]:
    file_index = locator >> LOCATOR_ROW_BITS
    file_row = locator & LOCATOR_ROW_MASK
    table_tasks = tasks[table_key]
    require(0 <= file_index < len(table_tasks), "locator file index invalid")
    return table_tasks[file_index], file_row


def _read_main_row(
    source_root: Path,
    task: Mapping[str, Any],
    file_row: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    relative = safe_relative_path(task.get("relative_path"), "main source path")
    path = secure_join(source_root, relative, field="main source Parquet")
    require(
        rich_identity(path, "main source Parquet") == task.get("source_rich_identity"),
        "main source identity differs from parent plan",
    )
    columns = require_list(
        config.get("primary_readback", {}).get("columns"), "readback columns"
    )
    parquet = pq.ParquetFile(path)
    require(
        parquet.metadata.num_rows == task.get("footer_rows"), "main footer rows differ"
    )
    expected_types = require_mapping(
        config.get("primary_readback", {}).get("expected_types"), "readback types"
    )
    schema = parquet.schema_arrow
    for name in columns:
        field = schema.field(name)
        require(
            str(field.type) == expected_types[name],
            f"main readback type mismatch: {name}",
        )
    offset = 0
    for row_group in range(parquet.metadata.num_row_groups):
        rows = parquet.metadata.row_group(row_group).num_rows
        if offset <= file_row < offset + rows:
            table = parquet.read_row_group(row_group, columns=columns)
            row = table.slice(file_row - offset, 1).to_pylist()[0]
            return {
                "relative_path": relative.as_posix(),
                "file_row": file_row,
                "row_group": row_group,
                "row_group_row": file_row - offset,
                **row,
            }
        offset += rows
    raise AdjudicationError("main locator row exceeds footer")


def _locate_group(events: np.memmap, key: int) -> np.ndarray:
    keys = events["key"]
    left = int(np.searchsorted(keys, np.uint64(key), side="left"))
    right = int(np.searchsorted(keys, np.uint64(key), side="right"))
    require(left < right, "expected conflict key is absent from sorted events")
    return np.array(events[left:right], dtype=EVENT_DTYPE, copy=True)


def adjudicate_conflicts(
    *,
    config: Mapping[str, Any],
    source_root: Path,
    parent_dir: Path,
    plan: Mapping[str, Any],
    sort_manifest: Mapping[str, Any],
) -> tuple[dict[int, int], dict[str, Any]]:
    sorted_path = secure_join(
        parent_dir,
        Path("work_topic_relation.events.sorted.bin"),
        field="parent sorted events",
    )
    event_rows = require_int(sort_manifest.get("event_rows"), "event rows")
    require(
        sorted_path.stat().st_size == event_rows * EVENT_DTYPE.itemsize,
        "sorted event size mismatch",
    )
    events = np.memmap(sorted_path, dtype=EVENT_DTYPE, mode="r", shape=(event_rows,))
    tasks = _task_paths(plan)
    readback = require_mapping(config.get("primary_readback"), "primary readback")
    minimum = float(readback.get("score_minimum_inclusive"))
    maximum = float(readback.get("score_maximum_inclusive"))
    adjudicated: dict[int, int] = {}
    conflicts: list[dict[str, Any]] = []
    try:
        for key in _expected_conflict_keys(config):
            group = _locate_group(events, key)
            main_rows = group[group["kind"] == MAIN_KIND]
            topic_rows = group[group["kind"] == TOPIC_KIND]
            require(len(main_rows) == 1, "conflict key lacks exactly one main marker")
            require(len(topic_rows) >= 2, "conflict key lacks duplicate topic rows")
            variants = sorted(set(int(x) for x in topic_rows["score_bits"]))
            require(len(variants) >= 2, "expected conflict has only one score")
            task, file_row = _decode_locator(
                int(main_rows[0]["locator"]), table_key="main", tasks=tasks
            )
            main = _read_main_row(source_root, task, file_row, config)
            work_numeric = key >> TOPIC_BITS
            topic_numeric = key & ((1 << TOPIC_BITS) - 1)
            require(
                main["id"] == f"{readback['work_prefix']}{work_numeric}",
                "main work ID differs from conflict key",
            )
            require(
                main["primary_topic__id"]
                == f"{readback['topic_prefix']}{topic_numeric}",
                "main primary topic differs from conflict key",
            )
            score = main.get("primary_topic__score")
            require(isinstance(score, float), "main primary score is not float64")
            require(minimum <= score <= maximum, "main primary score is out of range")
            canonical_bits = _score_bits(score)
            require(
                canonical_bits in variants,
                "main primary score is absent from child variants",
            )
            adjudicated[key] = canonical_bits
            raw_topic_rows = []
            for row in topic_rows:
                topic_task, topic_file_row = _decode_locator(
                    int(row["locator"]), table_key="topics", tasks=tasks
                )
                bits = int(row["score_bits"])
                raw_topic_rows.append(
                    {
                        "relative_path": str(topic_task["relative_path"]),
                        "file_row": topic_file_row,
                        "score": _score_from_bits(bits),
                        "score_bits_hex": f"{bits:016x}",
                        "selected": bits == canonical_bits,
                    }
                )
            conflicts.append(
                {
                    "work_numeric_id": work_numeric,
                    "topic_numeric_id": topic_numeric,
                    "main_primary_readback": {
                        "relative_path": main["relative_path"],
                        "file_row": main["file_row"],
                        "row_group": main["row_group"],
                        "row_group_row": main["row_group_row"],
                        "primary_topic_score": score,
                        "score_bits_hex": f"{canonical_bits:016x}",
                    },
                    "raw_topic_rows": raw_topic_rows,
                    "distinct_score_count": len(variants),
                    "canonical_score": score,
                    "canonical_score_bits_hex": f"{canonical_bits:016x}",
                    "decision": "SELECT_EXACT_FROZEN_MAIN_PRIMARY_SCORE",
                }
            )
    finally:
        del events
    sidecar = {
        "schema_version": SIDECAR_SCHEMA,
        "status": "PASS_ALL_PARENT_SCORE_CONFLICTS_PRIMARY_EXACT_ADJUDICATED",
        "policy": "unique_main_marker_and_exact_primary_id_score_child_match",
        "conflict_group_count": len(conflicts),
        "adjudicated_group_count": len(adjudicated),
        "unresolved_group_count": 0,
        "conflicts": conflicts,
    }
    return adjudicated, sidecar


def _group_blocks(
    events: np.memmap, *, chunk_rows: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    carry = np.empty(0, dtype=EVENT_DTYPE)
    offset = 0
    total = len(events)
    while offset < total:
        end = min(total, offset + chunk_rows)
        block = np.array(events[offset:end], dtype=EVENT_DTYPE, copy=True)
        if len(carry):
            block = np.concatenate([carry, block])
        keys = block["key"]
        boundaries = np.flatnonzero(keys[1:] != keys[:-1]) + 1
        if end < total:
            require(
                len(boundaries) > 0, "aggregation chunk cannot contain one giant key"
            )
            cut = int(boundaries[-1])
            complete = block[:cut]
            carry = block[cut:]
        else:
            complete = block
            carry = np.empty(0, dtype=EVENT_DTYPE)
        if len(complete):
            complete_keys = complete["key"]
            starts = np.r_[
                0, np.flatnonzero(complete_keys[1:] != complete_keys[:-1]) + 1
            ]
            ends = np.r_[starts[1:], len(complete)]
            yield complete, complete_keys[starts], starts, ends
        offset = end
    require(len(carry) == 0, "aggregation carry was not consumed")


def materialize_relation(
    *,
    config: Mapping[str, Any],
    parent_dir: Path,
    output_dir: Path,
    sort_manifest: Mapping[str, Any],
    adjudicated: Mapping[int, int],
) -> dict[str, Any]:
    sorted_path = secure_join(
        parent_dir,
        Path("work_topic_relation.events.sorted.bin"),
        field="parent sorted events",
    )
    event_rows = require_int(sort_manifest.get("event_rows"), "event rows")
    events = np.memmap(sorted_path, dtype=EVENT_DTYPE, mode="r", shape=(event_rows,))
    output_path = output_dir / "work_topic_edges.primary_adjudicated.sorted.bin"
    temporary = output_dir / f".{output_path.name}.tmp-{secrets.token_hex(8)}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    handle = os.fdopen(descriptor, "wb")
    expected_counts = require_mapping(
        config.get("blocked_parent", {}).get("required_full_counts"), "expected counts"
    )
    dedup_rows = 0
    topic_raw_rows = 0
    duplicate_groups = 0
    duplicate_excess = 0
    observed_conflicts: set[int] = set()
    publish = False
    try:
        chunk_rows = require_int(
            config.get("execution", {}).get("aggregation_chunk_rows"), "chunk rows"
        )
        for block, keys, starts, ends in _group_blocks(events, chunk_rows=chunk_rows):
            lengths = ends - starts
            is_main = block["kind"] == MAIN_KIND
            main_counts = np.add.reduceat(is_main.astype(np.int64), starts)
            topic_counts = lengths - main_counts
            topic_group = topic_counts > 0
            if not np.any(topic_group):
                continue
            group_indices = np.flatnonzero(topic_group)
            first_indices = starts[topic_group] + main_counts[topic_group]
            last_indices = ends[topic_group] - 1
            first_scores = block["score_bits"][first_indices]
            last_scores = block["score_bits"][last_indices]
            group_keys = keys[topic_group]
            conflicts = first_scores != last_scores
            for conflict_key in group_keys[conflicts]:
                numeric = int(conflict_key)
                require(numeric in adjudicated, "unexpected unresolved score conflict")
                observed_conflicts.add(numeric)
            records = np.empty(len(group_keys), dtype=DEDUP_DTYPE)
            records["key"] = group_keys
            records["score_bits"] = first_scores
            for key, score_bits in adjudicated.items():
                match = records["key"] == np.uint64(key)
                if np.any(match):
                    records["score_bits"][match] = np.uint64(score_bits)
            records.tofile(handle)
            dedup_rows += len(records)
            selected_counts = topic_counts[group_indices]
            topic_raw_rows += int(np.sum(selected_counts, dtype=np.int64))
            duplicate_groups += int(np.count_nonzero(selected_counts > 1))
            duplicate_excess += int(
                np.sum(np.maximum(selected_counts - 1, 0), dtype=np.int64)
            )
        require(
            observed_conflicts == set(adjudicated),
            "not every adjudicated conflict was observed",
        )
        require(
            dedup_rows == expected_counts["unique_work_topic_edges"],
            "dedup row count mismatch",
        )
        require(
            topic_raw_rows == expected_counts["topic_raw_rows"],
            "topic row count mismatch",
        )
        require(
            duplicate_groups == expected_counts["duplicate_edge_groups"],
            "duplicate group count mismatch",
        )
        require(
            duplicate_excess == expected_counts["duplicate_row_excess"],
            "duplicate excess mismatch",
        )
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, output_path)
        os.chmod(output_path, 0o600)
        publish = True
    finally:
        if not handle.closed:
            handle.close()
        if not publish and temporary.exists():
            temporary.unlink()
        del events
    identity = rich_identity(output_path, "canonical relation")
    require(
        identity["bytes"] == dedup_rows * DEDUP_DTYPE.itemsize,
        "dedup byte count mismatch",
    )
    return {
        "status": "PROMOTED_PRIMARY_EXACT_SCORE_ADJUDICATED_SET_VIEW",
        "relative_path": output_path.name,
        "rows": dedup_rows,
        "bytes": identity["bytes"],
        "sha256": sha256_file(output_path),
        "sort_order": ["packed_work_topic_key"],
        "record_dtype": [["key", "<u8"], ["score_bits", "<u8"]],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    evidence = report["adjudication_evidence"]
    relation = report["canonical_relation_artifact"]
    return "\n".join(
        [
            "# OpenAlex Work-Topic Primary-Score Adjudication",
            "",
            f"- Status: `{report['status']}`",
            f"- Parent status: `{report['blocked_parent']['status']}`",
            f"- Parent conflicts: `{evidence['parent_conflict_group_count']}`",
            f"- Adjudicated conflicts: `{evidence['adjudicated_group_count']}`",
            f"- Unresolved conflicts: `{evidence['unresolved_group_count']}`",
            f"- Canonical relation rows: `{relation['rows']:,}`",
            f"- Canonical relation SHA-256: `{relation['sha256']}`",
            "- GPU used: `false`",
            "- Topic classification rerun: `false`",
            "",
            "## Decisions",
            "",
            "| Work | Topic | Raw scores | Canonical score | Rule |",
            "| ---: | ---: | --- | ---: | --- |",
            *[
                "| W{work_numeric_id} | T{topic_numeric_id} | {scores} | {canonical_score:.4f} | main primary exact match |".format(
                    work_numeric_id=item["work_numeric_id"],
                    topic_numeric_id=item["topic_numeric_id"],
                    scores=", ".join(
                        f"{row['score']:.4f}" for row in item["raw_topic_rows"]
                    ),
                    canonical_score=item["canonical_score"],
                )
                for item in report["conflict_sidecar_summary"]
            ],
            "",
            "## Claim boundary",
            "",
            str(report["claim_boundary"]),
            "",
        ]
    )


def _artifact_contract(path: Path, output_dir: Path) -> dict[str, Any]:
    relative = path.relative_to(output_dir).as_posix()
    return {
        "relative_path": relative,
        "bytes": rich_identity(path, "output artifact")["bytes"],
        "sha256": sha256_file(path),
    }


def _validate_output_tree(output_dir: Path) -> None:
    require(
        stat.S_IMODE(output_dir.stat().st_mode) == 0o700,
        "output directory mode differs",
    )
    for path in output_dir.rglob("*"):
        require(not path.is_symlink(), "output tree contains a symlink")
        if path.is_dir():
            require(
                stat.S_IMODE(path.stat().st_mode) == 0o700,
                "output subdirectory mode differs",
            )
        elif path.is_file():
            require(
                stat.S_IMODE(path.stat().st_mode) == 0o600, "output file mode differs"
            )
        else:
            raise AdjudicationError("output tree contains a non-file entry")


def run(
    *,
    config_path: Path,
    parent_dir: Path | None,
    source_root: Path | None,
    output_dir: Path,
    allow_full_scope: bool,
) -> dict[str, Any]:
    require(allow_full_scope, "full-scope adjudication requires --allow-full-scope")
    config, config_sha, config_bytes = read_config(config_path.resolve())
    parent_contract = require_mapping(config.get("blocked_parent"), "blocked parent")
    source_contract = require_mapping(config.get("source"), "source")
    parent = resolve_path(
        parent_dir, str(parent_contract.get("output_dir_env")), field="parent output"
    )
    source = resolve_path(
        source_root, str(source_contract.get("root_env")), field="source root"
    )
    output = Path(os.path.abspath(os.fspath(output_dir.expanduser())))
    require(not output.exists(), "output directory must be fresh")
    require(
        not (output == parent or parent in output.parents or output in parent.parents),
        "output overlaps parent",
    )
    require(
        not (output == source or source in output.parents or output in source.parents),
        "output overlaps source",
    )
    output.mkdir(mode=0o700, parents=True)
    os.chmod(output, 0o700)
    _assert_directory_chain(output, field="output directory")

    receipt, parent_plan, parent_report, sort_manifest = _validate_parent(
        config, parent
    )
    adjudicated, sidecar = adjudicate_conflicts(
        config=config,
        source_root=source,
        parent_dir=parent,
        plan=parent_plan,
        sort_manifest=sort_manifest,
    )
    sidecar_path = output / "work_topic_score_conflicts.primary_adjudicated.json"
    atomic_write_json(sidecar_path, sidecar)

    required_rows = require_int(
        parent_contract.get("required_full_counts", {}).get("unique_work_topic_edges"),
        "unique edge rows",
    )
    required_bytes = required_rows * DEDUP_DTYPE.itemsize
    usage = shutil.disk_usage(output)
    reserve = require_int(
        config.get("execution", {}).get("disk_reserve_bytes"), "disk reserve"
    )
    require(usage.free >= required_bytes + reserve, "insufficient output disk")

    tool_path = Path(__file__).resolve()
    tool_sha = sha256_file(tool_path)
    plan_payload = {
        "schema_version": PLAN_SCHEMA,
        "label": config.get("label"),
        "config_sha256": config_sha,
        "config_bytes": config_bytes,
        "tool_sha256": tool_sha,
        "tool_relative_path": "scripts/adjudicate_openalex_work_topic_primary_scores.py",
        "parent_receipt_sha256": parent_contract.get("receipt_sha256"),
        "parent_result_fingerprint_sha256": receipt.get("result_fingerprint_sha256"),
        "parent_sorted_event_sha256": parent_contract["required_artifacts"][
            "work_topic_relation.events.sorted.bin"
        ]["sha256"],
        "source_root_id": source_contract.get("root_id"),
        "expected_conflict_keys": _expected_conflict_keys(config),
        "policy": config.get("adjudication_policy"),
        "full_scope": True,
        "gpu_used": False,
    }
    plan_payload["task_fingerprint_sha256"] = sha256_bytes(
        canonical_json_bytes(plan_payload)
    )
    plan_path = output / "adjudication_plan.json"
    atomic_write_json(plan_path, plan_payload)

    relation = materialize_relation(
        config=config,
        parent_dir=parent,
        output_dir=output,
        sort_manifest=sort_manifest,
        adjudicated=adjudicated,
    )
    sidecar_sha = sha256_file(sidecar_path)
    gates = dict(
        require_mapping(
            config.get("success_contract", {}).get("required_gate_states"),
            "required gate states",
        )
    )
    report_payload = {
        "schema_version": REPORT_SCHEMA,
        "label": config.get("label"),
        "status": FULL_STATUS,
        "blocked_parent": {
            "status": parent_report.get("status"),
            "receipt_sha256": parent_contract.get("receipt_sha256"),
            "result_fingerprint_sha256": receipt.get("result_fingerprint_sha256"),
            "every_receipt_artifact_rehashed": True,
            "receipt_artifact_count": len(receipt["artifacts"]),
        },
        "adjudication_evidence": {
            "parent_conflict_group_count": parent_contract["required_full_counts"][
                "score_conflict_edge_groups"
            ],
            "adjudicated_group_count": len(adjudicated),
            "unresolved_group_count": 0,
            "unexpected_group_count": 0,
            "sidecar_relative_path": sidecar_path.name,
            "sidecar_sha256": sidecar_sha,
        },
        "conflict_sidecar_summary": sidecar["conflicts"],
        "canonical_relation_artifact": relation,
        "gate_status": gates,
        "claim_boundary": config.get("claim_boundary"),
        "provenance": {
            "config_sha256": config_sha,
            "config_bytes": config_bytes,
            "tool_sha256": tool_sha,
            "tool_relative_path": "scripts/adjudicate_openalex_work_topic_primary_scores.py",
            "task_fingerprint_sha256": plan_payload["task_fingerprint_sha256"],
            "path_redacted": True,
        },
    }
    fingerprint_payload = json.loads(json.dumps(report_payload))
    result_fingerprint = sha256_bytes(canonical_json_bytes(fingerprint_payload))
    report_payload["provenance"]["result_fingerprint_sha256"] = result_fingerprint
    serialized = json.dumps(report_payload)
    require(
        all(prefix not in serialized for prefix in ("/home/", "/hdd/", "/ssd/")),
        "report leaked operator path",
    )

    report_path = output / "work_topic_relation_primary_adjudication.json"
    markdown_path = output / "work_topic_relation_primary_adjudication.md"
    state_path = output / "run_state.json"
    receipt_path = output / "work_topic_relation_primary_adjudication.outputs.json"
    atomic_write_json(report_path, report_payload)
    atomic_write_text(markdown_path, render_markdown(report_payload))
    atomic_write_json(
        state_path,
        {
            "schema_version": STATE_SCHEMA,
            "status": "COMPLETE_PENDING_RECEIPT",
            "completion_receipt_relative_path": receipt_path.name,
            "complete_only_if_receipt_exists": True,
            "task_fingerprint_sha256": plan_payload["task_fingerprint_sha256"],
            "result_fingerprint_sha256": result_fingerprint,
        },
    )
    artifact_paths = [
        plan_path,
        sidecar_path,
        output / relation["relative_path"],
        report_path,
        markdown_path,
        state_path,
    ]
    artifacts = {
        path.relative_to(output).as_posix(): _artifact_contract(path, output)
        for path in artifact_paths
    }
    output_receipt = {
        "schema_version": OUTPUTS_SCHEMA,
        "status": FULL_STATUS,
        "result_fingerprint_sha256": result_fingerprint,
        "artifacts": artifacts,
    }
    atomic_write_json(receipt_path, output_receipt)
    _validate_output_tree(output)
    disk_report = read_json(report_path, "published report")
    require(disk_report == report_payload, "published report differs")
    require(
        read_json(receipt_path, "published receipt") == output_receipt,
        "published receipt differs",
    )
    require(
        markdown_path.read_text(encoding="utf-8") == render_markdown(disk_report),
        "published Markdown differs",
    )
    for relative, contract in artifacts.items():
        path = secure_join(output, Path(relative), field="published artifact")
        require(
            path.stat().st_size == contract["bytes"], "published artifact size differs"
        )
        require(
            sha256_file(path) == contract["sha256"], "published artifact SHA differs"
        )
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-output-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-full-scope", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        raise AdjudicationError("execution requires --execute")
    report = run(
        config_path=args.config,
        parent_dir=args.parent_output_dir,
        source_root=args.source_root,
        output_dir=args.output_dir,
        allow_full_scope=args.allow_full_scope,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AdjudicationError, OSError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
