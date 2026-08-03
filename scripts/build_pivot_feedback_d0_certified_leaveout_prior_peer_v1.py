#!/usr/bin/env python3
"""Build exact-four incumbent, focal, and prior-target-peer leave-out measures."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import build_pivot_feedback_d0_certified_author_opportunity_rebuild_v1 as parent


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_certified_leaveout_prior_peer_v1.yaml"
)
CONFIG_SCHEMA = "pivot_feedback_d0_certified_leaveout_prior_peer_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_certified_leaveout_prior_peer_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_certified_leaveout_prior_peer_outputs.v1"
TARGET_SCHEMA_VERSION = "pivot_feedback_d0_certified_target_work_incumbency.v1"
EDGE_SCHEMA_VERSION = "pivot_feedback_d0_certified_prior_target_peer_edge.v1"
AUTHOR_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_author_leaveout_prior_peer_base.v1"
)
STATUS = (
    "PASS_EXACT_FOUR_OUTCOME_BLIND_INCUMBENT_FOCAL_AND_PRIOR_TARGET_PEER_LEAVEOUT"
)
PARTITIONS = 4096
CAPS = (10, 50, 100)
MAX_CONTROL_BYTES = 32 * 1024 * 1024
WINDOW_WORK_DTYPE = np.dtype(
    [
        ("work_numeric_id", "<u8"),
        ("episode_index", "u1"),
        ("publication_year", "<u2"),
        ("field_id", "u1"),
        ("declared_authors_count", "<u2"),
        ("temporal_disposition", "u1"),
    ],
    align=False,
)
WORK_AUTHOR_DTYPE = np.dtype(
    [
        ("episode_index", "u1"),
        ("work_numeric_id", "<u8"),
        ("author_numeric_id", "<u8"),
        ("publication_year", "<u2"),
        ("declared_authors_count", "<u2"),
    ],
    align=False,
)
PAIR_INCIDENCE_DTYPE = np.dtype(
    [
        ("episode_index", "u1"),
        ("candidate_author_id", "<u8"),
        ("peer_author_id", "<u8"),
        ("publication_year", "<u2"),
        ("declared_authors_count", "<u2"),
    ],
    align=False,
)
EDGE_DTYPE = np.dtype(
    [
        ("episode_index", "u1"),
        ("candidate_author_id", "<u8"),
        ("peer_author_id", "<u8"),
        ("minimum_declared_team_size", "<u2"),
        ("first_prior_collaboration_year", "<u2"),
        ("last_prior_collaboration_year", "<u2"),
        ("distinct_shared_work_count", "<u4"),
        ("peer_pre_signal_incumbent", "u1"),
    ],
    align=False,
)
EXCLUSION_DTYPE = np.dtype(
    [
        ("episode_index", "u1"),
        ("candidate_author_id", "<u8"),
        ("target_work_index", "<u4"),
        ("minimum_prior_tie_team_size", "<u2"),
    ],
    align=False,
)
TARGET_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=False),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("window_code", pa.uint8(), nullable=False),
        pa.field("valid_author_token_count", pa.uint32(), nullable=False),
        pa.field("pre_signal_incumbent_author_count", pa.uint32(), nullable=False),
        pa.field("has_pre_signal_incumbent_author", pa.bool_(), nullable=False),
        pa.field(
            "all_linked_authors_pre_signal_incumbents",
            pa.bool_(),
            nullable=False,
        ),
    ],
    metadata={b"schema_version": TARGET_SCHEMA_VERSION.encode()},
)
EDGE_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("candidate_author_id", pa.uint64(), nullable=False),
        pa.field("peer_author_id", pa.uint64(), nullable=False),
        pa.field("minimum_declared_team_size", pa.uint16(), nullable=False),
        pa.field("first_prior_collaboration_year", pa.uint16(), nullable=False),
        pa.field("last_prior_collaboration_year", pa.uint16(), nullable=False),
        pa.field("distinct_shared_work_count", pa.uint32(), nullable=False),
        pa.field("peer_pre_signal_incumbent", pa.bool_(), nullable=False),
    ],
    metadata={b"schema_version": EDGE_SCHEMA_VERSION.encode()},
)


def _author_schema() -> pa.Schema:
    fields = [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
    ]
    metrics = (
        "prior_target_active_peer_count",
        "prior_incumbent_peer_count",
        "prior_peer_target_work_count",
        "any_incumbent_attention_work_count",
        "any_incumbent_focal_leaveout_work_count",
        "any_incumbent_network_leaveout_work_count",
        "all_incumbent_attention_work_count",
        "all_incumbent_focal_leaveout_work_count",
        "all_incumbent_network_leaveout_work_count",
    )
    for cap in CAPS:
        for metric in metrics:
            fields.append(pa.field(f"{metric}_cap_{cap}", pa.uint32(), nullable=False))
        fields.append(
            pa.field(
                f"has_prior_target_active_peer_cap_{cap}",
                pa.bool_(),
                nullable=False,
            )
        )
    return pa.schema(
        fields,
        metadata={b"schema_version": AUTHOR_SCHEMA_VERSION.encode()},
    )


AUTHOR_OUTPUT_SCHEMA = _author_schema()


class CertifiedLeaveoutPriorPeerError(RuntimeError):
    """Fail-closed leave-out/prior-peer builder error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertifiedLeaveoutPriorPeerError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing field {dotted_path}",
        )
        current = current[part]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CertifiedLeaveoutPriorPeerError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise CertifiedLeaveoutPriorPeerError(f"invalid {label}: {error}") from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _validate_repo_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_repo_path(binding.get("relative_path"), "repository binding")
    observed_sha = _sha256(path)
    require(observed_sha == binding.get("sha256"), f"binding SHA mismatch: {path.name}")
    document = _read_yaml(path, path.name)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        f"binding schema mismatch: {path.name}",
    )
    require(
        document.get("status") == binding.get("required_status"),
        f"binding status mismatch: {path.name}",
    )
    for dotted, expected in _mapping(
        binding.get("required_fields", {}),
        "binding fields",
    ).items():
        observed = _nested(document, dotted)
        require(
            type(observed) is type(expected) and observed == expected,
            f"binding field mismatch: {dotted}",
        )
    return {
        "relative_path": binding["relative_path"],
        "sha256": observed_sha,
        "status": document["status"],
    }


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        relative = item.get("relative_path")
        require(isinstance(relative, str) and relative, "bad artifact path")
        require(relative not in result, "duplicate artifact path")
        result[relative] = item
    return result


def _load_parent(
    root: Path,
    binding: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid parent root")
    require(root.name == binding["root_basename"], "parent root basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "parent receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "parent receipt schema")
    require(receipt.get("status") == binding["receipt_status"], "parent receipt status")
    artifacts = _artifact_map(receipt)
    selected_path = root / str(binding["selected_work_basename"])
    author_path = root / str(binding["author_base_basename"])
    for path, sha_key in (
        (selected_path, "selected_work_sha256"),
        (author_path, "author_base_sha256"),
    ):
        require(_sha256(path) == binding[sha_key], f"parent artifact SHA: {path.name}")
        require(artifacts[path.name]["sha256"] == binding[sha_key], "parent receipt artifact differs")
    selected = pq.read_table(selected_path)
    authors = pq.read_table(
        author_path,
        columns=[
            "certified_episode_index",
            "field_id",
            "topic_id",
            "entry_year",
            "author_numeric_id",
        ],
    )
    require(
        (selected.schema.metadata or {}).get(b"schema_version", b"").decode()
        == binding["selected_work_schema_version"],
        "parent selected schema",
    )
    require(
        pq.ParquetFile(author_path).schema_arrow.metadata.get(
            b"schema_version",
            b"",
        ).decode()
        == binding["author_base_schema_version"],
        "parent author schema",
    )
    require(len(selected) == binding["selected_work_rows"], "parent selected rows")
    require(len(authors) == binding["author_base_rows"], "parent author rows")
    verification_path = root.parent / str(binding["verification_basename"])
    require(_sha256(verification_path) == binding["verification_sha256"], "parent verification SHA")
    verification = _read_json(verification_path, verification_path.name)
    require(
        verification.get("schema_version") == binding["verification_schema_version"],
        "parent verification schema",
    )
    require(
        verification.get("status") == binding["verification_status"],
        "parent verification status",
    )
    require(verification["scope"].get("outcomes_read") is False, "parent outcomes opened")
    return selected, authors, {
        "root_basename": root.name,
        "receipt_sha256": binding["receipt_sha256"],
        "selected_work_sha256": binding["selected_work_sha256"],
        "author_base_sha256": binding["author_base_sha256"],
        "verification_sha256": binding["verification_sha256"],
        "selected_work_rows": len(selected),
        "author_base_rows": len(authors),
    }


def _episodes(config: Mapping[str, Any]) -> list[dict[str, int]]:
    rows = [
        {key: int(value) if key != "prior_graph_window" else value for key, value in _mapping(raw, "episode").items()}
        for raw in _list(config.get("exact_episodes"), "episodes")
    ]
    require(
        [
            (
                row["certified_episode_index"],
                row["field_id"],
                row["field_candidate_rank"],
                row["topic_id"],
                row["entry_year"],
                row["prior_graph_window"],
            )
            for row in rows
        ]
        == [
            (1, 17, 1, 14430, 2003, [1998, 2002]),
            (2, 17, 45, 10579, 2012, [2007, 2011]),
            (3, 27, 5, 10151, 1993, [1988, 1992]),
            (4, 27, 6, 12421, 1953, [1948, 1952]),
        ],
        "exact episode contract differs",
    )
    return rows


def _membership(
    values: np.ndarray,
    sorted_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_ids, values)
    in_range = positions < len(sorted_ids)
    matched = np.zeros(len(values), dtype=bool)
    matched[in_range] = sorted_ids[positions[in_range]] == values[in_range]
    return matched, positions


def _scan_window_partition(
    pair: tuple[int, Path],
    *,
    fields: np.ndarray,
    dispositions: np.ndarray,
    types: np.ndarray,
    year_to_episode: np.ndarray,
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path)
    works = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(works % PARTITIONS == index)), f"work partition mismatch: {index}")
    year_array = table["publication_year"].combine_chunks()
    year_valid = year_array.is_valid().to_numpy(zero_copy_only=False)
    years = year_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint16, copy=False)
    field_array = table["primary_field_id"].combine_chunks()
    field_valid = field_array.is_valid().to_numpy(zero_copy_only=False)
    observed_fields = field_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
    topic_array = table["primary_topic_id"].combine_chunks()
    topic_valid = topic_array.is_valid().to_numpy(zero_copy_only=False)
    topics = topic_array.fill_null(0).to_numpy(zero_copy_only=False)
    observed_dispositions = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    observed_types = table["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)
    paratext = table["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)
    retracted_array = table["is_retracted"].combine_chunks()
    retracted_valid = retracted_array.is_valid().to_numpy(zero_copy_only=False)
    retracted = retracted_array.fill_null(False).to_numpy(zero_copy_only=False)
    teams = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    episode = np.zeros(len(table), dtype=np.uint8)
    in_year_range = years < len(year_to_episode)
    episode[in_year_range] = year_to_episode[years[in_year_range]]
    included = (
        year_valid
        & field_valid
        & topic_valid
        & (episode > 0)
        & np.isin(observed_fields, fields)
        & np.isin(observed_dispositions, dispositions)
        & np.isin(observed_types, types)
        & (topics > 0)
        & ~paratext
        & ~(retracted_valid & retracted)
        & (teams <= 100)
    )
    indices = np.flatnonzero(included)
    values = np.empty(len(indices), dtype=WINDOW_WORK_DTYPE)
    values["work_numeric_id"] = works[indices]
    values["episode_index"] = episode[indices]
    values["publication_year"] = years[indices]
    values["field_id"] = observed_fields[indices]
    values["declared_authors_count"] = teams[indices]
    values["temporal_disposition"] = observed_dispositions[indices]
    return {
        "rows": len(table),
        "window_prelink_rows": len(values),
        "by_episode": {
            episode_index: int((values["episode_index"] == episode_index).sum())
            for episode_index in range(1, 5)
        },
        "values": values,
    }


def _window_work_index(
    paths: Sequence[Path],
    *,
    fields: Sequence[int],
    dispositions: Sequence[int],
    types: Sequence[int],
    episodes: Sequence[Mapping[str, int]],
    workers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    year_to_episode = np.zeros(65536, dtype=np.uint8)
    for episode in episodes:
        low, high = episode["prior_graph_window"]
        require(not bool(np.any(year_to_episode[low : high + 1])), "graph windows overlap")
        year_to_episode[low : high + 1] = int(episode["certified_episode_index"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(
            executor.map(
                lambda pair: _scan_window_partition(
                    pair,
                    fields=np.asarray(fields, dtype=np.uint8),
                    dispositions=np.asarray(dispositions, dtype=np.uint8),
                    types=np.asarray(types, dtype=np.uint8),
                    year_to_episode=year_to_episode,
                ),
                enumerate(paths),
            )
        )
    accounting = {
        "source_rows_scanned": sum(int(item["rows"]) for item in chunks),
        "window_eligible_prelink_work_rows": sum(
            int(item["window_prelink_rows"]) for item in chunks
        ),
        "by_episode": {
            str(episode_index): sum(
                int(item["by_episode"][episode_index]) for item in chunks
            )
            for episode_index in range(1, 5)
        },
    }
    values = np.concatenate([item["values"] for item in chunks], dtype=WINDOW_WORK_DTYPE)
    del chunks
    order = np.argsort(values["work_numeric_id"], kind="stable")
    values = values[order]
    if len(values) > 1:
        require(
            not bool(
                np.any(
                    values["work_numeric_id"][1:] == values["work_numeric_id"][:-1]
                )
            ),
            "window work duplicated",
        )
    return values, accounting


def _selected_arrays(selected: pa.Table) -> dict[str, np.ndarray]:
    return {
        name: selected[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in selected.schema.names
    }


def _target_incidence(
    paths: Sequence[Path],
    selected: pa.Table,
) -> tuple[np.ndarray, dict[str, int]]:
    arrays = _selected_arrays(selected)
    order = np.argsort(arrays["work_numeric_id"], kind="stable")
    sorted_ids = arrays["work_numeric_id"][order]
    chunks = []
    scanned = 0
    for partition_index, path in enumerate(paths):
        table = pq.read_table(path)
        scanned += len(table)
        authors = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        require(
            bool(np.all(authors % PARTITIONS == partition_index)),
            f"author partition mismatch: {partition_index}",
        )
        works = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        matched, positions = _membership(works, sorted_ids)
        if not bool(np.any(matched)):
            continue
        source = np.flatnonzero(matched)
        target_index = order[positions[matched]]
        years = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)[source]
        fields = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)[source]
        teams = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)[source]
        dispositions = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)[source]
        require(bool(np.all(years == arrays["publication_year"][target_index])), "target event year mismatch")
        require(bool(np.all(fields == arrays["field_id"][target_index])), "target event field mismatch")
        require(bool(np.all(teams == arrays["declared_authors_count"][target_index])), "target event team mismatch")
        require(bool(np.all(np.isin(dispositions, [2, 3]))), "target event disposition mismatch")
        values = np.empty(len(source), dtype=WORK_AUTHOR_DTYPE)
        values["episode_index"] = arrays["certified_episode_index"][target_index]
        values["work_numeric_id"] = works[source]
        values["author_numeric_id"] = authors[source]
        values["publication_year"] = years
        values["declared_authors_count"] = teams
        chunks.append(values)
    require(bool(chunks), "no target author-work events found")
    incidence = np.concatenate(chunks, dtype=WORK_AUTHOR_DTYPE)
    order = np.lexsort(
        (
            incidence["author_numeric_id"],
            incidence["work_numeric_id"],
            incidence["episode_index"],
        )
    )
    incidence = incidence[order]
    if len(incidence) > 1:
        duplicate = (
            (incidence["episode_index"][1:] == incidence["episode_index"][:-1])
            & (incidence["work_numeric_id"][1:] == incidence["work_numeric_id"][:-1])
            & (
                incidence["author_numeric_id"][1:]
                == incidence["author_numeric_id"][:-1]
            )
        )
        require(not bool(np.any(duplicate)), "duplicate target author-work event")
    return incidence, {
        "source_rows_scanned": scanned,
        "target_author_incidence_rows": len(incidence),
    }


def _target_table(
    selected: pa.Table,
    incidence: np.ndarray,
    episodes: Sequence[Mapping[str, int]],
) -> tuple[pa.Table, dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    arrays = _selected_arrays(selected)
    work_order = np.argsort(arrays["work_numeric_id"], kind="stable")
    sorted_work = arrays["work_numeric_id"][work_order]
    positions = np.searchsorted(sorted_work, incidence["work_numeric_id"])
    require(
        bool(np.all(sorted_work[positions] == incidence["work_numeric_id"])),
        "target incidence work absent",
    )
    selected_index = work_order[positions]
    valid_count = np.bincount(selected_index, minlength=len(selected)).astype(np.uint32)
    incumbent_event = np.zeros(len(incidence), dtype=bool)
    target_authors: dict[int, np.ndarray] = {}
    incumbent_authors: dict[int, np.ndarray] = {}
    for episode in episodes:
        episode_index = int(episode["certified_episode_index"])
        episode_mask = incidence["episode_index"] == episode_index
        target_authors[episode_index] = np.unique(
            incidence["author_numeric_id"][episode_mask]
        )
        incumbent = np.unique(
            incidence["author_numeric_id"][
                episode_mask
                & (
                    incidence["publication_year"]
                    < int(episode["entry_year"]) - 6
                )
            ]
        )
        incumbent_authors[episode_index] = incumbent
        incumbent_event[episode_mask] = np.isin(
            incidence["author_numeric_id"][episode_mask],
            incumbent,
        )
    incumbent_count = np.bincount(
        selected_index,
        weights=incumbent_event.astype(np.uint8),
        minlength=len(selected),
    ).astype(np.uint32)
    has_any = incumbent_count > 0
    all_incumbent = (valid_count > 0) & (incumbent_count == valid_count)
    require(bool(np.all(valid_count > 0)), "selected target work without valid author")
    table = pa.Table.from_arrays(
        [
            selected["certified_episode_index"],
            selected["field_id"],
            selected["topic_id"],
            selected["entry_year"],
            selected["work_numeric_id"],
            selected["publication_year"],
            selected["work_type_code"],
            selected["declared_authors_count"],
            selected["window_code"],
            pa.array(valid_count, type=pa.uint32()),
            pa.array(incumbent_count, type=pa.uint32()),
            pa.array(has_any, type=pa.bool_()),
            pa.array(all_incumbent, type=pa.bool_()),
        ],
        schema=TARGET_OUTPUT_SCHEMA,
    )
    return table, target_authors, incumbent_authors, {
        "target_work_rows": len(table),
        "target_author_incidence_rows": len(incidence),
        "target_authors_by_episode": {
            str(key): len(value) for key, value in target_authors.items()
        },
        "pre_signal_incumbent_authors_by_episode": {
            str(key): len(value) for key, value in incumbent_authors.items()
        },
        "any_incumbent_target_work_rows": int(has_any.sum()),
        "all_incumbent_target_work_rows": int(all_incumbent.sum()),
    }


def _candidate_sets(
    authors: pa.Table,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, int]]:
    episode = authors["certified_episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
    author = authors["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    order = np.lexsort((author, episode))
    episode = episode[order]
    author = author[order]
    if len(author) > 1:
        require(
            not bool(
                np.any(
                    (episode[1:] == episode[:-1]) & (author[1:] == author[:-1])
                )
            ),
            "candidate author duplicated",
        )
    values = np.empty(
        len(author),
        dtype=[("episode_index", "u1"), ("author_numeric_id", "<u8")],
    )
    values["episode_index"] = episode
    values["author_numeric_id"] = author
    sets = {
        episode_index: author[episode == episode_index]
        for episode_index in range(1, 5)
    }
    return values, sets, {
        str(episode_index): len(sets[episode_index])
        for episode_index in range(1, 5)
    }


def _partition_author_sets(
    author_sets: Mapping[int, np.ndarray],
) -> dict[int, list[np.ndarray]]:
    partitioned: dict[int, list[np.ndarray]] = {}
    for episode_index in range(1, 5):
        authors = author_sets[episode_index]
        partitions = (authors % PARTITIONS).astype(np.uint16, copy=False)
        order = np.argsort(partitions, kind="stable")
        ordered_authors = authors[order]
        counts = np.bincount(partitions, minlength=PARTITIONS)
        stops = np.cumsum(counts, dtype=np.int64)
        starts = np.r_[0, stops[:-1]]
        partitioned[episode_index] = [
            ordered_authors[start:stop]
            for start, stop in zip(starts.tolist(), stops.tolist(), strict=True)
        ]
        require(
            sum(len(values) for values in partitioned[episode_index])
            == len(authors),
            f"author partition closure failed: {episode_index}",
        )
    return partitioned


def _scan_prior_events(
    paths: Sequence[Path],
    window_works: np.ndarray,
    *,
    candidate_sets: Mapping[int, np.ndarray],
    target_authors: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidate_chunks = []
    peer_chunks = []
    scanned = 0
    eligible_event_rows = 0
    candidate_partitions = _partition_author_sets(candidate_sets)
    target_partitions = _partition_author_sets(target_authors)
    work_by_year: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for year in np.unique(window_works["publication_year"]).tolist():
        index = np.flatnonzero(window_works["publication_year"] == year)
        work_by_year[int(year)] = (window_works["work_numeric_id"][index], index)
    graph_years = np.array(sorted(work_by_year), dtype=np.uint16)
    for partition_index, path in enumerate(paths):
        table = pq.read_table(path)
        scanned += len(table)
        authors = (
            table["author_numeric_id"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        require(
            bool(np.all(authors % PARTITIONS == partition_index)),
            f"author partition mismatch: {partition_index}",
        )
        works = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        years = (
            table["publication_year"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        fields = (
            table["primary_field_id"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        teams = (
            table["declared_authors_count"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        dispositions = (
            table["temporal_disposition"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        prefilter = (
            np.isin(years, graph_years)
            & (fields >= 11)
            & (fields <= 36)
            & np.isin(dispositions, [2, 3])
            & (teams <= 100)
        )
        if not bool(np.any(prefilter)):
            continue
        source_chunks = []
        metadata_chunks = []
        for year in graph_years.tolist():
            year_source = np.flatnonzero(prefilter & (years == year))
            if not len(year_source):
                continue
            year_work_ids, year_work_index = work_by_year[int(year)]
            matched, positions = _membership(works[year_source], year_work_ids)
            if bool(np.any(matched)):
                source_chunks.append(year_source[matched])
                metadata_chunks.append(year_work_index[positions[matched]])
        if not source_chunks:
            continue
        source = np.concatenate(source_chunks)
        metadata_index = np.concatenate(metadata_chunks)
        require(
            bool(np.all(years[source] == window_works["publication_year"][metadata_index])),
            "prior event year mismatch",
        )
        require(
            bool(np.all(fields[source] == window_works["field_id"][metadata_index])),
            "prior event field mismatch",
        )
        require(
            bool(
                np.all(
                    teams[source]
                    == window_works["declared_authors_count"][metadata_index]
                )
            ),
            "prior event team mismatch",
        )
        require(
            bool(
                np.all(
                    dispositions[source]
                    == window_works["temporal_disposition"][metadata_index]
                )
            ),
            "prior event disposition mismatch",
        )
        eligible_event_rows += len(source)
        event_episode = window_works["episode_index"][metadata_index]
        event_authors = authors[source]
        for episode_index in range(1, 5):
            episode_mask = event_episode == episode_index
            if not bool(np.any(episode_mask)):
                continue
            episode_source = source[episode_mask]
            episode_metadata = metadata_index[episode_mask]
            episode_authors = event_authors[episode_mask]
            candidate, _ = _membership(
                episode_authors,
                candidate_partitions[episode_index][partition_index],
            )
            peer, _ = _membership(
                episode_authors,
                target_partitions[episode_index][partition_index],
            )
            for keep, chunks in (
                (candidate, candidate_chunks),
                (peer, peer_chunks),
            ):
                if not bool(np.any(keep)):
                    continue
                values = np.empty(int(keep.sum()), dtype=WORK_AUTHOR_DTYPE)
                values["episode_index"] = episode_index
                values["work_numeric_id"] = works[episode_source[keep]]
                values["author_numeric_id"] = episode_authors[keep]
                values["publication_year"] = years[episode_source[keep]]
                values["declared_authors_count"] = teams[episode_source[keep]]
                chunks.append(values)
    candidates = (
        np.concatenate(candidate_chunks, dtype=WORK_AUTHOR_DTYPE)
        if candidate_chunks
        else np.empty(0, dtype=WORK_AUTHOR_DTYPE)
    )
    peers = (
        np.concatenate(peer_chunks, dtype=WORK_AUTHOR_DTYPE)
        if peer_chunks
        else np.empty(0, dtype=WORK_AUTHOR_DTYPE)
    )
    return candidates, peers, {
        "source_rows_scanned": scanned,
        "eligible_prior_window_author_event_rows": eligible_event_rows,
        "candidate_prior_work_event_rows": len(candidates),
        "target_author_prior_work_event_rows": len(peers),
    }


def _sort_unique_work_author(values: np.ndarray, label: str) -> np.ndarray:
    order = np.lexsort(
        (
            values["author_numeric_id"],
            values["episode_index"],
            values["work_numeric_id"],
        )
    )
    values = values[order]
    if len(values) > 1:
        duplicate = (
            (values["episode_index"][1:] == values["episode_index"][:-1])
            & (values["work_numeric_id"][1:] == values["work_numeric_id"][:-1])
            & (
                values["author_numeric_id"][1:]
                == values["author_numeric_id"][:-1]
            )
        )
        require(not bool(np.any(duplicate)), f"duplicate {label} work-author row")
    return values


def _reduce_pair_incidences(pairs: np.ndarray) -> np.ndarray:
    order = np.lexsort(
        (
            pairs["peer_author_id"],
            pairs["candidate_author_id"],
            pairs["episode_index"],
        )
    )
    pairs = pairs[order]
    if not len(pairs):
        return np.empty(0, dtype=EDGE_DTYPE)
    start = np.r_[
        0,
        np.flatnonzero(
            (pairs["episode_index"][1:] != pairs["episode_index"][:-1])
            | (
                pairs["candidate_author_id"][1:]
                != pairs["candidate_author_id"][:-1]
            )
            | (pairs["peer_author_id"][1:] != pairs["peer_author_id"][:-1])
        )
        + 1,
    ]
    ends = np.r_[start[1:], len(pairs)]
    edges = np.empty(len(start), dtype=EDGE_DTYPE)
    edges["episode_index"] = pairs["episode_index"][start]
    edges["candidate_author_id"] = pairs["candidate_author_id"][start]
    edges["peer_author_id"] = pairs["peer_author_id"][start]
    edges["minimum_declared_team_size"] = np.minimum.reduceat(
        pairs["declared_authors_count"],
        start,
    )
    edges["first_prior_collaboration_year"] = np.minimum.reduceat(
        pairs["publication_year"],
        start,
    )
    edges["last_prior_collaboration_year"] = np.maximum.reduceat(
        pairs["publication_year"],
        start,
    )
    edges["distinct_shared_work_count"] = (ends - start).astype(np.uint32)
    edges["peer_pre_signal_incumbent"] = 0
    return edges


def _merge_partial_edges(partials: Sequence[np.ndarray]) -> np.ndarray:
    if not partials:
        return np.empty(0, dtype=EDGE_DTYPE)
    values = np.concatenate(partials, dtype=EDGE_DTYPE)
    order = np.lexsort(
        (
            values["peer_author_id"],
            values["candidate_author_id"],
            values["episode_index"],
        )
    )
    values = values[order]
    start = np.r_[
        0,
        np.flatnonzero(
            (values["episode_index"][1:] != values["episode_index"][:-1])
            | (
                values["candidate_author_id"][1:]
                != values["candidate_author_id"][:-1]
            )
            | (
                values["peer_author_id"][1:]
                != values["peer_author_id"][:-1]
            )
        )
        + 1,
    ]
    edges = values[start].copy()
    edges["minimum_declared_team_size"] = np.minimum.reduceat(
        values["minimum_declared_team_size"],
        start,
    )
    edges["first_prior_collaboration_year"] = np.minimum.reduceat(
        values["first_prior_collaboration_year"],
        start,
    )
    edges["last_prior_collaboration_year"] = np.maximum.reduceat(
        values["last_prior_collaboration_year"],
        start,
    )
    counts = np.add.reduceat(
        values["distinct_shared_work_count"].astype(np.uint64),
        start,
    )
    require(
        bool(np.all(counts <= np.iinfo(np.uint32).max)),
        "shared work count overflow",
    )
    edges["distinct_shared_work_count"] = counts.astype(np.uint32)
    edges["peer_pre_signal_incumbent"] = 0
    return edges


def _pair_edges(
    candidate_events: np.ndarray,
    peer_events: np.ndarray,
    *,
    incumbent_authors: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, dict[str, int]]:
    candidates = _sort_unique_work_author(candidate_events, "candidate")
    peers = _sort_unique_work_author(peer_events, "peer")
    candidate_work, candidate_start, candidate_count = np.unique(
        candidates["work_numeric_id"],
        return_index=True,
        return_counts=True,
    )
    peer_work, peer_start, peer_count = np.unique(
        peers["work_numeric_id"],
        return_index=True,
        return_counts=True,
    )
    common, candidate_group, peer_group = np.intersect1d(
        candidate_work,
        peer_work,
        assume_unique=True,
        return_indices=True,
    )
    maximum_pairs = int(
        np.sum(
            candidate_count[candidate_group].astype(np.uint64)
            * peer_count[peer_group].astype(np.uint64),
            dtype=np.uint64,
        )
    )
    require(maximum_pairs <= 500_000_000, "candidate-target-peer pair expansion too large")
    for values, starts, counts, label in (
        (candidates, candidate_start, candidate_count, "candidate"),
        (peers, peer_start, peer_count, "peer"),
    ):
        for column in (
            "episode_index",
            "publication_year",
            "declared_authors_count",
        ):
            minimum = np.minimum.reduceat(values[column], starts)
            maximum = np.maximum.reduceat(values[column], starts)
            require(
                bool(np.all(minimum == maximum)),
                f"{label} work {column} differs",
            )
    candidate_representative = candidate_start[candidate_group]
    peer_representative = peer_start[peer_group]
    for column in (
        "episode_index",
        "publication_year",
        "declared_authors_count",
    ):
        require(
            bool(
                np.all(
                    candidates[column][candidate_representative]
                    == peers[column][peer_representative]
                )
            ),
            f"candidate/peer work {column} differs",
        )

    pair_counts = (
        candidate_count[candidate_group].astype(np.uint64)
        * peer_count[peer_group].astype(np.uint64)
    )
    cumulative = np.cumsum(pair_counts, dtype=np.uint64)
    partials = []
    observed_pairs = 0
    group_start = 0
    maximum_batch_pairs = 5_000_000
    while group_start < len(common):
        prior = int(cumulative[group_start - 1]) if group_start else 0
        group_end = int(
            np.searchsorted(
                cumulative,
                prior + maximum_batch_pairs,
                side="right",
            )
        )
        group_end = max(group_end, group_start + 1)
        batch_counts = pair_counts[group_start:group_end].astype(
            np.int64,
            copy=False,
        )
        batch_total = int(batch_counts.sum(dtype=np.int64))
        group_local = np.repeat(
            np.arange(len(batch_counts), dtype=np.int64),
            batch_counts,
        )
        batch_starts = np.r_[0, np.cumsum(batch_counts[:-1], dtype=np.int64)]
        local_pair = np.arange(batch_total, dtype=np.int64) - np.repeat(
            batch_starts,
            batch_counts,
        )
        candidate_groups = candidate_group[group_start:group_end][group_local]
        peer_groups = peer_group[group_start:group_end][group_local]
        peer_group_counts = peer_count[peer_groups].astype(np.int64)
        candidate_rows = (
            candidate_start[candidate_groups]
            + local_pair // peer_group_counts
        )
        peer_rows = (
            peer_start[peer_groups]
            + local_pair % peer_group_counts
        )
        keep = (
            candidates["author_numeric_id"][candidate_rows]
            != peers["author_numeric_id"][peer_rows]
        )
        count = int(keep.sum())
        observed_pairs += count
        if count:
            candidate_rows = candidate_rows[keep]
            peer_rows = peer_rows[keep]
            pairs = np.empty(count, dtype=PAIR_INCIDENCE_DTYPE)
            pairs["episode_index"] = candidates["episode_index"][candidate_rows]
            pairs["candidate_author_id"] = candidates["author_numeric_id"][
                candidate_rows
            ]
            pairs["peer_author_id"] = peers["author_numeric_id"][peer_rows]
            pairs["publication_year"] = candidates["publication_year"][
                candidate_rows
            ]
            pairs["declared_authors_count"] = candidates[
                "declared_authors_count"
            ][candidate_rows]
            partials.append(_reduce_pair_incidences(pairs))
        group_start = group_end
    edges = _merge_partial_edges(partials)
    for episode_index in range(1, 5):
        mask = edges["episode_index"] == episode_index
        edges["peer_pre_signal_incumbent"][mask] = np.isin(
            edges["peer_author_id"][mask],
            incumbent_authors[episode_index],
        ).astype(np.uint8)
    return edges, {
        "common_prior_work_rows": len(common),
        "candidate_target_peer_pair_expansion_before_self_exclusion": maximum_pairs,
        "candidate_target_peer_work_incidences": observed_pairs,
        "distinct_candidate_target_peer_edges": len(edges),
    }


def _edge_table(edges: np.ndarray) -> pa.Table:
    return pa.Table.from_arrays(
        [
            pa.array(edges["episode_index"], type=pa.uint8()),
            pa.array(edges["candidate_author_id"], type=pa.uint64()),
            pa.array(edges["peer_author_id"], type=pa.uint64()),
            pa.array(edges["minimum_declared_team_size"], type=pa.uint16()),
            pa.array(edges["first_prior_collaboration_year"], type=pa.uint16()),
            pa.array(edges["last_prior_collaboration_year"], type=pa.uint16()),
            pa.array(edges["distinct_shared_work_count"], type=pa.uint32()),
            pa.array(edges["peer_pre_signal_incumbent"].astype(bool), type=pa.bool_()),
        ],
        schema=EDGE_OUTPUT_SCHEMA,
    )


def _expand_peer_target_exclusions(
    edges: np.ndarray,
    target_incidence: np.ndarray,
    selected: pa.Table,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    selected_arrays = _selected_arrays(selected)
    selected_order = np.argsort(selected_arrays["work_numeric_id"], kind="stable")
    selected_ids = selected_arrays["work_numeric_id"][selected_order]
    incidence_position = np.searchsorted(selected_ids, target_incidence["work_numeric_id"])
    require(
        bool(np.all(incidence_position < len(selected_ids))),
        "target incidence work position exceeds selected works",
    )
    require(
        bool(
            np.all(
                selected_ids[incidence_position]
                == target_incidence["work_numeric_id"]
            )
        ),
        "target incidence work absent from selected works",
    )
    target_work_index = selected_order[incidence_position].astype(np.uint32)
    edge_order = np.lexsort(
        (
            edges["candidate_author_id"],
            edges["peer_author_id"],
            edges["episode_index"],
        )
    )
    edges = edges[edge_order]
    incidence_order = np.lexsort(
        (
            target_incidence["work_numeric_id"],
            target_incidence["author_numeric_id"],
            target_incidence["episode_index"],
        )
    )
    incidence = target_incidence[incidence_order]
    incidence_target_index = target_work_index[incidence_order]
    edge_peer_key = (
        edges["peer_author_id"].astype(np.uint64) * np.uint64(8)
        + edges["episode_index"].astype(np.uint64)
    )
    incidence_peer_key = (
        incidence["author_numeric_id"].astype(np.uint64) * np.uint64(8)
        + incidence["episode_index"].astype(np.uint64)
    )
    unique_peer, peer_start, peer_count = np.unique(
        incidence_peer_key,
        return_index=True,
        return_counts=True,
    )
    positions = np.searchsorted(unique_peer, edge_peer_key)
    matched = (positions < len(unique_peer)) & (
        unique_peer[np.minimum(positions, len(unique_peer) - 1)] == edge_peer_key
    )
    require(bool(np.all(matched)), "edge peer lacks target work")
    total = int(
        np.sum(peer_count[positions].astype(np.uint64), dtype=np.uint64)
    )
    peer_exclusions = np.empty(total, dtype=EXCLUSION_DTYPE)
    cursor = 0
    for edge_index, peer_group in enumerate(positions.tolist()):
        start = int(peer_start[peer_group])
        end = start + int(peer_count[peer_group])
        count = end - start
        target = slice(cursor, cursor + count)
        peer_exclusions["episode_index"][target] = edges["episode_index"][edge_index]
        peer_exclusions["candidate_author_id"][target] = edges[
            "candidate_author_id"
        ][edge_index]
        peer_exclusions["target_work_index"][target] = incidence_target_index[
            start:end
        ]
        peer_exclusions["minimum_prior_tie_team_size"][target] = edges[
            "minimum_declared_team_size"
        ][edge_index]
        cursor += count
    own = np.empty(len(incidence), dtype=EXCLUSION_DTYPE)
    own["episode_index"] = incidence["episode_index"]
    own["candidate_author_id"] = incidence["author_numeric_id"]
    own["target_work_index"] = incidence_target_index
    own["minimum_prior_tie_team_size"] = 0
    return peer_exclusions, own, {
        "peer_target_work_exclusion_incidences": len(peer_exclusions),
        "focal_target_work_exclusion_incidences_before_candidate_filter": len(own),
    }


def _filter_candidate_exclusions(
    values: np.ndarray,
    candidate_sets: Mapping[int, np.ndarray],
) -> np.ndarray:
    keep = np.zeros(len(values), dtype=bool)
    for episode_index in range(1, 5):
        episode_mask = values["episode_index"] == episode_index
        keep[episode_mask] = np.isin(
            values["candidate_author_id"][episode_mask],
            candidate_sets[episode_index],
        )
    return values[keep]


def _reduce_exclusions(values: np.ndarray) -> np.ndarray:
    order = np.lexsort(
        (
            values["target_work_index"],
            values["candidate_author_id"],
            values["episode_index"],
        )
    )
    values = values[order]
    if not len(values):
        return values
    start = np.r_[
        0,
        np.flatnonzero(
            (values["episode_index"][1:] != values["episode_index"][:-1])
            | (
                values["candidate_author_id"][1:]
                != values["candidate_author_id"][:-1]
            )
            | (
                values["target_work_index"][1:]
                != values["target_work_index"][:-1]
            )
        )
        + 1,
    ]
    result = values[start].copy()
    result["minimum_prior_tie_team_size"] = np.minimum.reduceat(
        values["minimum_prior_tie_team_size"],
        start,
    )
    return result


def _candidate_positions(
    candidate_values: np.ndarray,
    episodes: np.ndarray,
    authors: np.ndarray,
) -> np.ndarray:
    require(len(episodes) == len(authors), "measurement key length mismatch")
    require(
        bool(np.all(np.isin(episodes, [1, 2, 3, 4]))),
        "measurement episode outside exact four",
    )
    positions = np.empty(len(authors), dtype=np.int64)
    for episode_index in range(1, 5):
        value_mask = episodes == episode_index
        if not bool(np.any(value_mask)):
            continue
        candidate_mask = candidate_values["episode_index"] == episode_index
        require(
            bool(np.any(candidate_mask)),
            "measurement episode has no candidate rows",
        )
        candidate_authors = candidate_values["author_numeric_id"][candidate_mask]
        candidate_offset = int(np.flatnonzero(candidate_mask)[0])
        local = np.searchsorted(candidate_authors, authors[value_mask])
        require(
            bool(np.all(local < len(candidate_authors))),
            "measurement row candidate position exceeds episode candidates",
        )
        require(
            bool(np.all(candidate_authors[local] == authors[value_mask])),
            "measurement row candidate absent",
        )
        positions[value_mask] = candidate_offset + local
    return positions


def _count_rows_by_candidate(
    candidate_values: np.ndarray,
    episodes: np.ndarray,
    authors: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    positions = _candidate_positions(
        candidate_values,
        episodes[mask],
        authors[mask],
    )
    counts = np.bincount(positions, minlength=len(candidate_values))
    require(bool(np.all(counts <= np.iinfo(np.uint32).max)), "candidate count overflow")
    return counts.astype(np.uint32)


def _author_measurement_table(
    candidate_values: np.ndarray,
    authors: pa.Table,
    edges: np.ndarray,
    peer_exclusions: np.ndarray,
    own_exclusions: np.ndarray,
    network_exclusions: np.ndarray,
    target_table: pa.Table,
) -> tuple[pa.Table, dict[str, Any]]:
    author_episode = authors["certified_episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
    author_id = authors["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    author_order = np.lexsort((author_id, author_episode))
    authors = authors.take(pa.array(author_order, type=pa.int64()))
    require(
        bool(
            np.all(
                candidate_values["episode_index"]
                == authors["certified_episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
            )
        ),
        "candidate episode order differs",
    )
    require(
        bool(
            np.all(
                candidate_values["author_numeric_id"]
                == authors["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
            )
        ),
        "candidate author order differs",
    )
    target = {
        name: target_table[name].combine_chunks().to_numpy(zero_copy_only=False)
        for name in target_table.schema.names
    }
    arrays: dict[str, pa.Array] = {
        name: authors[name] for name in authors.schema.names
    }
    accounting: dict[str, Any] = {"by_cap": {}}
    for cap in CAPS:
        edge_mask = edges["minimum_declared_team_size"] <= cap
        peer_count = _count_rows_by_candidate(
            candidate_values,
            edges["episode_index"],
            edges["candidate_author_id"],
            edge_mask,
        )
        incumbent_peer_count = _count_rows_by_candidate(
            candidate_values,
            edges["episode_index"],
            edges["candidate_author_id"],
            edge_mask & edges["peer_pre_signal_incumbent"].astype(bool),
        )
        peer_target_mask = (
            peer_exclusions["minimum_prior_tie_team_size"] <= cap
        ) & (
            target["declared_authors_count"][
                peer_exclusions["target_work_index"]
            ]
            <= cap
        )
        peer_target_count = _count_rows_by_candidate(
            candidate_values,
            peer_exclusions["episode_index"],
            peer_exclusions["candidate_author_id"],
            peer_target_mask,
        )
        own_target_mask = (
            target["declared_authors_count"][own_exclusions["target_work_index"]]
            <= cap
        )
        network_target_mask = (
            network_exclusions["minimum_prior_tie_team_size"] <= cap
        ) & (
            target["declared_authors_count"][
                network_exclusions["target_work_index"]
            ]
            <= cap
        )
        global_any = np.zeros(5, dtype=np.uint32)
        global_all = np.zeros(5, dtype=np.uint32)
        for episode_index in range(1, 5):
            episode_target = target["certified_episode_index"] == episode_index
            cap_target = target["declared_authors_count"] <= cap
            global_any[episode_index] = int(
                (
                    episode_target
                    & cap_target
                    & target["has_pre_signal_incumbent_author"]
                ).sum()
            )
            global_all[episode_index] = int(
                (
                    episode_target
                    & cap_target
                    & target["all_linked_authors_pre_signal_incumbents"]
                ).sum()
            )
        candidate_episode = candidate_values["episode_index"]
        own_any_mask = own_target_mask & target["has_pre_signal_incumbent_author"][
            own_exclusions["target_work_index"]
        ]
        own_all_mask = own_target_mask & target[
            "all_linked_authors_pre_signal_incumbents"
        ][own_exclusions["target_work_index"]]
        network_any_mask = network_target_mask & target[
            "has_pre_signal_incumbent_author"
        ][network_exclusions["target_work_index"]]
        network_all_mask = network_target_mask & target[
            "all_linked_authors_pre_signal_incumbents"
        ][network_exclusions["target_work_index"]]
        own_any = _count_rows_by_candidate(
            candidate_values,
            own_exclusions["episode_index"],
            own_exclusions["candidate_author_id"],
            own_any_mask,
        )
        own_all = _count_rows_by_candidate(
            candidate_values,
            own_exclusions["episode_index"],
            own_exclusions["candidate_author_id"],
            own_all_mask,
        )
        network_any = _count_rows_by_candidate(
            candidate_values,
            network_exclusions["episode_index"],
            network_exclusions["candidate_author_id"],
            network_any_mask,
        )
        network_all = _count_rows_by_candidate(
            candidate_values,
            network_exclusions["episode_index"],
            network_exclusions["candidate_author_id"],
            network_all_mask,
        )
        require(
            bool(np.all(own_any <= network_any)),
            f"focal any exclusion exceeds network: {cap}",
        )
        require(
            bool(np.all(own_all <= network_all)),
            f"focal all exclusion exceeds network: {cap}",
        )
        global_any_rows = global_any[candidate_episode]
        global_all_rows = global_all[candidate_episode]
        require(bool(np.all(network_any <= global_any_rows)), "network any exceeds attention")
        require(bool(np.all(network_all <= global_all_rows)), "network all exceeds attention")
        values = {
            f"prior_target_active_peer_count_cap_{cap}": peer_count,
            f"prior_incumbent_peer_count_cap_{cap}": incumbent_peer_count,
            f"prior_peer_target_work_count_cap_{cap}": peer_target_count,
            f"any_incumbent_attention_work_count_cap_{cap}": global_any_rows,
            f"any_incumbent_focal_leaveout_work_count_cap_{cap}": global_any_rows
            - own_any,
            f"any_incumbent_network_leaveout_work_count_cap_{cap}": global_any_rows
            - network_any,
            f"all_incumbent_attention_work_count_cap_{cap}": global_all_rows,
            f"all_incumbent_focal_leaveout_work_count_cap_{cap}": global_all_rows
            - own_all,
            f"all_incumbent_network_leaveout_work_count_cap_{cap}": global_all_rows
            - network_all,
        }
        for name, value in values.items():
            arrays[name] = pa.array(value, type=pa.uint32())
        arrays[f"has_prior_target_active_peer_cap_{cap}"] = pa.array(
            peer_count > 0,
            type=pa.bool_(),
        )
        accounting["by_cap"][str(cap)] = {
            "candidate_rows_with_prior_target_active_peer": int((peer_count > 0).sum()),
            "prior_target_active_peer_edges": int(peer_count.sum(dtype=np.uint64)),
            "prior_incumbent_peer_edges": int(
                incumbent_peer_count.sum(dtype=np.uint64)
            ),
            "distinct_candidate_peer_target_work_rows": int(
                peer_target_count.sum(dtype=np.uint64)
            ),
            "focal_any_incumbent_target_work_exclusions": int(
                own_any.sum(dtype=np.uint64)
            ),
            "network_any_incumbent_target_work_exclusions": int(
                network_any.sum(dtype=np.uint64)
            ),
        }
    table = pa.Table.from_arrays(
        [arrays[name] for name in AUTHOR_OUTPUT_SCHEMA.names],
        schema=AUTHOR_OUTPUT_SCHEMA,
    )
    accounting["author_measurement_rows"] = len(table)
    return table, accounting


def _write_parquet(table: pa.Table, path: Path, *, compression_level: int = 9) -> dict[str, Any]:
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=compression_level,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        row_group_size=250_000,
    )
    os.chmod(path, 0o600)
    parquet = pq.ParquetFile(path)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Certified D0 incumbent and prior-target-peer leave-outs",
        "",
        f"Status: `{report['status']}`",
        "",
        "| Cap | Candidate rows with target-active peer | Prior target-peer edges | Network any-incumbent exclusions |",
        "|---:|---:|---:|---:|",
    ]
    for cap in CAPS:
        row = report["author_measurement_accounting"]["by_cap"][str(cap)]
        lines.append(
            f"| {cap} | "
            f"{row['candidate_rows_with_prior_target_active_peer']:,} | "
            f"{row['prior_target_active_peer_edges']:,} | "
            f"{row['network_any_incumbent_target_work_exclusions']:,} |"
        )
    lines.extend(
        [
            "",
            "All measures are predecision and outcome-blind. ORCID and",
            "same-affiliation proxies remain deferred to a receipt-bound raw",
            "authorship rescan. No team-cap profile is selected.",
            "",
        ]
    )
    return "\n".join(lines)


def _cleanup_temporary(temporary: Path | None, *, output: Path) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute(), "temporary path must be absolute")
    require(temporary.parent == output.parent, "temporary parent mismatch")
    require(temporary.name.startswith(f".{output.name}."), "temporary prefix mismatch")
    if temporary.exists():
        shutil.rmtree(temporary)


def build(
    *,
    config_path: Path,
    parent_root: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output: Path,
    workers: int,
) -> dict[str, Any]:
    started = time.monotonic()
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path, config_path.name)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V38",
        "component not authorized",
    )
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden boundary")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary opened")
    repository = _mapping(config.get("repository_bindings"), "repository bindings")
    gate_binding = _validate_repo_binding(_mapping(repository["gate_state"], "gate binding"))
    abi_binding = _validate_repo_binding(_mapping(repository["measurement_abi"], "ABI binding"))
    selected, author_base, parent_binding = _load_parent(
        parent_root,
        _mapping(config.get("verified_parent"), "verified parent"),
    )
    episodes = _episodes(config)
    observed_selected = [
        (
            int(row["certified_episode_index"]),
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        )
        for row in selected.select(
            [
                "certified_episode_index",
                "field_id",
                "topic_id",
                "entry_year",
            ]
        )
        .group_by(
            [
                "certified_episode_index",
                "field_id",
                "topic_id",
                "entry_year",
            ]
        )
        .aggregate([])
        .sort_by([("certified_episode_index", "ascending")])
        .to_pylist()
    ]
    require(
        observed_selected
        == [
            (
                episode["certified_episode_index"],
                episode["field_id"],
                episode["topic_id"],
                episode["entry_year"],
            )
            for episode in episodes
        ],
        "parent episodes differ",
    )
    inputs = _mapping(config.get("normalized_inputs"), "normalized inputs")
    work_paths, work_binding, work_stats = parent._validate_partitioned_input(
        cutoff_root,
        _mapping(inputs["cutoff_work_metadata"], "cutoff input"),
        schema=parent.WORK_INPUT_SCHEMA,
        workers=workers,
    )
    event_paths, event_binding, event_stats = parent._validate_partitioned_input(
        author_history_root,
        _mapping(inputs["author_history_events"], "event input"),
        schema=parent.EVENT_INPUT_SCHEMA,
        workers=workers,
    )
    contract = _mapping(config.get("measurement_contract"), "measurement contract")
    fields = [int(value) for value in contract["all_field_metric_field_ids"]]
    types = [int(value) for value in contract["inherited_work_type_codes"]]
    dispositions = [int(value) for value in contract["allowed_temporal_dispositions"]]
    require(fields == list(range(11, 37)), "field closure differs")
    require(types == [1, 2, 3, 4, 7, 16, 18, 21], "work types differ")
    require(dispositions == [2, 3], "dispositions differ")
    require(contract["graph_team_cap_profiles"] == [10, 50, 100], "graph caps differ")
    output = output.resolve()
    require(output.name == config["output_contract"]["root_basename"], "output basename mismatch")
    require(not output.exists(), "output already exists")
    require(output.parent.is_dir(), "output parent missing")
    for root in (parent_root.resolve(), cutoff_root.resolve(), author_history_root.resolve()):
        require(output != root and not output.is_relative_to(root), "output inside input")

    window_works, window_accounting = _window_work_index(
        work_paths,
        fields=fields,
        dispositions=dispositions,
        types=types,
        episodes=episodes,
        workers=workers,
    )
    require(
        window_accounting["source_rows_scanned"]
        == inputs["cutoff_work_metadata"]["rows"],
        "cutoff row closure failed",
    )
    target_incidence, target_incidence_accounting = _target_incidence(
        event_paths,
        selected,
    )
    require(
        target_incidence_accounting["source_rows_scanned"]
        == inputs["author_history_events"]["rows"],
        "target event row closure failed",
    )
    target_table, target_authors, incumbent_authors, target_accounting = _target_table(
        selected,
        target_incidence,
        episodes,
    )
    candidate_values, candidate_sets, candidate_accounting = _candidate_sets(
        author_base
    )
    candidate_events, peer_events, prior_event_accounting = _scan_prior_events(
        event_paths,
        window_works,
        candidate_sets=candidate_sets,
        target_authors=target_authors,
    )
    require(
        prior_event_accounting["source_rows_scanned"]
        == inputs["author_history_events"]["rows"],
        "prior event row closure failed",
    )
    edges, edge_accounting = _pair_edges(
        candidate_events,
        peer_events,
        incumbent_authors=incumbent_authors,
    )
    del candidate_events, peer_events, window_works
    edge_table = _edge_table(edges)
    peer_exclusions, own_exclusions, exclusion_accounting = (
        _expand_peer_target_exclusions(edges, target_incidence, selected)
    )
    own_exclusions = _filter_candidate_exclusions(
        own_exclusions,
        candidate_sets,
    )
    own_exclusions = _reduce_exclusions(own_exclusions)
    peer_exclusions = _reduce_exclusions(peer_exclusions)
    network_exclusions = _reduce_exclusions(
        np.concatenate([peer_exclusions, own_exclusions], dtype=EXCLUSION_DTYPE)
    )
    exclusion_accounting.update(
        {
            "focal_candidate_target_work_rows": len(own_exclusions),
            "distinct_candidate_peer_target_work_rows": len(peer_exclusions),
            "distinct_candidate_network_target_work_rows": len(network_exclusions),
        }
    )
    author_table, author_accounting = _author_measurement_table(
        candidate_values,
        author_base,
        edges,
        peer_exclusions,
        own_exclusions,
        network_exclusions,
        target_table,
    )
    require(len(author_table) == len(author_base), "author row closure failed")
    parent._validate_unchanged(work_paths, work_stats)
    parent._validate_unchanged(event_paths, event_stats)

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent)
    )
    os.chmod(temporary, 0o700)
    try:
        target_artifact = _write_parquet(
            target_table,
            temporary / "target_work_incumbency.parquet",
        )
        edge_artifact = _write_parquet(
            edge_table,
            temporary / "prior_target_peer_edges.parquet",
            compression_level=6,
        )
        author_artifact = _write_parquet(
            author_table,
            temporary / "author_leaveout_prior_peer_base.parquet",
            compression_level=6,
        )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA,
            "label": config["label"],
            "status": STATUS,
            "scope": {
                "exact_certified_episode_rows": 4,
                "incumbent_target_work_profiles_built": True,
                "focal_leaveouts_built": True,
                "prior_target_peer_edges_built": True,
                "network_leaveouts_built": True,
                "graph_team_cap_profiles": [10, 50, 100],
                "team_cap_selected": False,
                "all_team_graph_materialized": False,
                "raw_authorship_source_read": False,
                "orcid_stable_subset_built": False,
                "same_affiliation_bridge_built": False,
                "peer_visible_success_built": False,
                "outcomes_read": False,
                "entry_or_persistence_rows_read": False,
                "final_candidate_eligibility_established": False,
                "independent_verification_complete": False,
                "d0_complete": False,
                "model_fitting_performed": False,
                "g4_rows_read": False,
                "api_requests_created": 0,
                "gpu_used": False,
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "repository_bindings": {
                "gate_state": gate_binding,
                "measurement_abi": abi_binding,
            },
            "parent_binding": parent_binding,
            "normalized_input_bindings": {
                "cutoff_work_metadata": work_binding,
                "author_history_events": event_binding,
            },
            "episodes": episodes,
            "candidate_author_rows_by_episode": candidate_accounting,
            "window_work_accounting": window_accounting,
            "target_incidence_accounting": target_incidence_accounting,
            "target_work_accounting": {
                **target_accounting,
                "artifact": target_artifact,
            },
            "prior_event_accounting": prior_event_accounting,
            "prior_target_peer_edge_accounting": {
                **edge_accounting,
                "artifact": edge_artifact,
            },
            "exclusion_accounting": exclusion_accounting,
            "author_measurement_accounting": {
                **author_accounting,
                "artifact": author_artifact,
            },
            "deferred_components": config["deferred_components"],
            "authorization": {
                "independent_verification": "REQUIRED_NEXT",
                "raw_authorship_orcid_affiliation_successor": "PAUSED_PENDING_INDEPENDENT_VERIFICATION",
                "production_pivot_classification": "NOT_AUTHORIZED",
                "discovery_outcome_opening": "NOT_AUTHORIZED",
                "final_candidate_eligibility": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
            "execution": {
                "workload_class": "CPU_IO",
                "workers": workers,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
                "config_sha256": _sha256(config_path),
                "builder_relative_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
                "builder_sha256": _sha256(Path(__file__).resolve()),
                "producer_join_strategy": "NUMPY_WORK_GROUP_TARGET_AUTHOR_FILTERED_PAIR_REDUCTION",
            },
            "claim_boundary": config["claim_boundary"],
        }
        report_path = temporary / "certified_leaveout_prior_peer.json"
        _write_json(report_path, report)
        markdown_path = temporary / "CERTIFIED_LEAVEOUT_PRIOR_PEER.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts = [
            target_artifact,
            edge_artifact,
            author_artifact,
            {
                "relative_path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            },
            {
                "relative_path": markdown_path.name,
                "bytes": markdown_path.stat().st_size,
                "sha256": _sha256(markdown_path),
            },
        ]
        artifacts.sort(key=lambda item: str(item["relative_path"]))
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "status": STATUS,
            "artifact_count": len(artifacts),
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_bytes(
                    [
                        (item["relative_path"], item["bytes"], item["sha256"])
                        for item in artifacts
                    ]
                )
            ).hexdigest(),
            "incumbent_target_work_profiles_built": True,
            "focal_leaveouts_built": True,
            "prior_target_peer_edges_built": True,
            "network_leaveouts_built": True,
            "team_cap_selected": False,
            "raw_authorship_source_read": False,
            "orcid_stable_subset_built": False,
            "same_affiliation_bridge_built": False,
            "independent_verification_complete": False,
            "production_pivot_classification_authorized": False,
            "discovery_outcome_opening_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "outcomes_read": False,
            "gpu_used": False,
            "receipt_written_last": True,
        }
        _write_json(
            temporary / "certified_leaveout_prior_peer.outputs.json",
            receipt,
        )
        os.replace(temporary, output)
        temporary = None
        return report
    finally:
        _cleanup_temporary(temporary, output=output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--parent-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build(
        config_path=args.config,
        parent_root=args.parent_root,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output=args.output,
        workers=args.workers,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "target_work_accounting": report["target_work_accounting"],
                "prior_target_peer_edge_accounting": report[
                    "prior_target_peer_edge_accounting"
                ],
                "author_measurement_accounting": report[
                    "author_measurement_accounting"
                ],
                "execution": report["execution"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
