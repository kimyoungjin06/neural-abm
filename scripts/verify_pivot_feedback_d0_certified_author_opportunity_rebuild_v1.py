#!/usr/bin/env python3
"""Independently replay the certified D0 author-opportunity rebuild work-first."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_SCHEMA = "pivot_feedback_d0_certified_author_opportunity_rebuild_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_certified_author_opportunity_rebuild_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_certified_author_opportunity_rebuild_outputs.v1"
WORK_SCHEMA_VERSION = "pivot_feedback_d0_certified_selected_predecision_work.v1"
AUTHOR_SCHEMA_VERSION = (
    "pivot_feedback_d0_certified_pre_entry_author_opportunity_base.v1"
)
VERIFY_SCHEMA = (
    "pivot_feedback_d0_certified_author_opportunity_rebuild_verification.v1"
)
OUTPUT_STATUS = (
    "PASS_EXACT_FOUR_OUTCOME_BLIND_MEASUREMENT_ABI_AUTHOR_OPPORTUNITY_REBUILD"
)
VERIFY_STATUS = (
    "PASS_INDEPENDENT_EXACT_FOUR_OUTCOME_BLIND_MEASUREMENT_ABI_"
    "AUTHOR_OPPORTUNITY_REBUILD"
)
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
EXPECTED_FILES = {
    "CERTIFIED_AUTHOR_OPPORTUNITY_REBUILD.md",
    "certified_author_opportunity_rebuild.json",
    "certified_author_opportunity_rebuild.outputs.json",
    "certified_episode_selected_predecision_works.parquet",
    "certified_pre_entry_author_opportunity_base.parquet",
}
WORK_INPUT_SCHEMA = pa.schema(
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
EVENT_INPUT_SCHEMA = pa.schema(
    [
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("temporal_disposition", pa.uint8(), nullable=False),
        pa.field("declared_authors_count", pa.uint32(), nullable=False),
        pa.field("primary_field_id", pa.uint8(), nullable=False),
    ]
)
PRELINK_DTYPE = np.dtype(
    [
        ("work_numeric_id", "<u8"),
        ("publication_year", "<u2"),
        ("field_id", "u1"),
        ("topic_id", "<u4"),
        ("work_type_code", "u1"),
        ("declared_authors_count", "<u2"),
        ("is_retracted_null", "u1"),
        ("temporal_disposition", "u1"),
    ],
    align=False,
)
WORK_OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("work_numeric_id", pa.uint64(), nullable=False),
        pa.field("publication_year", pa.uint16(), nullable=False),
        pa.field("work_type_code", pa.uint8(), nullable=False),
        pa.field("declared_authors_count", pa.uint16(), nullable=False),
        pa.field("is_retracted_null", pa.bool_(), nullable=False),
        pa.field("window_code", pa.uint8(), nullable=False),
    ],
    metadata={b"schema_version": WORK_SCHEMA_VERSION.encode()},
)


def _author_schema() -> pa.Schema:
    fields = [
        pa.field("certified_episode_index", pa.uint8(), nullable=False),
        pa.field("field_id", pa.uint8(), nullable=False),
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.field("author_numeric_id", pa.uint64(), nullable=False),
    ]
    for stem in (
        "pre_entry_all_field_work_count",
        "pre_entry_focal_field_work_count",
        "portfolio_window_all_field_work_count",
        "portfolio_window_focal_field_work_count",
    ):
        for suffix in ("cap_10", "cap_50", "cap_100", "all"):
            fields.append(pa.field(f"{stem}_{suffix}", pa.uint32(), nullable=False))
    fields.extend(
        [
            pa.field("pre_entry_target_topic_work_count", pa.uint32(), nullable=False),
            pa.field("pre_signal_target_topic_work_count", pa.uint32(), nullable=False),
            pa.field("first_target_topic_year", pa.uint16(), nullable=True),
            pa.field("observed_no_prior_target_topic", pa.bool_(), nullable=False),
            pa.field("incumbent_before_signal_window", pa.bool_(), nullable=False),
        ]
    )
    return pa.schema(
        fields,
        metadata={b"schema_version": AUTHOR_SCHEMA_VERSION.encode()},
    )


AUTHOR_OUTPUT_SCHEMA = _author_schema()


class CertifiedAuthorOpportunityVerificationError(RuntimeError):
    """Fail-closed independent verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CertifiedAuthorOpportunityVerificationError(message)


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
        raise CertifiedAuthorOpportunityVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise CertifiedAuthorOpportunityVerificationError(
            f"invalid {label}: {error}"
        ) from error


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / pure).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        relative = item.get("relative_path")
        require(isinstance(relative, str) and relative, "bad artifact path")
        require(relative not in result, "duplicate receipt artifact")
        result[relative] = item
    return result


def _validate_output(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid output root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "output root mode mismatch")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set mismatch")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"output mode mismatch: {path.name}")
    receipt_path = root / "certified_author_opportunity_rebuild.outputs.json"
    receipt = _read_json(receipt_path, "rebuild receipt")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema mismatch")
    require(receipt.get("status") == OUTPUT_STATUS, "receipt status mismatch")
    require(receipt.get("artifact_count") == 4, "receipt artifact count mismatch")
    require(receipt.get("selected_work_manifest_built") is True, "work manifest absent")
    require(receipt.get("author_opportunity_base_built") is True, "author base absent")
    for key in (
        "independent_verification_complete",
        "leaveout_network_bridge_authorized",
        "discovery_outcome_opening_authorized",
        "model_fitting_authorized",
        "g4_open",
        "outcomes_read",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe receipt state: {key}")
    require(receipt.get("receipt_written_last") is True, "receipt order not declared")
    expected_artifacts = EXPECTED_FILES - {receipt_path.name}
    observed = []
    for relative, item in _artifact_map(receipt).items():
        require("/" not in relative and relative in expected_artifacts, "unexpected artifact")
        path = root / relative
        actual: dict[str, Any] = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if "rows" in item:
            parquet = pq.ParquetFile(path)
            actual["rows"] = parquet.metadata.num_rows
            actual["row_groups"] = parquet.metadata.num_row_groups
        require(actual == item, f"artifact differs: {relative}")
        observed.append(actual)
    observed.sort(key=lambda item: str(item["relative_path"]))
    require(
        {item["relative_path"] for item in observed} == expected_artifacts,
        "artifact closure mismatch",
    )
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(
            _canonical_bytes(
                [
                    (item["relative_path"], item["bytes"], item["sha256"])
                    for item in observed
                ]
            )
        ).hexdigest(),
        "artifact tree differs",
    )
    report = _read_json(
        root / "certified_author_opportunity_rebuild.json",
        "producer report",
    )
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == OUTPUT_STATUS, "report status mismatch")
    scope = _mapping(report.get("scope"), "report scope")
    require(scope.get("selected_work_manifest_built") is True, "report work manifest absent")
    require(scope.get("author_opportunity_base_built") is True, "report author base absent")
    for key in (
        "outcomes_read",
        "entry_or_persistence_rows_read",
        "realized_citation_rows_read",
        "final_candidate_eligibility_established",
        "leaveout_network_bridge_built",
        "independent_verification_complete",
        "d0_complete",
        "model_fitting_performed",
        "g4_rows_read",
        "gpu_used",
    ):
        require(scope.get(key) is False, f"unsafe report scope: {key}")
    require(scope.get("api_requests_created") == 0, "report API requests opened")
    require(scope.get("scientific_effect") == "NOT_ESTABLISHED", "effect overstated")
    return receipt, report, observed


def _validate_config(
    path: Path,
    report: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.resolve(strict=True)
    config = _read_yaml(path, path.name)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("status_before_execution") == "AUTHORIZED_BY_GATE_STATE_V37",
        "config authorization mismatch",
    )
    forbidden = _mapping(config.get("forbidden_inputs_and_outputs"), "forbidden boundary")
    require(forbidden and all(value is False for value in forbidden.values()), "forbidden boundary opened")
    for label in ("gate_state", "measurement_abi"):
        binding = _mapping(config["repository_bindings"][label], f"{label} binding")
        bound = _safe_repo_path(binding["relative_path"], f"{label} path")
        require(_sha256(bound) == binding["sha256"], f"{label} SHA mismatch")
        document = _read_yaml(bound, label)
        require(document.get("schema_version") == binding["required_schema_version"], f"{label} schema mismatch")
        require(document.get("status") == binding["required_status"], f"{label} status mismatch")
        for dotted, expected in _mapping(binding.get("required_fields", {}), f"{label} fields").items():
            observed = _nested(document, dotted)
            require(type(observed) is type(expected) and observed == expected, f"{label} field mismatch: {dotted}")
        require(
            report["repository_bindings"][label]
            == {
                "relative_path": binding["relative_path"],
                "sha256": binding["sha256"],
                "status": binding["required_status"],
            },
            f"reported {label} binding differs",
        )
    execution = _mapping(report.get("execution"), "execution")
    require(execution.get("config_relative_path") == path.relative_to(REPO_ROOT).as_posix(), "reported config path differs")
    require(execution.get("config_sha256") == _sha256(path), "reported config SHA differs")
    builder = _safe_repo_path(execution.get("builder_relative_path"), "reported builder")
    require(execution.get("builder_sha256") == _sha256(builder), "reported builder SHA differs")
    require("verify_pivot_feedback_d0_certified_author" not in builder.read_text(encoding="utf-8"), "producer imports verifier")
    projection = _mapping(config.get("measurement_projection"), "measurement projection")
    require([int(value) for value in projection["all_field_metric_field_ids"]] == list(range(11, 37)), "all-field closure differs")
    require([int(value) for value in projection["focal_field_ids"]] == [17, 27], "focal fields differ")
    episodes = [
        {
            "certified_episode_index": int(row["certified_episode_index"]),
            "field_id": int(row["field_id"]),
            "topic_id": int(row["topic_id"]),
            "entry_year": int(row["entry_year"]),
            "field_candidate_rank": int(row["field_candidate_rank"]),
            "subfield_id": int(row["subfield_id"]),
        }
        for row in config["certified_episode_selection"]["exact_rows"]
    ]
    require(
        [(row["certified_episode_index"], row["field_id"], row["topic_id"], row["entry_year"]) for row in episodes]
        == [(1, 17, 14430, 2003), (2, 17, 10579, 2012), (3, 27, 10151, 1993), (4, 27, 12421, 1953)],
        "episode contract differs",
    )
    reported_episodes = report["episodes"]
    require(
        [
            (int(row["certified_episode_index"]), int(row["field_id"]), int(row["topic_id"]), int(row["entry_year"]))
            for row in reported_episodes
        ]
        == [
            (row["certified_episode_index"], row["field_id"], row["topic_id"], row["entry_year"])
            for row in episodes
        ],
        "reported episodes differ",
    )
    return config, reported_episodes


def _validate_partitioned_input(
    root: Path,
    binding: Mapping[str, Any],
    *,
    schema: pa.Schema,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid input root")
    require(root.name == binding["expected_root_basename"], "input basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding["receipt_sha256"], "input receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(receipt.get("schema_version") == binding["receipt_schema_version"], "input receipt schema mismatch")
    require(receipt.get("status") == binding["required_status"], "input status mismatch")
    artifacts = {}
    for raw in _list(receipt.get("artifacts"), "input artifacts"):
        item = _mapping(raw, "input artifact")
        relative = item.get("relative_path")
        require(isinstance(relative, str) and relative not in artifacts, "bad input artifact")
        artifacts[relative] = item
    prefix = str(binding["artifact_prefix"])
    items = [item for relative, item in artifacts.items() if relative.startswith(prefix)]
    items.sort(key=lambda item: str(item["relative_path"]))
    require(len(items) == binding["artifact_count"] == PARTITIONS, "input partition count mismatch")
    paths = []
    stats = []
    for index, item in enumerate(items):
        relative = f"{prefix}part-{index:04d}.parquet"
        require(item["relative_path"] == relative, "input partition sequence mismatch")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root) and path.is_file() and not path.is_symlink(), "invalid input partition")
        require(pq.ParquetFile(path).schema_arrow == schema, "input schema mismatch")
        current = path.stat()
        require(current.st_size == item["bytes"], "input byte mismatch")
        paths.append(path)
        stats.append((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns))

    def rehash(pair: tuple[Path, Mapping[str, Any]]) -> tuple[str, int, str]:
        path, item = pair
        observed = _sha256(path)
        require(observed == item["sha256"], f"input SHA mismatch: {path.name}")
        return str(item["relative_path"]), int(item["bytes"]), observed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(rehash, zip(paths, items, strict=True)))
    if "artifact_bytes" in binding:
        require(sum(item[1] for item in verified) == binding["artifact_bytes"], "input byte total mismatch")
    return paths, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_prefix": prefix,
        "artifacts_rehashed": len(verified),
        "artifact_bytes": sum(item[1] for item in verified),
        "selected_tree_sha256": hashlib.sha256(_canonical_bytes(verified)).hexdigest(),
    }, stats


def _validate_unchanged(paths: Sequence[Path], stats: Sequence[tuple[int, int, int, int, int]]) -> None:
    observed = []
    for path in paths:
        current = path.stat()
        observed.append((current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns))
    require(observed == list(stats), "input changed during verification")


def _scan_prelink_partition(
    pair: tuple[int, Path],
    *,
    fields: np.ndarray,
    dispositions: np.ndarray,
    included_types: np.ndarray,
    latest_year: int,
) -> dict[str, Any]:
    index, path = pair
    table = pq.read_table(path)
    works = table["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(works % PARTITIONS == index)), f"work partition mismatch: {index}")
    year_array = table["publication_year"].combine_chunks()
    year_valid = year_array.is_valid().to_numpy(zero_copy_only=False)
    years = year_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint16, copy=False)
    observed_dispositions = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    types = table["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)
    paratext = table["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)
    retracted_array = table["is_retracted"].combine_chunks()
    retracted_valid = retracted_array.is_valid().to_numpy(zero_copy_only=False)
    retracted = retracted_array.fill_null(False).to_numpy(zero_copy_only=False)
    teams = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    topic_array = table["primary_topic_id"].combine_chunks()
    topic_valid = topic_array.is_valid().to_numpy(zero_copy_only=False)
    topics = topic_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint32, copy=False)
    field_array = table["primary_field_id"].combine_chunks()
    field_valid = field_array.is_valid().to_numpy(zero_copy_only=False)
    observed_fields = field_array.fill_null(0).to_numpy(zero_copy_only=False).astype(np.uint8, copy=False)
    scope = (
        year_valid
        & topic_valid
        & field_valid
        & np.isin(observed_fields, fields)
        & np.isin(observed_dispositions, dispositions)
        & (years > 0)
        & (years <= latest_year)
        & (topics > 0)
    )
    included = scope & np.isin(types, included_types)
    retracted_true = included & ~paratext & retracted_valid & retracted
    eligible = included & ~paratext & ~retracted_true
    indices = np.flatnonzero(eligible)
    records = np.empty(len(indices), dtype=PRELINK_DTYPE)
    records["work_numeric_id"] = works[indices]
    records["publication_year"] = years[indices]
    records["field_id"] = observed_fields[indices]
    records["topic_id"] = topics[indices]
    records["work_type_code"] = types[indices]
    records["declared_authors_count"] = teams[indices]
    records["is_retracted_null"] = (~retracted_valid[indices]).astype(np.uint8)
    records["temporal_disposition"] = observed_dispositions[indices]
    return {
        "rows": len(table),
        "scope_rows": int(scope.sum()),
        "included_type_rows": int(included.sum()),
        "paratext_excluded_rows": int((included & paratext).sum()),
        "retracted_true_excluded_rows": int(retracted_true.sum()),
        "eligible_prelink_rows": len(indices),
        "records": records,
    }


def _prelink_universe(
    paths: Sequence[Path],
    *,
    fields: Sequence[int],
    dispositions: Sequence[int],
    included_types: Sequence[int],
    latest_year: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, int]]:
    field_array = np.asarray(fields, dtype=np.uint8)
    disposition_array = np.asarray(dispositions, dtype=np.uint8)
    type_array = np.asarray(included_types, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        chunks = list(
            executor.map(
                lambda pair: _scan_prelink_partition(
                    pair,
                    fields=field_array,
                    dispositions=disposition_array,
                    included_types=type_array,
                    latest_year=latest_year,
                ),
                enumerate(paths),
            )
        )
    names = (
        "rows",
        "scope_rows",
        "included_type_rows",
        "paratext_excluded_rows",
        "retracted_true_excluded_rows",
        "eligible_prelink_rows",
    )
    accounting = {name: sum(int(item[name]) for item in chunks) for name in names}
    prelink = np.concatenate([item["records"] for item in chunks], dtype=PRELINK_DTYPE)
    del chunks
    order = np.argsort(prelink["work_numeric_id"], kind="stable")
    prelink = prelink[order]
    if len(prelink) > 1:
        require(not bool(np.any(prelink["work_numeric_id"][1:] == prelink["work_numeric_id"][:-1])), "prelink work duplicated")
    return prelink, accounting


def _membership(values: np.ndarray, sorted_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(sorted_ids, values)
    in_range = positions < len(sorted_ids)
    matched = np.zeros(len(values), dtype=bool)
    matched[in_range] = sorted_ids[positions[in_range]] == values[in_range]
    return matched, positions


def _counts(inverse: np.ndarray, mask: np.ndarray, size: int) -> np.ndarray:
    values = np.bincount(inverse[mask], minlength=size)
    require(bool(np.all(values <= np.iinfo(np.uint32).max)), "author count overflow")
    return values.astype(np.uint32, copy=False)


def _profiles(
    inverse: np.ndarray,
    mask: np.ndarray,
    teams: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected_inverse = inverse[mask]
    selected_teams = teams[mask]
    category = np.full(len(selected_teams), 3, dtype=np.uint8)
    category[selected_teams <= 100] = 2
    category[selected_teams <= 50] = 1
    category[selected_teams <= 10] = 0
    exact = np.bincount(
        selected_inverse.astype(np.int64, copy=False) * 4 + category,
        minlength=size * 4,
    ).reshape(size, 4)
    cumulative = np.cumsum(exact, axis=1, dtype=np.uint64)
    require(bool(np.all(cumulative <= np.iinfo(np.uint32).max)), "profile count overflow")
    cumulative = cumulative.astype(np.uint32, copy=False)
    return cumulative[:, 0], cumulative[:, 1], cumulative[:, 2], cumulative[:, 3]


def _expected_author_batches(
    authors: np.ndarray,
    works: np.ndarray,
    years: np.ndarray,
    teams: np.ndarray,
    fields: np.ndarray,
    *,
    episodes: Sequence[Mapping[str, Any]],
    target_ids: Mapping[int, np.ndarray],
) -> tuple[list[pa.Table], dict[int, dict[str, int]]]:
    unique_authors, inverse = np.unique(authors, return_inverse=True)
    size = len(unique_authors)
    batches = []
    accounting = {}
    for episode in episodes:
        episode_index = int(episode["certified_episode_index"])
        field_id = int(episode["field_id"])
        entry = int(episode["entry_year"])
        pre_entry = years < entry
        portfolio = pre_entry & (years >= entry - 3)
        focal = fields == field_id
        metrics = {}
        for stem, mask in (
            ("pre_entry_all_field_work_count", pre_entry),
            ("pre_entry_focal_field_work_count", pre_entry & focal),
            ("portfolio_window_all_field_work_count", portfolio),
            ("portfolio_window_focal_field_work_count", portfolio & focal),
        ):
            values = _profiles(inverse, mask, teams, size)
            for suffix, count in zip(("cap_10", "cap_50", "cap_100", "all"), values, strict=True):
                metrics[f"{stem}_{suffix}"] = count
        target, _ = _membership(works, target_ids[episode_index])
        require(not bool(np.any(target & (~pre_entry | ~focal))), "target event boundary differs")
        target_count = _counts(inverse, target, size)
        pre_signal = target & (years < entry - 6)
        pre_signal_count = _counts(inverse, pre_signal, size)
        first = np.full(size, np.iinfo(np.uint16).max, dtype=np.uint16)
        np.minimum.at(first, inverse[target], years[target].astype(np.uint16, copy=False))
        candidate = metrics["portfolio_window_focal_field_work_count_all"] > 0
        indices = np.flatnonzero(candidate)
        first_values = first[indices]
        arrays: dict[str, pa.Array] = {
            "certified_episode_index": pa.array(np.full(len(indices), episode_index), type=pa.uint8()),
            "field_id": pa.array(np.full(len(indices), field_id), type=pa.uint8()),
            "topic_id": pa.array(np.full(len(indices), int(episode["topic_id"])), type=pa.uint32()),
            "entry_year": pa.array(np.full(len(indices), entry), type=pa.uint16()),
            "author_numeric_id": pa.array(unique_authors[indices], type=pa.uint64()),
        }
        for name in AUTHOR_OUTPUT_SCHEMA.names[5:21]:
            arrays[name] = pa.array(metrics[name][indices], type=pa.uint32())
        arrays["pre_entry_target_topic_work_count"] = pa.array(target_count[indices], type=pa.uint32())
        arrays["pre_signal_target_topic_work_count"] = pa.array(pre_signal_count[indices], type=pa.uint32())
        arrays["first_target_topic_year"] = pa.array(first_values, mask=first_values == np.iinfo(np.uint16).max, type=pa.uint16())
        arrays["observed_no_prior_target_topic"] = pa.array(target_count[indices] == 0, type=pa.bool_())
        arrays["incumbent_before_signal_window"] = pa.array(pre_signal_count[indices] > 0, type=pa.bool_())
        batch = pa.Table.from_arrays([arrays[name] for name in AUTHOR_OUTPUT_SCHEMA.names], schema=AUTHOR_OUTPUT_SCHEMA)
        batches.append(batch)
        accounting[episode_index] = {
            "author_rows": len(batch),
            "observed_no_prior_target_topic_rows": int((target_count[indices] == 0).sum()),
            "prior_target_topic_rows": int((target_count[indices] > 0).sum()),
            "incumbent_before_signal_window_rows": int((pre_signal_count[indices] > 0).sum()),
            "portfolio_focal_work_events_all": int(
                metrics["portfolio_window_focal_field_work_count_all"][indices].sum(dtype=np.uint64)
            ),
        }
    return batches, accounting


def _selected_work_table(
    prelink: np.ndarray,
    linked_seen: np.ndarray,
    episodes: Sequence[Mapping[str, Any]],
) -> tuple[pa.Table, dict[str, Any]]:
    chunks = []
    by_episode = {}
    for episode in episodes:
        episode_index = int(episode["certified_episode_index"])
        entry = int(episode["entry_year"])
        keep = (
            linked_seen
            & (prelink["field_id"] == int(episode["field_id"]))
            & (prelink["topic_id"] == int(episode["topic_id"]))
            & (prelink["publication_year"] < entry)
        )
        values = prelink[keep]
        years = values["publication_year"]
        window = np.where(years <= entry - 7, 0, np.where(years <= entry - 4, 1, 2)).astype(np.uint8)
        chunks.append(
            pa.Table.from_arrays(
                [
                    pa.array(np.full(len(values), episode_index), type=pa.uint8()),
                    pa.array(values["field_id"], type=pa.uint8()),
                    pa.array(values["topic_id"], type=pa.uint32()),
                    pa.array(np.full(len(values), entry), type=pa.uint16()),
                    pa.array(values["work_numeric_id"], type=pa.uint64()),
                    pa.array(values["publication_year"], type=pa.uint16()),
                    pa.array(values["work_type_code"], type=pa.uint8()),
                    pa.array(values["declared_authors_count"], type=pa.uint16()),
                    pa.array(values["is_retracted_null"].astype(bool), type=pa.bool_()),
                    pa.array(window, type=pa.uint8()),
                ],
                schema=WORK_OUTPUT_SCHEMA,
            )
        )
        by_episode[str(episode_index)] = {
            "selected_work_rows": len(values),
            "pre_signal_work_rows": int((window == 0).sum()),
            "reference_window_work_rows": int((window == 1).sum()),
            "signal_window_work_rows": int((window == 2).sum()),
            "is_retracted_null_rows": int(values["is_retracted_null"].sum()),
        }
    table = pa.concat_tables(chunks).replace_schema_metadata(WORK_OUTPUT_SCHEMA.metadata)
    return table, {"selected_work_rows": len(table), "by_episode": by_episode}


def verify(
    *,
    config_path: Path,
    cutoff_root: Path,
    author_history_root: Path,
    output_root: Path,
    verification_report: Path,
    workers: int,
) -> dict[str, Any]:
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    receipt, report, output_artifacts = _validate_output(output_root)
    config, episodes = _validate_config(config_path, report)
    projection = config["measurement_projection"]
    fields = [int(value) for value in projection["all_field_metric_field_ids"]]
    dispositions = [int(value) for value in projection["allowed_temporal_dispositions"]]
    included_types = [int(value) for value in projection["inherited_work_type_codes"]]
    latest_year = int(projection["latest_predecision_year"])
    inputs = config["normalized_inputs"]
    work_paths, work_binding, work_stats = _validate_partitioned_input(
        cutoff_root,
        inputs["cutoff_work_metadata"],
        schema=WORK_INPUT_SCHEMA,
        workers=workers,
    )
    event_paths, event_binding, event_stats = _validate_partitioned_input(
        author_history_root,
        inputs["author_history_events"],
        schema=EVENT_INPUT_SCHEMA,
        workers=workers,
    )
    require(report["normalized_input_bindings"]["cutoff_work_metadata"] == work_binding, "reported work input differs")
    require(report["normalized_input_bindings"]["author_history_events"] == event_binding, "reported event input differs")
    prelink, work_accounting = _prelink_universe(
        work_paths,
        fields=fields,
        dispositions=dispositions,
        included_types=included_types,
        latest_year=latest_year,
        workers=workers,
    )
    require(work_accounting["rows"] == inputs["cutoff_work_metadata"]["rows"], "work row closure failed")
    target_ids = {}
    for episode in episodes:
        index = int(episode["certified_episode_index"])
        keep = (
            (prelink["field_id"] == int(episode["field_id"]))
            & (prelink["topic_id"] == int(episode["topic_id"]))
            & (prelink["publication_year"] < int(episode["entry_year"]))
        )
        target_ids[index] = prelink["work_numeric_id"][keep]
    linked_seen = np.zeros(len(prelink), dtype=bool)
    observed_author = pq.ParquetFile(
        output_root / "certified_pre_entry_author_opportunity_base.parquet"
    )
    require(observed_author.schema_arrow == AUTHOR_OUTPUT_SCHEMA, "author output schema mismatch")
    row_group = 0
    source_event_rows = 0
    eligible_event_rows = 0
    temporal_field_event_rows = 0
    duplicate_pairs = 0
    totals = {
        int(episode["certified_episode_index"]): {
            "author_rows": 0,
            "observed_no_prior_target_topic_rows": 0,
            "prior_target_topic_rows": 0,
            "incumbent_before_signal_window_rows": 0,
            "portfolio_focal_work_events_all": 0,
        }
        for episode in episodes
    }
    field_array = np.asarray(fields, dtype=np.uint8)
    disposition_array = np.asarray(dispositions, dtype=np.uint8)
    for partition_index, path in enumerate(event_paths):
        table = pq.read_table(path)
        source_event_rows += len(table)
        authors_all = table["author_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        require(bool(np.all(authors_all % PARTITIONS == partition_index)), f"author partition mismatch: {partition_index}")
        works_all = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
        years_all = table["publication_year"].combine_chunks().to_numpy(zero_copy_only=False)
        dispositions_all = table["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
        fields_all = table["primary_field_id"].combine_chunks().to_numpy(zero_copy_only=False)
        teams_all = table["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
        temporal = (
            np.isin(fields_all, field_array)
            & np.isin(dispositions_all, disposition_array)
            & (years_all > 0)
            & (years_all <= latest_year)
        )
        temporal_field_event_rows += int(temporal.sum())
        temporal_authors = authors_all[temporal]
        temporal_works = works_all[temporal]
        if len(temporal_works):
            order = np.lexsort((temporal_works, temporal_authors))
            duplicate_pairs += int(
                (
                    (temporal_authors[order][1:] == temporal_authors[order][:-1])
                    & (temporal_works[order][1:] == temporal_works[order][:-1])
                ).sum()
            )
        eligible, positions = _membership(works_all, prelink["work_numeric_id"])
        source = np.flatnonzero(eligible)
        matched = positions[eligible]
        if len(source):
            require(bool(np.all(years_all[source] == prelink["publication_year"][matched])), "event/work year mismatch")
            require(bool(np.all(fields_all[source] == prelink["field_id"][matched])), "event/work field mismatch")
            require(bool(np.all(dispositions_all[source] == prelink["temporal_disposition"][matched])), "event/work disposition mismatch")
            require(bool(np.all(teams_all[source] == prelink["declared_authors_count"][matched])), "event/work team mismatch")
            linked_seen[matched] = True
        eligible_event_rows += len(source)
        batches, accounting = _expected_author_batches(
            authors_all[eligible],
            works_all[eligible],
            years_all[eligible],
            teams_all[eligible],
            fields_all[eligible],
            episodes=episodes,
            target_ids=target_ids,
        )
        for batch in batches:
            if len(batch):
                require(row_group < observed_author.metadata.num_row_groups, "author output has too few row groups")
                observed = observed_author.read_row_group(row_group)
                require(batch.equals(observed, check_metadata=True), f"author replay differs at row group {row_group}")
                row_group += 1
        for episode_index, values in accounting.items():
            for key, value in values.items():
                totals[episode_index][key] += int(value)
    require(row_group == observed_author.metadata.num_row_groups, "author output has extra row groups")
    require(source_event_rows == inputs["author_history_events"]["rows"], "event row closure failed")
    require(duplicate_pairs == 0, "duplicate author-work events found")
    expected_selected, selected_accounting = _selected_work_table(prelink, linked_seen, episodes)
    observed_selected = pq.read_table(
        output_root / "certified_episode_selected_predecision_works.parquet"
    )
    require(observed_selected.schema == WORK_OUTPUT_SCHEMA, "selected output schema mismatch")
    require(expected_selected.equals(observed_selected, check_metadata=True), "selected-work replay differs")
    linked_rows = int(linked_seen.sum())
    work_replay = {
        **work_accounting,
        "linked_rows": linked_rows,
        "zero_valid_author_token_rows": len(prelink) - linked_rows,
        "is_retracted_null_linked_rows": int(prelink["is_retracted_null"][linked_seen].sum()),
    }
    require(report["work_abi_accounting"] == work_replay, "reported work ABI accounting differs")
    require(report["selected_work_accounting"]["selected_work_rows"] == selected_accounting["selected_work_rows"], "reported selected row count differs")
    require(report["selected_work_accounting"]["by_episode"] == selected_accounting["by_episode"], "reported selected episode accounting differs")
    author_replay = {
        "source_rows_scanned": source_event_rows,
        "eligible_author_work_event_rows": eligible_event_rows,
        "by_episode": {str(key): value for key, value in totals.items()},
    }
    require(report["author_opportunity_accounting"]["source_rows_scanned"] == source_event_rows, "reported event scan differs")
    require(report["author_opportunity_accounting"]["eligible_author_work_event_rows"] == eligible_event_rows, "reported eligible events differ")
    require(report["author_opportunity_accounting"]["by_episode"] == author_replay["by_episode"], "reported author accounting differs")
    require(report["author_event_index_accounting"]["event_rows_scanned"] == source_event_rows, "reported event index scan differs")
    require(report["author_event_index_accounting"]["abi_temporal_field_event_rows"] == temporal_field_event_rows, "reported temporal event rows differ")
    require(report["author_event_index_accounting"]["duplicate_author_work_event_rows"] == duplicate_pairs, "reported duplicate rows differ")
    _validate_unchanged(work_paths, work_stats)
    _validate_unchanged(event_paths, event_stats)
    result = {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "output_root_basename": output_root.resolve().name,
        "output_receipt_sha256": _sha256(
            output_root / "certified_author_opportunity_rebuild.outputs.json"
        ),
        "output_artifacts": output_artifacts,
        "replay_strategy": "WORK_FIRST_PRELINK_ELIGIBILITY_THEN_EVENT_LINKAGE_AND_ROW_GROUP_COMPARISON",
        "replay_accounting": {
            "cutoff_work_rows_scanned": work_accounting["rows"],
            "eligible_prelink_work_rows": len(prelink),
            "eligible_author_linked_work_rows": linked_rows,
            "author_event_rows_scanned": source_event_rows,
            "eligible_author_work_event_rows": eligible_event_rows,
            "selected_work_rows": len(expected_selected),
            "author_rows": observed_author.metadata.num_rows,
            "author_row_groups_compared": row_group,
            "by_episode": {
                key: {
                    "selected_work_rows": selected_accounting["by_episode"][key]["selected_work_rows"],
                    "author_rows": totals[int(key)]["author_rows"],
                }
                for key in sorted(selected_accounting["by_episode"])
            },
        },
        "scope": {
            "independent_replay_complete": True,
            "selected_work_manifest_verified": True,
            "author_opportunity_base_verified": True,
            "producer_imported": False,
            "outcomes_read": False,
            "entry_or_persistence_rows_read": False,
            "leaveout_network_bridge_built": False,
            "d0_complete": False,
            "model_fitting_performed": False,
            "g4_rows_read": False,
            "api_requests_created": 0,
            "gpu_used": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "authorization": {
            "gate_state_successor": "AUTHORIZED",
            "leaveout_network_bridge_build": "MAY_BE_AUTHORIZED_ONLY_BY_SUCCESSOR_GATE",
            "production_pivot_classification": "NOT_AUTHORIZED",
            "discovery_outcome_opening": "NOT_AUTHORIZED",
            "model_fitting": "NOT_AUTHORIZED",
            "g4_open": "NOT_AUTHORIZED",
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "verifier": {
            "relative_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256(Path(__file__).resolve()),
            "config_relative_path": config_path.resolve().relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(config_path.resolve()),
            "workers": workers,
        },
        "claim_boundary": (
            "The exact four certified predecision selected-work rows and "
            "outcome-blind author-opportunity base replayed independently from "
            "the frozen normalized inputs under the scholarly-work ABI. This "
            "does not establish natural-person identity, historical topic "
            "observability, final eligibility, leave-out or peer exposure, "
            "entry or persistence outcomes, an association, causality, D0 "
            "completion, model fitting, G4, or a scientific effect."
        ),
    }
    verification_report = verification_report.resolve()
    require(not verification_report.exists(), "verification report already exists")
    require(verification_report.parent.is_dir(), "verification report parent missing")
    require(
        verification_report.parent != output_root.resolve(),
        "verification report must be outside output root",
    )
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        verification_report,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        verification_report.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--author-history-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = verify(
        config_path=args.config,
        cutoff_root=args.cutoff_root,
        author_history_root=args.author_history_root,
        output_root=args.output_root,
        verification_report=args.verification_report,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
