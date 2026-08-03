#!/usr/bin/env python3
"""Build the exact local 96-title T12505 Medicine-rank-4 sample."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

import build_pivot_feedback_d0_t10543_targeted_title_sample_v1 as base


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_t12505_rank4_targeted_title_sample_v1.yaml"
)
DEFAULT_WORK_ROOT = base.DEFAULT_WORK_ROOT
DEFAULT_EVENT_ROOT = base.DEFAULT_EVENT_ROOT
DEFAULT_OPENALEX_ROOT = base.DEFAULT_OPENALEX_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t12505-rank4-targeted-title-sample-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t12505_rank4_targeted_title_sample_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t12505_rank4_targeted_title_sample_outputs.v1"
SAMPLE_SCHEMA = "pivot_feedback_d0_t12505_rank4_targeted_title_sample.v1"
STATUS = "PASS_EXACT_T12505_RANK4_96_TITLE_LOCAL_SAMPLE"


class T12505Rank4TargetedSampleBuildError(RuntimeError):
    """Fail-closed T12505 rank-4 targeted-sample build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T12505Rank4TargetedSampleBuildError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _dotted(document: Mapping[str, Any], key: str) -> Any:
    value: Any = document
    for part in key.split("."):
        value = _mapping(value, part)[part]
    return value


def _load_config(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status")
        == "FROZEN_EXACT_T12505_RANK4_96_TITLE_LOCAL_SAMPLE_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = _repo_file(parent)
    parent_document = _read_yaml(parent_path)
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    helpers = _mapping(config.get("implementation_helpers"), "helpers")
    helper_hashes = {
        key: _sha256(_repo_file(_mapping(helpers.get(key), key)))
        for key in helpers
    }
    target = _mapping(config.get("target"), "target")
    require(
        (
            target["field_id"],
            target["field_candidate_rank"],
            target["topic_id"],
            target["entry_year"],
        )
        == (27, 4, 12505, 1993),
        "target",
    )
    require(
        config["sampling"]["rows_per_window"] == 32
        and config["sampling"]["total_rows"] == 96,
        "sampling",
    )
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "general_raw_title_scan",
        "full_candidate_title_copy",
        "api_request_creation",
        "api_submission",
        "production_content_disposition",
        "selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config, {
        "parent_sha256": _sha256(parent_path),
        **{f"helper_{key}_sha256": value for key, value in helper_hashes.items()},
    }


def _window(year: int) -> int:
    if year <= 1986:
        return 0
    if year <= 1989:
        return 1
    require(year <= 1992, "window year")
    return 2


def _scan_work_candidates(
    pair: tuple[int, Path],
    *,
    included_types: np.ndarray,
    allowed_dispositions: np.ndarray,
) -> dict[str, Any]:
    index, work_path = pair
    work = pq.read_table(
        work_path,
        columns=[
            "work_id_numeric",
            "publication_year",
            "temporal_disposition",
            "work_type_code",
            "is_paratext",
            "is_retracted",
            "declared_authors_count",
            "primary_topic_id",
            "primary_field_id",
        ],
    )
    work_id = work["work_id_numeric"].combine_chunks().to_numpy(zero_copy_only=False)
    require(bool(np.all(work_id % base.PARTITIONS == index)), "work partition")
    year_array = work["publication_year"].combine_chunks()
    year_valid = year_array.is_valid().to_numpy(zero_copy_only=False)
    year = (
        year_array.fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.uint16, copy=False)
    )
    disposition = (
        work["temporal_disposition"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    type_code = work["work_type_code"].combine_chunks().to_numpy(zero_copy_only=False)
    is_paratext = work["is_paratext"].combine_chunks().to_numpy(zero_copy_only=False)
    retracted_array = work["is_retracted"].combine_chunks()
    retracted_valid = retracted_array.is_valid().to_numpy(zero_copy_only=False)
    is_retracted = retracted_array.fill_null(False).to_numpy(zero_copy_only=False)
    team = (
        work["declared_authors_count"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    topic_array = work["primary_topic_id"].combine_chunks()
    topic_valid = topic_array.is_valid().to_numpy(zero_copy_only=False)
    topic = (
        topic_array.fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.uint32, copy=False)
    )
    field_array = work["primary_field_id"].combine_chunks()
    field_valid = field_array.is_valid().to_numpy(zero_copy_only=False)
    field = (
        field_array.fill_null(0)
        .to_numpy(zero_copy_only=False)
        .astype(np.uint8, copy=False)
    )
    target_mask = (
        year_valid
        & topic_valid
        & field_valid
        & (topic == 12505)
        & (field == 27)
        & (year > 0)
        & (year <= 1992)
        & np.isin(disposition, allowed_dispositions)
        & np.isin(type_code, included_types)
        & ~is_paratext
        & ~(retracted_valid & is_retracted)
    )
    selected_indices = np.flatnonzero(target_mask)
    return {
        "rows": [
            {
                "work_numeric_id": int(work_id[source_index]),
                "publication_year": int(year[source_index]),
                "temporal_disposition": int(disposition[source_index]),
                "work_type_code": int(type_code[source_index]),
                "declared_authors_count": int(team[source_index]),
            }
            for source_index in selected_indices.tolist()
        ],
        "work_rows": len(work),
        "eligible_prelink_rows": len(selected_indices),
    }


def _select(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: str,
    per_window: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    for raw in rows:
        row = dict(raw)
        window = _window(int(row["publication_year"]))
        preimage = (
            f"{seed}\0{27}\0{12505}\0{1993}\0{window}\0"
            f"{int(row['work_numeric_id'])}"
        )
        row["window_code"] = window
        row["selection_sha256"] = hashlib.sha256(preimage.encode()).hexdigest()
        groups[window].append(row)
    selected: list[dict[str, Any]] = []
    pool: dict[str, Any] = {"window_pool_rows": {}, "selected_rows": {}}
    for window in (0, 1, 2):
        ordered = sorted(
            groups[window],
            key=lambda row: (row["selection_sha256"], row["work_numeric_id"]),
        )
        require(len(ordered) >= per_window, f"insufficient window: {window}")
        selected.extend(ordered[:per_window])
        pool["window_pool_rows"][str(window)] = len(ordered)
        pool["selected_rows"][str(window)] = per_window
    selected.sort(key=lambda row: (row["window_code"], row["selection_sha256"]))
    require(
        len(selected) == len({row["work_numeric_id"] for row in selected}) == 96,
        "sample",
    )
    return selected, pool


def _rank4_table(
    selected: Sequence[Mapping[str, Any]],
    found: Mapping[int, Mapping[str, Any]],
) -> pa.Table:
    adapted: dict[int, dict[str, Any]] = {}
    for work_id, raw_value in found.items():
        raw = dict(raw_value)
        require(
            raw["primary_topic__id"] == "https://openalex.org/T12505",
            "raw topic",
        )
        require(
            raw["primary_topic__field__id"] == "https://openalex.org/fields/27",
            "raw field",
        )
        raw["primary_topic__id"] = "https://openalex.org/T10543"
        adapted[work_id] = raw
    table = base._sample_table(selected, adapted)
    rank_index = table.schema.get_field_index("field_candidate_rank")
    topic_index = table.schema.get_field_index("topic_id")
    table = table.set_column(
        rank_index,
        pa.field("field_candidate_rank", pa.uint32(), nullable=False),
        pa.array([4] * len(table), type=pa.uint32()),
    )
    table = table.set_column(
        topic_index,
        pa.field("topic_id", pa.uint32(), nullable=False),
        pa.array([12505] * len(table), type=pa.uint32()),
    )
    return table.replace_schema_metadata({b"schema_version": SAMPLE_SCHEMA.encode()})


def _markdown(pool: Mapping[str, Any], raw_scan: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# T12505 Medicine rank-4 exact targeted title sample",
            "",
            f"Status: `{STATUS}`",
            "",
            f"- linked earlier pool: {pool['window_pool_rows']['0']:,}",
            f"- linked baseline pool: {pool['window_pool_rows']['1']:,}",
            f"- linked signal pool: {pool['window_pool_rows']['2']:,}",
            "- selected rows: 32 + 32 + 32 = 96",
            f"- raw files checked for exact IDs: {raw_scan['main_files_scanned']}",
            f"- content row groups read: {raw_scan['content_row_groups_read']}",
            "",
            "No network, credential, API, production disposition, certificate,",
            "reselection, outcome, or model operation occurred.",
            "",
        ]
    )


def build(
    *,
    config_path: Path = DEFAULT_CONFIG,
    work_root: Path = DEFAULT_WORK_ROOT,
    event_root: Path = DEFAULT_EVENT_ROOT,
    openalex_root: Path = DEFAULT_OPENALEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    workers: int = 16,
) -> dict[str, Any]:
    require(workers >= 1, "workers")
    config_path = config_path.resolve(strict=True)
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config, bindings = _load_config(config_path)
    normalized = _mapping(config.get("normalized_inputs"), "normalized inputs")
    work_paths, work_tree = base._validate_partition_root(
        work_root,
        _mapping(normalized.get("cutoff_work_metadata"), "work input"),
        workers=workers,
    )
    event_paths, event_tree = base._validate_partition_root(
        event_root,
        _mapping(normalized.get("author_history_events"), "event input"),
        workers=workers,
    )
    included_types = np.array(config["target"]["included_work_type_codes"], dtype=np.uint8)
    dispositions = np.array(
        config["target"]["allowed_temporal_dispositions"],
        dtype=np.uint8,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        work_parts = list(
            executor.map(
                lambda pair: _scan_work_candidates(
                    pair,
                    included_types=included_types,
                    allowed_dispositions=dispositions,
                ),
                enumerate(work_paths),
            )
        )
    prelink_all = [row for part in work_parts for row in part["rows"]]
    require(prelink_all, "no eligible work candidates")
    prelink_all.sort(key=lambda row: int(row["work_numeric_id"]))
    candidate_ids = np.array(
        [row["work_numeric_id"] for row in prelink_all],
        dtype=np.uint64,
    )
    require(len(candidate_ids) == len(np.unique(candidate_ids)), "candidate IDs")
    candidate_year = np.array(
        [row["publication_year"] for row in prelink_all],
        dtype=np.uint16,
    )
    candidate_disposition = np.array(
        [row["temporal_disposition"] for row in prelink_all],
        dtype=np.uint8,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        event_parts = list(
            executor.map(
                lambda pair: base._scan_event_links(
                    pair,
                    candidate_ids=candidate_ids,
                    candidate_year=candidate_year,
                    candidate_disposition=candidate_disposition,
                ),
                enumerate(event_paths),
            )
        )
    matched = [
        part["matched_work_ids"]
        for part in event_parts
        if len(part["matched_work_ids"])
    ]
    require(matched, "no matching author events")
    linked_ids, linked_counts = np.unique(np.concatenate(matched), return_counts=True)
    prelink_by_id = {row["work_numeric_id"]: row for row in prelink_all}
    pool_rows = [
        {
            **prelink_by_id[int(work_id)],
            "matching_author_event_rows": int(count),
        }
        for work_id, count in zip(
            linked_ids.tolist(),
            linked_counts.tolist(),
            strict=True,
        )
    ]
    selected, pool = _select(
        pool_rows,
        seed=str(config["sampling"]["seed"]),
        per_window=int(config["sampling"]["rows_per_window"]),
    )
    pool.update(
        {
            "schema_version": "pivot_feedback_d0_t12505_rank4_candidate_pool_summary.v1",
            "field_id": 27,
            "field_candidate_rank": 4,
            "topic_id": 12505,
            "entry_year": 1993,
            "normalized_work_rows_scanned": sum(
                part["work_rows"] for part in work_parts
            ),
            "normalized_event_rows_scanned": sum(
                part["event_rows"] for part in event_parts
            ),
            "eligible_prelink_rows": sum(
                part["eligible_prelink_rows"] for part in work_parts
            ),
            "author_linked_pool_rows": len(pool_rows),
            "recent_six_year_author_linked_rows": (
                int(pool["window_pool_rows"]["1"])
                + int(pool["window_pool_rows"]["2"])
            ),
        }
    )
    raw_files, columns, footer_rows = base._validate_raw_source(config, openalex_root)
    found, raw_scan = base._lookup_selected(
        raw_files,
        columns,
        selected,
        workers=workers,
    )
    table = _rank4_table(selected, found)
    pool["selected_title_rows"] = len(table)
    pool["missing_title_rows"] = int(pc.sum(pc.is_null(table["title"])).as_py())
    pool["raw_main_footer_rows"] = footer_rows
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["sample"] = base._write_parquet(
            table,
            temporary / str(contract["sample_basename"]),
        )
        pool_path = temporary / str(contract["pool_summary_basename"])
        base._write_private(pool_path, _canonical_json_bytes(pool))
        artifacts["pool_summary"] = {
            "basename": pool_path.name,
            "bytes": pool_path.stat().st_size,
            "sha256": _sha256(pool_path),
        }
        report_path = temporary / str(contract["report_basename"])
        base._write_private(report_path, _markdown(pool, raw_scan).encode())
        artifacts["report"] = {
            "basename": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": _sha256(report_path),
        }
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "producer_relative_path": (
                Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
            ),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "repository_bindings": bindings,
            "normalized_input_evidence": {
                "work_root_basename": work_root.name,
                "work_selected_tree_sha256": work_tree,
                "event_root_basename": event_root.name,
                "event_selected_tree_sha256": event_tree,
            },
            "pool": pool,
            "raw_scan": raw_scan,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "target_topic_ids_read": [12505],
            "selected_title_ids_read": 96,
            "general_raw_title_scan": False,
            "full_candidate_title_copy": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "content_disposition_created": False,
            "selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        receipt_path = temporary / str(contract["receipt_basename"])
        base._write_private(receipt_path, _canonical_json_bytes(receipt))
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--openalex-root", type=Path, default=DEFAULT_OPENALEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(
            config_path=args.config,
            work_root=args.work_root,
            event_root=args.event_root,
            openalex_root=args.openalex_root,
            output_root=args.output_root,
            workers=args.workers,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        T12505Rank4TargetedSampleBuildError,
        base.T10543TargetedSampleBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_T12505_RANK4_TARGETED_SAMPLE_V1", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
