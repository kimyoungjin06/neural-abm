#!/usr/bin/env python3
"""Build the exact local 96-title T10543 Medicine-rank-3 sample."""

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
import yaml

import build_pivot_feedback_d0_t10543_targeted_title_sample_v1 as base


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/"
    "pivot_feedback_d0_t10543_rank3_targeted_title_sample_v1.yaml"
)
DEFAULT_WORK_ROOT = base.DEFAULT_WORK_ROOT
DEFAULT_EVENT_ROOT = base.DEFAULT_EVENT_ROOT
DEFAULT_OPENALEX_ROOT = base.DEFAULT_OPENALEX_ROOT
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t10543-rank3-targeted-title-sample-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t10543_rank3_targeted_title_sample_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t10543_rank3_targeted_title_sample_outputs.v1"
SAMPLE_SCHEMA = "pivot_feedback_d0_t10543_rank3_targeted_title_sample.v1"
STATUS = "PASS_EXACT_T10543_RANK3_96_TITLE_LOCAL_SAMPLE"


class T10543Rank3TargetedSampleBuildError(RuntimeError):
    """Fail-closed T10543 rank-3 targeted-sample build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T10543Rank3TargetedSampleBuildError(message)


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
        == "FROZEN_EXACT_T10543_RANK3_96_TITLE_LOCAL_SAMPLE_AUTHORIZED",
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
        == (27, 3, 10543, 1986),
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
    if year <= 1979:
        return 0
    if year <= 1982:
        return 1
    require(year <= 1985, "window year")
    return 2


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
            f"{seed}\0{27}\0{10543}\0{1986}\0{window}\0"
            f"{int(row['work_numeric_id'])}"
        )
        row["window_code"] = window
        row["selection_sha256"] = hashlib.sha256(preimage.encode()).hexdigest()
        groups[window].append(row)
    selected: list[dict[str, Any]] = []
    pool = {"window_pool_rows": {}, "selected_rows": {}}
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
    require(len(selected) == len({row["work_numeric_id"] for row in selected}) == 96, "sample")
    return selected, pool


def _rank3_table(
    selected: Sequence[Mapping[str, Any]],
    found: Mapping[int, Mapping[str, Any]],
) -> pa.Table:
    table = base._sample_table(selected, found)
    rank_index = table.schema.get_field_index("field_candidate_rank")
    entry_index = table.schema.get_field_index("entry_year")
    table = table.set_column(
        rank_index,
        pa.field("field_candidate_rank", pa.uint32(), nullable=False),
        pa.array([3] * len(table), type=pa.uint32()),
    )
    table = table.set_column(
        entry_index,
        pa.field("entry_year", pa.uint16(), nullable=False),
        pa.array([1986] * len(table), type=pa.uint16()),
    )
    return table.replace_schema_metadata({b"schema_version": SAMPLE_SCHEMA.encode()})


def _markdown(pool: Mapping[str, Any], raw_scan: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# T10543 Medicine rank-3 exact targeted title sample",
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
                lambda pair: base._scan_work_candidates(
                    pair,
                    included_types=included_types,
                    allowed_dispositions=dispositions,
                ),
                enumerate(work_paths),
            )
        )
    prelink_all = [row for part in work_parts for row in part["rows"]]
    prelink_all.sort(key=lambda row: int(row["work_numeric_id"]))
    candidate_ids = np.array(
        [row["work_numeric_id"] for row in prelink_all],
        dtype=np.uint64,
    )
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
        if prelink_by_id[int(work_id)]["publication_year"] <= 1985
    ]
    selected, pool = _select(
        pool_rows,
        seed=str(config["sampling"]["seed"]),
        per_window=int(config["sampling"]["rows_per_window"]),
    )
    pool.update(
        {
            "schema_version": "pivot_feedback_d0_t10543_rank3_candidate_pool_summary.v1",
            "field_id": 27,
            "field_candidate_rank": 3,
            "topic_id": 10543,
            "entry_year": 1986,
            "normalized_work_rows_scanned": sum(
                part["work_rows"] for part in work_parts
            ),
            "normalized_event_rows_scanned": sum(
                part["event_rows"] for part in event_parts
            ),
            "eligible_prelink_rows": sum(
                row["publication_year"] <= 1985 for row in prelink_all
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
    table = _rank3_table(selected, found)
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
            "target_topic_ids_read": [10543],
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
        T10543Rank3TargetedSampleBuildError,
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
                {"status": "BLOCKED_T10543_RANK3_TARGETED_SAMPLE_V1", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
