#!/usr/bin/env python3
"""Independently replay and verify the exact T10543 rank-3 title sample."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq
import yaml

try:
    from scripts import (
        verify_pivot_feedback_d0_t10543_targeted_title_sample_v1 as base,
    )
except ModuleNotFoundError:
    import verify_pivot_feedback_d0_t10543_targeted_title_sample_v1 as base


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
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-t10543-rank3-targeted-title-sample-v1-20260729."
    "verification-v1.json"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t10543_rank3_targeted_title_sample_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t10543_rank3_targeted_title_sample_outputs.v1"
SAMPLE_SCHEMA = "pivot_feedback_d0_t10543_rank3_targeted_title_sample.v1"
VERIFY_SCHEMA = "pivot_feedback_d0_t10543_rank3_targeted_title_sample_verification.v1"
VERIFY_STATUS = "PASS_INDEPENDENT_EXACT_T10543_RANK3_96_TITLE_LOCAL_SAMPLE"


class T10543Rank3TargetedSampleVerificationError(RuntimeError):
    """Fail-closed T10543 rank-3 targeted-sample verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T10543Rank3TargetedSampleVerificationError(message)


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


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


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


def _validate_config(path: Path) -> dict[str, Any]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    helpers = _mapping(config.get("implementation_helpers"), "helpers")
    for key in helpers:
        _repo_file(_mapping(helpers.get(key), key))
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
    return config


def _verify_output(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = _mapping(config.get("output_contract"), "output contract")
    root = output_root.resolve(strict=True)
    require(root.name == contract["root_basename"], "output basename")
    require(root.is_dir() and not root.is_symlink(), "output root")
    require(root.stat().st_mode & 0o777 == 0o700, "output mode")
    paths = {
        "sample": root / str(contract["sample_basename"]),
        "pool_summary": root / str(contract["pool_summary_basename"]),
        "report": root / str(contract["report_basename"]),
    }
    receipt_path = root / str(contract["receipt_basename"])
    require(
        {path.name for path in root.iterdir()}
        == {path.name for path in paths.values()} | {receipt_path.name},
        "artifact set",
    )
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "output schema")
    require(
        receipt.get("status") == "PASS_EXACT_T10543_RANK3_96_TITLE_LOCAL_SAMPLE",
        "output status",
    )
    require(receipt.get("config_sha256") == config_sha256, "config SHA")
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"artifact: {key}")
        require(path.stat().st_mode & 0o777 == 0o600, f"artifact mode: {key}")
        artifact = _mapping(receipt["artifacts"][key], f"artifact: {key}")
        require(path.stat().st_size == artifact["bytes"], f"artifact bytes: {key}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {key}")
    require(receipt_path.stat().st_mode & 0o777 == 0o600, "receipt mode")
    require(
        hashlib.sha256(_canonical_json_bytes(receipt["artifacts"])).hexdigest()
        == receipt["artifact_tree_sha256"],
        "artifact tree",
    )
    return receipt, paths


def _window(year: int) -> int:
    if year <= 1979:
        return 0
    if year <= 1982:
        return 1
    require(year <= 1985, "window year")
    return 2


def _normalized_replay(
    config: Mapping[str, Any],
    work_paths: Sequence[Path],
    event_paths: Sequence[Path],
    *,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target = _mapping(config.get("target"), "target")
    included = np.array(target["included_work_type_codes"], dtype=np.uint8)
    dispositions = np.array(target["allowed_temporal_dispositions"], dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        work_parts = list(
            executor.map(
                lambda pair: base._work_candidates(
                    pair,
                    included_types=included,
                    dispositions=dispositions,
                ),
                enumerate(work_paths),
            )
        )
    candidates_all = [row for rows, _ in work_parts for row in rows]
    candidates_all.sort(key=lambda row: row["work_numeric_id"])
    ids = np.array([row["work_numeric_id"] for row in candidates_all], dtype=np.uint64)
    years = np.array([row["publication_year"] for row in candidates_all], dtype=np.uint16)
    candidate_dispositions = np.array(
        [row["temporal_disposition"] for row in candidates_all],
        dtype=np.uint8,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        event_parts = list(
            executor.map(
                lambda pair: base._linked_work_ids(
                    pair,
                    candidate_ids=ids,
                    candidate_year=years,
                    candidate_disposition=candidate_dispositions,
                ),
                enumerate(event_paths),
            )
        )
    arrays = [values for values, _ in event_parts if len(values)]
    linked_ids, linked_counts = np.unique(np.concatenate(arrays), return_counts=True)
    by_id = {row["work_numeric_id"]: row for row in candidates_all}
    linked = [
        {**by_id[int(work_id)], "matching_author_event_rows": int(count)}
        for work_id, count in zip(
            linked_ids.tolist(),
            linked_counts.tolist(),
            strict=True,
        )
        if by_id[int(work_id)]["publication_year"] <= 1985
    ]
    groups: dict[int, list[dict[str, Any]]] = {0: [], 1: [], 2: []}
    seed = str(config["sampling"]["seed"])
    for row in linked:
        window = _window(row["publication_year"])
        preimage = (
            f"{seed}\0{27}\0{10543}\0{1986}\0{window}\0"
            f"{row['work_numeric_id']}"
        )
        groups[window].append(
            {
                **row,
                "window_code": window,
                "selection_sha256": hashlib.sha256(preimage.encode()).hexdigest(),
            }
        )
    selected: list[dict[str, Any]] = []
    for window in (0, 1, 2):
        ordered = sorted(
            groups[window],
            key=lambda row: (row["selection_sha256"], row["work_numeric_id"]),
        )
        require(len(ordered) >= 32, f"window pool: {window}")
        selected.extend(ordered[:32])
    selected.sort(key=lambda row: (row["window_code"], row["selection_sha256"]))
    return selected, {
        "normalized_work_rows_scanned": sum(rows for _, rows in work_parts),
        "normalized_event_rows_scanned": sum(rows for _, rows in event_parts),
        "eligible_prelink_rows": sum(
            row["publication_year"] <= 1985 for row in candidates_all
        ),
        "author_linked_pool_rows": len(linked),
        "recent_six_year_author_linked_rows": len(groups[1]) + len(groups[2]),
        "window_pool_rows": {str(window): len(groups[window]) for window in (0, 1, 2)},
    }


def validate(
    config_path: Path = DEFAULT_CONFIG,
    work_root: Path = DEFAULT_WORK_ROOT,
    event_root: Path = DEFAULT_EVENT_ROOT,
    openalex_root: Path = DEFAULT_OPENALEX_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    workers: int = 16,
    replay_normalized: bool = True,
) -> dict[str, Any]:
    require(workers >= 1, "workers")
    config_path = config_path.resolve(strict=True)
    config_sha256 = _sha256(config_path)
    config = _validate_config(config_path)
    receipt, paths = _verify_output(
        config,
        output_root,
        config_sha256=config_sha256,
    )
    sample_file = pq.ParquetFile(paths["sample"])
    require(sample_file.metadata.num_rows == 96, "sample rows")
    require(
        sample_file.schema_arrow.metadata == {b"schema_version": SAMPLE_SCHEMA.encode()},
        "sample schema",
    )
    sample_rows = sample_file.read().to_pylist()
    require(
        {(row["field_candidate_rank"], row["entry_year"]) for row in sample_rows}
        == {(3, 1986)},
        "sample target",
    )
    pool = _read_json(paths["pool_summary"])
    require(pool == receipt["pool"], "pool receipt")
    raw = base._verify_raw_rows(config, openalex_root, sample_rows)
    normalized_replay: dict[str, Any] | None = None
    work_tree: str | None = None
    event_tree: str | None = None
    if replay_normalized:
        normalized = _mapping(config.get("normalized_inputs"), "normalized")
        work_paths, work_tree = base._partition_inputs(
            work_root,
            _mapping(normalized.get("cutoff_work_metadata"), "work input"),
            workers=workers,
        )
        event_paths, event_tree = base._partition_inputs(
            event_root,
            _mapping(normalized.get("author_history_events"), "event input"),
            workers=workers,
        )
        expected, normalized_replay = _normalized_replay(
            config,
            work_paths,
            event_paths,
            workers=workers,
        )
        require(
            [row["work_numeric_id"] for row in expected]
            == [row["work_numeric_id"] for row in sample_rows],
            "selected IDs",
        )
        for expected_row, observed in zip(expected, sample_rows, strict=True):
            for key in (
                "publication_year",
                "temporal_disposition",
                "work_type_code",
                "declared_authors_count",
                "matching_author_event_rows",
                "window_code",
                "selection_sha256",
            ):
                require(observed[key] == expected_row[key], f"sample metadata: {key}")
        for key, value in normalized_replay.items():
            require(pool[key] == value, f"pool replay: {key}")
        require(
            work_tree
            == receipt["normalized_input_evidence"]["work_selected_tree_sha256"],
            "work tree",
        )
        require(
            event_tree
            == receipt["normalized_input_evidence"]["event_selected_tree_sha256"],
            "event tree",
        )
    for key in (
        "general_raw_title_scan",
        "full_candidate_title_copy",
        "network_accessed",
        "credentials_accessed",
        "content_disposition_created",
        "selection_certificate_issued",
        "deterministic_reselection_executed",
        "outcomes_read",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt[key] is False, f"unsafe receipt flag: {key}")
    require(receipt["api_requests_created"] == 0, "API requests created")
    require(receipt["api_requests_submitted"] == 0, "API requests submitted")
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(
            output_root / str(config["output_contract"]["receipt_basename"])
        ),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "normalized_replay_executed": replay_normalized,
        "normalized_work_selected_tree_sha256": work_tree,
        "normalized_event_selected_tree_sha256": event_tree,
        "normalized_replay": normalized_replay,
        "raw_selected_row_replay": raw,
        "selected_title_rows": 96,
        "missing_title_rows": pool["missing_title_rows"],
        "recent_six_year_author_linked_rows": pool[
            "recent_six_year_author_linked_rows"
        ],
        "api_requests_submitted": 0,
        "content_disposition_created": False,
        "selection_certificate_issued": False,
        "outcomes_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--event-root", type=Path, default=DEFAULT_EVENT_ROOT)
    parser.add_argument("--openalex-root", type=Path, default=DEFAULT_OPENALEX_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            args.config,
            args.work_root,
            args.event_root,
            args.openalex_root,
            args.output_root,
            workers=args.workers,
        )
        report_path = args.report.absolute()
        require(not report_path.exists(), "verification report exists")
        require(report_path.parent == Path("/ssd"), "verification report parent")
        temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(result))
        temporary.chmod(0o600)
        os.replace(temporary, report_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        T10543Rank3TargetedSampleVerificationError,
        base.T10543TargetedSampleVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_T10543_RANK3_SAMPLE_VERIFICATION", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
