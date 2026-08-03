#!/usr/bin/env python3
"""Freeze Medicine rank-4 CONTENT_FAIL and adaptive frontier v5."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_rank4_production_disposition_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-rank4-production-disposition-v1-20260729"
)
OUTPUT_SCHEMA = "pivot_feedback_d0_rank4_production_disposition_outputs.v1"
STATUS = "PASS_RANK4_CONTENT_FAIL_FROZEN_FRONTIER_V5_RANK5_REQUIRED"


class Rank4ProductionDispositionBuildError(RuntimeError):
    """Fail-closed rank-4 production-disposition build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank4ProductionDispositionBuildError(message)


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


def _ssd_file(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "SSD root")
    path = root / str(binding[basename_key])
    require(_sha256(path) == binding[sha_key], f"input SHA: {path.name}")
    return path


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_yaml(path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_rank4_production_disposition_config.v1",
        "config schema",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "rank5_topic_card_design",
        "rank5_targeted_sample_execution",
        "new_title_extraction",
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "global_selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"unsafe authorization: {key}",
        )
    return config


def _inputs(
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("rank3_production"), "rank3 production")
    dispositions_path = _ssd_file(
        prior,
        "dispositions_basename",
        "dispositions_sha256",
    )
    frontier_path = _ssd_file(prior, "frontier_basename", "frontier_sha256")
    receipt_path = _ssd_file(prior, "receipt_basename", "receipt_sha256")
    dispositions = pq.read_table(dispositions_path)
    frontier = pq.read_table(frontier_path)
    rank4 = _mapping(frozen.get("rank4_dry_run"), "rank4")
    candidate_path = _ssd_file(rank4, "candidate_basename", "candidate_sha256")
    rank4_receipt = _ssd_file(rank4, "receipt_basename", "receipt_sha256")
    verification = Path("/ssd") / str(rank4["verification_basename"])
    require(_sha256(verification) == rank4["verification_sha256"], "verification SHA")
    candidate = pq.read_table(candidate_path).to_pylist()[0]
    return dispositions, frontier, candidate, {
        "rank3_dispositions_sha256": _sha256(dispositions_path),
        "rank3_frontier_v4_sha256": _sha256(frontier_path),
        "rank3_receipt_sha256": _sha256(receipt_path),
        "rank4_candidate_sha256": _sha256(candidate_path),
        "rank4_receipt_sha256": _sha256(rank4_receipt),
        "rank4_verification_sha256": _sha256(verification),
    }


def _dispositions(
    current: pa.Table,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pa.Table:
    frozen = _mapping(config.get("rank4_production_disposition"), "rank4 disposition")
    require(candidate["candidate_rule_v2_state"] == frozen["source_rule_state"], "source state")
    row = {
        "field_id": frozen["field_id"],
        "field_candidate_rank": frozen["field_candidate_rank"],
        "topic_id": frozen["topic_id"],
        "entry_year": frozen["entry_year"],
        "content_state": frozen["content_state"],
        "disposition_reason": frozen["disposition_reason"],
        "production_rule_version": frozen["production_rule_version"],
        "source_replay_state": frozen["source_rule_state"],
        "post_diagnostic_amendment_used": True,
        "human_review_used": frozen["human_review_used"],
        "api_labels_used_as_ground_truth": frozen["api_labels_used_as_ground_truth"],
    }
    rows = current.to_pylist() + [row]
    rows.sort(key=lambda item: (item["field_id"], item["field_candidate_rank"]))
    return pa.Table.from_pylist(
        rows,
        schema=current.schema.with_metadata(
            {b"schema_version": b"pivot_feedback_d0_production_candidate_dispositions.v4"}
        ),
    )


def _frontier(current: pa.Table, config: Mapping[str, Any]) -> tuple[pa.Table, dict[str, Any]]:
    rows = current.to_pylist()
    medicine = {row["field_candidate_rank"]: row for row in rows if row["field_id"] == 27}
    medicine[4].update(
        {
            "frontier_role": "FROZEN_PRODUCTION_CONTENT_FAIL",
            "content_state": "CONTENT_FAIL",
            "state_reason": "RECENT_SAMPLE_MAJORITY_OPAQUE_INDEXING_ARTIFACTS",
            "targeted_sample_required": False,
        }
    )
    medicine[5].update(
        {
            "frontier_role": "NEXT_REQUIRED_FIRST_CANDIDATE_TARGETED_SAMPLE",
            "content_state": "CONTENT_UNRESOLVED",
            "state_reason": "TARGETED_T10151_RANK5_SAMPLE_REQUIRED",
            "targeted_sample_required": True,
        }
    )
    medicine[6]["frontier_role"] = (
        "FROZEN_PRODUCTION_PASS_DEFERRED_BEHIND_REQUIRED_RANK5"
    )
    rows.sort(key=lambda row: (row["field_id"], row["field_candidate_rank"]))
    counts = {
        "total_frontier_rows": len(rows),
        "total_content_pass_rows": sum(row["content_state"] == "CONTENT_PASS" for row in rows),
        "total_content_fail_rows": sum(row["content_state"] == "CONTENT_FAIL" for row in rows),
        "total_content_unresolved_rows": sum(
            row["content_state"] == "CONTENT_UNRESOLVED" for row in rows
        ),
        "total_not_evaluated_rows": sum(
            row["content_state"] == "NOT_EVALUATED" for row in rows
        ),
        "field_certificates_issued": 1,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
    }
    for key, value in counts.items():
        require(value == config["frontier_v5"][key], f"frontier: {key}")
    table = pa.Table.from_pylist(
        rows,
        schema=current.schema.with_metadata(
            {b"schema_version": b"pivot_feedback_d0_adaptive_candidate_frontier.v5"}
        ),
    )
    return table, counts


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _write_parquet(path: Path, table: pa.Table) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    path.chmod(0o600)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(not output_root.exists(), "output root exists")
    config = _load_config(config_path)
    current_dispositions, current_frontier, candidate, evidence = _inputs(config)
    dispositions = _dispositions(current_dispositions, candidate, config)
    frontier, closure = _frontier(current_frontier, config)
    contract = _mapping(config.get("output_contract"), "output")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts = {
            "dispositions": _write_parquet(
                temporary / str(contract["dispositions_basename"]),
                dispositions,
            ),
            "frontier": _write_parquet(
                temporary / str(contract["frontier_basename"]),
                frontier,
            ),
        }
        report = temporary / str(contract["report_basename"])
        _write_private(
            report,
            (
                "# Rank-4 disposition and frontier v5\n\n"
                f"Status: `{STATUS}`\n\n"
                "Medicine rank 4 is CONTENT_FAIL. Rank 5 T10151 entry 1993 is "
                "the next required candidate. No selection certificate was issued.\n"
            ).encode(),
        )
        artifacts["report"] = {
            "basename": report.name,
            "bytes": report.stat().st_size,
            "sha256": _sha256(report),
        }
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "input_evidence": evidence,
            "cumulative_production_dispositions": len(dispositions),
            "frontier": closure,
            "targeted_successor": config["targeted_successor"],
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "new_titles_read": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "global_selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "outcomes_read": False,
            "g4_open": False,
        }
        receipt_path = temporary / str(contract["receipt_basename"])
        _write_private(receipt_path, _canonical_json_bytes(receipt))
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(build(args.config, args.output_root), indent=2, sort_keys=True))
        return 0
    except (
        Rank4ProductionDispositionBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_RANK4_PRODUCTION_DISPOSITION", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
