#!/usr/bin/env python3
"""Freeze rank-5 unresolved disposition and terminal certificate state."""

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
    / "experiments/configs/pivot_feedback_d0_rank5_terminal_disposition_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-rank5-terminal-disposition-v1-20260729"
)
OUTPUT_SCHEMA = "pivot_feedback_d0_rank5_terminal_disposition_outputs.v1"
STATUS = "PASS_RANK5_CONTENT_UNRESOLVED_TERMINAL_CERTIFICATE_BLOCK_FROZEN"


class Rank5TerminalDispositionBuildError(RuntimeError):
    """Fail-closed rank-5 terminal-disposition build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5TerminalDispositionBuildError(message)


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
        == "pivot_feedback_d0_rank5_terminal_disposition_config.v1",
        "config schema",
    )
    require(
        config.get("status")
        == "FROZEN_RANK5_CONTENT_UNRESOLVED_TERMINAL_CERTIFICATE_BLOCK_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    selection = _mapping(config.get("selection_contract"), "selection")
    selection_document = _read_yaml(_repo_file(selection))
    require(
        selection_document.get("schema_version") == selection["required_schema_version"],
        "selection schema",
    )
    contract = _mapping(selection_document.get("selection_contract"), "selection contract")
    require(contract.get("content_unresolved_is_pass") is False, "unresolved pass")
    require(
        _mapping(contract.get("lazy_certificate"), "lazy").get(
            "unresolved_required_row_blocks_certificate"
        )
        is True,
        "unresolved block",
    )
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "new_candidate_evaluation",
        "topic_card_amendment",
        "new_title_extraction",
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "selection_certificate",
        "deterministic_reselection",
        "author_opportunity_panel_rebuild",
        "production_pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def _inputs(
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("rank4_production"), "rank4 production")
    dispositions_path = _ssd_file(
        prior,
        "dispositions_basename",
        "dispositions_sha256",
    )
    frontier_path = _ssd_file(prior, "frontier_basename", "frontier_sha256")
    receipt_path = _ssd_file(prior, "receipt_basename", "receipt_sha256")
    dispositions = pq.read_table(dispositions_path)
    frontier = pq.read_table(frontier_path)
    require(len(dispositions) == prior["disposition_rows"], "prior disposition rows")
    require(len(frontier) == prior["frontier_rows"], "prior frontier rows")
    rank5 = _mapping(frozen.get("rank5_dry_run"), "rank5")
    candidate_path = _ssd_file(rank5, "candidate_basename", "candidate_sha256")
    receipt5 = _ssd_file(rank5, "receipt_basename", "receipt_sha256")
    verification = Path("/ssd") / str(rank5["verification_basename"])
    require(_sha256(verification) == rank5["verification_sha256"], "verification SHA")
    candidate_rows = pq.read_table(candidate_path).to_pylist()
    require(len(candidate_rows) == rank5["candidate_rows"] == 1, "candidate rows")
    return dispositions, frontier, candidate_rows[0], {
        "rank4_dispositions_sha256": _sha256(dispositions_path),
        "rank4_frontier_v5_sha256": _sha256(frontier_path),
        "rank4_receipt_sha256": _sha256(receipt_path),
        "rank5_candidate_sha256": _sha256(candidate_path),
        "rank5_receipt_sha256": _sha256(receipt5),
        "rank5_verification_sha256": _sha256(verification),
    }


def _dispositions(
    current: pa.Table,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pa.Table:
    frozen = _mapping(config.get("rank5_production_disposition"), "rank5 disposition")
    require(candidate["candidate_rule_v2_state"] == frozen["source_rule_state"], "source state")
    require(
        candidate["candidate_rule_v2_reason"] == frozen["disposition_reason"],
        "source reason",
    )
    row = {
        "field_id": frozen["field_id"],
        "field_candidate_rank": frozen["field_candidate_rank"],
        "topic_id": frozen["topic_id"],
        "entry_year": frozen["entry_year"],
        "content_state": frozen["content_state"],
        "disposition_reason": frozen["disposition_reason"],
        "production_rule_version": frozen["production_rule_version"],
        "source_replay_state": frozen["source_rule_state"],
        "post_diagnostic_amendment_used": frozen["post_diagnostic_amendment_used"],
        "human_review_used": frozen["human_review_used"],
        "api_labels_used_as_ground_truth": frozen["api_labels_used_as_ground_truth"],
    }
    rows = current.to_pylist() + [row]
    rows.sort(key=lambda item: (item["field_id"], item["field_candidate_rank"]))
    require(len(rows) == 8, "cumulative disposition rows")
    return pa.Table.from_pylist(
        rows,
        schema=current.schema.with_metadata(
            {b"schema_version": b"pivot_feedback_d0_production_candidate_dispositions.v5"}
        ),
    )


def _frontier(
    current: pa.Table,
    config: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    rows = current.to_pylist()
    medicine = {row["field_candidate_rank"]: row for row in rows if row["field_id"] == 27}
    medicine[5].update(
        {
            "frontier_role": "FROZEN_PRODUCTION_CONTENT_UNRESOLVED_BLOCKER",
            "content_state": "CONTENT_UNRESOLVED",
            "state_reason": "CONSERVATIVE_RECENT_EVIDENCE_NOT_DECISIVE",
            "targeted_sample_required": False,
        }
    )
    medicine[6].update(
        {
            "frontier_role": "FROZEN_PRODUCTION_PASS_BLOCKED_BY_EARLIER_UNRESOLVED",
            "targeted_sample_required": False,
        }
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
    expected = _mapping(config.get("frontier_v6"), "frontier")
    for key, value in counts.items():
        require(value == expected[key], f"frontier: {key}")
    table = pa.Table.from_pylist(
        rows,
        schema=current.schema.with_metadata(
            {b"schema_version": b"pivot_feedback_d0_adaptive_candidate_frontier.v6"}
        ),
    )
    return table, counts


def _certificate_status(config: Mapping[str, Any]) -> dict[str, Any]:
    status = dict(_mapping(config.get("terminal_certificate_status"), "certificate"))
    require(status["status"] == "BLOCKED_REQUIRED_RANK5_CONTENT_UNRESOLVED", "certificate")
    require(status["medicine_field_certificate_issued"] is False, "Medicine certificate")
    require(status["global_selection_certificate_issued"] is False, "global certificate")
    require(status["deterministic_reselection_executed"] is False, "reselection")
    return {
        "schema_version": "pivot_feedback_d0_terminal_selection_certificate_status.v1",
        **status,
    }


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
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config = _load_config(config_path)
    current_dispositions, current_frontier, candidate, evidence = _inputs(config)
    dispositions = _dispositions(current_dispositions, candidate, config)
    frontier, closure = _frontier(current_frontier, config)
    certificate = _certificate_status(config)
    contract = _mapping(config.get("output_contract"), "output")
    require(output_root.name == contract["root_basename"], "output basename")
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
        certificate_path = temporary / str(contract["certificate_status_basename"])
        _write_private(certificate_path, _canonical_json_bytes(certificate))
        artifacts["certificate_status"] = {
            "basename": certificate_path.name,
            "bytes": certificate_path.stat().st_size,
            "sha256": _sha256(certificate_path),
        }
        report = temporary / str(contract["report_basename"])
        _write_private(
            report,
            (
                "# Rank-5 terminal disposition and certificate status\n\n"
                f"Status: `{STATUS}`\n\n"
                "Medicine rank 5 is CONTENT_UNRESOLVED. The earlier unresolved "
                "required row blocks the Medicine and global selection "
                "certificates; no deterministic reselection was executed.\n"
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
            "terminal_certificate_status": certificate,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "new_candidate_evaluation_executed": False,
            "new_titles_read": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "computer_science_field_certificate_issued": True,
            "medicine_field_certificate_issued": False,
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
        Rank5TerminalDispositionBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_RANK5_TERMINAL_DISPOSITION", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
