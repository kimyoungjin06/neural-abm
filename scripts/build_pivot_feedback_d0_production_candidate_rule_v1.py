#!/usr/bin/env python3
"""Freeze production candidate dispositions and build adaptive frontier v2."""

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
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_production_candidate_rule_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-production-candidate-rule-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_production_candidate_rule_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_production_candidate_rule_outputs.v1"
DISPOSITION_SCHEMA = "pivot_feedback_d0_production_candidate_dispositions.v1"
FRONTIER_SCHEMA = "pivot_feedback_d0_adaptive_candidate_frontier.v2"
STATUS = "PASS_PRODUCTION_CANDIDATE_RULE_FROZEN_FRONTIER_V2_TARGETED_SAMPLE_REQUIRED"


class ProductionCandidateRuleBuildError(RuntimeError):
    """Fail-closed production-candidate-rule build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionCandidateRuleBuildError(message)


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
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ProductionCandidateRuleBuildError(f"invalid YAML: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionCandidateRuleBuildError(f"invalid JSON: {error}") from error
    return _mapping(value, path.name)


def _repo_path(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _ssd_root(value: Any) -> Path:
    require(
        isinstance(value, str)
        and value
        and "/" not in value
        and value not in {".", ".."},
        "unsafe SSD basename",
    )
    path = (Path("/ssd") / value).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "SSD root escapes")
    require(path.is_dir() and not path.is_symlink(), "invalid SSD root")
    return path


def _dotted(document: Mapping[str, Any], key: str) -> Any:
    value: Any = document
    for part in key.split("."):
        value = _mapping(value, part)[part]
    return value


def _validate_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status")
        == "FROZEN_PRODUCTION_CANDIDATE_RULE_AND_FRONTIER_V2_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = _repo_path(parent)
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent["required_schema_version"],
        "parent schema",
    )
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    rule = _mapping(config.get("production_rule"), "production rule")
    require(rule["accepted_without_change_from_dry_run"] is True, "rule not accepted")
    require(rule["human_review_required"] is False, "human review reintroduced")
    require(rule["api_labels_are_ground_truth"] is False, "API ground truth")
    boundary = _mapping(config.get("execution_boundary"), "execution boundary")
    require(all(value is False for value in boundary.values()), "execution opened")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "production_candidate_rule_freeze",
        "four_production_candidate_dispositions",
        "adaptive_frontier_v2_materialization",
        "computer_science_field_certificate",
        "targeted_t10543_sample_successor_contract",
    ):
        require(authorization.get(key) == "AUTHORIZED", f"safe work closed: {key}")
    for key in (
        "targeted_t10543_local_sample_execution",
        "general_raw_openalex_title_scan",
        "new_api_request_creation",
        "api_submission",
        "global_selection_certificate",
        "deterministic_reselection",
        "author_panel_rebuild",
        "pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"unsafe authorization: {key}",
        )
    return config, {"parent_sha256": _sha256(parent_path)}


def _load_inputs(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    candidate = _mapping(frozen.get("candidate_table"), "candidate table")
    candidate_root = _ssd_root(candidate["root_basename"])
    candidate_path = candidate_root / str(candidate["basename"])
    require(_sha256(candidate_path) == candidate["sha256"], "candidate SHA")
    require(pq.ParquetFile(candidate_path).metadata.num_rows == candidate["rows"], "candidate rows")
    candidates = pq.read_table(candidate_path).to_pylist()

    dry = _mapping(frozen.get("dry_run"), "dry-run")
    dry_config_path = _repo_path(_mapping(dry.get("config"), "dry config"))
    dry_summary_binding = _mapping(dry.get("summary"), "dry summary")
    dry_summary_path = _repo_path(dry_summary_binding)
    dry_summary = _read_json(dry_summary_path)
    require(
        dry_summary.get("schema_version")
        == dry_summary_binding["required_schema_version"],
        "dry summary schema",
    )
    require(
        dry_summary.get("status") == dry_summary_binding["required_status"],
        "dry summary status",
    )
    dry_root = _ssd_root(dry["operator_root_basename"])
    dry_candidate_path = dry_root / str(dry["candidate_basename"])
    require(_sha256(dry_candidate_path) == dry["candidate_sha256"], "dry candidate SHA")
    require(
        pq.ParquetFile(dry_candidate_path).metadata.num_rows
        == dry["candidate_rows"],
        "dry candidate rows",
    )
    dry_rows = pq.read_table(dry_candidate_path).to_pylist()
    receipt_path = dry_root / str(dry["receipt_basename"])
    verification_path = Path("/ssd") / str(dry["verification_report_basename"])
    require(_sha256(receipt_path) == dry["receipt_sha256"], "dry receipt SHA")
    require(
        _sha256(verification_path) == dry["verification_report_sha256"],
        "dry verification SHA",
    )
    return candidates, dry_rows, {
        "candidate_sha256": _sha256(candidate_path),
        "dry_config_sha256": _sha256(dry_config_path),
        "dry_summary_sha256": _sha256(dry_summary_path),
        "dry_candidate_sha256": _sha256(dry_candidate_path),
        "dry_receipt_sha256": _sha256(receipt_path),
        "dry_verification_sha256": _sha256(verification_path),
    }


def _build_dispositions(
    config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    dry_rows: Sequence[Mapping[str, Any]],
) -> pa.Table:
    candidate_by_key = {
        (
            int(row["field_id"]),
            int(row["field_candidate_rank"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        ): row
        for row in candidates
    }
    dry_by_key = {
        (
            int(row["field_id"]),
            int(row["field_candidate_rank"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        ): row
        for row in dry_rows
    }
    expected_dry_state = {
        "CONTENT_PASS": "WOULD_CONTENT_PASS",
        "CONTENT_FAIL": "WOULD_CONTENT_FAIL_INDEXING_ANOMALY",
    }
    output: list[dict[str, Any]] = []
    for frozen in config["production_candidate_dispositions"]:
        key = (
            int(frozen["field_id"]),
            int(frozen["field_candidate_rank"]),
            int(frozen["topic_id"]),
            int(frozen["entry_year"]),
        )
        require(key in candidate_by_key and key in dry_by_key, f"disposition key: {key}")
        candidate = candidate_by_key[key]
        dry = dry_by_key[key]
        state = str(frozen["content_state"])
        require(dry["dry_run_state"] == expected_dry_state[state], "dry state promotion")
        require(
            dry["dry_run_reason"] == frozen["disposition_reason"],
            "disposition reason",
        )
        output.append(
            {
                "field_id": key[0],
                "field_candidate_rank": key[1],
                "topic_id": key[2],
                "topic_name": str(candidate["topic_name"]),
                "subfield_id": int(candidate["subfield_id"]),
                "subfield_name": str(candidate["subfield_name"]),
                "entry_year": key[3],
                "content_state": state,
                "disposition_reason": str(frozen["disposition_reason"]),
                "recent_accept_lower_bound_fraction": float(
                    dry["recent_accept_lower_bound_fraction"]
                ),
                "baseline_to_signal_score_drop": float(
                    dry["baseline_to_signal_score_drop"]
                ),
                "baseline_to_signal_source_share_increase": float(
                    dry["baseline_to_signal_source_share_increase"]
                ),
                "source_dry_run_state": str(dry["dry_run_state"]),
                "human_review_used": False,
                "api_labels_used_as_ground_truth": False,
            }
        )
    output.sort(key=lambda row: (row["field_id"], row["field_candidate_rank"]))
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("content_state", pa.string(), nullable=False),
            pa.field("disposition_reason", pa.string(), nullable=False),
            pa.field("recent_accept_lower_bound_fraction", pa.float64(), nullable=False),
            pa.field("baseline_to_signal_score_drop", pa.float64(), nullable=False),
            pa.field(
                "baseline_to_signal_source_share_increase",
                pa.float64(),
                nullable=False,
            ),
            pa.field("source_dry_run_state", pa.string(), nullable=False),
            pa.field("human_review_used", pa.bool_(), nullable=False),
            pa.field("api_labels_used_as_ground_truth", pa.bool_(), nullable=False),
        ],
        metadata={b"schema_version": DISPOSITION_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(output, schema=schema)


def _frontier(
    candidates: Sequence[Mapping[str, Any]],
    dispositions: pa.Table,
    config: Mapping[str, Any],
) -> tuple[pa.Table, dict[str, Any]]:
    disposition_by_key = {
        (int(row["field_id"]), int(row["field_candidate_rank"])): row
        for row in dispositions.to_pylist()
    }
    rows: list[dict[str, Any]] = []
    by_field: dict[str, dict[str, Any]] = {}
    for field_id, last_rank in ((17, 45), (27, 6)):
        field = sorted(
            (row for row in candidates if int(row["field_id"]) == field_id),
            key=lambda row: int(row["field_candidate_rank"]),
        )[:last_rank]
        for candidate in field:
            rank = int(candidate["field_candidate_rank"])
            frozen = disposition_by_key.get((field_id, rank))
            if frozen is not None:
                state = str(frozen["content_state"])
                role = (
                    "FROZEN_PRODUCTION_CONTENT_PASS"
                    if state == "CONTENT_PASS"
                    else "FROZEN_PRODUCTION_CONTENT_FAIL"
                )
                reason = str(frozen["disposition_reason"])
            elif field_id == 27 and rank == 2:
                state = "CONTENT_UNRESOLVED"
                role = "NEXT_REQUIRED_CANDIDATE_TARGETED_SAMPLE"
                reason = "TARGETED_T10543_SAMPLE_REQUIRED"
            else:
                state = "NOT_EVALUATED"
                role = "STRUCTURALLY_NOT_REQUIRED_ON_CURRENT_BRANCH"
                reason = "INCOMPATIBLE_WITH_CURRENT_FIRST_PASS_BRANCH"
            rows.append(
                {
                    "field_id": field_id,
                    "field_candidate_rank": rank,
                    "topic_id": int(candidate["topic_id"]),
                    "topic_name": str(candidate["topic_name"]),
                    "subfield_id": int(candidate["subfield_id"]),
                    "subfield_name": str(candidate["subfield_name"]),
                    "entry_year": int(candidate["entry_year"]),
                    "frontier_role": role,
                    "content_state": state,
                    "state_reason": reason,
                    "targeted_sample_required": field_id == 27 and rank == 2,
                }
            )
        states = [row for row in rows if row["field_id"] == field_id]
        if field_id == 17:
            field_result = {
                "frontier_rows": 45,
                "content_pass_rows": 2,
                "content_fail_rows": 0,
                "content_unresolved_rows": 0,
                "not_evaluated_rows": 43,
                "field_selection_certificate_issued": True,
                "selected_candidate_ranks": [1, 45],
                "selected_topic_ids": [14430, 10579],
            }
        else:
            field_result = {
                "frontier_rows": 6,
                "content_pass_rows": 1,
                "content_fail_rows": 1,
                "content_unresolved_rows": 1,
                "not_evaluated_rows": 3,
                "field_selection_certificate_issued": False,
                "required_next_candidate_rank": 2,
                "required_next_topic_id": 10543,
                "required_next_entry_year": 1993,
                "later_compatible_pass_rank_under_rank_2_pass_branch": 6,
            }
        observed_counts = {
            "frontier_rows": len(states),
            "content_pass_rows": sum(row["content_state"] == "CONTENT_PASS" for row in states),
            "content_fail_rows": sum(row["content_state"] == "CONTENT_FAIL" for row in states),
            "content_unresolved_rows": sum(
                row["content_state"] == "CONTENT_UNRESOLVED" for row in states
            ),
            "not_evaluated_rows": sum(
                row["content_state"] == "NOT_EVALUATED" for row in states
            ),
        }
        require(
            observed_counts
            == {key: field_result[key] for key in observed_counts},
            f"field closure: {field_id}",
        )
        require(field_result == config["frontier_v2"]["fields"][field_id], "field config")
        by_field[str(field_id)] = field_result
    closure = {
        "fields": by_field,
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
    expected = {
        key: config["frontier_v2"][key]
        for key in closure
        if key != "fields"
    }
    require(
        {key: closure[key] for key in expected} == expected,
        "frontier closure",
    )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("frontier_role", pa.string(), nullable=False),
            pa.field("content_state", pa.string(), nullable=False),
            pa.field("state_reason", pa.string(), nullable=False),
            pa.field("targeted_sample_required", pa.bool_(), nullable=False),
        ],
        metadata={b"schema_version": FRONTIER_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(rows, schema=schema), closure


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    path.chmod(0o600)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def _markdown(dispositions: pa.Table, closure: Mapping[str, Any]) -> str:
    lines = [
        "# Production candidate rule and adaptive frontier v2",
        "",
        f"Status: `{STATUS}`",
        "",
        "| Candidate | Production state | Reason |",
        "|---|---|---|",
    ]
    for row in dispositions.to_pylist():
        lines.append(
            f"| F{row['field_id']}/r{row['field_candidate_rank']}/"
            f"T{row['topic_id']}/{row['entry_year']} | "
            f"`{row['content_state']}` | `{row['disposition_reason']}` |"
        )
    lines.extend(
        [
            "",
            "Computer Science ranks 1 and 45 have a field-level certificate.",
            "Medicine rank 1 is CONTENT_FAIL; rank 2 T10543 is",
            "CONTENT_UNRESOLVED and requires the exact targeted 96-title sample.",
            "The global selection certificate remains blocked.",
            "",
            f"Closure: {json.dumps(dict(closure), sort_keys=True)}",
            "",
        ]
    )
    return "\n".join(lines)


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(config_path.is_relative_to(REPO_ROOT), "config escapes repository")
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config, bindings = _validate_config(config_path)
    candidates, dry_rows, inputs = _load_inputs(config)
    dispositions = _build_dispositions(config, candidates, dry_rows)
    frontier, closure = _frontier(candidates, dispositions, config)
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")

    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["dispositions"] = _write_parquet(
            dispositions,
            temporary / str(contract["dispositions_basename"]),
        )
        artifacts["frontier"] = _write_parquet(
            frontier,
            temporary / str(contract["frontier_basename"]),
        )
        report_path = temporary / str(contract["report_basename"])
        _write_private(report_path, _markdown(dispositions, closure).encode("utf-8"))
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
            "producer_relative_path": Path(__file__).resolve().relative_to(REPO_ROOT).as_posix(),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "repository_bindings": bindings,
            "input_evidence": inputs,
            "production_candidate_dispositions": len(dispositions),
            "frontier": closure,
            "targeted_successor": config["targeted_successor"],
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "new_titles_read": False,
            "raw_openalex_main_table_read": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "global_selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
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
        result = build(args.config, args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        ProductionCandidateRuleBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_PRODUCTION_CANDIDATE_RULE_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
