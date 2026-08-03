#!/usr/bin/env python3
"""Freeze production candidate rule v2 and adaptive frontier v3."""

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
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_production_candidate_rule_v2.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-production-candidate-rule-v2-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_production_candidate_rule_config.v2"
OUTPUT_SCHEMA = "pivot_feedback_d0_production_candidate_rule_outputs.v2"
DISPOSITION_SCHEMA = "pivot_feedback_d0_production_candidate_dispositions.v2"
FRONTIER_SCHEMA = "pivot_feedback_d0_adaptive_candidate_frontier.v3"
STATUS = (
    "PASS_PRODUCTION_CANDIDATE_RULE_V2_FROZEN_FRONTIER_V3_TARGETED_RANK3_REQUIRED"
)


class ProductionCandidateRuleV2BuildError(RuntimeError):
    """Fail-closed production-candidate-rule-v2 build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionCandidateRuleV2BuildError(message)


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


def _ssd_file(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root_name = str(binding["root_basename"])
    require(root_name and "/" not in root_name, "unsafe SSD root")
    root = (Path("/ssd") / root_name).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "SSD root escapes")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), "frozen input")
    require(_sha256(path) == binding[sha_key], f"input SHA: {path.name}")
    return path


def _load_config(path: Path) -> dict[str, Any]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status")
        == "FROZEN_PRODUCTION_CANDIDATE_RULE_V2_AND_FRONTIER_V3_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    rule = _mapping(config.get("production_rule_v2"), "production rule")
    require(
        rule.get("amendment_disclosure")
        == "POST_DIAGNOSTIC_AFTER_T10543_RULE_V1_GAP",
        "amendment disclosure",
    )
    require(rule.get("amendment_is_out_of_sample_validated") is False, "validation claim")
    branch = _mapping(rule.get("new_branch"), "new branch")
    require(branch.get("minimum_recent_opaque_quarantine_fraction") == 0.5, "threshold")
    for key in (
        "source_name_used",
        "primary_topic_score_used",
        "api_label_used",
        "human_review_used",
    ):
        require(branch[key] is False, f"unsafe branch input: {key}")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "production_candidate_rule_v2_freeze",
        "five_production_candidate_dispositions",
        "adaptive_frontier_v3_materialization",
        "computer_science_field_certificate_preservation",
        "targeted_rank3_sample_successor_contract",
    ):
        require(authorization.get(key) == "AUTHORIZED", f"authorized work: {key}")
    for key in (
        "medicine_field_certificate",
        "targeted_rank3_sample_execution",
        "general_raw_openalex_title_scan",
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


def _load_inputs(
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, str],
]:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    candidate_binding = _mapping(frozen.get("candidate_table"), "candidates")
    candidate_path = _ssd_file(candidate_binding, "basename", "sha256")
    candidates = pq.read_table(candidate_path).to_pylist()
    require(len(candidates) == candidate_binding["rows"], "candidate rows")
    v1_binding = _mapping(frozen.get("production_rule_v1"), "production v1")
    v1_path = _ssd_file(v1_binding, "dispositions_basename", "dispositions_sha256")
    v1_rows = pq.read_table(v1_path).to_pylist()
    require(len(v1_rows) == v1_binding["disposition_rows"], "v1 disposition rows")
    v1_receipt = _ssd_file(v1_binding, "receipt_basename", "receipt_sha256")
    replay_binding = _mapping(
        frozen.get("candidate_rule_v2_replay"),
        "rule v2 replay",
    )
    replay_path = _ssd_file(replay_binding, "replay_basename", "replay_sha256")
    replay_rows = pq.read_table(replay_path).to_pylist()
    require(len(replay_rows) == replay_binding["replay_rows"], "replay rows")
    replay_receipt = _ssd_file(replay_binding, "receipt_basename", "receipt_sha256")
    verification = Path("/ssd") / str(replay_binding["verification_basename"])
    require(_sha256(verification) == replay_binding["verification_sha256"], "verification SHA")
    return candidates, v1_rows, replay_rows, {
        "candidate_sha256": _sha256(candidate_path),
        "v1_dispositions_sha256": _sha256(v1_path),
        "v1_receipt_sha256": _sha256(v1_receipt),
        "v2_replay_sha256": _sha256(replay_path),
        "v2_replay_receipt_sha256": _sha256(replay_receipt),
        "v2_replay_verification_sha256": _sha256(verification),
    }


def _dispositions(
    config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    v1_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
) -> pa.Table:
    candidate_keys = {
        (
            int(row["field_id"]),
            int(row["field_candidate_rank"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        )
        for row in candidates
    }
    v1_by_key = {
        (
            int(row["field_id"]),
            int(row["field_candidate_rank"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        ): row
        for row in v1_rows
    }
    replay_by_key = {
        (
            int(row["field_id"]),
            int(row["field_candidate_rank"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        ): row
        for row in replay_rows
    }
    output: list[dict[str, Any]] = []
    for frozen in config["production_candidate_dispositions"]:
        key = (
            int(frozen["field_id"]),
            int(frozen["field_candidate_rank"]),
            int(frozen["topic_id"]),
            int(frozen["entry_year"]),
        )
        require(key in candidate_keys and key in replay_by_key, "disposition key")
        replay = replay_by_key[key]
        expected_state = {
            "WOULD_CONTENT_PASS": "CONTENT_PASS",
            "WOULD_CONTENT_FAIL_INDEXING_ANOMALY": "CONTENT_FAIL",
            "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC": "CONTENT_FAIL",
        }.get(str(replay["candidate_rule_v2_state"]))
        require(expected_state == frozen["content_state"], "disposition state")
        require(
            replay["candidate_rule_v2_reason"] == frozen["disposition_reason"],
            "disposition reason",
        )
        if key in v1_by_key:
            v1 = v1_by_key[key]
            require(v1["content_state"] == frozen["content_state"], "v1 state changed")
            require(
                v1["disposition_reason"] == frozen["disposition_reason"],
                "v1 reason changed",
            )
        output.append(
            {
                **{name: value for name, value in zip(
                    ("field_id", "field_candidate_rank", "topic_id", "entry_year"),
                    key,
                    strict=True,
                )},
                "content_state": str(frozen["content_state"]),
                "disposition_reason": str(frozen["disposition_reason"]),
                "production_rule_version": "candidate_rule_v2",
                "source_replay_state": str(replay["candidate_rule_v2_state"]),
                "post_diagnostic_amendment_used": (
                    replay["candidate_rule_v2_branch"]
                    == "RECENT_MAJORITY_OPAQUE_INDEXING_ANOMALY"
                ),
                "human_review_used": False,
                "api_labels_used_as_ground_truth": False,
            }
        )
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("content_state", pa.string(), nullable=False),
            pa.field("disposition_reason", pa.string(), nullable=False),
            pa.field("production_rule_version", pa.string(), nullable=False),
            pa.field("source_replay_state", pa.string(), nullable=False),
            pa.field("post_diagnostic_amendment_used", pa.bool_(), nullable=False),
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
    disposition_by_rank = {
        (int(row["field_id"]), int(row["field_candidate_rank"])): row
        for row in dispositions.to_pylist()
    }
    rows: list[dict[str, Any]] = []
    for field_id, last_rank in ((17, 45), (27, 6)):
        field = sorted(
            (row for row in candidates if int(row["field_id"]) == field_id),
            key=lambda row: int(row["field_candidate_rank"]),
        )[:last_rank]
        for candidate in field:
            rank = int(candidate["field_candidate_rank"])
            frozen = disposition_by_rank.get((field_id, rank))
            if frozen is not None:
                state = str(frozen["content_state"])
                reason = str(frozen["disposition_reason"])
                if field_id == 27 and rank == 6:
                    role = "FROZEN_PRODUCTION_PASS_DEFERRED_BEHIND_REQUIRED_RANK3"
                else:
                    role = f"FROZEN_PRODUCTION_{state}"
            elif field_id == 27 and rank == 3:
                state = "CONTENT_UNRESOLVED"
                reason = "TARGETED_T10543_RANK3_SAMPLE_REQUIRED"
                role = "NEXT_REQUIRED_FIRST_CANDIDATE_TARGETED_SAMPLE"
            else:
                state = "NOT_EVALUATED"
                reason = "NOT_REQUIRED_BEFORE_CURRENT_UNRESOLVED_FRONTIER_ROW"
                role = "STRUCTURALLY_DEFERRED_ON_CURRENT_BRANCH"
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
                    "targeted_sample_required": field_id == 27 and rank == 3,
                }
            )
    by_field: dict[str, dict[str, Any]] = {}
    for field_id in (17, 27):
        field_rows = [row for row in rows if row["field_id"] == field_id]
        configured = dict(config["frontier_v3"]["fields"][field_id])
        counts = {
            "frontier_rows": len(field_rows),
            "content_pass_rows": sum(
                row["content_state"] == "CONTENT_PASS" for row in field_rows
            ),
            "content_fail_rows": sum(
                row["content_state"] == "CONTENT_FAIL" for row in field_rows
            ),
            "content_unresolved_rows": sum(
                row["content_state"] == "CONTENT_UNRESOLVED" for row in field_rows
            ),
            "not_evaluated_rows": sum(
                row["content_state"] == "NOT_EVALUATED" for row in field_rows
            ),
        }
        require(
            counts == {key: configured[key] for key in counts},
            f"field closure: {field_id}",
        )
        by_field[str(field_id)] = configured
    closure = {
        "fields": by_field,
        "total_frontier_rows": len(rows),
        "total_content_pass_rows": sum(
            row["content_state"] == "CONTENT_PASS" for row in rows
        ),
        "total_content_fail_rows": sum(
            row["content_state"] == "CONTENT_FAIL" for row in rows
        ),
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
    for key, value in closure.items():
        if key != "fields":
            require(value == config["frontier_v3"][key], f"frontier closure: {key}")
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


def _write_parquet(path: Path, table: pa.Table) -> dict[str, Any]:
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
        "# Production candidate rule v2 and adaptive frontier v3",
        "",
        f"Status: `{STATUS}`",
        "",
        "Rule v2 is disclosed as a post-diagnostic amendment.",
        "",
        "| Candidate | State | Reason |",
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
            "T10543 rank 2 is CONTENT_FAIL. Medicine rank 3 T10543 entry 1986",
            "is the next required candidate. The Computer Science field",
            "certificate is preserved; Medicine and the global certificate",
            "remain blocked.",
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
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config = _load_config(config_path)
    candidates, v1_rows, replay_rows, input_evidence = _load_inputs(config)
    dispositions = _dispositions(config, candidates, v1_rows, replay_rows)
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
            temporary / str(contract["dispositions_basename"]),
            dispositions,
        )
        artifacts["frontier"] = _write_parquet(
            temporary / str(contract["frontier_basename"]),
            frontier,
        )
        report_path = temporary / str(contract["report_basename"])
        _write_private(report_path, _markdown(dispositions, closure).encode())
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
            "input_evidence": input_evidence,
            "production_candidate_dispositions": len(dispositions),
            "production_rule_v2_post_diagnostic": True,
            "production_rule_v2_out_of_sample_validated": False,
            "frontier": closure,
            "targeted_successor": config["targeted_successor"],
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "existing_rule_replay_rows_read": 5,
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
        print(json.dumps(build(args.config, args.output_root), indent=2, sort_keys=True))
        return 0
    except (
        ProductionCandidateRuleV2BuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_PRODUCTION_CANDIDATE_RULE_V2", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
