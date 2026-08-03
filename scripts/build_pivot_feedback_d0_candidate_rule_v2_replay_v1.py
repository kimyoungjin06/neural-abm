#!/usr/bin/env python3
"""Replay candidate rule v2 across the five frozen title samples."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
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
    / "experiments/configs/"
    "pivot_feedback_d0_candidate_rule_v2_retrospective_replay_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-candidate-rule-v2-replay-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_candidate_rule_v2_replay_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_candidate_rule_v2_replay_outputs.v1"
REPLAY_SCHEMA = "pivot_feedback_d0_candidate_rule_v2_replay.v1"
STATUS = "PASS_CANDIDATE_RULE_V2_FIVE_CANDIDATE_RETROSPECTIVE_REPLAY"


class CandidateRuleV2ReplayBuildError(RuntimeError):
    """Fail-closed candidate-rule-v2 replay error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateRuleV2ReplayBuildError(message)


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
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, path.name)


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
        == "FROZEN_CANDIDATE_RULE_V2_RETROSPECTIVE_REPLAY_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    rule = _mapping(config.get("candidate_rule_v2"), "candidate rule")
    branch = _mapping(rule.get("new_branch"), "new branch")
    require(branch["minimum_opaque_quarantine_fraction"] == 0.5, "threshold")
    for key in (
        "source_name_used",
        "primary_topic_score_used",
        "api_label_used",
        "human_review_used",
    ):
        require(branch[key] is False, f"unsafe new branch: {key}")
    require(rule["replay_is_out_of_sample_validation"] is False, "out-of-sample claim")
    require(rule["replay_is_production_freeze"] is False, "production freeze")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "new_title_extraction",
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "production_candidate_rule_freeze",
        "production_content_disposition",
        "selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def _load_inputs(
    config: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, str]],
    list[dict[str, Any]],
    dict[str, str],
]:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    original_sample_binding = _mapping(frozen.get("original_sample"), "original sample")
    original_sample_path = _ssd_file(
        original_sample_binding,
        "sample_basename",
        "sample_sha256",
    )
    original_samples = pq.read_table(original_sample_path).to_pylist()
    require(len(original_samples) == original_sample_binding["sample_rows"], "original sample rows")
    original_routes_binding = _mapping(frozen.get("original_routes"), "original routes")
    original_routes_path = _ssd_file(
        original_routes_binding,
        "routes_basename",
        "routes_sha256",
    )
    with original_routes_path.open(newline="", encoding="utf-8") as handle:
        original_routes = list(csv.DictReader(handle))
    require(len(original_routes) == original_routes_binding["route_rows"], "original route rows")
    original_candidate_binding = _mapping(
        frozen.get("original_candidate_v1"),
        "original candidates",
    )
    original_candidate_path = _ssd_file(
        original_candidate_binding,
        "candidate_basename",
        "candidate_sha256",
    )
    original_candidates = pq.read_table(original_candidate_path).to_pylist()
    require(
        len(original_candidates) == original_candidate_binding["candidate_rows"],
        "original candidate rows",
    )
    target_sample_binding = _mapping(frozen.get("t10543_sample"), "target sample")
    target_sample_path = _ssd_file(
        target_sample_binding,
        "sample_basename",
        "sample_sha256",
    )
    target_samples = pq.read_table(target_sample_path).to_pylist()
    require(len(target_samples) == target_sample_binding["sample_rows"], "target sample rows")
    target_routes_binding = _mapping(frozen.get("t10543_routes"), "target routes")
    target_routes_path = _ssd_file(
        target_routes_binding,
        "routes_basename",
        "routes_sha256",
    )
    with target_routes_path.open(newline="", encoding="utf-8") as handle:
        target_routes = list(csv.DictReader(handle))
    require(len(target_routes) == target_routes_binding["route_rows"], "target route rows")
    target_candidate_binding = _mapping(
        frozen.get("t10543_candidate_v1"),
        "target candidate",
    )
    target_candidate_path = _ssd_file(
        target_candidate_binding,
        "candidate_basename",
        "candidate_sha256",
    )
    target_candidates = pq.read_table(target_candidate_path).to_pylist()
    require(
        len(target_candidates) == target_candidate_binding["candidate_rows"] == 1,
        "target candidate rows",
    )
    route_by_id = {
        row["openalex_id"]: row
        for row in original_routes + target_routes
    }
    require(len(route_by_id) == 480, "route IDs")
    samples = original_samples + target_samples
    require({str(row["openalex_id"]) for row in samples} == set(route_by_id), "sample routes")
    candidates = original_candidates + target_candidates
    evidence = {
        "original_sample_sha256": _sha256(original_sample_path),
        "original_routes_sha256": _sha256(original_routes_path),
        "original_candidate_v1_sha256": _sha256(original_candidate_path),
        "t10543_sample_sha256": _sha256(target_sample_path),
        "t10543_routes_sha256": _sha256(target_routes_path),
        "t10543_candidate_v1_sha256": _sha256(target_candidate_path),
    }
    return samples, list(route_by_id.values()), candidates, evidence


def _replay(
    samples: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, str]],
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    route_by_id = {row["openalex_id"]: row for row in routes}
    sample_by_topic: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        topic_id = int(sample["topic_id"])
        sample_by_topic.setdefault(topic_id, []).append(
            {**sample, **route_by_id[str(sample["openalex_id"])]}
        )
    rank_by_key = {
        (17, 14430, 2003): 1,
        (17, 10579, 2012): 45,
        (27, 12505, 1991): 1,
        (27, 10543, 1993): 2,
        (27, 12421, 1953): 6,
    }
    threshold = float(
        config["candidate_rule_v2"]["new_branch"][
            "minimum_opaque_quarantine_fraction"
        ]
    )
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        topic_id = int(candidate["topic_id"])
        topic_samples = sample_by_topic[topic_id]
        require(len(topic_samples) == 96, f"topic sample rows: {topic_id}")
        recent = [
            row
            for row in topic_samples
            if int(row["window_code"]) in (1, 2)
        ]
        require(len(recent) == 64, f"recent rows: {topic_id}")
        opaque = sum(
            row["local_reason"] == "OPAQUE_ARCHIVAL_IDENTIFIER"
            for row in recent
        )
        opaque_fraction = opaque / 64
        v1_state = str(candidate["dry_run_state"])
        v1_reason = str(candidate["dry_run_reason"])
        if opaque_fraction >= threshold:
            v2_state = "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
            v2_reason = "RECENT_SAMPLE_MAJORITY_OPAQUE_INDEXING_ARTIFACTS"
            branch = "RECENT_MAJORITY_OPAQUE_INDEXING_ANOMALY"
        else:
            v2_state = v1_state
            v2_reason = v1_reason
            branch = "RETAINED_V1_BRANCH"
        field_id = int(candidate["field_id"])
        entry_year = int(candidate["entry_year"])
        rank = rank_by_key[(field_id, topic_id, entry_year)]
        output.append(
            {
                "field_id": field_id,
                "field_candidate_rank": rank,
                "topic_id": topic_id,
                "topic_name": str(candidate["topic_name"]),
                "entry_year": entry_year,
                "recent_title_rows": 64,
                "recent_opaque_quarantine_rows": opaque,
                "recent_opaque_quarantine_fraction": opaque_fraction,
                "candidate_rule_v1_state": v1_state,
                "candidate_rule_v1_reason": v1_reason,
                "candidate_rule_v2_state": v2_state,
                "candidate_rule_v2_reason": v2_reason,
                "candidate_rule_v2_branch": branch,
                "state_changed": v1_state != v2_state,
                "production_disposition_created": False,
            }
        )
    output.sort(key=lambda row: (row["field_id"], row["field_candidate_rank"]))
    counts = Counter(row["candidate_rule_v2_state"] for row in output)
    changed = [row for row in output if row["state_changed"]]
    change_summary = {
        "schema_version": "pivot_feedback_d0_candidate_rule_v2_change_summary.v1",
        "candidate_rows": len(output),
        "unchanged_rows": len(output) - len(changed),
        "changed_rows": len(changed),
        "v2_would_content_pass_rows": counts["WOULD_CONTENT_PASS"],
        "v2_would_content_fail_indexing_anomaly_rows": counts[
            "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
        ],
        "v2_would_content_fail_semantic_off_topic_rows": counts[
            "WOULD_CONTENT_FAIL_SEMANTIC_OFF_TOPIC"
        ],
        "v2_would_content_unresolved_rows": counts["WOULD_CONTENT_UNRESOLVED"],
        "changed_candidates": changed,
        "production_candidate_rule_frozen": False,
        "production_content_dispositions_created": 0,
        "selection_certificate_issued": False,
    }
    expected = _mapping(config.get("expected_replay"), "expected replay")
    for key in (
        "candidate_rows",
        "unchanged_rows",
        "changed_rows",
        "v2_would_content_pass_rows",
        "v2_would_content_fail_indexing_anomaly_rows",
        "v2_would_content_fail_semantic_off_topic_rows",
        "v2_would_content_unresolved_rows",
        "production_candidate_rule_frozen",
        "production_content_dispositions_created",
        "selection_certificate_issued",
    ):
        require(change_summary[key] == expected[key], f"expected replay: {key}")
    expected_change = _mapping(expected.get("changed_candidate"), "expected change")
    require(len(changed) == 1, "changed candidate count")
    for key, value in expected_change.items():
        observed_key = {
            "v1_state": "candidate_rule_v1_state",
            "v2_state": "candidate_rule_v2_state",
            "v2_reason": "candidate_rule_v2_reason",
        }.get(key, key)
        require(changed[0][observed_key] == value, f"changed candidate: {key}")
    return output, change_summary


def _table(rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("recent_title_rows", pa.uint8(), nullable=False),
            pa.field("recent_opaque_quarantine_rows", pa.uint8(), nullable=False),
            pa.field("recent_opaque_quarantine_fraction", pa.float64(), nullable=False),
            pa.field("candidate_rule_v1_state", pa.string(), nullable=False),
            pa.field("candidate_rule_v1_reason", pa.string(), nullable=False),
            pa.field("candidate_rule_v2_state", pa.string(), nullable=False),
            pa.field("candidate_rule_v2_reason", pa.string(), nullable=False),
            pa.field("candidate_rule_v2_branch", pa.string(), nullable=False),
            pa.field("state_changed", pa.bool_(), nullable=False),
            pa.field("production_disposition_created", pa.bool_(), nullable=False),
        ],
        metadata={b"schema_version": REPLAY_SCHEMA.encode()},
    )
    return pa.Table.from_pylist(list(rows), schema=schema)


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


def _markdown(summary: Mapping[str, Any]) -> str:
    change = summary["changed_candidates"][0]
    return "\n".join(
        [
            "# Candidate rule v2 retrospective replay",
            "",
            f"Status: `{STATUS}`",
            "",
            "- replayed existing candidate samples: 5",
            f"- unchanged candidate states: {summary['unchanged_rows']}",
            f"- changed candidate states: {summary['changed_rows']}",
            (
                f"- changed candidate: F{change['field_id']}/"
                f"r{change['field_candidate_rank']}/T{change['topic_id']}/"
                f"{change['entry_year']}"
            ),
            f"- v1 state: `{change['candidate_rule_v1_state']}`",
            f"- v2 state: `{change['candidate_rule_v2_state']}`",
            f"- v2 reason: `{change['candidate_rule_v2_reason']}`",
            "",
            "The new branch is disclosed as post-diagnostic and this replay is",
            "not out-of-sample validation or a production-rule freeze. No new",
            "title, API request, production disposition, certificate, outcome,",
            "or model was opened.",
            "",
        ]
    )


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config = _load_config(config_path)
    samples, routes, candidates, input_evidence = _load_inputs(config)
    replay, change_summary = _replay(samples, routes, candidates, config)
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["replay"] = _write_parquet(
            temporary / str(contract["replay_basename"]),
            _table(replay),
        )
        change_path = temporary / str(contract["change_basename"])
        _write_private(change_path, _canonical_json_bytes(change_summary))
        artifacts["change"] = {
            "basename": change_path.name,
            "bytes": change_path.stat().st_size,
            "sha256": _sha256(change_path),
        }
        report_path = temporary / str(contract["report_basename"])
        _write_private(report_path, _markdown(change_summary).encode())
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
            "change_summary": change_summary,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "existing_title_rows_read": 480,
            "new_titles_read": False,
            "raw_openalex_main_table_read": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "production_candidate_rule_frozen": False,
            "production_content_dispositions_created": 0,
            "selection_certificate_issued": False,
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
        CandidateRuleV2ReplayBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_CANDIDATE_RULE_V2_REPLAY_V1", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
