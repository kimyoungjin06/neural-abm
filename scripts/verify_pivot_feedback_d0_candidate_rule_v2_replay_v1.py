#!/usr/bin/env python3
"""Independently verify the five-candidate rule-v2 retrospective replay."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

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
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-candidate-rule-v2-replay-v1-20260729."
    "verification-v1.json"
)
CONFIG_SCHEMA = "pivot_feedback_d0_candidate_rule_v2_replay_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_candidate_rule_v2_replay_outputs.v1"
VERIFY_SCHEMA = "pivot_feedback_d0_candidate_rule_v2_replay_verification.v1"
VERIFY_STATUS = "PASS_INDEPENDENT_CANDIDATE_RULE_V2_FIVE_CANDIDATE_REPLAY"


class CandidateRuleV2ReplayVerificationError(RuntimeError):
    """Fail-closed candidate-rule-v2 replay verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateRuleV2ReplayVerificationError(message)


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


def _validate_config(path: Path) -> dict[str, Any]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    rule = _mapping(config.get("candidate_rule_v2"), "candidate rule")
    require(
        rule.get("amendment_disclosure")
        == "POST_DIAGNOSTIC_AFTER_T10543_RULE_V1_GAP",
        "amendment disclosure",
    )
    branch = _mapping(rule.get("new_branch"), "new branch")
    require(branch.get("minimum_opaque_quarantine_fraction") == 0.5, "threshold")
    require(rule.get("replay_is_out_of_sample_validation") is False, "validation claim")
    require(rule.get("replay_is_production_freeze") is False, "production freeze")
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
        "replay": root / str(contract["replay_basename"]),
        "change": root / str(contract["change_basename"]),
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
        receipt.get("status")
        == "PASS_CANDIDATE_RULE_V2_FIVE_CANDIDATE_RETROSPECTIVE_REPLAY",
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


def _input_rows(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]], list[dict[str, Any]]]:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    samples: list[dict[str, Any]] = []
    routes: dict[str, dict[str, str]] = {}
    candidates: list[dict[str, Any]] = []
    for key in ("original_sample", "t10543_sample"):
        binding = _mapping(frozen.get(key), key)
        path = _ssd_file(binding, "sample_basename", "sample_sha256")
        rows = pq.read_table(path).to_pylist()
        require(len(rows) == binding["sample_rows"], f"sample rows: {key}")
        samples.extend(rows)
    for key in ("original_routes", "t10543_routes"):
        binding = _mapping(frozen.get(key), key)
        path = _ssd_file(binding, "routes_basename", "routes_sha256")
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == binding["route_rows"], f"route rows: {key}")
        for row in rows:
            require(row["openalex_id"] not in routes, "duplicate route ID")
            routes[row["openalex_id"]] = row
    for key in ("original_candidate_v1", "t10543_candidate_v1"):
        binding = _mapping(frozen.get(key), key)
        path = _ssd_file(binding, "candidate_basename", "candidate_sha256")
        rows = pq.read_table(path).to_pylist()
        require(len(rows) == binding["candidate_rows"], f"candidate rows: {key}")
        candidates.extend(rows)
    require(len(samples) == len(routes) == 480, "combined title rows")
    require({str(row["openalex_id"]) for row in samples} == set(routes), "sample routes")
    require(len(candidates) == 5, "combined candidates")
    return samples, routes, candidates


def _expected_replay(
    samples: Sequence[Mapping[str, Any]],
    routes: Mapping[str, Mapping[str, str]],
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_topic: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        topic_id = int(sample["topic_id"])
        by_topic.setdefault(topic_id, []).append(
            {**sample, **routes[str(sample["openalex_id"])]}
        )
    ranks = {
        (17, 14430, 2003): 1,
        (17, 10579, 2012): 45,
        (27, 12505, 1991): 1,
        (27, 10543, 1993): 2,
        (27, 12421, 1953): 6,
    }
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        topic_id = int(candidate["topic_id"])
        recent = [
            row
            for row in by_topic[topic_id]
            if int(row["window_code"]) in (1, 2)
        ]
        require(len(recent) == 64, f"recent rows: {topic_id}")
        opaque = sum(
            row["local_reason"] == "OPAQUE_ARCHIVAL_IDENTIFIER"
            for row in recent
        )
        fraction = opaque / 64
        v1_state = str(candidate["dry_run_state"])
        v1_reason = str(candidate["dry_run_reason"])
        if fraction >= 0.5:
            v2_state = "WOULD_CONTENT_FAIL_INDEXING_ANOMALY"
            v2_reason = "RECENT_SAMPLE_MAJORITY_OPAQUE_INDEXING_ARTIFACTS"
            branch = "RECENT_MAJORITY_OPAQUE_INDEXING_ANOMALY"
        else:
            v2_state = v1_state
            v2_reason = v1_reason
            branch = "RETAINED_V1_BRANCH"
        field_id = int(candidate["field_id"])
        entry_year = int(candidate["entry_year"])
        output.append(
            {
                "field_id": field_id,
                "field_candidate_rank": ranks[(field_id, topic_id, entry_year)],
                "topic_id": topic_id,
                "topic_name": str(candidate["topic_name"]),
                "entry_year": entry_year,
                "recent_title_rows": 64,
                "recent_opaque_quarantine_rows": opaque,
                "recent_opaque_quarantine_fraction": fraction,
                "candidate_rule_v1_state": v1_state,
                "candidate_rule_v1_reason": v1_reason,
                "candidate_rule_v2_state": v2_state,
                "candidate_rule_v2_reason": v2_reason,
                "candidate_rule_v2_branch": branch,
                "state_changed": v1_state != v2_state,
                "production_disposition_created": False,
            }
        )
    return sorted(output, key=lambda row: (row["field_id"], row["field_candidate_rank"]))


def validate(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config_sha256 = _sha256(config_path)
    config = _validate_config(config_path)
    receipt, paths = _verify_output(
        config,
        output_root,
        config_sha256=config_sha256,
    )
    samples, routes, candidates = _input_rows(config)
    expected = _expected_replay(samples, routes, candidates)
    observed = pq.read_table(paths["replay"]).to_pylist()
    require(observed == expected, "replay rows")
    change = _read_json(paths["change"])
    require(change == receipt["change_summary"], "change receipt")
    changed = [row for row in expected if row["state_changed"]]
    require(len(changed) == 1 and changed[0]["topic_id"] == 10543, "changed candidate")
    counts = Counter(row["candidate_rule_v2_state"] for row in expected)
    require(counts["WOULD_CONTENT_PASS"] == 3, "pass count")
    require(counts["WOULD_CONTENT_FAIL_INDEXING_ANOMALY"] == 2, "fail count")
    require(counts["WOULD_CONTENT_UNRESOLVED"] == 0, "unresolved count")
    for key in (
        "new_titles_read",
        "raw_openalex_main_table_read",
        "network_accessed",
        "credentials_accessed",
        "production_candidate_rule_frozen",
        "selection_certificate_issued",
        "deterministic_reselection_executed",
        "outcomes_read",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt[key] is False, f"unsafe receipt flag: {key}")
    require(receipt["api_requests_created"] == 0, "API requests created")
    require(receipt["api_requests_submitted"] == 0, "API requests submitted")
    require(
        receipt["production_content_dispositions_created"] == 0,
        "production dispositions",
    )
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
        "candidate_rows": 5,
        "unchanged_rows": 4,
        "changed_rows": 1,
        "changed_topic_id": 10543,
        "v2_would_content_pass_rows": 3,
        "v2_would_content_fail_indexing_anomaly_rows": 2,
        "v2_would_content_unresolved_rows": 0,
        "production_candidate_rule_frozen": False,
        "production_content_dispositions_created": 0,
        "selection_certificate_issued": False,
        "outcomes_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.config, args.output_root)
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
        CandidateRuleV2ReplayVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_CANDIDATE_RULE_V2_REPLAY_VERIFICATION_V1", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
