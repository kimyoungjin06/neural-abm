#!/usr/bin/env python3
"""Independently verify production candidate rule v2 and frontier v3."""

from __future__ import annotations

import argparse
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
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_production_candidate_rule_v2.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-production-candidate-rule-v2-20260729"
)
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-production-candidate-rule-v2-20260729."
    "verification-v1.json"
)
CONFIG_SCHEMA = "pivot_feedback_d0_production_candidate_rule_config.v2"
OUTPUT_SCHEMA = "pivot_feedback_d0_production_candidate_rule_outputs.v2"
VERIFY_SCHEMA = "pivot_feedback_d0_production_candidate_rule_verification.v2"
VERIFY_STATUS = "PASS_INDEPENDENT_PRODUCTION_CANDIDATE_RULE_V2_AND_FRONTIER_V3"


class ProductionCandidateRuleV2VerificationError(RuntimeError):
    """Fail-closed production-candidate-rule-v2 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionCandidateRuleV2VerificationError(message)


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
    rule = _mapping(config.get("production_rule_v2"), "production rule")
    require(
        rule.get("amendment_disclosure")
        == "POST_DIAGNOSTIC_AFTER_T10543_RULE_V1_GAP",
        "amendment disclosure",
    )
    require(rule.get("amendment_is_out_of_sample_validated") is False, "validation claim")
    authorization = _mapping(config.get("authorization"), "authorization")
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
        "dispositions": root / str(contract["dispositions_basename"]),
        "frontier": root / str(contract["frontier_basename"]),
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
        == "PASS_PRODUCTION_CANDIDATE_RULE_V2_FROZEN_FRONTIER_V3_TARGETED_RANK3_REQUIRED",
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    candidate_binding = _mapping(frozen.get("candidate_table"), "candidates")
    candidates = pq.read_table(
        _ssd_file(candidate_binding, "basename", "sha256")
    ).to_pylist()
    require(len(candidates) == candidate_binding["rows"], "candidate rows")
    v1_binding = _mapping(frozen.get("production_rule_v1"), "production v1")
    v1 = pq.read_table(
        _ssd_file(v1_binding, "dispositions_basename", "dispositions_sha256")
    ).to_pylist()
    require(len(v1) == v1_binding["disposition_rows"], "v1 rows")
    replay_binding = _mapping(
        frozen.get("candidate_rule_v2_replay"),
        "v2 replay",
    )
    replay = pq.read_table(
        _ssd_file(replay_binding, "replay_basename", "replay_sha256")
    ).to_pylist()
    require(len(replay) == replay_binding["replay_rows"], "replay rows")
    return candidates, v1, replay


def _verify_dispositions(
    config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    v1_rows: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> list[dict[str, Any]]:
    observed = pq.read_table(path).to_pylist()
    require(len(observed) == 5, "disposition rows")
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
    for frozen, row in zip(
        config["production_candidate_dispositions"],
        observed,
        strict=True,
    ):
        key = (
            int(frozen["field_id"]),
            int(frozen["field_candidate_rank"]),
            int(frozen["topic_id"]),
            int(frozen["entry_year"]),
        )
        require(key in candidate_keys and key in replay_by_key, "disposition key")
        for name, value in zip(
            ("field_id", "field_candidate_rank", "topic_id", "entry_year"),
            key,
            strict=True,
        ):
            require(row[name] == value, f"disposition: {name}")
        require(row["content_state"] == frozen["content_state"], "disposition state")
        require(
            row["disposition_reason"] == frozen["disposition_reason"],
            "disposition reason",
        )
        require(row["production_rule_version"] == "candidate_rule_v2", "rule version")
        require(row["human_review_used"] is False, "human review")
        require(row["api_labels_used_as_ground_truth"] is False, "API ground truth")
        replay = replay_by_key[key]
        require(
            row["source_replay_state"] == replay["candidate_rule_v2_state"],
            "source replay state",
        )
        require(
            row["post_diagnostic_amendment_used"]
            is (
                replay["candidate_rule_v2_branch"]
                == "RECENT_MAJORITY_OPAQUE_INDEXING_ANOMALY"
            ),
            "amendment flag",
        )
        if key in v1_by_key:
            require(row["content_state"] == v1_by_key[key]["content_state"], "v1 state")
    require(
        sum(row["post_diagnostic_amendment_used"] for row in observed) == 1,
        "amendment disposition count",
    )
    return observed


def _compatible(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        int(first["subfield_id"]) != int(second["subfield_id"])
        and abs(int(first["entry_year"]) - int(second["entry_year"])) >= 9
    )


def _lazy_pair(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int] | None:
    first: Mapping[str, Any] | None = None
    for row in rows:
        state = row["content_state"]
        if state == "CONTENT_FAIL":
            continue
        if state == "CONTENT_PASS":
            first = row
        break
    if first is None:
        return None
    for row in rows:
        if int(row["field_candidate_rank"]) <= int(first["field_candidate_rank"]):
            continue
        if not _compatible(first, row):
            continue
        if row["content_state"] == "CONTENT_FAIL":
            continue
        if row["content_state"] != "CONTENT_PASS":
            return None
        return (
            int(first["field_candidate_rank"]),
            int(row["field_candidate_rank"]),
        )
    return None


def _verify_frontier(
    config: Mapping[str, Any],
    path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    require(len(rows) == 51, "frontier rows")
    medicine = [row for row in rows if row["field_id"] == 27]
    require(
        [row["content_state"] for row in medicine]
        == [
            "CONTENT_FAIL",
            "CONTENT_FAIL",
            "CONTENT_UNRESOLVED",
            "NOT_EVALUATED",
            "NOT_EVALUATED",
            "CONTENT_PASS",
        ],
        "medicine states",
    )
    require(_lazy_pair(medicine) is None, "medicine certificate")
    rank3_pass = [dict(row) for row in medicine]
    rank3_pass[2]["content_state"] = "CONTENT_PASS"
    require(_lazy_pair(rank3_pass) == (3, 6), "rank3 pass branch")
    rank3_fail = [dict(row) for row in medicine]
    rank3_fail[2]["content_state"] = "CONTENT_FAIL"
    require(_lazy_pair(rank3_fail) is None, "rank3 fail branch")
    require(
        medicine[2]["targeted_sample_required"] is True
        and medicine[2]["topic_id"] == 10543
        and medicine[2]["entry_year"] == 1986,
        "targeted successor",
    )
    counts = {
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
    }
    for key, value in counts.items():
        require(value == config["frontier_v3"][key], f"frontier count: {key}")
    return rows, counts


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
    candidates, v1_rows, replay_rows = _input_rows(config)
    dispositions = _verify_dispositions(
        config,
        candidates,
        v1_rows,
        replay_rows,
        paths["dispositions"],
    )
    _, counts = _verify_frontier(config, paths["frontier"])
    require(receipt["production_candidate_dispositions"] == len(dispositions), "receipt dispositions")
    require(receipt["frontier"]["total_content_fail_rows"] == 2, "receipt frontier")
    require(receipt["production_rule_v2_post_diagnostic"] is True, "post diagnostic")
    require(
        receipt["production_rule_v2_out_of_sample_validated"] is False,
        "out-of-sample claim",
    )
    for key in (
        "new_titles_read",
        "raw_openalex_main_table_read",
        "network_accessed",
        "credentials_accessed",
        "global_selection_certificate_issued",
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
        "production_candidate_dispositions": 5,
        "post_diagnostic_amendment_dispositions": 1,
        "frontier_counts": counts,
        "computer_science_field_certificate_issued": True,
        "medicine_required_candidate_rank": 3,
        "medicine_required_topic_id": 10543,
        "medicine_required_entry_year": 1986,
        "medicine_field_certificate_issued": False,
        "global_selection_certificate_issued": False,
        "new_api_request_creation_authorized": False,
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
        ProductionCandidateRuleV2VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_PRODUCTION_CANDIDATE_RULE_V2_VERIFICATION", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
