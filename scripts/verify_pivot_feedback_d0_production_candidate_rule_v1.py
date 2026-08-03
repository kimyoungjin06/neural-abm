#!/usr/bin/env python3
"""Independently verify production dispositions and adaptive frontier v2."""

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
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_production_candidate_rule_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-production-candidate-rule-v1-20260729"
)
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-production-candidate-rule-v1-20260729.verification-v1.json"
)
VERIFY_STATUS = "PASS_INDEPENDENT_PRODUCTION_CANDIDATE_RULE_AND_FRONTIER_V2"


class ProductionCandidateRuleVerificationError(RuntimeError):
    """Fail-closed production-candidate-rule verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionCandidateRuleVerificationError(message)


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
        raise ProductionCandidateRuleVerificationError(
            f"invalid YAML: {error}"
        ) from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProductionCandidateRuleVerificationError(
            f"invalid JSON: {error}"
        ) from error
    return _mapping(value, path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _ssd_root(value: Any) -> Path:
    require(isinstance(value, str) and "/" not in value, "unsafe SSD basename")
    path = (Path("/ssd") / value).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "SSD root escapes")
    require(path.is_dir() and not path.is_symlink(), "invalid SSD root")
    return path


def _dotted(document: Mapping[str, Any], key: str) -> Any:
    value: Any = document
    for part in key.split("."):
        value = _mapping(value, part)[part]
    return value


def _verify_files(
    config: Mapping[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output mode")
    receipt_path = output_root / str(contract["receipt_basename"])
    receipt = _read_json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_PRODUCTION_CANDIDATE_RULE_FROZEN_FRONTIER_V2_TARGETED_SAMPLE_REQUIRED",
        "receipt status",
    )
    paths = {
        "dispositions": output_root / str(contract["dispositions_basename"]),
        "frontier": output_root / str(contract["frontier_basename"]),
        "report": output_root / str(contract["report_basename"]),
    }
    require(
        {path.name for path in output_root.iterdir()}
        == {path.name for path in paths.values()} | {receipt_path.name},
        "artifact set",
    )
    for key, path in paths.items():
        require(path.stat().st_mode & 0o777 == 0o600, f"artifact mode: {key}")
        require(
            _sha256(path) == receipt["artifacts"][key]["sha256"],
            f"artifact SHA: {key}",
        )
    require(receipt_path.stat().st_mode & 0o777 == 0o600, "receipt mode")
    require(
        hashlib.sha256(_canonical_json_bytes(receipt["artifacts"])).hexdigest()
        == receipt["artifact_tree_sha256"],
        "artifact tree",
    )
    return receipt, paths


def validate(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _read_yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_production_candidate_rule_config.v1",
        "config schema",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = _repo_file(parent)
    parent_document = _read_yaml(parent_path)
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "targeted_t10543_local_sample_execution",
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
    receipt, paths = _verify_files(config, output_root)

    frozen = _mapping(config.get("frozen_inputs"), "frozen inputs")
    candidate_binding = _mapping(frozen.get("candidate_table"), "candidate table")
    candidate_path = (
        _ssd_root(candidate_binding["root_basename"])
        / str(candidate_binding["basename"])
    )
    require(_sha256(candidate_path) == candidate_binding["sha256"], "candidate SHA")
    candidate_rows = pq.read_table(candidate_path).to_pylist()
    candidate_by_key = {
        (
            int(row["field_id"]),
            int(row["field_candidate_rank"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        ): row
        for row in candidate_rows
    }
    dry = _mapping(frozen.get("dry_run"), "dry-run")
    dry_root = _ssd_root(dry["operator_root_basename"])
    dry_path = dry_root / str(dry["candidate_basename"])
    require(_sha256(dry_path) == dry["candidate_sha256"], "dry candidate SHA")
    dry_rows = pq.read_table(dry_path).to_pylist()
    dry_by_key = {
        (
            int(row["field_id"]),
            int(row["field_candidate_rank"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        ): row
        for row in dry_rows
    }

    observed_dispositions = pq.read_table(paths["dispositions"]).to_pylist()
    require(len(observed_dispositions) == 4, "disposition rows")
    expected_states: dict[tuple[int, int], str] = {}
    for frozen_row, observed in zip(
        config["production_candidate_dispositions"],
        observed_dispositions,
        strict=True,
    ):
        key = (
            int(frozen_row["field_id"]),
            int(frozen_row["field_candidate_rank"]),
            int(frozen_row["topic_id"]),
            int(frozen_row["entry_year"]),
        )
        require(key in candidate_by_key and key in dry_by_key, "disposition key")
        require(observed["field_id"] == key[0], "disposition field")
        require(observed["field_candidate_rank"] == key[1], "disposition rank")
        require(observed["topic_id"] == key[2], "disposition topic")
        require(observed["entry_year"] == key[3], "disposition year")
        require(observed["content_state"] == frozen_row["content_state"], "state")
        require(
            observed["disposition_reason"] == frozen_row["disposition_reason"],
            "reason",
        )
        require(observed["human_review_used"] is False, "human review used")
        require(
            observed["api_labels_used_as_ground_truth"] is False,
            "API ground truth",
        )
        expected_states[(key[0], key[1])] = str(frozen_row["content_state"])

    observed_frontier = pq.read_table(paths["frontier"]).to_pylist()
    require(len(observed_frontier) == 51, "frontier rows")
    for row in observed_frontier:
        key = (int(row["field_id"]), int(row["field_candidate_rank"]))
        if key in expected_states:
            require(row["content_state"] == expected_states[key], "frozen frontier state")
        elif key == (27, 2):
            require(row["content_state"] == "CONTENT_UNRESOLVED", "target state")
            require(row["targeted_sample_required"] is True, "target sample flag")
            require(row["topic_id"] == 10543, "target topic")
        else:
            require(row["content_state"] == "NOT_EVALUATED", "structural state")
    counts = {
        "total_frontier_rows": len(observed_frontier),
        "total_content_pass_rows": sum(
            row["content_state"] == "CONTENT_PASS" for row in observed_frontier
        ),
        "total_content_fail_rows": sum(
            row["content_state"] == "CONTENT_FAIL" for row in observed_frontier
        ),
        "total_content_unresolved_rows": sum(
            row["content_state"] == "CONTENT_UNRESOLVED" for row in observed_frontier
        ),
        "total_not_evaluated_rows": sum(
            row["content_state"] == "NOT_EVALUATED" for row in observed_frontier
        ),
    }
    require(
        counts
        == {
            key: config["frontier_v2"][key]
            for key in counts
        },
        "frontier counts",
    )
    require(receipt["frontier"]["field_certificates_issued"] == 1, "field certificate")
    require(
        receipt["frontier"]["global_selection_certificate_issued"] is False,
        "global certificate",
    )
    for key in (
        "new_titles_read",
        "raw_openalex_main_table_read",
        "network_accessed",
        "credentials_accessed",
        "global_selection_certificate_issued",
        "deterministic_reselection_executed",
        "outcomes_read",
        "g4_open",
    ):
        require(receipt[key] is False, f"unsafe receipt flag: {key}")
    require(receipt["api_requests_created"] == 0, "API request creation")
    require(receipt["api_requests_submitted"] == 0, "API submission")
    return {
        "schema_version": "pivot_feedback_d0_production_candidate_rule_verification.v1",
        "status": VERIFY_STATUS,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": _sha256(config_path),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(
            output_root / str(config["output_contract"]["receipt_basename"])
        ),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "production_candidate_dispositions": 4,
        "frontier": receipt["frontier"],
        "computer_science_field_certificate_issued": True,
        "medicine_required_topic_id": 10543,
        "medicine_required_content_state": "CONTENT_UNRESOLVED",
        "global_selection_certificate_issued": False,
        "targeted_sample_execution_authorized": False,
        "new_api_request_creation_authorized": False,
        "deterministic_reselection_authorized": False,
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
        ProductionCandidateRuleVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_PRODUCTION_CANDIDATE_RULE_VERIFICATION_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
