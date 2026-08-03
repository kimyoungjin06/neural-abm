#!/usr/bin/env python3
"""Independently verify the local-abstract rank-5 terminal closeout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_rank5_local_abstract_closeout_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-rank5-local-abstract-closeout-v1-20260730"
)
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-rank5-local-abstract-closeout-v1-20260730."
    "verification-v1.json"
)
RESULT_STATUS = "PASS_RANK5_LOCAL_ABSTRACT_UNRESOLVED_TERMINAL_CLOSEOUT_FROZEN"
VERIFY_STATUS = "PASS_INDEPENDENT_RANK5_LOCAL_ABSTRACT_UNRESOLVED_TERMINAL_CLOSEOUT"
OUTPUT_FILES = {
    "cumulative_production_candidate_dispositions_v7.parquet",
    "adaptive_candidate_frontier_v8.parquet",
    "terminal_selection_certificate_status_v3.json",
    "RANK5_LOCAL_ABSTRACT_TERMINAL_CLOSEOUT.md",
    "rank5_local_abstract_closeout.outputs.json",
}


class Rank5LocalAbstractCloseoutVerificationError(RuntimeError):
    """Fail-closed local abstract closeout verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5LocalAbstractCloseoutVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _bound(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "input root")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), f"input: {path.name}")
    require(_sha256(path) == binding[sha_key], f"input SHA: {path.name}")
    return path


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def validate(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_rank5_local_abstract_closeout_config.v1",
        "config schema",
    )
    require(
        config.get("status")
        == "FROZEN_RANK5_LOCAL_ABSTRACT_UNRESOLVED_CLOSEOUT_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "parent path")
    require(_sha256(parent_path) == parent["sha256"], "parent SHA")
    parent_doc = _yaml(parent_path)
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")

    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.resolve(strict=True)
    require(output_root.parent == Path("/ssd").resolve(), "output parent")
    require(output_root.name == output["root_basename"], "output root")
    require({path.name for path in output_root.iterdir()} == OUTPUT_FILES, "output file set")
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"output mode: {path.name}")

    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("prior_closeout"), "prior")
    prior_dispositions_path = _bound(prior, "dispositions_basename", "dispositions_sha256")
    prior_frontier_path = _bound(prior, "frontier_basename", "frontier_sha256")
    prior_certificate_path = _bound(prior, "certificate_basename", "certificate_sha256")
    prior_receipt_path = _bound(prior, "receipt_basename", "receipt_sha256")
    prior_verification_path = Path("/ssd") / str(prior["verification_basename"])
    require(_sha256(prior_verification_path) == prior["verification_sha256"], "prior verification SHA")
    result = _mapping(frozen.get("abstract_result"), "abstract result")
    candidate_path = _bound(result, "candidate_basename", "candidate_sha256")
    decisions_path = _bound(result, "decisions_basename", "decisions_sha256")
    result_receipt_path = _bound(result, "receipt_basename", "receipt_sha256")
    result_verification_path = Path("/ssd") / str(result["verification_basename"])
    require(_sha256(result_verification_path) == result["verification_sha256"], "result verification SHA")
    candidate = _json(candidate_path)
    require(
        (
            candidate.get("final_accept_rows"),
            candidate.get("final_reject_rows"),
            candidate.get("final_unresolved_rows"),
        )
        == (28, 22, 14),
        "candidate counts",
    )
    require(candidate.get("candidate_content_state") == "CONTENT_UNRESOLVED", "candidate state")

    prior_dispositions = pq.read_table(prior_dispositions_path).to_pylist()
    expected_dispositions = [dict(row) for row in prior_dispositions]
    frozen_disposition = _mapping(config.get("rank5_production_disposition"), "disposition")
    rank5_disposition = next(
        row
        for row in expected_dispositions
        if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    )
    rank5_disposition.update(
        {
            "content_state": frozen_disposition["content_state"],
            "disposition_reason": frozen_disposition["disposition_reason"],
            "production_rule_version": frozen_disposition["production_rule_version"],
            "source_replay_state": frozen_disposition["source_replay_state"],
            "post_diagnostic_amendment_used": frozen_disposition["post_diagnostic_amendment_used"],
            "human_review_used": frozen_disposition["human_review_used"],
            "api_labels_used_as_ground_truth": frozen_disposition["api_labels_used_as_ground_truth"],
        }
    )
    dispositions_path = output_root / str(output["dispositions_basename"])
    dispositions = pq.read_table(dispositions_path)
    require(
        (dispositions.schema.metadata or {}).get(b"schema_version", b"").decode()
        == output["dispositions_schema_version"],
        "disposition schema",
    )
    require(dispositions.to_pylist() == expected_dispositions, "disposition rows")

    prior_frontier = pq.read_table(prior_frontier_path).to_pylist()
    expected_frontier = [dict(row) for row in prior_frontier]
    frontier_frozen = _mapping(config.get("frontier_v8"), "frontier")
    rank5 = next(row for row in expected_frontier if row["field_id"] == 27 and row["field_candidate_rank"] == 5)
    rank6 = next(row for row in expected_frontier if row["field_id"] == 27 and row["field_candidate_rank"] == 6)
    rank5.update(
        {
            "frontier_role": frontier_frozen["rank5_frontier_role"],
            "content_state": "CONTENT_UNRESOLVED",
            "state_reason": frontier_frozen["rank5_state_reason"],
            "targeted_sample_required": False,
        }
    )
    rank6.update(
        {
            "frontier_role": frontier_frozen["rank6_frontier_role"],
            "targeted_sample_required": False,
        }
    )
    counts = {
        "total_frontier_rows": len(expected_frontier),
        "total_content_pass_rows": sum(row["content_state"] == "CONTENT_PASS" for row in expected_frontier),
        "total_content_fail_rows": sum(row["content_state"] == "CONTENT_FAIL" for row in expected_frontier),
        "total_content_unresolved_rows": sum(row["content_state"] == "CONTENT_UNRESOLVED" for row in expected_frontier),
        "total_not_evaluated_rows": sum(row["content_state"] == "NOT_EVALUATED" for row in expected_frontier),
        "field_certificates_issued": 1,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
    }
    frontier_path = output_root / str(output["frontier_basename"])
    frontier = pq.read_table(frontier_path)
    require(
        (frontier.schema.metadata or {}).get(b"schema_version", b"").decode()
        == output["frontier_schema_version"],
        "frontier schema",
    )
    require(frontier.to_pylist() == expected_frontier, "frontier rows")
    for key, value in counts.items():
        require(value == frontier_frozen[key], f"frontier count: {key}")

    certificate_path = output_root / str(output["certificate_basename"])
    certificate = _json(certificate_path)
    require(certificate == config["terminal_certificate_status"], "certificate")
    require(
        certificate.get("status")
        == "BLOCKED_REQUIRED_RANK5_LOCAL_ABSTRACT_AUGMENTED_CONTENT_UNRESOLVED",
        "certificate status",
    )
    require(certificate.get("medicine_field_certificate_issued") is False, "Medicine certificate")
    require(certificate.get("global_selection_certificate_issued") is False, "global certificate")
    require(certificate.get("medicine_blocking_counts") == {"accept": 28, "reject": 22, "unresolved": 14}, "certificate counts")

    receipt_path = output_root / str(output["receipt_basename"])
    receipt = _json(receipt_path)
    require(receipt.get("status") == RESULT_STATUS, "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "receipt config SHA")
    expected_hashes = {
        "prior_dispositions_sha256": _sha256(prior_dispositions_path),
        "prior_frontier_sha256": _sha256(prior_frontier_path),
        "prior_certificate_sha256": _sha256(prior_certificate_path),
        "prior_receipt_sha256": _sha256(prior_receipt_path),
        "prior_verification_sha256": _sha256(prior_verification_path),
        "abstract_candidate_sha256": _sha256(candidate_path),
        "abstract_decisions_sha256": _sha256(decisions_path),
        "abstract_result_receipt_sha256": _sha256(result_receipt_path),
        "abstract_result_verification_sha256": _sha256(result_verification_path),
    }
    require(receipt.get("input_hashes") == expected_hashes, "input hashes")
    require(receipt.get("rank5_counts") == {"accept": 28, "reject": 22, "unresolved": 14}, "receipt counts")
    require(receipt.get("local_abstract_evidence_used") is True, "abstract evidence")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(receipt.get("provider_labels_used_as_ground_truth") is False, "ground truth")
    require(receipt.get("frontier") == counts, "receipt frontier")
    require(receipt.get("terminal_certificate_status") == certificate, "receipt certificate")
    require(receipt.get("outcomes_read") is False, "outcomes")
    expected_artifacts = {
        "dispositions": _artifact(dispositions_path, rows=8),
        "frontier": _artifact(frontier_path, rows=51),
        "certificate_status": _artifact(certificate_path),
        "report": _artifact(output_root / str(output["report_basename"])),
    }
    require(receipt.get("artifacts") == expected_artifacts, "receipt artifacts")
    return {
        "schema_version": "pivot_feedback_d0_rank5_local_abstract_closeout_verification.v1",
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "output_root_basename": output_root.name,
        "disposition_rows_verified": 8,
        "frontier_rows_verified": 51,
        "rank5_content_state": "CONTENT_UNRESOLVED",
        "final_accept_rows": 28,
        "final_reject_rows": 22,
        "final_unresolved_rows": 14,
        "local_abstract_evidence_used": True,
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_used_as_ground_truth": False,
        "medicine_field_certificate_issued": False,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
        "outcomes_read": False,
        "g4_open": False,
        "artifacts": {
            name: _artifact(output_root / name)
            for name in sorted(OUTPUT_FILES)
        },
    }


def verify(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT,
    report_path: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    result = validate(config_path=config_path, output_root=output_root)
    report_path = report_path.absolute()
    require(report_path.parent == Path("/ssd"), "report parent")
    require(not report_path.exists(), "report exists")
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report_path.chmod(0o600)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = (
            validate(config_path=args.config, output_root=args.output)
            if args.check_only
            else verify(
                config_path=args.config,
                output_root=args.output,
                report_path=args.report,
            )
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (Rank5LocalAbstractCloseoutVerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_LOCAL_ABSTRACT_CLOSEOUT_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
