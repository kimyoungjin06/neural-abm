#!/usr/bin/env python3
"""Independently verify the rank-5 API-augmented terminal closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_rank5_api_augmented_closeout_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-rank5-api-augmented-closeout-v1-20260730"
)
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-rank5-api-augmented-closeout-v1-20260730."
    "verification-v1.json"
)
RESULT_STATUS = "PASS_RANK5_API_AUGMENTED_UNRESOLVED_TERMINAL_CLOSEOUT_FROZEN"
VERIFY_STATUS = "PASS_INDEPENDENT_RANK5_API_AUGMENTED_UNRESOLVED_TERMINAL_CLOSEOUT"
OUTPUT_FILES = {
    "cumulative_production_candidate_dispositions_v6.parquet",
    "adaptive_candidate_frontier_v7.parquet",
    "terminal_selection_certificate_status_v2.json",
    "RANK5_API_AUGMENTED_TERMINAL_CLOSEOUT.md",
    "rank5_api_augmented_closeout.outputs.json",
}


class Rank5ApiAugmentedCloseoutVerificationError(RuntimeError):
    """Fail-closed closeout verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5ApiAugmentedCloseoutVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _nested(document: Mapping[str, Any], dotted: str) -> Any:
    value: Any = document
    for part in dotted.split("."):
        require(isinstance(value, Mapping) and part in value, f"missing {dotted}")
        value = value[part]
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _repo(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return _yaml(path)


def _bound(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "input root")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), f"input invalid: {path.name}")
    require(_sha256(path) == binding[sha_key], f"input SHA: {path.name}")
    return path


def _config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_rank5_api_augmented_closeout_config.v1",
        "config schema",
    )
    require(
        config.get("status")
        == "FROZEN_RANK5_API_AUGMENTED_UNRESOLVED_CLOSEOUT_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_doc = _repo(parent)
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    for dotted, expected in _mapping(parent.get("required_fields"), "parent fields").items():
        observed = _nested(parent_doc, dotted)
        require(type(observed) is type(expected) and observed == expected, f"parent field: {dotted}")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "new_api_request",
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


def _expected_inputs(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("prior_terminal"), "prior")
    dispositions_path = _bound(prior, "dispositions_basename", "dispositions_sha256")
    frontier_path = _bound(prior, "frontier_basename", "frontier_sha256")
    certificate_path = _bound(prior, "certificate_basename", "certificate_sha256")
    receipt_path = _bound(prior, "receipt_basename", "receipt_sha256")
    verification_path = Path("/ssd") / str(prior["verification_basename"])
    require(_sha256(verification_path) == prior["verification_sha256"], "prior verification SHA")
    dispositions = pq.read_table(dispositions_path).to_pylist()
    frontier = pq.read_table(frontier_path).to_pylist()
    require(len(dispositions) == 8 and len(frontier) == 51, "prior rows")
    require(_json(certificate_path).get("status") == "BLOCKED_REQUIRED_RANK5_CONTENT_UNRESOLVED", "prior certificate")
    result = _mapping(frozen.get("corrected_api_result"), "result")
    candidate_path = _bound(result, "candidate_basename", "candidate_sha256")
    decisions_path = _bound(result, "decisions_basename", "decisions_sha256")
    result_receipt = _bound(result, "receipt_basename", "receipt_sha256")
    result_verification = Path("/ssd") / str(result["verification_basename"])
    require(_sha256(result_verification) == result["verification_sha256"], "result verification SHA")
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        require(len(list(csv.DictReader(handle))) == 25, "decision rows")
    candidate = _json(candidate_path)
    require(candidate.get("candidate_content_state") == "CONTENT_UNRESOLVED", "candidate state")
    require(candidate.get("augmented_accept_rows") == 27, "candidate accept")
    require(candidate.get("augmented_reject_rows") == 21, "candidate reject")
    require(candidate.get("augmented_unresolved_rows") == 16, "candidate unresolved")
    require(_json(result_receipt).get("production_disposition_created") is False, "premature disposition")
    hashes = {
        "prior_dispositions_sha256": _sha256(dispositions_path),
        "prior_frontier_sha256": _sha256(frontier_path),
        "prior_certificate_sha256": _sha256(certificate_path),
        "prior_receipt_sha256": _sha256(receipt_path),
        "prior_verification_sha256": _sha256(verification_path),
        "corrected_candidate_sha256": _sha256(candidate_path),
        "corrected_decisions_sha256": _sha256(decisions_path),
        "corrected_result_receipt_sha256": _sha256(result_receipt),
        "corrected_result_verification_sha256": _sha256(result_verification),
    }
    return dispositions, frontier, candidate, hashes


def _expected_dispositions(
    rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frozen = _mapping(config.get("rank5_production_disposition"), "disposition")
    matches = [
        row
        for row in rows
        if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    ]
    require(len(matches) == 1, "rank5 disposition")
    matches[0].update(
        {
            "content_state": frozen["content_state"],
            "disposition_reason": frozen["disposition_reason"],
            "production_rule_version": frozen["production_rule_version"],
            "source_replay_state": frozen["source_replay_state"],
            "post_diagnostic_amendment_used": frozen["post_diagnostic_amendment_used"],
            "human_review_used": frozen["human_review_used"],
            "api_labels_used_as_ground_truth": frozen["api_labels_used_as_ground_truth"],
        }
    )
    return rows


def _expected_frontier(
    rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen = _mapping(config.get("frontier_v7"), "frontier")
    rank5 = next(
        row for row in rows if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    )
    rank6 = next(
        row for row in rows if row["field_id"] == 27 and row["field_candidate_rank"] == 6
    )
    rank5.update(
        {
            "frontier_role": frozen["rank5_frontier_role"],
            "content_state": "CONTENT_UNRESOLVED",
            "state_reason": frozen["rank5_state_reason"],
            "targeted_sample_required": False,
        }
    )
    rank6.update(
        {
            "frontier_role": frozen["rank6_frontier_role"],
            "targeted_sample_required": False,
        }
    )
    counts = {
        "total_frontier_rows": len(rows),
        "total_content_pass_rows": sum(row["content_state"] == "CONTENT_PASS" for row in rows),
        "total_content_fail_rows": sum(row["content_state"] == "CONTENT_FAIL" for row in rows),
        "total_content_unresolved_rows": sum(row["content_state"] == "CONTENT_UNRESOLVED" for row in rows),
        "total_not_evaluated_rows": sum(row["content_state"] == "NOT_EVALUATED" for row in rows),
        "field_certificates_issued": 1,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
    }
    for key, value in counts.items():
        require(value == frozen[key], f"frontier count: {key}")
    return rows, counts


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        result["rows"] = rows
    return result


def validate(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _config(config_path)
    output = output.resolve(strict=True)
    contract = _mapping(config.get("output_contract"), "output")
    require(output.name == contract["root_basename"], "output basename")
    require({path.name for path in output.iterdir()} == OUTPUT_FILES, "output files")
    for path in output.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"output mode: {path.name}")
    dispositions_in, frontier_in, candidate, input_hashes = _expected_inputs(config)
    expected_dispositions = _expected_dispositions(dispositions_in, config)
    expected_frontier, frontier_counts = _expected_frontier(frontier_in, config)
    dispositions_path = output / str(contract["dispositions_basename"])
    frontier_path = output / str(contract["frontier_basename"])
    dispositions = pq.read_table(dispositions_path)
    frontier = pq.read_table(frontier_path)
    require(
        (dispositions.schema.metadata or {}).get(b"schema_version", b"").decode()
        == contract["dispositions_schema_version"],
        "disposition schema version",
    )
    require(
        (frontier.schema.metadata or {}).get(b"schema_version", b"").decode()
        == contract["frontier_schema_version"],
        "frontier schema version",
    )
    require(dispositions.to_pylist() == expected_dispositions, "disposition rows")
    require(frontier.to_pylist() == expected_frontier, "frontier rows")
    certificate = _json(output / str(contract["certificate_basename"]))
    require(certificate == config["terminal_certificate_status"], "certificate status")
    require(certificate.get("status") == "BLOCKED_REQUIRED_RANK5_AUGMENTED_CONTENT_UNRESOLVED", "certificate block")
    receipt = _json(output / str(contract["receipt_basename"]))
    require(receipt.get("status") == RESULT_STATUS, "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "receipt config SHA")
    require(receipt.get("input_hashes") == input_hashes, "receipt input hashes")
    require(receipt.get("rank5_content_state") == "CONTENT_UNRESOLVED", "receipt state")
    require(receipt.get("corrected_api_evidence_used") is True, "correct API evidence")
    require(receipt.get("wrong_target_api_evidence_used") is False, "wrong API evidence")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(receipt.get("api_labels_used_as_ground_truth") is False, "ground truth")
    require(receipt.get("frontier") == frontier_counts, "receipt frontier")
    require(receipt.get("medicine_field_certificate_issued") is False, "Medicine certificate")
    require(receipt.get("global_selection_certificate_issued") is False, "global certificate")
    require(receipt.get("outcomes_read") is False, "outcomes")
    expected_artifacts = {
        "dispositions": _artifact(dispositions_path, rows=8),
        "frontier": _artifact(frontier_path, rows=51),
        "certificate_status": _artifact(output / str(contract["certificate_basename"])),
        "report": _artifact(output / str(contract["report_basename"])),
    }
    require(receipt.get("artifacts") == expected_artifacts, "receipt artifacts")
    return {
        "schema_version": "pivot_feedback_d0_rank5_api_augmented_closeout_verification.v1",
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "output_root_basename": output.name,
        "disposition_rows_verified": 8,
        "frontier_rows_verified": 51,
        "rank5_content_state": "CONTENT_UNRESOLVED",
        "augmented_accept_rows": candidate["augmented_accept_rows"],
        "augmented_reject_rows": candidate["augmented_reject_rows"],
        "augmented_unresolved_rows": candidate["augmented_unresolved_rows"],
        "corrected_api_evidence_used": True,
        "wrong_target_api_evidence_used": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "medicine_field_certificate_issued": False,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
        "outcomes_read": False,
        "g4_open": False,
        "artifacts": {
            name: _artifact(output / name)
            for name in sorted(OUTPUT_FILES)
        },
    }


def verify(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
    report: Path = DEFAULT_REPORT,
) -> dict[str, Any]:
    result = validate(config_path=config_path, output=output)
    report = report.absolute()
    require(not report.exists(), "verification report exists")
    with report.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    report.chmod(0o600)
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
            validate(config_path=args.config, output=args.output)
            if args.check_only
            else verify(config_path=args.config, output=args.output, report=args.report)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (Rank5ApiAugmentedCloseoutVerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_API_AUGMENTED_CLOSEOUT_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
