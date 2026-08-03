#!/usr/bin/env python3
"""Materialize the correct-HCV API-augmented rank-5 terminal closeout."""

from __future__ import annotations

import argparse
import csv
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
    / "experiments/configs/pivot_feedback_d0_rank5_api_augmented_closeout_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-rank5-api-augmented-closeout-v1-20260730"
)
STATUS = "PASS_RANK5_API_AUGMENTED_UNRESOLVED_TERMINAL_CLOSEOUT_FROZEN"


class Rank5ApiAugmentedCloseoutBuildError(RuntimeError):
    """Fail-closed closeout build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5ApiAugmentedCloseoutBuildError(message)


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


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _bound_file(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "input root")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), f"input invalid: {path.name}")
    require(_sha256(path) == binding[sha_key], f"input SHA: {path.name}")
    return path


def _load_config(path: Path) -> dict[str, Any]:
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
    parent_doc = _yaml(_repo_file(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    for dotted, expected in _mapping(parent.get("required_fields"), "parent fields").items():
        observed = _nested(parent_doc, dotted)
        require(type(observed) is type(expected) and observed == expected, f"parent field: {dotted}")
    disposition = _mapping(config.get("rank5_production_disposition"), "disposition")
    require(disposition.get("content_state") == "CONTENT_UNRESOLVED", "content state")
    require(disposition.get("human_review_used") is False, "human review")
    require(disposition.get("api_labels_used_as_ground_truth") is False, "ground truth")
    require(disposition.get("api_evidence_used") is True, "API evidence")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(
        authorization.get("production_disposition")
        == "AUTHORIZED_CONTENT_UNRESOLVED_ONLY",
        "disposition authorization",
    )
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


def _inputs(
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("prior_terminal"), "prior")
    dispositions_path = _bound_file(prior, "dispositions_basename", "dispositions_sha256")
    frontier_path = _bound_file(prior, "frontier_basename", "frontier_sha256")
    certificate_path = _bound_file(prior, "certificate_basename", "certificate_sha256")
    receipt_path = _bound_file(prior, "receipt_basename", "receipt_sha256")
    verification_path = Path("/ssd") / str(prior["verification_basename"])
    require(_sha256(verification_path) == prior["verification_sha256"], "prior verification SHA")
    dispositions = pq.read_table(dispositions_path)
    frontier = pq.read_table(frontier_path)
    require(dispositions.num_rows == prior["disposition_rows"] == 8, "disposition rows")
    require(frontier.num_rows == prior["frontier_rows"] == 51, "frontier rows")
    require(
        _json(certificate_path).get("status")
        == "BLOCKED_REQUIRED_RANK5_CONTENT_UNRESOLVED",
        "prior certificate status",
    )
    result = _mapping(frozen.get("corrected_api_result"), "corrected result")
    candidate_path = _bound_file(result, "candidate_basename", "candidate_sha256")
    decisions_path = _bound_file(result, "decisions_basename", "decisions_sha256")
    result_receipt_path = _bound_file(result, "receipt_basename", "receipt_sha256")
    result_verification_path = Path("/ssd") / str(result["verification_basename"])
    require(_sha256(result_verification_path) == result["verification_sha256"], "result verification SHA")
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        require(len(list(csv.DictReader(handle))) == result["rows"] == 25, "decision rows")
    candidate = _json(candidate_path)
    require(candidate.get("candidate_content_state") == "CONTENT_UNRESOLVED", "candidate state")
    require(_json(result_receipt_path).get("production_disposition_created") is False, "premature disposition")
    return dispositions, frontier, candidate, {
        "prior_dispositions_sha256": _sha256(dispositions_path),
        "prior_frontier_sha256": _sha256(frontier_path),
        "prior_certificate_sha256": _sha256(certificate_path),
        "prior_receipt_sha256": _sha256(receipt_path),
        "prior_verification_sha256": _sha256(verification_path),
        "corrected_candidate_sha256": _sha256(candidate_path),
        "corrected_decisions_sha256": _sha256(decisions_path),
        "corrected_result_receipt_sha256": _sha256(result_receipt_path),
        "corrected_result_verification_sha256": _sha256(result_verification_path),
    }


def _dispositions(
    table: pa.Table,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pa.Table:
    frozen = _mapping(config.get("rank5_production_disposition"), "disposition")
    for key in (
        "local_accept_rows",
        "local_reject_rows",
        "api_dual_high_accept_rows",
        "api_dual_high_reject_rows",
        "augmented_accept_rows",
        "augmented_reject_rows",
        "augmented_unresolved_rows",
    ):
        require(candidate.get(key) == frozen[key], f"candidate binding: {key}")
    rows = table.to_pylist()
    matches = [
        row
        for row in rows
        if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    ]
    require(len(matches) == 1, "rank5 disposition row")
    matches[0].update(
        {
            "content_state": frozen["content_state"],
            "disposition_reason": frozen["disposition_reason"],
            "production_rule_version": frozen["production_rule_version"],
            "source_replay_state": frozen["source_replay_state"],
            "post_diagnostic_amendment_used": frozen[
                "post_diagnostic_amendment_used"
            ],
            "human_review_used": frozen["human_review_used"],
            "api_labels_used_as_ground_truth": frozen[
                "api_labels_used_as_ground_truth"
            ],
        }
    )
    return pa.Table.from_pylist(
        rows,
        schema=table.schema.with_metadata(
            {
                b"schema_version": str(
                    _mapping(config["output_contract"], "output")[
                        "dispositions_schema_version"
                    ]
                ).encode()
            }
        ),
    )


def _frontier(table: pa.Table, config: Mapping[str, Any]) -> tuple[pa.Table, dict[str, Any]]:
    frozen = _mapping(config.get("frontier_v7"), "frontier")
    rows = table.to_pylist()
    rank5 = [
        row
        for row in rows
        if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    ]
    rank6 = [
        row
        for row in rows
        if row["field_id"] == 27 and row["field_candidate_rank"] == 6
    ]
    require(len(rank5) == len(rank6) == 1, "Medicine frontier rows")
    rank5[0].update(
        {
            "frontier_role": frozen["rank5_frontier_role"],
            "content_state": "CONTENT_UNRESOLVED",
            "state_reason": frozen["rank5_state_reason"],
            "targeted_sample_required": False,
        }
    )
    rank6[0].update(
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
    return (
        pa.Table.from_pylist(
            rows,
            schema=table.schema.with_metadata(
                {
                    b"schema_version": str(
                        _mapping(config["output_contract"], "output")[
                            "frontier_schema_version"
                        ]
                    ).encode()
                }
            ),
        ),
        counts,
    )


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    path.chmod(0o600)


def _artifact(path: Path, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def build(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _load_config(config_path)
    contract = _mapping(config.get("output_contract"), "output")
    output = output.absolute()
    require(output.parent == Path("/ssd"), "output parent")
    require(output.name == contract["root_basename"], "output basename")
    require(not output.exists(), "output exists")
    dispositions_in, frontier_in, candidate, input_hashes = _inputs(config)
    dispositions = _dispositions(dispositions_in, candidate, config)
    frontier, frontier_counts = _frontier(frontier_in, config)
    certificate = _mapping(config.get("terminal_certificate_status"), "certificate")
    require(certificate.get("medicine_field_certificate_issued") is False, "Medicine certificate")
    require(certificate.get("global_selection_certificate_issued") is False, "global certificate")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    temporary.chmod(0o700)
    try:
        dispositions_path = temporary / str(contract["dispositions_basename"])
        frontier_path = temporary / str(contract["frontier_basename"])
        pq.write_table(dispositions, dispositions_path, compression="zstd")
        pq.write_table(frontier, frontier_path, compression="zstd")
        dispositions_path.chmod(0o600)
        frontier_path.chmod(0o600)
        certificate_path = temporary / str(contract["certificate_basename"])
        _write_json(certificate_path, certificate)
        report_path = temporary / str(contract["report_basename"])
        report = "\n".join(
            [
                "# Rank-5 correct-HCV API-augmented terminal closeout",
                "",
                f"Status: `{STATUS}`",
                "",
                "- local lower bounds: accept 27/64, reject 12/64",
                "- dual-HIGH API additions: accept 0, reject 9",
                "- augmented lower bounds: accept 27/64, reject 21/64",
                "- unresolved remainder: 16/64",
                "- production state: `CONTENT_UNRESOLVED`",
                "- Medicine/global certificate: blocked",
                "",
                "Human review was not run. The wrong-target HBV API result is",
                "excluded. API labels are not ground truth. Outcomes remain unread.",
                "",
            ]
        )
        with report_path.open("x", encoding="utf-8") as handle:
            handle.write(report)
        report_path.chmod(0o600)
        artifacts = {
            "dispositions": _artifact(dispositions_path, rows=8),
            "frontier": _artifact(frontier_path, rows=51),
            "certificate_status": _artifact(certificate_path),
            "report": _artifact(report_path),
        }
        receipt = {
            "schema_version": "pivot_feedback_d0_rank5_api_augmented_closeout_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "input_hashes": input_hashes,
            "artifacts": artifacts,
            "cumulative_production_dispositions": 8,
            "rank5_content_state": "CONTENT_UNRESOLVED",
            "corrected_api_evidence_used": True,
            "wrong_target_api_evidence_used": False,
            "human_review": "NOT_RUN_BY_DESIGN",
            "api_labels_used_as_ground_truth": False,
            "frontier": frontier_counts,
            "terminal_certificate_status": certificate,
            "computer_science_field_certificate_issued": True,
            "medicine_field_certificate_issued": False,
            "global_selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "new_api_requests_created": 0,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        _write_json(temporary / str(contract["receipt_basename"]), receipt)
        os.replace(temporary, output)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            require(temporary.parent == output.parent, "temporary parent")
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(config_path=args.config, output=args.output), indent=2, sort_keys=True))
        return 0
    except (Rank5ApiAugmentedCloseoutBuildError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_API_AUGMENTED_CLOSEOUT_BUILD", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
