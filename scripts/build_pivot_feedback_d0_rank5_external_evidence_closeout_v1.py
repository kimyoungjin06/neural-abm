#!/usr/bin/env python3
"""Materialize the external-evidence-augmented rank-5 terminal closeout."""

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
    / "experiments/configs/pivot_feedback_d0_rank5_external_evidence_closeout_v1.yaml"
)
DEFAULT_OUTPUT = Path(
    "/ssd/pivot-feedback-d0-rank5-external-evidence-closeout-v1-20260730"
)
STATUS = "PASS_RANK5_EXTERNAL_EVIDENCE_UNRESOLVED_TERMINAL_CLOSEOUT_FROZEN"


class Rank5ExternalCloseoutError(RuntimeError):
    """Fail-closed external-evidence closeout error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5ExternalCloseoutError(message)


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


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _bound(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "input root")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), f"input: {path.name}")
    require(_sha256(path) == binding[sha_key], f"input SHA: {path.name}")
    return path


def load_config(path: Path) -> dict[str, Any]:
    config = _yaml(path.resolve(strict=True))
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_rank5_external_evidence_closeout_config.v1",
        "config schema",
    )
    require(
        config.get("status")
        == "FROZEN_RANK5_EXTERNAL_EVIDENCE_UNRESOLVED_CLOSEOUT_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    disposition = _mapping(config.get("rank5_production_disposition"), "disposition")
    require(disposition.get("content_state") == "CONTENT_UNRESOLVED", "content state")
    require(disposition.get("human_review_used") is False, "human review")
    require(disposition.get("api_labels_used_as_ground_truth") is False, "ground truth")
    require(
        disposition["final_accept_rows"]
        + disposition["final_reject_rows"]
        + disposition["final_unresolved_rows"]
        == disposition["recent_denominator_rows"]
        == 64,
        "denominator",
    )
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("production_disposition") == "AUTHORIZED_CONTENT_UNRESOLVED_ONLY", "production")
    for key in (
        "new_api_request",
        "selection_certificate",
        "deterministic_reselection",
        "production_pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def load_inputs(
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any], dict[str, str]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("prior_closeout"), "prior")
    dispositions_path = _bound(prior, "dispositions_basename", "dispositions_sha256")
    frontier_path = _bound(prior, "frontier_basename", "frontier_sha256")
    certificate_path = _bound(prior, "certificate_basename", "certificate_sha256")
    receipt_path = _bound(prior, "receipt_basename", "receipt_sha256")
    verification_path = Path("/ssd") / str(prior["verification_basename"])
    require(_sha256(verification_path) == prior["verification_sha256"], "prior verification SHA")
    require(_json(receipt_path).get("status") == prior["receipt_status"], "prior receipt status")
    require(_json(verification_path).get("status") == prior["verification_status"], "prior verification status")
    dispositions = pq.read_table(dispositions_path)
    frontier = pq.read_table(frontier_path)
    require(dispositions.num_rows == prior["disposition_rows"] == 8, "disposition rows")
    require(frontier.num_rows == prior["frontier_rows"] == 51, "frontier rows")
    require(
        _json(certificate_path).get("status")
        == "BLOCKED_REQUIRED_RANK5_LOCAL_ABSTRACT_AUGMENTED_CONTENT_UNRESOLVED",
        "prior certificate",
    )
    result = _mapping(frozen.get("external_result"), "external result")
    decisions_path = _bound(result, "decisions_basename", "decisions_sha256")
    candidate_path = _bound(result, "candidate_basename", "candidate_sha256")
    result_receipt_path = _bound(result, "receipt_basename", "receipt_sha256")
    result_verification_path = Path("/ssd") / str(result["verification_basename"])
    require(_sha256(result_verification_path) == result["verification_sha256"], "result verification SHA")
    require(_json(result_receipt_path).get("status") == result["receipt_status"], "result receipt")
    require(_json(result_verification_path).get("status") == result["verification_status"], "result verification")
    with decisions_path.open(newline="", encoding="utf-8") as handle:
        require(len(list(csv.DictReader(handle))) == result["decision_rows"] == 11, "decision rows")
    candidate = _json(candidate_path)
    require(candidate.get("candidate_content_state") == "CONTENT_UNRESOLVED", "candidate state")
    require(candidate.get("production_disposition_created") is False, "premature production")
    hashes = {
        "prior_dispositions_sha256": _sha256(dispositions_path),
        "prior_frontier_sha256": _sha256(frontier_path),
        "prior_certificate_sha256": _sha256(certificate_path),
        "prior_receipt_sha256": _sha256(receipt_path),
        "prior_verification_sha256": _sha256(verification_path),
        "external_decisions_sha256": _sha256(decisions_path),
        "external_candidate_sha256": _sha256(candidate_path),
        "external_receipt_sha256": _sha256(result_receipt_path),
        "external_verification_sha256": _sha256(result_verification_path),
    }
    return dispositions, frontier, candidate, hashes


def update_tables(
    dispositions: pa.Table,
    frontier: pa.Table,
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[str, Any]]:
    frozen = _mapping(config.get("rank5_production_disposition"), "disposition")
    for key in (
        "previous_accept_rows",
        "previous_reject_rows",
        "previous_unresolved_rows",
        "external_dual_high_accept_rows",
        "external_dual_high_reject_rows",
        "external_reviewed_unresolved_rows",
        "external_holdout_rows",
        "final_accept_rows",
        "final_reject_rows",
        "final_unresolved_rows",
    ):
        require(candidate.get(key) == frozen[key], f"candidate binding: {key}")
    disposition_rows = dispositions.to_pylist()
    rank5 = [
        row
        for row in disposition_rows
        if row["field_id"] == frozen["field_id"]
        and row["field_candidate_rank"] == frozen["field_candidate_rank"]
    ]
    require(len(rank5) == 1, "rank5 disposition")
    rank5[0].update(
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
    output = _mapping(config.get("output_contract"), "output")
    dispositions_out = pa.Table.from_pylist(
        disposition_rows,
        schema=dispositions.schema.with_metadata(
            {b"schema_version": str(output["dispositions_schema_version"]).encode()}
        ),
    )
    frontier_frozen = _mapping(config.get("frontier_v9"), "frontier")
    frontier_rows = frontier.to_pylist()
    rank5_frontier = next(
        row for row in frontier_rows if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    )
    rank6_frontier = next(
        row for row in frontier_rows if row["field_id"] == 27 and row["field_candidate_rank"] == 6
    )
    rank5_frontier.update(
        {
            "frontier_role": frontier_frozen["rank5_frontier_role"],
            "content_state": "CONTENT_UNRESOLVED",
            "state_reason": frontier_frozen["rank5_state_reason"],
            "targeted_sample_required": False,
        }
    )
    rank6_frontier.update(
        {
            "frontier_role": frontier_frozen["rank6_frontier_role"],
            "targeted_sample_required": False,
        }
    )
    counts = {
        "total_frontier_rows": len(frontier_rows),
        "total_content_pass_rows": sum(row["content_state"] == "CONTENT_PASS" for row in frontier_rows),
        "total_content_fail_rows": sum(row["content_state"] == "CONTENT_FAIL" for row in frontier_rows),
        "total_content_unresolved_rows": sum(row["content_state"] == "CONTENT_UNRESOLVED" for row in frontier_rows),
        "total_not_evaluated_rows": sum(row["content_state"] == "NOT_EVALUATED" for row in frontier_rows),
        "field_certificates_issued": 1,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
    }
    for key, value in counts.items():
        require(value == frontier_frozen[key], f"frontier count: {key}")
    frontier_out = pa.Table.from_pylist(
        frontier_rows,
        schema=frontier.schema.with_metadata(
            {b"schema_version": str(output["frontier_schema_version"]).encode()}
        ),
    )
    return dispositions_out, frontier_out, counts


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
    output_root: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    output = _mapping(config.get("output_contract"), "output")
    output_root = output_root.absolute()
    require(output_root.parent == Path("/ssd"), "output parent")
    require(output_root.name == output["root_basename"], "output name")
    require(not output_root.exists(), "output exists")
    dispositions_in, frontier_in, candidate, hashes = load_inputs(config)
    dispositions, frontier, frontier_counts = update_tables(
        dispositions_in, frontier_in, candidate, config
    )
    certificate = _mapping(config.get("terminal_certificate_status"), "certificate")
    require(certificate.get("medicine_field_certificate_issued") is False, "Medicine certificate")
    require(certificate.get("global_selection_certificate_issued") is False, "global certificate")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir="/ssd"))
    temporary.chmod(0o700)
    try:
        dispositions_path = temporary / str(output["dispositions_basename"])
        frontier_path = temporary / str(output["frontier_basename"])
        pq.write_table(dispositions, dispositions_path, compression="zstd")
        pq.write_table(frontier, frontier_path, compression="zstd")
        dispositions_path.chmod(0o600)
        frontier_path.chmod(0o600)
        certificate_path = temporary / str(output["certificate_basename"])
        _write_json(certificate_path, certificate)
        report_path = temporary / str(output["report_basename"])
        report_path.write_text(
            "\n".join(
                [
                    "# Rank-5 external-evidence augmented terminal closeout",
                    "",
                    f"Status: `{STATUS}`",
                    "",
                    "- prior local-abstract counts: 28 accept / 22 reject / 14 unresolved",
                    "- external dual-HIGH additions: 2 accept / 3 reject",
                    "- final counts: 30 accept / 25 reject / 9 unresolved",
                    "- production state: `CONTENT_UNRESOLVED`",
                    "- Medicine/global certificate: blocked",
                    "- human review: `NOT_RUN_BY_DESIGN`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        report_path.chmod(0o600)
        artifacts = {
            "dispositions": _artifact(dispositions_path, rows=8),
            "frontier": _artifact(frontier_path, rows=51),
            "certificate_status": _artifact(certificate_path),
            "report": _artifact(report_path),
        }
        receipt = {
            "schema_version": "pivot_feedback_d0_rank5_external_evidence_closeout_outputs.v1",
            "status": STATUS,
            "config_sha256": _sha256(config_path),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "input_hashes": hashes,
            "artifacts": artifacts,
            "cumulative_production_dispositions": 8,
            "rank5_content_state": "CONTENT_UNRESOLVED",
            "rank5_counts": {"accept": 30, "reject": 25, "unresolved": 9},
            "external_evidence_used": True,
            "human_review": "NOT_RUN_BY_DESIGN",
            "provider_labels_used_as_ground_truth": False,
            "api_cost": config["api_cost"],
            "frontier": frontier_counts,
            "terminal_certificate_status": certificate,
            "medicine_field_certificate_issued": False,
            "global_selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "new_api_requests_created": 0,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        _write_json(temporary / str(output["receipt_basename"]), receipt)
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            require(temporary.parent == Path("/ssd"), "temporary parent")
            shutil.rmtree(temporary)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(build(config_path=args.config, output_root=args.output), indent=2, sort_keys=True))
        return 0
    except (
        Rank5ExternalCloseoutError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_EXTERNAL_CLOSEOUT", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
