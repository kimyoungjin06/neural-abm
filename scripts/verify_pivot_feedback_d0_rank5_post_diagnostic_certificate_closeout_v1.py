#!/usr/bin/env python3
"""Independently verify the rank-5 pass and global certificate closeout."""

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
    / "experiments/configs/pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_v1.yaml"
)
DEFAULT_ROOT = Path(
    "/ssd/pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-rank5-post-diagnostic-certificate-closeout-v1-20260730.verification-v1.json"
)
STATUS = "PASS_INDEPENDENT_RANK5_POST_DIAGNOSTIC_CONTENT_PASS_GLOBAL_CERTIFICATES"


class PostDiagnosticCertificateVerificationError(RuntimeError):
    """Fail-closed certificate verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PostDiagnosticCertificateVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _compatible(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        first["subfield_id"] != second["subfield_id"]
        and abs(int(first["entry_year"]) - int(second["entry_year"])) >= 9
    )


def _independent_pair(rows: Sequence[Mapping[str, Any]], field_id: int) -> list[int]:
    field_rows = sorted(
        (row for row in rows if row["field_id"] == field_id),
        key=lambda row: row["field_candidate_rank"],
    )
    first: Mapping[str, Any] | None = None
    for row in field_rows:
        if row["content_state"] == "CONTENT_FAIL":
            continue
        require(row["content_state"] == "CONTENT_PASS", f"blocked first: field {field_id}")
        first = row
        break
    require(first is not None, f"missing first: field {field_id}")
    for row in field_rows:
        if row["field_candidate_rank"] <= first["field_candidate_rank"]:
            continue
        if not _compatible(first, row):
            continue
        if row["content_state"] == "CONTENT_FAIL":
            continue
        require(row["content_state"] == "CONTENT_PASS", f"blocked second: field {field_id}")
        return [int(first["field_candidate_rank"]), int(row["field_candidate_rank"])]
    raise PostDiagnosticCertificateVerificationError(f"missing second: field {field_id}")


def verify(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), "config")
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_config.v1",
        "config schema",
    )
    output = _mapping(config.get("output_contract"), "output")
    root = output_root.resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "root parent")
    require(root.name == output["root_basename"], "root name")
    require(root.stat().st_mode & 0o077 == 0, "root mode")
    receipt_path = root / str(output["receipt_basename"])
    receipt = _json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_RANK5_POST_DIAGNOSTIC_CONTENT_PASS_GLOBAL_CERTIFICATES_FROZEN",
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    artifacts = _mapping(receipt.get("artifacts"), "artifacts")
    for label, artifact_value in artifacts.items():
        artifact = _mapping(artifact_value, label)
        path = root / str(artifact["basename"])
        require(path.is_file() and not path.is_symlink(), f"artifact: {label}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {label}")
        require(path.stat().st_mode & 0o077 == 0, f"artifact mode: {label}")

    dispositions_path = root / str(output["dispositions_basename"])
    dispositions = pq.read_table(dispositions_path)
    require(dispositions.num_rows == 8, "disposition rows")
    require(
        (dispositions.schema.metadata or {}).get(b"schema_version", b"").decode()
        == output["dispositions_schema_version"],
        "disposition schema",
    )
    disposition_rows = dispositions.to_pylist()
    rank5_dispositions = [
        row
        for row in disposition_rows
        if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    ]
    require(len(rank5_dispositions) == 1, "rank5 disposition identity")
    rank5_disposition = rank5_dispositions[0]
    frozen_disposition = _mapping(config.get("rank5_production_disposition"), "disposition")
    for key in (
        "content_state",
        "disposition_reason",
        "production_rule_version",
        "source_replay_state",
        "post_diagnostic_amendment_used",
        "human_review_used",
        "api_labels_used_as_ground_truth",
    ):
        require(rank5_disposition[key] == frozen_disposition[key], f"disposition: {key}")
    require(rank5_disposition["content_state"] == "CONTENT_PASS", "rank5 state")

    frontier_path = root / str(output["frontier_basename"])
    frontier = pq.read_table(frontier_path)
    require(frontier.num_rows == 51, "frontier rows")
    require(
        (frontier.schema.metadata or {}).get(b"schema_version", b"").decode()
        == output["frontier_schema_version"],
        "frontier schema",
    )
    rows = frontier.to_pylist()
    counts = {
        "total_frontier_rows": len(rows),
        "total_content_pass_rows": sum(row["content_state"] == "CONTENT_PASS" for row in rows),
        "total_content_fail_rows": sum(row["content_state"] == "CONTENT_FAIL" for row in rows),
        "total_content_unresolved_rows": sum(row["content_state"] == "CONTENT_UNRESOLVED" for row in rows),
        "total_not_evaluated_rows": sum(row["content_state"] == "NOT_EVALUATED" for row in rows),
    }
    expected_frontier = _mapping(config.get("frontier_v10"), "frontier")
    for key, value in counts.items():
        require(value == expected_frontier[key], f"frontier count: {key}")
    rank5 = next(row for row in rows if row["field_id"] == 27 and row["field_candidate_rank"] == 5)
    rank6 = next(row for row in rows if row["field_id"] == 27 and row["field_candidate_rank"] == 6)
    require(rank5["content_state"] == "CONTENT_PASS", "rank5 frontier state")
    require(rank5["frontier_role"] == expected_frontier["rank5_frontier_role"], "rank5 role")
    require(rank5["state_reason"] == expected_frontier["rank5_state_reason"], "rank5 reason")
    require(rank6["content_state"] == "CONTENT_PASS", "rank6 state")
    require(rank6["frontier_role"] == expected_frontier["rank6_frontier_role"], "rank6 role")
    pairs = {
        "computer_science": _independent_pair(rows, 17),
        "medicine": _independent_pair(rows, 27),
    }
    require(pairs == {"computer_science": [1, 45], "medicine": [5, 6]}, "selected pairs")
    require(receipt.get("selected_candidate_ranks") == pairs, "receipt pairs")

    certificate_path = root / str(output["certificate_basename"])
    certificate = _json(certificate_path)
    require(certificate == config["terminal_certificate_status"], "certificate binding")
    require(certificate.get("status") == "PASS_GLOBAL_SELECTION_CERTIFICATE_ISSUED", "certificate status")
    require(certificate.get("computer_science_selected_candidate_ranks") == pairs["computer_science"], "CS ranks")
    require(certificate.get("medicine_selected_candidate_ranks") == pairs["medicine"], "Medicine ranks")
    require(certificate.get("field_certificates_issued") == 2, "field certificates")
    require(certificate.get("global_selection_certificate_issued") is True, "global certificate")
    require(certificate.get("deterministic_reselection_executed") is False, "reselection")
    require(certificate.get("deterministic_reselection_successor_authorized") is False, "reselection successor")
    require(certificate.get("retrospective_post_diagnostic") is True, "retrospective")
    require(certificate.get("out_of_sample_validation") is False, "out of sample")
    require(certificate.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(certificate.get("provider_labels_are_ground_truth") is False, "provider ground truth")
    require(certificate.get("pubmed_indexing_is_ground_truth") is False, "PubMed ground truth")
    require(receipt.get("rank5_counts") == {"accept": 32, "reject": 25, "unresolved": 7}, "rank5 counts")
    require(receipt.get("rank5_content_state") == "CONTENT_PASS", "receipt rank5 state")
    require(receipt.get("global_selection_certificate_issued") is True, "receipt global certificate")
    require(receipt.get("deterministic_reselection_executed") is False, "receipt reselection")
    require(receipt.get("deterministic_reselection_successor_authorized") is False, "receipt reselection successor")
    for key in ("outcomes_read", "g4_open"):
        require(receipt.get(key) is False, f"closed boundary: {key}")
    return {
        "schema_version": "pivot_feedback_d0_rank5_post_diagnostic_certificate_closeout_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "receipt_sha256": _sha256(receipt_path),
        "dispositions_sha256": _sha256(dispositions_path),
        "frontier_sha256": _sha256(frontier_path),
        "certificate_status_sha256": _sha256(certificate_path),
        "rank5_counts": {"accept": 32, "reject": 25, "unresolved": 7},
        "rank5_content_state": "CONTENT_PASS",
        "frontier_counts": counts,
        "selected_candidate_ranks": pairs,
        "field_certificates_issued": 2,
        "global_selection_certificate_issued": True,
        "deterministic_reselection_executed": False,
        "deterministic_reselection_successor_authorized": False,
        "retrospective_post_diagnostic": True,
        "out_of_sample_validation": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "outcomes_read": False,
        "g4_open": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--verification-output", type=Path, default=DEFAULT_VERIFICATION)
    args = parser.parse_args(argv)
    try:
        result = verify(args.config, args.output_root)
        output = args.verification_output.absolute()
        require(output.parent == Path("/ssd"), "verification parent")
        require(not output.exists(), "verification exists")
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.chmod(0o600)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        PostDiagnosticCertificateVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_POST_DIAGNOSTIC_CERTIFICATE_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
