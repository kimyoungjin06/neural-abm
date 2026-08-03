#!/usr/bin/env python3
"""Independently verify the rank-5 external-evidence terminal closeout."""

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
    / "experiments/configs/pivot_feedback_d0_rank5_external_evidence_closeout_v1.yaml"
)
DEFAULT_ROOT = Path(
    "/ssd/pivot-feedback-d0-rank5-external-evidence-closeout-v1-20260730"
)
DEFAULT_VERIFICATION = Path(
    "/ssd/pivot-feedback-d0-rank5-external-evidence-closeout-v1-20260730.verification-v1.json"
)
STATUS = "PASS_INDEPENDENT_RANK5_EXTERNAL_EVIDENCE_UNRESOLVED_TERMINAL_CLOSEOUT"


class Rank5ExternalCloseoutVerificationError(RuntimeError):
    """Fail-closed external closeout verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5ExternalCloseoutVerificationError(message)


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


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def verify(
    *,
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _yaml(config_path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_rank5_external_evidence_closeout_config.v1",
        "config schema",
    )
    output = _mapping(config.get("output_contract"), "output")
    root = output_root.resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "output parent")
    require(root.name == output["root_basename"], "output name")
    require(root.stat().st_mode & 0o077 == 0, "root mode")
    receipt_path = root / str(output["receipt_basename"])
    receipt = _json(receipt_path)
    require(
        receipt.get("status")
        == "PASS_RANK5_EXTERNAL_EVIDENCE_UNRESOLVED_TERMINAL_CLOSEOUT_FROZEN",
        "receipt status",
    )
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    require(receipt.get("rank5_counts") == {"accept": 30, "reject": 25, "unresolved": 9}, "counts")
    require(receipt.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(receipt.get("provider_labels_used_as_ground_truth") is False, "ground truth")
    require(receipt.get("medicine_field_certificate_issued") is False, "Medicine certificate")
    require(receipt.get("global_selection_certificate_issued") is False, "global certificate")
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
    disposition = [
        row
        for row in dispositions.to_pylist()
        if row["field_id"] == 27 and row["field_candidate_rank"] == 5
    ]
    require(len(disposition) == 1, "rank5 disposition")
    frozen = _mapping(config.get("rank5_production_disposition"), "frozen disposition")
    for key in (
        "content_state",
        "disposition_reason",
        "production_rule_version",
        "source_replay_state",
        "post_diagnostic_amendment_used",
        "human_review_used",
        "api_labels_used_as_ground_truth",
    ):
        require(disposition[0][key] == frozen[key], f"disposition: {key}")

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
    expected_frontier = _mapping(config.get("frontier_v9"), "frontier")
    for key, value in counts.items():
        require(value == expected_frontier[key], f"frontier count: {key}")
    rank5 = next(row for row in rows if row["field_id"] == 27 and row["field_candidate_rank"] == 5)
    rank6 = next(row for row in rows if row["field_id"] == 27 and row["field_candidate_rank"] == 6)
    require(rank5["frontier_role"] == expected_frontier["rank5_frontier_role"], "rank5 role")
    require(rank5["state_reason"] == expected_frontier["rank5_state_reason"], "rank5 reason")
    require(rank5["content_state"] == "CONTENT_UNRESOLVED", "rank5 state")
    require(rank6["frontier_role"] == expected_frontier["rank6_frontier_role"], "rank6 role")
    require(rank6["content_state"] == "CONTENT_PASS", "rank6 state")

    certificate_path = root / str(output["certificate_basename"])
    certificate = _json(certificate_path)
    require(certificate == config["terminal_certificate_status"], "certificate")
    require(
        certificate.get("status")
        == "BLOCKED_REQUIRED_RANK5_EXTERNAL_EVIDENCE_AUGMENTED_CONTENT_UNRESOLVED",
        "certificate status",
    )
    require(certificate.get("medicine_blocking_counts") == {"accept": 30, "reject": 25, "unresolved": 9}, "blocking counts")
    require(certificate.get("computer_science_selected_candidate_ranks") == [1, 45], "CS ranks")
    require(certificate.get("medicine_field_certificate_issued") is False, "Medicine blocked")
    require(certificate.get("global_selection_certificate_issued") is False, "global blocked")
    require(receipt.get("api_cost") == config["api_cost"], "API cost")
    return {
        "schema_version": "pivot_feedback_d0_rank5_external_evidence_closeout_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(config_path),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(receipt_path),
        "dispositions_sha256": _sha256(dispositions_path),
        "frontier_sha256": _sha256(frontier_path),
        "certificate_status_sha256": _sha256(certificate_path),
        "rank5_counts": {"accept": 30, "reject": 25, "unresolved": 9},
        "rank5_content_state": "CONTENT_UNRESOLVED",
        "frontier_counts": counts,
        "computer_science_selected_candidate_ranks": [1, 45],
        "medicine_field_certificate_issued": False,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "provider_labels_used_as_ground_truth": False,
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
        result = verify(config_path=args.config, output_root=args.output_root)
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
        Rank5ExternalCloseoutVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_RANK5_EXTERNAL_CLOSEOUT_VERIFICATION", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
