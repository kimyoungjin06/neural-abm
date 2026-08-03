#!/usr/bin/env python3
"""Verify rank-5 external evidence collection and review gate v32."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v32.yaml"
STATUS = "PASS_RANK5_EXACT_14_EXTERNAL_COLLECTION_EXACT_11_REVIEW_GATE"


class GateStateV32VerificationError(RuntimeError):
    """Fail-closed gate v32 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV32VerificationError(message)


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


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _ssd(binding: Mapping[str, Any]) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "SSD root")
    path = root / str(binding["basename"])
    require(_sha256(path) == binding["sha256"], f"SSD SHA: {path.name}")
    return path


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate_path = gate_path.resolve(strict=True)
    gate = _yaml(gate_path)
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v32", "schema")
    require(
        gate.get("status")
        == "VALID_RANK5_EXACT_14_EXTERNAL_EVIDENCE_COLLECTION_AND_EXACT_11_REVIEW_AUTHORIZED",
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    frozen = _mapping(gate.get("frozen_input"), "input")
    recovery = pq.read_table(_ssd(_mapping(frozen.get("local_recovery"), "recovery")))
    require(recovery.num_rows == 16, "recovery rows")
    resolved = {
        "https://openalex.org/W2412228338",
        "https://openalex.org/W2917660375",
    }
    ids = sorted(
        str(row["openalex_id"])
        for row in recovery.to_pylist()
        if str(row["openalex_id"]) not in resolved
    )
    require(len(ids) == frozen["unresolved_rows"] == 14, "unresolved rows")
    digest = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    require(digest == frozen["sorted_openalex_ids_sha256"], "ID digest")
    _ssd(_mapping(frozen.get("local_abstract_decisions"), "decisions"))
    collection = _mapping(gate.get("external_collection_contract"), "collection")
    require(collection.get("exact_rows") == 14, "collection rows")
    require(collection.get("retries") == 0, "collection retries")
    require(collection.get("credentials_required") is False, "credentials")
    review = _mapping(gate.get("review_eligibility_contract"), "review")
    require(review.get("exact_review_rows") == 11, "review rows")
    require(review.get("exact_holdout_rows") == 3, "holdout rows")
    require(review.get("provider_calls") == 2 and review.get("retries") == 0, "provider calls")
    require(review.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(review.get("provider_labels_are_ground_truth") is False, "ground truth")
    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_exact_14_external_metadata_collection")
        == "AUTHORIZED_READ_ONLY",
        "collection authorization",
    )
    require(
        authorization.get("d0_rank5_exact_11_paid_api_submission")
        == "AUTHORIZED_TWO_CALLS_NO_RETRY_AFTER_INDEPENDENT_PACKET_VERIFICATION",
        "API authorization",
    )
    require(str(authorization.get("d0_rank5_production_disposition")).startswith("NOT_AUTHORIZED"), "production")
    require(authorization.get("d0_human_review") == "NOT_RUN_BY_DESIGN", "human")
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v32",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "external_collection_rows_authorized": 14,
        "external_review_rows_authorized": 11,
        "holdout_rows": 3,
        "provider_calls_authorized": 2,
        "retries_authorized": 0,
        "production_disposition_authorized": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "outcomes_read_authorized": False,
        "g4_open": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate(args.gate), indent=2, sort_keys=True))
        return 0
    except (
        GateStateV32VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V32", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
