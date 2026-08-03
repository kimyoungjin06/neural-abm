#!/usr/bin/env python3
"""Verify additive pivot-feedback gate-state v24."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v24.yaml"
STATUS = "PASS_EXACT_RANK5_SAMPLE_VERIFIED_LOCAL_ROUTING_ONLY_OUTCOMES_FAIL_CLOSED"


class GateStateV24VerificationError(RuntimeError):
    """Fail-closed v24 gate verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV24VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate_path = gate_path.resolve(strict=True)
    gate = _read_yaml(gate_path)
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v24", "schema")
    require(
        gate.get("status")
        == (
            "VALID_EXACT_T10151_RANK5_SAMPLE_VERIFIED_ZERO_COST_LOCAL_"
            "ROUTING_AUTHORIZED_OUTCOMES_FAIL_CLOSED"
        ),
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    require(
        _read_yaml(_repo_file(parent)).get("status") == parent["required_status"],
        "parent status",
    )
    evidence = _mapping(gate.get("rank5_sample_evidence"), "evidence")
    for key in ("config", "producer", "verifier"):
        _repo_file(_mapping(evidence.get(key), key))
    local = _mapping(evidence.get("operator_local"), "operator")
    root = (Path("/ssd") / str(local["root_basename"])).resolve(strict=True)
    sample = root / str(local["sample_basename"])
    receipt = root / str(local["receipt_basename"])
    verification = Path("/ssd") / str(local["verification_basename"])
    require(_sha256(sample) == local["sample_sha256"], "sample SHA")
    require(_sha256(receipt) == local["receipt_sha256"], "receipt SHA")
    require(_sha256(verification) == local["verification_sha256"], "verification SHA")
    require(pq.ParquetFile(sample).metadata.num_rows == local["sample_rows"], "sample rows")
    require(_read_json(receipt).get("status") == local["receipt_status"], "receipt status")
    require(
        _read_json(verification).get("status") == local["verification_status"],
        "verification status",
    )
    observed = _mapping(gate.get("verified_sample"), "sample result")
    require(
        (
            observed["field_candidate_rank"],
            observed["topic_id"],
            observed["entry_year"],
            observed["eligible_prelink_rows"],
            observed["author_linked_pool_rows"],
            observed["missing_title_rows"],
        )
        == (5, 10151, 1993, 4549, 3797, 0),
        "sample result",
    )
    successor = _mapping(gate.get("routing_successor"), "successor")
    _repo_file(_mapping(successor.get("topic_card_source"), "card source"))
    _repo_file(_mapping(successor.get("production_rule_v2_source"), "rule source"))
    require(successor.get("sample_rows_to_read") == 96, "routing rows")
    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_local_routing") == "AUTHORIZED_ZERO_COST_ONLY",
        "routing authorization",
    )
    require(
        authorization.get("d0_rank5_candidate_rule_v2_dry_run") == "AUTHORIZED",
        "dry-run authorization",
    )
    for key in (
        "d0_rank5_production_content_disposition",
        "d0_raw_openalex_title_scan",
        "d0_new_title_extraction",
        "d0_new_api_request_creation",
        "d0_paid_api_exception_submission",
        "d0_selection_certificate",
        "d0_deterministic_reselection",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"unsafe authorization: {key}",
        )
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v24",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "rank5_sample_rows_verified": 96,
        "rank5_local_routing_authorized": True,
        "rank5_production_disposition_authorized": False,
        "new_api_request_creation_authorized": False,
        "outcomes_read_authorized": False,
        "g4_open": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(validate(args.gate), indent=2, sort_keys=True))
        return 0
    except (
        GateStateV24VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V24", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
