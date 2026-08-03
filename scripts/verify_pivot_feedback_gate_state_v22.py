#!/usr/bin/env python3
"""Verify additive pivot-feedback gate-state v22."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v22.yaml"
STATUS = "PASS_RANK4_CONTENT_FAIL_FRONTIER_V5_ONLY_OUTCOMES_FAIL_CLOSED"


class GateStateV22VerificationError(RuntimeError):
    """Fail-closed v22 gate verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV22VerificationError(message)


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
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v22", "schema")
    require(
        gate.get("status")
        == (
            "VALID_MEDICINE_RANK4_INDEXING_FAIL_VERIFIED_PRODUCTION_"
            "DISPOSITION_AND_FRONTIER_V5_AUTHORIZED_OUTCOMES_FAIL_CLOSED"
        ),
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    require(
        _read_yaml(_repo_file(parent)).get("status") == parent["required_status"],
        "parent status",
    )
    evidence = _mapping(gate.get("rank4_local_routing_evidence"), "evidence")
    for key in ("config", "producer", "verifier", "summary_markdown"):
        _repo_file(_mapping(evidence.get(key), key))
    summary_binding = _mapping(evidence.get("summary_json"), "summary")
    summary = _read_json(_repo_file(summary_binding))
    require(
        summary.get("schema_version") == summary_binding["required_schema_version"],
        "summary schema",
    )
    require(summary.get("status") == summary_binding["required_status"], "summary status")
    local = _mapping(evidence.get("operator_local"), "operator")
    root = (Path("/ssd") / str(local["root_basename"])).resolve(strict=True)
    receipt = root / str(local["receipt_basename"])
    verification = Path("/ssd") / str(local["verification_basename"])
    require(_sha256(receipt) == local["receipt_sha256"], "receipt SHA")
    require(_sha256(verification) == local["verification_sha256"], "verification SHA")
    require(_read_json(receipt).get("status") == local["receipt_status"], "receipt status")
    require(
        _read_json(verification).get("status") == local["verification_status"],
        "verification status",
    )
    result = _mapping(gate.get("verified_rank4_result"), "rank4 result")
    require(
        (
            result["field_candidate_rank"],
            result["topic_id"],
            result["entry_year"],
            result["candidate_rule_v2_state"],
            result["production_content_state_to_freeze"],
        )
        == (4, 12505, 1993, "WOULD_CONTENT_FAIL_INDEXING_ANOMALY", "CONTENT_FAIL"),
        "rank4 result",
    )
    successor = _mapping(gate.get("authorized_successor"), "successor")
    target = _mapping(
        successor.get("expected_next_required_candidate_if_rank4_fails"),
        "next target",
    )
    require(
        (
            target["field_id"],
            target["field_candidate_rank"],
            target["topic_id"],
            target["entry_year"],
        )
        == (27, 5, 10151, 1993),
        "next target",
    )
    authorization = _mapping(gate.get("authorization"), "authorization")
    for key in (
        "d0_rank4_production_content_disposition",
        "d0_cumulative_seven_production_dispositions",
        "d0_adaptive_frontier_v5_regeneration",
    ):
        require(authorization.get(key) == "AUTHORIZED", f"authorized work: {key}")
    require(
        authorization.get("d0_selection_certificate_from_frontier_v5")
        == "AUTHORIZED_ONLY_IF_LAZY_STATE_MACHINE_CLOSES",
        "certificate condition",
    )
    for key in (
        "d0_new_title_extraction",
        "d0_raw_openalex_title_scan",
        "d0_new_api_request_creation",
        "d0_paid_api_exception_submission",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"unsafe authorization: {key}",
        )
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v22",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "rank4_production_content_fail_authorized": True,
        "cumulative_seven_dispositions_authorized": True,
        "adaptive_frontier_v5_authorized": True,
        "rank5_topic_id": 10151,
        "rank5_entry_year": 1993,
        "selection_certificate_conditioned_on_lazy_closure": True,
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
        GateStateV22VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V22", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
