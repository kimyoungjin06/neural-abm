#!/usr/bin/env python3
"""Verify rank-5 corrected API-result production closeout gate v29."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v29.yaml"
STATUS = "PASS_RANK5_HCV_AUGMENTED_UNRESOLVED_PRODUCTION_CLOSEOUT_ONLY"


class GateStateV29VerificationError(RuntimeError):
    """Fail-closed gate v29 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV29VerificationError(message)


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


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate = _yaml(gate_path.resolve(strict=True))
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v29", "schema")
    require(
        gate.get("status")
        == "VALID_RANK5_HCV_API_AUGMENTED_UNRESOLVED_PRODUCTION_CLOSEOUT_AUTHORIZED",
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    evidence = _mapping(gate.get("corrected_result_evidence"), "evidence")
    for key in ("config", "producer", "verifier"):
        binding = _mapping(evidence.get(key), key)
        path = _repo(binding)
        if "required_schema_version" in binding:
            document = _yaml(path)
            require(document.get("schema_version") == binding["required_schema_version"], f"{key} schema")
            require(document.get("status") == binding["required_status"], f"{key} status")
    local = _mapping(evidence.get("operator_local"), "operator")
    root = (Path("/ssd") / str(local["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "operator root")
    decisions = root / str(local["decisions_basename"])
    candidate = root / str(local["candidate_basename"])
    receipt = root / str(local["receipt_basename"])
    verification = Path("/ssd") / str(local["verification_basename"])
    require(_sha256(decisions) == local["decisions_sha256"], "decisions SHA")
    require(_sha256(candidate) == local["candidate_sha256"], "candidate SHA")
    require(_sha256(receipt) == local["receipt_sha256"], "receipt SHA")
    require(_sha256(verification) == local["verification_sha256"], "verification SHA")
    with decisions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 25, "decision rows")
    require(_json(receipt).get("status") == local["receipt_status"], "receipt status")
    require(_json(verification).get("status") == local["verification_status"], "verification status")
    candidate_doc = _json(candidate)
    result = _mapping(gate.get("verified_corrected_result"), "result")
    required = {
        "local_accept_rows": 27,
        "local_reject_rows": 12,
        "local_unresolved_rows": 25,
        "api_dual_high_accept_rows": 0,
        "api_dual_high_reject_rows": 9,
        "api_remains_unresolved_rows": 16,
        "augmented_accept_rows": 27,
        "augmented_reject_rows": 21,
        "augmented_unresolved_rows": 16,
    }
    for key, expected in required.items():
        require(candidate_doc.get(key) == result.get(key) == expected, f"result: {key}")
    require(candidate_doc.get("candidate_content_state") == result["augmented_content_state"] == "CONTENT_UNRESOLVED", "content state")
    require(result.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(result.get("api_labels_are_ground_truth") is False, "ground truth")
    successor = _mapping(gate.get("authorized_successor"), "successor")
    require(successor.get("adaptive_frontier_version") == "v7", "frontier version")
    require(
        successor.get("terminal_certificate_state")
        == "BLOCKED_REQUIRED_RANK5_AUGMENTED_CONTENT_UNRESOLVED",
        "terminal state",
    )
    require(successor.get("medicine_field_certificate_issued") is False, "Medicine certificate")
    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_augmented_production_disposition")
        == "AUTHORIZED_CONTENT_UNRESOLVED_ONLY",
        "disposition authorization",
    )
    for key in (
        "d0_new_api_request_creation",
        "d0_paid_api_submission",
        "d0_new_candidate_evaluation",
        "d0_selection_certificate",
        "d0_deterministic_reselection",
        "d0_author_opportunity_panel_rebuild",
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(str(authorization.get(key)).startswith("NOT_AUTHORIZED"), f"unsafe authorization: {key}")
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v29",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "corrected_api_result_verified": True,
        "augmented_accept_rows": 27,
        "augmented_reject_rows": 21,
        "augmented_unresolved_rows": 16,
        "rank5_content_state_authorized": "CONTENT_UNRESOLVED",
        "frontier_v7_authorized": True,
        "selection_certificate_authorized": False,
        "new_api_requests_authorized": False,
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
    except (GateStateV29VerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V29", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
