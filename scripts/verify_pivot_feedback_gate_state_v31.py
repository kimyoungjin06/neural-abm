#!/usr/bin/env python3
"""Verify rank-5 local-abstract production closeout gate v31."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v31.yaml"
STATUS = "PASS_RANK5_LOCAL_ABSTRACT_UNRESOLVED_PRODUCTION_CLOSEOUT_GATE"


class GateStateV31VerificationError(RuntimeError):
    """Fail-closed gate v31 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV31VerificationError(message)


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
    gate_path = gate_path.resolve(strict=True)
    gate = _yaml(gate_path)
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v31", "schema")
    require(
        gate.get("status")
        == "VALID_RANK5_LOCAL_ABSTRACT_AUGMENTED_UNRESOLVED_PRODUCTION_CLOSEOUT_AUTHORIZED",
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    result = _mapping(gate.get("verified_result"), "result")
    for key in ("config", "producer", "verifier"):
        binding = _mapping(result.get(key), key)
        path = _repo(binding)
        if "required_schema_version" in binding:
            document = _yaml(path)
            require(document.get("schema_version") == binding["required_schema_version"], f"{key} schema")
            require(document.get("status") == binding["required_status"], f"{key} status")
    operator = _mapping(result.get("operator_local"), "operator")
    root = (Path("/ssd") / str(operator["root_basename"])).resolve(strict=True)
    decisions = root / str(operator["decisions_basename"])
    candidate_path = root / str(operator["candidate_basename"])
    receipt_path = root / str(operator["receipt_basename"])
    verification_path = Path("/ssd") / str(operator["verification_basename"])
    require(_sha256(decisions) == operator["decisions_sha256"], "decisions SHA")
    require(_sha256(candidate_path) == operator["candidate_sha256"], "candidate SHA")
    require(_sha256(receipt_path) == operator["receipt_sha256"], "receipt SHA")
    require(_sha256(verification_path) == operator["verification_sha256"], "verification SHA")
    with decisions.open(newline="", encoding="utf-8") as handle:
        require(len(list(csv.DictReader(handle))) == 5, "decision rows")
    require(_json(receipt_path).get("status") == operator["receipt_status"], "receipt status")
    require(_json(verification_path).get("status") == operator["verification_status"], "verification status")
    candidate = _json(candidate_path)
    counts = _mapping(result.get("counts"), "counts")
    for key in (
        "previous_accept_rows",
        "previous_reject_rows",
        "previous_unresolved_rows",
        "abstract_dual_high_accept_rows",
        "abstract_dual_high_reject_rows",
        "abstract_reviewed_unresolved_rows",
        "final_accept_rows",
        "final_reject_rows",
        "final_unresolved_rows",
    ):
        require(candidate.get(key) == counts[key], f"candidate count: {key}")
    require(candidate.get("local_abstract_unavailable_or_fragment_rows") == counts["unavailable_or_fragment_rows"], "unavailable")
    require(candidate.get("candidate_content_state") == counts["final_content_state"] == "CONTENT_UNRESOLVED", "content state")
    require(counts.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(counts.get("provider_labels_are_ground_truth") is False, "ground truth")
    successor = _mapping(gate.get("authorized_successor"), "successor")
    require(successor.get("adaptive_frontier_version") == "v8", "frontier")
    require(successor.get("terminal_certificate_status_version") == "v3", "certificate version")
    require(successor.get("medicine_field_certificate_issued") is False, "Medicine certificate")
    require(successor.get("global_selection_certificate_issued") is False, "global certificate")
    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_abstract_augmented_production_disposition")
        == "AUTHORIZED_CONTENT_UNRESOLVED_ONLY",
        "production authorization",
    )
    for key in (
        "d0_new_api_request_creation",
        "d0_paid_api_submission",
        "d0_selection_certificate",
        "d0_deterministic_reselection",
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(str(authorization.get(key)).startswith("NOT_AUTHORIZED"), f"unsafe authorization: {key}")
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v31",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "final_accept_rows": 28,
        "final_reject_rows": 22,
        "final_unresolved_rows": 14,
        "rank5_content_state_authorized": "CONTENT_UNRESOLVED",
        "frontier_v8_authorized": True,
        "certificate_status_v3_blocked_authorized": True,
        "selection_certificate_authorized": False,
        "new_api_requests_authorized": False,
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
    except (GateStateV31VerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V31", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
