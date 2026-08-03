#!/usr/bin/env python3
"""Verify additive pivot-feedback gate-state v17."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v17.yaml"
STATUS = (
    "PASS_T10543_LOCAL_ROUTING_UNRESOLVED_RULE_V2_REPLAY_ONLY_"
    "OUTCOMES_FAIL_CLOSED"
)


class GateStateV17VerificationError(RuntimeError):
    """Fail-closed v17 gate verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV17VerificationError(message)


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
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(value, path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate_path = gate_path.resolve(strict=True)
    gate = _read_yaml(gate_path)
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v17", "schema")
    require(
        gate.get("status")
        == "VALID_T10543_LOCAL_ROUTING_UNRESOLVED_PERSISTENT_OPAQUE_RULE_GAP_IDENTIFIED_OUTCOMES_FAIL_CLOSED",
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    evidence = _mapping(gate.get("t10543_local_routing_evidence"), "evidence")
    for key in ("config", "producer", "verifier", "summary_markdown"):
        _repo_file(_mapping(evidence.get(key), key))
    summary_binding = _mapping(evidence.get("summary_json"), "summary")
    summary = _read_json(_repo_file(summary_binding))
    require(summary.get("schema_version") == summary_binding["required_schema_version"], "summary schema")
    require(summary.get("status") == summary_binding["required_status"], "summary status")
    local = _mapping(evidence.get("operator_local"), "operator evidence")
    root = (Path("/ssd") / str(local["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "operator root")
    receipt_path = root / str(local["receipt_basename"])
    verification_path = Path("/ssd") / str(local["verification_basename"])
    require(_sha256(receipt_path) == local["receipt_sha256"], "receipt SHA")
    require(_sha256(verification_path) == local["verification_sha256"], "verification SHA")
    require(_read_json(receipt_path).get("status") == local["receipt_status"], "receipt status")
    require(
        _read_json(verification_path).get("status") == local["verification_status"],
        "verification status",
    )
    observed = _mapping(gate.get("observed_result"), "observed")
    require(
        (
            observed["recent_rule_accept_rows"],
            observed["recent_rule_reject_rows"],
            observed["recent_informative_ambiguous_rows"],
            observed["recent_opaque_quarantine_rows"],
        )
        == (8, 1, 3, 52),
        "routing counts",
    )
    require(observed["candidate_rule_v1_state"] == "WOULD_CONTENT_UNRESOLVED", "v1 state")
    require(observed["production_disposition_created"] is False, "production disposition")
    proposal = _mapping(
        _mapping(gate.get("rule_gap"), "rule gap").get(
            "simplest_candidate_rule_v2_proposal"
        ),
        "v2 proposal",
    )
    require(proposal["threshold"] == 0.5, "v2 threshold")
    require(proposal["amendment_is_post_diagnostic"] is True, "diagnostic disclosure")
    require(
        proposal["retrospective_replay_required_before_production_use"] is True,
        "replay requirement",
    )
    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_candidate_rule_v2_post_diagnostic_amendment_design")
        == "AUTHORIZED",
        "v2 design",
    )
    require(
        authorization.get("d0_candidate_rule_v2_five_candidate_retrospective_replay")
        == "AUTHORIZED_ZERO_COST_ONLY",
        "v2 replay",
    )
    for key in (
        "d0_new_title_extraction",
        "d0_raw_openalex_title_scan",
        "d0_new_api_request_creation",
        "d0_paid_api_exception_submission",
        "d0_global_selection_certificate",
        "d0_episode_reselection_successor",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"unsafe authorization: {key}",
        )
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v17",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "candidate_rule_v1_state": observed["candidate_rule_v1_state"],
        "recent_opaque_quarantine_rows": observed["recent_opaque_quarantine_rows"],
        "candidate_rule_v2_replay_authorized": True,
        "new_api_request_creation_authorized": False,
        "production_disposition_authorized": False,
        "selection_certificate_authorized": False,
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
        GateStateV17VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V17", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
