#!/usr/bin/env python3
"""Verify the additive rank-5 production-and-certificate gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v35.yaml"
STATUS = "PASS_PIVOT_FEEDBACK_GATE_STATE_V35"


class GateStateV35VerificationError(RuntimeError):
    """Fail-closed v35 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV35VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def verify(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = config_path.resolve(strict=True)
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    require(config.get("schema_version") == "pivot_feedback_gate_state.v35", "schema")
    require(
        config.get("status")
        == "VALID_RANK5_POST_DIAGNOSTIC_CONTENT_PASS_PRODUCTION_AND_CERTIFICATES_AUTHORIZED",
        "status",
    )
    parent = _mapping(config.get("additive_parent"), "parent")
    parent_doc = _mapping(yaml.safe_load(_repo(parent).read_text(encoding="utf-8")), "parent doc")
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    result = _mapping(config.get("verified_result"), "verified result")
    _repo(_mapping(result.get("config"), "result config"))
    _repo(_mapping(result.get("producer"), "producer"))
    _repo(_mapping(result.get("verifier"), "verifier"))
    operator = _mapping(result.get("operator_local"), "operator")
    root = (Path("/ssd") / str(operator["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "operator root")
    for basename_key, sha_key in (
        ("replay_basename", "replay_sha256"),
        ("candidate_basename", "candidate_sha256"),
        ("receipt_basename", "receipt_sha256"),
    ):
        artifact = root / str(operator[basename_key])
        require(_sha256(artifact) == operator[sha_key], f"operator SHA: {artifact.name}")
    verification = (Path("/ssd") / str(operator["verification_basename"])).resolve(strict=True)
    require(verification.parent == Path("/ssd").resolve(), "verification parent")
    require(_sha256(verification) == operator["verification_sha256"], "verification SHA")
    require(_json(root / str(operator["receipt_basename"])).get("status") == operator["receipt_status"], "receipt status")
    require(_json(verification).get("status") == operator["verification_status"], "verification status")
    counts = _mapping(result.get("counts"), "counts")
    require(
        {
            "replay_rows": counts["replay_rows"],
            "new_accept_rows": counts["new_accept_rows"],
            "new_reject_rows": counts["new_reject_rows"],
            "remaining_unresolved_rows": counts["remaining_unresolved_rows"],
            "final_accept_rows": counts["final_accept_rows"],
            "final_reject_rows": counts["final_reject_rows"],
            "final_unresolved_rows": counts["final_unresolved_rows"],
            "final_content_state": counts["final_content_state"],
        }
        == {
            "replay_rows": 9,
            "new_accept_rows": 2,
            "new_reject_rows": 0,
            "remaining_unresolved_rows": 7,
            "final_accept_rows": 32,
            "final_reject_rows": 25,
            "final_unresolved_rows": 7,
            "final_content_state": "CONTENT_PASS",
        },
        "counts",
    )
    require(counts.get("retrospective_post_diagnostic") is True, "retrospective")
    require(counts.get("out_of_sample_validation") is False, "out of sample")
    successor = _mapping(config.get("authorized_successor"), "successor")
    require(successor.get("computer_science_selected_candidate_ranks") == [1, 45], "CS ranks")
    require(successor.get("medicine_selected_candidate_ranks") == [5, 6], "Medicine ranks")
    require(successor.get("field_certificates_issued") == 2, "field certificates")
    require(successor.get("global_selection_certificate_issued") is True, "global certificate")
    require(successor.get("deterministic_reselection_executed") is False, "reselection execution")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(authorization.get("d0_rank5_production_disposition") == "AUTHORIZED_CONTENT_PASS_ONLY", "production")
    require(authorization.get("d0_selection_certificate") == "AUTHORIZED_EXACT_RANKS_ONLY", "certificate")
    require(
        authorization.get("d0_deterministic_reselection")
        == "NOT_AUTHORIZED_UNTIL_CLOSEOUT_INDEPENDENTLY_VERIFIED",
        "reselection",
    )
    for key in (
        "d0_new_api_request_creation",
        "d0_paid_api_submission",
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return {
        "schema_version": "pivot_feedback_gate_state_v35_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(path),
        "rank5_content_state": "CONTENT_PASS",
        "computer_science_selected_candidate_ranks": [1, 45],
        "medicine_selected_candidate_ranks": [5, 6],
        "global_selection_certificate_authorized": True,
        "deterministic_reselection_authorized": False,
        "retrospective_post_diagnostic": True,
        "out_of_sample_validation": False,
        "outcomes_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.config), indent=2, sort_keys=True))
        return 0
    except (GateStateV35VerificationError, OSError, ValueError, TypeError, KeyError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V35", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
