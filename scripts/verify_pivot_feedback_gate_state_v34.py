#!/usr/bin/env python3
"""Verify the additive rank-5 post-diagnostic external-rule replay gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v34.yaml"
STATUS = "PASS_PIVOT_FEEDBACK_GATE_STATE_V34"


class GateStateV34VerificationError(RuntimeError):
    """Fail-closed gate-state verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV34VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = config_path.resolve(strict=True)
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    require(config.get("schema_version") == "pivot_feedback_gate_state.v34", "schema")
    require(
        config.get("status")
        == "VALID_RANK5_POST_DIAGNOSTIC_ACCEPT_ONLY_EXTERNAL_RULE_REPLAY_AUTHORIZED",
        "status",
    )
    parent = _mapping(config.get("additive_parent"), "parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "parent escapes repository")
    require(_sha256(parent_path) == parent["sha256"], "parent SHA")
    parent_doc = _mapping(
        yaml.safe_load(parent_path.read_text(encoding="utf-8")), "parent document"
    )
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    basis = _mapping(config.get("diagnostic_basis"), "diagnostic basis")
    require(basis.get("unresolved_rows") == 9, "unresolved rows")
    require(basis.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(basis.get("rule_selected_after_examining_disagreements") is True, "retrospective")
    require(basis.get("out_of_sample_validation_claimed") is False, "out of sample")
    replay = _mapping(config.get("authorized_replay"), "replay")
    require(replay.get("cohort") == "ALL_EXACT_NINE_EXTERNAL_EVIDENCE_UNRESOLVED_ROWS", "cohort")
    require(replay.get("rule_kind") == "DETERMINISTIC_ACCEPT_ONLY", "rule kind")
    require(replay.get("negative_override") == "FORBIDDEN", "negative override")
    require(replay.get("absence_is_off_topic") is False, "absence semantics")
    require(replay.get("expected_new_accept_rows") == 2, "accept rows")
    require(replay.get("expected_remaining_unresolved_rows") == 7, "remaining unresolved")
    require(replay.get("expected_candidate_state") == "CONTENT_PASS", "candidate state")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_external_controlled_subject_rule_replay")
        == "AUTHORIZED_EXACT_NINE_ONLY",
        "replay authorization",
    )
    require(
        authorization.get("d0_rank5_production_disposition_update")
        == "NOT_AUTHORIZED_UNTIL_INDEPENDENT_REPLAY_VERIFICATION",
        "production authorization",
    )
    require(
        authorization.get("d0_selection_certificate")
        == "NOT_AUTHORIZED_UNTIL_INDEPENDENT_REPLAY_VERIFICATION",
        "certificate authorization",
    )
    for key in (
        "d0_deterministic_reselection",
        "d0_new_api_request_creation",
        "d0_paid_api_submission",
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return {
        "schema_version": "pivot_feedback_gate_state_v34_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(path),
        "parent_sha256": _sha256(parent_path),
        "replay_rows": 9,
        "expected_new_accept_rows": 2,
        "production_disposition_authorized": False,
        "selection_certificate_authorized": False,
        "deterministic_reselection_authorized": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "out_of_sample_validation_claimed": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.config), indent=2, sort_keys=True))
        return 0
    except (GateStateV34VerificationError, OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V34", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
