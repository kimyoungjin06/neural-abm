#!/usr/bin/env python3
"""Verify rank-5 target-identity correction gate v28."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v28.yaml"
STATUS = "PASS_RANK5_HCV_TARGET_CORRECTION_ONLY_OUTCOMES_FAIL_CLOSED"


class GateStateV28VerificationError(RuntimeError):
    """Fail-closed gate v28 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV28VerificationError(message)


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


def _repo(relative: str, digest: str) -> Path:
    path = (REPO_ROOT / relative).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == digest, f"repository SHA: {path.name}")
    return path


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate = _yaml(gate_path.resolve(strict=True))
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v28", "schema")
    require(
        gate.get("status")
        == "VALID_RANK5_HCV_TARGET_CORRECTION_TWO_CALL_SUCCESSOR_AUTHORIZED_OUTCOMES_FAIL_CLOSED",
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_doc = _yaml(_repo(parent["relative_path"], parent["sha256"]))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")

    target = _mapping(gate.get("authoritative_target"), "target")
    source = _yaml(_repo(target["source_relative_path"], target["source_sha256"]))
    require(source.get("schema_version") == target["source_schema_version"], "target source schema")
    source_target = _mapping(source.get("rank5_target"), "source target")
    require(
        (
            source_target["field_id"],
            source_target["field_candidate_rank"],
            source_target["topic_id"],
            source_target["topic_name"],
            source_target["entry_year"],
        )
        == (27, 5, 10151, "Hepatitis C virus research", 1993),
        "authoritative target",
    )
    card = _mapping(source.get("frozen_topic_card"), "topic card")
    require(card.get("positive_patterns") == [
        r"\bhepatitis c\b",
        r"\bhcv\b",
        r"\bhepacivirus\b",
        r"\bnon a non b hepatitis\b",
    ], "positive anchors")
    require(target.get("generic_hepatitis_is_ambiguous") is True, "generic hepatitis")

    invalid = _mapping(gate.get("invalidated_api_evidence"), "invalid evidence")
    _repo(invalid["wrong_target_contract_relative_path"], invalid["wrong_target_contract_sha256"])
    _repo(invalid["transport_successor_relative_path"], invalid["transport_successor_sha256"])
    result_root = (Path("/ssd") / str(invalid["result_root_basename"])).resolve(strict=True)
    require(result_root.parent == Path("/ssd").resolve(), "invalid result root")
    receipt = result_root / str(invalid["result_receipt_basename"])
    verification = Path("/ssd") / str(invalid["result_verification_basename"])
    require(_sha256(receipt) == invalid["result_receipt_sha256"], "invalid result receipt SHA")
    require(_sha256(verification) == invalid["result_verification_sha256"], "invalid verification SHA")
    require(_json(receipt).get("production_disposition_created") is False, "invalid production use")
    require(invalid.get("wrong_target_name") == "Hepatitis B Virus Infection", "wrong target name")
    require(invalid.get("may_enter_rank5_consensus") is False, "invalid consensus reuse")
    require(invalid.get("may_change_production_disposition") is False, "invalid disposition use")

    successor = _mapping(gate.get("corrected_successor"), "successor")
    require(successor.get("exact_request_rows") == 25, "request rows")
    for key in (
        "source_row_set_change_allowed",
        "prompt_classification_rules_change_allowed",
        "models_change_allowed",
        "labels_change_allowed",
        "confidence_change_allowed",
        "dual_high_consensus_change_allowed",
        "visible_fields_change_allowed",
    ):
        require(successor.get(key) is False, f"scope expansion: {key}")
    require(successor.get("target_topic_name") == "Hepatitis C virus research", "corrected target")
    require(successor.get("additional_provider_calls_authorized") == 2, "provider calls")
    require(successor.get("retries_within_successor") == 0, "retries")
    cost = _mapping(gate.get("cost_control"), "cost")
    require(
        Decimal(str(cost["incurred_conservative_upper_bound_usd"]))
        + Decimal(str(cost["corrected_successor_request_upper_bound_usd"]))
        == Decimal(str(cost["cumulative_upper_bound_usd"])),
        "cumulative cost arithmetic",
    )
    require(
        Decimal(str(cost["cumulative_upper_bound_usd"]))
        <= Decimal(str(cost["cumulative_authorized_cap_usd"]))
        == Decimal("0.10"),
        "cumulative cost cap",
    )
    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_corrected_submission")
        == "AUTHORIZED_TWO_CALLS_NO_RETRY",
        "corrected authorization",
    )
    for key in (
        "d0_rank5_augmented_disposition",
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
        "schema_version": "pivot_feedback_gate_state_verification.v28",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "corrected_target_topic_id": 10151,
        "corrected_target_topic_name": "Hepatitis C virus research",
        "invalid_prior_results_reusable": False,
        "additional_provider_calls_authorized": 2,
        "cumulative_cost_upper_bound_usd": str(cost["cumulative_upper_bound_usd"]),
        "cumulative_cost_cap_usd": str(cost["cumulative_authorized_cap_usd"]),
        "augmented_disposition_authorized": False,
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
    except (GateStateV28VerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V28", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
