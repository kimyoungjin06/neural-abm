#!/usr/bin/env python3
"""Verify the transport-only rank-5 API successor authorization."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v27.yaml"
STATUS = "PASS_RANK5_TRANSPORT_ONLY_SUCCESSOR_OUTCOMES_FAIL_CLOSED"


class GateStateV27VerificationError(RuntimeError):
    """Fail-closed gate v27 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV27VerificationError(message)


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


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate = _yaml(gate_path.resolve(strict=True))
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v27", "schema")
    require(
        gate.get("status")
        == "VALID_RANK5_TRANSPORT_ONLY_SUCCESSOR_AUTHORIZED_CUMULATIVE_COST_FAIL_CLOSED",
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "parent escapes")
    require(_sha256(parent_path) == parent["sha256"], "parent SHA")
    parent_doc = _yaml(parent_path)
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")

    failed = _mapping(gate.get("failed_attempt"), "failed attempt")
    contract = (REPO_ROOT / str(failed["contract_relative_path"])).resolve(strict=True)
    require(contract.is_relative_to(REPO_ROOT), "contract escapes")
    require(_sha256(contract) == failed["contract_sha256"], "contract SHA")
    require(_yaml(contract).get("schema_version") == failed["required_contract_schema"], "contract schema")
    prepared = Path("/ssd") / str(failed["prepared_root_basename"])
    execution = Path("/ssd") / str(failed["execution_root_basename"])
    require(
        _sha256(prepared / "cross_provider_preparation.outputs.json")
        == failed["preparation_receipt_sha256"],
        "preparation receipt SHA",
    )
    verification = Path("/ssd") / (
        str(failed["prepared_root_basename"]) + ".verification-v1.json"
    )
    require(_sha256(verification) == failed["preparation_verification_sha256"], "preparation verification SHA")
    failure_path = execution / str(failed["failure_basename"])
    require(_sha256(failure_path) == failed["failure_sha256"], "failure SHA")
    failure = _json(failure_path)
    require(failure.get("status") == failed["failure_status"], "failure status")
    require(failure.get("provider") == "claude", "failed provider")
    require(failure.get("http_status") == 400, "failed HTTP")
    require(failed.get("usable_provider_results_preserved") == 0, "usable first result")
    require(failed.get("first_attempt_result_may_enter_consensus") is False, "first result reuse")

    successor = _mapping(gate.get("transport_only_successor"), "successor")
    for key in (
        "request_row_set_change_allowed",
        "prompt_change_allowed",
        "label_change_allowed",
        "confidence_change_allowed",
        "consensus_rule_change_allowed",
        "model_change_allowed",
        "input_visibility_change_allowed",
        "openai_request_change_allowed",
    ):
        require(successor.get(key) is False, f"scope expansion: {key}")
    correction = _mapping(successor.get("claude_request_change"), "Claude correction")
    require(correction.get("remove_unsupported_array_minItems") is True, "minItems correction")
    require(correction.get("remove_unsupported_array_maxItems") is True, "maxItems correction")
    require(
        correction.get(
            "exact_25_unique_rows_remains_post_response_verification_requirement"
        )
        is True,
        "exact verification",
    )
    require(successor.get("additional_provider_calls_authorized") == 2, "provider calls")
    require(successor.get("retries_within_successor") == 0, "retries")
    cumulative = Decimal(str(successor["cumulative_attempt_cost_upper_bound_usd"]))
    cap = Decimal(str(successor["cumulative_authorized_cap_usd"]))
    require(cumulative <= cap == Decimal("0.10"), "cumulative cost cap")

    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_transport_successor_submission")
        == "AUTHORIZED_TWO_CALLS_NO_RETRY",
        "successor authorization",
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
        "schema_version": "pivot_feedback_gate_state_verification.v27",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "transport_only_successor_authorized": True,
        "additional_provider_calls_authorized": 2,
        "cumulative_cost_upper_bound_usd": str(cumulative),
        "cumulative_cost_cap_usd": str(cap),
        "first_attempt_results_reusable": False,
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
    except (GateStateV27VerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V27", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
