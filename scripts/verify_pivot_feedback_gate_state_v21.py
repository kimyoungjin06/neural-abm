#!/usr/bin/env python3
"""Verify additive pivot-feedback gate-state v21."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v21.yaml"
STATUS = "PASS_FRONTIER_V4_FROZEN_EXACT_RANK4_SAMPLE_ONLY_OUTCOMES_FAIL_CLOSED"


class GateStateV21VerificationError(RuntimeError):
    """Fail-closed v21 gate verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV21VerificationError(message)


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
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v21", "schema")
    require(
        gate.get("status")
        == (
            "VALID_RANK3_CONTENT_FAIL_FRONTIER_V4_FROZEN_EXACT_MEDICINE_"
            "RANK4_SAMPLE_AUTHORIZED_OUTCOMES_FAIL_CLOSED"
        ),
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")

    evidence = _mapping(gate.get("rank3_production_evidence"), "evidence")
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
    require(root.parent == Path("/ssd").resolve(), "operator root")
    receipt_path = root / str(local["receipt_basename"])
    verification_path = Path("/ssd") / str(local["verification_basename"])
    require(_sha256(receipt_path) == local["receipt_sha256"], "receipt SHA")
    require(_sha256(verification_path) == local["verification_sha256"], "verification SHA")
    require(
        _read_json(receipt_path).get("status") == local["receipt_status"],
        "receipt status",
    )
    require(
        _read_json(verification_path).get("status")
        == local["verification_status"],
        "verification status",
    )

    result = _mapping(gate.get("production_result"), "production result")
    require(
        (
            result["cumulative_production_dispositions"],
            result["content_pass_rows"],
            result["content_fail_rows"],
            result["medicine_required_candidate_rank"],
            result["medicine_required_topic_id"],
            result["medicine_required_entry_year"],
        )
        == (6, 3, 3, 4, 12505, 1993),
        "production result",
    )
    require(result["medicine_field_certificate_issued"] is False, "medicine certificate")
    require(result["global_selection_certificate_issued"] is False, "global certificate")

    target = _mapping(gate.get("exact_rank4_sample_authorization"), "target")
    require(
        (
            target["field_id"],
            target["field_candidate_rank"],
            target["topic_id"],
            target["entry_year"],
            target["rows_per_window"],
            target["total_rows"],
        )
        == (27, 4, 12505, 1993, 32, 96),
        "target contract",
    )
    require(target["api_request_creation"] == "NOT_AUTHORIZED", "target API")

    authorization = _mapping(gate.get("authorization"), "authorization")
    for key in (
        "d0_exact_rank4_normalized_candidate_scan",
        "d0_exact_rank4_96_id_sample",
        "d0_exact_rank4_selected_id_title_lookup",
        "d0_reuse_t12505_topic_card",
        "d0_rank4_local_routing",
    ):
        require(
            str(authorization.get(key)).startswith("AUTHORIZED"),
            f"authorized work: {key}",
        )
    for key in (
        "d0_general_raw_openalex_title_scan",
        "d0_full_candidate_title_copy",
        "d0_new_api_request_creation",
        "d0_paid_api_exception_submission",
        "d0_rank4_production_content_disposition",
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
        "schema_version": "pivot_feedback_gate_state_verification.v21",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "rank4_sample_authorized": True,
        "rank4_topic_id": 12505,
        "rank4_entry_year": 1993,
        "rank4_title_rows": 96,
        "new_api_request_creation_authorized": False,
        "rank4_production_disposition_authorized": False,
        "global_selection_certificate_authorized": False,
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
        GateStateV21VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V21", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
