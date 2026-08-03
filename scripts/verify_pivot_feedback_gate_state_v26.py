#!/usr/bin/env python3
"""Verify the narrow rank-5 cross-provider API exception in gate-state v26."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v26.yaml"
STATUS = "PASS_RANK5_25_ROW_API_EXCEPTION_ONLY_OUTCOMES_FAIL_CLOSED"


class GateStateV26VerificationError(RuntimeError):
    """Fail-closed gate v26 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV26VerificationError(message)


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


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _ssd_file(binding: Mapping[str, Any], basename_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "SSD root differs")
    path = root / str(binding[basename_key])
    require(path.is_file() and not path.is_symlink(), f"input invalid: {path.name}")
    require(_sha256(path) == binding["sha256"], f"input SHA: {path.name}")
    return path


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate = _yaml(gate_path.resolve(strict=True))
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v26", "schema")
    require(
        gate.get("status")
        == "VALID_RANK5_25_ROW_CROSS_PROVIDER_EXCEPTION_AUTHORIZED_OUTCOMES_FAIL_CLOSED",
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_doc = _yaml(_repo_file(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")

    frozen = _mapping(gate.get("frozen_rank5_inputs"), "frozen inputs")
    sample_binding = _mapping(frozen.get("sample"), "sample")
    routes_binding = _mapping(frozen.get("local_routes"), "routes")
    sample_path = _ssd_file(sample_binding, "basename")
    routes_path = _ssd_file(routes_binding, "basename")
    require(pq.ParquetFile(sample_path).metadata.num_rows == sample_binding["rows"] == 96, "sample rows")
    with routes_path.open(newline="", encoding="utf-8") as handle:
        routes = list(csv.DictReader(handle))
    require(len(routes) == routes_binding["rows"] == 96, "route rows")
    recent = [row for row in routes if int(row["window_code"]) in {1, 2}]
    require(len(recent) == routes_binding["recent_rows"] == 64, "recent rows")
    counts = {
        route: sum(row["local_route"] == route for row in recent)
        for route in {
            "NO_API_RULE_ACCEPT",
            "NO_API_RULE_REJECT",
            "API_REQUIRED_AMBIGUOUS",
            "QUARANTINE_UNRESOLVED",
        }
    }
    require(counts["NO_API_RULE_ACCEPT"] == routes_binding["recent_accept_rows"] == 27, "accept rows")
    require(counts["NO_API_RULE_REJECT"] == routes_binding["recent_reject_rows"] == 12, "reject rows")
    require(
        counts["API_REQUIRED_AMBIGUOUS"] == routes_binding["recent_api_required_rows"] == 25,
        "API rows",
    )
    require(counts["QUARANTINE_UNRESOLVED"] == routes_binding["recent_opaque_rows"] == 0, "opaque rows")

    terminal = _mapping(frozen.get("terminal_disposition"), "terminal")
    receipt = Path("/ssd") / str(terminal["root_basename"]) / str(terminal["receipt_basename"])
    report = Path("/ssd") / str(terminal["verification_basename"])
    require(_sha256(receipt) == terminal["receipt_sha256"], "terminal receipt SHA")
    require(_sha256(report) == terminal["verification_sha256"], "terminal verification SHA")
    require(_json(receipt).get("status") == terminal["receipt_status"], "terminal receipt status")
    require(_json(report).get("status") == terminal["verification_status"], "terminal verification status")

    exception = _mapping(gate.get("frozen_exception"), "exception")
    require(exception.get("exact_request_rows") == 25, "request rows")
    require(exception.get("provider_calls") == 2, "provider calls")
    require(float(exception.get("maximum_authorized_total_usd")) == 0.10, "cost cap")
    require(exception.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(exception.get("consensus_accept") == "BOTH_PROVIDERS_ON_TOPIC_AND_HIGH", "accept rule")
    require(exception.get("consensus_reject") == "BOTH_PROVIDERS_OFF_TOPIC_AND_HIGH", "reject rule")
    require(exception.get("all_other_results") == "REMAIN_UNRESOLVED", "fallback rule")
    require(exception.get("single_provider_promotion") is False, "single provider promotion")
    require(exception.get("outcome_fields_visible") is False, "outcome visibility")

    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_paid_api_exception_submission")
        == "AUTHORIZED_MAXIMUM_TOTAL_USD_0_10",
        "API authorization",
    )
    require(
        authorization.get("d0_rank5_augmented_disposition")
        == "NOT_AUTHORIZED_UNTIL_API_RESULT_VERIFIED",
        "disposition boundary",
    )
    for key in (
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
        "schema_version": "pivot_feedback_gate_state_verification.v26",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "api_request_rows_authorized": 25,
        "provider_calls_authorized": 2,
        "maximum_authorized_total_usd": 0.10,
        "human_review_required": False,
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
    except (GateStateV26VerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V26", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
