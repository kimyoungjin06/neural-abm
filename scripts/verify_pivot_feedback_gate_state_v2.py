#!/usr/bin/env python3
"""Verify the additive post-execution pivot-feedback gate ledger v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v2"
STATUS = "PASS_G_MINUS_1A_COMPLETE_S0_PARTIAL_D0_FAIL_CLOSED"
EXPECTED_ORDER = [
    "G_MINUS_1A",
    "S0",
    "D0",
    "G0",
    "G0_5",
    "G1",
    "G2",
    "G3",
    "G4",
    "G5",
]


class GateStateError(RuntimeError):
    """Fail-closed gate-state validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _safe_path(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(isinstance(current, Mapping) and part in current, f"missing {dotted_path}")
        current = current[part]
    return current


def _read(path: Path, *, yaml_document: bool) -> dict[str, Any]:
    try:
        value = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_document
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise GateStateError(f"invalid document {path.name}: {error}") from error
    return _mapping(value, path.name)


def _validate_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    yaml_document: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = dict(raw)
    path = _safe_path(repo_root, item.get("relative_path"), "bound evidence path")
    observed_sha = _sha256(path)
    require(observed_sha == item.get("sha256"), f"bound evidence SHA mismatch: {path.name}")
    document = _read(path, yaml_document=yaml_document)
    require(document.get("schema_version") == item.get("required_schema_version"), "bound schema mismatch")
    require(document.get("status") == item.get("required_status"), "bound status mismatch")
    for dotted_path, expected in _mapping(
        item.get("required_fields", {}), "required fields"
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"bound field mismatch: {dotted_path}",
        )
    return document, {
        "relative_path": item["relative_path"],
        "sha256": observed_sha,
        "status": document["status"],
    }


def validate(path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    path = path.resolve(strict=True)
    ledger = _read(path, yaml_document=True)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema mismatch")
    require(ledger.get("label") == "pivot_feedback_gate_state_v2", "ledger label mismatch")
    require(ledger.get("mode") == "ADDITIVE_POST_EXECUTION_GATE_LEDGER", "ledger mode mismatch")
    require(
        ledger.get("status") == "VALID_G_MINUS_1A_PASS_S0_PARTIAL_D0_NOT_AUTHORIZED",
        "ledger status mismatch",
    )
    parent_raw = _mapping(ledger.get("acyclic_order_parent"), "acyclic parent")
    require(parent_raw.get("mutation_allowed") is False, "v1 parent mutation is allowed")
    parent, parent_result = _validate_binding(
        parent_raw, repo_root=repo_root, yaml_document=True
    )
    parent_order = [
        item["gate_id"]
        for item in sorted(parent["gate_nodes"], key=lambda value: int(value["order"]))
    ]
    require(parent_order == EXPECTED_ORDER, "parent gate order mismatch")

    evidence_raw = _mapping(ledger.get("execution_evidence"), "execution evidence")
    require(
        set(evidence_raw) == {"g_minus_1a_receipt", "s0_core_inventory"},
        "execution evidence set mismatch",
    )
    _, g1_result = _validate_binding(
        _mapping(evidence_raw["g_minus_1a_receipt"], "G-1A evidence"),
        repo_root=repo_root,
        yaml_document=False,
    )
    _, s0_result = _validate_binding(
        _mapping(evidence_raw["s0_core_inventory"], "S0 evidence"),
        repo_root=repo_root,
        yaml_document=False,
    )

    states = _mapping(ledger.get("gate_state"), "gate state")
    require(list(states) == EXPECTED_ORDER, "ledger gate order mismatch")
    g1a = _mapping(states["G_MINUS_1A"], "G-1A state")
    s0 = _mapping(states["S0"], "S0 state")
    d0 = _mapping(states["D0"], "D0 state")
    require(g1a == {"state": "PASS_BOUNDED", "complete": True, "authorizes": ["S0"]}, "G-1A state differs")
    require(s0.get("state") == "PARTIAL_CORE_PASS_DEFERRED_COMPONENTS_NOT_RUN", "S0 state differs")
    require(s0.get("complete") is False and s0.get("authorizes") == [], "S0 improperly complete")
    require(
        set(s0.get("missing_components", []))
        == {
            "ORCID_TOKEN_STABILITY_FIELD_YEAR",
            "AUTHOR_WORK_AFFILIATION_FIELD_YEAR",
            "DISTINCT_COAUTHOR_GRAPH_FIELD_YEAR",
        },
        "S0 missing component set differs",
    )
    require(
        d0 == {"state": "NOT_AUTHORIZED", "complete": False, "requires": ["S0"]},
        "D0 state differs",
    )
    require(states["G4"] == {"state": "SEALED_NOT_OPENED", "complete": False}, "G4 is not sealed")

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(authorization.get("s0_remaining_components") == "AUTHORIZED", "S0 remainder is not authorized")
    for key in (
        "d0_discovery_pilot",
        "g0_measurement_freeze",
        "final_panel_materialization",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "effect is overstated")
    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(next_step.get("gate") == "S0", "next gate differs")
    require(next_step.get("workload_class") == "CPU_IO", "workload class differs")
    require(next_step.get("gpu_required") is False, "GPU is improperly required")
    require(next_step.get("d0_may_start_before_completion") is False, "D0 may start early")

    return {
        "schema_version": "pivot_feedback_gate_state_verification.v2",
        "status": STATUS,
        "ledger_sha256": _sha256(path),
        "acyclic_parent": parent_result,
        "execution_evidence": {
            "g_minus_1a_receipt": g1_result,
            "s0_core_inventory": s0_result,
        },
        "g_minus_1a": "PASS_BOUNDED",
        "s0": "PARTIAL_CORE_PASS_DEFERRED_COMPONENTS_NOT_RUN",
        "s0_missing_component_count": 3,
        "d0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "next_workload_class": "CPU_IO",
        "gpu_required": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v2.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(validate(args.ledger), indent=2, sort_keys=True))
    except (GateStateError, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED_GATE_STATE_V2", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
