#!/usr/bin/env python3
"""Verify the additive S0-complete pivot-feedback gate ledger v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v3"
LEDGER_STATUS = "VALID_G_MINUS_1A_AND_S0_PASS_D0_DISCOVERY_ONLY_AUTHORIZED"
VERIFY_STATUS = (
    "PASS_G_MINUS_1A_AND_S0_COMPLETE_D0_DISCOVERY_ONLY_AUTHORIZED"
)
S0_RECEIPT_STATUS = (
    "PASS_S0_THRESHOLD_FREE_FIELD_AUTHOR_GRAPH_AND_AFFILIATION_INVENTORY_"
    "D0_DISCOVERY_ONLY_AUTHORIZED"
)
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
S0_VERIFIER_PATHS = {
    "core": "scripts/verify_pivot_feedback_s0_threshold_free_inventory.py",
    "orcid_affiliation": (
        "scripts/verify_pivot_feedback_s0_authorship_orcid_affiliation.py"
    ),
    "coauthor_graph": "scripts/verify_pivot_feedback_s0_coauthor_graph_profile.py",
}
S0_COMPLETION_VERIFIER_PATH = (
    "scripts/verify_pivot_feedback_s0_completion_v1.py"
)


class GateStateV3Error(RuntimeError):
    """Fail-closed gate-state v3 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV3Error(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_path(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing {dotted_path}",
        )
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
        raise GateStateV3Error(f"invalid document {path.name}: {error}") from error
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
    require(
        observed_sha == item.get("sha256"),
        f"bound evidence SHA mismatch: {path.name}",
    )
    document = _read(path, yaml_document=yaml_document)
    require(
        document.get("schema_version") == item.get("required_schema_version"),
        "bound schema mismatch",
    )
    require(
        document.get("status") == item.get("required_status"),
        "bound status mismatch",
    )
    for dotted_path, expected in _mapping(
        item.get("required_fields", {}),
        "required fields",
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
    require(ledger.get("label") == "pivot_feedback_gate_state_v3", "ledger label mismatch")
    require(
        ledger.get("mode") == "ADDITIVE_S0_COMPLETION_GATE_LEDGER",
        "ledger mode mismatch",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status mismatch")

    parent_raw = _mapping(ledger.get("additive_parent"), "additive parent")
    require(parent_raw.get("mutation_allowed") is False, "v2 parent mutation is allowed")
    parent, parent_result = _validate_binding(
        parent_raw,
        repo_root=repo_root,
        yaml_document=True,
    )
    parent_states = _mapping(parent.get("gate_state"), "parent gate state")
    require(list(parent_states) == EXPECTED_ORDER, "parent gate order mismatch")
    require(
        _mapping(parent_states["S0"], "parent S0").get("complete") is False,
        "v2 parent no longer records partial S0",
    )
    require(
        _mapping(parent_states["D0"], "parent D0").get("state")
        == "NOT_AUTHORIZED",
        "v2 parent D0 state changed",
    )

    evidence_raw = _mapping(ledger.get("execution_evidence"), "execution evidence")
    require(
        set(evidence_raw) == {"g_minus_1a_receipt", "s0_completion_receipt"},
        "execution evidence set mismatch",
    )
    _, g1_result = _validate_binding(
        _mapping(evidence_raw["g_minus_1a_receipt"], "G-1A evidence"),
        repo_root=repo_root,
        yaml_document=False,
    )
    s0_receipt, s0_result = _validate_binding(
        _mapping(evidence_raw["s0_completion_receipt"], "S0 evidence"),
        repo_root=repo_root,
        yaml_document=False,
    )
    require(
        s0_receipt.get("status") == S0_RECEIPT_STATUS,
        "S0 completion status mismatch",
    )
    require(
        s0_receipt.get("completion_verifier_sha256")
        == _sha256(repo_root / S0_COMPLETION_VERIFIER_PATH),
        "S0 completion verifier SHA mismatch",
    )
    component_receipts = _mapping(
        s0_receipt.get("component_receipts"),
        "S0 component receipts",
    )
    require(
        set(component_receipts) == set(S0_VERIFIER_PATHS),
        "S0 component receipt set mismatch",
    )
    for label, verifier_relative_path in S0_VERIFIER_PATHS.items():
        component = _mapping(component_receipts[label], f"{label} component")
        require(
            component.get("verifier_sha256")
            == _sha256(repo_root / verifier_relative_path),
            f"{label} verifier SHA mismatch",
        )

    states = _mapping(ledger.get("gate_state"), "gate state")
    require(list(states) == EXPECTED_ORDER, "ledger gate order mismatch")
    require(
        _mapping(states["G_MINUS_1A"], "G-1A state")
        == {"state": "PASS_BOUNDED", "complete": True, "authorizes": ["S0"]},
        "G-1A state differs",
    )
    require(
        _mapping(states["S0"], "S0 state")
        == {
            "state": "PASS_THRESHOLD_FREE_INVENTORY",
            "complete": True,
            "authorizes": ["D0"],
        },
        "S0 state differs",
    )
    require(
        _mapping(states["D0"], "D0 state")
        == {
            "state": "AUTHORIZED_NOT_RUN",
            "complete": False,
            "requires": ["S0"],
            "entry_year_end": 2012,
        },
        "D0 state differs",
    )
    require(
        _mapping(states["G0"], "G0 state").get("state") == "PENDING",
        "G0 is not pending",
    )
    require(
        _mapping(states["G4"], "G4 state")
        == {"state": "SEALED_NOT_OPENED", "complete": False},
        "G4 is not sealed",
    )

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(
        authorization.get("d0_discovery_pilot")
        == "AUTHORIZED_DISCOVERY_ONLY_ENTRY_YEAR_AT_MOST_2012",
        "D0 authorization mismatch",
    )
    for key in (
        "g0_measurement_freeze",
        "final_panel_materialization",
        "model_fitting",
        "g4_open",
    ):
        require(
            authorization.get(key) == "NOT_AUTHORIZED",
            f"unsafe authorization: {key}",
        )
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect is overstated",
    )

    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(next_step.get("gate") == "D0", "next gate differs")
    require(next_step.get("entry_year_end") == 2012, "D0 discovery cutoff differs")
    require(
        next_step.get("final_field_choice_by_effect_size_allowed") is False,
        "effect-size field choice is allowed",
    )
    require(
        next_step.get("confirmatory_year_changes_allowed") is False,
        "confirmatory year changes are allowed",
    )
    require(next_step.get("gpu_required") is False, "GPU is improperly required")

    return {
        "schema_version": "pivot_feedback_gate_state_verification.v3",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(path),
        "additive_parent": parent_result,
        "execution_evidence": {
            "g_minus_1a_receipt": g1_result,
            "s0_completion_receipt": s0_result,
        },
        "g_minus_1a": "PASS_BOUNDED",
        "s0": "PASS_THRESHOLD_FREE_INVENTORY",
        "d0": "AUTHORIZED_NOT_RUN",
        "d0_entry_year_end": 2012,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "next_workload_class": next_step.get("workload_class"),
        "gpu_required": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v3.yaml",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(validate(args.ledger), indent=2, sort_keys=True))
    except (GateStateV3Error, OSError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED_GATE_STATE_V3", "error": str(error)}))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
