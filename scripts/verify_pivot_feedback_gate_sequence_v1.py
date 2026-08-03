#!/usr/bin/env python3
"""Validate the versioned acyclic pivot-feedback gate-sequence amendment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from yaml.constructor import ConstructorError
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPO_ROOT / "experiments/configs/pivot_feedback_gate_sequence_v1.yaml"
SCHEMA = "pivot_feedback_gate_sequence.v1"
STATUS = "PASS_ACYCLIC_GATE_SEQUENCE_CONTRACT_ONLY"
EXPECTED_SEQUENCE = [
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


class GateSequenceError(RuntimeError):
    """Fail-closed gate-sequence validation error."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate contract keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateSequenceError(message)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _list(value: Any, field: str) -> list[Any]:
    require(isinstance(value, list), f"{field} must be an array")
    return list(value)


def _safe_relative(value: Any, field: str) -> Path:
    require(isinstance(value, str) and value, f"{field} must be a path")
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, f"unsafe {field}")
    require(path.as_posix() == value, f"noncanonical {field}")
    return path


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, field: str) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    require(isinstance(value, dict), f"{field} must be an object")
    return dict(value)


def _load_contract(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.resolve(strict=True).read_bytes()
    try:
        value = yaml.load(raw, Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise GateSequenceError("invalid gate-sequence YAML") from exc
    require(isinstance(value, dict), "gate-sequence contract must be an object")
    return dict(value), hashlib.sha256(raw).hexdigest()


def _validate_parent(
    raw: Mapping[str, Any], *, expect_status: bool
) -> dict[str, Any]:
    item = dict(raw)
    relative = _safe_relative(item.get("relative_path"), "parent path")
    path = REPO_ROOT / relative
    require(path.is_file() and not path.is_symlink(), f"missing parent {relative}")
    require(_sha256_file(path) == item.get("sha256"), f"parent SHA mismatch: {relative}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"parent must be object: {relative}")
    require(value.get("schema_version") == item.get("schema_version"), f"parent schema mismatch: {relative}")
    if expect_status:
        observed_status = value.get("status")
        if observed_status is None:
            observed_status = _mapping(value.get("lifecycle"), "parent lifecycle").get(
                "freeze_status"
            )
        require(observed_status == item.get("status"), f"parent status mismatch: {relative}")
    require(item.get("mutation_allowed") is False, f"parent mutation must be forbidden: {relative}")
    return dict(value)


def _validate_evidence(raw: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(raw)
    relative = _safe_relative(item.get("relative_path"), "evidence path")
    path = REPO_ROOT / relative
    require(path.is_file() and not path.is_symlink(), f"missing evidence {relative}")
    require(_sha256_file(path) == item.get("sha256"), f"evidence SHA mismatch: {relative}")
    value = _read_json(path, f"evidence {relative}")
    require(value.get("status") == item.get("required_status"), f"evidence status mismatch: {relative}")
    return value


def _topological_order(nodes: Sequence[Mapping[str, Any]]) -> list[str]:
    by_id = {str(node["gate_id"]): dict(node) for node in nodes}
    require(len(by_id) == len(nodes), "duplicate gate ID")
    indegree = {gate_id: 0 for gate_id in by_id}
    children = {gate_id: [] for gate_id in by_id}
    for gate_id, node in by_id.items():
        dependencies = [str(value) for value in _list(node.get("requires"), f"{gate_id} requires")]
        require(len(dependencies) == len(set(dependencies)), f"duplicate dependency for {gate_id}")
        for dependency in dependencies:
            require(dependency in by_id, f"unknown dependency {dependency} for {gate_id}")
            require(
                int(by_id[dependency]["order"]) < int(node["order"]),
                f"backward or same-order dependency {dependency}->{gate_id}",
            )
            indegree[gate_id] += 1
            children[dependency].append(gate_id)
    ready = sorted(
        (gate_id for gate_id, degree in indegree.items() if degree == 0),
        key=lambda gate_id: int(by_id[gate_id]["order"]),
    )
    result: list[str] = []
    while ready:
        gate_id = ready.pop(0)
        result.append(gate_id)
        for child in sorted(children[gate_id], key=lambda item: int(by_id[item]["order"])):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=lambda item: int(by_id[item]["order"]))
    require(len(result) == len(nodes), "gate graph contains a cycle")
    return result


def validate(path: Path) -> dict[str, Any]:
    contract, contract_sha = _load_contract(path)
    require(contract.get("schema_version") == SCHEMA, "gate-sequence schema mismatch")
    require(contract.get("label") == "pivot_feedback_gate_sequence_v1", "gate-sequence label mismatch")
    require(contract.get("mode") == "PROSPECTIVE_ACYCLIC_GATE_ORDER_AMENDMENT", "gate-sequence mode mismatch")
    require(contract.get("status") == "VALID_PENDING_SUCCESSOR_IMPLEMENTATION", "gate-sequence pending status mismatch")

    lifecycle = _mapping(contract.get("lifecycle"), "lifecycle")
    require(lifecycle.get("legacy_v0_parents_mutated") is False, "legacy parents marked mutated")
    require(lifecycle.get("execution_authorized") is False, "execution is authorized")
    require(lifecycle.get("scientific_effect") == "NOT_ESTABLISHED", "scientific effect is overstated")

    parents = _mapping(contract.get("immutable_parents"), "immutable parents")
    require(
        set(parents)
        == {
            "legacy_g_minus_1_registry",
            "pending_g0_overlay",
            "cross_field_matrix",
            "peer_mechanism_registry",
        },
        "immutable parent set mismatch",
    )
    for key, raw in parents.items():
        _validate_parent(_mapping(raw, key), expect_status=key in {"legacy_g_minus_1_registry", "pending_g0_overlay"})

    evidence = _mapping(contract.get("completed_bounded_evidence"), "bounded evidence")
    require(set(evidence) == {"cutoff_primitives", "cutoff_source_replay"}, "bounded evidence set mismatch")
    replay = _validate_evidence(_mapping(evidence["cutoff_source_replay"], "source replay evidence"))
    _validate_evidence(_mapping(evidence["cutoff_primitives"], "cutoff evidence"))
    require(replay["scope"]["production_source_replay_pass"] is True, "source replay is not full")
    require(replay["scope"]["source_content_rehashed_in_this_run"] is False, "unexpected source rehash claim")
    require(replay["row_level_verification"]["all_projected_values_and_null_masks_exact"] is True, "source replay row equality missing")
    require(replay["gate_status"]["g_minus_1_data_readiness"] == "PENDING", "source replay improperly passes G-1")

    nodes = [
        _mapping(raw, "gate node")
        for raw in _list(contract.get("gate_nodes"), "gate nodes")
    ]
    require(
        [node.get("gate_id") for node in sorted(nodes, key=lambda item: int(item["order"]))]
        == EXPECTED_SEQUENCE,
        "declared gate order mismatch",
    )
    topological = _topological_order(nodes)
    require(topological == EXPECTED_SEQUENCE, "topological gate order mismatch")
    by_id = {str(node["gate_id"]): node for node in nodes}
    require(by_id["G0"]["requires"] == ["D0"], "G0 must depend only on D0")
    require(by_id["G0_5"]["requires"] == ["G0"], "G0.5 must depend on G0")
    require(by_id["G1"]["requires"] == ["G0_5"], "G1 must depend on G0.5")
    owners = [gate_id for gate_id, node in by_id.items() if node.get("owns_final_cutoff_panel")]
    require(owners == ["G0_5"], "final cutoff panel must have one G0.5 owner")
    require(by_id["G4"].get("current_state") == "SEALED_NOT_OPENED", "G4 is not sealed")

    rules = _mapping(contract.get("separation_rules"), "separation rules")
    require(rules.get("final_cutoff_panel_owner") == "G0_5", "final-panel rule differs")
    require(rules.get("final_cutoff_panel_is_g_minus_1a_prerequisite") is False, "G-1A still depends on final panel")
    require(rules.get("final_cutoff_panel_is_g0_prerequisite") is False, "G0 still depends on final panel")
    require(rules.get("g0_freezes_definitions_before_final_panel_materialization") is True, "G0/G0.5 order differs")
    forbidden = set(_list(rules.get("s0_forbidden_metrics"), "S0 forbidden metrics"))
    require(
        {
            "eligible_author_count",
            "matured_entry_opportunity_count",
            "fixed_horizon_outcome_coverage",
            "independent_attention_shock_count",
            "entry_effect_size",
            "persistence_effect_size",
            "overshoot_timing_gap",
            "predictive_score",
        }
        <= forbidden,
        "S0 forbidden metric boundary is incomplete",
    )
    require(rules.get("d0_entry_end_year") == 2012, "D0 end year differs")
    require(rules.get("d0_may_select_final_fields_by_effect_size") is False, "D0 may select fields by effect")
    require(rules.get("d0_may_change_confirmatory_years") is False, "D0 may change confirmatory years")
    require(rules.get("g4_open_before_g3_resolution") is False, "G4 can open too early")

    completion = _mapping(contract.get("completion_contract"), "completion contract")
    require(completion.get("passing_validator_status") == STATUS, "validator status contract mismatch")
    for field in (
        "config_alone_can_pass_g_minus_1a",
        "config_alone_can_pass_g0",
        "config_alone_can_authorize_execution",
        "config_alone_can_establish_scientific_effect",
    ):
        require(completion.get(field) is False, f"completion boundary differs: {field}")

    return {
        "schema_version": "pivot_feedback_gate_sequence_validation.v1",
        "status": STATUS,
        "contract_relative_path": path.resolve(strict=True).relative_to(REPO_ROOT).as_posix(),
        "contract_sha256": contract_sha,
        "topological_order": topological,
        "node_count": len(nodes),
        "edge_count": sum(len(_list(node["requires"], "requires")) for node in nodes),
        "immutable_parent_count": len(parents),
        "bounded_evidence_count": len(evidence),
        "legacy_v0_parents_unchanged": True,
        "cycle_present": False,
        "final_cutoff_panel_owner": "G0_5",
        "execution_authorized": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "next_required_implementation": completion["next_required_implementation"],
        "claim_boundary": contract["claim_boundary"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate(args.contract)
    except (GateSequenceError, OSError, ValueError, yaml.YAMLError) as exc:
        print(json.dumps({"status": "BLOCKED_INVALID_GATE_SEQUENCE_CONTRACT", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
