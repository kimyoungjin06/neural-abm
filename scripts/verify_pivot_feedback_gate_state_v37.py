#!/usr/bin/env python3
"""Verify the certified-reselection panel-rebuild successor gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v37.yaml"
STATUS = "PASS_PIVOT_FEEDBACK_GATE_STATE_V37"


class GateStateV37VerificationError(RuntimeError):
    """Fail-closed v37 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV37VerificationError(message)


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
    require(config.get("schema_version") == "pivot_feedback_gate_state.v37", "schema")
    require(
        config.get("status")
        == "VALID_CERTIFIED_EPISODE_RESELECTION_COMPLETE_EXACT_OUTCOME_BLIND_PANEL_REBUILD_AUTHORIZED",
        "status",
    )
    parent = _mapping(config.get("additive_parent"), "parent")
    parent_doc = _mapping(yaml.safe_load(_repo(parent).read_text(encoding="utf-8")), "parent doc")
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")

    reselection = _mapping(config.get("verified_reselection"), "reselection")
    _repo(_mapping(reselection.get("config"), "reselection config"))
    _repo(_mapping(reselection.get("producer"), "producer"))
    _repo(_mapping(reselection.get("verifier"), "verifier"))
    operator = _mapping(reselection.get("operator_local"), "operator")
    root = (Path("/ssd") / str(operator["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "operator root")
    for basename_key, sha_key in (
        ("selection_basename", "selection_sha256"),
        ("decision_basename", "decision_sha256"),
        ("receipt_basename", "receipt_sha256"),
    ):
        artifact = root / str(operator[basename_key])
        require(_sha256(artifact) == operator[sha_key], f"artifact SHA: {artifact.name}")
    verification = (Path("/ssd") / str(operator["verification_basename"])).resolve(strict=True)
    require(verification.parent == Path("/ssd").resolve(), "verification parent")
    require(_sha256(verification) == operator["verification_sha256"], "verification SHA")
    receipt = _json(root / str(operator["receipt_basename"]))
    verification_doc = _json(verification)
    require(receipt.get("status") == operator["receipt_status"], "receipt status")
    require(verification_doc.get("status") == operator["verification_status"], "verification status")
    require(receipt.get("deterministic_reselection_executed") is True, "reselection execution")
    require(verification_doc.get("deterministic_reselection_executed") is True, "verified execution")
    require(receipt.get("outcomes_read") is False, "receipt outcomes")
    require(verification_doc.get("outcomes_read") is False, "verification outcomes")

    selection = pq.read_table(root / str(operator["selection_basename"]))
    require(selection.num_rows == 4, "selection rows")
    rows = sorted(selection.to_pylist(), key=lambda row: row["certified_episode_index"])
    require(
        [(row["field_id"], row["field_candidate_rank"]) for row in rows]
        == [(17, 1), (17, 45), (27, 5), (27, 6)],
        "exact selected ranks",
    )
    require(all(row["content_state"] == "CONTENT_PASS" for row in rows), "content states")
    result = _mapping(reselection.get("result"), "result")
    require(result.get("selected_episode_rows") == 4, "result rows")
    require(result.get("retained_episode_rows") == 3, "retained rows")
    require(result.get("newly_selected_episode_rows") == 1, "new rows")
    require(result.get("deselected_prior_episode_rows") == 1, "deselected rows")
    require(result.get("computer_science_selected_candidate_ranks") == [1, 45], "CS ranks")
    require(result.get("medicine_selected_candidate_ranks") == [5, 6], "Medicine ranks")
    require(result.get("author_opportunity_panel_rebuild_executed") is False, "panel execution")

    scope = _mapping(config.get("authorized_rebuild_scope"), "scope")
    require(scope.get("selected_episode_rows") == 4, "scope rows")
    require(scope.get("exact_field_candidate_ranks") == {
        "computer_science": [1, 45],
        "medicine": [5, 6],
    }, "scope ranks")
    require(scope.get("outcome_blind") is True, "outcome blind")
    require(scope.get("candidate_substitution") == "FORBIDDEN", "candidate substitution")
    require(scope.get("field_substitution") == "FORBIDDEN", "field substitution")
    require(scope.get("measurement_abi_amendment") == "FORBIDDEN", "ABI amendment")
    require(scope.get("entry_or_persistence_outcome_read") == "FORBIDDEN", "outcome read")
    require(scope.get("current_old_panel_role") == "SUPERSEDED_PROVISIONAL_DO_NOT_REUSE", "old panel")

    authorization = _mapping(config.get("authorization"), "authorization")
    require(
        authorization.get("d0_exact_selected_work_manifest_rebuild")
        == "AUTHORIZED_OUTCOME_BLIND_EXACT_FOUR_ONLY",
        "work manifest",
    )
    require(
        authorization.get("d0_author_opportunity_panel_rebuild")
        == "AUTHORIZED_OUTCOME_BLIND_EXACT_FOUR_ONLY",
        "panel rebuild",
    )
    require(
        authorization.get("d0_leaveout_network_bridge_build")
        == "PAUSED_UNTIL_REBUILT_PANEL_INDEPENDENTLY_VERIFIED",
        "leaveout boundary",
    )
    for key in (
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "d0_new_api_request_creation",
        "d0_paid_api_submission",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return {
        "schema_version": "pivot_feedback_gate_state_v37_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(path),
        "certified_episode_reselection_complete": True,
        "selected_episode_rows": 4,
        "selected_field_candidate_ranks": {
            "computer_science": [1, 45],
            "medicine": [5, 6],
        },
        "exact_selected_work_manifest_rebuild_authorized": True,
        "author_opportunity_panel_rebuild_authorized": True,
        "leaveout_network_bridge_build_authorized": False,
        "production_pivot_classification_authorized": False,
        "outcomes_read": False,
        "g4_open": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(verify(args.config), indent=2, sort_keys=True))
        return 0
    except (
        GateStateV37VerificationError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V37", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
