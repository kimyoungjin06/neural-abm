#!/usr/bin/env python3
"""Verify the exact deterministic-reselection successor gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v36.yaml"
STATUS = "PASS_PIVOT_FEEDBACK_GATE_STATE_V36"


class GateStateV36VerificationError(RuntimeError):
    """Fail-closed v36 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV36VerificationError(message)


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
    require(config.get("schema_version") == "pivot_feedback_gate_state.v36", "schema")
    require(
        config.get("status")
        == "VALID_GLOBAL_SELECTION_CERTIFICATE_EXACT_DETERMINISTIC_RESELECTION_SUCCESSOR_AUTHORIZED",
        "status",
    )
    parent = _mapping(config.get("additive_parent"), "parent")
    parent_doc = _mapping(yaml.safe_load(_repo(parent).read_text(encoding="utf-8")), "parent doc")
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    closeout = _mapping(config.get("verified_closeout"), "closeout")
    _repo(_mapping(closeout.get("config"), "closeout config"))
    _repo(_mapping(closeout.get("producer"), "producer"))
    _repo(_mapping(closeout.get("verifier"), "verifier"))
    operator = _mapping(closeout.get("operator_local"), "operator")
    root = (Path("/ssd") / str(operator["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "operator root")
    for basename_key, sha_key in (
        ("dispositions_basename", "dispositions_sha256"),
        ("frontier_basename", "frontier_sha256"),
        ("certificate_basename", "certificate_sha256"),
        ("receipt_basename", "receipt_sha256"),
    ):
        artifact = root / str(operator[basename_key])
        require(_sha256(artifact) == operator[sha_key], f"artifact SHA: {artifact.name}")
    verification = (Path("/ssd") / str(operator["verification_basename"])).resolve(strict=True)
    require(verification.parent == Path("/ssd").resolve(), "verification parent")
    require(_sha256(verification) == operator["verification_sha256"], "verification SHA")
    require(_json(root / str(operator["receipt_basename"])).get("status") == operator["receipt_status"], "receipt status")
    require(_json(verification).get("status") == operator["verification_status"], "verification status")
    certificate = _json(root / str(operator["certificate_basename"]))
    require(certificate.get("global_selection_certificate_issued") is True, "global certificate")
    require(certificate.get("computer_science_selected_candidate_ranks") == [1, 45], "CS ranks")
    require(certificate.get("medicine_selected_candidate_ranks") == [5, 6], "Medicine ranks")
    require(certificate.get("deterministic_reselection_executed") is False, "reselection execution")

    frontier = pq.read_table(root / str(operator["frontier_basename"])).to_pylist()
    scope = _mapping(config.get("exact_reselection_scope"), "scope")
    episodes = scope.get("episodes")
    require(isinstance(episodes, list) and len(episodes) == scope["episode_rows"] == 4, "episode rows")
    actual: list[dict[str, int]] = []
    for episode_value in episodes:
        episode = _mapping(episode_value, "episode")
        matches = [
            row
            for row in frontier
            if row["field_id"] == episode["field_id"]
            and row["field_candidate_rank"] == episode["field_candidate_rank"]
        ]
        require(len(matches) == 1, "frontier episode identity")
        row = matches[0]
        require(row["content_state"] == "CONTENT_PASS", "episode content pass")
        for key in ("topic_id", "entry_year", "subfield_id"):
            require(row[key] == episode[key], f"episode binding: {key}")
        actual.append(
            {
                key: int(episode[key])
                for key in (
                    "field_id",
                    "field_candidate_rank",
                    "topic_id",
                    "entry_year",
                    "subfield_id",
                )
            }
        )
    require(
        [(row["field_id"], row["field_candidate_rank"]) for row in actual]
        == [(17, 1), (17, 45), (27, 5), (27, 6)],
        "exact ranks",
    )
    require(scope.get("candidate_substitution") == "FORBIDDEN", "substitution")
    require(scope.get("content_reinterpretation") == "FORBIDDEN", "reinterpretation")
    require(scope.get("outcome_access_during_reselection") == "FORBIDDEN", "outcomes")
    require(scope.get("deterministic_reselection_executed") is False, "execution")
    authorization = _mapping(config.get("authorization"), "authorization")
    require(
        authorization.get("d0_exact_four_episode_deterministic_reselection_successor")
        == "AUTHORIZED",
        "reselection authorization",
    )
    require(
        authorization.get("d0_author_opportunity_panel_rebuild")
        == "NOT_AUTHORIZED_UNTIL_RESELECTION_INDEPENDENTLY_VERIFIED",
        "panel rebuild",
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
        "schema_version": "pivot_feedback_gate_state_v36_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(path),
        "global_selection_certificate_verified": True,
        "authorized_episode_rows": 4,
        "authorized_field_candidate_ranks": {
            "computer_science": [1, 45],
            "medicine": [5, 6],
        },
        "deterministic_reselection_successor_authorized": True,
        "deterministic_reselection_executed": False,
        "author_opportunity_panel_rebuild_authorized": False,
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
    except (GateStateV36VerificationError, OSError, ValueError, TypeError, KeyError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V36", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
