#!/usr/bin/env python3
"""Verify the exact-four author-panel leaveout/network successor gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v38.yaml"
STATUS = "PASS_PIVOT_FEEDBACK_GATE_STATE_V38"


class GateStateV38VerificationError(RuntimeError):
    """Fail-closed v38 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV38VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing field {dotted_path}",
        )
        current = current[part]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(path.is_file() and not path.is_symlink(), "invalid repository binding")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def verify(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    path = config_path.resolve(strict=True)
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), "config")
    require(config.get("schema_version") == "pivot_feedback_gate_state.v38", "schema")
    require(
        config.get("status")
        == "VALID_EXACT_FOUR_AUTHOR_OPPORTUNITY_REBUILD_INDEPENDENTLY_VERIFIED_LEAVEOUT_NETWORK_BRIDGE_AUTHORIZED",
        "status",
    )
    parent = _mapping(config.get("additive_parent"), "parent")
    parent_doc = _mapping(
        yaml.safe_load(_repo(parent).read_text(encoding="utf-8")),
        "parent document",
    )
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")
    for dotted_path, expected in _mapping(
        parent.get("required_fields", {}),
        "parent required fields",
    ).items():
        observed = _nested(parent_doc, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"parent field: {dotted_path}",
        )

    rebuild = _mapping(config.get("verified_rebuild"), "verified rebuild")
    for label in ("config", "producer", "verifier", "summary_markdown"):
        _repo(_mapping(rebuild.get(label), label))
    summary_binding = _mapping(rebuild.get("summary_json"), "summary JSON")
    summary = _json(_repo(summary_binding))
    require(
        summary.get("schema_version") == summary_binding["required_schema_version"],
        "summary schema",
    )
    require(summary.get("status") == summary_binding["required_status"], "summary status")

    operator = _mapping(rebuild.get("operator_local"), "operator local")
    root = (Path("/ssd") / str(operator["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "operator root parent")
    require(root.is_dir() and not root.is_symlink(), "operator root")
    for basename_key, sha_key in (
        ("receipt_basename", "receipt_sha256"),
        ("report_basename", "report_sha256"),
        ("selected_work_basename", "selected_work_sha256"),
        ("author_base_basename", "author_base_sha256"),
    ):
        artifact = root / str(operator[basename_key])
        require(_sha256(artifact) == operator[sha_key], f"operator SHA: {artifact.name}")
    verification_path = (Path("/ssd") / str(operator["verification_basename"])).resolve(strict=True)
    require(verification_path.parent == Path("/ssd").resolve(), "verification parent")
    require(_sha256(verification_path) == operator["verification_sha256"], "verification SHA")
    receipt = _json(root / str(operator["receipt_basename"]))
    report = _json(root / str(operator["report_basename"]))
    verification = _json(verification_path)
    require(receipt.get("status") == operator["receipt_status"], "receipt status")
    require(report.get("status") == operator["receipt_status"], "report status")
    require(verification.get("status") == operator["verification_status"], "verification status")
    require(receipt.get("outcomes_read") is False, "receipt outcomes")
    require(report["scope"].get("outcomes_read") is False, "report outcomes")
    require(verification["scope"].get("outcomes_read") is False, "verification outcomes")
    require(verification["scope"].get("producer_imported") is False, "verifier independence")
    require(verification["scope"].get("independent_replay_complete") is True, "replay incomplete")

    selected = pq.ParquetFile(root / str(operator["selected_work_basename"]))
    authors = pq.ParquetFile(root / str(operator["author_base_basename"]))
    require(selected.metadata.num_rows == 30765, "selected work rows")
    require(authors.metadata.num_rows == 3256271, "author rows")
    require(authors.metadata.num_row_groups == 16384, "author row groups")
    replay = _mapping(verification.get("replay_accounting"), "replay")
    require(replay.get("selected_work_rows") == selected.metadata.num_rows, "replay selected rows")
    require(replay.get("author_rows") == authors.metadata.num_rows, "replay author rows")
    require(replay.get("author_row_groups_compared") == authors.metadata.num_row_groups, "replay row groups")
    require(replay.get("cutoff_work_rows_scanned") == 492293206, "work replay rows")
    require(replay.get("author_event_rows_scanned") == 779630258, "event replay rows")

    result = _mapping(rebuild.get("result"), "result")
    require(result.get("exact_certified_episode_rows") == 4, "episode rows")
    require(result.get("selected_work_rows") == selected.metadata.num_rows, "result selected rows")
    require(result.get("author_opportunity_rows") == authors.metadata.num_rows, "result author rows")
    require(result.get("author_row_groups_independently_compared") == 16384, "result row groups")
    require(result.get("duplicate_author_work_event_rows") == 0, "duplicate events")
    require(result.get("selected_work_rows_by_episode") == [15099, 8068, 3797, 3801], "episode work rows")
    require(result.get("author_rows_by_episode") == [549629, 1215043, 1330658, 160941], "episode author rows")
    require(result.get("outcomes_read") is False, "result outcomes")
    require(result.get("api_requests_created") == 0, "result API requests")

    surface = _mapping(config.get("current_measurement_surface"), "surface")
    require(
        surface.get("exact_episode_rows")
        == [
            [17, 1, 14430, 2003],
            [17, 45, 10579, 2012],
            [27, 5, 10151, 1993],
            [27, 6, 12421, 1953],
        ],
        "surface episodes",
    )
    require(
        surface.get("author_opportunity_base_role")
        == "CURRENT_INDEPENDENTLY_VERIFIED_EXACT_FOUR_OUTCOME_BLIND_MEASUREMENT_BASE",
        "current author base",
    )
    require(
        surface.get("old_author_panel_role")
        == "SUPERSEDED_PROVISIONAL_DO_NOT_REUSE",
        "old panel role",
    )
    require(surface.get("team_cap_selected") is False, "team cap selected")
    require(surface.get("final_minimum_history_works") == "UNRESOLVED", "history threshold")
    require(surface.get("final_candidate_eligibility_established") is False, "eligibility")

    next_scope = _mapping(config.get("authorized_next_scope"), "next scope")
    require(next_scope.get("exact_certified_episode_rows") == 4, "next episode rows")
    require(next_scope.get("outcome_blind") is True, "next outcome blind")
    require(next_scope.get("predecision_only") is True, "next predecision")
    require(next_scope.get("author_membership_rule_may_use_outcomes") is False, "membership outcomes")
    require(next_scope.get("final_minimum_history_threshold_selection") == "NOT_AUTHORIZED", "history selection")
    require(next_scope.get("team_cap_selection") == "NOT_AUTHORIZED", "team cap selection")
    require(next_scope.get("episode_or_field_substitution") == "FORBIDDEN", "substitution")
    require(next_scope.get("measurement_abi_amendment") == "FORBIDDEN", "ABI amendment")
    for key in (
        "focal_author_leaveout",
        "incumbent_only_target_attention",
        "prior_collaborator_graph",
        "registered_local_network_leaveout",
        "prior_peer_target_activity",
    ):
        require(next_scope.get(key) == "AUTHORIZED", f"next component: {key}")
    require(next_scope.get("temporal_leakage_independent_verification") == "REQUIRED", "temporal verifier")

    authorization = _mapping(config.get("authorization"), "authorization")
    require(
        authorization.get("d0_exact_selected_work_manifest_rebuild")
        == "COMPLETE_INDEPENDENTLY_VERIFIED",
        "work manifest completion",
    )
    require(
        authorization.get("d0_author_opportunity_panel_rebuild")
        == "COMPLETE_INDEPENDENTLY_VERIFIED",
        "author panel completion",
    )
    require(
        authorization.get("d0_leaveout_network_bridge_contract_and_build")
        == "AUTHORIZED_OUTCOME_BLIND_EXACT_FOUR_ONLY",
        "leaveout authorization",
    )
    require(
        authorization.get("d0_leaveout_network_bridge_independent_verification")
        == "REQUIRED",
        "leaveout verification",
    )
    require(
        authorization.get("d0_production_pivot_classification")
        == "PAUSED_UNTIL_LEAVEOUT_NETWORK_BRIDGE_INDEPENDENTLY_VERIFIED",
        "pivot boundary",
    )
    for key in (
        "d0_discovery_outcome_opening",
        "d0_new_api_request_creation",
        "d0_paid_api_submission",
        "final_candidate_eligibility",
        "g0_measurement_freeze",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "scientific effect")
    return {
        "schema_version": "pivot_feedback_gate_state_v38_verification.v1",
        "status": STATUS,
        "config_sha256": _sha256(path),
        "author_opportunity_rebuild_complete": True,
        "independent_replay_complete": True,
        "selected_work_rows": selected.metadata.num_rows,
        "author_opportunity_rows": authors.metadata.num_rows,
        "leaveout_network_bridge_build_authorized": True,
        "production_pivot_classification_authorized": False,
        "discovery_outcome_opening_authorized": False,
        "outcomes_read": False,
        "api_requests_created": 0,
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
        GateStateV38VerificationError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V38",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
