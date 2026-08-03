#!/usr/bin/env python3
"""Verify the additive D0-B-author-foundation gate ledger v5."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v5"
LEDGER_STATUS = (
    "VALID_G_MINUS_1A_S0_D0A_AND_D0B_AUTHOR_FOUNDATION_"
    "REMAINING_D0B_MEASUREMENTS_AUTHORIZED"
)
VERIFY_STATUS = (
    "PASS_G_MINUS_1A_S0_D0A_D0B_AUTHOR_FOUNDATION_"
    "REMAINING_D0B_ONLY_AUTHORIZED"
)
SUMMARY_STATUS = (
    "PASS_INDEPENDENT_D0B_AUTHOR_OPPORTUNITY_FOUNDATION_D0_AND_D0B_"
    "PARTIAL_REMAINING_MEASUREMENTS_ONLY_AUTHORIZED"
)
OUTPUT_STATUS = "PASS_D0B_OUTCOME_BLIND_PRE_ENTRY_AUTHOR_OPPORTUNITY_FOUNDATION"
OUTPUT_VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0B_OUTCOME_BLIND_PRE_ENTRY_AUTHOR_OPPORTUNITY_"
    "FOUNDATION_D0B_AND_D0_STILL_PARTIAL"
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
FROZEN_EPISODES = [
    (17, 14430, 2003),
    (17, 10579, 2012),
    (27, 11361, 2007),
    (27, 12505, 1991),
]
REQUIRED_COMPONENTS = [
    "current_snapshot_topic_family_content_coherence_audit",
    "pre_event_indexing_discontinuity_diagnostic",
    "incumbent_membership_frozen_before_signal_window",
    "incumbent_only_attention_available_by_t_minus_1",
    "focal_author_leaveout",
    "registered_local_network_leaveout",
    "prior_peer_activity_available_by_t_minus_1",
    "recent_collaboration_existing_by_t_minus_1",
    "orcid_stable_subset_flag",
    "same_affiliation_existing_by_t_minus_1",
    "independent_temporal_leakage_verification",
]


class GateStateV5Error(RuntimeError):
    """Fail-closed gate-state v5 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV5Error(message)


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
    require(isinstance(value, str) and value, f"{label} must be text")
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
    require(path.is_file() and not path.is_symlink(), f"invalid document: {path}")
    try:
        value = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_document
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise GateStateV5Error(f"invalid document {path.name}: {error}") from error
    return _mapping(value, path.name)


def _validate_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    yaml_document: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    item = dict(raw)
    path = _safe_path(repo_root, item.get("relative_path"), "bound evidence")
    digest = _sha256(path)
    require(digest == item.get("sha256"), f"bound SHA mismatch: {path.name}")
    document = _read(path, yaml_document=yaml_document)
    require(
        document.get("schema_version") == item.get("required_schema_version"),
        "bound schema mismatch",
    )
    require(document.get("status") == item.get("required_status"), "bound status mismatch")
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
        "sha256": digest,
        "status": document["status"],
    }


def _summary_episodes(summary: Mapping[str, Any]) -> list[tuple[int, int, int]]:
    rows = summary.get("selected_work_measurement", {}).get("by_episode")
    require(isinstance(rows, list), "summary episodes are not a list")
    return [
        (int(row["field_id"]), int(row["topic_id"]), int(row["entry_year"]))
        for row in rows
    ]


def _validate_live_foundation(
    summary: Mapping[str, Any],
    *,
    repo_root: Path,
    output_root: Path | None,
    verification_report: Path | None,
) -> dict[str, Any]:
    binding = _mapping(summary.get("output_binding"), "output binding")
    root = (
        Path("/ssd") / str(binding["output_root_basename"])
        if output_root is None
        else output_root.resolve(strict=True)
    )
    require(root.is_dir() and not root.is_symlink(), "D0-B output root invalid")
    require(root.name == binding.get("output_root_basename"), "output basename mismatch")
    receipt_path = root / "d0b_author_opportunity_foundation.outputs.json"
    receipt_sha = _sha256(receipt_path)
    require(receipt_sha == binding.get("output_receipt_sha256"), "output receipt SHA mismatch")
    receipt = _read(receipt_path, yaml_document=False)
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0b_author_opportunity_foundation_outputs.v1",
        "output receipt schema mismatch",
    )
    require(receipt.get("status") == OUTPUT_STATUS, "output status mismatch")
    require(receipt.get("d0b_author_foundation_built") is True, "foundation absent")
    for key in (
        "d0b_complete",
        "d0_complete",
        "discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe output flag: {key}")
    require(
        receipt.get("artifact_tree_sha256") == binding.get("artifact_tree_sha256"),
        "artifact tree mismatch",
    )

    report_path = (
        Path("/ssd") / str(binding["verification_report_basename"])
        if verification_report is None
        else verification_report.resolve(strict=True)
    )
    report_sha = _sha256(report_path)
    require(
        report_sha == binding.get("verification_report_sha256"),
        "verification report SHA mismatch",
    )
    report = _read(report_path, yaml_document=False)
    require(
        report.get("schema_version")
        == "pivot_feedback_d0b_author_opportunity_foundation_verification.v1",
        "verification schema mismatch",
    )
    require(report.get("status") == OUTPUT_VERIFY_STATUS, "verification status mismatch")
    replay = _mapping(report.get("replay"), "verification replay")
    require(replay.get("source_rows_replayed_total") == 1_271_923_464, "replay rows differ")
    require(replay.get("producer_module_imported") is False, "producer imported")
    require(replay.get("forbidden_columns_absent") is True, "forbidden columns present")
    require(
        replay.get("selected_works", {}).get("exact_output_match") is True,
        "selected works do not match",
    )
    require(
        replay.get("author_measurement_base", {}).get("exact_output_match") is True,
        "author output does not match",
    )
    scope = _mapping(report.get("scope"), "verification scope")
    require(
        scope.get("d0b_author_foundation_independently_verified") is True,
        "foundation not independently verified",
    )
    for key in (
        "d0b_complete",
        "d0_complete",
        "discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(scope.get(key) is False, f"unsafe verification scope: {key}")

    execution = _mapping(summary.get("execution"), "summary execution")
    for key in ("builder", "verifier", "config"):
        path = _safe_path(
            repo_root,
            execution.get(f"{key}_relative_path"),
            f"{key} path",
        )
        require(
            _sha256(path) == execution.get(f"{key}_sha256"),
            f"{key} SHA mismatch",
        )
    require(
        report.get("verifier", {}).get("sha256") == execution["verifier_sha256"],
        "live verifier hash differs",
    )
    return {
        "output_root_basename": root.name,
        "output_receipt_sha256": receipt_sha,
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "verification_report_basename": report_path.name,
        "verification_report_sha256": report_sha,
        "verification_status": report["status"],
        "source_rows_replayed_total": replay["source_rows_replayed_total"],
        "builder_sha256": execution["builder_sha256"],
        "verifier_sha256": execution["verifier_sha256"],
    }


def validate(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path | None = None,
    verification_report: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    path = path.resolve(strict=True)
    ledger = _read(path, yaml_document=True)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema mismatch")
    require(ledger.get("label") == "pivot_feedback_gate_state_v5", "ledger label mismatch")
    require(
        ledger.get("mode")
        == "ADDITIVE_D0B_AUTHOR_OPPORTUNITY_FOUNDATION_GATE_LEDGER",
        "ledger mode mismatch",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status mismatch")

    parent_raw = _mapping(ledger.get("additive_parent"), "additive parent")
    require(parent_raw.get("mutation_allowed") is False, "v4 mutation allowed")
    parent, parent_result = _validate_binding(
        parent_raw,
        repo_root=repo_root,
        yaml_document=True,
    )
    parent_states = _mapping(parent.get("gate_state"), "parent states")
    require(list(parent_states) == EXPECTED_ORDER, "parent gate order mismatch")
    require(
        parent_states["D0"].get("state") == "PARTIAL_D0A_STRUCTURAL_FREEZE_PASS",
        "parent D0 state differs",
    )

    evidence = _mapping(ledger.get("execution_evidence"), "execution evidence")
    require(
        set(evidence) == {"d0b_author_opportunity_foundation"},
        "execution evidence set mismatch",
    )
    summary, summary_result = _validate_binding(
        _mapping(
            evidence["d0b_author_opportunity_foundation"],
            "D0-B foundation evidence",
        ),
        repo_root=repo_root,
        yaml_document=False,
    )
    require(summary.get("status") == SUMMARY_STATUS, "summary status mismatch")
    require(_summary_episodes(summary) == FROZEN_EPISODES, "summary episodes differ")
    live = _validate_live_foundation(
        summary,
        repo_root=repo_root,
        output_root=output_root,
        verification_report=verification_report,
    )

    states = _mapping(ledger.get("gate_state"), "gate state")
    require(list(states) == EXPECTED_ORDER, "gate order mismatch")
    require(
        states["G_MINUS_1A"]
        == {"state": "PASS_BOUNDED", "complete": True, "authorizes": ["S0"]},
        "G-1A state differs",
    )
    require(
        states["S0"]
        == {
            "state": "PASS_THRESHOLD_FREE_INVENTORY",
            "complete": True,
            "authorizes": ["D0"],
        },
        "S0 state differs",
    )
    require(
        states["D0"]
        == {
            "state": "PARTIAL_D0A_AND_D0B_AUTHOR_FOUNDATION_PASS",
            "complete": False,
            "requires": ["S0"],
            "entry_year_end": 2012,
            "completed_components": [
                "D0A_OUTCOME_BLIND_FIELD_AND_EPISODE_FREEZE",
                "D0B_PRE_ENTRY_AUTHOR_OPPORTUNITY_FOUNDATION",
            ],
            "authorizes": ["D0B_REMAINING_OUTCOME_BLIND_MEASUREMENTS"],
        },
        "D0 state differs",
    )
    require(states["G0"] == {"state": "PENDING", "complete": False}, "G0 differs")
    require(
        states["G4"] == {"state": "SEALED_NOT_OPENED", "complete": False},
        "G4 is not sealed",
    )

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(authorization.get("d0a_structural_freeze") == "COMPLETE", "D0-A differs")
    require(
        authorization.get("d0b_author_opportunity_foundation") == "COMPLETE",
        "foundation differs",
    )
    require(
        authorization.get("d0b_remaining_outcome_blind_measurements")
        == "AUTHORIZED_FOUR_FROZEN_EPISODES_ONLY",
        "remaining D0-B authorization differs",
    )
    require(
        authorization.get("d0_discovery_outcome_opening") == "NOT_YET_AUTHORIZED",
        "D0 outcomes opened",
    )
    for key in (
        "g0_measurement_freeze",
        "final_panel_materialization",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect overstated",
    )

    frozen = _mapping(ledger.get("frozen_scope"), "frozen scope")
    require(frozen.get("field_ids") == [17, 27], "field IDs differ")
    require(
        [tuple(row) for row in frozen["episode_topic_years"]] == FROZEN_EPISODES,
        "frozen episodes differ",
    )
    require(frozen.get("final_g0_selection") is False, "scope promoted to G0")
    foundation = _mapping(ledger.get("d0b_author_foundation"), "foundation")
    require(foundation.get("selected_predecision_work_rows") == 83_565, "work rows differ")
    require(foundation.get("author_episode_rows") == 5_849_627, "author rows differ")
    require(foundation.get("team_cap_selected") is False, "team cap selected")
    require(
        foundation.get("final_candidate_eligibility_established") is False,
        "candidate eligibility claimed",
    )
    require(
        foundation.get("pre_signal_incumbent_and_field_active_rows_by_episode")
        == [182, 130, 3368, 5449],
        "incumbent intersection differs",
    )

    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(next_step.get("gate") == "D0B_REMAINING", "next gate differs")
    require(
        [tuple(row) for row in next_step["allowed_episode_topic_years"]]
        == FROZEN_EPISODES,
        "next-step episode scope differs",
    )
    require(next_step.get("required_components") == REQUIRED_COMPONENTS, "required components differ")
    require(
        next_step.get("same_affiliation_dependency")
        == "NEW_RECEIPT_BOUND_AUTHOR_WORK_INSTITUTION_SURFACE_REQUIRED",
        "affiliation dependency differs",
    )
    for key in (
        "entry_or_persistence_outcomes_may_be_opened",
        "threshold_or_team_cap_selection_allowed",
        "final_candidate_eligibility_may_be_claimed",
        "final_field_choice_by_effect_size_allowed",
        "confirmatory_year_changes_allowed",
        "gpu_required",
    ):
        require(next_step.get(key) is False, f"unsafe next-step state: {key}")
    require("D0-B and D0 remain incomplete" in str(ledger.get("claim_boundary")), "partial boundary absent")
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v5",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(path),
        "additive_parent": parent_result,
        "d0b_author_foundation_summary": summary_result,
        "live_d0b_author_foundation": live,
        "gate_order": EXPECTED_ORDER,
        "d0a_complete": True,
        "d0b_author_opportunity_foundation_complete": True,
        "d0b_complete": False,
        "d0_complete": False,
        "d0b_remaining_outcome_blind_measurements_authorized": True,
        "d0_discovery_outcome_opening_authorized": False,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v5.yaml",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--verification-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            args.ledger,
            output_root=args.output_root,
            verification_report=args.verification_report,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except (GateStateV5Error, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V5",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
