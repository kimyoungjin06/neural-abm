#!/usr/bin/env python3
"""Verify the additive D0 linkage-review gate ledger v6."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v6"
LEDGER_STATUS = (
    "VALID_D0_MEASUREMENT_REVIEW_REQUIRED_LEAVEOUT_AND_OUTCOMES_FAIL_CLOSED"
)
VERIFY_STATUS = (
    "PASS_D0_LINKAGE_GAP_REVIEW_ABI_REDESIGN_ONLY_AUTHORIZED_"
    "LEAVEOUT_AND_OUTCOMES_FAIL_CLOSED"
)
SUMMARY_STATUS = (
    "PASS_INDEPENDENT_D0B_LINKAGE_AUDIT_MEASUREMENT_REVIEW_REQUIRED_"
    "LEAVEOUT_BUILD_PAUSED"
)
OUTPUT_STATUS = (
    "PASS_D0B_OUTCOME_BLIND_SELECTED_WORK_AUTHOR_LINKAGE_AUDIT_"
    "NO_EPISODE_DECISION"
)
OUTPUT_VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0B_SELECTED_WORK_AUTHOR_LINKAGE_AUDIT_"
    "SEVERE_HETEROGENEOUS_LINKAGE_GAP_REQUIRES_MEASUREMENT_REVIEW"
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
PROVISIONAL_EPISODES = [
    (17, 14430, 2003),
    (17, 10579, 2012),
    (27, 11361, 2007),
    (27, 12505, 1991),
]
REQUIRED_ORDER = [
    "freeze_scholarly_work_type_eligibility_without_outcomes",
    "freeze_is_paratext_and_retraction_treatment_without_outcomes",
    "freeze_valid_author_token_linkage_requirement_and_missingness_reporting",
    "rebuild_all_field_topic_year_attention_candidates_under_the_new_abi",
    "rerun_deterministic_two_field_four_episode_selection_or_fail_coverage",
    "audit_content_coherence_and_indexing_discontinuity_before_leaveouts",
    "independently_replay_the_successor_freeze",
]


class GateStateV6Error(RuntimeError):
    """Fail-closed gate-state v6 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV6Error(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing {dotted_path}",
        )
        current = current[part]
    return current


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


def _read(path: Path, *, yaml_document: bool) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid document: {path}")
    try:
        value = (
            yaml.safe_load(path.read_text(encoding="utf-8"))
            if yaml_document
            else json.loads(path.read_text(encoding="utf-8"))
        )
    except (UnicodeDecodeError, yaml.YAMLError, json.JSONDecodeError) as error:
        raise GateStateV6Error(f"invalid document {path.name}: {error}") from error
    return _mapping(value, path.name)


def _validate_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    yaml_document: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = dict(raw)
    path = _safe_path(repo_root, binding.get("relative_path"), "bound evidence")
    digest = _sha256(path)
    require(digest == binding.get("sha256"), f"bound SHA mismatch: {path.name}")
    document = _read(path, yaml_document=yaml_document)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        "bound schema mismatch",
    )
    require(
        document.get("status") == binding.get("required_status"),
        "bound status mismatch",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        "required fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"bound field mismatch: {dotted_path}",
        )
    return document, {
        "relative_path": binding["relative_path"],
        "sha256": digest,
        "status": document["status"],
    }


def _validate_live_linkage(
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
    require(root.is_dir() and not root.is_symlink(), "linkage output root invalid")
    require(root.name == binding.get("output_root_basename"), "output basename mismatch")
    receipt_path = root / "selected_work_author_linkage_audit.outputs.json"
    receipt_sha = _sha256(receipt_path)
    require(receipt_sha == binding.get("output_receipt_sha256"), "output receipt SHA mismatch")
    receipt = _read(receipt_path, yaml_document=False)
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0b_selected_work_author_linkage_audit_outputs.v1",
        "output receipt schema mismatch",
    )
    require(receipt.get("status") == OUTPUT_STATUS, "output status mismatch")
    require(receipt.get("selected_work_author_linkage_audited") is True, "linkage audit absent")
    for key in (
        "episode_exclusion_decision_made",
        "linkage_threshold_selected",
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
        == "pivot_feedback_d0b_selected_work_author_linkage_audit_verification.v1",
        "verification schema mismatch",
    )
    require(report.get("status") == OUTPUT_VERIFY_STATUS, "verification status mismatch")
    replay = _mapping(report.get("replay"), "verification replay")
    require(replay.get("source_rows_replayed_total") == 1_271_923_464, "replay rows differ")
    for key in (
        "work_metadata_exact_match",
        "author_incidence_exact_match",
        "work_linkage_count_exact_match",
        "profile_exact_match",
        "forbidden_columns_absent",
    ):
        require(replay.get(key) is True, f"replay check failed: {key}")
    require(replay.get("producer_module_imported") is False, "producer imported")
    critical = _mapping(report.get("critical_readout"), "critical readout")
    require(critical.get("episode_3_signal_selected_work_rows") == 23_073, "signal rows differ")
    require(critical.get("episode_3_signal_linked_work_rows") == 1_790, "signal links differ")
    require(
        critical.get("episode_3_signal_zero_valid_author_token_work_rows")
        == 21_283,
        "signal zero-token rows differ",
    )
    require(critical.get("episode_3_other_type_selected_work_rows") == 22_451, "other rows differ")
    require(critical.get("episode_3_other_type_linked_work_rows") == 14, "other links differ")
    require(
        critical.get("interpretation")
        == "SEVERE_HETEROGENEOUS_WORK_TYPE_AND_AUTHOR_LINKAGE_GAP_BLOCKS_UNQUALIFIED_INCUMBENT_ATTENTION_USE",
        "critical interpretation differs",
    )
    scope = _mapping(report.get("scope"), "verification scope")
    require(scope.get("linkage_audit_independently_verified") is True, "audit unverified")
    require(scope.get("measurement_review_required") is True, "review not required")
    for key in (
        "episode_exclusion_decision_made",
        "threshold_selected",
        "incumbent_attention_validated",
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
    for label in ("builder", "verifier", "config"):
        path = _safe_path(
            repo_root,
            execution.get(f"{label}_relative_path"),
            f"{label} path",
        )
        require(
            _sha256(path) == execution.get(f"{label}_sha256"),
            f"{label} SHA mismatch",
        )
    require(
        report.get("verifier", {}).get("sha256") == execution["verifier_sha256"],
        "live verifier SHA differs",
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
    require(ledger.get("label") == "pivot_feedback_gate_state_v6", "ledger label mismatch")
    require(
        ledger.get("mode")
        == "ADDITIVE_D0B_LINKAGE_AUDIT_MEASUREMENT_REVIEW_GATE_LEDGER",
        "ledger mode mismatch",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status mismatch")

    parent_raw = _mapping(ledger.get("additive_parent"), "additive parent")
    require(parent_raw.get("mutation_allowed") is False, "v5 mutation allowed")
    parent, parent_result = _validate_binding(
        parent_raw,
        repo_root=repo_root,
        yaml_document=True,
    )
    require(
        list(_mapping(parent.get("gate_state"), "parent states"))
        == EXPECTED_ORDER,
        "parent gate order differs",
    )

    evidence = _mapping(ledger.get("execution_evidence"), "execution evidence")
    require(
        set(evidence) == {"selected_work_author_linkage_audit"},
        "execution evidence set differs",
    )
    summary, summary_result = _validate_binding(
        _mapping(
            evidence["selected_work_author_linkage_audit"],
            "linkage summary",
        ),
        repo_root=repo_root,
        yaml_document=False,
    )
    require(summary.get("status") == SUMMARY_STATUS, "summary status mismatch")
    live = _validate_live_linkage(
        summary,
        repo_root=repo_root,
        output_root=output_root,
        verification_report=verification_report,
    )

    states = _mapping(ledger.get("gate_state"), "gate states")
    require(list(states) == EXPECTED_ORDER, "gate order differs")
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
    d0 = _mapping(states["D0"], "D0 state")
    require(
        d0.get("state")
        == "PARTIAL_MEASUREMENT_REVIEW_REQUIRED_AFTER_LINKAGE_AUDIT",
        "D0 state differs",
    )
    require(d0.get("complete") is False, "D0 improperly complete")
    require(
        d0.get("authorizes")
        == [
            "D0_OUTCOME_BLIND_MEASUREMENT_ABI_REDESIGN",
            "D0_OUTCOME_BLIND_EPISODE_RESELECTION_SUCCESSOR",
        ],
        "D0 authorization differs",
    )
    require(
        d0.get("blocked_components")
        == [
            "INCUMBENT_ONLY_ATTENTION",
            "FOCAL_AUTHOR_LEAVEOUT",
            "REGISTERED_LOCAL_NETWORK_LEAVEOUT",
            "PRIOR_PEER_ACTIVITY",
            "RECENT_COLLABORATION_BRIDGE",
            "SAME_AFFILIATION_BRIDGE",
        ],
        "blocked component set differs",
    )
    require(states["G0"] == {"state": "PENDING", "complete": False}, "G0 differs")
    require(
        states["G4"] == {"state": "SEALED_NOT_OPENED", "complete": False},
        "G4 is not sealed",
    )

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(
        authorization.get("d0_measurement_abi_redesign")
        == "AUTHORIZED_OUTCOME_BLIND",
        "ABI redesign not authorized",
    )
    require(
        authorization.get("d0_episode_reselection_successor")
        == "AUTHORIZED_AFTER_ABI_FREEZE",
        "reselection authorization differs",
    )
    require(
        authorization.get("d0b_leaveout_network_bridge_build")
        == "PAUSED_PENDING_ABI_AND_RESELECTION",
        "leaveout build not paused",
    )
    require(
        authorization.get("d0_discovery_outcome_opening") == "NOT_AUTHORIZED",
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

    diagnosis = _mapping(ledger.get("measurement_diagnosis"), "measurement diagnosis")
    require(diagnosis.get("selected_work_rows") == 83_565, "selected rows differ")
    require(diagnosis.get("linked_work_rows") == 48_245, "linked rows differ")
    require(diagnosis.get("zero_valid_author_token_work_rows") == 35_320, "zero rows differ")
    require(diagnosis.get("heterogeneous_linkage_gap") is True, "gap absent")
    decisive = _mapping(diagnosis.get("decisive_episode"), "decisive episode")
    require(
        (
            decisive.get("field_id"),
            decisive.get("topic_id"),
            decisive.get("entry_year"),
        )
        == (27, 11361, 2007),
        "decisive episode differs",
    )
    for key in (
        "current_all_work_attention_valid_for_unqualified_author_level_use",
        "current_four_episode_freeze_valid_for_leaveout_build",
        "episode_exclusion_decision_made",
        "eligible_work_types_selected",
        "valid_author_linkage_rule_selected",
    ):
        require(diagnosis.get(key) is False, f"unsafe diagnosis state: {key}")

    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(next_step.get("gate") == "D0_MEASUREMENT_ABI_REDESIGN", "next gate differs")
    require(next_step.get("required_order") == REQUIRED_ORDER, "required order differs")
    require(
        [tuple(row) for row in next_step["current_episode_topic_years_are_provisional_not_authorized_for_leaveout"]]
        == PROVISIONAL_EPISODES,
        "provisional episodes differ",
    )
    for key in (
        "selection_must_use_outcomes",
        "entry_or_persistence_outcomes_may_be_opened",
        "leaveout_or_peer_measurement_may_be_built_before_successor_freeze",
        "linkage_threshold_may_be_selected_from_entry_or_persistence",
        "final_candidate_eligibility_may_be_claimed",
        "final_field_choice_by_effect_size_allowed",
        "confirmatory_year_changes_allowed",
        "gpu_required",
    ):
        require(next_step.get(key) is False, f"unsafe next-step state: {key}")
    require(
        "not authorized for further leave-out" in str(ledger.get("claim_boundary")),
        "claim boundary omits leaveout pause",
    )
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v6",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(path),
        "additive_parent": parent_result,
        "linkage_audit_summary": summary_result,
        "live_linkage_audit": live,
        "gate_order": EXPECTED_ORDER,
        "measurement_review_required": True,
        "measurement_abi_redesign_authorized": True,
        "episode_reselection_successor_authorized_after_abi": True,
        "leaveout_network_bridge_build_authorized": False,
        "d0b_complete": False,
        "d0_complete": False,
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
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v6.yaml",
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
    except (GateStateV6Error, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V6",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
