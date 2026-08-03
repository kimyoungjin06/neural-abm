#!/usr/bin/env python3
"""Verify the additive D0 human-review packet-preparation gate ledger v8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v8"
LEDGER_STATUS = (
    "VALID_D0_REVIEW_PACKETS_PREPARED_HUMAN_REVIEW_NOT_RUN_"
    "OUTCOMES_FAIL_CLOSED"
)
VERIFY_STATUS = (
    "PASS_D0_REVIEW_PACKETS_PREPARED_AND_INDEPENDENTLY_VERIFIED_"
    "HUMAN_REVIEW_NOT_RUN_OUTCOMES_FAIL_CLOSED"
)
PACKET_STATUS = (
    "PASS_D0_BLINDED_TWO_HUMAN_REVIEWER_PACKETS_PREPARED_REVIEW_NOT_RUN"
)
PACKET_VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0_BLINDED_TWO_HUMAN_REVIEWER_PACKETS_REVIEW_NOT_RUN"
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
REQUIRED_NEXT_ORDER = [
    "assign_two_distinct_human_reviewers",
    "complete_reviewer_1_packet_without_cross_review",
    "complete_reviewer_2_packet_without_cross_review",
    "freeze_and_hash_both_completed_reviewer_files",
    "verify_complete_allowed_labels_and_required_notes",
    "compute_exact_agreement_and_cohen_kappa",
    "adjudicate_every_disagreement_with_a_third_human",
    "freeze_adjudicated_review_receipt",
    "define_candidate_level_indexing_contamination_rule_without_outcomes",
    "apply_rule_to_complete_candidate_universe",
    "rerun_deterministic_two_field_four_episode_selection_or_fail_coverage",
    "independently_replay_the_successor_freeze",
    "rebuild_author_opportunity_panel_for_successor_episodes",
]


class GateStateV8Error(RuntimeError):
    """Fail-closed gate-state v8 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV8Error(message)


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
        raise GateStateV8Error(f"invalid document {path.name}: {error}") from error
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


def _validate_live_packet_evidence(
    summary: Mapping[str, Any],
    packet_state: Mapping[str, Any],
    *,
    repo_root: Path,
    output_root: Path | None,
    verification_report: Path | None,
) -> dict[str, Any]:
    output = _mapping(summary.get("output"), "packet output")
    root = (
        Path("/ssd") / str(output["root_basename"])
        if output_root is None
        else output_root.resolve(strict=True)
    )
    require(root.is_dir() and not root.is_symlink(), "packet root invalid")
    require(root.name == output.get("root_basename"), "packet root basename differs")
    require(root.stat().st_mode & 0o777 == 0o700, "packet root mode differs")
    receipt_path = root / "content_review_packets.outputs.json"
    receipt_sha = _sha256(receipt_path)
    require(receipt_sha == output.get("receipt_sha256"), "packet receipt SHA mismatch")
    receipt = _read(receipt_path, yaml_document=False)
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_content_review_packet_outputs.v1",
        "packet receipt schema differs",
    )
    require(receipt.get("status") == PACKET_STATUS, "packet receipt status differs")
    require(
        receipt.get("artifact_tree_sha256") == output.get("artifact_tree_sha256"),
        "packet artifact tree differs",
    )
    require(receipt.get("reviewer_packet_rows_each") == 384, "packet rows differ")
    require(receipt.get("packet_orders_differ") is True, "packet orders identical")
    require(
        receipt.get("packet_content_sets_match") is True,
        "packet content sets differ",
    )
    require(receipt.get("reviewer_labels_filled") == 0, "packet labels filled")
    for key in (
        "human_review_complete",
        "automatic_title_classification_performed",
        "outcomes_read",
        "d0_complete",
        "leaveout_network_bridge_authorized",
        "discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe packet receipt flag: {key}")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid packet file: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"packet file mode differs: {path.name}")

    verification = _mapping(
        summary.get("independent_verification"),
        "independent verification",
    )
    report_path = (
        Path("/ssd") / str(verification["report_basename"])
        if verification_report is None
        else verification_report.resolve(strict=True)
    )
    report_sha = _sha256(report_path)
    require(
        report_sha == verification.get("report_sha256"),
        "packet verification report SHA mismatch",
    )
    report = _read(report_path, yaml_document=False)
    require(
        report.get("schema_version")
        == "pivot_feedback_d0_content_review_packet_verification.v1",
        "packet verification schema differs",
    )
    require(report.get("status") == PACKET_VERIFY_STATUS, "packet verification status differs")
    require(report.get("packet_content_exact_match") is True, "packet content unverified")
    require(report.get("adjudication_key_exact_match") is True, "adjudication key unverified")
    require(report.get("packet_orders_differ") is True, "packet order verification failed")
    require(report.get("reviewer_labels_filled") == 0, "verification labels filled")
    require(report.get("producer_module_imported") is False, "producer imported")
    for key in (
        "human_review_complete",
        "outcomes_read",
        "d0_complete",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(report.get(key) is False, f"unsafe packet verification flag: {key}")

    producer = _mapping(summary.get("producer"), "packet producer")
    for key in ("script", "verifier", "config"):
        path = _safe_path(repo_root, producer.get(key), f"{key} path")
        require(_sha256(path) == producer.get(f"{key}_sha256"), f"{key} SHA mismatch")
    require(
        report.get("verifier_sha256") == producer.get("verifier_sha256"),
        "live verifier SHA differs",
    )
    require(
        packet_state.get("receipt_sha256") == receipt_sha,
        "ledger packet receipt differs",
    )
    require(
        packet_state.get("verification_report_sha256") == report_sha,
        "ledger packet verification differs",
    )
    return {
        "root_basename": root.name,
        "receipt_sha256": receipt_sha,
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "verification_report_basename": report_path.name,
        "verification_report_sha256": report_sha,
        "reviewer_packet_rows_each": 384,
        "reviewer_labels_filled": 0,
        "human_review_complete": False,
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
    require(ledger.get("label") == "pivot_feedback_gate_state_v8", "ledger label mismatch")
    require(
        ledger.get("mode")
        == "ADDITIVE_D0_HUMAN_REVIEW_PACKET_PREPARATION_GATE_LEDGER",
        "ledger mode mismatch",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status mismatch")

    parent_raw = _mapping(ledger.get("additive_parent"), "additive parent")
    require(parent_raw.get("mutation_allowed") is False, "v7 mutation allowed")
    _, parent_result = _validate_binding(
        parent_raw,
        repo_root=repo_root,
        yaml_document=True,
    )
    evidence = _mapping(ledger.get("execution_evidence"), "execution evidence")
    require(list(evidence) == ["blinded_human_review_packets"], "evidence set differs")
    summary, summary_result = _validate_binding(
        _mapping(evidence["blinded_human_review_packets"], "packet evidence"),
        repo_root=repo_root,
        yaml_document=False,
    )

    states = _mapping(ledger.get("gate_state"), "gate states")
    require(list(states) == EXPECTED_ORDER, "gate order differs")
    d0 = _mapping(states["D0"], "D0 state")
    require(
        d0.get("state") == "PARTIAL_REVIEW_PACKETS_PREPARED_HUMAN_REVIEW_NOT_RUN",
        "D0 state differs",
    )
    require(d0.get("complete") is False, "D0 improperly complete")
    require(
        d0.get("authorizes")
        == [
            "D0_TWO_HUMAN_TITLE_REVIEWS",
            "D0_THIRD_PARTY_ADJUDICATION_AFTER_REVIEWER_FREEZE",
            "D0_OUTCOME_BLIND_INDEXING_CONTAMINATION_SUCCESSOR_AFTER_ADJUDICATION",
        ],
        "D0 authorization differs",
    )
    require(states["G0"] == {"state": "PENDING", "complete": False}, "G0 differs")
    require(
        states["G4"] == {"state": "SEALED_NOT_OPENED", "complete": False},
        "G4 is not sealed",
    )

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(
        authorization.get("d0_blinded_review_packet_preparation")
        == "COMPLETE_INDEPENDENTLY_VERIFIED",
        "packet preparation not complete",
    )
    require(
        authorization.get("d0_adjudication")
        == "AUTHORIZED_AFTER_BOTH_REVIEWER_FILES_FROZEN",
        "adjudication order differs",
    )
    require(
        authorization.get("d0b_author_opportunity_foundation_v1")
        == "SUPERSEDED_AUDIT_BASE_NOT_CURRENT_PANEL",
        "old author panel not superseded",
    )
    require(
        authorization.get("d0b_leaveout_network_bridge_build")
        == "PAUSED_PENDING_REVIEW_AND_RESELECTION_SUCCESSOR",
        "leaveout build not paused",
    )
    for key in (
        "d0_discovery_outcome_opening",
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

    packet_state = _mapping(ledger.get("review_packet_state"), "review packet state")
    require(packet_state.get("reviewer_packet_rows_each") == 384, "packet rows differ")
    for key in (
        "packet_orders_differ",
        "packet_content_sets_match",
        "packet_content_exact_match",
        "adjudication_key_exact_match",
    ):
        require(packet_state.get(key) is True, f"packet state failed: {key}")
    for key in (
        "reviewer_1_completed_rows",
        "reviewer_2_completed_rows",
        "adjudicated_rows",
    ):
        require(packet_state.get(key) == 0, f"review completion overstated: {key}")
    require(packet_state.get("human_review_complete") is False, "human review complete")
    require(
        packet_state.get("artificial_agent_may_substitute_for_human_reviewer")
        is False,
        "artificial reviewer substitution allowed",
    )

    scope = _mapping(
        ledger.get("provisional_scientific_scope"),
        "provisional scientific scope",
    )
    require(scope.get("episode_3_quarantined_pending_review") is True, "episode 3 not quarantined")
    for key in (
        "remaining_episodes_automatically_promoted",
        "current_four_episode_set_valid_for_leaveout_or_author_panel",
        "prior_author_episode_rows_are_current_panel",
        "entry_or_persistence_outcomes_read",
    ):
        require(scope.get(key) is False, f"unsafe scientific scope: {key}")

    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(next_step.get("gate") == "D0_HUMAN_CONTENT_REVIEW_EXECUTION", "next gate differs")
    require(next_step.get("required_order") == REQUIRED_NEXT_ORDER, "required order differs")
    for key in (
        "reviewer_discussion_before_independent_freeze_allowed",
        "artificial_reviewer_substitution_allowed",
        "entry_or_persistence_outcomes_may_be_opened",
        "leaveout_or_peer_measurement_may_be_built_before_successor_freeze",
        "current_episode_set_may_be_reused_as_final",
        "current_author_panel_may_be_reused_as_final",
        "manual_episode_replacement_without_complete_universe_rule_allowed",
        "final_candidate_eligibility_may_be_claimed",
        "final_field_choice_by_effect_size_allowed",
        "confirmatory_year_changes_allowed",
        "gpu_required",
    ):
        require(next_step.get(key) is False, f"unsafe next-step state: {key}")

    live = _validate_live_packet_evidence(
        summary,
        packet_state,
        repo_root=repo_root,
        output_root=output_root,
        verification_report=verification_report,
    )
    require(
        "No title has been classified" in str(ledger.get("claim_boundary")),
        "claim boundary omits review-not-run state",
    )
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v8",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(path),
        "additive_parent": parent_result,
        "packet_summary": summary_result,
        "live_packet_evidence": live,
        "gate_order": EXPECTED_ORDER,
        "packets_prepared": True,
        "packets_independently_verified": True,
        "reviewer_1_completed_rows": 0,
        "reviewer_2_completed_rows": 0,
        "adjudicated_rows": 0,
        "human_review_complete": False,
        "episode_3_quarantined": True,
        "remaining_episodes_promoted": False,
        "old_author_panel_current": False,
        "leaveout_network_bridge_build_authorized": False,
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
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v8.yaml",
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
    except (GateStateV8Error, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V8",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
