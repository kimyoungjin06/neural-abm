#!/usr/bin/env python3
"""Verify the additive D0 ABI-reselection and content-review gate ledger v7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v7"
LEDGER_STATUS = (
    "VALID_D0_CONTENT_REVIEW_REQUIRED_EPISODE_3_QUARANTINED_"
    "OUTCOMES_FAIL_CLOSED"
)
VERIFY_STATUS = (
    "PASS_D0_ABI_RESELECTION_AND_AUTOMATED_CONTENT_AUDIT_"
    "HUMAN_DOUBLE_REVIEW_REQUIRED_OUTCOMES_FAIL_CLOSED"
)
RESELECTION_STATUS = (
    "PASS_D0_OUTCOME_BLIND_MEASUREMENT_ABI_TWO_FIELD_FOUR_EPISODE_RESELECTION"
)
CONTENT_STATUS = "PASS_D0_CONTENT_INDEXING_EXTRACTION_HUMAN_REVIEW_REQUIRED"
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
    (27, 12505, 1991),
    (27, 12421, 1953),
]
ALLOWED_LABELS = ["ON_TOPIC", "AMBIGUOUS", "OFF_TOPIC", "UNREVIEWABLE"]
REVIEW_COLUMNS = [
    "reviewer_1_label",
    "reviewer_2_label",
    "adjudicated_label",
    "review_notes",
]
REQUIRED_NEXT_ORDER = [
    "generate_two_separately_ordered_blinded_human_review_packets",
    "collect_two_independent_complete_title_reviews",
    "adjudicate_all_disagreements_without_outcomes",
    "freeze_review_receipt_and_inter_reviewer_agreement",
    "define_candidate_level_indexing_contamination_rule_without_outcomes",
    "apply_rule_to_complete_candidate_universe",
    "rerun_deterministic_two_field_four_episode_selection_or_fail_coverage",
    "independently_replay_the_successor_freeze",
    "rebuild_author_opportunity_panel_for_successor_episodes",
]


class GateStateV7Error(RuntimeError):
    """Fail-closed gate-state v7 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV7Error(message)


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
        raise GateStateV7Error(f"invalid document {path.name}: {error}") from error
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


def _validate_reselection_live(
    summary: Mapping[str, Any],
    *,
    output_root: Path | None,
) -> dict[str, Any]:
    output = _mapping(summary.get("output"), "reselection output")
    root = (
        Path("/ssd") / str(output["root_basename"])
        if output_root is None
        else output_root.resolve(strict=True)
    )
    require(root.is_dir() and not root.is_symlink(), "reselection root invalid")
    require(root.name == output.get("root_basename"), "reselection basename differs")
    receipt_path = root / "measurement_abi_reselection.outputs.json"
    report_path = root / "measurement_abi_reselection.json"
    selected_path = root / "selected_episode_predecision_works.parquet"
    require(_sha256(receipt_path) == output.get("receipt_sha256"), "receipt SHA mismatch")
    require(_sha256(report_path) == output.get("report_sha256"), "report SHA mismatch")
    require(
        _sha256(selected_path) == output.get("selected_work_sha256"),
        "selected-work SHA mismatch",
    )
    receipt = _read(receipt_path, yaml_document=False)
    report = _read(report_path, yaml_document=False)
    require(receipt.get("status") == RESELECTION_STATUS, "receipt status differs")
    require(report.get("status") == RESELECTION_STATUS, "report status differs")
    require(
        receipt.get("artifact_tree_sha256") == output.get("artifact_tree_sha256"),
        "artifact tree differs",
    )
    require(receipt.get("selected_episode_count") == 4, "episode count differs")
    require(
        receipt.get("selected_episode_predecision_work_rows") == 43_659,
        "selected-work rows differ",
    )
    for key in (
        "content_and_indexing_audit_complete",
        "d0_complete",
        "g0_authorized",
        "leaveout_network_bridge_authorized",
        "model_fitting_authorized",
        "g4_open",
        "outcomes_read",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe reselection receipt flag: {key}")
    episodes = [
        (
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        )
        for row in report["episode_reselection"]["selected_episodes"]
    ]
    require(episodes == PROVISIONAL_EPISODES, "live reselection episodes differ")
    summary_episodes = [
        (
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        )
        for row in summary["reselection"]["selected_episodes"]
    ]
    require(summary_episodes == episodes, "compact reselection episodes differ")
    require(
        report["work_abi_accounting"]["linked_rows"]
        == summary["measurement_abi"]["linked_rows"],
        "linked-row summary differs",
    )
    return {
        "root_basename": root.name,
        "receipt_sha256": _sha256(receipt_path),
        "report_sha256": _sha256(report_path),
        "selected_work_sha256": _sha256(selected_path),
        "selected_episode_count": len(episodes),
        "selected_episode_predecision_work_rows": 43_659,
    }


def _validate_content_live(
    summary: Mapping[str, Any],
    review_state: Mapping[str, Any],
    *,
    output_root: Path | None,
    review_template: Path | None,
) -> dict[str, Any]:
    output = _mapping(summary.get("output"), "content output")
    root = (
        Path("/ssd") / str(output["root_basename"])
        if output_root is None
        else output_root.resolve(strict=True)
    )
    require(root.is_dir() and not root.is_symlink(), "content root invalid")
    require(root.name == output.get("root_basename"), "content basename differs")
    receipt_path = root / "content_indexing_audit.outputs.json"
    report_path = root / "content_indexing_audit.json"
    template_path = (
        root / str(review_state["source_review_template_basename"])
        if review_template is None
        else review_template.resolve(strict=True)
    )
    require(_sha256(receipt_path) == output.get("receipt_sha256"), "content receipt SHA mismatch")
    require(_sha256(report_path) == output.get("report_sha256"), "content report SHA mismatch")
    require(
        _sha256(template_path) == review_state.get("source_review_template_sha256"),
        "review template SHA mismatch",
    )
    receipt = _read(receipt_path, yaml_document=False)
    report = _read(report_path, yaml_document=False)
    require(receipt.get("status") == CONTENT_STATUS, "content receipt status differs")
    require(report.get("status") == CONTENT_STATUS, "content report status differs")
    require(receipt.get("sample_rows") == 384, "content sample rows differ")
    require(receipt.get("human_review_complete") is False, "review already complete")
    for key in (
        "d0_complete",
        "g0_authorized",
        "leaveout_network_bridge_authorized",
        "model_fitting_authorized",
        "g4_open",
        "outcomes_read",
        "gpu_used",
    ):
        require(receipt.get(key) is False, f"unsafe content receipt flag: {key}")
    require(report["human_review"]["required"] is True, "human review not required")
    require(report["human_review"]["complete"] is False, "human review complete")
    require(
        int(report["indexing_review_flags"][0]["episode_index"]) == 3,
        "decisive review episode differs",
    )

    with template_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "review template lacks header")
        for column in REVIEW_COLUMNS:
            require(column in reader.fieldnames, f"review column missing: {column}")
        rows = list(reader)
    require(len(rows) == 384, "review template row count differs")
    for row in rows:
        for column in REVIEW_COLUMNS:
            require(not str(row[column]).strip(), f"review template not blank: {column}")
    require(
        len({str(row["openalex_id"]) for row in rows}) == 384,
        "review template work IDs not unique",
    )
    return {
        "root_basename": root.name,
        "receipt_sha256": _sha256(receipt_path),
        "report_sha256": _sha256(report_path),
        "review_template_sha256": _sha256(template_path),
        "review_rows": len(rows),
        "reviewer_1_completed_rows": 0,
        "reviewer_2_completed_rows": 0,
        "adjudicated_rows": 0,
    }


def validate(
    path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    reselection_root: Path | None = None,
    content_root: Path | None = None,
    review_template: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    path = path.resolve(strict=True)
    ledger = _read(path, yaml_document=True)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema mismatch")
    require(ledger.get("label") == "pivot_feedback_gate_state_v7", "ledger label mismatch")
    require(
        ledger.get("mode")
        == "ADDITIVE_D0_MEASUREMENT_ABI_RESELECTION_AND_CONTENT_REVIEW_GATE_LEDGER",
        "ledger mode mismatch",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status mismatch")

    parent_raw = _mapping(ledger.get("additive_parent"), "additive parent")
    require(parent_raw.get("mutation_allowed") is False, "v6 mutation allowed")
    _, parent_result = _validate_binding(
        parent_raw,
        repo_root=repo_root,
        yaml_document=True,
    )

    evidence = _mapping(ledger.get("execution_evidence"), "execution evidence")
    require(
        list(evidence) == ["measurement_abi_reselection", "content_indexing_audit"],
        "execution evidence order differs",
    )
    reselection_summary, reselection_result = _validate_binding(
        _mapping(evidence["measurement_abi_reselection"], "reselection evidence"),
        repo_root=repo_root,
        yaml_document=False,
    )
    content_summary, content_result = _validate_binding(
        _mapping(evidence["content_indexing_audit"], "content evidence"),
        repo_root=repo_root,
        yaml_document=False,
    )

    states = _mapping(ledger.get("gate_state"), "gate states")
    require(list(states) == EXPECTED_ORDER, "gate order differs")
    require(states["G_MINUS_1A"]["state"] == "PASS_BOUNDED", "G-1A differs")
    require(states["S0"]["state"] == "PASS_THRESHOLD_FREE_INVENTORY", "S0 differs")
    d0 = _mapping(states["D0"], "D0 state")
    require(
        d0.get("state") == "PARTIAL_ABI_RESELECTION_PASS_CONTENT_REVIEW_REQUIRED",
        "D0 state differs",
    )
    require(d0.get("complete") is False, "D0 improperly complete")
    require(
        d0.get("authorizes")
        == [
            "D0_HUMAN_DOUBLE_CONTENT_REVIEW",
            "D0_OUTCOME_BLIND_INDEXING_CONTAMINATION_SUCCESSOR_AFTER_REVIEW",
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
        authorization.get("d0_human_double_content_review")
        == "AUTHORIZED_OUTCOME_BLIND",
        "human review not authorized",
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

    scope = _mapping(ledger.get("current_provisional_scope"), "provisional scope")
    require(
        [tuple(row) for row in scope.get("episode_topic_years", [])]
        == PROVISIONAL_EPISODES,
        "provisional episode set differs",
    )
    require(scope.get("episode_3_quarantined_pending_review") is True, "episode 3 not quarantined")
    for key in (
        "remaining_episodes_automatically_promoted",
        "current_four_episode_set_valid_for_leaveout_or_author_panel",
        "prior_author_episode_rows_are_current_panel",
    ):
        require(scope.get(key) is False, f"unsafe provisional scope: {key}")

    review_state = _mapping(ledger.get("human_review_state"), "human review state")
    require(review_state.get("sample_rows") == 384, "review sample rows differ")
    for key in (
        "reviewer_1_completed_rows",
        "reviewer_2_completed_rows",
        "adjudicated_rows",
    ):
        require(review_state.get(key) == 0, f"review completion overstated: {key}")
    require(review_state.get("minimum_independent_reviewers") == 2, "reviewer count differs")
    require(review_state.get("allowed_labels") == ALLOWED_LABELS, "review labels differ")
    require(
        review_state.get("artificial_agent_may_substitute_for_human_reviewer")
        is False,
        "artificial reviewer substitution allowed",
    )
    require(review_state.get("review_freeze_complete") is False, "review freeze complete")

    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(
        next_step.get("gate")
        == "D0_CONTENT_REVIEW_AND_INDEXING_CONTAMINATION_SUCCESSOR",
        "next gate differs",
    )
    require(next_step.get("required_order") == REQUIRED_NEXT_ORDER, "required order differs")
    for key in (
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

    live_reselection = _validate_reselection_live(
        reselection_summary,
        output_root=reselection_root,
    )
    live_content = _validate_content_live(
        content_summary,
        review_state,
        output_root=content_root,
        review_template=review_template,
    )
    require(
        "prior author-opportunity foundation is not the current analytic panel"
        in str(ledger.get("claim_boundary")),
        "claim boundary omits superseded author panel",
    )
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v7",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(path),
        "additive_parent": parent_result,
        "reselection_summary": reselection_result,
        "content_summary": content_result,
        "live_reselection": live_reselection,
        "live_content": live_content,
        "gate_order": EXPECTED_ORDER,
        "human_double_review_required": True,
        "human_double_review_complete": False,
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
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v7.yaml",
    )
    parser.add_argument("--reselection-root", type=Path)
    parser.add_argument("--content-root", type=Path)
    parser.add_argument("--review-template", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(
            args.ledger,
            reselection_root=args.reselection_root,
            content_root=args.content_root,
            review_template=args.review_template,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    except (GateStateV7Error, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V7",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
