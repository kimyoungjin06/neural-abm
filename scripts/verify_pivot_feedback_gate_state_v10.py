#!/usr/bin/env python3
"""Verify the additive D0 cross-provider result gate ledger v10."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v10"
LEDGER_STATUS = (
    "VALID_D0_API_REVIEW_COMPLETE_HUMAN_REVIEW_PENDING_OUTCOMES_FAIL_CLOSED"
)
SUMMARY_STATUS = (
    "PASS_INDEPENDENT_D0_CROSS_PROVIDER_RESULT_VERIFICATION_HUMAN_REVIEW_PENDING"
)
RESULT_STATUS = "PASS_VALID_CROSS_PROVIDER_RESULTS_HUMAN_REVIEW_PENDING"
RESULT_VERIFY_STATUS = "PASS_INDEPENDENT_D0_CROSS_PROVIDER_RESULT_VERIFICATION"
VERIFY_STATUS = (
    "PASS_D0_VALID_CROSS_PROVIDER_API_REVIEW_HUMAN_REVIEW_PENDING_"
    "OUTCOMES_FAIL_CLOSED"
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
    "distribute_separately_ordered_233_row_blinded_packets",
    "complete_both_packets_without_cross_review_or_api_output_access",
    "freeze_and_hash_both_completed_reviewer_files",
    "compute_exact_human_agreement_and_cohen_kappa",
    "adjudicate_every_human_disagreement_with_a_third_human",
    "test_api_error_rates_against_frozen_human_calibration",
    "apply_fail_close_rules_or_freeze_calibrated_review_receipt",
    "freeze_candidate_universe_screen_successor_before_any_full_screen",
]


class GateStateV10Error(RuntimeError):
    """Fail-closed gate-state v10 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV10Error(message)


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
        raise GateStateV10Error(f"invalid document {path.name}: {error}") from error
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


def _validate_live_evidence(
    summary: Mapping[str, Any],
    state: Mapping[str, Any],
    *,
    repo_root: Path,
    output_root: Path | None,
    verification_report: Path | None,
) -> dict[str, Any]:
    execution = _mapping(summary.get("execution"), "execution")
    v3_root = Path("/ssd") / str(execution["v3_execution_root_basename"])
    require(
        v3_root.is_dir() and not v3_root.is_symlink(),
        "v3 execution root invalid",
    )
    cancel_path = v3_root / "openai_batch_cancel_final_status.json"
    require(
        _sha256(cancel_path)
        == execution["v3_openai_final_status_snapshot_sha256"],
        "v3 final cancellation snapshot SHA differs",
    )
    cancel = _read(cancel_path, yaml_document=False)
    require(
        cancel.get("batch_id") == execution["v3_openai_batch_id"]
        and cancel.get("status") == "cancelled",
        "v3 final cancellation status differs",
    )
    require(
        _mapping(cancel.get("request_counts"), "v3 request counts")
        == {"completed": 0, "failed": 0, "total": 0},
        "v3 cancelled request counts differ",
    )
    require(
        cancel.get("output_file_id") is None
        and cancel.get("error_file_id") is None
        and cancel.get("credential_values_persisted") is False,
        "v3 cancellation boundary differs",
    )
    output = _mapping(summary.get("output"), "result output")
    root = (
        Path("/ssd") / str(output["root_basename"])
        if output_root is None
        else output_root.resolve(strict=True)
    )
    require(root.is_dir() and not root.is_symlink(), "result output root invalid")
    require(root.name == output["root_basename"], "result output basename differs")
    require(root.stat().st_mode & 0o777 == 0o700, "result output mode differs")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid result: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"result mode differs: {path.name}")
    receipt_path = root / "cross_provider_result.outputs.json"
    require(_sha256(receipt_path) == output["receipt_sha256"], "result receipt SHA differs")
    receipt = _read(receipt_path, yaml_document=False)
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_cross_provider_result_outputs.v6",
        "result receipt schema differs",
    )
    require(receipt.get("status") == RESULT_STATUS, "result receipt status differs")
    require(
        receipt.get("artifact_tree_sha256") == output["artifact_tree_sha256"],
        "result artifact tree differs",
    )
    require(receipt.get("openai_valid_structured_rows") == 384, "OpenAI rows differ")
    require(receipt.get("claude_valid_structured_rows") == 384, "Claude rows differ")
    require(receipt.get("shared_expected_custom_ids") == 384, "shared IDs differ")
    reduction = _mapping(receipt.get("reduction"), "result reduction")
    expected_reduction = {
        "label_agreement_rows": 306,
        "label_disagreement_rows": 78,
        "api_escalation_rows": 187,
        "base_calibration_rows": 96,
        "calibration_escalation_overlap_rows": 50,
        "blinded_human_queue_rows": 233,
        "dual_high_confidence_on_topic_rows": 197,
    }
    require(reduction == expected_reduction, "result reduction differs")
    require(
        receipt.get("human_packets_blinded_from_api_outputs") is True,
        "human packet blinding differs",
    )
    for key in (
        "human_calibration_complete",
        "human_adjudication_complete",
        "candidate_universe_screen_complete",
        "d0_complete",
        "outcomes_read",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"unsafe result flag: {key}")

    report_path = (
        Path("/ssd") / str(output["verification_report_basename"])
        if verification_report is None
        else verification_report.resolve(strict=True)
    )
    require(
        _sha256(report_path) == output["verification_report_sha256"],
        "result verification report SHA differs",
    )
    report = _read(report_path, yaml_document=False)
    require(
        report.get("schema_version")
        == "pivot_feedback_d0_cross_provider_result_verification.v6",
        "result verification schema differs",
    )
    require(report.get("status") == RESULT_VERIFY_STATUS, "result verification status differs")
    verifier = _safe_path(
        repo_root,
        _mapping(summary.get("producer"), "producer").get("verifier"),
        "result verifier",
    )
    live_process = subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--output-root",
            str(root),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    require(live_process.returncode == 0, "live result verification failed")
    live = _mapping(json.loads(live_process.stdout), "live result verification")
    require(live == report, "stored result verification differs from live")
    require(live.get("producer_module_imported") is False, "result producer imported")
    require(live.get("human_packets_blinded_from_api_outputs") is True, "blinding unverified")
    require(live.get("human_calibration_complete") is False, "human review overstated")
    require(live.get("d0_complete") is False, "D0 overstated")
    require(live.get("outcomes_read") is False, "outcomes read")
    producer = _mapping(summary.get("producer"), "producer")
    for key in ("script", "verifier", "config"):
        path = _safe_path(repo_root, producer.get(key), f"{key} path")
        require(_sha256(path) == producer.get(f"{key}_sha256"), f"{key} SHA differs")
    require(state.get("receipt_sha256") == _sha256(receipt_path), "ledger receipt differs")
    require(
        state.get("verification_report_sha256") == _sha256(report_path),
        "ledger report differs",
    )
    return {
        "root_basename": root.name,
        "receipt_sha256": _sha256(receipt_path),
        "verification_report_sha256": _sha256(report_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "live_verification_exact_match": True,
        "v3_final_cancellation_snapshot_sha256": _sha256(cancel_path),
    }


def validate(
    ledger_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path | None = None,
    verification_report: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = _read(ledger_path, yaml_document=True)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema differs")
    require(ledger.get("status") == LEDGER_STATUS, "ledger status differs")
    _, parent = _validate_binding(
        _mapping(ledger.get("additive_parent"), "parent"),
        repo_root=repo_root,
        yaml_document=True,
    )
    summary, evidence = _validate_binding(
        _mapping(
            _mapping(ledger.get("execution_evidence"), "execution evidence").get(
                "cross_provider_api_review"
            ),
            "cross-provider evidence",
        ),
        repo_root=repo_root,
        yaml_document=False,
    )
    require(summary.get("status") == SUMMARY_STATUS, "summary status differs")
    state = _mapping(ledger.get("api_review_state"), "API review state")
    live = _validate_live_evidence(
        summary,
        state,
        repo_root=repo_root,
        output_root=output_root,
        verification_report=verification_report,
    )

    gate_state = _mapping(ledger.get("gate_state"), "gate state")
    require(list(gate_state) == EXPECTED_ORDER, "gate order differs")
    require(
        gate_state["D0"].get("state")
        == "PARTIAL_VALID_API_REVIEW_COMPLETE_HUMAN_REVIEW_PENDING",
        "D0 state differs",
    )
    require(gate_state["D0"].get("complete") is False, "D0 complete")
    require(gate_state["G4"].get("state") == "SEALED_NOT_OPENED", "G4 opened")
    require(gate_state["G4"].get("complete") is False, "G4 complete")

    exact_state = {
        "valid_rows_per_provider": 384,
        "shared_expected_custom_ids": 384,
        "label_agreement_rows": 306,
        "label_disagreement_rows": 78,
        "dual_high_confidence_on_topic_rows": 197,
        "api_escalation_rows": 187,
        "calibration_rows": 96,
        "human_queue_rows_per_reviewer": 233,
        "human_reviewer_1_completed_rows": 0,
        "human_reviewer_2_completed_rows": 0,
    }
    for key, expected in exact_state.items():
        require(state.get(key) == expected, f"API review state differs: {key}")
    for key in (
        "human_packets_blinded_from_api_outputs",
        "api_submitted",
        "api_responses_received",
        "api_review_complete_for_frozen_set",
    ):
        require(state.get(key) is True, f"completed state missing: {key}")
    for key in (
        "human_calibration_complete",
        "candidate_universe_screen_complete",
        "artificial_agent_may_substitute_for_human_reviewer",
    ):
        require(state.get(key) is False, f"unsafe API review state: {key}")

    cost = _mapping(summary.get("cost_control"), "cost")
    require(
        Decimal(cost["cumulative_actual_cost_usd"])
        == Decimal("0.352772")
        < Decimal(cost["cumulative_authorized_maximum_usd"])
        == Decimal("1.00"),
        "cost boundary differs",
    )
    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(
        authorization.get("d0_api_response_ingestion")
        == "COMPLETE_INDEPENDENTLY_VERIFIED",
        "API review completion differs",
    )
    require(
        authorization.get("d0_candidate_universe_api_screen")
        == "NOT_AUTHORIZED_PENDING_HUMAN_CALIBRATION",
        "candidate screen prematurely authorized",
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

    scope = _mapping(ledger.get("provisional_scientific_scope"), "scope")
    require(scope.get("episode_3_quarantined_pending_review") is True, "episode 3 opened")
    require(scope.get("remaining_episodes_automatically_promoted") is False, "episodes promoted")
    require(scope.get("prior_author_episode_rows_are_current_panel") is False, "old panel current")
    require(scope.get("entry_or_persistence_outcomes_read") is False, "outcomes read")
    require(scope.get("scientific_effect") == "NOT_ESTABLISHED", "effect overstated")
    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(next_step.get("required_order") == REQUIRED_NEXT_ORDER, "next order differs")
    for key in (
        "reviewer_discussion_before_independent_freeze_allowed",
        "api_outputs_visible_to_human_reviewers",
        "artificial_reviewer_substitution_allowed",
        "api_only_may_complete_d0",
        "candidate_universe_screen_may_run_before_calibration",
        "entry_or_persistence_outcomes_may_be_opened",
        "leaveout_or_peer_measurement_may_be_built_before_successor_freeze",
        "current_episode_set_may_be_reused_as_final",
        "current_author_panel_may_be_reused_as_final",
        "final_candidate_eligibility_may_be_claimed",
        "gpu_required",
    ):
        require(next_step.get(key) is False, f"unsafe next-step flag: {key}")
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v10",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(ledger_path),
        "parent": parent,
        "evidence": evidence,
        "live": live,
        "api_submitted": True,
        "api_responses_received": True,
        "api_review_complete_for_frozen_set": True,
        "openai_valid_structured_rows": 384,
        "claude_valid_structured_rows": 384,
        "label_agreement_rows": 306,
        "label_disagreement_rows": 78,
        "dual_high_confidence_on_topic_rows": 197,
        "human_queue_rows_per_reviewer": 233,
        "human_packets_blinded_from_api_outputs": True,
        "human_calibration_complete": False,
        "episode_3_quarantined": True,
        "remaining_episodes_promoted": False,
        "old_author_panel_current": False,
        "d0_complete": False,
        "candidate_universe_screen_authorized": False,
        "d0_discovery_outcome_opening_authorized": False,
        "g4_open": False,
        "outcomes_read": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        type=Path,
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v10.yaml",
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
        return 0
    except (
        GateStateV10Error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V10",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
