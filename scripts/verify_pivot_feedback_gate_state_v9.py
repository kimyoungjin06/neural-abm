#!/usr/bin/env python3
"""Verify the additive D0 API-review preparation gate ledger v9."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot_feedback_gate_state.v9"
LEDGER_STATUS = (
    "VALID_D0_API_REVIEW_INPUTS_PREPARED_NOT_SUBMITTED_OUTCOMES_FAIL_CLOSED"
)
SUMMARY_STATUS = (
    "PASS_INDEPENDENT_D0_API_ASSISTED_REVIEW_INPUT_VERIFICATION_NOT_SUBMITTED"
)
PREPARATION_STATUS = (
    "PASS_D0_API_ASSISTED_REVIEW_BATCH_REQUESTS_PREPARED_NOT_SUBMITTED"
)
VERIFY_STATUS = (
    "PASS_D0_API_REVIEW_INPUTS_PREPARED_AND_INDEPENDENTLY_VERIFIED_"
    "NOT_SUBMITTED_OUTCOMES_FAIL_CLOSED"
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
    "confirm_operator_credential_and_cost_control",
    "upload_both_frozen_jsonl_files_with_batch_purpose",
    "submit_two_24_hour_responses_batches",
    "preserve_upload_file_ids_batch_ids_and_request_ids",
    "download_and_hash_output_and_error_files",
    "validate_all_768_responses_against_frozen_schema",
    "derive_deduplicated_base_plus_escalation_human_queue",
    "assign_two_distinct_human_reviewers",
    "complete_both_blinded_human_packets_without_cross_review",
    "freeze_and_hash_both_completed_reviewer_files",
    "compute_exact_agreement_and_cohen_kappa",
    "adjudicate_every_human_disagreement_with_a_third_human",
    "apply_fail_close_rules_or_freeze_calibrated_review_receipt",
    "freeze_candidate_universe_screen_successor_before_any_full_screen",
]


class GateStateV9Error(RuntimeError):
    """Fail-closed gate-state v9 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV9Error(message)


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
        raise GateStateV9Error(f"invalid document {path.name}: {error}") from error
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
    output = _mapping(summary.get("output"), "API review output")
    root = (
        Path("/ssd") / str(output["root_basename"])
        if output_root is None
        else output_root.resolve(strict=True)
    )
    require(root.is_dir() and not root.is_symlink(), "API output root invalid")
    require(root.name == output.get("root_basename"), "API output basename differs")
    require(root.stat().st_mode & 0o777 == 0o700, "API output mode differs")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid API output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"API file mode differs: {path.name}")

    receipt_path = root / "api_review_preparation.outputs.json"
    receipt_sha = _sha256(receipt_path)
    require(receipt_sha == output.get("receipt_sha256"), "API receipt SHA mismatch")
    receipt = _read(receipt_path, yaml_document=False)
    require(
        receipt.get("schema_version")
        == "pivot_feedback_d0_api_review_preparation_outputs.v2",
        "API receipt schema differs",
    )
    require(receipt.get("status") == PREPARATION_STATUS, "API receipt status differs")
    require(
        receipt.get("artifact_tree_sha256") == output.get("artifact_tree_sha256"),
        "API artifact tree differs",
    )
    require(receipt.get("artifact_count") == 8, "API artifact count differs")
    require(receipt.get("request_rows_per_judge") == 384, "request rows differ")
    require(receipt.get("total_request_rows") == 768, "total requests differ")
    require(receipt.get("base_calibration_rows") == 96, "calibration rows differ")
    require(receipt.get("base_calibration_strata") == 12, "calibration strata differ")
    require(
        receipt.get("request_bodies_contain_forbidden_fields") is False,
        "forbidden fields reported",
    )
    require(receipt.get("episode_3_quarantined") is True, "episode 3 not quarantined")
    for key in (
        "api_submitted",
        "api_responses_received",
        "api_review_complete",
        "human_calibration_complete",
        "legacy_human_review_complete",
        "candidate_universe_screen_complete",
        "remaining_episodes_promoted",
        "old_author_panel_current",
        "d0_complete",
        "discovery_outcome_opening_authorized",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
        "outcomes_read",
    ):
        require(receipt.get(key) is False, f"unsafe API receipt flag: {key}")
    require(receipt.get("scientific_effect") == "NOT_ESTABLISHED", "effect overstated")

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
        "API verification report SHA mismatch",
    )
    report = _read(report_path, yaml_document=False)
    require(
        report.get("schema_version")
        == "pivot_feedback_d0_api_review_preparation_verification.v2",
        "API verification schema differs",
    )
    require(report.get("status") == SUMMARY_STATUS, "API verification status differs")
    for key in (
        "request_content_exact_match",
        "request_visible_fields_exact_match",
        "calibration_packets_exact_match",
        "calibration_key_exact_match",
    ):
        require(report.get(key) is True, f"API verification failed: {key}")
    require(
        report.get("request_bodies_contain_forbidden_fields") is False,
        "verification found forbidden fields",
    )
    require(report.get("producer_module_imported") is False, "producer imported")
    for key in (
        "api_submitted",
        "api_responses_received",
        "human_calibration_complete",
        "d0_complete",
        "outcomes_read",
        "g0_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(report.get(key) is False, f"unsafe API verification flag: {key}")

    producer = _mapping(summary.get("producer"), "producer")
    for key in ("script", "verifier", "config"):
        path = _safe_path(repo_root, producer.get(key), f"{key} path")
        require(_sha256(path) == producer.get(f"{key}_sha256"), f"{key} SHA mismatch")
    require(
        report.get("verifier_sha256") == producer.get("verifier_sha256"),
        "live verifier SHA differs",
    )
    require(state.get("receipt_sha256") == receipt_sha, "ledger receipt differs")
    require(
        state.get("verification_report_sha256") == report_sha,
        "ledger verification report differs",
    )
    return {
        "root_basename": root.name,
        "receipt_sha256": receipt_sha,
        "verification_report_sha256": report_sha,
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
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
        _mapping(ledger.get("additive_parent"), "additive parent"),
        repo_root=repo_root,
        yaml_document=True,
    )
    summary, evidence = _validate_binding(
        _mapping(
            _mapping(ledger.get("execution_evidence"), "execution evidence").get(
                "api_review_preparation"
            ),
            "API review evidence",
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
    require(gate_state["D0"].get("complete") is False, "D0 completion overstated")
    require(
        gate_state["D0"].get("state")
        == "PARTIAL_API_REVIEW_INPUTS_PREPARED_NOT_SUBMITTED",
        "D0 state differs",
    )
    require(gate_state["G4"].get("state") == "SEALED_NOT_OPENED", "G4 opened")
    require(gate_state["G4"].get("complete") is False, "G4 completion overstated")

    require(state.get("request_rows_per_judge") == 384, "ledger request rows differ")
    require(state.get("total_request_rows") == 768, "ledger total requests differ")
    require(state.get("calibration_rows") == 96, "ledger calibration rows differ")
    require(state.get("calibration_strata") == 12, "ledger strata differ")
    require(state.get("request_content_exact_match") is True, "requests unverified")
    require(
        state.get("forbidden_request_fields_present") is False,
        "ledger reports forbidden fields",
    )
    require(state.get("calibration_packets_exact_match") is True, "packets unverified")
    for key in (
        "api_submitted",
        "api_responses_received",
        "human_calibration_complete",
        "candidate_universe_screen_complete",
    ):
        require(state.get(key) is False, f"execution completion overstated: {key}")
    require(
        state.get("artificial_agent_may_substitute_for_human_reviewer") is False,
        "artificial reviewer substitution allowed",
    )

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(
        authorization.get("d0_api_batch_submission")
        == "AUTHORIZED_ONLY_WITH_OPERATOR_CREDENTIAL_AND_COST_CONTROL",
        "API submission authorization differs",
    )
    require(
        authorization.get("d0_candidate_universe_api_screen")
        == "NOT_AUTHORIZED_PENDING_CALIBRATION",
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
    require(scope.get("scientific_effect") == "NOT_ESTABLISHED", "scope effect overstated")

    next_step = _mapping(ledger.get("next_required_implementation"), "next step")
    require(next_step.get("required_order") == REQUIRED_NEXT_ORDER, "next order differs")
    for key in (
        "reviewer_discussion_before_independent_freeze_allowed",
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
        "schema_version": "pivot_feedback_gate_state_verification.v9",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(ledger_path),
        "parent": parent,
        "evidence": evidence,
        "live": live,
        "batch_requests_prepared": True,
        "batch_requests_independently_verified": True,
        "api_submitted": False,
        "api_responses_received": False,
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
        default=REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v9.yaml",
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
    except (
        GateStateV9Error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V9",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
