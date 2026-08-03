#!/usr/bin/env python3
"""Verify the API-only zero-cost topic-routing gate ledger v12."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v12.yaml"
SCHEMA = "pivot_feedback_gate_state.v12"
LEDGER_STATUS = (
    "VALID_API_ONLY_TOPIC_ROUTING_DRY_RUN_COMPLETE_MANIFEST_NOT_RUN_"
    "OUTCOMES_FAIL_CLOSED"
)
VERIFY_STATUS = (
    "PASS_API_ONLY_TOPIC_ROUTING_DRY_RUN_COMPLETE_MANIFEST_NOT_RUN_"
    "OUTCOMES_FAIL_CLOSED"
)
EXPECTED_GATE_ORDER = [
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
EXPECTED_NEXT_ORDER = [
    "freeze_complete_candidate_work_universe_definition",
    "materialize_title_bearing_candidate_works_for_all_candidate_topic_year_windows",
    "independently_verify_ids_windows_deduplication_provenance_and_counts",
    "apply_frozen_local_router_to_complete_manifest_without_api_submission",
    "freeze_actual_api_exception_count_and_provider_cost_estimate",
    "obtain_explicit_paid_submission_authorization",
    "prepare_and_execute_pinned_cross_provider_exception_screen",
    "quarantine_every_unresolved_candidate",
    "independently_verify_full_screen_receipt",
    "deterministically_reselect_attention_episodes",
    "rebuild_author_opportunity_panel",
    "classify_observable_history_new_topic_pivots",
]
EXPECTED_ROUTES = {
    "NO_API_RULE_ACCEPT": 229,
    "NO_API_RULE_REJECT": 24,
    "API_REQUIRED_AMBIGUOUS": 94,
    "QUARANTINE_UNRESOLVED": 37,
}


class GateStateV12Error(RuntimeError):
    """Fail-closed gate-state v12 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV12Error(message)


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


def _safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid YAML: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise GateStateV12Error(f"invalid YAML {path.name}: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateStateV12Error(f"invalid JSON {path.name}: {error}") from error
    return _mapping(value, path.name)


def _validate_document_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    binding = dict(raw)
    path = _safe_repo_path(repo_root, binding.get("relative_path"), label)
    require(_sha256(path) == binding.get("sha256"), f"{label} SHA mismatch")
    require(binding.get("mutation_allowed") is False, f"{label} mutation allowed")
    document = _read_json(path) if path.suffix == ".json" else _read_yaml(path)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        f"{label} schema mismatch",
    )
    require(
        document.get("status") == binding.get("required_status"),
        f"{label} status mismatch",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        f"{label} fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"{label} field mismatch: {dotted_path}",
        )
    return path, document


def _validate_file_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> Path:
    binding = dict(raw)
    path = _safe_repo_path(repo_root, binding.get("relative_path"), label)
    require(_sha256(path) == binding.get("sha256"), f"{label} SHA mismatch")
    require(binding.get("mutation_allowed") is False, f"{label} mutation allowed")
    return path


def _ssd_root(value: Any) -> Path:
    require(
        isinstance(value, str)
        and value
        and "/" not in value
        and value not in {".", ".."},
        "unsafe SSD root basename",
    )
    path = (Path("/ssd") / value).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "SSD root escapes /ssd")
    require(path.is_dir() and not path.is_symlink(), "invalid SSD root")
    return path


def _run_routing_verifier(
    verifier_path: Path,
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    process = subprocess.run(
        [
            sys.executable,
            str(verifier_path),
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    require(process.returncode == 0, "topic-routing verifier failed")
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise GateStateV12Error("topic-routing verifier emitted invalid JSON") from error
    return _mapping(value, "topic-routing verification")


def validate(
    ledger_path: Path = DEFAULT_LEDGER,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = _read_yaml(ledger_path)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema mismatch")
    require(ledger.get("label") == "pivot_feedback_gate_state_v12", "ledger label")
    require(
        ledger.get("mode") == "ADDITIVE_API_ONLY_ZERO_COST_TOPIC_ROUTING_FREEZE",
        "ledger mode mismatch",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status mismatch")

    parent_binding = _mapping(ledger.get("additive_parent"), "parent")
    parent_path, parent = _validate_document_binding(
        parent_binding,
        repo_root=repo_root,
        label="parent",
    )

    freeze = _mapping(ledger.get("topic_routing_freeze"), "routing freeze")
    config_binding = _mapping(freeze.get("config"), "routing config")
    config_path, config = _validate_document_binding(
        config_binding,
        repo_root=repo_root,
        label="routing config",
    )
    producer_path = _validate_file_binding(
        _mapping(freeze.get("producer"), "producer"),
        repo_root=repo_root,
        label="producer",
    )
    verifier_path = _validate_file_binding(
        _mapping(freeze.get("verifier"), "verifier"),
        repo_root=repo_root,
        label="verifier",
    )
    summary_path, summary = _validate_document_binding(
        _mapping(freeze.get("summary_json"), "summary JSON"),
        repo_root=repo_root,
        label="summary JSON",
    )
    summary_markdown_path = _validate_file_binding(
        _mapping(freeze.get("summary_markdown"), "summary Markdown"),
        repo_root=repo_root,
        label="summary Markdown",
    )
    require(
        config["repository_parent"]["sha256"] == _sha256(parent_path),
        "routing config parent differs",
    )
    require(
        config["authorization"]["paid_api_submission"] == "NOT_AUTHORIZED",
        "routing config permits paid API",
    )
    require(
        config["authorization"]["production_rule_accept_or_reject"]
        == "NOT_AUTHORIZED",
        "routing config permits production dispositions",
    )

    local = _mapping(freeze.get("operator_local_evidence"), "local evidence")
    output_root = _ssd_root(local.get("root_basename"))
    receipt_path = output_root / str(local.get("receipt_basename"))
    require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        "routing receipt missing",
    )
    require(_sha256(receipt_path) == local.get("receipt_sha256"), "receipt SHA")
    verification_report_path = Path("/ssd") / str(
        local.get("verification_report_basename")
    )
    require(
        verification_report_path.is_file()
        and not verification_report_path.is_symlink(),
        "verification report missing",
    )
    require(
        _sha256(verification_report_path)
        == local.get("verification_report_sha256"),
        "verification report SHA",
    )
    frozen_verification = _read_json(verification_report_path)
    require(
        frozen_verification.get("status") == local.get("verification_status"),
        "frozen verification status",
    )
    live_verification = _run_routing_verifier(
        verifier_path,
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
    )
    require(live_verification == frozen_verification, "live verification differs")

    expected_result = {
        "scope": "FROZEN_384_TITLE_DIAGNOSTIC_SAMPLE_ONLY",
        "sample_rows": 384,
        "topic_card_count": 4,
        "no_api_rule_accept_rows": 229,
        "no_api_rule_reject_rows": 24,
        "api_required_ambiguous_rows": 94,
        "quarantine_unresolved_rows": 37,
        "avoided_api_request_rows": 290,
        "counterfactual_api_request_rows": 94,
        "api_request_avoidance_fraction": "0.755208",
        "network_accessed": False,
        "credentials_accessed": False,
        "api_requests_submitted": 0,
        "api_cost_usd": "0.000000",
        "rules_are_ground_truth": False,
        "prior_api_labels_are_ground_truth": False,
        "accuracy_established": False,
        "error_rate_established": False,
        "candidate_universe_screen_complete": False,
        "complete_candidate_work_manifest_materialized": False,
        "production_dispositions_authorized": False,
    }
    result = _mapping(ledger.get("routing_result"), "routing result")
    require(result == expected_result, "routing result differs")
    require(live_verification.get("route_counts") == EXPECTED_ROUTES, "route counts")
    require(
        live_verification.get("avoided_api_request_rows") == 290,
        "avoided API rows",
    )
    require(
        live_verification.get("counterfactual_api_request_rows") == 94,
        "API exception rows",
    )
    require(live_verification.get("accuracy_established") is False, "accuracy opened")

    bindings = _mapping(summary.get("repository_bindings"), "summary bindings")
    require(
        bindings
        == {
            "config_relative_path": config_binding["relative_path"],
            "config_sha256": _sha256(config_path),
            "producer_relative_path": freeze["producer"]["relative_path"],
            "producer_sha256": _sha256(producer_path),
            "verifier_relative_path": freeze["verifier"]["relative_path"],
            "verifier_sha256": _sha256(verifier_path),
        },
        "summary repository bindings differ",
    )
    require(summary.get("overall", {}).get("route_counts") == EXPECTED_ROUTES, "summary routes")
    require(
        summary.get("operator_local_evidence", {}).get("receipt_sha256")
        == _sha256(receipt_path),
        "summary receipt differs",
    )
    require(
        summary.get("operator_local_evidence", {}).get(
            "verification_report_sha256"
        )
        == _sha256(verification_report_path),
        "summary verification differs",
    )
    summary_boundaries = _mapping(summary.get("boundaries"), "summary boundaries")
    for key in (
        "topic_cards_are_semantic_ground_truth",
        "current_openalex_topic_metadata_reconstructs_historical_semantics",
        "prior_api_labels_are_ground_truth",
        "accuracy_established",
        "error_rate_established",
        "candidate_universe_screen_complete",
        "production_dispositions_authorized",
        "paid_api_submission_authorized",
        "episode_reselection_authorized",
        "pivot_classification_authorized",
        "outcomes_read",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(summary_boundaries.get(key) is False, f"summary opened: {key}")
    require(
        summary_boundaries.get("human_review") == "NOT_RUN_BY_DESIGN",
        "summary human state",
    )

    gates = _mapping(ledger.get("gate_state"), "gate state")
    require(list(gates) == EXPECTED_GATE_ORDER, "gate order differs")
    d0 = _mapping(gates.get("D0"), "D0")
    require(
        d0.get("state")
        == "PARTIAL_API_ONLY_TOPIC_ROUTING_DRY_RUN_COMPLETE_MANIFEST_NOT_RUN",
        "D0 state differs",
    )
    require(d0.get("complete") is False, "D0 complete")
    require(
        d0.get("authorizes")
        == [
            "D0_COMPLETE_CANDIDATE_WORK_MANIFEST_DESIGN",
            "D0_ZERO_COST_CANDIDATE_WORK_MANIFEST_MATERIALIZATION",
            "D0_ZERO_COST_COMPLETE_MANIFEST_LOCAL_ROUTING",
        ],
        "D0 authorization differs",
    )
    require(
        "COMPLETE_TITLE_BEARING_CANDIDATE_WORK_MANIFEST"
        in d0.get("blocked_components", []),
        "complete manifest not blocked",
    )
    require(gates["G4"].get("state") == "SEALED_NOT_OPENED", "G4 opened")
    require(gates["G4"].get("complete") is False, "G4 complete")

    authorization = _mapping(ledger.get("authorization"), "authorization")
    for key in (
        "d0_complete_candidate_work_manifest_design",
        "d0_zero_cost_candidate_work_manifest_materialization",
        "d0_zero_cost_complete_manifest_local_routing",
    ):
        require(authorization.get(key) == "AUTHORIZED", f"safe work closed: {key}")
    require(
        authorization.get("d0_human_review") == "NOT_RUN_BY_DESIGN",
        "human review state",
    )
    for key in (
        "d0_paid_api_exception_submission",
        "d0_api_exception_response_ingestion",
        "d0_episode_reselection_successor",
        "d0_author_opportunity_panel_rebuild",
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "g0_measurement_freeze",
        "final_panel_materialization",
        "model_fitting",
        "g4_open",
    ):
        require(
            authorization.get(key) == "NOT_AUTHORIZED",
            f"unsafe authorization: {key}",
        )
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect overstated",
    )

    next_step = _mapping(
        ledger.get("next_required_implementation"),
        "next implementation",
    )
    require(
        next_step.get("gate")
        == "D0_COMPLETE_TITLE_BEARING_CANDIDATE_WORK_MANIFEST",
        "next gate differs",
    )
    require(next_step.get("required_order") == EXPECTED_NEXT_ORDER, "next order")
    require(
        next_step.get(
            "existing_70277_rows_are_topic_year_candidates_not_candidate_works"
        )
        is True,
        "70,277-row boundary missing",
    )
    for key in (
        "api_required_for_every_candidate",
        "human_review_may_be_claimed_complete",
        "human_accuracy_may_be_reported",
        "api_or_rule_accuracy_may_be_reported",
        "unresolved_candidates_may_be_forced_to_decisive_label",
        "paid_submission_may_run_from_this_ledger",
        "current_384_sample_may_be_treated_as_candidate_universe",
        "current_episode_set_may_be_reused_as_final",
        "current_author_panel_may_be_reused_as_final",
        "pivot_distance_may_gate_primary_event",
        "persistence_may_enter_primary_event",
        "entry_or_persistence_outcomes_may_be_opened",
        "model_fitting_may_begin",
        "g4_may_open",
        "gpu_required",
    ):
        require(next_step.get(key) is False, f"unsafe next-step flag: {key}")

    return {
        "schema_version": "pivot_feedback_gate_state_verification.v12",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(ledger_path),
        "parent_sha256": _sha256(parent_path),
        "routing_config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(producer_path),
        "routing_verifier_sha256": _sha256(verifier_path),
        "summary_json_sha256": _sha256(summary_path),
        "summary_markdown_sha256": _sha256(summary_markdown_path),
        "routing_receipt_sha256": _sha256(receipt_path),
        "routing_verification_sha256": _sha256(verification_report_path),
        "sample_rows": 384,
        "route_counts": EXPECTED_ROUTES,
        "avoided_api_request_rows": 290,
        "api_exception_rows": 94,
        "api_request_avoidance_fraction": "0.755208",
        "api_required_for_every_candidate": False,
        "complete_candidate_work_manifest_materialized": False,
        "candidate_universe_screen_complete": False,
        "human_review": "NOT_RUN_BY_DESIGN",
        "paid_api_submission_authorized": False,
        "production_pivot_classification_authorized": False,
        "d0_complete": False,
        "outcomes_read": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "scientific_effect": "NOT_ESTABLISHED",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.ledger)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        GateStateV12Error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V12",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
