#!/usr/bin/env python3
"""Verify the scope-corrected adaptive-frontier gate ledger v13."""

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
DEFAULT_LEDGER = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v13.yaml"
SCHEMA = "pivot_feedback_gate_state.v13"
LEDGER_STATUS = (
    "VALID_SCOPE_AUDIT_COMPLETE_ADAPTIVE_FRONTIER_PROOF_NOT_RUN_"
    "OUTCOMES_FAIL_CLOSED"
)
VERIFY_STATUS = (
    "PASS_SCOPE_AUDIT_COMPLETE_ADAPTIVE_FRONTIER_PROOF_NOT_RUN_"
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
    "freeze_adaptive_frontier_state_machine_and_certificate_schema",
    "prove_eager_lazy_selection_equivalence_on_exhaustive_small_fixtures",
    "independently_verify_certificate_rejection_of_unresolved_or_skipped_competitors",
    "materialize_initial_structural_frontier_from_frozen_candidate_ranks",
    "freeze_candidate_level_content_disposition_rule_without_outcomes",
    "freeze_deterministic_frontier_title_sampling_rule",
    "estimate_frontier_work_scan_and_api_exception_workload",
    "authorize_raw_title_scan_only_in_a_versioned_successor",
    "route_frontier_samples_with_local_rules",
    "freeze_actual_ambiguous_request_count_and_cost",
    "obtain_explicit_paid_submission_authorization",
    "execute_pinned_cross_provider_ambiguous_only_screen",
    "issue_selection_equivalence_certificate_or_remain_blocked",
]
EXPECTED_SCOPE = {
    "candidate_rows": 70277,
    "distinct_candidate_topics": 994,
    "candidate_window_rows": 421662,
    "unique_candidate_topic_year_cells": 99433,
    "unique_candidate_window_linked_work_rows": 22437851,
    "candidate_window_linked_work_memberships": 102900039,
    "frozen_topic_card_count": 4,
    "card_covered_candidate_topics": 4,
    "uncovered_candidate_topics": 990,
    "card_covered_candidate_rows": 236,
    "full_router_topic_coverage_fraction": "0.004024",
    "full_router_candidate_row_coverage_fraction": "0.003358",
    "current_four_card_router_can_screen_complete_candidate_universe": False,
    "complete_title_bearing_work_manifest_materialized": False,
    "complete_candidate_universe_screened": False,
    "adaptive_frontier_equivalence_proven": False,
}


class GateStateV13Error(RuntimeError):
    """Fail-closed gate-state v13 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV13Error(message)


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
        raise GateStateV13Error(f"invalid YAML {path.name}: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateStateV13Error(f"invalid JSON {path.name}: {error}") from error
    return _mapping(value, path.name)


def _validate_document_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    binding = dict(raw)
    path = _safe_repo_path(repo_root, binding.get("relative_path"), label)
    require(_sha256(path) == binding.get("sha256"), f"{label} SHA")
    require(binding.get("mutation_allowed") is False, f"{label} mutable")
    document = _read_json(path) if path.suffix == ".json" else _read_yaml(path)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        f"{label} schema",
    )
    require(
        document.get("status") == binding.get("required_status"),
        f"{label} status",
    )
    for dotted_path, expected in _mapping(
        binding.get("required_fields", {}),
        f"{label} fields",
    ).items():
        observed = _nested(document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"{label} field: {dotted_path}",
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
    require(_sha256(path) == binding.get("sha256"), f"{label} SHA")
    require(binding.get("mutation_allowed") is False, f"{label} mutable")
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


def _run_scope_verifier(
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
    require(process.returncode == 0, "scope verifier failed")
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise GateStateV13Error("scope verifier emitted invalid JSON") from error
    return _mapping(value, "scope verification")


def validate(
    ledger_path: Path = DEFAULT_LEDGER,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = _read_yaml(ledger_path)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema")
    require(ledger.get("label") == "pivot_feedback_gate_state_v13", "ledger label")
    require(
        ledger.get("mode")
        == "ADDITIVE_SCOPE_CORRECTION_ADAPTIVE_FRONTIER_PREPARATION",
        "ledger mode",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status")

    parent_path, _ = _validate_document_binding(
        _mapping(ledger.get("additive_parent"), "parent"),
        repo_root=repo_root,
        label="parent",
    )
    freeze = _mapping(ledger.get("scope_audit_freeze"), "scope freeze")
    config_path, _ = _validate_document_binding(
        _mapping(freeze.get("config"), "scope config"),
        repo_root=repo_root,
        label="scope config",
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

    local = _mapping(freeze.get("operator_local_evidence"), "local evidence")
    output_root = _ssd_root(local.get("root_basename"))
    receipt_path = output_root / str(local.get("receipt_basename"))
    require(receipt_path.is_file() and not receipt_path.is_symlink(), "receipt")
    require(_sha256(receipt_path) == local.get("receipt_sha256"), "receipt SHA")
    verification_path = Path("/ssd") / str(
        local.get("verification_report_basename")
    )
    require(
        verification_path.is_file() and not verification_path.is_symlink(),
        "verification report",
    )
    require(
        _sha256(verification_path) == local.get("verification_report_sha256"),
        "verification report SHA",
    )
    frozen_verification = _read_json(verification_path)
    require(
        frozen_verification.get("status") == local.get("verification_status"),
        "frozen verification status",
    )
    live_verification = _run_scope_verifier(
        verifier_path,
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
    )
    require(live_verification == frozen_verification, "live scope replay differs")
    require(ledger.get("scope_result") == EXPECTED_SCOPE, "scope result")
    closure = live_verification["closure"]
    for key in (
        "candidate_rows",
        "distinct_candidate_topics",
        "candidate_window_rows",
        "unique_candidate_topic_year_cells",
        "unique_candidate_window_linked_work_rows",
        "candidate_window_linked_work_memberships",
        "frozen_topic_card_count",
        "card_covered_candidate_topics",
        "uncovered_candidate_topics",
        "card_covered_candidate_rows",
        "full_router_topic_coverage_fraction",
        "full_router_candidate_row_coverage_fraction",
    ):
        require(closure.get(key) == EXPECTED_SCOPE[key], f"live scope: {key}")

    bindings = _mapping(summary.get("repository_bindings"), "summary bindings")
    require(bindings["config_sha256"] == _sha256(config_path), "summary config")
    require(bindings["producer_sha256"] == _sha256(producer_path), "summary producer")
    require(bindings["verifier_sha256"] == _sha256(verifier_path), "summary verifier")
    require(
        summary.get("scope_closure", {}).get("candidate_rows") == 70277,
        "summary candidates",
    )
    require(
        summary.get("scope_closure", {}).get(
            "unique_candidate_window_linked_work_rows"
        )
        == 22437851,
        "summary work scope",
    )
    require(
        summary.get("router_coverage", {}).get(
            "current_four_card_router_can_screen_complete_candidate_universe"
        )
        is False,
        "summary router promoted",
    )
    require(
        summary.get("decision", {}).get("adaptive_frontier_equivalence_proven")
        is False,
        "summary frontier proof invented",
    )
    for key, value in _mapping(summary.get("boundaries"), "summary boundaries").items():
        if key == "scientific_effect":
            require(value == "NOT_ESTABLISHED", "summary scientific effect")
        else:
            require(value is False, f"summary opened: {key}")

    strategy = _mapping(ledger.get("corrected_strategy"), "strategy")
    require(
        strategy.get("name") == "DETERMINISTIC_ADAPTIVE_CANDIDATE_FRONTIER",
        "strategy name",
    )
    require(
        strategy.get("required_candidate_order") == "FIELD_CANDIDATE_RANK_ASCENDING",
        "candidate order",
    )
    require(
        strategy.get("required_content_states")
        == [
            "CONTENT_PASS",
            "CONTENT_FAIL",
            "CONTENT_UNRESOLVED",
            "NOT_EVALUATED",
        ],
        "content states",
    )
    require(strategy.get("fail_closed_state") == "CONTENT_UNRESOLVED", "fail close")
    require(strategy.get("eager_lazy_equivalence_required") is True, "proof not required")
    for key in (
        "eager_lazy_equivalence_proven",
        "initial_frontier_materialized",
        "candidate_content_rule_frozen",
        "title_sampling_rule_frozen",
        "all_994_topic_cards_required",
        "api_required_for_every_candidate",
    ):
        require(strategy.get(key) is False, f"strategy flag differs: {key}")
    require(
        strategy.get("api_only_for_ambiguous_frontier_samples") is True,
        "ambiguous-only API boundary missing",
    )

    gates = _mapping(ledger.get("gate_state"), "gate state")
    require(list(gates) == EXPECTED_GATE_ORDER, "gate order")
    d0 = _mapping(gates.get("D0"), "D0")
    require(
        d0.get("state")
        == "PARTIAL_SCOPE_AUDIT_COMPLETE_ADAPTIVE_FRONTIER_PROOF_NOT_RUN",
        "D0 state",
    )
    require(d0.get("complete") is False, "D0 complete")
    require(
        d0.get("authorizes")
        == [
            "D0_ADAPTIVE_FRONTIER_CONTRACT_AND_EQUIVALENCE_FIXTURES",
            "D0_ZERO_COST_INITIAL_STRUCTURAL_FRONTIER_MATERIALIZATION",
        ],
        "D0 authorization",
    )
    require(gates["G4"].get("state") == "SEALED_NOT_OPENED", "G4 opened")

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(
        authorization.get("d0_candidate_scope_audit")
        == "COMPLETE_INDEPENDENTLY_VERIFIED",
        "scope audit incomplete",
    )
    require(
        authorization.get("d0_complete_title_bearing_work_manifest")
        == "WITHDRAWN_PENDING_EXECUTABLE_SCREENING_DESIGN",
        "full manifest remains immediate",
    )
    for key in (
        "d0_adaptive_frontier_contract_and_equivalence_fixtures",
        "d0_zero_cost_initial_structural_frontier_materialization",
    ):
        require(authorization.get(key) == "AUTHORIZED", f"safe work closed: {key}")
    for key in (
        "d0_raw_openalex_title_scan",
        "d0_candidate_content_dispositions",
        "d0_topic_card_expansion",
        "d0_paid_api_exception_submission",
        "d0_api_exception_response_ingestion",
        "d0_episode_reselection_successor",
        "d0_author_opportunity_panel_rebuild",
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(
            authorization.get(key) == "NOT_AUTHORIZED",
            f"unsafe authorization: {key}",
        )
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect",
    )

    next_step = _mapping(
        ledger.get("next_required_implementation"),
        "next implementation",
    )
    require(
        next_step.get("gate") == "D0_ADAPTIVE_CANDIDATE_FRONTIER_PROOF",
        "next gate",
    )
    require(next_step.get("required_order") == EXPECTED_NEXT_ORDER, "next order")
    for key in (
        "full_eager_title_manifest_required",
        "all_994_topic_cards_required",
        "api_required_for_every_candidate",
        "unresolved_candidate_may_be_skipped",
        "unevaluated_competitor_may_be_skipped_without_structural_reason",
        "current_four_episode_set_may_be_reused_as_final",
        "current_author_panel_may_be_reused_as_final",
        "paid_submission_may_run_from_this_ledger",
        "outcomes_may_be_opened",
        "model_fitting_may_begin",
        "g4_may_open",
        "gpu_required",
    ):
        require(next_step.get(key) is False, f"unsafe next-step flag: {key}")

    return {
        "schema_version": "pivot_feedback_gate_state_verification.v13",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(ledger_path),
        "parent_sha256": _sha256(parent_path),
        "scope_config_sha256": _sha256(config_path),
        "scope_producer_sha256": _sha256(producer_path),
        "scope_verifier_sha256": _sha256(verifier_path),
        "scope_summary_json_sha256": _sha256(summary_path),
        "scope_summary_markdown_sha256": _sha256(summary_markdown_path),
        "scope_receipt_sha256": _sha256(receipt_path),
        "scope_verification_sha256": _sha256(verification_path),
        "candidate_rows": 70277,
        "candidate_topics": 994,
        "candidate_window_rows": 421662,
        "unique_candidate_window_linked_work_rows": 22437851,
        "candidate_window_linked_work_memberships": 102900039,
        "router_covered_topics": 4,
        "router_covered_candidate_rows": 236,
        "complete_candidate_work_manifest_materialized": False,
        "complete_candidate_universe_screened": False,
        "adaptive_frontier_equivalence_required": True,
        "adaptive_frontier_equivalence_proven": False,
        "raw_title_scan_authorized": False,
        "api_required_for_every_candidate": False,
        "paid_api_submission_authorized": False,
        "deterministic_reselection_authorized": False,
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
        GateStateV13Error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V13",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
