#!/usr/bin/env python3
"""Verify the adaptive-frontier proof freeze gate ledger v14."""

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
DEFAULT_LEDGER = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v14.yaml"
SCHEMA = "pivot_feedback_gate_state.v14"
LEDGER_STATUS = (
    "VALID_ADAPTIVE_FRONTIER_PROOF_COMPLETE_CONTENT_DISPOSITIONS_NOT_RUN_"
    "OUTCOMES_FAIL_CLOSED"
)
VERIFY_STATUS = (
    "PASS_ADAPTIVE_FRONTIER_PROOF_COMPLETE_CONTENT_DISPOSITIONS_NOT_RUN_"
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
    "freeze_candidate_content_states_and_fail_closed_precedence",
    "freeze_candidate_level_aggregation_over_earlier_baseline_and_recent_windows",
    "distinguish_indexing_anomaly_from_semantic_off_topic_evidence",
    "freeze_deterministic_frontier_title_sampling_rule",
    "run_zero_cost_dry_run_on_existing_384_title_evidence",
    "report_sensitivity_without_calling_api_or_rules_ground_truth",
    "keep_every_required_candidate_unresolved_if_rule_is_not_decisive",
    "estimate_incremental_frontier_scan_and_ambiguous_api_cost",
    "authorize_raw_title_scan_or_paid_api_only_in_a_versioned_successor",
    "freeze_four_required_content_dispositions_or_expand_frontier",
    "issue_selection_certificate_only_if_all_required_states_are_decisive",
]


class GateStateV14Error(RuntimeError):
    """Fail-closed gate-state v14 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV14Error(message)


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
        raise GateStateV14Error(f"invalid YAML {path.name}: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateStateV14Error(f"invalid JSON {path.name}: {error}") from error
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


def _run_frontier_verifier(
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
    require(process.returncode == 0, "frontier verifier failed")
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise GateStateV14Error("frontier verifier emitted invalid JSON") from error
    return _mapping(value, "frontier verification")


def validate(
    ledger_path: Path = DEFAULT_LEDGER,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = _read_yaml(ledger_path)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema")
    require(ledger.get("label") == "pivot_feedback_gate_state_v14", "ledger label")
    require(
        ledger.get("mode")
        == "ADDITIVE_ADAPTIVE_FRONTIER_PROOF_FREEZE_CONTENT_RULE_PREPARATION",
        "ledger mode",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status")

    parent_path, _ = _validate_document_binding(
        _mapping(ledger.get("additive_parent"), "parent"),
        repo_root=repo_root,
        label="parent",
    )
    freeze = _mapping(ledger.get("adaptive_frontier_freeze"), "frontier freeze")
    config_path, _ = _validate_document_binding(
        _mapping(freeze.get("config"), "frontier config"),
        repo_root=repo_root,
        label="frontier config",
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
    live = _run_frontier_verifier(
        verifier_path,
        config_path=config_path,
        output_root=output_root,
        repo_root=repo_root,
    )
    require(live == frozen_verification, "live frontier replay differs")

    proof = _mapping(ledger.get("proof_result"), "proof result")
    require(
        proof
        == {
            "total_fixture_cases": 299584,
            "eager_pair_cases": 118404,
            "required_state_unresolved_mutation_cases": 331424,
            "structurally_incompatible_not_evaluated_cases": 146832,
            "bounded_fixture_equivalence_passed": True,
            "general_formal_proof_claimed": False,
        },
        "proof result",
    )
    require(live["proof"]["total_fixture_cases"] == 299584, "live fixture cases")
    require(live["proof"]["eager_pair_cases"] == 118404, "live eager pairs")
    frontier = _mapping(ledger.get("frontier_result"), "frontier result")
    require(
        frontier
        == {
            "total_frontier_rows": 51,
            "total_structurally_incompatible_rows": 47,
            "total_content_required_rows": 4,
            "distinct_frontier_topics": 22,
            "content_required_topic_ids": [10579, 12421, 12505, 14430],
            "every_content_required_topic_has_frozen_card": True,
            "content_required_rows_resolved": 0,
            "rank_1_pass_assumption_is_content_disposition": False,
            "selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
        },
        "frontier result",
    )
    require(live["frontier"]["total_frontier_rows"] == 51, "live frontier rows")
    require(
        live["frontier"]["total_content_required_rows"] == 4,
        "live content rows",
    )
    require(
        live["frontier"]["initial_selection_certificate_issued"] is False,
        "live certificate",
    )

    bindings = _mapping(summary.get("repository_bindings"), "summary bindings")
    require(bindings["config_sha256"] == _sha256(config_path), "summary config")
    require(bindings["producer_sha256"] == _sha256(producer_path), "summary producer")
    require(bindings["verifier_sha256"] == _sha256(verifier_path), "summary verifier")
    require(
        summary["exhaustive_small_fixture_proof"]["general_formal_proof_claimed"]
        is False,
        "summary formal proof overclaim",
    )
    require(
        summary["initial_structural_frontier"]["content_required_rows_resolved"] == 0,
        "summary resolved content",
    )
    require(
        summary["initial_structural_frontier"]["selection_certificate_issued"]
        is False,
        "summary certificate",
    )
    for key, value in _mapping(summary.get("boundaries"), "summary boundaries").items():
        if key == "scientific_effect":
            require(value == "NOT_ESTABLISHED", "summary effect")
        else:
            require(value is False, f"summary opened: {key}")

    gates = _mapping(ledger.get("gate_state"), "gate state")
    require(list(gates) == EXPECTED_GATE_ORDER, "gate order")
    d0 = _mapping(gates.get("D0"), "D0")
    require(
        d0.get("state")
        == "PARTIAL_ADAPTIVE_FRONTIER_PROOF_COMPLETE_CONTENT_DISPOSITIONS_NOT_RUN",
        "D0 state",
    )
    require(d0.get("complete") is False, "D0 complete")
    require(
        d0.get("authorizes")
        == [
            "D0_CANDIDATE_CONTENT_DISPOSITION_RULE_DESIGN",
            "D0_FRONTIER_TITLE_SAMPLING_RULE_DESIGN",
            "D0_EXISTING_384_TITLE_ZERO_COST_CANDIDATE_DISPOSITION_DRY_RUN",
        ],
        "D0 authorization",
    )
    require(gates["G4"].get("state") == "SEALED_NOT_OPENED", "G4 opened")

    authorization = _mapping(ledger.get("authorization"), "authorization")
    for key, expected in (
        ("d0_candidate_scope_audit", "COMPLETE_INDEPENDENTLY_VERIFIED"),
        (
            "d0_adaptive_frontier_proof",
            "COMPLETE_INDEPENDENTLY_VERIFIED_BOUNDED_FIXTURES",
        ),
        ("d0_initial_structural_frontier", "COMPLETE_INDEPENDENTLY_VERIFIED"),
        ("d0_human_review", "NOT_RUN_BY_DESIGN"),
        ("d0_candidate_content_disposition_rule_design", "AUTHORIZED"),
        ("d0_frontier_title_sampling_rule_design", "AUTHORIZED"),
        (
            "d0_existing_384_title_candidate_disposition_dry_run",
            "AUTHORIZED_ZERO_COST_ONLY",
        ),
    ):
        require(authorization.get(key) == expected, f"authorization: {key}")
    for key in (
        "d0_raw_openalex_title_scan",
        "d0_new_api_request_creation",
        "d0_paid_api_exception_submission",
        "d0_candidate_content_dispositions",
        "d0_selection_certificate",
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
        next_step.get("gate") == "D0_CANDIDATE_CONTENT_DISPOSITION_RULE",
        "next gate",
    )
    require(next_step.get("required_order") == EXPECTED_NEXT_ORDER, "next order")
    require(
        next_step.get("existing_384_titles_may_be_reused_for_zero_cost_dry_run")
        is True,
        "existing evidence reuse closed",
    )
    for key in (
        "existing_384_titles_may_establish_accuracy",
        "dry_run_may_create_production_content_dispositions",
        "full_eager_title_manifest_required",
        "all_994_topic_cards_required",
        "api_required_for_every_candidate",
        "unresolved_candidate_may_be_skipped",
        "selection_certificate_may_be_issued_now",
        "deterministic_reselection_may_run_now",
        "raw_title_scan_may_run_now",
        "new_api_requests_may_be_created_now",
        "paid_submission_may_run_now",
        "outcomes_may_be_opened",
        "model_fitting_may_begin",
        "g4_may_open",
        "gpu_required",
    ):
        require(next_step.get(key) is False, f"unsafe next-step flag: {key}")

    return {
        "schema_version": "pivot_feedback_gate_state_verification.v14",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(ledger_path),
        "parent_sha256": _sha256(parent_path),
        "frontier_config_sha256": _sha256(config_path),
        "frontier_producer_sha256": _sha256(producer_path),
        "frontier_verifier_sha256": _sha256(verifier_path),
        "frontier_summary_json_sha256": _sha256(summary_path),
        "frontier_summary_markdown_sha256": _sha256(summary_markdown_path),
        "frontier_receipt_sha256": _sha256(receipt_path),
        "frontier_verification_sha256": _sha256(verification_path),
        "bounded_fixture_cases": 299584,
        "bounded_eager_pair_cases": 118404,
        "initial_frontier_rows": 51,
        "structurally_incompatible_rows": 47,
        "content_required_rows": 4,
        "content_required_rows_resolved": 0,
        "general_formal_proof_claimed": False,
        "selection_certificate_issued": False,
        "raw_title_scan_authorized": False,
        "new_api_request_creation_authorized": False,
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
        GateStateV14Error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V14",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
