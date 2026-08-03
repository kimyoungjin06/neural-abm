#!/usr/bin/env python3
"""Verify the API-only pivot-definition freeze gate ledger v11."""

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
DEFAULT_LEDGER = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v11.yaml"
SCHEMA = "pivot_feedback_gate_state.v11"
LEDGER_STATUS = (
    "VALID_API_ONLY_DEFINITION_FROZEN_FULL_SCREEN_NOT_RUN_OUTCOMES_FAIL_CLOSED"
)
VERIFY_STATUS = (
    "PASS_API_ONLY_PIVOT_DEFINITION_FROZEN_FULL_SCREEN_NOT_RUN_OUTCOMES_FAIL_CLOSED"
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
    "freeze_topic_specific_inclusion_exclusion_cards",
    "freeze_api_only_consensus_and_quarantine_rule",
    "materialize_complete_candidate_universe_request_manifest",
    "run_zero_cost_fixture_and_bounded_dry_run",
    "freeze_request_count_cost_estimate_and_operator_cost_cap",
    "obtain_explicit_paid_submission_authorization",
    "execute_pinned_cross_provider_candidate_screen",
    "quarantine_every_unresolved_candidate",
    "independently_verify_full_screen_receipt",
    "deterministically_reselect_attention_episodes",
    "rebuild_author_opportunity_panel",
    "classify_observable_history_new_topic_pivots",
]


class GateStateV11Error(RuntimeError):
    """Fail-closed gate-state v11 validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV11Error(message)


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
    require(path.is_relative_to(repo_root.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _read_yaml(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"invalid YAML: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise GateStateV11Error(f"invalid YAML {path.name}: {error}") from error
    return _mapping(value, path.name)


def _validate_binding(
    raw: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = dict(raw)
    path = _safe_repo_path(repo_root, binding.get("relative_path"), "bound path")
    digest = _sha256(path)
    require(digest == binding.get("sha256"), f"bound SHA mismatch: {path.name}")
    document = _read_yaml(path)
    require(
        document.get("schema_version") == binding.get("required_schema_version"),
        f"bound schema mismatch: {path.name}",
    )
    require(
        document.get("status") == binding.get("required_status"),
        f"bound status mismatch: {path.name}",
    )
    require(binding.get("mutation_allowed") is False, "bound mutation allowed")
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


def _run_definition_verifier(
    repo_root: Path,
    definition_path: Path,
) -> dict[str, Any]:
    verifier = _safe_repo_path(
        repo_root,
        "scripts/verify_pivot_feedback_researcher_pivot_definition_v1.py",
        "definition verifier",
    )
    process = subprocess.run(
        [
            sys.executable,
            str(verifier),
            "--contract",
            str(definition_path),
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    require(process.returncode == 0, "definition verifier failed")
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise GateStateV11Error("definition verifier emitted invalid JSON") from error
    result = _mapping(result, "definition verification")
    require(
        result.get("status")
        == "PASS_SIMPLE_PAST_ONLY_RESEARCHER_PIVOT_DEFINITION_FROZEN",
        "definition verification status differs",
    )
    require(result.get("past_only") is True, "definition is not past-only")
    require(
        result.get("observable_history_claim_only") is True,
        "lifetime-first claim opened",
    )
    require(result.get("pivot_distance_gates_event") is False, "distance gates event")
    require(
        result.get("persistence_is_future_outcome") is True,
        "persistence entered event",
    )
    require(result.get("human_review") == "NOT_RUN_BY_DESIGN", "human state differs")
    require(result.get("human_accuracy") == "NOT_AVAILABLE", "human accuracy invented")
    require(
        result.get("candidate_universe_screen_complete") is False,
        "candidate screen complete",
    )
    require(
        result.get("production_execution_authorized") is False,
        "production classification opened",
    )
    return result


def validate(
    ledger_path: Path = DEFAULT_LEDGER,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    ledger_path = ledger_path.resolve(strict=True)
    ledger = _read_yaml(ledger_path)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema mismatch")
    require(ledger.get("label") == "pivot_feedback_gate_state_v11", "ledger label")
    require(
        ledger.get("mode")
        == "ADDITIVE_API_ONLY_DEFINITION_FREEZE_HUMAN_REVIEW_NOT_RUN_BY_DESIGN",
        "ledger mode mismatch",
    )
    require(ledger.get("status") == LEDGER_STATUS, "ledger status mismatch")

    parent_document, parent = _validate_binding(
        _mapping(ledger.get("additive_parent"), "parent"),
        repo_root=repo_root,
    )
    definition_document, definition = _validate_binding(
        _mapping(ledger.get("definition_freeze"), "definition"),
        repo_root=repo_root,
    )
    definition_binding = _mapping(ledger.get("definition_freeze"), "definition")
    definition_path = _safe_repo_path(
        repo_root,
        definition["relative_path"],
        "definition path",
    )
    definition_verifier_path = _safe_repo_path(
        repo_root,
        definition_binding.get("verifier_relative_path"),
        "definition verifier path",
    )
    require(
        _sha256(definition_verifier_path)
        == definition_binding.get("verifier_sha256"),
        "definition verifier SHA mismatch",
    )
    definition_verification = _run_definition_verifier(repo_root, definition_path)
    require(
        parent_document["api_review_state"]["dual_high_confidence_on_topic_rows"]
        == definition_document["api_only_content_boundary"][
            "current_strict_dual_high_on_topic_rows"
        ]
        == 197,
        "strict API row count differs",
    )
    require(
        parent_document["api_review_state"]["api_escalation_rows"]
        == definition_document["api_only_content_boundary"][
            "current_unresolved_quarantined_rows"
        ]
        == 187,
        "quarantined API row count differs",
    )

    branch = _mapping(ledger.get("api_only_branch"), "API-only branch")
    exact_branch = {
        "human_review": "NOT_RUN_BY_DESIGN",
        "human_calibration": "NOT_RUN_BY_DESIGN",
        "human_accuracy": "NOT_AVAILABLE",
        "human_cohen_kappa": "NOT_AVAILABLE",
        "api_consensus_is_ground_truth": False,
        "prior_human_packets_retained_as_historical_artifacts": True,
        "prior_human_packets_are_active_next_step": False,
    }
    for key, expected in exact_branch.items():
        require(
            type(branch.get(key)) is type(expected) and branch.get(key) == expected,
            f"API-only branch differs: {key}",
        )
    diagnostic = _mapping(
        branch.get("frozen_384_title_diagnostic"),
        "frozen diagnostic",
    )
    require(
        diagnostic
        == {
            "total_rows": 384,
            "strict_dual_high_on_topic_rows": 197,
            "unresolved_quarantined_rows": 187,
            "title_sample_only": True,
            "candidate_universe_screen_complete": False,
            "d0_complete": False,
        },
        "diagnostic scope differs",
    )

    gate_state = _mapping(ledger.get("gate_state"), "gate state")
    require(list(gate_state) == EXPECTED_ORDER, "gate order differs")
    d0 = _mapping(gate_state.get("D0"), "D0")
    require(
        d0.get("state")
        == "PARTIAL_API_ONLY_DIAGNOSTIC_COMPLETE_PIVOT_DEFINITION_FROZEN",
        "D0 state differs",
    )
    require(d0.get("complete") is False, "D0 complete")
    require(
        d0.get("not_run_by_design")
        == [
            "TWO_INDEPENDENT_HUMAN_CALIBRATION_AND_ESCALATION_REVIEWS",
            "THIRD_PARTY_HUMAN_DISAGREEMENT_ADJUDICATION",
            "HUMAN_CALIBRATED_API_ERROR_RATE_ESTIMATION",
        ],
        "human not-run boundary differs",
    )
    require(gate_state["G4"].get("state") == "SEALED_NOT_OPENED", "G4 opened")
    require(gate_state["G4"].get("complete") is False, "G4 complete")

    authorization = _mapping(ledger.get("authorization"), "authorization")
    require(
        authorization.get("d0_frozen_384_api_diagnostic")
        == "COMPLETE_INDEPENDENTLY_VERIFIED",
        "diagnostic completion differs",
    )
    require(
        authorization.get("d0_human_review") == "NOT_RUN_BY_DESIGN",
        "human review state differs",
    )
    require(
        authorization.get("d0_api_only_topic_card_and_request_successor_preparation")
        == "AUTHORIZED",
        "safe successor preparation closed",
    )
    for key in (
        "d0_api_only_paid_candidate_universe_submission",
        "d0_api_only_candidate_universe_response_ingestion",
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
        "effect overstated",
    )

    next_step = _mapping(
        ledger.get("next_required_implementation"),
        "next implementation",
    )
    require(
        next_step.get("gate") == "D0_API_ONLY_CANDIDATE_SCREEN_SUCCESSOR",
        "next gate differs",
    )
    require(next_step.get("required_order") == REQUIRED_NEXT_ORDER, "next order differs")
    for key in (
        "human_review_may_be_claimed_complete",
        "human_accuracy_may_be_reported",
        "human_cohen_kappa_may_be_reported",
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
        "schema_version": "pivot_feedback_gate_state_verification.v11",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(ledger_path),
        "parent": parent,
        "definition": definition,
        "definition_verification_sha256": _sha256(
            definition_verifier_path
        ),
        "definition_contract_sha256": definition_verification["contract_sha256"],
        "human_review": "NOT_RUN_BY_DESIGN",
        "human_accuracy": "NOT_AVAILABLE",
        "human_cohen_kappa": "NOT_AVAILABLE",
        "strict_api_consensus_rows": 197,
        "unresolved_quarantined_rows": 187,
        "candidate_universe_screen_complete": False,
        "paid_candidate_screen_authorized": False,
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
        GateStateV11Error,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V11",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
