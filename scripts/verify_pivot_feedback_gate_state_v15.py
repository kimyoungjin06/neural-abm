#!/usr/bin/env python3
"""Verify the candidate-content dry-run gate ledger v15."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v15.yaml"
SCHEMA = "pivot_feedback_gate_state.v15"
STATUS = (
    "VALID_CANDIDATE_CONTENT_DRY_RUN_COMPLETE_PRODUCTION_DISPOSITIONS_"
    "NOT_FROZEN_OUTCOMES_FAIL_CLOSED"
)
VERIFY_STATUS = (
    "PASS_CANDIDATE_CONTENT_DRY_RUN_COMPLETE_PRODUCTION_DISPOSITIONS_"
    "NOT_FROZEN_OUTCOMES_FAIL_CLOSED"
)


class GateStateV15Error(RuntimeError):
    """Fail-closed v15 gate verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV15Error(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise GateStateV15Error(f"invalid YAML: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateStateV15Error(f"invalid JSON: {error}") from error
    return _mapping(value, path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "binding escapes repository")
    require(_sha256(path) == binding["sha256"], f"binding SHA: {path.name}")
    return path


def _dotted(document: Mapping[str, Any], key: str) -> Any:
    value: Any = document
    for part in key.split("."):
        value = _mapping(value, part)[part]
    return value


def _ssd_root(basename: Any) -> Path:
    require(
        isinstance(basename, str)
        and basename
        and "/" not in basename
        and basename not in {".", ".."},
        "unsafe SSD basename",
    )
    path = (Path("/ssd") / basename).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "SSD root escapes /ssd")
    require(path.is_dir() and not path.is_symlink(), "invalid SSD root")
    return path


def validate(ledger_path: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    ledger_path = ledger_path.resolve(strict=True)
    ledger = _read_yaml(ledger_path)
    require(ledger.get("schema_version") == SCHEMA, "ledger schema")
    require(ledger.get("status") == STATUS, "ledger status")

    parent = _mapping(ledger.get("additive_parent"), "parent")
    parent_path = _repo_file(parent)
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent["required_schema_version"],
        "parent schema",
    )
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")

    freeze = _mapping(
        ledger.get("candidate_content_dry_run_freeze"),
        "dry-run freeze",
    )
    config_binding = _mapping(freeze.get("config"), "dry-run config")
    config_path = _repo_file(config_binding)
    config = _read_yaml(config_path)
    require(
        config.get("schema_version") == config_binding["required_schema_version"],
        "dry-run config schema",
    )
    require(config.get("status") == config_binding["required_status"], "config status")
    producer_path = _repo_file(_mapping(freeze.get("producer"), "producer"))
    verifier_path = _repo_file(_mapping(freeze.get("verifier"), "verifier"))
    summary_binding = _mapping(freeze.get("summary_json"), "summary")
    summary_path = _repo_file(summary_binding)
    summary = _read_json(summary_path)
    require(
        summary.get("schema_version") == summary_binding["required_schema_version"],
        "summary schema",
    )
    require(summary.get("status") == summary_binding["required_status"], "summary status")
    summary_markdown = _repo_file(
        _mapping(freeze.get("summary_markdown"), "summary markdown")
    )

    operator = _mapping(freeze.get("operator_local_evidence"), "operator evidence")
    output_root = _ssd_root(operator["root_basename"])
    receipt_path = output_root / str(operator["receipt_basename"])
    verification_path = Path("/ssd") / str(operator["verification_report_basename"])
    require(_sha256(receipt_path) == operator["receipt_sha256"], "receipt SHA")
    require(
        _sha256(verification_path) == operator["verification_report_sha256"],
        "verification SHA",
    )
    receipt = _read_json(receipt_path)
    verification = _read_json(verification_path)
    require(
        receipt["artifact_tree_sha256"] == operator["artifact_tree_sha256"],
        "artifact tree",
    )
    require(verification["status"] == operator["verification_status"], "verification status")
    require(
        verification["closure"] == receipt["closure"] == config["expected_dry_run"],
        "independent closure",
    )

    expected_dry_run = {
        "candidate_rows": 4,
        "would_content_pass_rows": 3,
        "would_content_fail_indexing_anomaly_rows": 1,
        "would_content_fail_semantic_off_topic_rows": 0,
        "would_content_unresolved_rows": 0,
        "indexing_anomaly_topic_ids": [12505],
        "production_content_dispositions_created": 0,
        "production_content_required_rows_resolved": 0,
        "selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
        "semantic_ground_truth_established": False,
        "accuracy_established": False,
    }
    require(ledger.get("dry_run_result") == expected_dry_run, "dry-run result")
    require(
        summary["dry_run_result"]["production_content_dispositions_created"] == 0,
        "summary production disposition",
    )
    require(
        summary["dry_run_result"]["production_content_required_rows_resolved"] == 0,
        "summary resolved rows",
    )
    require(
        summary["dry_run_result"]["selection_certificate_issued"] is False,
        "summary certificate",
    )

    expected_sensitivity = {
        "accept_threshold_0_50": {
            "would_pass": 3,
            "would_fail_indexing": 1,
            "would_unresolved": 0,
        },
        "accept_threshold_0_60": {
            "would_pass": 3,
            "would_fail_indexing": 1,
            "would_unresolved": 0,
        },
        "accept_threshold_0_70": {
            "would_pass": 2,
            "would_fail_indexing": 1,
            "would_unresolved": 1,
        },
        "t12505_indexing_anomaly_stable_over_declared_grid": True,
        "t12421_pass_is_threshold_sensitive": True,
    }
    require(ledger.get("sensitivity_result") == expected_sensitivity, "sensitivity")

    counterfactual = _mapping(
        ledger.get("counterfactual_next_frontier"),
        "counterfactual",
    )
    require(counterfactual["applied_to_production_states"] is False, "state applied")
    require(
        counterfactual["next_required_candidate_if_t12505_were_later_frozen_fail"]
        == "F27_RANK2_T10543_1993",
        "next candidate",
    )
    require(
        counterfactual["recent_six_year_author_linked_work_memberships"] == 13080,
        "membership estimate",
    )
    require(counterfactual["target_title_sample_rows"] == 96, "sample estimate")
    for key in (
        "raw_title_scan_authorized",
        "new_api_request_creation_authorized",
        "api_submission_authorized",
    ):
        require(counterfactual[key] is False, f"unsafe counterfactual: {key}")

    gate = _mapping(ledger.get("gate_state"), "gate state")
    require(gate["D0"]["complete"] is False, "D0 complete")
    require(gate["G4"]["state"] == "SEALED_NOT_OPENED", "G4 open")
    authorization = _mapping(ledger.get("authorization"), "authorization")
    for key in (
        "d0_raw_openalex_title_scan",
        "d0_new_title_extraction",
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
    require(authorization.get("scientific_effect") == "NOT_ESTABLISHED", "effect")

    next_step = _mapping(
        ledger.get("next_required_implementation"),
        "next implementation",
    )
    require(next_step["human_review_required"] is False, "human review reintroduced")
    require(next_step["api_required_for_every_candidate"] is False, "API universal")
    for key in (
        "dry_run_may_create_production_dispositions",
        "raw_title_scan_may_run_now",
        "new_title_extraction_may_run_now",
        "new_api_requests_may_be_created_now",
        "paid_submission_may_run_now",
        "selection_certificate_may_be_issued_now",
        "deterministic_reselection_may_run_now",
        "outcomes_may_be_opened",
        "model_fitting_may_begin",
        "g4_may_open",
        "gpu_required",
    ):
        require(next_step[key] is False, f"unsafe next step: {key}")

    return {
        "schema_version": "pivot_feedback_gate_state_verification.v15",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(ledger_path),
        "parent_sha256": _sha256(parent_path),
        "dry_run_config_sha256": _sha256(config_path),
        "dry_run_producer_sha256": _sha256(producer_path),
        "dry_run_verifier_sha256": _sha256(verifier_path),
        "dry_run_summary_json_sha256": _sha256(summary_path),
        "dry_run_summary_markdown_sha256": _sha256(summary_markdown),
        "dry_run_receipt_sha256": _sha256(receipt_path),
        "dry_run_verification_sha256": _sha256(verification_path),
        "candidate_rows": 4,
        "would_content_pass_rows": 3,
        "would_content_fail_indexing_anomaly_rows": 1,
        "production_content_dispositions_created": 0,
        "production_content_required_rows_resolved": 0,
        "selection_certificate_issued": False,
        "next_target_topic_id_if_rule_later_frozen": 10543,
        "next_target_title_sample_rows": 96,
        "raw_title_scan_authorized": False,
        "new_api_request_creation_authorized": False,
        "paid_api_submission_authorized": False,
        "deterministic_reselection_authorized": False,
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
    except (GateStateV15Error, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V15",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
