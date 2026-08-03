#!/usr/bin/env python3
"""Verify the production-rule and exact-target authorization gate v16."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v16.yaml"
VERIFY_STATUS = (
    "PASS_PRODUCTION_CANDIDATE_RULE_FROZEN_EXACT_T10543_LOCAL_SAMPLE_"
    "AUTHORIZED_OUTCOMES_FAIL_CLOSED"
)


class GateStateV16Error(RuntimeError):
    """Fail-closed v16 gate verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV16Error(message)


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
        raise GateStateV16Error(f"invalid YAML: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateStateV16Error(f"invalid JSON: {error}") from error
    return _mapping(value, path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def _dotted(document: Mapping[str, Any], key: str) -> Any:
    value: Any = document
    for part in key.split("."):
        value = _mapping(value, part)[part]
    return value


def _ssd_root(value: Any) -> Path:
    require(isinstance(value, str) and "/" not in value, "unsafe SSD basename")
    path = (Path("/ssd") / value).resolve(strict=True)
    require(path.parent == Path("/ssd").resolve(), "SSD root escapes")
    require(path.is_dir() and not path.is_symlink(), "invalid SSD root")
    return path


def validate(ledger_path: Path = DEFAULT_LEDGER) -> dict[str, Any]:
    ledger_path = ledger_path.resolve(strict=True)
    ledger = _read_yaml(ledger_path)
    require(ledger.get("schema_version") == "pivot_feedback_gate_state.v16", "schema")
    require(
        ledger.get("status")
        == "VALID_PRODUCTION_CANDIDATE_RULE_FROZEN_EXACT_T10543_LOCAL_SAMPLE_AUTHORIZED_OUTCOMES_FAIL_CLOSED",
        "status",
    )
    parent = _mapping(ledger.get("additive_parent"), "parent")
    parent_path = _repo_file(parent)
    parent_document = _read_yaml(parent_path)
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")

    freeze = _mapping(ledger.get("production_rule_freeze"), "rule freeze")
    config_binding = _mapping(freeze.get("config"), "rule config")
    config_path = _repo_file(config_binding)
    config = _read_yaml(config_path)
    require(
        config.get("schema_version") == config_binding["required_schema_version"],
        "rule config schema",
    )
    require(config.get("status") == config_binding["required_status"], "rule config status")
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
    summary_markdown_path = _repo_file(
        _mapping(freeze.get("summary_markdown"), "summary markdown")
    )
    operator = _mapping(freeze.get("operator_local_evidence"), "operator")
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
    require(verification["status"] == operator["verification_status"], "verification")

    production = _mapping(ledger.get("production_result"), "production result")
    require(
        production
        == {
            "production_candidate_dispositions": 4,
            "content_pass_rows": 3,
            "content_fail_rows": 1,
            "computer_science_field_certificate_issued": True,
            "medicine_required_topic_id": 10543,
            "medicine_required_entry_year": 1993,
            "medicine_required_content_state": "CONTENT_UNRESOLVED",
            "global_selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
        },
        "production result",
    )
    require(
        verification["medicine_required_topic_id"] == 10543,
        "verified target topic",
    )
    require(
        verification["global_selection_certificate_issued"] is False,
        "verified global certificate",
    )

    target = _mapping(
        ledger.get("exact_targeted_sample_authorization"),
        "target authorization",
    )
    require(
        {
            "field_id": target["field_id"],
            "field_candidate_rank": target["field_candidate_rank"],
            "topic_id": target["topic_id"],
            "entry_year": target["entry_year"],
            "rows_per_window": target["rows_per_window"],
            "total_rows": target["total_rows"],
        }
        == {
            "field_id": 27,
            "field_candidate_rank": 2,
            "topic_id": 10543,
            "entry_year": 1993,
            "rows_per_window": 32,
            "total_rows": 96,
        },
        "target",
    )
    require(target["exact_work_type_and_author_linkage_abi_required"] is True, "ABI")
    require(target["deterministic_lowest_hash_selection_required"] is True, "sample")
    require(target["exact_selected_id_title_lookup_only"] is True, "title lookup")
    require(target["local_routing_before_api_required"] is True, "local first")
    require(target["api_request_creation"] == "NOT_AUTHORIZED", "API creation")
    require(target["api_submission"] == "NOT_AUTHORIZED", "API submission")

    authorization = _mapping(ledger.get("authorization"), "authorization")
    for key in (
        "d0_exact_t10543_normalized_candidate_scan",
        "d0_exact_t10543_96_id_sample",
        "d0_exact_selected_id_title_lookup",
        "d0_t10543_topic_card_design",
        "d0_t10543_local_routing",
    ):
        require(
            str(authorization.get(key)).startswith("AUTHORIZED"),
            f"targeted work closed: {key}",
        )
    for key in (
        "d0_general_raw_openalex_title_scan",
        "d0_full_candidate_title_copy",
        "d0_new_api_request_creation",
        "d0_paid_api_exception_submission",
        "d0_t10543_production_content_disposition",
        "d0_global_selection_certificate",
        "d0_episode_reselection_successor",
        "d0_author_opportunity_panel_rebuild",
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"unsafe authorization: {key}",
        )
    require(authorization["scientific_effect"] == "NOT_ESTABLISHED", "effect")
    require(ledger["gate_state"]["D0"]["complete"] is False, "D0 complete")
    require(ledger["gate_state"]["G4"]["state"] == "SEALED_NOT_OPENED", "G4")

    return {
        "schema_version": "pivot_feedback_gate_state_verification.v16",
        "status": VERIFY_STATUS,
        "ledger_sha256": _sha256(ledger_path),
        "parent_sha256": _sha256(parent_path),
        "rule_config_sha256": _sha256(config_path),
        "rule_producer_sha256": _sha256(producer_path),
        "rule_verifier_sha256": _sha256(verifier_path),
        "rule_summary_json_sha256": _sha256(summary_path),
        "rule_summary_markdown_sha256": _sha256(summary_markdown_path),
        "rule_receipt_sha256": _sha256(receipt_path),
        "rule_verification_sha256": _sha256(verification_path),
        "production_candidate_dispositions": 4,
        "computer_science_field_certificate_issued": True,
        "medicine_required_topic_id": 10543,
        "exact_target_title_rows_authorized": 96,
        "general_raw_scan_authorized": False,
        "full_title_copy_authorized": False,
        "new_api_request_creation_authorized": False,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_authorized": False,
        "outcomes_read": False,
        "model_fitting_authorized": False,
        "g4_open": False,
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
    except (GateStateV16Error, OSError, ValueError, KeyError, TypeError) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V16",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
