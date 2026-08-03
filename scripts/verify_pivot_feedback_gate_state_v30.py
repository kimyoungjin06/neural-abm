#!/usr/bin/env python3
"""Verify rank-5 local-abstract recovery and review gate v30."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v30.yaml"
STATUS = "PASS_RANK5_LOCAL_ABSTRACT_RECOVERY_AND_FIVE_ROW_REVIEW_GATE"


class GateStateV30VerificationError(RuntimeError):
    """Fail-closed gate v30 verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV30VerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _repo(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate_path = gate_path.resolve(strict=True)
    gate = _yaml(gate_path)
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v30", "schema")
    require(
        gate.get("status")
        == "VALID_RANK5_LOCAL_ABSTRACT_RECOVERY_AND_FIVE_ROW_DUAL_HIGH_REVIEW_AUTHORIZED",
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_doc = _yaml(_repo(parent))
    require(parent_doc.get("schema_version") == parent["required_schema_version"], "parent schema")
    require(parent_doc.get("status") == parent["required_status"], "parent status")

    terminal = _mapping(gate.get("terminal_parent_evidence"), "terminal")
    operator = _mapping(terminal.get("operator_local"), "operator")
    root = (Path("/ssd") / str(operator["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "operator root")
    receipt = root / str(operator["receipt_basename"])
    verification = Path("/ssd") / str(operator["verification_basename"])
    require(_sha256(receipt) == operator["receipt_sha256"], "terminal receipt SHA")
    require(_sha256(verification) == operator["verification_sha256"], "terminal verification SHA")
    require(_json(receipt).get("status") == operator["receipt_status"], "terminal receipt status")
    require(_json(verification).get("status") == operator["verification_status"], "terminal verification status")
    rank5 = _mapping(terminal.get("rank5_state"), "rank5 state")
    require(
        (
            rank5.get("augmented_accept_rows"),
            rank5.get("augmented_reject_rows"),
            rank5.get("augmented_unresolved_rows"),
        )
        == (27, 21, 16),
        "rank5 counts",
    )
    require(rank5.get("content_state") == "CONTENT_UNRESOLVED", "rank5 state")

    frozen = _mapping(gate.get("frozen_unresolved_input"), "frozen input")
    result_root = (Path("/ssd") / str(frozen["root_basename"])).resolve(strict=True)
    decisions = result_root / str(frozen["decisions_basename"])
    require(_sha256(decisions) == frozen["decisions_sha256"], "decisions SHA")
    with decisions.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == frozen["decisions_rows"] == 25, "decision rows")
    selected = [row for row in rows if row["consensus_route"] == frozen["selected_route"]]
    require(len(selected) == frozen["selected_rows"] == 16, "selected rows")
    ids = sorted(row["openalex_id"] for row in selected)
    ids_sha = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    require(ids_sha == frozen["sorted_openalex_ids_sha256"], "selected ID digest")

    source = _mapping(gate.get("local_abstract_source"), "source")
    require(source.get("required_root_id") == "openalex_works_20260330_repairreplay_20260410_190630", "source root")
    require(source.get("expected_parquet_files") == 493, "source files")
    require(source.get("expected_footer_rows") == 492293207, "source rows")
    scope = _mapping(source.get("read_scope"), "read scope")
    require(scope.get("full_scan_columns") == ["id"], "full scan columns")
    require(scope.get("selected_row_group_columns") == ["id", "value"], "selected columns")
    require(scope.get("full_value_column_scan") is False, "value scan")

    eligibility = _mapping(gate.get("abstract_eligibility_contract"), "eligibility")
    require(eligibility.get("minimum_whitespace_words") == 20, "minimum words")
    require(eligibility.get("maximum_position_collisions") == 0, "collisions")
    require(
        (
            eligibility.get("expected_source_rows_found"),
            eligibility.get("expected_reconstructable_rows"),
            eligibility.get("expected_eligible_review_rows"),
            eligibility.get("expected_missing_rows"),
            eligibility.get("expected_fragment_rows"),
        )
        == (16, 6, 5, 10, 1),
        "eligibility counts",
    )
    require("not content-blind" in str(eligibility.get("freeze_note")), "exposure disclosure")
    require(eligibility.get("local_semantic_labeling") == "NOT_AUTHORIZED", "local labeling")

    review = _mapping(gate.get("abstract_review_contract"), "review")
    require(review.get("exact_review_rows") == 5, "review rows")
    require(review.get("provider_calls") == 2 and review.get("retries") == 0, "provider calls")
    require(review.get("consensus_accept") == "BOTH_LABEL_ON_TOPIC_AND_CONFIDENCE_HIGH", "accept rule")
    require(review.get("consensus_reject") == "BOTH_LABEL_OFF_TOPIC_AND_CONFIDENCE_HIGH", "reject rule")
    require(review.get("human_review") == "NOT_RUN_BY_DESIGN", "human review")
    require(review.get("provider_labels_are_ground_truth") is False, "ground truth")

    cost = _mapping(gate.get("cost_control"), "cost")
    require(float(cost["projected_cumulative_upper_bound_usd"]) <= float(cost["cumulative_authorized_cap_usd"]), "cost cap")
    authorization = _mapping(gate.get("authorization"), "authorization")
    require(authorization.get("d0_rank5_exact_local_abstract_id_scan") == "AUTHORIZED_READ_ONLY", "ID scan")
    require(
        authorization.get("d0_rank5_eligible_abstract_paid_api_submission")
        == "AUTHORIZED_TWO_CALLS_NO_RETRY",
        "API authorization",
    )
    for key in (
        "d0_rank5_production_disposition",
        "d0_selection_certificate",
        "d0_deterministic_reselection",
        "d0_production_pivot_classification",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(str(authorization.get(key)).startswith("NOT_AUTHORIZED"), f"unsafe authorization: {key}")
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v30",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "frozen_unresolved_ids": 16,
        "expected_reconstructable_abstracts": 6,
        "eligible_review_rows_authorized": 5,
        "provider_calls_authorized": 2,
        "human_review": "NOT_RUN_BY_DESIGN",
        "production_disposition_authorized": False,
        "selection_certificate_authorized": False,
        "outcomes_read_authorized": False,
        "g4_open": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(validate(args.gate), indent=2, sort_keys=True))
        return 0
    except (GateStateV30VerificationError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V30", "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
