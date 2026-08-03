#!/usr/bin/env python3
"""Verify additive pivot-feedback gate-state v23."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GATE = REPO_ROOT / "experiments/configs/pivot_feedback_gate_state_v23.yaml"
STATUS = "PASS_FRONTIER_V5_RANK5_CARD_FROZEN_EXACT_SAMPLE_ONLY_OUTCOMES_FAIL_CLOSED"


class GateStateV23VerificationError(RuntimeError):
    """Fail-closed v23 gate verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateStateV23VerificationError(message)


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
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


def _repo_file(binding: Mapping[str, Any]) -> Path:
    path = (REPO_ROOT / str(binding["relative_path"])).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT), "repository binding escapes")
    require(_sha256(path) == binding["sha256"], f"repository SHA: {path.name}")
    return path


def validate(gate_path: Path = DEFAULT_GATE) -> dict[str, Any]:
    gate_path = gate_path.resolve(strict=True)
    gate = _read_yaml(gate_path)
    require(gate.get("schema_version") == "pivot_feedback_gate_state.v23", "schema")
    require(
        gate.get("status")
        == (
            "VALID_FRONTIER_V5_RANK5_REQUIRED_TOPIC_CARD_FROZEN_EXACT_"
            "SAMPLE_AUTHORIZED_OUTCOMES_FAIL_CLOSED"
        ),
        "status",
    )
    parent = _mapping(gate.get("additive_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")

    evidence = _mapping(gate.get("rank4_production_evidence"), "evidence")
    for key in ("config", "producer", "verifier"):
        _repo_file(_mapping(evidence.get(key), key))
    local = _mapping(evidence.get("operator_local"), "operator")
    root = (Path("/ssd") / str(local["root_basename"])).resolve(strict=True)
    dispositions = root / str(local["dispositions_basename"])
    frontier = root / str(local["frontier_basename"])
    receipt = root / str(local["receipt_basename"])
    require(_sha256(dispositions) == local["dispositions_sha256"], "dispositions SHA")
    require(_sha256(frontier) == local["frontier_sha256"], "frontier SHA")
    require(_sha256(receipt) == local["receipt_sha256"], "receipt SHA")
    require(_read_json(receipt).get("status") == local["receipt_status"], "receipt status")
    require(pq.ParquetFile(dispositions).metadata.num_rows == 7, "disposition rows")
    require(pq.ParquetFile(frontier).metadata.num_rows == 51, "frontier rows")

    target = _mapping(gate.get("rank5_target"), "target")
    require(
        (
            target["field_id"],
            target["field_candidate_rank"],
            target["topic_id"],
            target["entry_year"],
            target["subfield_id"],
            target["rows_per_window"],
            target["total_rows"],
        )
        == (27, 5, 10151, 1993, 2721, 32, 96),
        "rank5 target",
    )
    card = _mapping(gate.get("frozen_topic_card"), "card")
    require(
        card.get("freeze_timing") == "BEFORE_T10151_TITLE_SAMPLE_READ",
        "card freeze timing",
    )
    require(card.get("minimum_informative_token_count") == 2, "minimum tokens")
    require(
        card.get("precedence")
        == [
            "QUARANTINE_IF_OPAQUE_OR_UNINFORMATIVE",
            "ACCEPT_IF_POSITIVE_AND_NOT_NEGATIVE",
            "REJECT_IF_NEGATIVE_AND_NOT_POSITIVE",
            "API_REQUIRED_OTHERWISE",
        ],
        "routing precedence",
    )
    positives = card.get("positive_patterns")
    negatives = card.get("negative_patterns")
    require(isinstance(positives, list) and len(positives) == 4, "positive patterns")
    require(isinstance(negatives, list) and len(negatives) == 9, "negative patterns")
    for pattern in [*positives, *negatives, *card["global_opaque_title_patterns"]]:
        re.compile(pattern)

    authorization = _mapping(gate.get("authorization"), "authorization")
    require(
        authorization.get("d0_rank5_topic_card_design")
        == "COMPLETE_FROZEN_BEFORE_SAMPLE_READ",
        "card authorization",
    )
    for key in (
        "d0_exact_rank5_normalized_candidate_scan",
        "d0_exact_rank5_selected_id_title_lookup",
    ):
        require(authorization.get(key) == "AUTHORIZED_READ_ONLY", key)
    require(
        authorization.get("d0_exact_rank5_96_id_sample") == "AUTHORIZED_CREATE_ONLY",
        "sample authorization",
    )
    for key in (
        "d0_rank5_local_routing",
        "d0_rank5_production_content_disposition",
        "d0_general_raw_openalex_title_scan",
        "d0_full_candidate_title_copy",
        "d0_new_api_request_creation",
        "d0_paid_api_exception_submission",
        "d0_selection_certificate",
        "d0_deterministic_reselection",
        "d0_discovery_outcome_opening",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"unsafe authorization: {key}",
        )
    return {
        "schema_version": "pivot_feedback_gate_state_verification.v23",
        "status": STATUS,
        "gate_sha256": _sha256(gate_path),
        "rank5_topic_id": 10151,
        "rank5_entry_year": 1993,
        "topic_card_frozen_before_sample_read": True,
        "exact_rank5_sample_authorized": True,
        "local_routing_authorized": False,
        "new_api_request_creation_authorized": False,
        "outcomes_read_authorized": False,
        "g4_open": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, default=DEFAULT_GATE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(json.dumps(validate(args.gate), indent=2, sort_keys=True))
        return 0
    except (
        GateStateV23VerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_PIVOT_FEEDBACK_GATE_STATE_V23", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
