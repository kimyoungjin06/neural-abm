#!/usr/bin/env python3
"""Independently verify rank-5 terminal disposition and certificate block."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_rank5_terminal_disposition_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-rank5-terminal-disposition-v1-20260729"
)
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-rank5-terminal-disposition-v1-20260729."
    "verification-v1.json"
)
OUTPUT_SCHEMA = "pivot_feedback_d0_rank5_terminal_disposition_outputs.v1"
VERIFY_SCHEMA = "pivot_feedback_d0_rank5_terminal_disposition_verification.v1"
VERIFY_STATUS = "PASS_INDEPENDENT_RANK5_UNRESOLVED_TERMINAL_CERTIFICATE_BLOCK"


class Rank5TerminalDispositionVerificationError(RuntimeError):
    """Fail-closed rank-5 terminal-disposition verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank5TerminalDispositionVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), path.name)


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), path.name)


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


def _ssd_file(binding: Mapping[str, Any], basename_key: str, sha_key: str) -> Path:
    root = (Path("/ssd") / str(binding["root_basename"])).resolve(strict=True)
    require(root.parent == Path("/ssd").resolve(), "SSD root")
    path = root / str(binding[basename_key])
    require(_sha256(path) == binding[sha_key], f"input SHA: {path.name}")
    return path


def _validate_config(path: Path) -> dict[str, Any]:
    config = _read_yaml(path)
    require(
        config.get("schema_version")
        == "pivot_feedback_d0_rank5_terminal_disposition_config.v1",
        "config schema",
    )
    require(
        config.get("status")
        == "FROZEN_RANK5_CONTENT_UNRESOLVED_TERMINAL_CERTIFICATE_BLOCK_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    selection = _mapping(config.get("selection_contract"), "selection")
    selection_document = _read_yaml(_repo_file(selection))
    selection_contract = _mapping(
        selection_document.get("selection_contract"),
        "selection contract",
    )
    require(selection_contract.get("content_unresolved_is_pass") is False, "unresolved pass")
    require(
        _mapping(selection_contract.get("lazy_certificate"), "lazy").get(
            "unresolved_required_row_blocks_certificate"
        )
        is True,
        "unresolved block",
    )
    disposition = _mapping(config.get("rank5_production_disposition"), "disposition")
    require(
        (
            disposition["field_candidate_rank"],
            disposition["topic_id"],
            disposition["entry_year"],
            disposition["content_state"],
            disposition["source_rule_state"],
        )
        == (5, 10151, 1993, "CONTENT_UNRESOLVED", "WOULD_CONTENT_UNRESOLVED"),
        "rank5 disposition",
    )
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "new_candidate_evaluation",
        "topic_card_amendment",
        "new_title_extraction",
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "selection_certificate",
        "deterministic_reselection",
        "author_opportunity_panel_rebuild",
        "production_pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config


def _verify_inputs(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("rank4_production"), "rank4")
    disposition_path = _ssd_file(prior, "dispositions_basename", "dispositions_sha256")
    frontier_path = _ssd_file(prior, "frontier_basename", "frontier_sha256")
    _ssd_file(prior, "receipt_basename", "receipt_sha256")
    dispositions = pq.read_table(disposition_path).to_pylist()
    frontier = pq.read_table(frontier_path).to_pylist()
    require(len(dispositions) == prior["disposition_rows"] == 7, "prior dispositions")
    require(len(frontier) == prior["frontier_rows"] == 51, "prior frontier")
    rank5 = _mapping(frozen.get("rank5_dry_run"), "rank5")
    candidate_path = _ssd_file(rank5, "candidate_basename", "candidate_sha256")
    _ssd_file(rank5, "receipt_basename", "receipt_sha256")
    verification = Path("/ssd") / str(rank5["verification_basename"])
    require(_sha256(verification) == rank5["verification_sha256"], "verification SHA")
    candidate_rows = pq.read_table(candidate_path).to_pylist()
    require(len(candidate_rows) == 1, "candidate rows")
    candidate = candidate_rows[0]
    require(
        (
            candidate["field_candidate_rank"],
            candidate["topic_id"],
            candidate["entry_year"],
            candidate["candidate_rule_v2_state"],
            candidate["candidate_rule_v2_reason"],
        )
        == (
            5,
            10151,
            1993,
            "WOULD_CONTENT_UNRESOLVED",
            "CONSERVATIVE_RECENT_EVIDENCE_NOT_DECISIVE",
        ),
        "rank5 candidate",
    )
    return dispositions, frontier, candidate


def _verify_output(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = _mapping(config.get("output_contract"), "output contract")
    root = output_root.resolve(strict=True)
    require(root.name == contract["root_basename"], "output basename")
    require(root.is_dir() and not root.is_symlink(), "output root")
    require(root.stat().st_mode & 0o777 == 0o700, "output mode")
    paths = {
        "dispositions": root / str(contract["dispositions_basename"]),
        "frontier": root / str(contract["frontier_basename"]),
        "certificate_status": root / str(contract["certificate_status_basename"]),
        "report": root / str(contract["report_basename"]),
    }
    receipt_path = root / str(contract["receipt_basename"])
    require(
        {path.name for path in root.iterdir()}
        == {path.name for path in paths.values()} | {receipt_path.name},
        "artifact set",
    )
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "output schema")
    require(
        receipt.get("status")
        == "PASS_RANK5_CONTENT_UNRESOLVED_TERMINAL_CERTIFICATE_BLOCK_FROZEN",
        "output status",
    )
    require(receipt.get("config_sha256") == config_sha256, "config SHA")
    for key, path in paths.items():
        require(path.is_file() and not path.is_symlink(), f"artifact: {key}")
        require(path.stat().st_mode & 0o777 == 0o600, f"artifact mode: {key}")
        artifact = _mapping(receipt["artifacts"][key], f"artifact: {key}")
        require(path.stat().st_size == artifact["bytes"], f"artifact bytes: {key}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {key}")
    require(receipt_path.stat().st_mode & 0o777 == 0o600, "receipt mode")
    require(
        hashlib.sha256(_canonical_json_bytes(receipt["artifacts"])).hexdigest()
        == receipt["artifact_tree_sha256"],
        "artifact tree",
    )
    return receipt, paths


def _compatible(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        first["subfield_id"] != second["subfield_id"]
        and abs(int(first["entry_year"]) - int(second["entry_year"])) >= 9
    )


def _lazy_pair(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int] | None:
    first: Mapping[str, Any] | None = None
    for row in rows:
        state = row["content_state"]
        if state == "CONTENT_FAIL":
            continue
        if state == "CONTENT_PASS":
            first = row
        break
    if first is None:
        return None
    for row in rows:
        if row["field_candidate_rank"] <= first["field_candidate_rank"]:
            continue
        if not _compatible(first, row):
            continue
        if row["content_state"] == "CONTENT_FAIL":
            continue
        if row["content_state"] != "CONTENT_PASS":
            return None
        return first["field_candidate_rank"], row["field_candidate_rank"]
    return None


def _expected_dispositions(
    prior: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    frozen = _mapping(config.get("rank5_production_disposition"), "disposition")
    row = {
        "field_id": frozen["field_id"],
        "field_candidate_rank": frozen["field_candidate_rank"],
        "topic_id": frozen["topic_id"],
        "entry_year": frozen["entry_year"],
        "content_state": frozen["content_state"],
        "disposition_reason": frozen["disposition_reason"],
        "production_rule_version": frozen["production_rule_version"],
        "source_replay_state": frozen["source_rule_state"],
        "post_diagnostic_amendment_used": frozen["post_diagnostic_amendment_used"],
        "human_review_used": frozen["human_review_used"],
        "api_labels_used_as_ground_truth": frozen["api_labels_used_as_ground_truth"],
    }
    rows = [dict(value) for value in prior] + [row]
    return sorted(rows, key=lambda value: (value["field_id"], value["field_candidate_rank"]))


def _expected_frontier(
    prior: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [dict(value) for value in prior]
    medicine = {row["field_candidate_rank"]: row for row in rows if row["field_id"] == 27}
    medicine[5].update(
        {
            "frontier_role": "FROZEN_PRODUCTION_CONTENT_UNRESOLVED_BLOCKER",
            "content_state": "CONTENT_UNRESOLVED",
            "state_reason": "CONSERVATIVE_RECENT_EVIDENCE_NOT_DECISIVE",
            "targeted_sample_required": False,
        }
    )
    medicine[6].update(
        {
            "frontier_role": "FROZEN_PRODUCTION_PASS_BLOCKED_BY_EARLIER_UNRESOLVED",
            "targeted_sample_required": False,
        }
    )
    return sorted(rows, key=lambda row: (row["field_id"], row["field_candidate_rank"]))


def validate(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config_sha256 = _sha256(config_path)
    config = _validate_config(config_path)
    prior_dispositions, prior_frontier, _ = _verify_inputs(config)
    receipt, paths = _verify_output(
        config,
        output_root,
        config_sha256=config_sha256,
    )
    dispositions_file = pq.ParquetFile(paths["dispositions"])
    frontier_file = pq.ParquetFile(paths["frontier"])
    require(
        (dispositions_file.schema_arrow.metadata or {}).get(b"schema_version")
        == b"pivot_feedback_d0_production_candidate_dispositions.v5",
        "disposition schema",
    )
    require(
        (frontier_file.schema_arrow.metadata or {}).get(b"schema_version")
        == b"pivot_feedback_d0_adaptive_candidate_frontier.v6",
        "frontier schema",
    )
    dispositions = dispositions_file.read().to_pylist()
    frontier = frontier_file.read().to_pylist()
    require(dispositions == _expected_dispositions(prior_dispositions, config), "dispositions")
    require(frontier == _expected_frontier(prior_frontier), "frontier")
    require(len(dispositions) == 8 and len(frontier) == 51, "output rows")
    medicine = [row for row in frontier if row["field_id"] == 27]
    require(_lazy_pair(medicine) is None, "Medicine lazy pair")
    require(
        medicine[4]["content_state"] == "CONTENT_UNRESOLVED"
        and medicine[5]["content_state"] == "CONTENT_PASS",
        "Medicine terminal states",
    )
    computer_science = [row for row in frontier if row["field_id"] == 17]
    require(_lazy_pair(computer_science) == (1, 45), "Computer Science pair")
    certificate = _read_json(paths["certificate_status"])
    expected_certificate = {
        "schema_version": "pivot_feedback_d0_terminal_selection_certificate_status.v1",
        **_mapping(config.get("terminal_certificate_status"), "certificate"),
    }
    require(certificate == expected_certificate, "certificate status")
    require(receipt["terminal_certificate_status"] == certificate, "certificate receipt")
    require(receipt["cumulative_production_dispositions"] == 8, "receipt dispositions")
    require(receipt["computer_science_field_certificate_issued"] is True, "CS certificate")
    for key in (
        "new_candidate_evaluation_executed",
        "medicine_field_certificate_issued",
        "global_selection_certificate_issued",
        "deterministic_reselection_executed",
        "outcomes_read",
        "g4_open",
    ):
        require(receipt[key] is False, f"unsafe receipt flag: {key}")
    require(receipt["new_titles_read"] is False, "new titles")
    require(receipt["api_requests_created"] == 0, "API requests created")
    require(receipt["api_requests_submitted"] == 0, "API requests submitted")
    return {
        "schema_version": VERIFY_SCHEMA,
        "status": VERIFY_STATUS,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(
            output_root / str(config["output_contract"]["receipt_basename"])
        ),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "cumulative_production_dispositions": 8,
        "frontier_rows": 51,
        "computer_science_selected_candidate_ranks": [1, 45],
        "medicine_lazy_pair": None,
        "medicine_blocking_candidate_rank": 5,
        "medicine_blocking_content_state": "CONTENT_UNRESOLVED",
        "terminal_certificate_status": certificate["status"],
        "medicine_field_certificate_issued": False,
        "global_selection_certificate_issued": False,
        "deterministic_reselection_executed": False,
        "api_requests_created": 0,
        "outcomes_read": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.config, args.output_root)
        report_path = args.report.absolute()
        require(not report_path.exists(), "verification report exists")
        require(report_path.parent == Path("/ssd"), "verification report parent")
        temporary = report_path.with_name(f".{report_path.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(result))
        temporary.chmod(0o600)
        os.replace(temporary, report_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        Rank5TerminalDispositionVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_RANK5_TERMINAL_DISPOSITION_VERIFICATION", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
