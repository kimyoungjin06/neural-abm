#!/usr/bin/env python3
"""Independently verify rank-4 CONTENT_FAIL and frontier v5."""

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
    / "experiments/configs/pivot_feedback_d0_rank4_production_disposition_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-rank4-production-disposition-v1-20260729"
)
DEFAULT_REPORT = Path(
    "/ssd/pivot-feedback-d0-rank4-production-disposition-v1-20260729."
    "verification-v1.json"
)
VERIFY_STATUS = "PASS_INDEPENDENT_RANK4_CONTENT_FAIL_AND_FRONTIER_V5"


class Rank4ProductionDispositionVerificationError(RuntimeError):
    """Fail-closed rank-4 disposition verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rank4ProductionDispositionVerificationError(message)


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
        == "pivot_feedback_d0_rank4_production_disposition_config.v1",
        "config schema",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_document = _read_yaml(_repo_file(parent))
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "rank5_topic_card_design",
        "rank5_targeted_sample_execution",
        "new_title_extraction",
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "global_selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(
            str(authorization.get(key)).startswith("NOT_AUTHORIZED"),
            f"unsafe authorization: {key}",
        )
    return config


def _verify_output(
    config: Mapping[str, Any],
    output_root: Path,
    *,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    contract = _mapping(config.get("output_contract"), "output")
    root = output_root.resolve(strict=True)
    require(root.name == contract["root_basename"], "output basename")
    require(root.stat().st_mode & 0o777 == 0o700, "output mode")
    paths = {
        "dispositions": root / str(contract["dispositions_basename"]),
        "frontier": root / str(contract["frontier_basename"]),
        "report": root / str(contract["report_basename"]),
    }
    receipt_path = root / str(contract["receipt_basename"])
    require(
        {path.name for path in root.iterdir()}
        == {path.name for path in paths.values()} | {receipt_path.name},
        "artifact set",
    )
    receipt = _read_json(receipt_path)
    require(receipt.get("status") == contract["status"], "output status")
    require(receipt.get("config_sha256") == config_sha256, "config SHA")
    for key, path in paths.items():
        require(path.stat().st_mode & 0o777 == 0o600, f"artifact mode: {key}")
        artifact = _mapping(receipt["artifacts"][key], f"artifact: {key}")
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {key}")
    require(
        hashlib.sha256(_canonical_json_bytes(receipt["artifacts"])).hexdigest()
        == receipt["artifact_tree_sha256"],
        "artifact tree",
    )
    return receipt, paths


def _inputs(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    frozen = _mapping(config.get("frozen_inputs"), "inputs")
    prior = _mapping(frozen.get("rank3_production"), "prior")
    dispositions = pq.read_table(
        _ssd_file(prior, "dispositions_basename", "dispositions_sha256")
    ).to_pylist()
    frontier = pq.read_table(
        _ssd_file(prior, "frontier_basename", "frontier_sha256")
    ).to_pylist()
    rank4 = _mapping(frozen.get("rank4_dry_run"), "rank4")
    candidate = pq.read_table(
        _ssd_file(rank4, "candidate_basename", "candidate_sha256")
    ).to_pylist()[0]
    return dispositions, frontier, candidate


def _compatible(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    return (
        first["subfield_id"] != second["subfield_id"]
        and abs(first["entry_year"] - second["entry_year"]) >= 9
    )


def _lazy_pair(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int] | None:
    first = None
    for row in rows:
        if row["content_state"] == "CONTENT_FAIL":
            continue
        if row["content_state"] == "CONTENT_PASS":
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


def validate(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config_sha256 = _sha256(config_path)
    config = _validate_config(config_path)
    receipt, paths = _verify_output(
        config,
        output_root,
        config_sha256=config_sha256,
    )
    prior_dispositions, prior_frontier, candidate = _inputs(config)
    dispositions = pq.read_table(paths["dispositions"]).to_pylist()
    require(len(dispositions) == 7, "disposition rows")
    key_fields = ("field_id", "field_candidate_rank", "topic_id", "entry_year")
    prior_by_key = {
        tuple(row[field] for field in key_fields): row for row in prior_dispositions
    }
    observed_by_key = {
        tuple(row[field] for field in key_fields): row for row in dispositions
    }
    require(len(prior_by_key) == 6, "prior disposition keys")
    require(len(observed_by_key) == 7, "cumulative disposition keys")
    for key, prior_row in prior_by_key.items():
        require(observed_by_key.get(key) == prior_row, f"prior disposition: {key}")
    rank4 = observed_by_key.get((27, 4, 12505, 1993))
    require(rank4 is not None, "rank4 disposition key")
    require(
        (
            rank4["content_state"],
            rank4["disposition_reason"],
            rank4["post_diagnostic_amendment_used"],
        )
        == (
            "CONTENT_FAIL",
            "RECENT_SAMPLE_MAJORITY_OPAQUE_INDEXING_ARTIFACTS",
            True,
        ),
        "rank4 disposition",
    )
    require(
        candidate["candidate_rule_v2_state"] == "WOULD_CONTENT_FAIL_INDEXING_ANOMALY",
        "rank4 source state",
    )
    frontier = pq.read_table(paths["frontier"]).to_pylist()
    require(len(frontier) == len(prior_frontier) == 51, "frontier rows")
    medicine = [row for row in frontier if row["field_id"] == 27]
    require(
        [row["content_state"] for row in medicine]
        == [
            "CONTENT_FAIL",
            "CONTENT_FAIL",
            "CONTENT_FAIL",
            "CONTENT_FAIL",
            "CONTENT_UNRESOLVED",
            "CONTENT_PASS",
        ],
        "medicine states",
    )
    require(_lazy_pair(medicine) is None, "medicine certificate")
    rank5_pass = [dict(row) for row in medicine]
    rank5_pass[4]["content_state"] = "CONTENT_PASS"
    require(_lazy_pair(rank5_pass) == (5, 6), "rank5 pass branch")
    require(
        medicine[4]["targeted_sample_required"] is True
        and medicine[4]["topic_id"] == 10151
        and medicine[4]["entry_year"] == 1993,
        "rank5 target",
    )
    require(receipt["cumulative_production_dispositions"] == 7, "receipt dispositions")
    require(receipt["global_selection_certificate_issued"] is False, "certificate")
    require(receipt["api_requests_created"] == 0, "API requests")
    require(receipt["outcomes_read"] is False, "outcomes")
    return {
        "schema_version": "pivot_feedback_d0_rank4_production_disposition_verification.v1",
        "status": VERIFY_STATUS,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "producer_sha256": receipt["producer_sha256"],
        "receipt_sha256": _sha256(
            output_root / str(config["output_contract"]["receipt_basename"])
        ),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "cumulative_production_dispositions": 7,
        "rank4_content_state": "CONTENT_FAIL",
        "frontier_v5_required_candidate_rank": 5,
        "frontier_v5_required_topic_id": 10151,
        "frontier_v5_required_entry_year": 1993,
        "medicine_field_certificate_issued": False,
        "global_selection_certificate_issued": False,
        "outcomes_read": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        result = validate(args.config, args.output_root)
        report = args.report.absolute()
        require(not report.exists(), "verification report exists")
        temporary = report.with_name(f".{report.name}.tmp-{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(result))
        temporary.chmod(0o600)
        os.replace(temporary, report)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        Rank4ProductionDispositionVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_RANK4_DISPOSITION_VERIFICATION", "error": str(error)}
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
