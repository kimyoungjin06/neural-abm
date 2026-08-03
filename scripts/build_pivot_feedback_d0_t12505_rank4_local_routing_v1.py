#!/usr/bin/env python3
"""Build zero-cost rank-4 T12505 routing and rule-v2 dry-run."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

import build_pivot_feedback_d0_t10543_local_routing_v1 as base
import build_pivot_feedback_d0_t10543_rank3_local_routing_v1 as engine


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_t12505_rank4_local_routing_v1.yaml"
)
DEFAULT_SAMPLE_ROOT = Path(
    "/ssd/pivot-feedback-d0-t12505-rank4-targeted-title-sample-v1-20260729"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-t12505-rank4-local-routing-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_t12505_rank4_local_routing_config.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_t12505_rank4_local_routing_outputs.v1"
STATUS = "PASS_ZERO_COST_T12505_RANK4_LOCAL_ROUTING_AND_RULE_V2_DRY_RUN"


class T12505Rank4LocalRoutingBuildError(RuntimeError):
    """Fail-closed rank-4 local-routing build error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise T12505Rank4LocalRoutingBuildError(message)


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


def _load_config(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config = _read_yaml(path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status") == "FROZEN_T12505_RANK4_LOCAL_ROUTING_DRY_RUN_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = _repo_file(parent)
    parent_document = _read_yaml(parent_path)
    require(parent_document.get("status") == parent["required_status"], "parent status")
    for dotted, expected in parent["required_fields"].items():
        require(_dotted(parent_document, dotted) == expected, f"parent field: {dotted}")
    card_source = _mapping(
        _mapping(config.get("topic_card"), "topic card").get("source"),
        "card source",
    )
    card_path = _repo_file(card_source)
    source_config = _read_yaml(card_path)
    require(card_source["topic_id"] == 12505, "card topic")
    require(card_source["reuse_routing_contract_without_change"] is True, "routing reuse")
    require(card_source["reuse_topic_card_without_change"] is True, "card reuse")
    topic_cards = _mapping(source_config.get("topic_cards"), "topic cards")
    raw_card = topic_cards.get(12505, topic_cards.get("12505"))
    card = _mapping(raw_card, "T12505 card")
    card["topic_id"] = 12505
    card_config = {
        "routing_contract": source_config["routing_contract"],
        "topic_card": card,
    }
    rule_source = _mapping(
        _mapping(config.get("production_rule_v2"), "rule").get("source"),
        "rule source",
    )
    rule_path = _repo_file(rule_source)
    require(rule_source["reuse_without_change"] is True, "rule reuse")
    authorization = _mapping(config.get("authorization"), "authorization")
    for key in (
        "raw_title_scan",
        "new_api_request_creation",
        "api_submission",
        "production_content_disposition",
        "selection_certificate",
        "deterministic_reselection",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(authorization.get(key) == "NOT_AUTHORIZED", f"unsafe authorization: {key}")
    return config, card_config, {
        "parent_sha256": _sha256(parent_path),
        "topic_card_config_sha256": _sha256(card_path),
        "production_rule_v2_config_sha256": _sha256(rule_path),
    }


def _load_sample(
    config: Mapping[str, Any],
    sample_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    binding = _mapping(config.get("frozen_input"), "frozen input")
    root = sample_root.resolve(strict=True)
    require(root.name == binding["root_basename"], "sample root")
    sample_path = root / str(binding["sample_basename"])
    receipt_path = root / str(binding["receipt_basename"])
    verification_path = Path("/ssd") / str(binding["verification_basename"])
    require(_sha256(sample_path) == binding["sample_sha256"], "sample SHA")
    require(_sha256(receipt_path) == binding["receipt_sha256"], "receipt SHA")
    require(_sha256(verification_path) == binding["verification_sha256"], "verification SHA")
    require(_read_json(receipt_path).get("status") == binding["receipt_status"], "sample status")
    require(
        _read_json(verification_path).get("status") == binding["verification_status"],
        "verification status",
    )
    parquet = pq.ParquetFile(sample_path)
    require(parquet.metadata.num_rows == binding["sample_rows"], "sample rows")
    require(
        (parquet.schema_arrow.metadata or {}).get(b"schema_version", b"").decode()
        == binding["sample_schema_version"],
        "sample schema",
    )
    rows = parquet.read().to_pylist()
    require(
        {
            (
                row["field_id"],
                row["field_candidate_rank"],
                row["topic_id"],
                row["entry_year"],
            )
            for row in rows
        }
        == {(27, 4, 12505, 1993)},
        "sample target",
    )
    require(
        Counter(row["window_code"] for row in rows) == {0: 32, 1: 32, 2: 32},
        "sample windows",
    )
    return rows, {
        "sample_sha256": _sha256(sample_path),
        "sample_receipt_sha256": _sha256(receipt_path),
        "sample_verification_sha256": _sha256(verification_path),
    }


def _candidate(
    samples: Sequence[Mapping[str, Any]],
    routes: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = engine._candidate(samples, routes, config)
    candidate.update(
        {
            "field_candidate_rank": 4,
            "topic_id": 12505,
            "topic_name": "Peptidase Inhibition and Analysis",
            "entry_year": 1993,
        }
    )
    return candidate


def _table(row: Mapping[str, Any]) -> pa.Table:
    return engine._table(row).replace_schema_metadata(
        {b"schema_version": b"pivot_feedback_d0_t12505_rank4_rule_v2_dry_run.v1"}
    )


def _markdown(candidate: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# T12505 Medicine rank-4 local routing and rule-v2 dry-run",
            "",
            f"Status: `{STATUS}`",
            "",
            f"- recent accept: {candidate['recent_accept_rows']}/64",
            f"- recent reject: {candidate['recent_reject_rows']}/64",
            f"- recent ambiguous: {candidate['recent_ambiguous_rows']}/64",
            f"- recent opaque quarantine: {candidate['recent_opaque_quarantine_rows']}/64",
            f"- score drop: {candidate['baseline_to_signal_score_drop']:.6f}",
            (
                "- source-share increase: "
                f"{candidate['baseline_to_signal_source_share_increase']:.6f}"
            ),
            f"- rule-v2 branch: `{candidate['candidate_rule_v2_branch']}`",
            f"- rule-v2 state: `{candidate['candidate_rule_v2_state']}`",
            "",
            "No API request, production disposition, certificate, reselection,",
            "outcome read, or model fit occurred.",
            "",
        ]
    )


def build(
    config_path: Path = DEFAULT_CONFIG,
    sample_root: Path = DEFAULT_SAMPLE_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(not output_root.exists(), "output root exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config, card_config, repository_bindings = _load_config(config_path)
    samples, input_evidence = _load_sample(config, sample_root)
    routes = base._route_rows(samples, card_config)
    candidate = _candidate(samples, routes, config)
    expected = _mapping(config.get("expected_dry_run"), "expected")
    for key, value in expected.items():
        require(candidate[key] == value, f"expected dry-run: {key}")
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["routes"] = base._write_csv(
            temporary / str(contract["routes_basename"]),
            routes,
        )
        artifacts["candidate"] = base._write_parquet(
            temporary / str(contract["candidate_basename"]),
            _table(candidate),
        )
        report_path = temporary / str(contract["report_basename"])
        base._write_private(report_path, _markdown(candidate).encode())
        artifacts["report"] = {
            "basename": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": _sha256(report_path),
        }
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "producer_relative_path": (
                Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
            ),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "repository_bindings": repository_bindings,
            "input_evidence": input_evidence,
            "candidate": candidate,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "sample_title_rows_read": 96,
            "raw_openalex_main_table_read": False,
            "new_titles_read": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "production_content_dispositions_created": 0,
            "selection_certificate_issued": False,
            "deterministic_reselection_executed": False,
            "outcomes_read": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        receipt_path = temporary / str(contract["receipt_basename"])
        base._write_private(receipt_path, _canonical_json_bytes(receipt))
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        print(
            json.dumps(
                build(args.config, args.sample_root, args.output_root),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (
        T12505Rank4LocalRoutingBuildError,
        base.T10543LocalRoutingBuildError,
        engine.T10543Rank3LocalRoutingBuildError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        yaml.YAMLError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {"status": "BLOCKED_T12505_RANK4_LOCAL_ROUTING_V1", "error": str(error)},
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
