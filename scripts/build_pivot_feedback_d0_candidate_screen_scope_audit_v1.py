#!/usr/bin/env python3
"""Build the zero-cost candidate-screen scope and router-coverage audit v1."""

from __future__ import annotations

import argparse
from collections import defaultdict
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_candidate_screen_scope_audit_v1.yaml"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/ssd/pivot-feedback-d0-candidate-screen-scope-audit-v1-20260729"
)
CONFIG_SCHEMA = "pivot_feedback_d0_candidate_screen_scope_audit_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_candidate_screen_scope_audit_report.v1"
OUTPUT_SCHEMA = "pivot_feedback_d0_candidate_screen_scope_audit_outputs.v1"
STATUS = "PASS_ZERO_COST_CANDIDATE_SCREEN_SCOPE_AND_ROUTER_COVERAGE_AUDIT"
WINDOW_SCHEMA = "pivot_feedback_d0_candidate_window_scope.v1"
TOPIC_SCHEMA = "pivot_feedback_d0_candidate_topic_scope.v1"


class CandidateScopeAuditError(RuntimeError):
    """Fail-closed candidate-screen scope audit error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateScopeAuditError(message)


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
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise CandidateScopeAuditError(f"invalid YAML {path.name}: {error}") from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateScopeAuditError(f"invalid JSON {path.name}: {error}") from error
    return _mapping(value, path.name)


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


def _schema_version(path: Path) -> str:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    value = metadata.get(b"schema_version")
    require(value is not None, f"missing schema version: {path.name}")
    return value.decode("utf-8")


def _validate_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(
        config.get("status")
        == "FROZEN_ZERO_COST_CANDIDATE_SCREEN_SCOPE_AUDIT_AUTHORIZED",
        "config status mismatch",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = (REPO_ROOT / str(parent["relative_path"])).resolve(strict=True)
    require(parent_path.is_relative_to(REPO_ROOT), "parent escapes repository")
    require(_sha256(parent_path) == parent["sha256"], "parent SHA mismatch")
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent["required_schema_version"],
        "parent schema mismatch",
    )
    require(
        parent_document.get("status") == parent["required_status"],
        "parent status mismatch",
    )

    inputs = _mapping(config.get("frozen_inputs"), "inputs")
    router = _mapping(inputs.get("frozen_topic_router"), "router")
    router_path = (REPO_ROOT / str(router["relative_path"])).resolve(strict=True)
    require(router_path.is_relative_to(REPO_ROOT), "router escapes repository")
    require(_sha256(router_path) == router["sha256"], "router SHA mismatch")
    router_document = _read_yaml(router_path)
    require(
        router_document.get("schema_version") == router["required_schema_version"],
        "router schema mismatch",
    )
    require(
        router_document.get("status") == router["required_status"],
        "router status mismatch",
    )
    require(len(router_document["topic_cards"]) == router["card_count"], "card count")

    boundary = _mapping(config.get("execution_boundary"), "execution boundary")
    require(boundary.get("scope_audit_only") is True, "scope-only flag closed")
    for key in (
        "title_bearing_work_manifest_materialization",
        "raw_openalex_scan",
        "topic_card_expansion",
        "api_request_creation",
        "api_submission",
        "production_dispositions",
        "deterministic_reselection",
        "author_panel_rebuild",
        "pivot_classification",
        "outcomes_read",
        "model_fitting",
        "g4_open",
    ):
        require(
            boundary.get(key) == "NOT_AUTHORIZED_BY_THIS_CONTRACT"
            if key in {
                "title_bearing_work_manifest_materialization",
                "raw_openalex_scan",
            }
            else boundary.get(key) == "NOT_AUTHORIZED",
            f"unsafe execution authorization: {key}",
        )
    return config, {
        "parent_sha256": _sha256(parent_path),
        "router_sha256": _sha256(router_path),
    }


def _load_inputs(
    config: Mapping[str, Any],
) -> tuple[pa.Table, pa.Table, dict[int, dict[str, Any]], dict[str, Any]]:
    inputs = _mapping(config.get("frozen_inputs"), "inputs")
    measurement = _mapping(
        inputs.get("measurement_abi_reselection"),
        "measurement input",
    )
    root = _ssd_root(measurement["root_basename"])
    receipt_path = root / str(measurement["receipt_basename"])
    candidate_path = root / str(measurement["candidate_basename"])
    topic_year_path = root / str(measurement["topic_year_basename"])
    for path in (receipt_path, candidate_path, topic_year_path):
        require(path.is_file() and not path.is_symlink(), f"invalid input: {path.name}")
    require(_sha256(receipt_path) == measurement["receipt_sha256"], "receipt SHA")
    receipt = _read_json(receipt_path)
    require(receipt.get("status") == measurement["required_status"], "receipt status")
    require(_sha256(candidate_path) == measurement["candidate_sha256"], "candidate SHA")
    require(
        _schema_version(candidate_path) == measurement["candidate_schema_version"],
        "candidate schema",
    )
    require(
        pq.ParquetFile(candidate_path).metadata.num_rows
        == measurement["candidate_rows"],
        "candidate rows",
    )
    require(
        _sha256(topic_year_path) == measurement["topic_year_sha256"],
        "topic-year SHA",
    )
    require(
        _schema_version(topic_year_path) == measurement["topic_year_schema_version"],
        "topic-year schema",
    )
    require(
        pq.ParquetFile(topic_year_path).metadata.num_rows
        == measurement["topic_year_rows"],
        "topic-year rows",
    )
    candidates = pq.read_table(candidate_path)
    topic_year = pq.read_table(topic_year_path)

    router = _mapping(inputs.get("frozen_topic_router"), "router")
    router_document = _read_yaml(REPO_ROOT / str(router["relative_path"]))
    cards = {
        int(topic_id): _mapping(card, f"card {topic_id}")
        for topic_id, card in _mapping(
            router_document.get("topic_cards"),
            "topic cards",
        ).items()
    }
    evidence = {
        "measurement_receipt_sha256": _sha256(receipt_path),
        "candidate_sha256": _sha256(candidate_path),
        "topic_year_sha256": _sha256(topic_year_path),
    }
    return candidates, topic_year, cards, evidence


def _materialize_scope(
    candidates: pa.Table,
    topic_year: pa.Table,
    cards: Mapping[int, Mapping[str, Any]],
) -> tuple[pa.Table, pa.Table, dict[str, Any]]:
    counts: dict[tuple[int, int, int], int] = {}
    topic_metadata: dict[tuple[int, int], tuple[str, int, str]] = {}
    for row in topic_year.to_pylist():
        key = (
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["publication_year"]),
        )
        require(key not in counts, "duplicate topic-year cell")
        counts[key] = int(row["author_linked_works"])
        topic_metadata[(key[0], key[1])] = (
            str(row["topic_name"]),
            int(row["subfield_id"]),
            str(row["subfield_name"]),
        )

    window_rows: list[dict[str, Any]] = []
    topic_candidates: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    topic_year_sets: dict[tuple[int, int], set[int]] = defaultdict(set)
    topic_memberships: dict[tuple[int, int], int] = defaultdict(int)
    offsets = [-6, -5, -4, -3, -2, -1]
    candidate_rows = candidates.to_pylist()
    for row_number, row in enumerate(candidate_rows, start=1):
        field_id = int(row["field_id"])
        topic_id = int(row["topic_id"])
        entry_year = int(row["entry_year"])
        topic_key = (field_id, topic_id)
        topic_candidates[topic_key].append(row)
        require(topic_key in topic_metadata, "candidate topic missing from profile")
        for offset in offsets:
            publication_year = entry_year + offset
            linked = counts.get((field_id, topic_id, publication_year), 0)
            topic_year_sets[topic_key].add(publication_year)
            topic_memberships[topic_key] += linked
            window_rows.append(
                {
                    "candidate_row_number": row_number,
                    "field_id": field_id,
                    "field_candidate_rank": int(row["field_candidate_rank"]),
                    "topic_id": topic_id,
                    "topic_name": str(row["topic_name"]),
                    "subfield_id": int(row["subfield_id"]),
                    "entry_year": entry_year,
                    "window_code": 1 if offset <= -4 else 2,
                    "publication_year": publication_year,
                    "author_linked_works": linked,
                    "topic_card_available": topic_id in cards,
                }
            )

    window_schema = pa.schema(
        [
            pa.field("candidate_row_number", pa.uint32(), nullable=False),
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("window_code", pa.uint8(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=False),
            pa.field("author_linked_works", pa.uint64(), nullable=False),
            pa.field("topic_card_available", pa.bool_(), nullable=False),
        ],
        metadata={b"schema_version": WINDOW_SCHEMA.encode()},
    )
    window_table = pa.Table.from_pylist(window_rows, schema=window_schema)

    topic_rows: list[dict[str, Any]] = []
    for (field_id, topic_id), rows in sorted(topic_candidates.items()):
        name, subfield_id, subfield_name = topic_metadata[(field_id, topic_id)]
        years = topic_year_sets[(field_id, topic_id)]
        ranks = [int(row["field_candidate_rank"]) for row in rows]
        entries = [int(row["entry_year"]) for row in rows]
        topic_rows.append(
            {
                "field_id": field_id,
                "topic_id": topic_id,
                "topic_name": name,
                "subfield_id": subfield_id,
                "subfield_name": subfield_name,
                "candidate_rows": len(rows),
                "minimum_field_candidate_rank": min(ranks),
                "maximum_field_candidate_rank": max(ranks),
                "minimum_entry_year": min(entries),
                "maximum_entry_year": max(entries),
                "candidate_window_rows": len(rows) * 6,
                "unique_candidate_window_years": len(years),
                "unique_candidate_window_linked_work_rows": sum(
                    counts.get((field_id, topic_id, year), 0) for year in years
                ),
                "candidate_window_linked_work_memberships": topic_memberships[
                    (field_id, topic_id)
                ],
                "topic_card_available": topic_id in cards,
            }
        )
    topic_schema = pa.schema(
        [
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("topic_name", pa.string(), nullable=False),
            pa.field("subfield_id", pa.uint16(), nullable=False),
            pa.field("subfield_name", pa.string(), nullable=False),
            pa.field("candidate_rows", pa.uint32(), nullable=False),
            pa.field("minimum_field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("maximum_field_candidate_rank", pa.uint32(), nullable=False),
            pa.field("minimum_entry_year", pa.uint16(), nullable=False),
            pa.field("maximum_entry_year", pa.uint16(), nullable=False),
            pa.field("candidate_window_rows", pa.uint32(), nullable=False),
            pa.field("unique_candidate_window_years", pa.uint16(), nullable=False),
            pa.field(
                "unique_candidate_window_linked_work_rows",
                pa.uint64(),
                nullable=False,
            ),
            pa.field(
                "candidate_window_linked_work_memberships",
                pa.uint64(),
                nullable=False,
            ),
            pa.field("topic_card_available", pa.bool_(), nullable=False),
        ],
        metadata={b"schema_version": TOPIC_SCHEMA.encode()},
    )
    topic_table = pa.Table.from_pylist(topic_rows, schema=topic_schema)

    all_topic_rows = topic_table.to_pylist()
    covered = [row for row in all_topic_rows if row["topic_card_available"]]
    closure = {
        "selected_field_ids": sorted(
            {int(row["field_id"]) for row in candidate_rows}
        ),
        "candidate_rows": len(candidate_rows),
        "distinct_candidate_topics": len(all_topic_rows),
        "candidate_window_rows": len(window_table),
        "unique_candidate_topic_year_cells": sum(
            int(row["unique_candidate_window_years"]) for row in all_topic_rows
        ),
        "unique_candidate_window_linked_work_rows": sum(
            int(row["unique_candidate_window_linked_work_rows"])
            for row in all_topic_rows
        ),
        "candidate_window_linked_work_memberships": sum(
            int(row["candidate_window_linked_work_memberships"])
            for row in all_topic_rows
        ),
        "frozen_topic_card_count": len(cards),
        "card_covered_candidate_topics": len(covered),
        "uncovered_candidate_topics": len(all_topic_rows) - len(covered),
        "card_covered_candidate_rows": sum(
            int(row["candidate_rows"]) for row in covered
        ),
        "card_covered_candidate_window_rows": sum(
            int(row["candidate_window_rows"]) for row in covered
        ),
        "card_covered_unique_topic_year_cells": sum(
            int(row["unique_candidate_window_years"]) for row in covered
        ),
        "card_covered_unique_linked_work_rows": sum(
            int(row["unique_candidate_window_linked_work_rows"])
            for row in covered
        ),
        "card_covered_linked_work_memberships": sum(
            int(row["candidate_window_linked_work_memberships"])
            for row in covered
        ),
        "full_router_topic_coverage_fraction": f"{len(covered) / len(all_topic_rows):.6f}",
        "full_router_candidate_row_coverage_fraction": (
            f"{sum(int(row['candidate_rows']) for row in covered) / len(candidate_rows):.6f}"
        ),
    }
    return window_table, topic_table, closure


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    path.chmod(0o600)
    return {
        "basename": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def _write_private(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)


def _markdown(closure: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Candidate-screen scope audit",
            "",
            f"Status: `{STATUS}`",
            "",
            f"- candidate rows: {closure['candidate_rows']:,}",
            f"- candidate topics: {closure['distinct_candidate_topics']:,}",
            f"- six-year candidate-window rows: {closure['candidate_window_rows']:,}",
            (
                "- unique candidate topic-year cells: "
                f"{closure['unique_candidate_topic_year_cells']:,}"
            ),
            (
                "- unique linked works represented by those cells: "
                f"{closure['unique_candidate_window_linked_work_rows']:,}"
            ),
            (
                "- overlapping candidate-window work memberships: "
                f"{closure['candidate_window_linked_work_memberships']:,}"
            ),
            f"- frozen topic cards: {closure['frozen_topic_card_count']:,}",
            (
                "- card-covered candidate topics: "
                f"{closure['card_covered_candidate_topics']:,}"
            ),
            (
                "- card-covered candidate rows: "
                f"{closure['card_covered_candidate_rows']:,}"
            ),
            "",
            "The four-card router cannot screen the 994-topic candidate universe.",
            "This audit read no title, raw OpenAlex main row, credential, API, or outcome.",
            "It does not authorize a full work copy, paid submission, or reselection.",
            "",
        ]
    )


def build(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    require(config_path.is_relative_to(REPO_ROOT), "config escapes repository")
    require(not output_root.exists(), "output root already exists")
    require(output_root.parent.resolve(strict=True) == Path("/ssd"), "output parent")
    config, repository_bindings = _validate_config(config_path)
    output_contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == output_contract["root_basename"], "output basename")
    candidates, topic_year, cards, input_evidence = _load_inputs(config)
    window_table, topic_table, closure = _materialize_scope(
        candidates,
        topic_year,
        cards,
    )
    require(closure == config["expected_closure"], "expected closure mismatch")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=str(output_root.parent))
    )
    temporary.chmod(0o700)
    try:
        artifacts: dict[str, dict[str, Any]] = {}
        artifacts["candidate_window_map"] = _write_parquet(
            window_table,
            temporary / str(output_contract["candidate_window_map_basename"]),
        )
        artifacts["topic_scope_profile"] = _write_parquet(
            topic_table,
            temporary / str(output_contract["topic_scope_profile_basename"]),
        )
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": STATUS,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": _sha256(config_path),
            "repository_bindings": repository_bindings,
            "input_evidence": input_evidence,
            "closure": closure,
            "decision": {
                "current_four_card_router_can_screen_complete_candidate_universe": False,
                "reason": "FOUR_CARDS_COVER_ONLY_4_OF_994_CANDIDATE_TOPICS",
                "full_title_bearing_work_copy_is_required_before_next_design": False,
                "preferred_successor": "DETERMINISTIC_ADAPTIVE_CANDIDATE_FRONTIER",
            },
            "execution": {
                "raw_openalex_main_table_read": False,
                "titles_read": False,
                "work_ids_materialized": False,
                "network_accessed": False,
                "credentials_accessed": False,
                "api_requests_created": 0,
                "api_requests_submitted": 0,
                "api_cost_usd": "0.000000",
                "outcomes_read": False,
            },
            "authorization": {
                "complete_title_bearing_work_manifest": "NOT_AUTHORIZED",
                "topic_card_expansion": "NOT_AUTHORIZED",
                "api_submission": "NOT_AUTHORIZED",
                "production_dispositions": "NOT_AUTHORIZED",
                "deterministic_reselection": "NOT_AUTHORIZED",
                "author_panel_rebuild": "NOT_AUTHORIZED",
                "pivot_classification": "NOT_AUTHORIZED",
                "model_fitting": "NOT_AUTHORIZED",
                "g4_open": "NOT_AUTHORIZED",
                "scientific_effect": "NOT_ESTABLISHED",
            },
        }
        report_path = temporary / str(output_contract["report_basename"])
        _write_private(report_path, _canonical_json_bytes(report))
        artifacts["report"] = {
            "basename": report_path.name,
            "bytes": report_path.stat().st_size,
            "sha256": _sha256(report_path),
        }
        markdown_path = temporary / str(output_contract["markdown_basename"])
        _write_private(markdown_path, _markdown(closure).encode("utf-8"))
        artifacts["markdown"] = {
            "basename": markdown_path.name,
            "bytes": markdown_path.stat().st_size,
            "sha256": _sha256(markdown_path),
        }
        receipt = {
            "schema_version": OUTPUT_SCHEMA,
            "status": STATUS,
            "config_relative_path": config_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(config_path),
            "producer_relative_path": Path(__file__)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix(),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "repository_bindings": repository_bindings,
            "input_evidence": input_evidence,
            "closure": closure,
            "artifacts": artifacts,
            "artifact_tree_sha256": hashlib.sha256(
                _canonical_json_bytes(artifacts)
            ).hexdigest(),
            "raw_openalex_main_table_read": False,
            "titles_read": False,
            "work_ids_materialized": False,
            "network_accessed": False,
            "credentials_accessed": False,
            "api_requests_created": 0,
            "api_requests_submitted": 0,
            "api_cost_usd": "0.000000",
            "outcomes_read": False,
            "complete_candidate_work_manifest_materialized": False,
            "complete_candidate_universe_screened": False,
            "deterministic_reselection_authorized": False,
            "production_dispositions_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
        }
        receipt_path = temporary / str(output_contract["receipt_basename"])
        _write_private(receipt_path, _canonical_json_bytes(receipt))
        os.replace(temporary, output_root)
        temporary = None
        return receipt
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = build(args.config, args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        CandidateScopeAuditError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_CANDIDATE_SCREEN_SCOPE_AUDIT_V1",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
