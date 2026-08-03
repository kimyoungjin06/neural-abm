#!/usr/bin/env python3
"""Independently verify the candidate-screen scope audit v1."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
RESULT_STATUS = "PASS_ZERO_COST_CANDIDATE_SCREEN_SCOPE_AND_ROUTER_COVERAGE_AUDIT"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_ZERO_COST_CANDIDATE_SCREEN_SCOPE_AND_ROUTER_COVERAGE_AUDIT"
)
WINDOW_SCHEMA = "pivot_feedback_d0_candidate_window_scope.v1"
TOPIC_SCHEMA = "pivot_feedback_d0_candidate_topic_scope.v1"


class CandidateScopeVerificationError(RuntimeError):
    """Fail-closed scope-audit verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateScopeVerificationError(message)


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


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise CandidateScopeVerificationError(
            f"invalid YAML {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CandidateScopeVerificationError(
            f"invalid JSON {path.name}: {error}"
        ) from error
    return _mapping(value, path.name)


def _safe_repo_path(repo_root: Path, value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be text")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (repo_root / value).resolve(strict=True)
    require(path.is_relative_to(repo_root), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
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


def _parquet_schema_version(path: Path) -> str:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    value = metadata.get(b"schema_version")
    require(value is not None, f"missing schema version: {path.name}")
    return value.decode("utf-8")


def _validate_config(
    config_path: Path,
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    config = _read_yaml(config_path)
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema")
    require(
        config.get("status")
        == "FROZEN_ZERO_COST_CANDIDATE_SCREEN_SCOPE_AUDIT_AUTHORIZED",
        "config status",
    )
    parent = _mapping(config.get("repository_parent"), "parent")
    parent_path = _safe_repo_path(
        repo_root,
        parent.get("relative_path"),
        "parent",
    )
    require(_sha256(parent_path) == parent.get("sha256"), "parent SHA")
    parent_document = _read_yaml(parent_path)
    require(
        parent_document.get("schema_version") == parent.get("required_schema_version"),
        "parent schema",
    )
    require(
        parent_document.get("status") == parent.get("required_status"),
        "parent status",
    )
    for dotted_path, expected in _mapping(
        parent.get("required_fields"),
        "parent fields",
    ).items():
        observed = _nested(parent_document, dotted_path)
        require(
            type(observed) is type(expected) and observed == expected,
            f"parent field: {dotted_path}",
        )

    inputs = _mapping(config.get("frozen_inputs"), "inputs")
    router = _mapping(inputs.get("frozen_topic_router"), "router")
    router_path = _safe_repo_path(
        repo_root,
        router.get("relative_path"),
        "router",
    )
    require(_sha256(router_path) == router.get("sha256"), "router SHA")
    router_document = _read_yaml(router_path)
    require(
        router_document.get("schema_version") == router.get("required_schema_version"),
        "router schema",
    )
    require(
        router_document.get("status") == router.get("required_status"),
        "router status",
    )
    require(
        len(router_document["topic_cards"]) == router.get("card_count") == 4,
        "router cards",
    )

    boundary = _mapping(config.get("execution_boundary"), "execution boundary")
    require(boundary.get("scope_audit_only") is True, "scope-only flag")
    require(
        boundary.get("candidate_window_map_materialization") == "AUTHORIZED",
        "window map closed",
    )
    require(
        boundary.get("topic_scope_profile_materialization") == "AUTHORIZED",
        "topic profile closed",
    )
    for key in (
        "title_bearing_work_manifest_materialization",
        "raw_openalex_scan",
    ):
        require(
            boundary.get(key) == "NOT_AUTHORIZED_BY_THIS_CONTRACT",
            f"unsafe scope authorization: {key}",
        )
    for key in (
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
            boundary.get(key) == "NOT_AUTHORIZED",
            f"unsafe scope authorization: {key}",
        )
    decision = _mapping(config.get("decision_rule"), "decision")
    require(
        decision.get("current_four_card_router_can_screen_complete_candidate_universe")
        is False,
        "router promoted",
    )
    require(
        decision.get("adaptive_frontier_equivalence_must_be_proven_before_use")
        is True,
        "frontier proof waived",
    )
    return config, {
        "parent_sha256": _sha256(parent_path),
        "router_sha256": _sha256(router_path),
    }


def _load_inputs(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[int], dict[str, str]]:
    inputs = _mapping(config.get("frozen_inputs"), "inputs")
    measurement = _mapping(
        inputs.get("measurement_abi_reselection"),
        "measurement",
    )
    root = _ssd_root(measurement.get("root_basename"))
    receipt_path = root / str(measurement["receipt_basename"])
    candidate_path = root / str(measurement["candidate_basename"])
    topic_year_path = root / str(measurement["topic_year_basename"])
    for path in (receipt_path, candidate_path, topic_year_path):
        require(path.is_file() and not path.is_symlink(), f"invalid input: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"input mode: {path.name}")
    require(_sha256(receipt_path) == measurement["receipt_sha256"], "receipt SHA")
    receipt = _read_json(receipt_path)
    require(receipt.get("status") == measurement["required_status"], "receipt status")
    require(_sha256(candidate_path) == measurement["candidate_sha256"], "candidate SHA")
    require(
        _parquet_schema_version(candidate_path)
        == measurement["candidate_schema_version"],
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
        _parquet_schema_version(topic_year_path)
        == measurement["topic_year_schema_version"],
        "topic-year schema",
    )
    require(
        pq.ParquetFile(topic_year_path).metadata.num_rows
        == measurement["topic_year_rows"],
        "topic-year rows",
    )
    router = _mapping(inputs.get("frozen_topic_router"), "router")
    router_path = _safe_repo_path(
        repo_root,
        router.get("relative_path"),
        "router",
    )
    router_document = _read_yaml(router_path)
    cards = {
        int(topic_id)
        for topic_id in _mapping(router_document.get("topic_cards"), "cards")
    }
    evidence = {
        "measurement_receipt_sha256": _sha256(receipt_path),
        "candidate_sha256": _sha256(candidate_path),
        "topic_year_sha256": _sha256(topic_year_path),
    }
    return (
        pq.read_table(candidate_path).to_pylist(),
        pq.read_table(topic_year_path).to_pylist(),
        cards,
        evidence,
    )


def _replay(
    candidate_rows: Sequence[Mapping[str, Any]],
    topic_year_rows: Sequence[Mapping[str, Any]],
    cards: set[int],
) -> tuple[pa.Table, pa.Table, dict[str, Any]]:
    counts: dict[tuple[int, int, int], int] = {}
    metadata: dict[tuple[int, int], tuple[str, int, str]] = {}
    for row in topic_year_rows:
        field_id = int(row["field_id"])
        topic_id = int(row["topic_id"])
        year = int(row["publication_year"])
        key = (field_id, topic_id, year)
        require(key not in counts, "duplicate topic-year input")
        counts[key] = int(row["author_linked_works"])
        topic_key = (field_id, topic_id)
        value = (
            str(row["topic_name"]),
            int(row["subfield_id"]),
            str(row["subfield_name"]),
        )
        require(
            topic_key not in metadata or metadata[topic_key] == value,
            "topic metadata differs",
        )
        metadata[topic_key] = value

    window_rows: list[dict[str, Any]] = []
    by_topic: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    years_by_topic: dict[tuple[int, int], set[int]] = defaultdict(set)
    memberships: dict[tuple[int, int], int] = defaultdict(int)
    for row_number, row in enumerate(candidate_rows, start=1):
        field_id = int(row["field_id"])
        topic_id = int(row["topic_id"])
        entry_year = int(row["entry_year"])
        topic_key = (field_id, topic_id)
        require(topic_key in metadata, "candidate topic missing")
        by_topic[topic_key].append(row)
        for offset in (-6, -5, -4, -3, -2, -1):
            year = entry_year + offset
            linked = counts.get((field_id, topic_id, year), 0)
            years_by_topic[topic_key].add(year)
            memberships[topic_key] += linked
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
                    "publication_year": year,
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
    window = pa.Table.from_pylist(window_rows, schema=window_schema)

    topics: list[dict[str, Any]] = []
    for (field_id, topic_id), rows in sorted(by_topic.items()):
        name, subfield_id, subfield_name = metadata[(field_id, topic_id)]
        years = years_by_topic[(field_id, topic_id)]
        ranks = [int(row["field_candidate_rank"]) for row in rows]
        entries = [int(row["entry_year"]) for row in rows]
        topics.append(
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
                "candidate_window_linked_work_memberships": memberships[
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
    topic_table = pa.Table.from_pylist(topics, schema=topic_schema)
    covered = [row for row in topics if row["topic_card_available"]]
    closure = {
        "selected_field_ids": sorted(
            {int(row["field_id"]) for row in candidate_rows}
        ),
        "candidate_rows": len(candidate_rows),
        "distinct_candidate_topics": len(topics),
        "candidate_window_rows": len(window),
        "unique_candidate_topic_year_cells": sum(
            int(row["unique_candidate_window_years"]) for row in topics
        ),
        "unique_candidate_window_linked_work_rows": sum(
            int(row["unique_candidate_window_linked_work_rows"]) for row in topics
        ),
        "candidate_window_linked_work_memberships": sum(
            int(row["candidate_window_linked_work_memberships"]) for row in topics
        ),
        "frozen_topic_card_count": len(cards),
        "card_covered_candidate_topics": len(covered),
        "uncovered_candidate_topics": len(topics) - len(covered),
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
        "full_router_topic_coverage_fraction": f"{len(covered) / len(topics):.6f}",
        "full_router_candidate_row_coverage_fraction": (
            f"{sum(int(row['candidate_rows']) for row in covered) / len(candidate_rows):.6f}"
        ),
    }
    return window, topic_table, closure


def validate(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    output_root = output_root.resolve(strict=True)
    require(output_root.is_dir() and not output_root.is_symlink(), "output root")
    require(output_root.stat().st_mode & 0o777 == 0o700, "output root mode")
    config, repository_bindings = _validate_config(
        config_path,
        repo_root=repo_root,
    )
    contract = _mapping(config.get("output_contract"), "output contract")
    require(output_root.name == contract["root_basename"], "output basename")
    expected_names = {
        contract["candidate_window_map_basename"],
        contract["topic_scope_profile_basename"],
        contract["report_basename"],
        contract["markdown_basename"],
        contract["receipt_basename"],
    }
    require(
        {path.name for path in output_root.iterdir()} == expected_names,
        "output file set",
    )
    for path in output_root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(path.stat().st_mode & 0o777 == 0o600, f"output mode: {path.name}")

    receipt_path = output_root / str(contract["receipt_basename"])
    receipt = _read_json(receipt_path)
    require(receipt.get("schema_version") == OUTPUT_SCHEMA, "receipt schema")
    require(receipt.get("status") == RESULT_STATUS, "receipt status")
    require(receipt.get("config_sha256") == _sha256(config_path), "config SHA")
    producer_path = _safe_repo_path(
        repo_root,
        receipt.get("producer_relative_path"),
        "producer",
    )
    require(receipt.get("producer_sha256") == _sha256(producer_path), "producer SHA")
    require(receipt.get("repository_bindings") == repository_bindings, "bindings")

    artifacts = _mapping(receipt.get("artifacts"), "artifacts")
    require(
        set(artifacts)
        == {"candidate_window_map", "topic_scope_profile", "report", "markdown"},
        "artifact set",
    )
    for raw in artifacts.values():
        artifact = _mapping(raw, "artifact")
        path = output_root / str(artifact["basename"])
        require(_sha256(path) == artifact["sha256"], f"artifact SHA: {path.name}")
        require(path.stat().st_size == artifact["bytes"], f"artifact size: {path.name}")
        if "rows" in artifact:
            require(
                pq.ParquetFile(path).metadata.num_rows == artifact["rows"],
                f"artifact rows: {path.name}",
            )
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(_canonical_json_bytes(artifacts)).hexdigest(),
        "artifact tree SHA",
    )

    candidate_rows, topic_year_rows, cards, input_evidence = _load_inputs(
        config,
        repo_root=repo_root,
    )
    expected_window, expected_topics, closure = _replay(
        candidate_rows,
        topic_year_rows,
        cards,
    )
    require(closure == config.get("expected_closure"), "config closure")
    require(receipt.get("closure") == closure, "receipt closure")
    require(receipt.get("input_evidence") == input_evidence, "input evidence")
    observed_window_path = output_root / str(contract["candidate_window_map_basename"])
    observed_topics_path = output_root / str(contract["topic_scope_profile_basename"])
    require(
        _parquet_schema_version(observed_window_path) == WINDOW_SCHEMA,
        "window schema",
    )
    require(
        _parquet_schema_version(observed_topics_path) == TOPIC_SCHEMA,
        "topic schema",
    )
    require(
        pq.read_table(observed_window_path).equals(expected_window),
        "window replay mismatch",
    )
    require(
        pq.read_table(observed_topics_path).equals(expected_topics),
        "topic replay mismatch",
    )

    report_path = output_root / str(contract["report_basename"])
    report = _read_json(report_path)
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema")
    require(report.get("status") == RESULT_STATUS, "report status")
    require(report.get("closure") == closure, "report closure")
    require(report.get("input_evidence") == input_evidence, "report evidence")
    require(
        report.get("decision", {}).get(
            "current_four_card_router_can_screen_complete_candidate_universe"
        )
        is False,
        "report promotes router",
    )
    for document, label in ((receipt, "receipt"), (report["execution"], "report")):
        for key in (
            "raw_openalex_main_table_read",
            "titles_read",
            "work_ids_materialized",
            "network_accessed",
            "credentials_accessed",
            "outcomes_read",
        ):
            require(document.get(key) is False, f"{label} unsafe flag: {key}")
        require(document.get("api_requests_created") == 0, f"{label} request creation")
        require(
            document.get("api_requests_submitted") == 0,
            f"{label} request submission",
        )
        require(document.get("api_cost_usd") == "0.000000", f"{label} API cost")
    for key in (
        "complete_candidate_work_manifest_materialized",
        "complete_candidate_universe_screened",
        "deterministic_reselection_authorized",
        "production_dispositions_authorized",
        "model_fitting_authorized",
        "g4_open",
    ):
        require(receipt.get(key) is False, f"unsafe receipt flag: {key}")

    return {
        "schema_version": "pivot_feedback_d0_candidate_screen_scope_audit_verification.v1",
        "status": VERIFY_STATUS,
        "config_sha256": _sha256(config_path),
        "producer_sha256": _sha256(producer_path),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
        "receipt_sha256": _sha256(receipt_path),
        "artifact_tree_sha256": receipt["artifact_tree_sha256"],
        "closure": closure,
        "raw_openalex_main_table_read": False,
        "titles_read": False,
        "work_ids_materialized": False,
        "network_accessed": False,
        "credentials_accessed": False,
        "api_requests_submitted": 0,
        "api_cost_usd": "0.000000",
        "complete_candidate_work_manifest_materialized": False,
        "complete_candidate_universe_screened": False,
        "adaptive_frontier_equivalence_proven": False,
        "deterministic_reselection_authorized": False,
        "production_dispositions_authorized": False,
        "outcomes_read": False,
        "model_fitting_authorized": False,
        "g4_open": False,
    }


def _write_report(path: Path, result: Mapping[str, Any]) -> None:
    require(not path.exists(), "verification report already exists")
    require(path.parent.is_dir(), "verification report parent missing")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json_bytes(result))
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = validate(args.config, args.output_root)
        if args.report is not None:
            _write_report(args.report.absolute(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (
        CandidateScopeVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_PIVOT_FEEDBACK_D0_CANDIDATE_SCREEN_SCOPE_AUDIT_V1_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
