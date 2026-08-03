#!/usr/bin/env python3
"""Independently replay and verify the D0-A structural pilot freeze."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "experiments/configs/pivot_feedback_d0_structural_pilot_freeze_v1.yaml"
)
BUILDER_PATH = (
    REPO_ROOT / "scripts/build_pivot_feedback_d0_structural_pilot_freeze.py"
)
CONFIG_SCHEMA = "pivot_feedback_d0_structural_pilot_freeze_config.v1"
REPORT_SCHEMA = "pivot_feedback_d0_structural_pilot_freeze_report.v1"
RECEIPT_SCHEMA = "pivot_feedback_d0_structural_pilot_freeze_outputs.v1"
FIELD_SCHEMA = "pivot_feedback_d0_field_structural_ranking.v1"
TOPIC_YEAR_SCHEMA = "pivot_feedback_d0_selected_field_topic_year_attention.v1"
CANDIDATE_SCHEMA = "pivot_feedback_d0_episode_candidates.v1"
OUTPUT_STATUS = "PASS_D0A_OUTCOME_BLIND_TWO_FIELD_FOUR_EPISODE_STRUCTURAL_FREEZE"
VERIFY_STATUS = (
    "PASS_INDEPENDENT_D0A_OUTCOME_BLIND_STRUCTURAL_FREEZE_"
    "D0B_AUTHOR_PANEL_STILL_NOT_RUN"
)
EXPECTED_FILES = {
    "d0_structural_pilot_freeze.json",
    "d0_structural_pilot_freeze.md",
    "d0_structural_pilot_freeze.outputs.json",
    "episode_candidates.parquet",
    "field_structural_ranking.parquet",
    "selected_field_topic_year_attention.parquet",
}
PARTITIONS = 4096
MAX_CONTROL_BYTES = 32 * 1024 * 1024
FIELD_METRICS = (
    "work_volume",
    "usable_date_rate",
    "history_usable_author_work_rate",
    "stable_orcid_author_work_density",
    "affiliation_row_density",
    "institution_row_density",
    "cap100_graph_nodes",
    "cap100_graph_edges",
)
FORBIDDEN_COLUMNS = {
    "entry",
    "entry_effect_size",
    "persistence",
    "persistence_effect_size",
    "citation_success",
    "overshoot_timing_gap",
    "predictive_score",
    "g4_outcome",
    "selected_best_threshold",
}
ID_RE = {
    "topic": re.compile(r"^https://openalex[.]org/T([1-9][0-9]*)$"),
    "field": re.compile(r"^https://openalex[.]org/fields/([1-9][0-9]*)$"),
    "subfield": re.compile(r"^https://openalex[.]org/subfields/([1-9][0-9]*)$"),
}


class D0StructuralVerificationError(RuntimeError):
    """Fail-closed D0-A independent-verification error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise D0StructuralVerificationError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be an object")
    return dict(value)


def _list(value: Any, label: str) -> list[Any]:
    require(isinstance(value, list), f"{label} must be a list")
    return list(value)


def _nested(value: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        require(
            isinstance(current, Mapping) and part in current,
            f"missing field {dotted_path}",
        )
        current = current[part]
    return current


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D0StructuralVerificationError(f"invalid {label}: {error}") from error


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"missing {label}")
    require(path.stat().st_size <= MAX_CONTROL_BYTES, f"{label} is too large")
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), label)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise D0StructuralVerificationError(f"invalid {label}: {error}") from error


def _schema_version(schema: pa.Schema) -> str | None:
    value = (schema.metadata or {}).get(b"schema_version")
    return None if value is None else value.decode("utf-8")


def _safe_repo_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value, f"{label} must be a string")
    pure = PurePosixPath(value)
    require(not pure.is_absolute() and ".." not in pure.parts, f"unsafe {label}")
    path = (REPO_ROOT / value).resolve(strict=True)
    require(path.is_relative_to(REPO_ROOT.resolve()), f"{label} escapes repository")
    require(path.is_file() and not path.is_symlink(), f"invalid {label}")
    return path


def _write_report(path: Path, result: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve(strict=True)
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    require(
        target != root and not target.is_relative_to(root),
        "verification report must be outside output root",
    )
    payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        target,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(target, 0o600, follow_symlinks=False)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_output_identity(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid output root")
    require(stat.S_IMODE(root.stat().st_mode) == 0o700, "output root mode mismatch")
    require({path.name for path in root.iterdir()} == EXPECTED_FILES, "output file set mismatch")
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink(), f"invalid output: {path.name}")
        require(stat.S_IMODE(path.stat().st_mode) == 0o600, f"output mode mismatch: {path.name}")
    receipt_path = root / "d0_structural_pilot_freeze.outputs.json"
    receipt = _read_json(receipt_path, "D0-A output receipt")
    require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema mismatch")
    require(receipt.get("status") == OUTPUT_STATUS, "receipt status mismatch")
    require(receipt.get("artifact_count") == 5, "receipt artifact count mismatch")
    require(receipt.get("d0a_structural_freeze_complete") is True, "D0-A receipt incomplete")
    require(receipt.get("d0_complete") is False, "receipt improperly completes D0")
    require(
        receipt.get("d0b_outcome_blind_author_panel_authorized") is True,
        "receipt does not authorize bounded D0-B",
    )
    for key in ("g0_authorized", "model_fitting_authorized", "g4_open", "gpu_used"):
        require(receipt.get(key) is False, f"unsafe receipt state: {key}")
    expected_artifacts = EXPECTED_FILES - {receipt_path.name}
    observed = []
    for raw in _list(receipt.get("artifacts"), "receipt artifacts"):
        item = _mapping(raw, "receipt artifact")
        relative = item.get("relative_path")
        require(
            isinstance(relative, str)
            and "/" not in relative
            and relative in expected_artifacts,
            "unexpected artifact path",
        )
        path = root / relative
        actual: dict[str, Any] = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if "rows" in item:
            actual["rows"] = pq.ParquetFile(path).metadata.num_rows
        require(actual == item, f"artifact identity mismatch: {relative}")
        observed.append(actual)
    observed.sort(key=lambda item: str(item["relative_path"]))
    require(
        {item["relative_path"] for item in observed} == expected_artifacts,
        "receipt artifact set mismatch",
    )
    require(
        receipt.get("artifact_tree_sha256")
        == hashlib.sha256(
            _canonical_bytes(
                [
                    (item["relative_path"], item["bytes"], item["sha256"])
                    for item in observed
                ]
            )
        ).hexdigest(),
        "artifact tree mismatch",
    )
    report = _read_json(root / "d0_structural_pilot_freeze.json", "D0-A report")
    require(report.get("schema_version") == REPORT_SCHEMA, "report schema mismatch")
    require(report.get("status") == OUTPUT_STATUS, "report status mismatch")
    scope = _mapping(report.get("scope"), "report scope")
    expected_false = (
        "d0_complete",
        "outcomes_read",
        "author_entry_rows_read",
        "persistence_rows_read",
        "citations_realized_read",
        "predictive_scores_computed",
        "threshold_tuning_performed",
        "model_fitting_performed",
        "g4_rows_read",
        "gpu_used",
    )
    require(scope.get("d0a_structural_freeze_complete") is True, "report D0-A incomplete")
    for key in expected_false:
        require(scope.get(key) is False, f"unsafe report scope: {key}")
    require(scope.get("entry_year_end_inclusive") == 2012, "report cutoff mismatch")
    require(scope.get("scientific_effect") == "NOT_ESTABLISHED", "effect claim overstated")
    return receipt, report, observed


def _validate_repository_bindings(
    config: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _mapping(config.get("repository_bindings"), "repository bindings")
    reported = _mapping(report.get("repository_bindings"), "reported bindings")
    result = {}
    for label, yaml_document in (
        ("gate_state", True),
        ("s0_completion", False),
        ("candidate_universe", True),
        ("taxonomy_reference", False),
    ):
        item = _mapping(contract[label], f"{label} contract")
        path = _safe_repo_path(item.get("relative_path"), f"{label} path")
        observed_sha = _sha256(path)
        require(observed_sha == item.get("sha256"), f"{label} SHA mismatch")
        document = (
            _read_yaml(path, label)
            if yaml_document
            else _read_json(path, label)
        )
        require(
            document.get("schema_version") == item.get("required_schema_version"),
            f"{label} schema mismatch",
        )
        if "required_status" in item:
            require(
                document.get("status") == item.get("required_status"),
                f"{label} status mismatch",
            )
        for dotted_path, expected in _mapping(
            item.get("required_fields", {}),
            f"{label} required fields",
        ).items():
            observed_value = _nested(document, dotted_path)
            require(
                type(observed_value) is type(expected)
                and observed_value == expected,
                f"{label} field mismatch: {dotted_path}",
            )
        result[label] = {
            "relative_path": item["relative_path"],
            "sha256": observed_sha,
            "schema_version": document["schema_version"],
            "status": document.get("status"),
        }
        require(reported.get(label) == result[label], f"reported {label} binding mismatch")
    return {
        "bindings": result,
        "candidate_universe_document": _read_yaml(
            _safe_repo_path(
                contract["candidate_universe"]["relative_path"],
                "candidate universe",
            ),
            "candidate universe",
        ),
    }


def _artifact_map(receipt: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for raw in _list(receipt.get("artifacts"), "input receipt artifacts"):
        item = _mapping(raw, "input receipt artifact")
        relative = str(item.get("relative_path"))
        require(relative not in result, "duplicate input artifact")
        result[relative] = item
    return result


def _validate_small_input(
    root: Path,
    binding: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid normalized input root")
    require(root.name == binding.get("expected_root_basename"), "input root basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), "input receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "input receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "input status mismatch")
    artifact = _mapping(binding.get("artifact"), "input artifact")
    relative = str(artifact["relative_path"])
    require(
        _artifact_map(receipt).get(relative, {}).get("sha256")
        == artifact.get("sha256"),
        "input receipt artifact mismatch",
    )
    path = (root / relative).resolve(strict=True)
    require(path.is_relative_to(root), "input artifact escapes root")
    require(path.is_file() and not path.is_symlink(), "invalid input artifact")
    require(_sha256(path) == artifact.get("sha256"), "input artifact SHA mismatch")
    require(pq.ParquetFile(path).metadata.num_rows == artifact.get("rows"), "input row mismatch")
    return path, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_relative_path": relative,
        "artifact_sha256": artifact["sha256"],
        "artifact_rows": artifact["rows"],
    }


def _validate_work_input(
    root: Path,
    binding: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[list[Path], dict[str, Any], list[tuple[int, int, int, int, int]]]:
    root = root.resolve(strict=True)
    require(root.is_dir() and not root.is_symlink(), "invalid cutoff input")
    require(root.name == binding.get("expected_root_basename"), "cutoff basename mismatch")
    receipt_path = root / str(binding["receipt_basename"])
    require(_sha256(receipt_path) == binding.get("receipt_sha256"), "cutoff receipt SHA mismatch")
    receipt = _read_json(receipt_path, receipt_path.name)
    require(
        receipt.get("schema_version") == binding.get("receipt_schema_version"),
        "cutoff receipt schema mismatch",
    )
    require(receipt.get("status") == binding.get("required_status"), "cutoff status mismatch")
    prefix = str(binding["artifact_prefix"])
    items = [
        item
        for relative, item in _artifact_map(receipt).items()
        if relative.startswith(prefix)
    ]
    items.sort(key=lambda item: str(item["relative_path"]))
    require(len(items) == binding.get("artifact_count") == PARTITIONS, "cutoff partition count mismatch")
    paths = []
    pre_stats = []
    for index, item in enumerate(items):
        relative = f"{prefix}part-{index:04d}.parquet"
        require(item.get("relative_path") == relative, "cutoff sequence mismatch")
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root), "cutoff artifact escapes root")
        require(path.is_file() and not path.is_symlink(), "invalid cutoff artifact")
        current = path.stat()
        require(current.st_size == item.get("bytes"), "cutoff byte mismatch")
        paths.append(path)
        pre_stats.append(
            (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
        )

    def rehash(pair: tuple[Path, Mapping[str, Any]]) -> tuple[str, int, str]:
        path, item = pair
        observed = _sha256(path)
        require(observed == item.get("sha256"), f"cutoff artifact SHA mismatch: {path.name}")
        return str(item["relative_path"]), int(item["bytes"]), observed

    with ThreadPoolExecutor(max_workers=workers) as executor:
        verified = list(executor.map(rehash, zip(paths, items, strict=True)))
    return paths, {
        "root_basename": root.name,
        "receipt_basename": receipt_path.name,
        "receipt_sha256": binding["receipt_sha256"],
        "artifact_prefix": prefix,
        "artifacts_rehashed": len(verified),
        "artifact_bytes": sum(item[1] for item in verified),
        "selected_tree_sha256": hashlib.sha256(
            _canonical_bytes(verified)
        ).hexdigest(),
    }, pre_stats


def _validate_stats_unchanged(
    paths: Sequence[Path],
    pre_stats: Sequence[tuple[int, int, int, int, int]],
) -> None:
    observed = []
    for path in paths:
        current = path.stat()
        observed.append(
            (
                current.st_dev,
                current.st_ino,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            )
        )
    require(observed == list(pre_stats), "cutoff input changed during verification")


def _parse_id(value: Any, entity: str) -> int:
    require(isinstance(value, str), f"{entity} ID is not text")
    match = ID_RE[entity].fullmatch(value)
    require(match is not None, f"invalid {entity} ID")
    return int(match.group(1))


def _taxonomy(
    path: Path,
    binding: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), "invalid taxonomy projection")
    require(path.name == binding.get("expected_basename"), "taxonomy basename mismatch")
    require(_sha256(path) == binding.get("sha256"), "taxonomy SHA mismatch")
    parquet = pq.ParquetFile(path)
    require(parquet.metadata.num_rows == binding.get("rows"), "taxonomy row mismatch")
    required_columns = _mapping(binding.get("required_columns"), "taxonomy columns")
    for name, expected_type in required_columns.items():
        require(name in parquet.schema_arrow.names, f"taxonomy column absent: {name}")
        require(
            str(parquet.schema_arrow.field(name).type) == expected_type,
            f"taxonomy type mismatch: {name}",
        )
    rows = pq.read_table(path, columns=list(required_columns)).to_pylist()
    result = {}
    for row in rows:
        topic_id = _parse_id(row["id"], "topic")
        require(topic_id not in result, "duplicate taxonomy topic")
        result[topic_id] = {
            "field_id": _parse_id(row["field__id"], "field"),
            "field_name": str(row["field__display_name"]),
            "subfield_id": _parse_id(row["subfield__id"], "subfield"),
            "subfield_name": str(row["subfield__display_name"]),
            "topic_name": str(row["display_name"]),
        }
    require(len(result) == binding.get("rows"), "taxonomy unique row closure failed")
    return result, {
        "basename": path.name,
        "sha256": binding["sha256"],
        "rows": len(result),
        "role": binding["role"],
        "canonical_taxonomy_parent_sha256": binding[
            "canonical_taxonomy_parent_sha256"
        ],
    }


def _candidate_fields(document: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    universe = _mapping(document.get("candidate_universe"), "candidate universe")
    result = {}
    for raw_domain in _list(universe.get("domains"), "domains"):
        domain = _mapping(raw_domain, "domain")
        for raw_field in _list(domain.get("fields"), "fields"):
            field = _mapping(raw_field, "field")
            field_id = int(field["field_id"])
            require(field_id not in result, "duplicate field")
            result[field_id] = {
                "domain_id": str(domain["domain_id"]),
                "domain_name": str(domain["display_name"]),
                "domain_rng_code": int(domain["rng_code"]),
                "field_name": str(field["display_name"]),
                "field_slug": str(field["field_slug"]),
            }
    require(set(result) == set(range(11, 37)), "candidate universe differs")
    return result


def _aggregate_rows(
    path: Path,
    *,
    entry_year_end: int,
    cap100_only: bool = False,
) -> dict[int, dict[str, int]]:
    result: dict[int, dict[str, int]] = {}
    table = pq.read_table(path)
    for row in table.to_pylist():
        year = row["publication_year"]
        if year is None or int(year) > entry_year_end:
            continue
        if cap100_only and int(row["declared_team_cap"]) != 100:
            continue
        field_id = int(row["field_id"])
        cell = result.setdefault(field_id, {})
        for name, value in row.items():
            if name in {"field_id", "publication_year", "declared_team_cap"}:
                continue
            if isinstance(value, int):
                cell[name] = cell.get(name, 0) + value
    return result


def _min_rank_percentiles(values: Mapping[int, Fraction]) -> dict[int, Fraction]:
    ordered = sorted(values.values())
    require(len(ordered) == 26, "metric percentile field count mismatch")
    return {
        field_id: Fraction(ordered.index(value), 25)
        for field_id, value in values.items()
    }


def _replay_field_ranking(
    core_path: Path,
    orcid_path: Path,
    graph_path: Path,
    *,
    fields: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    core = _aggregate_rows(core_path, entry_year_end=2012)
    orcid = _aggregate_rows(orcid_path, entry_year_end=2012)
    graph = _aggregate_rows(graph_path, entry_year_end=2012, cap100_only=True)
    require(set(core) == set(orcid) == set(graph) == set(range(11, 37)), "field margins differ")
    raw = {}
    metrics: dict[str, dict[int, Fraction]] = {
        metric: {} for metric in FIELD_METRICS
    }
    for field_id in range(11, 37):
        c, o, g = core[field_id], orcid[field_id], graph[field_id]
        require(
            c["total_canonical_works"] > 0
            and c["distinct_author_work_pairs"] > 0
            and o["grammar_valid_author_id_rows"] > 0,
            "zero structural denominator",
        )
        values = {
            "work_volume": Fraction(c["total_canonical_works"]),
            "usable_date_rate": Fraction(
                c["usable_date_works"],
                c["total_canonical_works"],
            ),
            "history_usable_author_work_rate": Fraction(
                c["history_usable_distinct_author_work_pairs"],
                c["distinct_author_work_pairs"],
            ),
            "stable_orcid_author_work_density": Fraction(
                o["stable_orcid_author_work_pairs"],
                o["grammar_valid_author_id_rows"],
            ),
            "affiliation_row_density": Fraction(
                o["author_work_affiliation_rows"],
                o["grammar_valid_author_id_rows"],
            ),
            "institution_row_density": Fraction(
                o["author_work_institution_rows"],
                o["grammar_valid_author_id_rows"],
            ),
            "cap100_graph_nodes": Fraction(g["graph_node_tokens"]),
            "cap100_graph_edges": Fraction(g["distinct_undirected_edges"]),
        }
        for name, value in values.items():
            metrics[name][field_id] = value
        raw[field_id] = {
            "field_id": field_id,
            **fields[field_id],
            "work_volume": c["total_canonical_works"],
            "usable_date_works": c["usable_date_works"],
            "distinct_author_work_pairs": c["distinct_author_work_pairs"],
            "history_usable_distinct_author_work_pairs": c[
                "history_usable_distinct_author_work_pairs"
            ],
            "grammar_valid_author_id_rows": o["grammar_valid_author_id_rows"],
            "stable_orcid_author_work_pairs": o["stable_orcid_author_work_pairs"],
            "author_work_affiliation_rows": o["author_work_affiliation_rows"],
            "author_work_institution_rows": o["author_work_institution_rows"],
            "cap100_graph_nodes": g["graph_node_tokens"],
            "cap100_graph_edges": g["distinct_undirected_edges"],
        }
    percentiles = {
        name: _min_rank_percentiles(values)
        for name, values in metrics.items()
    }
    scores = {}
    for field_id in range(11, 37):
        values = [percentiles[name][field_id] for name in FIELD_METRICS]
        scores[field_id] = (
            min(values),
            sum(values, Fraction()) / len(values),
        )
    ranked = sorted(
        range(11, 37),
        key=lambda field_id: (
            scores[field_id][0],
            scores[field_id][1],
            -field_id,
        ),
        reverse=True,
    )
    selected = [
        ranked[0],
        next(
            field_id
            for field_id in ranked[1:]
            if fields[field_id]["domain_id"] != fields[ranked[0]]["domain_id"]
        ),
    ]
    rows = []
    for rank, field_id in enumerate(ranked, start=1):
        row = {
            **raw[field_id],
            "usable_date_rate": float(metrics["usable_date_rate"][field_id]),
            "history_usable_author_work_rate": float(
                metrics["history_usable_author_work_rate"][field_id]
            ),
            "stable_orcid_author_work_density": float(
                metrics["stable_orcid_author_work_density"][field_id]
            ),
            "affiliation_row_density": float(
                metrics["affiliation_row_density"][field_id]
            ),
            "institution_row_density": float(
                metrics["institution_row_density"][field_id]
            ),
        }
        for name in FIELD_METRICS:
            row[f"{name}_percentile"] = float(percentiles[name][field_id])
        row.update(
            {
                "minimum_metric_percentile": float(scores[field_id][0]),
                "mean_metric_percentile": float(scores[field_id][1]),
                "global_rank": rank,
                "pilot_selected": field_id in selected,
                "pilot_selection_order": (
                    selected.index(field_id) + 1
                    if field_id in selected
                    else None
                ),
            }
        )
        rows.append(row)
    return rows, selected, ranked


def _replay_topic_years(
    paths: Sequence[Path],
    *,
    selected_fields: Sequence[int],
    taxonomy: Mapping[int, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected_values = np.asarray(selected_fields, dtype=np.uint8)
    dispositions = np.asarray([2, 3], dtype=np.uint8)
    counts: dict[int, int] = {}
    accounting = {
        "work_rows_scanned": 0,
        "selected_field_rows_with_usable_predecision_year_and_topic": 0,
        "taxonomy_missing_topic_field_pairs": 0,
        "taxonomy_field_mismatch_topic_field_pairs": 0,
    }
    for path in paths:
        table = pq.read_table(
            path,
            columns=[
                "publication_year",
                "temporal_disposition",
                "primary_topic_id",
                "primary_field_id",
            ],
        )
        accounting["work_rows_scanned"] += len(table)
        years = (
            table["publication_year"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        disposition = (
            table["temporal_disposition"]
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
        )
        topics = (
            table["primary_topic_id"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        fields = (
            table["primary_field_id"]
            .combine_chunks()
            .fill_null(0)
            .to_numpy(zero_copy_only=False)
            .astype(np.uint64, copy=False)
        )
        keep = (
            np.isin(fields, selected_values)
            & np.isin(disposition, dispositions)
            & (years > 0)
            & (years <= 2011)
            & (topics > 0)
        )
        pairs = np.unique(
            (fields[keep] << np.uint64(32)) | topics[keep]
        )
        for packed in pairs.tolist():
            field_id = int(packed >> 32)
            topic_id = int(packed & 0xFFFFFFFF)
            item = taxonomy.get(topic_id)
            if item is None:
                accounting["taxonomy_missing_topic_field_pairs"] += 1
            elif int(item["field_id"]) != field_id:
                accounting["taxonomy_field_mismatch_topic_field_pairs"] += 1
        keys = (
            (fields[keep] << np.uint64(48))
            | (topics[keep] << np.uint64(16))
            | years[keep]
        )
        unique, frequencies = np.unique(keys, return_counts=True)
        for key, frequency in zip(
            unique.tolist(),
            frequencies.tolist(),
            strict=True,
        ):
            counts[int(key)] = counts.get(int(key), 0) + int(frequency)
        accounting[
            "selected_field_rows_with_usable_predecision_year_and_topic"
        ] += int(keep.sum())
    require(accounting["taxonomy_missing_topic_field_pairs"] == 0, "taxonomy topic absent")
    require(
        accounting["taxonomy_field_mismatch_topic_field_pairs"] == 0,
        "taxonomy field mismatch",
    )
    rows = []
    for key in sorted(counts):
        field_id = key >> 48
        topic_id = (key >> 16) & 0xFFFFFFFF
        year = key & 0xFFFF
        item = taxonomy[topic_id]
        rows.append(
            {
                "field_id": field_id,
                "topic_id": topic_id,
                "topic_name": item["topic_name"],
                "subfield_id": item["subfield_id"],
                "subfield_name": item["subfield_name"],
                "publication_year": year,
                "usable_primary_topic_works": counts[key],
            }
        )
    accounting["topic_year_cells"] = len(rows)
    accounting["distinct_topics"] = len(
        {(row["field_id"], row["topic_id"]) for row in rows}
    )
    return rows, accounting


def _episode_key(row: Mapping[str, Any]) -> tuple[Fraction, int, int, int, int]:
    return (
        Fraction(
            int(row["standardized_gain_numerator"]),
            int(row["standardized_gain_denominator"]),
        ),
        int(row["minimum_baseline_recent_count"]),
        int(row["absolute_gain"]),
        -int(row["topic_id"]),
        -int(row["entry_year"]),
    )


def _replay_candidates(
    topic_rows: Sequence[Mapping[str, Any]],
    selected_fields: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for row in topic_rows:
        key = (int(row["field_id"]), int(row["topic_id"]))
        item = grouped.setdefault(
            key,
            {
                "topic_name": row["topic_name"],
                "subfield_id": int(row["subfield_id"]),
                "subfield_name": row["subfield_name"],
                "counts": {},
            },
        )
        item["counts"][int(row["publication_year"])] = int(
            row["usable_primary_topic_works"]
        )
    by_field: dict[int, list[dict[str, Any]]] = {}
    for (field_id, topic_id), item in grouped.items():
        years = item["counts"]
        for entry_year in range(min(years) + 6, 2013):
            baseline = sum(years.get(year, 0) for year in range(entry_year - 6, entry_year - 3))
            recent = sum(years.get(year, 0) for year in range(entry_year - 3, entry_year))
            if baseline <= 0 or recent <= baseline:
                continue
            gain = recent - baseline
            row = {
                "field_id": field_id,
                "topic_id": topic_id,
                "topic_name": item["topic_name"],
                "subfield_id": item["subfield_id"],
                "subfield_name": item["subfield_name"],
                "entry_year": entry_year,
                "baseline_start_year": entry_year - 6,
                "baseline_end_year": entry_year - 4,
                "recent_start_year": entry_year - 3,
                "recent_end_year": entry_year - 1,
                "reserved_response_end_year": entry_year + 2,
                "baseline_works": baseline,
                "recent_works": recent,
                "absolute_gain": gain,
                "minimum_baseline_recent_count": min(baseline, recent),
                "standardized_gain_numerator": gain * gain,
                "standardized_gain_denominator": baseline + 1,
                "standardized_gain": gain / math.sqrt(baseline + 1),
                "field_candidate_rank": 0,
                "pilot_episode_selected": False,
                "pilot_episode_order": None,
            }
            by_field.setdefault(field_id, []).append(row)
    selected = []
    for field_id in selected_fields:
        rows = by_field.get(field_id, [])
        require(rows, f"no replayed candidate for field {field_id}")
        rows.sort(key=_episode_key, reverse=True)
        for rank, row in enumerate(rows, start=1):
            row["field_candidate_rank"] = rank
        first = rows[0]
        second = next(
            row
            for row in rows[1:]
            if row["subfield_id"] != first["subfield_id"]
            and abs(row["entry_year"] - first["entry_year"]) >= 9
        )
        for order, row in enumerate((first, second), start=1):
            row["pilot_episode_selected"] = True
            row["pilot_episode_order"] = order
            selected.append(row)
    all_rows = [
        row
        for field_id in selected_fields
        for row in by_field[field_id]
    ]
    all_rows.sort(
        key=lambda row: (
            selected_fields.index(int(row["field_id"])),
            int(row["field_candidate_rank"]),
        )
    )
    selected.sort(
        key=lambda row: (
            selected_fields.index(int(row["field_id"])),
            int(row["pilot_episode_order"]),
        )
    )
    return all_rows, selected, {
        "candidate_rows": len(all_rows),
        "candidate_rows_by_field": {
            str(field_id): len(by_field[field_id])
            for field_id in selected_fields
        },
        "selected_episode_count": len(selected),
        "minimum_entry_year_separation": 9,
        "distinct_subfields_within_each_field": all(
            len(
                {
                    row["subfield_id"]
                    for row in selected
                    if row["field_id"] == field_id
                }
            )
            == 2
            for field_id in selected_fields
        ),
    }


def _assert_rows_equal(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    float_columns: set[str] | None = None,
    label: str,
) -> None:
    float_columns = set() if float_columns is None else float_columns
    require(len(observed) == len(expected), f"{label} row count mismatch")
    for index, (left, right) in enumerate(
        zip(observed, expected, strict=True)
    ):
        require(set(left) == set(right), f"{label} column set mismatch")
        for key in left:
            if key in float_columns:
                require(
                    math.isclose(
                        float(left[key]),
                        float(right[key]),
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ),
                    f"{label} float mismatch at row {index}:{key}",
                )
            else:
                require(
                    left[key] == right[key],
                    f"{label} mismatch at row {index}:{key}",
                )


def verify(
    root: Path,
    *,
    s0_core_root: Path,
    s0_orcid_root: Path,
    s0_graph_root: Path,
    cutoff_root: Path,
    taxonomy_topics: Path,
    workers: int,
) -> dict[str, Any]:
    require(1 <= workers <= 32, "workers must be in [1, 32]")
    root = root.resolve(strict=True)
    receipt, report, artifacts = _validate_output_identity(root)
    config = _read_yaml(CONFIG_PATH, "D0-A config")
    require(config.get("schema_version") == CONFIG_SCHEMA, "config schema mismatch")
    require(report.get("claim_boundary") == config.get("claim_boundary"), "claim boundary mismatch")
    forbidden = _mapping(
        config.get("forbidden_inputs_and_outputs"),
        "forbidden boundary",
    )
    require(
        forbidden and all(value is False for value in forbidden.values()),
        "config opens a forbidden D0-A input or output",
    )
    execution = _mapping(report.get("execution"), "execution")
    require(execution.get("config_sha256") == _sha256(CONFIG_PATH), "reported config SHA mismatch")
    require(execution.get("builder_sha256") == _sha256(BUILDER_PATH), "reported builder SHA mismatch")
    bindings = _validate_repository_bindings(config, report)
    fields = _candidate_fields(bindings["candidate_universe_document"])

    input_contract = _mapping(config.get("normalized_inputs"), "normalized inputs")
    core_path, core_binding = _validate_small_input(
        s0_core_root,
        _mapping(input_contract["s0_core"], "S0 core"),
    )
    orcid_path, orcid_binding = _validate_small_input(
        s0_orcid_root,
        _mapping(input_contract["s0_orcid_affiliation"], "S0 ORCID"),
    )
    graph_path, graph_binding = _validate_small_input(
        s0_graph_root,
        _mapping(input_contract["s0_coauthor_graph"], "S0 graph"),
    )
    work_paths, cutoff_binding, pre_stats = _validate_work_input(
        cutoff_root,
        _mapping(input_contract["cutoff_work_metadata"], "cutoff input"),
        workers=workers,
    )
    taxonomy, taxonomy_binding = _taxonomy(
        taxonomy_topics,
        _mapping(input_contract["topic_taxonomy_projection"], "taxonomy input"),
    )
    require(
        report.get("normalized_input_bindings")
        == {
            "s0_core": core_binding,
            "s0_orcid_affiliation": orcid_binding,
            "s0_coauthor_graph": graph_binding,
            "cutoff_work_metadata": cutoff_binding,
            "topic_taxonomy_projection": taxonomy_binding,
        },
        "reported normalized input bindings differ",
    )

    field_table = pq.read_table(root / "field_structural_ranking.parquet")
    require(_schema_version(field_table.schema) == FIELD_SCHEMA, "field table schema mismatch")
    require(FORBIDDEN_COLUMNS.isdisjoint(field_table.column_names), "forbidden field column")
    expected_field_rows, selected_fields, reserve_order = _replay_field_ranking(
        core_path,
        orcid_path,
        graph_path,
        fields=fields,
    )
    _assert_rows_equal(
        field_table.to_pylist(),
        expected_field_rows,
        float_columns={
            name
            for name in field_table.column_names
            if name.endswith("_rate")
            or name.endswith("_density")
            or name.endswith("_percentile")
        },
        label="field ranking",
    )
    field_report = _mapping(report.get("field_selection"), "field selection")
    require(field_report.get("reserve_order_field_ids") == reserve_order, "reserve order mismatch")
    expected_selected_fields = []
    by_field = {int(row["field_id"]): row for row in expected_field_rows}
    for field_id in selected_fields:
        row = by_field[field_id]
        expected_selected_fields.append(
            {
                "field_id": field_id,
                "field_name": row["field_name"],
                "field_slug": row["field_slug"],
                "domain_id": row["domain_id"],
                "domain_name": row["domain_name"],
                "global_rank": row["global_rank"],
                "pilot_selection_order": row["pilot_selection_order"],
                "minimum_metric_percentile": row[
                    "minimum_metric_percentile"
                ],
                "mean_metric_percentile": row["mean_metric_percentile"],
            }
        )
    _assert_rows_equal(
        field_report["selected_pilot_fields"],
        expected_selected_fields,
        float_columns={
            "minimum_metric_percentile",
            "mean_metric_percentile",
        },
        label="selected pilot fields",
    )

    topic_rows, topic_accounting = _replay_topic_years(
        work_paths,
        selected_fields=selected_fields,
        taxonomy=taxonomy,
    )
    require(
        topic_accounting["work_rows_scanned"]
        == input_contract["cutoff_work_metadata"]["rows"],
        "work row closure mismatch",
    )
    topic_table = pq.read_table(
        root / "selected_field_topic_year_attention.parquet"
    )
    require(_schema_version(topic_table.schema) == TOPIC_YEAR_SCHEMA, "topic-year schema mismatch")
    require(FORBIDDEN_COLUMNS.isdisjoint(topic_table.column_names), "forbidden topic-year column")
    _assert_rows_equal(
        topic_table.to_pylist(),
        topic_rows,
        label="topic-year attention",
    )
    reported_topic = _mapping(report.get("topic_year_attention"), "topic-year report")
    for key, expected in topic_accounting.items():
        require(reported_topic.get(key) == expected, f"topic accounting mismatch: {key}")

    candidate_rows, selected_episodes, candidate_accounting = _replay_candidates(
        topic_rows,
        selected_fields,
    )
    candidate_table = pq.read_table(root / "episode_candidates.parquet")
    require(_schema_version(candidate_table.schema) == CANDIDATE_SCHEMA, "candidate schema mismatch")
    require(FORBIDDEN_COLUMNS.isdisjoint(candidate_table.column_names), "forbidden candidate column")
    _assert_rows_equal(
        candidate_table.to_pylist(),
        candidate_rows,
        float_columns={"standardized_gain"},
        label="episode candidates",
    )
    episode_report = _mapping(report.get("episode_selection"), "episode report")
    for key, expected in candidate_accounting.items():
        require(episode_report.get(key) == expected, f"candidate accounting mismatch: {key}")
    expected_selected = [
        (row["field_id"], row["topic_id"], row["entry_year"])
        for row in selected_episodes
    ]
    selected_keys = list(episode_report["selected_pilot_episodes"][0])
    expected_selected_rows = [
        {key: row[key] for key in selected_keys}
        for row in selected_episodes
    ]
    _assert_rows_equal(
        episode_report["selected_pilot_episodes"],
        expected_selected_rows,
        float_columns={"standardized_gain"},
        label="selected pilot episodes",
    )
    require(len(expected_selected) == 4, "selected episode count mismatch")
    _validate_stats_unchanged(work_paths, pre_stats)

    artifact_by_name = {
        item["relative_path"]: item
        for item in artifacts
    }
    require(
        field_report.get("artifact")
        == artifact_by_name["field_structural_ranking.parquet"],
        "field artifact report binding mismatch",
    )
    require(
        reported_topic.get("artifact")
        == artifact_by_name["selected_field_topic_year_attention.parquet"],
        "topic-year artifact report binding mismatch",
    )
    require(
        episode_report.get("artifact")
        == artifact_by_name["episode_candidates.parquet"],
        "episode artifact report binding mismatch",
    )
    expected_result_fingerprint = hashlib.sha256(
        _canonical_bytes(
            {
                "status": OUTPUT_STATUS,
                "config_sha256": _sha256(CONFIG_PATH),
                "selected_field_ids": selected_fields,
                "selected_episodes": expected_selected,
                "artifacts": [
                    (
                        item["relative_path"],
                        item["bytes"],
                        item["sha256"],
                    )
                    for item in artifacts
                ],
            }
        )
    ).hexdigest()
    require(
        receipt.get("result_fingerprint_sha256")
        == expected_result_fingerprint,
        "receipt result fingerprint mismatch",
    )

    authorization = _mapping(report.get("authorization"), "authorization")
    require(
        authorization.get("d0b_outcome_blind_author_panel") == "AUTHORIZED",
        "bounded D0-B authorization missing",
    )
    for key in (
        "d0_discovery_outcome_opening",
        "g0_measurement_freeze",
        "final_panel_materialization",
        "model_fitting",
        "g4_open",
    ):
        expected = (
            "NOT_YET_AUTHORIZED"
            if key == "d0_discovery_outcome_opening"
            else "NOT_AUTHORIZED"
        )
        require(authorization.get(key) == expected, f"unsafe authorization: {key}")
    require(
        authorization.get("scientific_effect") == "NOT_ESTABLISHED",
        "scientific effect overstated",
    )
    return {
        "schema_version": "pivot_feedback_d0_structural_pilot_freeze_verification.v1",
        "status": VERIFY_STATUS,
        "output_root_basename": root.name,
        "output_receipt_sha256": _sha256(
            root / "d0_structural_pilot_freeze.outputs.json"
        ),
        "artifacts_and_receipt_rehashed": len(artifacts) + 1,
        "source_work_artifacts_rehashed": len(work_paths),
        "source_work_rows_replayed": topic_accounting["work_rows_scanned"],
        "field_ranking_rows_replayed": len(expected_field_rows),
        "topic_year_rows_replayed": len(topic_rows),
        "episode_candidate_rows_replayed": len(candidate_rows),
        "selected_field_ids": selected_fields,
        "selected_episodes": [
            {
                "field_id": field_id,
                "topic_id": topic_id,
                "entry_year": entry_year,
            }
            for field_id, topic_id, entry_year in expected_selected
        ],
        "outcomes_read": False,
        "d0a_structural_freeze_complete": True,
        "d0_complete": False,
        "d0b_outcome_blind_author_panel_authorized": True,
        "d0_discovery_outcome_opening_authorized": False,
        "g0_authorized": False,
        "model_fitting_authorized": False,
        "g4_open": False,
        "gpu_used": False,
        "scientific_effect": "NOT_ESTABLISHED",
        "builder_sha256": _sha256(BUILDER_PATH),
        "verifier_sha256": _sha256(Path(__file__).resolve()),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--s0-core-root", type=Path, required=True)
    parser.add_argument("--s0-orcid-root", type=Path, required=True)
    parser.add_argument("--s0-graph-root", type=Path, required=True)
    parser.add_argument("--cutoff-root", type=Path, required=True)
    parser.add_argument("--taxonomy-topics", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--report",
        type=Path,
        help="write one new private verification report outside the output root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = verify(
            args.output_root,
            s0_core_root=args.s0_core_root,
            s0_orcid_root=args.s0_orcid_root,
            s0_graph_root=args.s0_graph_root,
            cutoff_root=args.cutoff_root,
            taxonomy_topics=args.taxonomy_topics,
            workers=args.workers,
        )
        if args.report is not None:
            _write_report(args.report, result, root=args.output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    except (
        D0StructuralVerificationError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        pa.ArrowException,
    ) as error:
        print(
            json.dumps(
                {
                    "status": "BLOCKED_D0A_STRUCTURAL_FREEZE_VERIFICATION",
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
