#!/usr/bin/env python3
"""Prepare the outcome-blind D0 content and indexing review.

The command binds the measurement-ABI reselection, profiles the four selected
episodes before entry, and extracts a deterministic current-snapshot OpenAlex
title/metadata sample. It never reads entry, persistence, citation, prediction,
or G4 outcomes. Automated success still requires human double review before
leave-out or network construction.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT / "experiments/configs/pivot_feedback_d0_content_indexing_audit_v1.yaml"
)
OPENALEX_WORK_PREFIX = "https://openalex.org/W"
OPENALEX_TOPIC_PREFIX = "https://openalex.org/T"


class AuditError(RuntimeError):
    """Fail-closed D0 content/indexing audit error."""


@dataclass(frozen=True)
class SampleSpec:
    episode_index: int
    field_id: int
    topic_id: int
    entry_year: int
    work_numeric_id: int
    publication_year: int
    work_type_code: int
    window_code: int

    @property
    def openalex_id(self) -> str:
        return f"{OPENALEX_WORK_PREFIX}{self.work_numeric_id}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> Mapping[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(raw, "config")


def _load_json(path: Path) -> Mapping[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _mapping(raw, path.name)


def _episode_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["episode_index"]),
        int(row["field_id"]),
        int(row["topic_id"]),
        int(row["entry_year"]),
    )


def _validate_parent(
    config: Mapping[str, Any],
    reselection_root: Path,
) -> tuple[Mapping[str, Any], Path]:
    parent = _mapping(config["parent_reselection"], "parent reselection")
    receipt_path = reselection_root / str(parent["receipt_basename"])
    report_path = reselection_root / str(parent["report_basename"])
    selected_path = reselection_root / str(parent["selected_work_basename"])
    for path in (receipt_path, report_path, selected_path):
        require(path.is_file(), f"missing parent artifact: {path.name}")
    require(_sha256(receipt_path) == parent["receipt_sha256"], "receipt SHA mismatch")
    require(_sha256(report_path) == parent["report_sha256"], "report SHA mismatch")
    require(
        _sha256(selected_path) == parent["selected_work_sha256"],
        "selected-work SHA mismatch",
    )
    receipt = _load_json(receipt_path)
    report = _load_json(report_path)
    require(receipt["status"] == parent["required_status"], "receipt status mismatch")
    require(report["status"] == parent["required_status"], "report status mismatch")
    require(receipt["outcomes_read"] is False, "parent read outcomes")
    require(report["scope"]["outcomes_read"] is False, "parent report read outcomes")
    require(report["abi"]["sha256"] == parent["abi_sha256"], "ABI SHA mismatch")
    require(receipt["d0_complete"] is False, "parent incorrectly completed D0")
    require(receipt["g0_authorized"] is False, "parent incorrectly authorized G0")
    require(receipt["g4_open"] is False, "parent opened G4")
    observed = {
        (
            index,
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        )
        for index, row in enumerate(
            report["episode_reselection"]["selected_episodes"], start=1
        )
    }
    expected = {_episode_key(row) for row in parent["expected_episodes"]}
    require(observed == expected, "selected episode contract mismatch")
    return report, selected_path


def _read_selected(
    path: Path,
    *,
    expected_rows: int,
    expected_episodes: Sequence[Mapping[str, Any]],
) -> pa.Table:
    table = pq.read_table(path)
    required = {
        "episode_index",
        "field_id",
        "topic_id",
        "entry_year",
        "work_numeric_id",
        "publication_year",
        "work_type_code",
        "declared_authors_count",
        "is_retracted_null",
        "window_code",
    }
    require(set(table.column_names) == required, "selected-work schema mismatch")
    require(len(table) == expected_rows, "selected-work row count mismatch")
    observed = {
        (
            int(row["episode_index"]),
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        )
        for row in table.select(
            ["episode_index", "field_id", "topic_id", "entry_year"]
        ).to_pylist()
    }
    expected = {_episode_key(row) for row in expected_episodes}
    require(observed == expected, "selected-work episodes mismatch")
    work = table["work_numeric_id"].combine_chunks().to_numpy(zero_copy_only=False)
    episode = table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False)
    order = np.lexsort((work, episode))
    require(
        not bool(
            np.any(
                (episode[order][1:] == episode[order][:-1])
                & (work[order][1:] == work[order][:-1])
            )
        ),
        "duplicate selected episode-work row",
    )
    return table


def _sample_specs(
    table: pa.Table,
    *,
    seed: str,
    per_group: int,
) -> list[SampleSpec]:
    grouped: dict[tuple[int, int], list[tuple[bytes, SampleSpec]]] = {}
    for row in table.to_pylist():
        spec = SampleSpec(
            episode_index=int(row["episode_index"]),
            field_id=int(row["field_id"]),
            topic_id=int(row["topic_id"]),
            entry_year=int(row["entry_year"]),
            work_numeric_id=int(row["work_numeric_id"]),
            publication_year=int(row["publication_year"]),
            work_type_code=int(row["work_type_code"]),
            window_code=int(row["window_code"]),
        )
        payload = (
            f"{seed}\0{spec.episode_index}\0{spec.window_code}\0{spec.work_numeric_id}"
        ).encode()
        grouped.setdefault((spec.episode_index, spec.window_code), []).append(
            (hashlib.sha256(payload).digest(), spec)
        )
    require(len(grouped) == 12, "expected three windows for four episodes")
    selected: list[SampleSpec] = []
    for key, rows in sorted(grouped.items()):
        require(len(rows) >= per_group, f"insufficient rows for sample group {key}")
        rows.sort(key=lambda item: (item[0], item[1].work_numeric_id))
        selected.extend(item[1] for item in rows[:per_group])
    selected.sort(
        key=lambda item: (
            item.episode_index,
            item.window_code,
            item.work_numeric_id,
        )
    )
    require(
        len({item.openalex_id for item in selected}) == len(selected),
        "sampled work appears in multiple episode-window groups",
    )
    return selected


def _annual_profile(
    table: pa.Table,
    *,
    ratio_threshold: float,
    minimum_previous_count: int,
) -> pa.Table:
    rows = table.to_pylist()
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    episode_meta: dict[int, tuple[int, int, int]] = {}
    for row in rows:
        episode = int(row["episode_index"])
        year = int(row["publication_year"])
        grouped.setdefault((episode, year), []).append(row)
        episode_meta[episode] = (
            int(row["field_id"]),
            int(row["topic_id"]),
            int(row["entry_year"]),
        )
    output: list[dict[str, Any]] = []
    for episode in sorted(episode_meta):
        field, topic, entry = episode_meta[episode]
        years = sorted(
            year for item_episode, year in grouped if item_episode == episode
        )
        previous = None
        for year in years:
            items = grouped[(episode, year)]
            type_counts: dict[int, int] = {}
            authors = []
            for item in items:
                code = int(item["work_type_code"])
                type_counts[code] = type_counts.get(code, 0) + 1
                authors.append(int(item["declared_authors_count"]))
            dominant_code, dominant_count = min(
                type_counts.items(), key=lambda item: (-item[1], item[0])
            )
            count = len(items)
            ratio = count / previous if previous else None
            abrupt = bool(
                previous is not None
                and previous >= minimum_previous_count
                and ratio is not None
                and ratio >= ratio_threshold
            )
            output.append(
                {
                    "episode_index": episode,
                    "field_id": field,
                    "topic_id": topic,
                    "entry_year": entry,
                    "publication_year": year,
                    "window_code": (
                        0 if year <= entry - 7 else (1 if year <= entry - 4 else 2)
                    ),
                    "author_linked_work_count": count,
                    "previous_year_work_count": previous,
                    "year_over_year_ratio": ratio,
                    "abrupt_attention_flag": abrupt,
                    "dominant_work_type_code": dominant_code,
                    "dominant_work_type_share": dominant_count / count,
                    "median_declared_authors_count": float(np.median(authors)),
                    "is_retracted_null_count": sum(
                        bool(item["is_retracted_null"]) for item in items
                    ),
                }
            )
            previous = count
    schema = pa.schema(
        [
            pa.field("episode_index", pa.uint8(), nullable=False),
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("publication_year", pa.uint16(), nullable=False),
            pa.field("window_code", pa.uint8(), nullable=False),
            pa.field("author_linked_work_count", pa.uint64(), nullable=False),
            pa.field("previous_year_work_count", pa.uint64()),
            pa.field("year_over_year_ratio", pa.float64()),
            pa.field("abrupt_attention_flag", pa.bool_(), nullable=False),
            pa.field("dominant_work_type_code", pa.uint8(), nullable=False),
            pa.field("dominant_work_type_share", pa.float64(), nullable=False),
            pa.field("median_declared_authors_count", pa.float64(), nullable=False),
            pa.field("is_retracted_null_count", pa.uint64(), nullable=False),
        ],
        metadata={
            b"schema_version": b"pivot_feedback_d0_episode_annual_indexing_profile.v1"
        },
    )
    return pa.Table.from_pylist(output, schema=schema)


def _validate_source(
    config: Mapping[str, Any],
    openalex_root: Path,
) -> tuple[list[Path], list[str]]:
    source = _mapping(config["current_snapshot_source"], "current snapshot source")
    require(openalex_root.name == source["root_id"], "OpenAlex root ID mismatch")
    table_dir = openalex_root / str(source["main_table_directory"])
    require(table_dir.is_dir(), "OpenAlex main table directory missing")
    files = sorted(table_dir.glob("part-*.parquet"), key=lambda path: path.name)
    require(
        len(files) == source["expected_main_file_count"], "main file count mismatch"
    )
    footer_rows = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    require(footer_rows == source["expected_main_footer_rows"], "main row mismatch")
    required_columns = [str(value) for value in source["required_columns"]]
    available = set(pq.ParquetFile(files[0]).schema_arrow.names)
    require(set(required_columns) <= available, "required source columns missing")
    return files, required_columns


def _extract_sample(
    files: Sequence[Path],
    columns: Sequence[str],
    specs: Sequence[SampleSpec],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_id = {item.openalex_id: item for item in specs}
    value_set = pa.array(sorted(by_id), type=pa.string())
    found: dict[str, dict[str, Any]] = {}
    row_groups_scanned = 0
    content_row_groups_read = 0
    for file_index, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        for row_group in range(parquet.metadata.num_row_groups):
            row_groups_scanned += 1
            identifiers = parquet.read_row_group(row_group, columns=["id"])["id"]
            mask = pc.fill_null(pc.is_in(identifiers, value_set=value_set), False)
            if not pc.any(mask).as_py():
                continue
            content_row_groups_read += 1
            content = parquet.read_row_group(row_group, columns=list(columns))
            selected = content.filter(
                pc.fill_null(pc.is_in(content["id"], value_set=value_set), False)
            )
            for row in selected.to_pylist():
                openalex_id = str(row["id"])
                require(
                    openalex_id not in found, f"duplicate source work {openalex_id}"
                )
                spec = by_id[openalex_id]
                row["episode_index"] = spec.episode_index
                row["field_id"] = spec.field_id
                row["topic_id"] = spec.topic_id
                row["entry_year"] = spec.entry_year
                row["work_numeric_id"] = spec.work_numeric_id
                row["expected_publication_year"] = spec.publication_year
                row["expected_work_type_code"] = spec.work_type_code
                row["window_code"] = spec.window_code
                row["source_file_index"] = file_index
                row["source_row_group"] = row_group
                found[openalex_id] = row
    rows = [found[item.openalex_id] for item in specs if item.openalex_id in found]
    return rows, {
        "main_files_scanned": len(files),
        "row_groups_scanned": row_groups_scanned,
        "content_row_groups_read": content_row_groups_read,
        "sample_rows_requested": len(specs),
        "sample_rows_found": len(rows),
    }


def _work_type_mapping(parent_report: Mapping[str, Any]) -> dict[int, str]:
    codes = [int(value) for value in parent_report["abi"]["included_work_type_codes"]]
    names = [str(value) for value in parent_report["abi"]["included_work_type_names"]]
    require(len(codes) == len(names), "parent ABI type mapping mismatch")
    return dict(zip(codes, names, strict=True))


def _sample_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    type_names: Mapping[int, str],
) -> pa.Table:
    output = []
    for row in rows:
        expected_code = int(row["expected_work_type_code"])
        output.append(
            {
                "episode_index": int(row["episode_index"]),
                "field_id": int(row["field_id"]),
                "topic_id": int(row["topic_id"]),
                "entry_year": int(row["entry_year"]),
                "window_code": int(row["window_code"]),
                "work_numeric_id": int(row["work_numeric_id"]),
                "openalex_id": str(row["id"]),
                "expected_publication_year": int(row["expected_publication_year"]),
                "observed_publication_year": (
                    int(row["publication_year"])
                    if row["publication_year"] is not None
                    else None
                ),
                "expected_work_type_code": expected_code,
                "expected_work_type_name": type_names[expected_code],
                "observed_work_type": row["type"],
                "title": row["title"] or row["display_name"],
                "language": row["language"],
                "authors_count": row["authors_count"],
                "is_paratext": row["is_paratext"],
                "is_retracted": row["is_retracted"],
                "observed_primary_topic_id": row["primary_topic__id"],
                "observed_primary_topic_name": row["primary_topic__display_name"],
                "primary_topic_score": row["primary_topic__score"],
                "observed_primary_field_id": row["primary_topic__field__id"],
                "observed_primary_subfield_id": row["primary_topic__subfield__id"],
                "source_id": row["primary_location__source__id"],
                "source_name": row["primary_location__source__display_name"],
                "created_date": row["created_date"],
                "updated_date": row["updated_date"],
                "source_file_index": int(row["source_file_index"]),
                "source_row_group": int(row["source_row_group"]),
                "reviewer_1_label": None,
                "reviewer_2_label": None,
                "adjudicated_label": None,
                "review_notes": None,
            }
        )
    schema = pa.schema(
        [
            pa.field("episode_index", pa.uint8(), nullable=False),
            pa.field("field_id", pa.uint8(), nullable=False),
            pa.field("topic_id", pa.uint32(), nullable=False),
            pa.field("entry_year", pa.uint16(), nullable=False),
            pa.field("window_code", pa.uint8(), nullable=False),
            pa.field("work_numeric_id", pa.uint64(), nullable=False),
            pa.field("openalex_id", pa.string(), nullable=False),
            pa.field("expected_publication_year", pa.uint16(), nullable=False),
            pa.field("observed_publication_year", pa.uint16()),
            pa.field("expected_work_type_code", pa.uint8(), nullable=False),
            pa.field("expected_work_type_name", pa.string(), nullable=False),
            pa.field("observed_work_type", pa.string()),
            pa.field("title", pa.string()),
            pa.field("language", pa.string()),
            pa.field("authors_count", pa.int64()),
            pa.field("is_paratext", pa.bool_()),
            pa.field("is_retracted", pa.bool_()),
            pa.field("observed_primary_topic_id", pa.string()),
            pa.field("observed_primary_topic_name", pa.string()),
            pa.field("primary_topic_score", pa.float64()),
            pa.field("observed_primary_field_id", pa.string()),
            pa.field("observed_primary_subfield_id", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("source_name", pa.string()),
            pa.field("created_date", pa.string()),
            pa.field("updated_date", pa.string()),
            pa.field("source_file_index", pa.uint16(), nullable=False),
            pa.field("source_row_group", pa.uint16(), nullable=False),
            pa.field("reviewer_1_label", pa.string()),
            pa.field("reviewer_2_label", pa.string()),
            pa.field("adjudicated_label", pa.string()),
            pa.field("review_notes", pa.string()),
        ],
        metadata={b"schema_version": b"pivot_feedback_d0_content_review_sample.v1"},
    )
    return pa.Table.from_pylist(output, schema=schema)


def _group_diagnostics(
    sample: pa.Table,
    *,
    type_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in sample.to_pylist():
        grouped.setdefault(
            (int(row["episode_index"]), int(row["window_code"])), []
        ).append(row)
    diagnostics = []
    for (episode, window), rows in sorted(grouped.items()):
        count = len(rows)
        scores = [
            float(row["primary_topic_score"])
            for row in rows
            if row["primary_topic_score"] is not None
        ]
        sources: dict[str, int] = {}
        for row in rows:
            key = str(row["source_id"] or "<MISSING>")
            sources[key] = sources.get(key, 0) + 1
        dominant_source, dominant_count = min(
            sources.items(), key=lambda item: (-item[1], item[0])
        )
        diagnostics.append(
            {
                "episode_index": episode,
                "window_code": window,
                "sample_rows": count,
                "missing_title_rows": sum(not row["title"] for row in rows),
                "missing_title_fraction": sum(not row["title"] for row in rows) / count,
                "missing_topic_score_rows": count - len(scores),
                "missing_topic_score_fraction": (count - len(scores)) / count,
                "median_topic_score": float(np.median(scores)) if scores else None,
                "dominant_source_id": dominant_source,
                "dominant_source_share": dominant_count / count,
                "publication_year_mismatch_rows": sum(
                    row["observed_publication_year"] != row["expected_publication_year"]
                    for row in rows
                ),
                "primary_topic_mismatch_rows": sum(
                    row["observed_primary_topic_id"]
                    != f"{OPENALEX_TOPIC_PREFIX}{row['topic_id']}"
                    for row in rows
                ),
                "work_type_mismatch_rows": sum(
                    row["observed_work_type"]
                    != type_names[int(row["expected_work_type_code"])]
                    for row in rows
                ),
            }
        )
    return diagnostics


def _review_flags(
    parent_report: Mapping[str, Any],
    diagnostics: Sequence[Mapping[str, Any]],
    annual: pa.Table,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    selected = parent_report["episode_reselection"]["selected_episodes"]
    link_threshold = float(thresholds["absolute_baseline_signal_link_rate_shift"])
    by_group = {
        (int(row["episode_index"]), int(row["window_code"])): row for row in diagnostics
    }
    score_threshold = float(
        thresholds["absolute_baseline_signal_median_topic_score_shift"]
    )
    source_threshold = float(
        thresholds["absolute_baseline_signal_dominant_source_share_shift"]
    )
    for episode, selected_row in enumerate(selected, start=1):
        link_shift = float(selected_row["recent_link_rate"]) - float(
            selected_row["baseline_link_rate"]
        )
        if abs(link_shift) >= link_threshold:
            flags.append(
                {
                    "episode_index": episode,
                    "code": "BASELINE_SIGNAL_LINK_RATE_SHIFT",
                    "observed": link_shift,
                    "threshold": link_threshold,
                }
            )
        baseline = by_group[(episode, 1)]
        signal = by_group[(episode, 2)]
        if (
            baseline["median_topic_score"] is not None
            and signal["median_topic_score"] is not None
        ):
            score_shift = float(signal["median_topic_score"]) - float(
                baseline["median_topic_score"]
            )
            if abs(score_shift) >= score_threshold:
                flags.append(
                    {
                        "episode_index": episode,
                        "code": "SAMPLED_MEDIAN_TOPIC_SCORE_SHIFT",
                        "observed": score_shift,
                        "threshold": score_threshold,
                    }
                )
        source_shift = float(signal["dominant_source_share"]) - float(
            baseline["dominant_source_share"]
        )
        if abs(source_shift) >= source_threshold:
            flags.append(
                {
                    "episode_index": episode,
                    "code": "SAMPLED_DOMINANT_SOURCE_SHARE_SHIFT",
                    "observed": source_shift,
                    "threshold": source_threshold,
                }
            )
    for row in annual.to_pylist():
        if row["abrupt_attention_flag"]:
            flags.append(
                {
                    "episode_index": int(row["episode_index"]),
                    "code": "SINGLE_YEAR_ATTENTION_RATIO",
                    "publication_year": int(row["publication_year"]),
                    "observed": float(row["year_over_year_ratio"]),
                    "previous_count": int(row["previous_year_work_count"]),
                }
            )
    return flags


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
    )
    os.chmod(path, 0o600)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "rows": len(table),
        "sha256": _sha256(path),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o600)


def _write_review_csv(table: pa.Table, path: Path) -> dict[str, Any]:
    rows = table.to_pylist()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=table.column_names)
        writer.writeheader()
        writer.writerows(rows)
    os.chmod(path, 0o600)
    return {
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "sha256": _sha256(path),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# D0 content and indexing audit",
        "",
        f"Status: `{report['status']}`",
        "",
        "The outcome-blind extraction passed its automated integrity checks.",
        "Human double review of sampled titles is still required, so D0, G0,",
        "leave-out/network construction, model fitting, and G4 remain closed.",
        "",
        "## Sample diagnostics",
        "",
        "| Episode | Window | Rows | Missing title | Median topic score |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["sample_diagnostics"]:
        score = row["median_topic_score"]
        lines.append(
            f"| {row['episode_index']} | {row['window_code']} | "
            f"{row['sample_rows']} | {row['missing_title_rows']} | "
            f"{score:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Indexing-review flags: `{len(report['indexing_review_flags'])}`",
            "",
            "These flags are review prompts, not evidence that a scientific",
            "transition or an indexing artifact occurred.",
            "",
        ]
    )
    return "\n".join(lines)


def _artifact_tree(artifacts: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in sorted(artifacts, key=lambda row: str(row["relative_path"])):
        digest.update(
            f"{item['relative_path']}\0{item['sha256']}\0{item['bytes']}\n".encode()
        )
    return digest.hexdigest()


def _cleanup_temporary_directory(
    temporary: Path | None,
    *,
    output: Path,
) -> None:
    if temporary is None:
        return
    require(temporary.is_absolute(), "temporary cleanup path must be absolute")
    require(
        temporary.parent == output.parent.resolve(),
        "temporary cleanup path must be an output sibling",
    )
    require(
        temporary.name.startswith(f".{output.name}.tmp-"),
        "temporary cleanup prefix mismatch",
    )
    if temporary.exists():
        shutil.rmtree(temporary)


def run_audit(
    *,
    config_path: Path,
    reselection_root: Path,
    openalex_root: Path,
    output: Path,
) -> Mapping[str, Any]:
    config_path = config_path.expanduser().resolve()
    reselection_root = reselection_root.expanduser().resolve()
    openalex_root = openalex_root.expanduser().resolve()
    output = output.expanduser().resolve()
    require(not output.exists(), "output directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = _load_yaml(config_path)
    require(
        config["schema_version"]
        == "pivot_feedback_d0_content_indexing_audit_config.v1",
        "config schema mismatch",
    )
    parent_report, selected_path = _validate_parent(config, reselection_root)
    parent = _mapping(config["parent_reselection"], "parent reselection")
    selected = _read_selected(
        selected_path,
        expected_rows=int(parent["selected_work_rows"]),
        expected_episodes=parent["expected_episodes"],
    )
    sampling = _mapping(config["sampling"], "sampling")
    specs = _sample_specs(
        selected,
        seed=str(sampling["seed"]),
        per_group=int(sampling["per_episode_window"]),
    )
    checks = _mapping(config["automated_checks"], "automated checks")
    jump = _mapping(
        _mapping(checks["indexing_review_flags"], "review flags")[
            "single_year_attention_ratio_with_minimum_previous_count"
        ],
        "annual jump",
    )
    annual = _annual_profile(
        selected,
        ratio_threshold=float(jump["ratio"]),
        minimum_previous_count=int(jump["minimum_previous_count"]),
    )
    files, columns = _validate_source(config, openalex_root)
    started = _utc_now()
    extracted, scan = _extract_sample(files, columns, specs)
    require(
        scan["sample_rows_found"] == scan["sample_rows_requested"],
        "not every sampled work was found exactly once",
    )
    type_names = _work_type_mapping(parent_report)
    sample = _sample_table(extracted, type_names=type_names)
    diagnostics = _group_diagnostics(sample, type_names=type_names)
    maximum_title = float(checks["maximum_missing_title_fraction_per_episode_window"])
    maximum_score = float(
        checks["maximum_missing_topic_score_fraction_per_episode_window"]
    )
    blocking_issues = []
    if scan["sample_rows_found"] != scan["sample_rows_requested"]:
        blocking_issues.append("SAMPLED_WORK_NOT_FOUND_EXACTLY_ONCE")
    if any(row["publication_year_mismatch_rows"] for row in diagnostics):
        blocking_issues.append("PUBLICATION_YEAR_MISMATCH")
    if any(row["primary_topic_mismatch_rows"] for row in diagnostics):
        blocking_issues.append("PRIMARY_TOPIC_MISMATCH")
    if any(row["work_type_mismatch_rows"] for row in diagnostics):
        blocking_issues.append("WORK_TYPE_MISMATCH")
    if any(row["missing_title_fraction"] > maximum_title for row in diagnostics):
        blocking_issues.append("TITLE_MISSINGNESS_THRESHOLD_EXCEEDED")
    if any(row["missing_topic_score_fraction"] > maximum_score for row in diagnostics):
        blocking_issues.append("TOPIC_SCORE_MISSINGNESS_THRESHOLD_EXCEEDED")
    flags = _review_flags(
        parent_report,
        diagnostics,
        annual,
        _mapping(checks["indexing_review_flags"], "indexing review thresholds"),
    )
    output_contract = _mapping(config["output_contract"], "output contract")
    status = (
        output_contract["blocked_status"]
        if blocking_issues
        else output_contract["automated_pass_status"]
    )
    report = {
        "schema_version": output_contract["report_schema_version"],
        "status": status,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "parent": {
            "reselection_root_basename": reselection_root.name,
            "receipt_sha256": parent["receipt_sha256"],
            "report_sha256": parent["report_sha256"],
            "selected_work_sha256": parent["selected_work_sha256"],
            "selected_work_rows": len(selected),
            "abi_sha256": parent["abi_sha256"],
        },
        "source": {
            "root_id": openalex_root.name,
            "main_file_count": len(files),
            "main_footer_rows": int(
                config["current_snapshot_source"]["expected_main_footer_rows"]
            ),
            "path_redacted": True,
            "current_snapshot_primary_topic_semantics": True,
        },
        "sampling": {
            "algorithm": sampling["algorithm"],
            "seed": sampling["seed"],
            "per_episode_window": sampling["per_episode_window"],
            **scan,
        },
        "sample_diagnostics": diagnostics,
        "indexing_review_flags": flags,
        "blocking_issues": sorted(set(blocking_issues)),
        "human_review": {
            "required": True,
            "complete": False,
            "minimum_two_reviewers": True,
            "allowed_labels": config["human_review"]["labels"],
        },
        "authorization": {
            "d0_complete": False,
            "g0_authorized": False,
            "leaveout_network_bridge_authorized": False,
            "discovery_outcome_opening_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "scientific_effect": "NOT_ESTABLISHED",
        },
        "forbidden_inputs_observed": {
            "entry_or_persistence": False,
            "realized_citation": False,
            "prediction": False,
            "g4": False,
        },
        "gpu_used": False,
        "claim_boundary": config["claim_boundary"],
    }
    temporary: Path | None = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    ).resolve()
    os.chmod(temporary, 0o700)
    try:
        artifacts = [
            _write_parquet(sample, temporary / "content_review_sample.parquet"),
            _write_review_csv(sample, temporary / "content_review_template.csv"),
            _write_parquet(
                annual, temporary / "episode_annual_indexing_profile.parquet"
            ),
        ]
        report_path = temporary / "content_indexing_audit.json"
        _write_json(report_path, report)
        artifacts.append(
            {
                "relative_path": report_path.name,
                "bytes": report_path.stat().st_size,
                "sha256": _sha256(report_path),
            }
        )
        markdown_path = temporary / "content_indexing_audit.md"
        markdown_path.write_text(_markdown(report), encoding="utf-8")
        os.chmod(markdown_path, 0o600)
        artifacts.append(
            {
                "relative_path": markdown_path.name,
                "bytes": markdown_path.stat().st_size,
                "sha256": _sha256(markdown_path),
            }
        )
        receipt = {
            "schema_version": output_contract["receipt_schema_version"],
            "status": status,
            "artifact_count": len(artifacts),
            "artifact_tree_sha256": _artifact_tree(artifacts),
            "artifacts": sorted(artifacts, key=lambda row: row["relative_path"]),
            "sample_rows": len(sample),
            "indexing_review_flag_count": len(flags),
            "blocking_issue_count": len(set(blocking_issues)),
            "human_review_complete": False,
            "d0_complete": False,
            "g0_authorized": False,
            "leaveout_network_bridge_authorized": False,
            "model_fitting_authorized": False,
            "g4_open": False,
            "outcomes_read": False,
            "gpu_used": False,
        }
        _write_json(temporary / "content_indexing_audit.outputs.json", receipt)
        os.replace(temporary, output)
        temporary = None
    finally:
        _cleanup_temporary_directory(temporary, output=output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reselection-root", type=Path, required=True)
    parser.add_argument("--openalex-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.openalex_root is None:
        config = _load_yaml(args.config)
        env_name = str(config["current_snapshot_source"]["root_env"])
        value = os.environ.get(env_name)
        if not value:
            parser.error(f"set {env_name} or pass --openalex-root")
        args.openalex_root = Path(value)
    return args


def main() -> int:
    args = parse_args()
    try:
        report = run_audit(
            config_path=args.config,
            reselection_root=args.reselection_root,
            openalex_root=args.openalex_root,
            output=args.output,
        )
    except (AuditError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "sample_rows": report["sampling"]["sample_rows_found"],
                "indexing_review_flags": len(report["indexing_review_flags"]),
                "blocking_issues": len(report["blocking_issues"]),
                "human_review_complete": False,
                "d0_complete": False,
                "outcomes_read": False,
                "gpu_used": False,
            },
            sort_keys=True,
        )
    )
    return 0 if not report["blocking_issues"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
