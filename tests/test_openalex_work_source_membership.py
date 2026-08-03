from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_work_source_membership.py"
CONFIG_PATH = (
    REPO_ROOT / "experiments/configs/openalex_g1_work_source_membership_v0.yaml"
)


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_work_source_membership_module", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _masks(parsed: Any) -> dict[str, list[bool]]:
    return {
        "valid": parsed.grammar_valid.tolist(),
        "null": parsed.null.tolist(),
        "w_minus_1": parsed.w_minus_1.tolist(),
        "malformed": parsed.malformed_other.tolist(),
    }


def test_partition_work_ids_is_exact_and_mutually_exclusive() -> None:
    maximum = "https://openalex.org/W18446744073709551615"
    overflow = "https://openalex.org/W18446744073709551616"
    values = pa.array(
        [
            "https://openalex.org/W1",
            audit.W_ZERO,
            audit.W_MINUS_1,
            None,
            "https://openalex.org/W01",
            "https://openalex.org/A1",
            "https://openalex.org/W",
            "https://openalex.org/W+1",
            maximum,
            overflow,
        ],
        type=pa.string(),
    )

    parsed = audit.partition_ids(values, prefix=audit.WORK_PREFIX)

    assert _masks(parsed) == {
        "valid": [True, True, False, False, False, False, False, False, True, False],
        "null": [False, False, False, True, False, False, False, False, False, False],
        "w_minus_1": [
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        "malformed": [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            False,
            True,
        ],
    }
    assert parsed.numeric[0] == 1
    assert parsed.numeric[1] == 0
    assert parsed.numeric[8] == np.iinfo(np.uint64).max
    category_sum = (
        parsed.grammar_valid.astype(np.uint8)
        + parsed.null.astype(np.uint8)
        + parsed.w_minus_1.astype(np.uint8)
        + parsed.malformed_other.astype(np.uint8)
    )
    np.testing.assert_array_equal(category_sum, np.ones(len(values), dtype=np.uint8))


def test_partition_author_ids_does_not_special_case_negative_one() -> None:
    values = pa.array(
        [
            "https://openalex.org/A0",
            "https://openalex.org/A7",
            "https://openalex.org/A-1",
            "https://openalex.org/A07",
            audit.W_MINUS_1,
            None,
        ],
        type=pa.string(),
    )

    parsed = audit.partition_ids(values, prefix=audit.AUTHOR_PREFIX)

    assert parsed.grammar_valid.tolist() == [True, True, False, False, False, False]
    assert parsed.null.tolist() == [False, False, False, False, False, True]
    assert not np.any(parsed.w_minus_1)
    assert parsed.malformed_other.tolist() == [False, False, True, True, True, False]
    np.testing.assert_array_equal(parsed.numeric[:2], np.array([0, 7], dtype=np.uint64))


def test_dense_membership_is_exact_at_zero_bounds_and_gaps() -> None:
    master = np.array([0, 2, 7], dtype=np.uint64)

    membership = audit.build_membership_index(master, max_id=7)
    observed = audit.member_mask(
        np.array([0, 1, 2, 6, 7, 8, np.iinfo(np.uint64).max], dtype=np.uint64),
        membership,
    )

    assert membership.dtype == np.uint8
    assert membership.tolist() == [1, 0, 1, 0, 0, 0, 0, 1]
    assert observed.tolist() == [True, False, True, False, True, False, False]


def test_dense_membership_rejects_duplicate_master_ids() -> None:
    with pytest.raises(audit.MembershipAuditError, match="popcount mismatch"):
        audit.build_membership_index(np.array([0, 2, 2], dtype=np.uint64), max_id=2)


def _scan_config(table: str) -> dict[str, Any]:
    return {
        "scope": {
            "source_id_column": "id",
            "main_table": "main",
            "child_source_tables": [table],
            "work_target_columns": {table: "target"},
            "authorship": {
                "table": "authorships",
                "author_id_column": "author_id",
                "author_orcid_column": "orcid",
            },
            "main_strata": {
                "publication_year_column": "year",
                "primary_field_column": "field",
                "declared_authors_column": "authors_count",
            },
        }
    }


def _resource_safety(
    *, maximum_author_numeric_id_inclusive: int = 1_000_000
) -> dict[str, Any]:
    return {
        "maximum_author_numeric_id_inclusive": maximum_author_numeric_id_inclusive,
        "maximum_estimated_dense_bytes": 100_000_000,
        "minimum_available_to_estimated_ratio": 1.0,
    }


def _file_task(path: Path, *, root: Path, table: str, mode: str, index: int = 0) -> Any:
    file_stat = path.stat()
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow.remove_metadata()
    return audit.FileTask(
        index=index,
        table=table,
        mode=mode,
        record={
            "relative_path": path.relative_to(root).as_posix(),
            "bytes": file_stat.st_size,
            "mtime_ns": file_stat.st_mtime_ns,
            "footer_rows": parquet.metadata.num_rows,
            "row_groups": parquet.num_row_groups,
            "physical_schema_fingerprint_sha256": hashlib.sha256(
                schema.serialize().to_pybytes()
            ).hexdigest(),
            "schema_fields": {
                field.name: {
                    "type": str(field.type),
                    "nullable": bool(field.nullable),
                }
                for field in schema
            },
        },
        lineage="part",
    )


def test_synthetic_target_scan_keeps_source_and_target_partitions_distinct(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    table = "references"
    path = root / table / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(
                    [
                        audit.W_ZERO,
                        "https://openalex.org/W1",
                        "https://openalex.org/W3",
                        audit.W_MINUS_1,
                        None,
                        "https://openalex.org/W01",
                        "https://openalex.org/A2",
                        "https://openalex.org/W18446744073709551616",
                    ],
                    type=pa.string(),
                ),
                "target": pa.array(
                    [
                        audit.W_ZERO,
                        "https://openalex.org/W2",
                        "https://openalex.org/W4",
                        audit.W_MINUS_1,
                        None,
                        "https://openalex.org/W02",
                        "not-an-openalex-id",
                        "https://openalex.org/W18446744073709551616",
                    ],
                    type=pa.string(),
                ),
            }
        ),
        path,
        row_group_size=3,
    )
    task = _file_task(path, root=root, table=table, mode="target")
    membership = audit.build_membership_index(
        np.array([0, 1, 2], dtype=np.uint64), max_id=2
    )

    metric = audit.scan_file(
        root=root,
        output_dir=tmp_path / "output",
        task=task,
        config=_scan_config(table),
        plan={"plan_sha256": "synthetic-plan"},
        membership=membership,
        batch_size=2,
    )
    counts = metric["counts"]

    assert counts["raw_rows"] == 8
    assert counts["source_member_rows"] == 2
    assert counts["source_orphan_rows"] == 1
    assert counts["source_w_minus_1_rows"] == 1
    assert counts["source_w_zero_rows"] == 1
    assert counts["source_null_rows"] == 1
    assert counts["source_malformed_other_rows"] == 3
    assert (
        counts["source_member_rows"]
        + counts["source_orphan_rows"]
        + counts["source_w_minus_1_rows"]
        + counts["source_null_rows"]
        + counts["source_malformed_other_rows"]
        == counts["raw_rows"]
    )

    # Target W0 is a separate frozen diagnostic category, not also an
    # in-snapshot target row in the mutually-exclusive target partition.
    assert counts["target_in_snapshot_rows"] == 1
    assert counts["target_external_rows"] == 1
    assert counts["target_w_zero_rows"] == 1
    assert counts["target_w_minus_1_rows"] == 1
    assert counts["target_null_rows"] == 1
    assert counts["target_malformed_rows"] == 3
    assert (
        counts["target_in_snapshot_rows"]
        + counts["target_external_rows"]
        + counts["target_w_zero_rows"]
        + counts["target_w_minus_1_rows"]
        + counts["target_null_rows"]
        + counts["target_malformed_rows"]
        == counts["raw_rows"]
    )
    assert counts["target_self_loop_rows"] == 0
    resumed = audit.scan_file(
        root=root,
        output_dir=tmp_path / "output",
        task=task,
        config=_scan_config(table),
        plan={"plan_sha256": "synthetic-plan"},
        membership=membership,
        batch_size=2,
    )
    assert resumed["artifact"]["sha256"] == metric["artifact"]["sha256"]


def _table_row_for_blocking_ledger(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "table": metric["table"],
        "mode": metric["mode"],
        "counts": metric["counts"],
        "blocking_identifier_ledger": copy.deepcopy(
            metric["blocking_identifier_locators"]
        ),
    }


def _rehash_locator(locator: dict[str, Any]) -> None:
    payload = {key: value for key, value in locator.items() if key != "row_digest_sha256"}
    locator["row_digest_sha256"] = audit.sha256_bytes(
        audit.canonical_json_bytes(payload)
    )


@pytest.mark.parametrize("tamper", ["path", "row_group", "raw_identifier"])
def test_blocking_ledger_is_bound_to_actual_parquet_row(
    tmp_path: Path, tamper: str
) -> None:
    root = tmp_path / "root"
    table = "references"
    path = root / table / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(
                    [None, "https://openalex.org/W3", "not-a-work"],
                    type=pa.string(),
                ),
                "target": pa.array(
                    [audit.W_ZERO, audit.W_MINUS_1, "not-a-target"],
                    type=pa.string(),
                ),
            }
        ),
        path,
        row_group_size=2,
    )
    task = _file_task(path, root=root, table=table, mode="target")
    config = _scan_config(table)
    membership = audit.build_membership_index(
        np.array([0, 1, 2], dtype=np.uint64), max_id=2
    )
    metric = audit.scan_file(
        root=root,
        output_dir=tmp_path / "output",
        task=task,
        config=config,
        plan={"plan_sha256": "synthetic-plan"},
        membership=membership,
        batch_size=1,
    )
    table_row = _table_row_for_blocking_ledger(metric)
    clean = audit.validate_blocking_identifier_ledger(
        [table_row],
        config,
        root=root,
        tasks=[task],
        metrics=[metric],
        membership=membership,
    )
    assert clean["pass"] is True
    assert clean["source_row_binding"]["checked_source_rows_by_domain"] == {
        "source": 3,
        "target": 3,
        "author": 0,
        "main_field": 0,
    }

    locator = next(
        row
        for row in table_row["blocking_identifier_ledger"]
        if row["disposition"].startswith("source_")
    )
    if tamper == "path":
        locator["relative_path"] = "references/nonexistent.parquet"
    elif tamper == "row_group":
        locator["row_group"] = 99
    else:
        locator["raw_identifier"] = "https://openalex.org/W1"
    _rehash_locator(locator)

    tampered = audit.validate_blocking_identifier_ledger(
        [table_row],
        config,
        root=root,
        tasks=[task],
        metrics=[metric],
        membership=membership,
    )
    assert tampered["pass"] is False
    assert tampered["domain_passes"]["source"] is False
    assert tampered["domain_passes"]["target"] is True


def test_blocking_ledger_counts_are_exact_per_checkpoint_file(tmp_path: Path) -> None:
    root = tmp_path / "root"
    table = "awards"
    paths = [root / table / "part-0.parquet", root / table / "part-1.parquet"]
    for path, value in zip(paths, [None, "https://openalex.org/W1"], strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"id": pa.array([value], type=pa.string())}), path)
    tasks = [
        _file_task(path, root=root, table=table, mode="source", index=index)
        for index, path in enumerate(paths)
    ]
    membership = np.ones(2, dtype=np.uint8)
    metrics = [
        audit.scan_file(
            root=root,
            output_dir=tmp_path / "output",
            task=task,
            config=_scan_config(table),
            plan={"plan_sha256": "synthetic-plan"},
            membership=membership,
            batch_size=1,
        )
        for task in tasks
    ]
    clean, _ = audit.validate_per_file_blocking_identifier_ledgers(
        tasks=tasks, metrics=metrics
    )
    assert clean["pass"] is True

    shifted = copy.deepcopy(metrics)
    shifted[0]["counts"]["source_null_rows"] = 0
    shifted[1]["counts"]["source_null_rows"] = 1
    tampered, _ = audit.validate_per_file_blocking_identifier_ledgers(
        tasks=tasks, metrics=shifted
    )
    assert tampered["pass"] is False
    assert tampered["domain_passes"]["source"] is False

    boolean_coordinate = copy.deepcopy(metrics)
    boolean_coordinate[0]["blocking_identifier_locators"][0]["row_group"] = False
    _rehash_locator(boolean_coordinate[0]["blocking_identifier_locators"][0])
    tampered, _ = audit.validate_per_file_blocking_identifier_ledgers(
        tasks=tasks, metrics=boolean_coordinate
    )
    assert tampered["pass"] is False

    boolean_task_index = copy.deepcopy(metrics)
    boolean_task_index[0]["task_index"] = False
    tampered, _ = audit.validate_per_file_blocking_identifier_ledgers(
        tasks=tasks, metrics=boolean_task_index
    )
    assert tampered["pass"] is False


@pytest.mark.parametrize(
    ("live_type", "frozen_type"),
    [
        (pa.large_string(), None),
        (pa.string(), "large_string"),
    ],
    ids=("live-arrow-type", "frozen-manifest-type"),
)
def test_scan_rejects_selected_column_type_mismatch_in_live_or_frozen_schema(
    tmp_path: Path,
    live_type: pa.DataType,
    frozen_type: str | None,
) -> None:
    root = tmp_path / "root"
    table = "awards"
    path = root / table / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(
                    ["https://openalex.org/W1"],
                    type=live_type,
                )
            }
        ),
        path,
    )
    task = _file_task(path, root=root, table=table, mode="source")
    if frozen_type is not None:
        task.record["schema_fields"]["id"]["type"] = frozen_type

    with pytest.raises(
        audit.MembershipAuditError, match="selected column type differs"
    ):
        audit.scan_file(
            root=root,
            output_dir=tmp_path / "output",
            task=task,
            config=_scan_config(table),
            plan={"plan_sha256": "synthetic-plan"},
            membership=np.ones(2, dtype=np.uint8),
            batch_size=1,
        )


def test_author_file_aggregate_and_status_codes_cover_all_four_work_classes() -> None:
    ids = np.array([2, 1, 1, 3, 2, 4], dtype=np.uint64)
    # 0=null, 1=grammar-valid, 2=non-null malformed.
    categories = np.array([1, 0, 1, 2, 0, 1], dtype=np.uint8)

    aggregated = audit._aggregate_author_file(ids, categories)

    assert aggregated["id"].tolist() == [1, 2, 3, 4]
    assert aggregated["slots"].tolist() == [2, 2, 1, 1]
    assert aggregated["null_ids"].tolist() == [1, 1, 0, 0]
    assert aggregated["valid_ids"].tolist() == [1, 1, 0, 1]
    assert aggregated["invalid_ids"].tolist() == [0, 0, 1, 0]
    np.testing.assert_array_equal(
        aggregated["slots"],
        aggregated["null_ids"] + aggregated["valid_ids"] + aggregated["invalid_ids"],
    )

    status = audit.author_status_codes(
        np.array([1, 2, 2, 0], dtype=np.uint32),
        np.array([1, 1, 0, 0], dtype=np.uint32),
    )
    assert status.tolist() == [0, 1, 2, 3]


def _store_artifact(
    output_dir: Path, task: Any, kind: str, array: np.ndarray
) -> dict[str, Any]:
    path = audit.artifact_path(output_dir, task, kind)
    return {"kind": kind, **audit.atomic_save_array(path, array)}


def test_author_work_profile_includes_complete_partial_none_and_zero_row(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    main_task = audit.FileTask(
        0, "main", "main", {"relative_path": "main/part-0.parquet"}, "part"
    )
    author_task = audit.FileTask(
        1,
        "authorships",
        "authorship",
        {"relative_path": "authorships/base-0.parquet"},
        "base",
    )
    main = np.zeros(4, dtype=audit.MAIN_DTYPE)
    main["id"] = [0, 1, 2, 3]
    main["year"] = [audit.YEAR_NULL, 2020, 2020, 2021]
    main["field"] = [audit.FIELD_NULL, 1, 1, 2]
    main["declared_authors"] = [0, 1, 2, 2]
    authors = np.zeros(3, dtype=audit.AUTHOR_DTYPE)
    authors["id"] = [1, 2, 3]
    authors["slots"] = [1, 2, 2]
    authors["null_ids"] = [0, 1, 2]
    authors["valid_ids"] = [1, 1, 0]
    authors["invalid_ids"] = [0, 0, 0]
    metrics = {
        0: {"artifact": _store_artifact(output_dir, main_task, "main", main)},
        1: {
            "artifact": _store_artifact(output_dir, author_task, "author", authors),
            "extra_artifacts": [
                {
                    **_store_artifact(
                        output_dir,
                        author_task,
                        "author_ids",
                        np.array([3, 10], dtype=np.uint64),
                    ),
                    "maximum_numeric_id": 10,
                }
            ],
            "counts": {
                "source_member_rows": 5,
                "author_null_id_rows": 3,
                "author_valid_id_rows": 2,
                "author_invalid_id_rows": 0,
                "author_orcid_null_rows": 2,
                "author_orcid_nonnull_rows": 3,
            },
        },
    }

    profile = audit.author_work_profile(
        output_dir=output_dir,
        tasks=[main_task, author_task],
        metrics_by_index=metrics,
        max_id=3,
        resource_safety=_resource_safety(),
    )

    assert profile["master_work_rows_profiled"] == 4
    assert profile["grammar_usable_author_id_status"] == {
        "complete": 1,
        "partial": 1,
        "none_usable": 1,
        "zero_authorship_row": 1,
    }
    assert profile["raw_null_only_status"] == {
        "complete": 1,
        "partial": 1,
        "none_usable": 1,
        "zero_authorship_row": 1,
    }
    assert profile["authorship_partition_conserved"] is True
    assert profile["declared_authors_count"] == {
        "null": 0,
        "invalid_nonnull": 0,
        "matches_observed_slots": 4,
        "mismatches_observed_slots": 0,
    }
    assert profile["distinct_grammar_valid_nonnull_author_ids"] == 2
    assert profile["retained_authorship_rows"] == 5
    assert profile["author_orcid_rows"] == {"null": 2, "non_null": 3}
    assert (
        sum(profile["grammar_usable_author_id_status"].values())
        == (profile["master_work_rows_profiled"])
    )
    assert (
        sum(profile["raw_null_only_status"].values())
        == (profile["master_work_rows_profiled"])
    )
    assert (
        sum(profile["declared_authors_count"].values())
        == (profile["master_work_rows_profiled"])
    )
    assert profile["by_publication_year"]
    assert profile["by_primary_field"]
    assert profile["by_observed_team_size"]
    assert profile["by_authorship_source_lineage_mask"]
    for margin_name in (
        "by_publication_year",
        "by_primary_field",
        "by_observed_team_size",
        "by_authorship_source_lineage_mask",
    ):
        assert (
            sum(sum(row["statuses"].values()) for row in profile[margin_name])
            == profile["master_work_rows_profiled"]
        )
    assert (
        sum(row["works"] for row in profile["joint_sparse_cube"])
        == (profile["master_work_rows_profiled"])
    )
    assert profile["w0"] == {
        "observed_authorship_slots": 0,
        "valid_author_ids": 0,
        "null_author_ids": 0,
        "invalid_author_ids": 0,
        "status": "zero_authorship_row",
    }


def test_author_work_profile_rejects_excessive_maximum_before_dense_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    task = audit.FileTask(
        0,
        "authorships",
        "authorship",
        {"relative_path": "authorships/base-0.parquet"},
        "base",
    )
    authors = np.zeros(1, dtype=audit.AUTHOR_DTYPE)
    authors["id"] = [1]
    authors["slots"] = [1]
    authors["valid_ids"] = [1]
    excessive_author_id = 1_000_000_000_000
    metrics = {
        0: {
            "artifact": _store_artifact(output_dir, task, "author", authors),
            "extra_artifacts": [
                {
                    **_store_artifact(
                        output_dir,
                        task,
                        "author_ids",
                        np.array([excessive_author_id], dtype=np.uint64),
                    ),
                    "maximum_numeric_id": excessive_author_id,
                }
            ],
            "counts": {
                "source_member_rows": 1,
                "author_orcid_null_rows": 1,
                "author_orcid_nonnull_rows": 0,
            },
        }
    }
    original_zeros = audit.np.zeros

    def guarded_zeros(shape: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        if isinstance(shape, (int, np.integer)) and int(shape) > 1_000_000:
            pytest.fail(
                "dense author-ID allocation was attempted before cap validation"
            )
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(audit.np, "zeros", guarded_zeros)

    with pytest.raises(
        audit.MembershipAuditError, match="(?i)author.*(cap|maximum|dense)"
    ):
        audit.author_work_profile(
            output_dir=output_dir,
            tasks=[task],
            metrics_by_index=metrics,
            max_id=1,
            resource_safety=_resource_safety(
                maximum_author_numeric_id_inclusive=1_000_000
            ),
        )


def test_audit_preflights_worst_case_dense_memory_before_membership_allocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[str] = []
    config = {
        "execution": {
            "workers": 1,
            "batch_size": 1,
            "resource_safety": _resource_safety(),
        }
    }

    class SyntheticMaster:
        array = np.array([0, 7], dtype=np.uint64)

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(audit, "load_config", lambda _path: config)
    monkeypatch.setattr(audit, "validate_evidence", lambda _config: {})
    monkeypatch.setattr(
        audit, "validate_parent", lambda **_kwargs: ({}, "1" * 64, {})
    )
    monkeypatch.setattr(audit, "build_tasks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        audit,
        "plan_payload",
        lambda **_kwargs: {"plan_sha256": "synthetic-plan"},
    )
    monkeypatch.setattr(audit, "prepare_output_dir", lambda *_args: None)
    monkeypatch.setattr(audit, "validate_master_ids", lambda *_args: SyntheticMaster())

    def fail_preflight(**kwargs: Any) -> dict[str, Any]:
        events.append(f"preflight:{kwargs['maximum_author_id']}")
        raise audit.MembershipAuditError("synthetic resource rejection")

    monkeypatch.setattr(audit, "validate_dense_resource_preflight", fail_preflight)
    monkeypatch.setattr(
        audit,
        "build_membership_index",
        lambda *_args: pytest.fail("membership allocated before resource preflight"),
    )

    with pytest.raises(audit.MembershipAuditError, match="resource rejection"):
        audit.audit(
            root=tmp_path / "root",
            output_dir=tmp_path / "output",
            config_path=tmp_path / "config.yaml",
            manifest_path=tmp_path / "manifest.json",
            master_ids_path=tmp_path / "master.u64",
        )

    assert events == ["preflight:1000000", "close"]


def test_resource_preflight_evidence_excludes_nondeterministic_available_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safety = _resource_safety(maximum_author_numeric_id_inclusive=10)
    monkeypatch.setattr(audit, "available_memory_bytes", lambda: 10_000_000)
    first = audit.validate_dense_resource_preflight(
        max_work_id=7, maximum_author_id=10, resource_safety=safety
    )
    monkeypatch.setattr(audit, "available_memory_bytes", lambda: 20_000_000)
    second = audit.validate_dense_resource_preflight(
        max_work_id=7, maximum_author_id=10, resource_safety=safety
    )

    assert first == second
    assert "available_bytes_at_preflight" not in first


def test_field_parser_preserves_null_and_rejects_non_normal_or_out_of_range_ids() -> (
    None
):
    values = pa.array(
        [
            None,
            "https://openalex.org/fields/1",
            "https://openalex.org/fields/254",
            "https://openalex.org/fields/0",
            "https://openalex.org/fields/01",
            "https://openalex.org/fields/255",
            "https://openalex.org/W1",
        ],
        type=pa.string(),
    )

    fields, null, invalid = audit.parse_field(values)

    assert fields.tolist() == [0, 1, 254, 255, 255, 255, 255]
    assert null.tolist() == [True, False, False, False, False, False, False]
    assert invalid.tolist() == [False, False, False, True, True, True, True]


def test_authorship_scan_commits_distinct_author_id_artifact_and_maximum(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    table = "authorships"
    path = root / table / "base-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["https://openalex.org/W1", "https://openalex.org/W2"]),
                "author_id": pa.array(
                    ["https://openalex.org/A10", "https://openalex.org/A3"]
                ),
                "orcid": pa.array([None, "0000-0001"], type=pa.string()),
            }
        ),
        path,
    )
    task = _file_task(path, root=root, table=table, mode="authorship")
    membership = audit.build_membership_index(
        np.array([0, 1, 2], dtype=np.uint64), max_id=2
    )

    metric = audit.scan_file(
        root=root,
        output_dir=tmp_path / "output",
        task=task,
        config=_scan_config(table),
        plan={"plan_sha256": "synthetic-plan"},
        membership=membership,
        batch_size=1,
    )

    extra = next(
        record for record in metric["extra_artifacts"] if record["kind"] == "author_ids"
    )
    assert extra["maximum_numeric_id"] == 10
    author_ids = audit.load_extra_artifact(
        tmp_path / "output", task, metric, "author_ids"
    )
    assert author_ids.tolist() == [3, 10]
    assert metric["counts"]["author_orcid_null_rows"] == 1
    assert metric["counts"]["author_orcid_nonnull_rows"] == 1
    resumed = audit.scan_file(
        root=root,
        output_dir=tmp_path / "output",
        task=task,
        config=_scan_config(table),
        plan={"plan_sha256": "synthetic-plan"},
        membership=membership,
        batch_size=1,
    )
    assert resumed["artifact"]["sha256"] == metric["artifact"]["sha256"]
    assert audit.canonical_json_bytes(
        resumed["extra_artifacts"]
    ) == audit.canonical_json_bytes(metric["extra_artifacts"])


def test_main_artifact_checkpoint_clean_resume_replays_exactly(tmp_path: Path) -> None:
    root = tmp_path / "root"
    table = "main"
    path = root / table / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(["https://openalex.org/W1"], type=pa.string()),
                "year": pa.array([2020.0], type=pa.float64()),
                "field": pa.array(
                    ["https://openalex.org/fields/1"], type=pa.string()
                ),
                "authors_count": pa.array([1], type=pa.int64()),
            }
        ),
        path,
    )
    task = _file_task(path, root=root, table=table, mode="main")
    config = _scan_config(table)
    config["work_profile_strata"] = {
        "dimensions": {
            "publication_year": {"physical_type": "double"},
            "primary_field": {"physical_type": "string"},
            "authors_count": {"physical_type": "int64"},
        }
    }
    membership = np.ones(2, dtype=np.uint8)
    output_dir = tmp_path / "output"
    metric = audit.scan_file(
        root=root,
        output_dir=output_dir,
        task=task,
        config=config,
        plan={"plan_sha256": "synthetic-plan"},
        membership=membership,
        batch_size=1,
    )
    resumed = audit.scan_file(
        root=root,
        output_dir=output_dir,
        task=task,
        config=config,
        plan={"plan_sha256": "synthetic-plan"},
        membership=membership,
        batch_size=1,
    )

    assert resumed["artifact"]["sha256"] == metric["artifact"]["sha256"]


def test_year_parser_distinguishes_arrow_null_from_nonfinite_and_nonintegral() -> None:
    values = pa.array(
        [None, np.nan, 2020.0, 2020.5, np.inf, -32767.0, 32767.0, 32768.0],
        type=pa.float64(),
    )

    years, invalid = audit.parse_year(values)

    assert years.tolist() == [
        audit.YEAR_NULL,
        audit.YEAR_NULL,
        2020,
        audit.YEAR_NULL,
        audit.YEAR_NULL,
        -32767,
        32767,
        audit.YEAR_NULL,
    ]
    assert invalid.tolist() == [False, True, False, True, True, False, False, True]


def _checkpoint_metric(
    *, task: Any, plan: dict[str, Any], artifact: dict[str, Any] | None
) -> dict[str, Any]:
    metric = {
        "schema_version": audit.CHECKPOINT_SCHEMA_VERSION,
        "plan_sha256": plan["plan_sha256"],
        "task_index": task.index,
        "table": task.table,
        "relative_path": task.record["relative_path"],
        "file_record_sha256": audit.file_record_sha256(task.record),
        "footer_rows": task.record["footer_rows"],
        "counts": {"raw_rows": task.record["footer_rows"]},
        "artifact": artifact,
    }
    metric["metric_sha256"] = audit.checkpoint_digest(metric)
    return metric


def _checkpoint_task() -> Any:
    return audit.FileTask(
        index=0,
        table="references",
        mode="target",
        record={"relative_path": "references/part-0.parquet", "footer_rows": 2},
        lineage="part",
    )


def test_checkpoint_metric_tamper_is_rejected(tmp_path: Path) -> None:
    task = _checkpoint_task()
    plan = {"plan_sha256": "synthetic-plan"}
    metric = _checkpoint_metric(task=task, plan=plan, artifact=None)
    path = audit.checkpoint_path(tmp_path, task)
    audit.atomic_write_json(path, metric)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["counts"]["raw_rows"] = 1
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(audit.MembershipAuditError, match="checkpoint validation"):
        audit.load_checkpoint(output_dir=tmp_path, task=task, plan=plan)


def test_same_size_artifact_tamper_is_rejected_before_use(tmp_path: Path) -> None:
    task = _checkpoint_task()
    plan = {"plan_sha256": "synthetic-plan"}
    array = np.array([(1, 1), (2, 1)], dtype=audit.TARGET_DTYPE)
    artifact = _store_artifact(tmp_path, task, "target", array)
    metric = _checkpoint_metric(task=task, plan=plan, artifact=artifact)
    audit.atomic_write_json(audit.checkpoint_path(tmp_path, task), metric)
    artifact_file = audit.artifact_path(tmp_path, task, "target")
    payload = bytearray(artifact_file.read_bytes())
    payload[-1] ^= 0x01
    artifact_file.write_bytes(payload)

    with pytest.raises(audit.MembershipAuditError, match="artifact SHA-256 changed"):
        resumed = audit.load_checkpoint(output_dir=tmp_path, task=task, plan=plan)
        assert resumed is not None
        audit.load_artifact(tmp_path, task, resumed, "target")


def test_valid_checkpoint_resume_rejects_changed_live_source_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    table = "awards"
    path = root / table / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(
                    ["https://openalex.org/W1"],
                    type=pa.string(),
                )
            }
        ),
        path,
    )
    task = _file_task(path, root=root, table=table, mode="source")
    output_dir = tmp_path / "output"
    plan = {"plan_sha256": "synthetic-plan"}
    membership = np.ones(2, dtype=np.uint8)

    metric = audit.scan_file(
        root=root,
        output_dir=output_dir,
        task=task,
        config=_scan_config(table),
        plan=plan,
        membership=membership,
        batch_size=1,
    )
    assert audit.checkpoint_path(output_dir, task).is_file()
    assert metric["start_file_identity"] == metric["end_file_identity"]

    before = path.stat()
    replacement = path.with_name("part-0.replacement.parquet")
    replacement.write_bytes(path.read_bytes())
    os.utime(
        replacement,
        ns=(before.st_atime_ns, before.st_mtime_ns),
    )
    replacement.replace(path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.stat()
    assert after.st_size == before.st_size
    assert after.st_mtime_ns == before.st_mtime_ns
    assert (after.st_dev, after.st_ino, after.st_ctime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_ctime_ns,
    )

    with pytest.raises(
        audit.MembershipAuditError,
        match="checkpoint source identity changed",
    ):
        audit.scan_file(
            root=root,
            output_dir=output_dir,
            task=task,
            config=_scan_config(table),
            plan=plan,
            membership=membership,
            batch_size=1,
        )


def test_checkpoint_resume_replays_source_semantics_before_trusting_self_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    table = "awards"
    path = root / table / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"id": pa.array([None], type=pa.string())}),
        path,
    )
    task = _file_task(path, root=root, table=table, mode="source")
    output_dir = tmp_path / "output"
    plan = {"plan_sha256": "synthetic-plan"}
    membership = np.ones(2, dtype=np.uint8)
    audit.scan_file(
        root=root,
        output_dir=output_dir,
        task=task,
        config=_scan_config(table),
        plan=plan,
        membership=membership,
        batch_size=1,
    )
    audit.scan_file(
        root=root,
        output_dir=output_dir,
        task=task,
        config=_scan_config(table),
        plan=plan,
        membership=membership,
        batch_size=1,
    )
    checkpoint = audit.checkpoint_path(output_dir, task)
    metric = json.loads(checkpoint.read_text(encoding="utf-8"))
    metric["counts"]["source_null_rows"] = 0
    metric["counts"]["source_member_rows"] = 1
    metric["blocking_identifier_locators"] = []
    metric["metric_sha256"] = audit.checkpoint_digest(metric)
    audit.atomic_write_json(checkpoint, metric)

    with pytest.raises(
        audit.MembershipAuditError, match="checkpoint semantic replay differs"
    ):
        audit.scan_file(
            root=root,
            output_dir=output_dir,
            task=task,
            config=_scan_config(table),
            plan=plan,
            membership=membership,
            batch_size=1,
        )


def test_output_location_rejects_symlink_ancestor_into_source(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)

    with pytest.raises(audit.MembershipAuditError, match="must not contain symlinks"):
        audit.validate_output_location(source, alias / "audit-output")


def test_scan_rejects_preexisting_symlinked_table_output_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    table = "awards"
    path = root / table / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({"id": pa.array(["https://openalex.org/W1"], type=pa.string())}),
        path,
    )
    task = _file_task(path, root=root, table=table, mode="source")
    output_dir = tmp_path / "output"
    (output_dir / "checkpoints").mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (output_dir / "checkpoints" / table).symlink_to(
        external, target_is_directory=True
    )

    with pytest.raises(
        audit.MembershipAuditError, match="invalid output table directory"
    ):
        audit.scan_file(
            root=root,
            output_dir=output_dir,
            task=task,
            config=_scan_config(table),
            plan={"plan_sha256": "synthetic-plan"},
            membership=np.ones(2, dtype=np.uint8),
            batch_size=1,
        )
    assert list(external.iterdir()) == []


def test_scan_rejects_same_fd_end_identity_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    table = "awards"
    path = root / table / "part-0.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(
                    ["https://openalex.org/W1"],
                    type=pa.string(),
                )
            }
        ),
        path,
    )
    task = _file_task(path, root=root, table=table, mode="source")
    original_file_identity = audit.file_identity
    calls = 0

    def tampered_file_identity(file_stat: os.stat_result) -> dict[str, int]:
        nonlocal calls
        calls += 1
        identity = original_file_identity(file_stat)
        if calls == 2:
            identity = {**identity, "ctime_ns": identity["ctime_ns"] + 1}
        return identity

    monkeypatch.setattr(audit, "file_identity", tampered_file_identity)

    with pytest.raises(audit.MembershipAuditError, match="input changed while open"):
        audit.scan_file(
            root=root,
            output_dir=tmp_path / "output",
            task=task,
            config=_scan_config(table),
            plan={"plan_sha256": "synthetic-plan"},
            membership=np.ones(2, dtype=np.uint8),
            batch_size=1,
        )
    assert calls == 2


def test_master_id_symlink_is_rejected(tmp_path: Path) -> None:
    values = np.array([0, 2], dtype="<u8")
    real_path = tmp_path / "master-real.u64"
    values.tofile(real_path)
    symlink_path = tmp_path / "master-link.u64"
    symlink_path.symlink_to(real_path)
    config = {
        "parents": {
            "grammar_recount": {
                "sorted_ids_bytes": real_path.stat().st_size,
                "sorted_ids_sha256": _sha256(real_path),
                "sorted_ids_rows": len(values),
                "min_numeric_id": int(values[0]),
                "max_numeric_id": int(values[-1]),
            }
        }
    }

    with pytest.raises(audit.MembershipAuditError, match="regular file"):
        audit.validate_master_ids(symlink_path, config)


def test_sentinel_contract_ignores_expected_zero_tables_when_comparing_incidence() -> (
    None
):
    config = {
        "scope": {"main_table": "not-scanned-in-this-unit-test"},
        "quarantine": {
            "expected_w_minus_1": {
                "main": {
                    "files": [{"relative_path": "main/part-0.parquet", "rows": 1}]
                },
                "empty": {"files": []},
            },
            "expected_w0": {
                "main": {
                    "files": [{"relative_path": "main/part-1.parquet", "rows": 1}]
                },
                "empty": {"files": []},
            },
        },
    }
    table_rows = [
        {
            "table": "main",
            "w_minus_1_affected_files": {"main/part-0.parquet": 1},
            "w_zero_affected_files": {"main/part-1.parquet": 1},
        },
        {
            "table": "empty",
            "w_minus_1_affected_files": {},
            "w_zero_affected_files": {},
        },
    ]

    result = audit.validate_sentinel_contract(
        table_rows=table_rows, config=config, full_scope=True
    )

    assert result["w_minus_1_exact_match"] is True
    assert result["w0_exact_match"] is True


def _write_config(
    tmp_path: Path,
    raw: dict[str, Any],
    name: str,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    changed = copy.deepcopy(raw)
    mutate(changed)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(changed, sort_keys=False), encoding="utf-8")
    return path


def test_config_loader_rejects_identity_scope_and_quarantine_tamper(
    tmp_path: Path,
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    invalid_hash = _write_config(
        tmp_path,
        raw,
        "invalid-hash.yaml",
        lambda value: value["parent_inventory"].__setitem__("manifest_sha256", "x"),
    )
    with pytest.raises(audit.MembershipAuditError, match="lowercase SHA-256"):
        audit.load_config(invalid_hash)

    missing_table = _write_config(
        tmp_path,
        raw,
        "missing-table.yaml",
        lambda value: value["child_source_membership"]["tables"].pop(
            next(iter(value["child_source_membership"]["tables"]))
        ),
    )
    with pytest.raises(audit.MembershipAuditError, match="exactly 22 child tables"):
        audit.load_config(missing_table)

    wrong_quarantine = _write_config(
        tmp_path,
        raw,
        "wrong-quarantine.yaml",
        lambda value: value["sentinel_propagation"]["w_minus_1"].__setitem__(
            "raw_value", "https://openalex.org/W-2"
        ),
    )
    with pytest.raises(audit.MembershipAuditError, match="freeze exact W-1"):
        audit.load_config(wrong_quarantine)


@pytest.mark.parametrize(
    ("variant", "message"),
    [
        ("main-table", "canonical main table"),
        ("target-table", "canonical work-target columns"),
        ("target-column", "canonical work-target columns"),
        ("success-status", "canonical success status"),
    ],
)
def test_config_loader_rejects_canonical_name_and_status_variants(
    tmp_path: Path,
    variant: str,
    message: str,
) -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def mutate(value: dict[str, Any]) -> None:
        if variant == "main-table":
            value["main_accounting"]["table"] = "openalex_works"
        elif variant == "target-table":
            tables = value["work_target_coverage"]["tables"]
            canonical = "openalex_works_20260330__referenced_works"
            tables[f"{canonical}_alias"] = tables.pop(canonical)
        elif variant == "target-column":
            tables = value["work_target_coverage"]["tables"]
            tables["openalex_works_20260330__referenced_works"]["target_column"] = (
                "references"
            )
        else:
            value["success_contract"]["success_status"] = (
                "PASS_WORK_SOURCE_MEMBERSHIP_AUDIT"
            )

    path = _write_config(
        tmp_path,
        raw,
        f"canonical-{variant}.yaml",
        mutate,
    )

    with pytest.raises(audit.MembershipAuditError, match=message):
        audit.load_config(path)


def test_tracked_config_conserves_counts_and_binds_parent_evidence() -> None:
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    loaded = audit.load_config(CONFIG_PATH)

    children = raw["child_source_membership"]
    assert len(children["tables"]) == children["expected_child_table_count"] == 22
    assert (
        sum(row["expected_files"] for row in children["tables"].values())
        == (children["expected_child_files"])
    )
    assert (
        sum(row["expected_raw_rows"] for row in children["tables"].values())
        == (children["expected_raw_child_rows"])
    )
    assert children["expected_raw_child_rows"] == (
        children["expected_retained_child_rows_on_pass"]
        + children["expected_frozen_quarantine_child_rows"]
    )

    main = raw["main_accounting"]
    assert main["expected_raw_rows"] == (
        main["expected_retained_rows"] + main["expected_frozen_quarantine_rows"]
    )
    global_identity = raw["source_row_partition"]["global_identity_on_pass"]
    assert global_identity["raw_rows"] == (
        global_identity["retained_rows"]
        + global_identity["frozen_w_minus_1_quarantine_rows"]
        + global_identity["all_other_partition_rows"]
    )

    incidence = raw["sentinel_propagation"]["source_incidence_by_table"]
    assert sum(row["w_minus_1"]["expected_rows"] for row in incidence.values()) == 21
    assert sum(row["w_zero"]["expected_rows"] for row in incidence.values()) == 6
    assert (
        sum(bool(row["w_minus_1"]["expected_rows"]) for row in incidence.values()) == 6
    )
    assert sum(bool(row["w_zero"]["expected_rows"]) for row in incidence.values()) == 6

    parents = (
        (
            raw["parent_inventory"],
            "manifest_relative_path",
            "manifest_sha256",
        ),
        (
            raw["parent_inventory"],
            "summary_relative_path",
            "summary_sha256",
        ),
        (
            raw["parent_inventory"],
            "config_relative_path",
            "config_sha256",
        ),
        (
            raw["parent_grammar_valid_main_id_recount"],
            "summary_relative_path",
            "summary_sha256",
        ),
        (
            raw["parent_grammar_valid_main_id_recount"],
            "config_relative_path",
            "config_sha256",
        ),
        (
            raw["parent_sentinel_incidence"],
            "summary_relative_path",
            "summary_sha256",
        ),
        (
            raw["parent_sentinel_incidence"],
            "config_relative_path",
            "config_sha256",
        ),
    )
    for parent, path_key, hash_key in parents:
        assert parent[hash_key] == _sha256(REPO_ROOT / parent[path_key])

    assert loaded["scope"]["main_table"] == raw["main_accounting"]["table"]
    assert len(loaded["scope"]["child_source_tables"]) == 22
    assert loaded["quarantine"]["raw_value"] == audit.W_MINUS_1
    assert loaded["quarantine"]["w0_value"] == audit.W_ZERO
    assert raw["success_contract"]["required_component_states"]["pass_g1"] is False


def _full_pass_author_profile() -> dict[str, Any]:
    statuses = {
        "complete": 1,
        "partial": 1,
        "none_usable": 1,
        "zero_authorship_row": 1,
    }
    return {
        "master_work_rows_profiled": 4,
        "grammar_usable_author_id_status": dict(statuses),
        "raw_null_only_status": dict(statuses),
        "invalid_nonnull_author_id_rows": 0,
        "author_id_rows": {
            "null": 3,
            "grammar_valid_non_null": 2,
            "invalid_non_null": 0,
        },
        "distinct_grammar_valid_nonnull_author_ids": 2,
        "retained_authorship_rows": 5,
        "author_orcid_rows": {"null": 2, "non_null": 3},
        "authorship_partition_conserved": True,
        "declared_authors_count": {
            "null": 0,
            "invalid_nonnull": 0,
            "matches_observed_slots": 4,
            "mismatches_observed_slots": 0,
        },
        "by_declared_authors_count_band": [
            {"declared_authors_band": "1", "statuses": dict(statuses)}
        ],
        "declared_authors_count_mismatch_profile": [],
        "by_publication_year": [{"year": 2020, "statuses": dict(statuses)}],
        "by_primary_field": [{"field_id": 1, "statuses": dict(statuses)}],
        "joint_sparse_cube": [
            {
                "publication_year": 2020,
                "primary_field_id": 1,
                "declared_authors_band": "1",
                "author_id_status": name,
                "works": count,
            }
            for name, count in statuses.items()
        ],
        "by_observed_team_size": [{"team_size": "1", "statuses": dict(statuses)}],
        "by_authorship_source_lineage_mask": [
            {"lineage_mask": 1, "statuses": dict(statuses)}
        ],
        "w0": {
            "observed_authorship_slots": 1,
            "valid_author_ids": 0,
            "null_author_ids": 1,
            "invalid_author_ids": 0,
            "status": "none_usable",
        },
        "candidate_eligibility_stratum": "NOT_EVALUABLE_G0_PENDING",
        "author_entity_validity_established": False,
    }


def _aggregate_full_pass_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: dict[str, Any],
) -> dict[str, Any]:
    def counts(**updates: int) -> dict[str, int]:
        value = {
            "raw_rows": 0,
            "source_member_rows": 0,
            "source_orphan_rows": 0,
            "source_null_rows": 0,
            "source_w_minus_1_rows": 0,
            "source_w_zero_rows": 0,
            "source_malformed_other_rows": 0,
            "target_in_snapshot_rows": 0,
            "target_external_rows": 0,
            "target_null_rows": 0,
            "target_w_minus_1_rows": 0,
            "target_w_zero_rows": 0,
            "target_malformed_rows": 0,
            "target_self_loop_rows": 0,
            "author_raw_null_id_rows": 0,
            "author_raw_invalid_id_rows": 0,
            "author_w_minus_1_null_id_rows": 0,
            "author_w_zero_null_id_rows": 0,
            "main_year_invalid_rows": 0,
            "main_field_invalid_rows": 0,
            "main_declared_authors_invalid_rows": 0,
        }
        value.update(updates)
        return value

    def table_row(table: str, mode: str, row_counts: dict[str, int]) -> dict[str, Any]:
        return {
            "table": table,
            "mode": mode,
            "expected_parent_files": 1,
            "scanned_files": 1,
            "expected_parent_footer_rows": row_counts["raw_rows"],
            "scanned_footer_rows": row_counts["raw_rows"],
            "counts": row_counts,
            "source_row_conservation": True,
            "blocking_identifier_ledger": [],
        }

    table_rows = [
        table_row("main", "main", counts(raw_rows=4, source_member_rows=4)),
        table_row(
            "authorships",
            "authorship",
            counts(
                raw_rows=6,
                source_member_rows=5,
                source_w_minus_1_rows=1,
                source_w_zero_rows=1,
                author_raw_null_id_rows=3,
                author_w_minus_1_null_id_rows=1,
                author_w_zero_null_id_rows=1,
            ),
        ),
        table_row("references", "target", counts()),
        table_row("related", "target", counts()),
    ]
    monkeypatch.setattr(audit, "aggregate_table_metrics", lambda *_args: table_rows)
    monkeypatch.setattr(
        audit,
        "validate_sentinel_contract",
        lambda **_kwargs: {
            "w_minus_1_exact_match": True,
            "w0_exact_match": True,
            "w_minus_1_locator_ledger_exact_match": True,
            "w0_locator_ledger_exact_match": True,
            "main_locator_exact_match": {"w_minus_1": True, "w_zero": True},
        },
    )
    monkeypatch.setattr(
        audit,
        "target_unique_coverage",
        lambda **_kwargs: {
            "rle_input_rows": 0,
            "per_file_rle_records": 0,
            "unique_valid_targets": 0,
            "unique_targets_in_snapshot": 0,
            "unique_targets_external_to_snapshot": 0,
            "unique_target_coverage": None,
            "external_ids_above_main_max": 0,
        },
    )
    monkeypatch.setattr(audit, "author_work_profile", lambda **_kwargs: profile)
    monkeypatch.setattr(
        audit, "git_metadata", lambda: {"commit": "synthetic", "dirty": False}
    )
    config = {
        "label": "synthetic-full-pass-boundary",
        "scope": {
            "source_id_column": "id",
            "main_table": "main",
            "child_source_tables": ["authorships", "references", "related"],
            "work_target_columns": {
                "references": "referenced_works",
                "related": "related_works",
            },
            "authorship": {"table": "authorships"},
        },
        "authorship_author_id_work_profile": {
            "expected_raw_rows": 6,
            "expected_retained_rows_on_membership_pass": 5,
            "expected_raw_author_id_null_rows_from_footer": 3,
            "w_zero_expected_source_rows": 1,
        },
        "execution": {
            "resource_safety": {
                "maximum_author_numeric_id_inclusive": 100,
                "maximum_estimated_dense_bytes": 10_000,
                "minimum_available_to_estimated_ratio": 1.0,
            }
        },
        "source_row_partition": {
            "global_identity_on_pass": {
                "raw_rows": 10,
                "retained_rows": 9,
                "frozen_w_minus_1_quarantine_rows": 1,
                "all_other_partition_rows": 0,
            }
        },
        "parent_inventory": {"expected_parquet_file_count": 0},
        "success_contract": {
            "success_status": "PASS_SYNTHETIC_COMPONENT",
            "required_component_states": {
                "main_accounting": "PASS_PARENT_RECOUNT_REVALIDATED",
                "child_source_membership": "PASS",
                "source_row_conservation": "PASS",
                "declared_work_target_scan": "COMPLETE_COVERAGE_MEASURED",
                "authorship_author_id_work_profile": "COMPLETE_PROFILE_MEASURED",
                "work_profile_strata": "COMPLETE_PROFILE_MEASURED",
                "raw_source_id_validity": "BLOCKED",
                "w_zero_entity_validity": "UNRESOLVED",
                "parent_content_identity": "UNVERIFIED",
                "g_minus_1_data_readiness": "PENDING",
                "empirical_g1_signature": "NOT_RUN",
                "pass_g1": False,
            },
        },
        "parents": {
            "inventory": {
                "source_root_id": "synthetic-root",
                "expected_parquet_file_count": 0,
                "inventory_fingerprint_sha256": "1" * 64,
                "stability_fingerprint_sha256": "2" * 64,
            },
            "grammar_recount": {
                "summary_sha256": "3" * 64,
                "sorted_ids_sha256": "4" * 64,
                "sorted_ids_rows": 4,
            },
            "sentinel_incidence": {"result_fingerprint_sha256": "5" * 64},
        },
        "pending": ["synthetic-pending"],
        "claim_boundary": "Synthetic component only; pass_g1=false.",
    }
    return audit.aggregate_report(
        root=tmp_path,
        config=config,
        config_path=CONFIG_PATH,
        tool_path=SCRIPT_PATH,
        manifest={},
        manifest_sha256="6" * 64,
        plan={
            "full_scope": True,
            "plan_sha256": "7" * 64,
            "config_sha256": "8" * 64,
            "tool_sha256": "9" * 64,
            "batch_size": 2,
            "workers": 1,
        },
        tasks=[],
        metrics=[],
        output_dir=tmp_path / "output",
        membership=np.ones(4, dtype=np.uint8),
        started_at="2026-07-17T00:00:00+00:00",
        elapsed_seconds=1.0,
        parent_stable=True,
    )


def test_full_pass_boundary_accepts_only_fully_conserved_author_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _aggregate_full_pass_report(
        monkeypatch, tmp_path, _full_pass_author_profile()
    )

    assert report["status"] == "PASS_SYNTHETIC_COMPONENT"
    assert report["gate_status"]["work_source_membership_component"] == "PASS"


def test_blocking_ledger_domain_failure_is_attributed_to_target_component(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        audit,
        "validate_blocking_identifier_ledger",
        lambda *_args, **_kwargs: {
            "pass": False,
            "domain_passes": {
                "source": True,
                "target": False,
                "author": True,
                "main_field": True,
            },
        },
    )

    report = _aggregate_full_pass_report(
        monkeypatch, tmp_path, _full_pass_author_profile()
    )

    assert report["gate_status"]["source_row_conservation"] == "PASS"
    assert report["gate_status"]["child_source_membership"] == "PASS"
    assert report["gate_status"]["declared_work_target_scan"] == "NOT_COMPLETE"
    assert (
        report["gate_status"]["authorship_author_id_work_profile"]
        == "COMPLETE_PROFILE_MEASURED"
    )
    assert report["gate_status"]["work_profile_strata"] == "COMPLETE_PROFILE_MEASURED"
    assert report["gate_status"]["work_source_membership_component"] == "NOT_PASS"


def _set_nested(profile: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = profile
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("master_work_rows_profiled",), 3),
        (("retained_authorship_rows",), 4),
        (("grammar_usable_author_id_status", "complete"), 2),
        (("raw_null_only_status", "complete"), 2),
        (("declared_authors_count", "matches_observed_slots"), 3),
        (("by_publication_year",), []),
        (("by_primary_field",), []),
        (("by_observed_team_size",), []),
        (("by_authorship_source_lineage_mask",), []),
        (("joint_sparse_cube",), []),
    ],
    ids=(
        "master-row-count",
        "retained-authorship-rows",
        "grammar-status-partition",
        "raw-null-status-partition",
        "declared-author-partition",
        "year-margin",
        "field-margin",
        "team-margin",
        "lineage-margin",
        "joint-cube",
    ),
)
def test_full_pass_boundary_rejects_incomplete_or_nonconserved_author_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    path: tuple[str, ...],
    value: Any,
) -> None:
    profile = _full_pass_author_profile()
    _set_nested(profile, path, value)

    report = _aggregate_full_pass_report(monkeypatch, tmp_path, profile)

    assert report["status"] == "BLOCKED"
    assert report["gate_status"]["work_source_membership_component"] == "NOT_PASS"


OUTPUT_MANIFEST_NAME = "work_source_membership_audit.outputs.json"
OUTPUT_MANIFEST_SCHEMA = "openalex_g1_work_source_membership_outputs.v1"


def _synthetic_main(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "artifacts").mkdir()
    (output_dir / "audit_plan.json").write_text("{}\n", encoding="utf-8")
    report = {
        "schema_version": audit.REPORT_SCHEMA_VERSION,
        "status": "PASS_SYNTHETIC",
        "gate_status": {
            "g_minus_1_data_readiness": "PENDING",
            "g1_empirical_signature": "NOT_RUN",
        },
        "provenance": {"result_fingerprint_sha256": "8" * 64},
        "output_evidence_expectation": {
            "checkpoint_files": 0,
            "checkpoint_path_set_sha256": hashlib.sha256(b"").hexdigest(),
            "data_artifact_files": 0,
            "data_artifact_path_set_sha256": hashlib.sha256(b"").hexdigest(),
        },
    }
    args = SimpleNamespace(
        config=CONFIG_PATH,
        inventory_manifest=tmp_path / "manifest.json",
        root=root,
        master_ids=tmp_path / "master.u64",
        output_dir=output_dir,
        workers=None,
        batch_size=None,
        max_files_per_table=None,
        tables=None,
        require_pass=False,
    )
    monkeypatch.setattr(audit, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(audit, "load_config", lambda _path: {"synthetic": True})
    monkeypatch.setattr(audit, "resolve_root", lambda _config, _explicit: root)
    monkeypatch.setattr(
        audit, "validate_output_location", lambda _root, _output: output_dir
    )
    monkeypatch.setattr(audit, "audit", lambda **_kwargs: report)
    monkeypatch.setattr(audit, "render_markdown", lambda _report: "# Synthetic\n")
    return output_dir, report


def _assert_output_manifest(output_dir: Path, report: dict[str, Any]) -> Path:
    manifest_path = output_dir / OUTPUT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == OUTPUT_MANIFEST_SCHEMA
    assert (
        manifest["result_fingerprint_sha256"]
        == report["provenance"]["result_fingerprint_sha256"]
    )
    expected = {
        "report_json": "work_source_membership_audit.json",
        "report_markdown": "work_source_membership_audit.md",
    }
    assert set(manifest["artifacts"]) == set(expected)
    for kind, name in expected.items():
        artifact = output_dir / name
        record = manifest["artifacts"][kind]
        assert record["relative_path"] == name
        assert record["bytes"] == artifact.stat().st_size
        assert record["sha256"] == _sha256(artifact)
    return manifest_path


def test_final_output_manifest_commits_json_markdown_and_clean_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir, report = _synthetic_main(monkeypatch, tmp_path)

    assert audit.main([]) == 0
    _assert_output_manifest(output_dir, report)
    assert audit.main([]) == 0


def _replace_same_size(path: Path, old: str, new: str) -> None:
    assert len(old) == len(new)
    content = path.read_text(encoding="utf-8")
    assert old in content
    path.write_text(content.replace(old, new, 1), encoding="utf-8")


@pytest.mark.parametrize("tamper", ["json", "markdown", "manifest"])
def test_final_output_resume_rejects_same_size_tamper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    output_dir, report = _synthetic_main(monkeypatch, tmp_path)
    assert audit.main([]) == 0
    manifest_path = _assert_output_manifest(output_dir, report)

    if tamper == "json":
        _replace_same_size(
            output_dir / "work_source_membership_audit.json",
            "PASS_SYNTHETIC",
            "FAIL_SYNTHETIC",
        )
    elif tamper == "markdown":
        _replace_same_size(
            output_dir / "work_source_membership_audit.md", "# Synthetic", "! Synthetic"
        )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        original = manifest["artifacts"]["report_markdown"]["sha256"]
        replacement = ("0" if original[0] != "0" else "1") + original[1:]
        _replace_same_size(manifest_path, original, replacement)

    assert audit.main([]) == 2


@pytest.mark.parametrize(
    "mutation",
    ["extra_top", "extra_artifact_field", "string_bytes", "boolean_file_count"],
)
def test_final_output_resume_rejects_noncanonical_manifest_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    output_dir, report = _synthetic_main(monkeypatch, tmp_path)
    assert audit.main([]) == 0
    manifest_path = _assert_output_manifest(output_dir, report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "extra_top":
        manifest["unexpected"] = True
    elif mutation == "extra_artifact_field":
        manifest["artifacts"]["report_json"]["unexpected"] = True
    elif mutation == "string_bytes":
        manifest["artifacts"]["report_json"]["bytes"] = str(
            manifest["artifacts"]["report_json"]["bytes"]
        )
    else:
        manifest["checkpoint_tree"]["files"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert audit.main([]) == 2


@pytest.mark.parametrize("tree_name", ["checkpoints", "artifacts"])
def test_final_output_resume_rejects_empty_nested_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tree_name: str
) -> None:
    output_dir, _ = _synthetic_main(monkeypatch, tmp_path)
    assert audit.main([]) == 0
    (output_dir / tree_name / "unexpected" / "empty").mkdir(parents=True)

    assert audit.main([]) == 2


def test_output_tree_commitment_fails_closed_on_unreadable_directory(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "checkpoints"
    blocked = tree / "blocked"
    blocked.mkdir(parents=True)
    blocked.chmod(0)
    try:
        with pytest.raises(audit.MembershipAuditError, match="cannot traverse"):
            audit.output_tree_commitment(tmp_path, "checkpoints")
    finally:
        blocked.chmod(0o700)


def test_output_lock_symlink_is_never_followed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir, _ = _synthetic_main(monkeypatch, tmp_path)
    assert audit.main([]) == 0
    external = tmp_path / "external-lock-target"
    external.write_text("must remain unchanged", encoding="utf-8")
    lock_path = output_dir / "audit.lock"
    lock_path.unlink()
    lock_path.symlink_to(external)

    assert audit.main([]) == 2
    assert external.read_text(encoding="utf-8") == "must remain unchanged"


def test_final_output_resume_rejects_wrong_report_schema_even_with_updated_sha(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir, report = _synthetic_main(monkeypatch, tmp_path)
    assert audit.main([]) == 0
    manifest_path = _assert_output_manifest(output_dir, report)
    report_path = output_dir / "work_source_membership_audit.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["schema_version"] = "wrong-report-schema"
    report_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["report_json"] = audit._final_artifact_record(report_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert audit.main([]) == 2
