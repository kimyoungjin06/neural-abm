from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_source_inventory.py"
CONFIG_PATH = REPO_ROOT / "experiments/configs/openalex_g1_source_inventory_v0.yaml"
SUMMARY_PATH = (
    REPO_ROOT
    / "experiments/evidence/results/openalex_g1_source_inventory_20260716.summary.json"
)
FILENAME_PATTERN = (
    r"^(?P<prefix>part|base|delta|repair)-"
    r"(?P<index>[0-9]+)[.]parquet$"
)


def _load_inventory_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_source_inventory_module",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inventory = _load_inventory_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(autouse=True)
def _stable_git_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inventory,
        "git_metadata",
        lambda: {"commit": "synthetic-test", "dirty": False, "status_error": None},
    )


def _write_parquet(
    path: Path,
    columns: list[tuple[str, pa.Array]],
    *,
    row_group_size: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_arrays(
        [array for _name, array in columns],
        names=[name for name, _array in columns],
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _base_table_spec(
    *,
    role: str = "main_works",
    expected_file_count: int = 1,
    expected_footer_rows: int = 2,
) -> dict[str, Any]:
    return {
        "role": role,
        "expected_file_count": expected_file_count,
        "expected_footer_rows": expected_footer_rows,
        "required_columns": ["id", "value"],
        "required_column_types": {"id": "string", "value": "int64"},
        "expected_prefix_contract": {
            "part": {"count": expected_file_count, "min": 0, "max": 0, "gaps": []}
        },
    }


def _config(
    root: Path,
    tables: dict[str, dict[str, Any]],
    *,
    content_sha256_required: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": "openalex_g1_source_inventory_config.v1",
        "label": "synthetic_openalex_inventory",
        "root": {
            "env": "SYNTHETIC_OPENALEX_ROOT",
            "id": root.name,
            "snapshot_date": "2026-03-30",
            "lineage_id": "synthetic-lineage",
            "coverage": "synthetic",
            "historical_reference_id": "synthetic-reference",
        },
        "inventory": {
            "expected_table_count": len(tables),
            "allowed_filename_pattern": FILENAME_PATTERN,
            "reject_unexpected_table_directories": True,
            "reject_unexpected_table_entries": True,
            "content_sha256_required_for_pass_metadata": content_sha256_required,
        },
        "feature_roles": {},
        "tables": tables,
        "pending_full_scan_gates": ["synthetic_exact_recount"],
    }


def _audit(
    tmp_path: Path,
    root: Path,
    tables: dict[str, dict[str, Any]],
    *,
    content_sha256: bool = False,
    content_sha256_required: bool = True,
) -> dict[str, Any]:
    config = _config(
        root,
        tables,
        content_sha256_required=content_sha256_required,
    )
    config_path = tmp_path / "synthetic_inventory.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return inventory.audit_source(
        root=root,
        config=config,
        config_path=config_path,
        workers=1,
        content_sha256=content_sha256,
    )


def _table(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    return next(table for table in manifest["tables"] if table["name"] == name)


def _check(container: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in container["checks"] if item["name"] == name)


def _make_normal_root(tmp_path: Path) -> tuple[Path, str, dict[str, Any]]:
    root = tmp_path / "synthetic_snapshot"
    table_name = "works"
    _write_parquet(
        root / table_name / "part-0.parquet",
        [
            ("id", pa.array(["W1", "W2"], type=pa.string())),
            ("value", pa.array([1, 2], type=pa.int64())),
        ],
    )
    return root, table_name, _base_table_spec()


def test_normal_audit_is_inventory_only_and_redacts_source_path(
    tmp_path: Path,
) -> None:
    root, table_name, table_spec = _make_normal_root(tmp_path)

    manifest = _audit(tmp_path, root, {table_name: table_spec})

    assert manifest["status"] == "INVENTORY_ONLY"
    assert manifest["pass_metadata"] is False
    assert manifest["pass_g1"] is False
    assert manifest["execution"]["path_redacted"] is True
    assert manifest["summary"]["block_count"] == 0
    assert str(root) not in json.dumps(manifest, sort_keys=True)
    assert manifest["tables"][0]["files"][0]["relative_path"] == (
        "works/part-0.parquet"
    )


def test_tracked_summary_is_bound_to_the_tool_and_source_contract() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert summary["status"] == "INVENTORY_ONLY"
    assert summary["pass_metadata"] is False
    assert summary["pass_g1"] is False
    assert summary["checks"] == {
        "blocks": 0,
        "warnings": 3,
        "pending_metadata_checks": 23,
    }
    assert summary["tool_sha256"] == _sha256(SCRIPT_PATH)
    assert summary["config_sha256"] == _sha256(CONFIG_PATH)


def test_content_hashing_promotes_clean_audit_to_pass_metadata(
    tmp_path: Path,
) -> None:
    root, table_name, table_spec = _make_normal_root(tmp_path)

    manifest = _audit(
        tmp_path,
        root,
        {table_name: table_spec},
        content_sha256=True,
    )

    assert manifest["status"] == "PASS_METADATA"
    assert manifest["pass_metadata"] is True
    assert manifest["pass_g1"] is False
    table = _table(manifest, table_name)
    assert _check(table, "content_sha256")["status"] == "PASS"
    assert len(table["files"][0]["content_sha256"]) == 64


def test_frozen_path_and_size_inventory_fingerprint_mismatch_blocks(
    tmp_path: Path,
) -> None:
    root, table_name, table_spec = _make_normal_root(tmp_path)
    config = _config(root, {table_name: table_spec})
    config["inventory"]["expected_inventory_fingerprint_sha256"] = "0" * 64
    config_path = tmp_path / "mismatched_inventory.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    manifest = inventory.audit_source(
        root=root,
        config=config,
        config_path=config_path,
        workers=1,
        content_sha256=False,
    )

    assert manifest["status"] == "BLOCKED"
    frozen_check = _check(manifest, "frozen_inventory_fingerprint")
    assert frozen_check["status"] == "BLOCK"
    assert frozen_check["observed"] != frozen_check["expected"]


def test_footer_row_and_required_type_mismatches_block(
    tmp_path: Path,
) -> None:
    root, table_name, table_spec = _make_normal_root(tmp_path)
    table_spec["expected_footer_rows"] = 3
    table_spec["required_column_types"]["value"] = "double"

    manifest = _audit(tmp_path, root, {table_name: table_spec})

    assert manifest["status"] == "BLOCKED"
    table = _table(manifest, table_name)
    assert _check(table, "historical_footer_rows")["status"] == "BLOCK"
    type_check = _check(table, "required_column_types")
    assert type_check["status"] == "BLOCK"
    assert next(iter(type_check["observed"].values()))["value"] == {
        "observed": "int64",
        "expected": "double",
    }


def test_column_order_drift_is_canonical_pass_and_physical_warning(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ordered_snapshot"
    table_name = "authorships"
    table_dir = root / table_name
    ids = pa.array(["W1", "W2"], type=pa.string())
    authors = pa.array(["A1", "A2"], type=pa.string())
    _write_parquet(table_dir / "base-0.parquet", [("id", ids), ("author_id", authors)])
    _write_parquet(
        table_dir / "delta-0.parquet",
        [("author_id", authors), ("id", ids)],
    )
    table_spec = {
        "role": "works_authorships",
        "expected_file_count": 2,
        "expected_footer_rows": 4,
        "allowed_prefixes": ["base", "delta"],
        "expected_prefix_contract": {
            "base": {"count": 1, "min": 0, "max": 0, "gaps": []},
            "delta": {"count": 1, "min": 0, "max": 0, "gaps": []},
        },
        "required_columns": ["id", "author_id"],
        "required_column_types": {"id": "string", "author_id": "string"},
    }

    manifest = _audit(tmp_path, root, {table_name: table_spec})

    assert manifest["status"] == "INVENTORY_ONLY"
    table = _table(manifest, table_name)
    assert _check(table, "single_canonical_schema_fingerprint")["status"] == "PASS"
    physical = _check(table, "physical_schema_order_variants")
    assert physical["status"] == "WARN"
    assert len(table["canonical_schema_fingerprints"]) == 1
    assert len(table["physical_schema_fingerprints"]) == 2


@pytest.mark.parametrize(
    ("filenames", "expected_contract", "blocked_check"),
    [
        (
            ["base-0.parquet"],
            {"part": {"count": 1, "min": 0, "max": 0, "gaps": []}},
            "allowed_filename_prefixes",
        ),
        (
            ["part-0.parquet", "part-2.parquet"],
            {"part": {"count": 2, "min": 0, "max": 2, "gaps": []}},
            "filename_sequence_gaps",
        ),
    ],
)
def test_sequence_or_prefix_contract_mismatch_blocks(
    tmp_path: Path,
    filenames: list[str],
    expected_contract: dict[str, dict[str, Any]],
    blocked_check: str,
) -> None:
    root = tmp_path / f"contract_snapshot_{blocked_check}"
    table_name = "works"
    for index, filename in enumerate(filenames):
        _write_parquet(
            root / table_name / filename,
            [
                ("id", pa.array([f"W{index}"], type=pa.string())),
                ("value", pa.array([index], type=pa.int64())),
            ],
            row_group_size=1,
        )
    table_spec = _base_table_spec(
        expected_file_count=len(filenames),
        expected_footer_rows=len(filenames),
    )
    table_spec["expected_prefix_contract"] = expected_contract

    manifest = _audit(tmp_path, root, {table_name: table_spec})

    assert manifest["status"] == "BLOCKED"
    table = _table(manifest, table_name)
    assert _check(table, blocked_check)["status"] == "BLOCK"
    assert _check(table, "filename_prefix_contract")["status"] == "BLOCK"


def test_author_id_footer_nulls_are_aggregated_exactly_and_warn(
    tmp_path: Path,
) -> None:
    root = tmp_path / "author_null_snapshot"
    table_name = "authorships"
    table_dir = root / table_name
    _write_parquet(
        table_dir / "base-0.parquet",
        [
            ("id", pa.array(["W1", "W2"], type=pa.string())),
            ("author_id", pa.array(["A1", None], type=pa.string())),
        ],
        row_group_size=1,
    )
    _write_parquet(
        table_dir / "delta-0.parquet",
        [
            ("id", pa.array(["W3", "W4"], type=pa.string())),
            ("author_id", pa.array([None, "A4"], type=pa.string())),
        ],
        row_group_size=1,
    )
    table_spec = {
        "role": "works_authorships",
        "expected_file_count": 2,
        "expected_footer_rows": 4,
        "allowed_prefixes": ["base", "delta"],
        "expected_prefix_contract": {
            "base": {"count": 1, "min": 0, "max": 0, "gaps": []},
            "delta": {"count": 1, "min": 0, "max": 0, "gaps": []},
        },
        "required_columns": ["id", "author_id"],
        "required_column_types": {"id": "string", "author_id": "string"},
        "footer_null_checks": {
            "id": {"expected_null_count": 0, "nonzero_status": "BLOCK"},
            "author_id": {
                "expected_null_count": 2,
                "mismatch_status": "BLOCK",
                "nonzero_status": "WARN",
            },
        },
    }

    manifest = _audit(tmp_path, root, {table_name: table_spec})

    assert manifest["status"] == "INVENTORY_ONLY"
    table = _table(manifest, table_name)
    assert _check(table, "footer_null_count:id")["status"] == "PASS"
    author_check = _check(table, "footer_null_count:author_id")
    assert author_check["status"] == "WARN"
    assert table["footer_null_statistics"]["author_id"] == {
        "null_count": 2,
        "footer_rows": 4,
        "null_fraction": 0.5,
        "statistics_complete": True,
    }
    per_file_nulls = [
        record["footer_null_statistics"]["author_id"]["null_count"]
        for record in table["files"]
    ]
    assert per_file_nulls == [1, 1]


def test_symlink_and_unexpected_table_entry_block(
    tmp_path: Path,
) -> None:
    root, table_name, table_spec = _make_normal_root(tmp_path)
    table_dir = root / table_name
    external = tmp_path / "external.parquet"
    _write_parquet(
        external,
        [
            ("id", pa.array(["W9"], type=pa.string())),
            ("value", pa.array([9], type=pa.int64())),
        ],
        row_group_size=1,
    )
    (table_dir / "part-1.parquet").symlink_to(external)
    (table_dir / "notes.txt").write_text("unexpected", encoding="utf-8")

    manifest = _audit(tmp_path, root, {table_name: table_spec})

    assert manifest["status"] == "BLOCKED"
    table = _table(manifest, table_name)
    unexpected = _check(table, "unexpected_table_entries")
    assert unexpected["status"] == "BLOCK"
    assert unexpected["observed"] == ["notes.txt", "part-1.parquet"]
    assert table["file_count"] == 1
    assert [record["relative_path"] for record in table["files"]] == [
        "works/part-0.parquet"
    ]
