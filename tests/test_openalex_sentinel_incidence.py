from __future__ import annotations

import hashlib
import importlib.util
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
SCRIPT_PATH = REPO_ROOT / "scripts/audit_openalex_sentinel_incidence.py"
W_MINUS_1 = "https://openalex.org/W-1"
W_ZERO = "https://openalex.org/W0"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "test_openalex_sentinel_incidence_module", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_module()


@pytest.fixture(autouse=True)
def _stable_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        audit,
        "git_metadata",
        lambda: {"commit": "synthetic", "dirty": False, "status_error": None},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_table(path: Path, **columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                name: pa.array(values, type=pa.string())
                for name, values in columns.items()
            }
        ),
        path,
        row_group_size=2,
    )


def _snapshot(root: Path) -> list[Any]:
    rows = []
    for path in sorted(root.glob("*/*.parquet")):
        file_stat = path.stat()
        rows.append(
            audit.FileSnapshot(
                relative_path=path.relative_to(root).as_posix(),
                bytes=file_stat.st_size,
                mtime_ns=file_stat.st_mtime_ns,
            )
        )
    return rows


def _prepare_contract(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    root = tmp_path / "synthetic_openalex"
    _write_table(
        root / "main" / "part-0.parquet",
        id=["https://openalex.org/W1", W_MINUS_1, W_ZERO],
    )
    _write_table(
        root / "references" / "base-0.parquet",
        id=[W_MINUS_1, W_MINUS_1, "https://openalex.org/W2", W_ZERO],
        referenced_works=[W_ZERO, W_MINUS_1, W_MINUS_1, "https://openalex.org/W3"],
    )
    _write_table(
        root / "references" / "delta-0.parquet",
        id=[W_ZERO, "https://openalex.org/W4"],
        referenced_works=[W_ZERO, "https://openalex.org/W5"],
    )
    _write_table(
        root / "related" / "base-0.parquet",
        id=["https://openalex.org/W6", W_MINUS_1],
        related_works=[W_MINUS_1, W_ZERO],
    )
    snapshots = _snapshot(root)
    inventory_sha = audit.inventory_fingerprint(snapshots)
    stability_sha = audit.stability_fingerprint(snapshots)

    tables = []
    for table_name in ("main", "references", "related"):
        records = []
        for path in sorted((root / table_name).glob("*.parquet")):
            file_stat = path.stat()
            parquet = pq.ParquetFile(path)
            records.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "bytes": file_stat.st_size,
                    "mtime_ns": file_stat.st_mtime_ns,
                    "footer_rows": parquet.metadata.num_rows,
                    "columns": parquet.schema_arrow.names,
                    "error": None,
                }
            )
        tables.append(
            {
                "name": table_name,
                "file_count": len(records),
                "files": records,
            }
        )
    manifest = {
        "schema_version": "openalex_source_inventory.v1",
        "manifest_id": "synthetic_inventory",
        "summary": {"block_count": 0},
        "root": {
            "id": root.name,
            "table_count": 3,
            "parquet_file_count": len(snapshots),
            "inventory_fingerprint_sha256": inventory_sha,
            "stability_fingerprint_sha256": stability_sha,
        },
        "tables": tables,
    }
    manifest_path = tmp_path / "parent.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = {
        "schema_version": "openalex_sentinel_incidence_config.v1",
        "label": "synthetic_sentinel_incidence",
        "parent_inventory": {
            "schema_version": "openalex_source_inventory.v1",
            "manifest_id": "synthetic_inventory",
            "manifest_sha256": _sha256(manifest_path),
            "source_root_env": "SYNTHETIC_OPENALEX_ROOT",
            "source_root_id": root.name,
            "expected_table_count": 3,
            "inventory_fingerprint_sha256": inventory_sha,
            "stability_fingerprint_sha256": stability_sha,
        },
        "targeted_incidence": {
            "source_id_column": "id",
            "sentinels": {
                "w_minus_1": {
                    "value": W_MINUS_1,
                    "disposition": "quarantine_candidate",
                },
                "w_zero": {
                    "value": W_ZERO,
                    "disposition": "unresolved_diagnostic",
                },
            },
            "work_target_columns": {
                "references": "referenced_works",
                "related": "related_works",
            },
            "batch_size": 2,
            "workers": 1,
            "predicate_pushdown_required": True,
        },
        "claim_boundary": "targeted incidence only; no row conservation",
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return root, manifest_path, config_path, audit.load_config(config_path)


def _run(tmp_path: Path) -> tuple[dict[str, Any], Path, Path, Path]:
    root, manifest_path, config_path, config = _prepare_contract(tmp_path)
    report = audit.audit_sentinel_incidence(
        root=root,
        config=config,
        config_path=config_path,
        manifest_path=manifest_path,
        workers=1,
    )
    return report, root, manifest_path, config_path


def _table(scope: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in scope["tables"] if item["table"] == name)


def test_exact_source_and_target_counts_are_separate_and_path_redacted(
    tmp_path: Path,
) -> None:
    report, root, manifest_path, config_path = _run(tmp_path)

    assert report["status"] == "COMPLETE_TARGETED_INCIDENCE_AUDIT"
    assert report["audit_complete"] is True
    assert report["pass_g1"] is False
    assert report["analysis_view_quarantine_applied"] is False
    assert report["scope"] == {
        "kind": "targeted_incidence_only",
        "row_conservation_claimed": False,
        "all_non_sentinel_identifiers_validated": False,
    }
    assert report["sentinels"]["w_minus_1"]["disposition"] == ("quarantine_candidate")
    assert report["sentinels"]["w_zero"]["disposition"] == ("unresolved_diagnostic")

    source = report["incidence"]["source_id"]
    assert source["table_count"] == 3
    assert source["files_evaluated"] == 4
    assert source["totals"]["w_minus_1"]["exact_row_count"] == 4
    assert source["totals"]["w_zero"]["exact_row_count"] == 3
    references = _table(source, "references")
    assert references["sentinels"]["w_minus_1"] == {
        "value": W_MINUS_1,
        "exact_row_count": 2,
        "affected_file_count": 1,
        "affected_files": [
            {
                "relative_path": "references/base-0.parquet",
                "exact_row_count": 2,
            }
        ],
    }

    targets = report["incidence"]["work_targets"]
    assert targets["table_column_count"] == 2
    assert targets["files_evaluated"] == 3
    assert targets["totals"]["w_minus_1"]["exact_row_count"] == 3
    assert targets["totals"]["w_zero"]["exact_row_count"] == 3
    assert (
        _table(targets, "references")["sentinels"]["w_zero"]["affected_file_count"] == 2
    )

    encoded = json.dumps(report, sort_keys=True)
    markdown = audit.render_markdown(report)
    assert str(root) not in encoded
    assert str(root) not in markdown
    assert report["provenance"]["path_redacted"] is True
    assert report["provenance"]["parent_manifest_sha256"] == _sha256(manifest_path)
    assert report["provenance"]["config_sha256"] == _sha256(config_path)
    assert report["provenance"]["tool_sha256"] == _sha256(SCRIPT_PATH)


def test_scanner_receives_isin_filter_and_only_target_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _manifest_path, _config_path, _config = _prepare_contract(tmp_path)
    captured: dict[str, Any] = {}

    class FakeScanner:
        def to_batches(self) -> list[pa.RecordBatch]:
            return [pa.record_batch([pa.array([W_MINUS_1])], names=["id"])]

    class FakeDataset:
        def scanner(self, **kwargs: Any) -> FakeScanner:
            captured.update(kwargs)
            return FakeScanner()

    monkeypatch.setattr(audit.ds, "dataset", lambda *_args, **_kwargs: FakeDataset())
    result = audit.scan_targeted_file(
        root=root,
        task=audit.ScanTask(
            scope="source_id",
            table="main",
            column="id",
            relative_path="main/part-0.parquet",
        ),
        sentinels={"w_minus_1": W_MINUS_1, "w_zero": W_ZERO},
        batch_size=17,
    )

    assert result.counts == {"w_minus_1": 1, "w_zero": 0}
    assert captured["columns"] == ["id"]
    assert captured["filter"] is not None
    assert "is_in" in str(captured["filter"])
    assert W_MINUS_1 in str(captured["filter"])
    assert W_ZERO in str(captured["filter"])
    assert captured["batch_size"] == 17


def test_starting_source_drift_is_fail_closed(tmp_path: Path) -> None:
    root, manifest_path, config_path, config = _prepare_contract(tmp_path)
    changed = root / "main/part-0.parquet"
    changed.write_bytes(changed.read_bytes() + b"drift")

    with pytest.raises(audit.SentinelAuditError, match="live path/size/mtime"):
        audit.audit_sentinel_incidence(
            root=root,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            workers=1,
        )


def test_end_source_drift_is_fail_closed_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, config_path, config = _prepare_contract(tmp_path)
    original = audit.execute_tasks

    def execute_then_drift(**kwargs: Any) -> list[Any]:
        results = original(**kwargs)
        path = root / "main/part-0.parquet"
        path.touch()
        return results

    monkeypatch.setattr(audit, "execute_tasks", execute_then_drift)
    output_json = tmp_path / "result.json"
    output_md = tmp_path / "result.md"

    with pytest.raises(audit.SentinelAuditError, match="live path/size/mtime"):
        report = audit.audit_sentinel_incidence(
            root=root,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            workers=1,
        )
        audit.write_output_pair(
            json_path=output_json, markdown_path=output_md, report=report
        )
    assert not output_json.exists()
    assert not output_md.exists()


def test_scan_error_publishes_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, config_path, config = _prepare_contract(tmp_path)
    output_json = tmp_path / "result.json"
    output_md = tmp_path / "result.md"

    def fail_scan(**_kwargs: Any) -> Any:
        raise OSError("synthetic read failure")

    monkeypatch.setattr(audit, "scan_targeted_file", fail_scan)
    with pytest.raises(audit.SentinelAuditError, match="synthetic read failure"):
        report = audit.audit_sentinel_incidence(
            root=root,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            workers=1,
        )
        audit.write_output_pair(
            json_path=output_json, markdown_path=output_md, report=report
        )
    assert not output_json.exists()
    assert not output_md.exists()


def test_parent_manifest_sha_and_required_target_column_are_fail_closed(
    tmp_path: Path,
) -> None:
    root, manifest_path, config_path, config = _prepare_contract(tmp_path)
    config["parent_inventory"]["manifest_sha256"] = "0" * 64
    with pytest.raises(audit.SentinelAuditError, match="manifest SHA-256"):
        audit.audit_sentinel_incidence(
            root=root,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            workers=1,
        )

    root, manifest_path, config_path, config = _prepare_contract(
        tmp_path / "missing-column"
    )
    config["targeted_incidence"]["work_target_columns"]["references"] = "missing"
    with pytest.raises(audit.SentinelAuditError, match="required targeted columns"):
        audit.audit_sentinel_incidence(
            root=root,
            config=config,
            config_path=config_path,
            manifest_path=manifest_path,
            workers=1,
        )


def test_cli_writes_atomic_pair_and_refuses_existing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, manifest_path, config_path, _config = _prepare_contract(tmp_path)
    output_json = tmp_path / "outputs/result.json"
    output_md = tmp_path / "outputs/result.md"
    monkeypatch.setenv("SYNTHETIC_OPENALEX_ROOT", str(root))

    exit_code = audit.main(
        [
            "--config",
            str(config_path),
            "--inventory-manifest",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--workers",
            "1",
        ]
    )
    assert exit_code == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["pass_g1"] is False
    assert output_md.read_text(encoding="utf-8").startswith(
        "# OpenAlex Sentinel Targeted-Incidence Audit"
    )

    second_exit = audit.main(
        [
            "--config",
            str(config_path),
            "--inventory-manifest",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )
    assert second_exit == 2
