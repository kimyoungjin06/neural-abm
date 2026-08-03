from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_SCRIPT = REPO_ROOT / "scripts/audit_openalex_source_inventory.py"
RECOUNT_SCRIPT = REPO_ROOT / "scripts/recount_openalex_g1_canonical_quarantine.py"
BASE_SCRIPT = REPO_ROOT / "scripts/recount_openalex_g1.py"
BLOCKED_EVIDENCE = (
    REPO_ROOT / "experiments/evidence/results/"
    "openalex_g1_main_id_recount_20260716.blocked.summary.json"
)
PREFIX = "https://openalex.org/W"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inventory = _load_module("test_quarantine_inventory_module", INVENTORY_SCRIPT)
recount = _load_module("test_grammar_quarantine_recount_module", RECOUNT_SCRIPT)


@pytest.fixture(autouse=True)
def _stable_git_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    stable = lambda: {  # noqa: E731
        "commit": "synthetic",
        "dirty": False,
        "status_error": None,
    }
    monkeypatch.setattr(inventory, "git_metadata", stable)
    monkeypatch.setattr(recount.base, "git_metadata", stable)


def _write_ids(path: Path, ids: list[str | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids, type=pa.string()),
                "publication_year": pa.array([2020] * len(ids), type=pa.int64()),
            }
        ),
        path,
        row_group_size=2,
    )


def _inventory_config(root: Path, *, files: int, rows: int) -> dict[str, Any]:
    return {
        "schema_version": "openalex_g1_source_inventory_config.v1",
        "label": "synthetic_inventory",
        "root": {
            "env": "SYNTHETIC_ROOT",
            "id": root.name,
            "snapshot_date": "2026-03-30",
            "lineage_id": "synthetic",
            "coverage": "synthetic",
            "historical_reference_id": "synthetic",
        },
        "inventory": {
            "expected_table_count": 1,
            "allowed_filename_pattern": (
                r"^(?P<prefix>part|base|delta|repair)-"
                r"(?P<index>[0-9]+)[.]parquet$"
            ),
            "reject_unexpected_table_directories": True,
            "reject_unexpected_table_entries": True,
            "content_sha256_required_for_pass_metadata": False,
        },
        "tables": {
            "works": {
                "role": "main_works",
                "expected_file_count": files,
                "expected_footer_rows": rows,
                "required_columns": ["id", "publication_year"],
                "required_column_types": {"id": "string"},
            }
        },
        "pending_full_scan_gates": [],
    }


def _prepare_parent(
    tmp_path: Path, files: list[list[str | None]]
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "synthetic_snapshot"
    for index, ids in enumerate(files):
        _write_ids(root / "works" / f"part-{index}.parquet", ids)
    row_count = sum(len(ids) for ids in files)
    config = _inventory_config(root, files=len(files), rows=row_count)
    config_path = tmp_path / "inventory.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = inventory.audit_source(
        root=root,
        config=config,
        config_path=config_path,
        workers=1,
        content_sha256=False,
    )
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path, manifest


def _location(
    raw_value: str, relative_path: str, row_group: int, file_row: int
) -> dict[str, Any]:
    return {
        "raw_value": raw_value,
        "expected_count": 1,
        "expected_relative_path": relative_path,
        "expected_row_group": row_group,
        "expected_file_row": file_row,
    }


def _recount_config(
    tmp_path: Path,
    root: Path,
    manifest: dict[str, Any],
    *,
    expected_valid_rows: int,
    expected_distinct: int,
    expected_duplicate_excess: int,
    quarantine: dict[str, Any],
    zero: dict[str, Any],
) -> Path:
    blocked = json.loads(BLOCKED_EVIDENCE.read_text(encoding="utf-8"))
    config = {
        "schema_version": "openalex_g1_grammar_quarantine_recount_config.v1",
        "label": "synthetic_grammar_quarantine_recount",
        "parent_inventory": {
            "schema_version": manifest["schema_version"],
            "manifest_id": manifest["manifest_id"],
            "source_root_env": "SYNTHETIC_ROOT",
            "source_root_id": root.name,
            "inventory_fingerprint_sha256": manifest["root"][
                "inventory_fingerprint_sha256"
            ],
        },
        "parent_evidence": {
            "blocked_zero_invalid_recount": {
                "relative_path": BLOCKED_EVIDENCE.relative_to(REPO_ROOT).as_posix(),
                "sha256": recount.sha256_file(BLOCKED_EVIDENCE),
                "raw_report_sha256": blocked["provenance"]["raw_report_sha256"],
                "status": "BLOCKED",
            }
        },
        "tool_dependency": {
            "relative_path": BASE_SCRIPT.relative_to(REPO_ROOT).as_posix(),
            "sha256": recount.sha256_file(BASE_SCRIPT),
        },
        "main_id_recount": {
            "table": "works",
            "column": "id",
            "grammar_prefix": PREFIX,
            "expected_files": manifest["root"]["parquet_file_count"],
            "expected_body_rows": manifest["root"]["footer_rows"],
            "expected_null_rows": 0,
            "expected_grammar_valid_rows": expected_valid_rows,
            "expected_grammar_valid_distinct_ids": expected_distinct,
            "expected_grammar_valid_duplicate_row_excess": (expected_duplicate_excess),
            "spool_dtype": "uint64",
            "batch_size": 1,
            "duplicate_scan_chunk_rows": 2,
            "invalid_sample_limit": 5,
        },
        "quarantine": {
            "policy": "exact_raw_value_and_location_allowlist.v1",
            "require_no_unlisted_invalid": True,
            "expected_rows": 1,
            "entries": [quarantine],
        },
        "zero_id_diagnostic": {
            "warning_code": "WARN_ZERO_ID_UNRESOLVED",
            "retained_in_grammar_valid_spool": True,
            "entity_validity": "UNRESOLVED",
            "api_404_is_exclusion_evidence": False,
            "entry": zero,
        },
        "claim_boundary": (
            "Synthetic grammar-valid recount only; W0 entity validity remains "
            "unresolved and pass_g1 is false."
        ),
        "pending_after_grammar_valid_main_id_recount": [
            "w0_entity_validity_and_exclusion_sensitivity"
        ],
    }
    path = tmp_path / "grammar_quarantine_recount.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def _run(
    *,
    config_path: Path,
    manifest_path: Path,
    root: Path,
    output_dir: Path,
    max_files: int | None = None,
    pause_after_new_files: int | None = None,
) -> dict[str, Any]:
    return recount.run_recount(
        config_path=config_path,
        parent_path=manifest_path,
        root=root,
        output_dir=output_dir,
        max_files=max_files,
        batch_size=1,
        pause_after_new_files=pause_after_new_files,
    )


def _happy_contract(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            [f"{PREFIX}0", f"{PREFIX}1"],
            [f"{PREFIX}2", f"{PREFIX}-1", f"{PREFIX}2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_valid_rows=4,
        expected_distinct=3,
        expected_duplicate_excess=1,
        quarantine=_location(f"{PREFIX}-1", "works/part-1.parquet", 0, 1),
        zero=_location(f"{PREFIX}0", "works/part-0.parquet", 0, 0),
    )
    return root, manifest_path, config_path, tmp_path / "output"


def _check(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["checks"] if item["name"] == name)


def test_compact_parser_retains_w0_without_a_numeric_malformed_sentinel() -> None:
    numeric, nulls, malformed, zero_indices = (
        recount.parse_grammar_valid_openalex_ids_compact(
            pa.array(
                [f"{PREFIX}0", f"{PREFIX}-1", f"{PREFIX}1", None],
                type=pa.string(),
            ),
            prefix=PREFIX,
        )
    )

    assert numeric.tolist() == [0, 1]
    assert nulls == 1
    assert malformed == [(1, f"{PREFIX}-1")]
    assert zero_indices == [0]


def test_full_recount_is_compact_exact_and_keeps_w0_as_warning(
    tmp_path: Path,
) -> None:
    root, manifest_path, config_path, output_dir = _happy_contract(tmp_path)

    report = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
    )

    assert (
        report["status"] == "PASS_GRAMMAR_VALID_MAIN_ID_RECOUNT_WITH_FROZEN_QUARANTINE"
    )
    assert report["pass_body_recount"] is True
    assert report["quarantine_contract_satisfied"] is True
    assert report["pass_grammar_valid_main_id_recount"] is True
    assert report["source_key_integrity_clean"] is False
    assert report["entity_validity_established"] is False
    assert report["parent_content_identity_verified"] is False
    assert report["g_minus_1_data_readiness"] == "PENDING"
    assert report["g1_empirical_signature"] == "NOT_RUN"
    assert report["pass_g1"] is False
    assert "WARN_PARENT_CONTENT_SHA256_PENDING" in {
        item["code"] for item in report["warnings"]
    }
    assert report["zero_id_diagnostic"]["warning_code"] == ("WARN_ZERO_ID_UNRESOLVED")
    assert report["zero_id_diagnostic"]["retained_in_grammar_valid_spool"] is True
    assert report["zero_id_diagnostic"]["api_404_is_exclusion_evidence"] is False
    assert report["counts"]["body_rows_scanned"] == 5
    assert report["counts"]["grammar_valid_rows"] == 4
    assert report["counts"]["quarantined_malformed_rows"] == 1
    assert report["counts"]["zero_id_rows_retained"] == 1
    assert report["counts"]["grammar_valid_distinct_ids"] == 3
    assert report["counts"]["grammar_valid_duplicate_row_excess"] == 1
    assert sum(report["diagnostic_buckets"]["histogram"]) == 4
    assert report["diagnostic_buckets"]["input_universe"] == (
        "grammar_valid_rows_only_including_w0"
    )

    raw = output_dir / "main_ids.grammar_valid.unsorted.u64"
    sorted_path = output_dir / "main_ids.grammar_valid.sorted.u64"
    assert raw.stat().st_size == 4 * 8
    assert np.fromfile(raw, dtype="<u8").tolist() == [0, 1, 2, 2]
    assert np.fromfile(sorted_path, dtype="<u8").tolist() == [0, 1, 2, 2]
    quarantine = json.loads(
        (output_dir / "malformed_quarantine.json").read_text(encoding="utf-8")
    )
    assert quarantine["numeric_sentinel_used"] is False
    assert quarantine["full_scope_contract_satisfied"] is True

    plan = json.loads((output_dir / "recount_plan.json").read_text())
    assert plan["tool_sha256"] == recount.sha256_file(RECOUNT_SCRIPT)
    assert plan["base_dependency"]["sha256"] == recount.sha256_file(BASE_SCRIPT)
    assert (
        plan["blocked_zero_invalid_parent_evidence"]["raw_report_sha256"]
        == "ceb318b378ba23a5418754f4e5c906d71f4c22ceb6e05c1f1c11cfd6f26e9483"
    )


@pytest.mark.parametrize("mismatch", ["quarantine_location", "w0_location"])
def test_exact_frozen_location_mismatch_blocks_without_adaptive_exclusion(
    tmp_path: Path, mismatch: str
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [[f"{PREFIX}0", f"{PREFIX}1", f"{PREFIX}-1"]],
    )
    quarantine_row = 1 if mismatch == "quarantine_location" else 2
    zero_row = 1 if mismatch == "w0_location" else 0
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_valid_rows=2,
        expected_distinct=2,
        expected_duplicate_excess=0,
        quarantine=_location(f"{PREFIX}-1", "works/part-0.parquet", 1, quarantine_row),
        zero=_location(f"{PREFIX}0", "works/part-0.parquet", 0, zero_row),
    )

    report = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=tmp_path / "mismatch_output",
    )

    assert report["status"] == "BLOCKED"
    assert report["pass_grammar_valid_main_id_recount"] is False
    if mismatch == "quarantine_location":
        assert report["counts"]["unexpected_malformed_rows"] == 1
        assert report["counts"]["quarantined_malformed_rows"] == 0
        assert (
            _check(report, "frozen_malformed_quarantine_exact_multiset")["status"]
            == "BLOCK"
        )
    else:
        assert report["counts"]["zero_id_rows_retained"] == 1
        assert (
            _check(report, "frozen_w0_diagnostic_exact_multiset")["status"] == "BLOCK"
        )


def test_engineering_pilot_leaves_full_quarantine_and_w0_checks_pending(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [[f"{PREFIX}1"], [f"{PREFIX}0", f"{PREFIX}-1"]],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_valid_rows=2,
        expected_distinct=2,
        expected_duplicate_excess=0,
        quarantine=_location(f"{PREFIX}-1", "works/part-1.parquet", 0, 1),
        zero=_location(f"{PREFIX}0", "works/part-1.parquet", 0, 0),
    )

    report = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=tmp_path / "pilot_output",
        max_files=1,
    )

    assert report["status"] == "PASS_ENGINEERING_PILOT"
    assert report["pass_body_recount"] is False
    assert report["quarantine_contract_satisfied"] is False
    assert report["pass_grammar_valid_main_id_recount"] is False
    assert report["pass_g1"] is False
    for name in (
        "frozen_malformed_quarantine_row_count",
        "frozen_malformed_quarantine_exact_multiset",
        "frozen_w0_diagnostic_row_count",
        "frozen_w0_diagnostic_exact_multiset",
    ):
        assert _check(report, name)["status"] == "PENDING"


def test_pause_resume_tracks_separate_body_and_grammar_valid_offsets(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            [f"{PREFIX}0", f"{PREFIX}-1", f"{PREFIX}1"],
            [f"{PREFIX}2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_valid_rows=3,
        expected_distinct=3,
        expected_duplicate_excess=0,
        quarantine=_location(f"{PREFIX}-1", "works/part-0.parquet", 0, 1),
        zero=_location(f"{PREFIX}0", "works/part-0.parquet", 0, 0),
    )
    output_dir = tmp_path / "resume_output"

    paused = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
        pause_after_new_files=1,
    )
    assert paused["status"] == "PAUSED"
    assert paused["body_rows_scanned"] == 3
    assert paused["grammar_valid_rows_written"] == 2
    state = json.loads((output_dir / "recount_state.json").read_text())
    metric = state["file_metrics"][0]
    assert metric["body_start_row"] == 0
    assert metric["body_rows"] == 3
    assert metric["grammar_valid_start_row"] == 0
    assert metric["grammar_valid_rows"] == 2
    assert metric["grammar_valid_id_sha256"]
    assert metric["quarantine_multiset"][0]["raw_value"] == f"{PREFIX}-1"

    report = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
    )
    assert report["pass_grammar_valid_main_id_recount"] is True
    final_state = json.loads((output_dir / "recount_state.json").read_text())
    assert final_state["body_rows_scanned"] == 4
    assert final_state["grammar_valid_rows_written"] == 3


@pytest.mark.parametrize("tamper", ["spool", "checkpoint_quarantine"])
def test_resume_rejects_segment_or_quarantine_checkpoint_tampering(
    tmp_path: Path, tamper: str
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            [f"{PREFIX}0", f"{PREFIX}-1", f"{PREFIX}1"],
            [f"{PREFIX}2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_valid_rows=3,
        expected_distinct=3,
        expected_duplicate_excess=0,
        quarantine=_location(f"{PREFIX}-1", "works/part-0.parquet", 0, 1),
        zero=_location(f"{PREFIX}0", "works/part-0.parquet", 0, 0),
    )
    output_dir = tmp_path / f"tamper_{tamper}"
    _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
        pause_after_new_files=1,
    )
    if tamper == "spool":
        with (output_dir / "main_ids.grammar_valid.unsorted.u64").open("r+b") as handle:
            handle.write(b"\xff")
        message = "segment checksum changed"
    else:
        state_path = output_dir / "recount_state.json"
        state = json.loads(state_path.read_text())
        state["file_metrics"][0]["quarantine_multiset"][0]["file_row"] = 99
        state_path.write_text(json.dumps(state), encoding="utf-8")
        message = "checkpoint metadata mismatch"

    with pytest.raises(recount.RecountError, match=message):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=output_dir,
        )


def test_complete_resume_validates_all_committed_artifacts(tmp_path: Path) -> None:
    root, manifest_path, config_path, output_dir = _happy_contract(tmp_path)
    first = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
    )
    resumed = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
    )
    assert resumed == first

    with (output_dir / "malformed_quarantine.json").open("r+b") as handle:
        handle.seek(0)
        handle.write(b"X")
    with pytest.raises(recount.RecountError, match="artifact checksum mismatch"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=output_dir,
        )


@pytest.mark.parametrize("managed_name", [None, "main_ids.grammar_valid.unsorted.u64"])
def test_output_and_managed_path_symlinks_fail_closed(
    tmp_path: Path, managed_name: str | None
) -> None:
    root, manifest_path, config_path, _ = _happy_contract(tmp_path)
    target = tmp_path / "outside_target"
    if managed_name is None:
        target.mkdir()
        output_dir = tmp_path / "output_link"
        output_dir.symlink_to(target, target_is_directory=True)
    else:
        output_dir = tmp_path / "managed_link_output"
        output_dir.mkdir()
        target.write_bytes(b"outside sentinel")
        (output_dir / managed_name).symlink_to(target)

    with pytest.raises(recount.RecountError, match="must not be a symlink"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=output_dir,
        )
    if managed_name is None:
        assert list(target.iterdir()) == []
    else:
        assert target.read_bytes() == b"outside sentinel"
