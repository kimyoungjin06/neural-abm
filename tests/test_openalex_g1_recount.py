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
RECOUNT_SCRIPT = REPO_ROOT / "scripts/recount_openalex_g1.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inventory = _load_module("test_recount_inventory_module", INVENTORY_SCRIPT)
recount = _load_module("test_openalex_g1_recount_module", RECOUNT_SCRIPT)


@pytest.fixture(autouse=True)
def _stable_git_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        inventory,
        "git_metadata",
        lambda: {"commit": "synthetic", "dirty": False, "status_error": None},
    )
    monkeypatch.setattr(
        recount,
        "git_metadata",
        lambda: {"commit": "synthetic", "dirty": False, "status_error": None},
    )


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


def _inventory_config(root: Path, file_count: int, rows: int) -> dict[str, Any]:
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
                "expected_file_count": file_count,
                "expected_footer_rows": rows,
                "required_columns": ["id", "publication_year"],
                "required_column_types": {"id": "string"},
            }
        },
        "pending_full_scan_gates": [],
    }


def _prepare_parent(
    tmp_path: Path,
    files: list[list[str | None]],
) -> tuple[Path, Path, dict[str, Any]]:
    root = tmp_path / "synthetic_snapshot"
    for index, ids in enumerate(files):
        _write_ids(root / "works" / f"part-{index}.parquet", ids)
    rows = sum(len(ids) for ids in files)
    inventory_config = _inventory_config(root, len(files), rows)
    inventory_config_path = tmp_path / "inventory.yaml"
    inventory_config_path.write_text(yaml.safe_dump(inventory_config), encoding="utf-8")
    manifest = inventory.audit_source(
        root=root,
        config=inventory_config,
        config_path=inventory_config_path,
        workers=1,
        content_sha256=False,
    )
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return root, manifest_path, manifest


def _recount_config(
    tmp_path: Path,
    root: Path,
    manifest: dict[str, Any],
    *,
    expected_distinct: int,
    expected_duplicate_excess: int,
    expected_null: int = 0,
    expected_invalid: int = 0,
) -> Path:
    config = {
        "schema_version": "openalex_g1_recount_config.v1",
        "label": "synthetic_recount",
        "parent_inventory": {
            "schema_version": manifest["schema_version"],
            "manifest_id": manifest["manifest_id"],
            "source_root_env": "SYNTHETIC_ROOT",
            "source_root_id": root.name,
            "inventory_fingerprint_sha256": manifest["root"][
                "inventory_fingerprint_sha256"
            ],
        },
        "main_id_recount": {
            "table": "works",
            "column": "id",
            "canonical_prefix": "https://openalex.org/W",
            "expected_files": manifest["root"]["parquet_file_count"],
            "expected_rows": manifest["root"]["footer_rows"],
            "expected_null_ids": expected_null,
            "expected_invalid_ids": expected_invalid,
            "expected_distinct_ids": expected_distinct,
            "expected_duplicate_row_excess": expected_duplicate_excess,
            "spool_dtype": "uint64",
            "batch_size": 2,
            "duplicate_scan_chunk_rows": 2,
            "invalid_sample_limit": 5,
        },
        "claim_boundary": "Synthetic main-ID recount only.",
        "pending_after_main_id_recount": ["synthetic_pending_gate"],
    }
    path = tmp_path / "recount.yaml"
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
        batch_size=2,
        pause_after_new_files=pause_after_new_files,
    )


def test_canonical_id_parser_is_injective_and_reports_bad_values() -> None:
    values = pa.array(
        [
            "https://openalex.org/W0",
            "https://openalex.org/W1",
            "https://openalex.org/W01",
            "https://example.org/W2",
            None,
        ],
        type=pa.string(),
    )

    numeric, nulls, invalid, samples = recount.parse_canonical_openalex_ids(
        values,
        prefix="https://openalex.org/W",
        sample_limit=5,
    )

    assert numeric.tolist() == [0, 1, 0, 0, 0]
    assert nulls == 1
    assert invalid == 2
    assert samples == ["https://openalex.org/W01", "https://example.org/W2"]

    single_numeric, _, single_invalid, _ = recount.parse_canonical_openalex_ids(
        pa.array(["https://openalex.org/W01"], type=pa.string()),
        prefix="https://openalex.org/W",
        sample_limit=5,
    )
    assert single_numeric.tolist() == [0]
    assert single_invalid == 1

    boundary_numeric, _, boundary_invalid, _ = recount.parse_canonical_openalex_ids(
        pa.array(
            [
                "https://openalex.org/W18446744073709551615",
                "https://openalex.org/W18446744073709551616",
            ],
            type=pa.string(),
        ),
        prefix="https://openalex.org/W",
        sample_limit=5,
    )
    assert boundary_numeric.tolist() == [np.iinfo(np.uint64).max, 0]
    assert boundary_invalid == 1


def test_canonical_id_parser_requires_exact_numeric_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_cast = recount.pc.cast

    def permissive_uint64_cast(
        values: pa.Array, target_type: pa.DataType, *args: Any, **kwargs: Any
    ) -> pa.Array:
        if pa.types.is_uint64(target_type) and values.to_pylist() == ["+1"]:
            return pa.array([1], type=pa.uint64())
        return real_cast(values, target_type, *args, **kwargs)

    monkeypatch.setattr(recount.pc, "cast", permissive_uint64_cast)
    numeric, nulls, invalid, samples = recount.parse_canonical_openalex_ids(
        pa.array(["https://openalex.org/W+1"], type=pa.string()),
        prefix="https://openalex.org/W",
        sample_limit=5,
    )

    assert numeric.tolist() == [0]
    assert nulls == 0
    assert invalid == 1
    assert samples == ["https://openalex.org/W+1"]


def test_splitmix64_vectors_and_exact_duplicates_cross_chunks() -> None:
    observed = recount.splitmix64(np.array([0, 1], dtype=np.uint64))
    assert [f"{int(value):016x}" for value in observed] == [
        "e220a8397b1dcdaf",
        "910a2dec89025cc1",
    ]
    values = np.array([1, 1, 2, 2, 2, 3, 4, 4], dtype=np.uint64)
    assert recount.exact_duplicate_statistics(values, chunk_rows=2) == {
        "distinct_ids": 4,
        "duplicate_row_excess": 4,
        "duplicate_id_groups": 3,
        "max_multiplicity": 3,
        "min_numeric_id": 1,
        "max_numeric_id": 4,
    }


def test_full_main_id_recount_is_exact_and_keeps_g1_false(tmp_path: Path) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1", "https://openalex.org/W2"],
            ["https://openalex.org/W2", "https://openalex.org/W3"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=3,
        expected_duplicate_excess=1,
    )

    report = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=tmp_path / "full_output",
    )

    assert report["status"] == "PASS_MAIN_ID_RECOUNT"
    assert report["pass_main_id_recount"] is True
    assert report["pass_g1"] is False
    assert report["counts"]["data_page_rows"] == 4
    assert report["counts"]["distinct_ids"] == 3
    assert report["counts"]["duplicate_row_excess"] == 1
    assert sum(report["diagnostic_buckets"]["histogram"]) == 4
    assert report["diagnostic_buckets"]["hash_is_equality_key"] is False
    assert set(report["artifact_sha256"]) == {
        "main_ids.unsorted.u64",
        "main_ids.sorted.u64",
    }


def test_engineering_file_pilot_cannot_promote_main_recount(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1", "https://openalex.org/W2"],
            ["https://openalex.org/W3", "https://openalex.org/W4"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=4,
        expected_duplicate_excess=0,
    )

    report = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=tmp_path / "pilot_output",
        max_files=1,
    )

    assert report["status"] == "PASS_ENGINEERING_PILOT"
    assert report["scope"] == "engineering_file_pilot"
    assert report["pass_main_id_recount"] is False
    assert report["pass_g1"] is False
    assert (
        next(
            item
            for item in report["checks"]
            if item["name"] == "exact_distinct_main_work_ids"
        )["status"]
        == "PENDING"
    )


def test_pause_is_file_atomic_and_resume_verifies_segments(tmp_path: Path) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
            ["https://openalex.org/W3"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=3,
        expected_duplicate_excess=0,
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
    assert paused["completed_files"] == 1
    state = json.loads((output_dir / "recount_state.json").read_text())
    assert state["completed_files"] == 1
    assert len(state["file_metrics"]) == 1

    report = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
    )
    assert report["status"] == "PASS_MAIN_ID_RECOUNT"
    final_state = json.loads((output_dir / "recount_state.json").read_text())
    assert final_state["completed_files"] == 3
    assert len(final_state["file_metrics"]) == 3


def test_existing_plan_requires_valid_self_hash_and_exact_current_payload(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )
    output_dir = tmp_path / "plan_tamper_output"
    _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
        pause_after_new_files=1,
    )
    state_path = output_dir / "recount_state.json"
    frozen_state = state_path.read_bytes()
    plan_path = output_dir / "recount_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["batch_size"] = int(plan["batch_size"]) + 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(recount.RecountError, match="different recount plan"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=output_dir,
        )
    assert state_path.read_bytes() == frozen_state

    unhashed_plan = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = recount.sha256_bytes(
        recount.canonical_json_bytes(unhashed_plan)
    )
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(recount.RecountError, match="different recount plan"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=output_dir,
        )
    assert state_path.read_bytes() == frozen_state


def test_missing_state_refuses_existing_recount_outputs(tmp_path: Path) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )
    output_dir = tmp_path / "missing_state_output"
    _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
    )
    state_path = output_dir / "recount_state.json"
    state_path.unlink()
    artifacts = (
        output_dir / "main_ids.unsorted.u64",
        output_dir / "main_ids.sorted.u64",
        output_dir / "main_id_recount.json",
        output_dir / "main_id_recount.md",
    )
    frozen_hashes = {path.name: recount.sha256_file(path) for path in artifacts}

    with pytest.raises(recount.RecountError, match="state is missing"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=output_dir,
        )

    assert not state_path.exists()
    assert {path.name: recount.sha256_file(path) for path in artifacts} == frozen_hashes


def test_resume_rejects_spool_segment_or_checkpoint_metadata_tampering(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )

    spool_output = tmp_path / "spool_tamper_output"
    _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=spool_output,
        pause_after_new_files=1,
    )
    spool = spool_output / "main_ids.unsorted.u64"
    with spool.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\xff")
    with pytest.raises(recount.RecountError, match="segment checksum changed"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=spool_output,
        )

    state_output = tmp_path / "state_tamper_output"
    _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=state_output,
        pause_after_new_files=1,
    )
    state_path = state_output / "recount_state.json"
    state = json.loads(state_path.read_text())
    state["file_metrics"][0]["start_row"] = 1
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(recount.RecountError, match="checkpoint metadata mismatch"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=state_output,
        )


def test_complete_resume_revalidates_report_and_binary_artifacts(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )
    output_dir = tmp_path / "complete_tamper_output"
    _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=output_dir,
    )
    sorted_artifact = output_dir / "main_ids.sorted.u64"
    with sorted_artifact.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\xff")

    with pytest.raises(recount.RecountError, match="artifact checksum mismatch"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=output_dir,
        )


def test_output_inside_frozen_source_is_rejected_before_creation(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )
    forbidden = root / "forbidden-output"

    with pytest.raises(recount.RecountError, match="inside the source root"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=forbidden,
        )
    assert not forbidden.exists()


@pytest.mark.parametrize("target_exists", [False, True])
def test_output_directory_symlink_is_rejected_without_touching_target(
    tmp_path: Path,
    target_exists: bool,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )
    target = tmp_path / "outside-output-target"
    if target_exists:
        target.mkdir()
    output_link = tmp_path / "output-link"
    output_link.symlink_to(target, target_is_directory=True)

    with pytest.raises(recount.RecountError, match="must not be a symlink"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=output_link,
        )

    if target_exists:
        assert list(target.iterdir()) == []
    else:
        assert not target.exists()


def test_all_managed_output_symlinks_are_rejected_fail_closed(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )
    managed_names = (
        "recount.lock",
        "recount_plan.json",
        "recount_state.json",
        "main_ids.unsorted.u64",
        "main_ids.sorted.u64",
        "main_id_recount.json",
        "main_id_recount.md",
        "recount_plan.json.partial",
        "recount_state.json.partial",
        "main_id_recount.json.partial",
        "main_id_recount.md.partial",
    )
    for target_exists in (False, True):
        for index, name in enumerate(managed_names):
            output_dir = tmp_path / f"managed-{int(target_exists)}-{index}"
            output_dir.mkdir()
            target = tmp_path / f"outside-{int(target_exists)}-{index}"
            if target_exists:
                target.write_bytes(b"outside sentinel")
            (output_dir / name).symlink_to(target)

            with pytest.raises(
                recount.RecountError,
                match=rf"managed recount path must not be a symlink: {name}",
            ):
                _run(
                    config_path=config_path,
                    manifest_path=manifest_path,
                    root=root,
                    output_dir=output_dir,
                )

            if target_exists:
                assert target.read_bytes() == b"outside sentinel"
            else:
                assert not target.exists()


def test_output_containment_checks_lexical_and_resolved_paths(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )
    outside = tmp_path / "outside-source"
    outside.mkdir()
    lexical_alias = root / "alias-to-outside"
    lexical_alias.symlink_to(outside, target_is_directory=True)
    lexical_output = lexical_alias / "recount"
    with pytest.raises(recount.RecountError, match="lexically inside"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=lexical_output,
        )
    assert not (outside / "recount").exists()

    resolved_alias = tmp_path / "alias-to-source"
    resolved_alias.symlink_to(root, target_is_directory=True)
    resolved_output = resolved_alias / "recount"
    with pytest.raises(recount.RecountError, match="resolve inside"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=resolved_output,
        )
    assert not (root / "recount").exists()


def test_invalid_body_id_blocks_even_when_expected_source_count_matches(
    tmp_path: Path,
) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1", "https://openalex.org/W01"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
        expected_invalid=0,
    )

    report = _run(
        config_path=config_path,
        manifest_path=manifest_path,
        root=root,
        output_dir=tmp_path / "invalid_output",
    )

    assert report["status"] == "BLOCKED"
    assert report["pass_main_id_recount"] is False
    assert report["counts"]["invalid_ids"] == 1
    invalid_check = next(
        item for item in report["checks"] if item["name"] == "invalid_main_work_ids"
    )
    assert invalid_check["status"] == "BLOCK"


def test_parent_file_drift_refuses_recount(tmp_path: Path) -> None:
    root, manifest_path, manifest = _prepare_parent(
        tmp_path,
        [
            ["https://openalex.org/W1"],
            ["https://openalex.org/W2"],
        ],
    )
    config_path = _recount_config(
        tmp_path,
        root,
        manifest,
        expected_distinct=2,
        expected_duplicate_excess=0,
    )
    source = root / "works/part-0.parquet"
    source.touch()

    with pytest.raises(recount.RecountError, match="mtime drift"):
        _run(
            config_path=config_path,
            manifest_path=manifest_path,
            root=root,
            output_dir=tmp_path / "drift_output",
        )
