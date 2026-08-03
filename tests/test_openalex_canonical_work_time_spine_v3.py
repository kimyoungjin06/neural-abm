from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v3.py"
SPEC = importlib.util.spec_from_file_location("canonical_spine_v3_runtime_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _source_parquet(
    path: Path,
    *,
    ids: list[str],
    dates: list[str | None] | None = None,
    years: list[int | None] | None = None,
    row_group_size: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dates = dates or ["2020-01-01"] * len(ids)
    years = years or [2020] * len(ids)
    table = pa.Table.from_arrays(
        [
            pa.array(ids, type=pa.string()),
            pa.array(dates, type=pa.string()),
            pa.array(years, type=pa.int64()),
        ],
        names=["id", "publication_date", "publication_year"],
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _record(work_id: int) -> object:
    return MODULE.Record(work_id, 18_262, 2020, 2)


def _fast_fixture_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "PARTITION_COUNT", 4)
    monkeypatch.setattr(
        MODULE,
        "_enforce_resource_preflight",
        lambda *_args, **_kwargs: {
            "observed_free_disk_bytes": 10**12,
            "observed_available_ram_bytes": 10**12,
            "observed_nofile_soft_limit": 16_384,
        },
    )


def _fixture_source(root: Path, *, two_files: bool = False) -> Path:
    source = root / "source"
    source.mkdir(mode=0o700)
    _source_parquet(
        source / "a.parquet",
        ids=["https://openalex.org/W4", "https://openalex.org/W1"],
        dates=[None, "2020-01-01"],
        years=[None, 2020],
        row_group_size=1,
    )
    if two_files:
        _source_parquet(
            source / "b.parquet",
            ids=["https://openalex.org/W2", "https://openalex.org/W3"],
            row_group_size=1,
        )
    return source


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_temporal_v3_json(path: Path, value: object) -> None:
    path.write_bytes(MODULE._canonical_json_bytes(value) + b"\n")
    path.chmod(0o600)


def _spool_checkpoint(output: Path, relative: str, records: list[object]) -> dict:
    token = MODULE._sha256_bytes(relative.encode("utf-8"))
    path = output / "staging/main" / f"{token}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(record.encode() for record in records)
    path.write_bytes(payload)
    path.chmod(0o600)
    return {
        "relative_path": relative,
        "staging_shard_relative_path": path.relative_to(output).as_posix(),
        "staging_shard_rows": len(records),
        "staging_shard_bytes": len(payload),
        "staging_shard_sha256": MODULE._sha256_file(path),
        "row_level_disposition_payload_sha256": MODULE._record_digest_file(
            path, len(records)
        ),
    }


def test_authorization_modes_are_fail_closed_and_pairwise_distinct() -> None:
    with pytest.raises(MODULE.SpineAuditError, match="requires"):
        MODULE._authorize_execution(
            execute=True,
            authorize_full=True,
            acknowledge_exclusive_source_scan=False,
            max_files=None,
        )
    assert (
        MODULE._authorize_execution(
            execute=True,
            authorize_full=True,
            acknowledge_exclusive_source_scan=True,
            max_files=None,
        )
        == MODULE.FULL_SCOPE
    )
    with pytest.raises(MODULE.SpineAuditError, match="forbidden"):
        MODULE._authorize_execution(
            execute=True,
            authorize_full=True,
            acknowledge_exclusive_source_scan=True,
            max_files=1,
        )
    assert (
        MODULE._authorize_execution(
            execute=True,
            authorize_full=False,
            acknowledge_exclusive_source_scan=False,
            max_files=1,
        )
        == MODULE.PILOT_SCOPE
    )
    with pytest.raises(MODULE.SpineAuditError, match="exactly one"):
        MODULE._authorize_execution(
            execute=True,
            authorize_full=False,
            acknowledge_exclusive_source_scan=False,
            max_files=2,
        )
    statuses = {
        MODULE._status(MODULE.FULL_SCOPE),
        MODULE._status(MODULE.PILOT_SCOPE),
        MODULE._status(MODULE.CLOSED_FIXTURE_SCOPE),
    }
    assert len(statuses) == 3
    schema_sets = [set(MODULE._schemas(scope).values()) for scope in (
        MODULE.FULL_SCOPE,
        MODULE.PILOT_SCOPE,
        MODULE.CLOSED_FIXTURE_SCOPE,
    )]
    assert all(left.isdisjoint(right) for index, left in enumerate(schema_sets) for right in schema_sets[index + 1 :])


@pytest.mark.parametrize(
    ("execute", "authorize_full", "acknowledge"),
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
    ],
)
def test_every_incomplete_full_authorization_tuple_fails_before_path_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    execute: bool,
    authorize_full: bool,
    acknowledge: bool,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_env_or_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authorization failure reached environment access")
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "run_production",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("authorization failure reached the runner")
        ),
    )
    argv = ["--output-dir", str(tmp_path / "unused")]
    if execute:
        argv.append("--execute")
    if authorize_full:
        argv.append("--authorize-full")
    if acknowledge:
        argv.append("--acknowledge-exclusive-source-scan")
    assert MODULE.main(argv) == 2


@pytest.mark.parametrize("max_files", [0, 2, 492])
def test_invalid_pilot_file_bounds_fail_before_path_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, max_files: int
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_env_or_path",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("pilot failure reached environment access")
        ),
    )
    assert (
        MODULE.main(
            [
                "--output-dir",
                str(tmp_path / "unused"),
                "--execute",
                "--max-files",
                str(max_files),
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    "mode_args",
    [
        [
            "--execute",
            "--authorize-full",
            "--acknowledge-exclusive-source-scan",
        ],
        ["--execute", "--max-files", "1"],
    ],
)
def test_valid_authorization_modes_reach_only_the_mocked_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode_args: list[str]
) -> None:
    monkeypatch.setattr(
        MODULE, "_env_or_path", lambda *_args, **_kwargs: tmp_path / "mock"
    )
    calls: list[dict] = []

    def mock_runner(**kwargs):
        calls.append(kwargs)
        return {"status": "MOCKED_AFTER_AUTHORIZATION"}

    monkeypatch.setattr(MODULE, "run_production", mock_runner)
    assert (
        MODULE.main(["--output-dir", str(tmp_path / "unused"), *mode_args])
        == 0
    )
    assert len(calls) == 1


def test_nondefault_config_is_rejected_before_yaml_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    copied = tmp_path / "copied.yaml"
    copied.write_text("not: parsed\n", encoding="utf-8")
    touched = False

    def forbidden_load(_path: Path):
        nonlocal touched
        touched = True
        raise AssertionError("nondefault config reached YAML parsing")

    monkeypatch.setattr(MODULE._V2, "_load_yaml_snapshot", forbidden_load)
    with pytest.raises(MODULE.SpineAuditError, match="repository default config"):
        MODULE._load_config(copied)
    assert touched is False


def test_temporal_repository_override_is_rejected_before_config_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    touched = False

    def forbidden_load(_path: Path):
        nonlocal touched
        touched = True
        raise AssertionError("repository override reached config access")

    monkeypatch.setattr(MODULE, "_load_config", forbidden_load)
    with pytest.raises(MODULE.SpineAuditError, match="override is forbidden"):
        MODULE.run_production(
            source_root=tmp_path / "missing-source",
            content_receipt_path=tmp_path / "missing-content",
            temporal_receipt_path=tmp_path / "missing-temporal",
            temporal_config_path=tmp_path / "missing-config",
            temporal_repo_root=tmp_path / "wrong-repository",
            output_dir=tmp_path / "missing-output",
            execute=True,
            authorize_full=True,
            acknowledge_exclusive_source_scan=True,
        )
    assert touched is False


def test_production_sizes_are_frozen_before_config_or_source_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    touched = False

    def forbidden_load(_path: Path) -> None:
        nonlocal touched
        touched = True
        raise AssertionError("config must not be opened")

    monkeypatch.setattr(MODULE, "_load_config", forbidden_load)
    with pytest.raises(MODULE.SpineAuditError, match="batch size"):
        MODULE.run_production(
            source_root=tmp_path / "missing-source",
            content_receipt_path=tmp_path / "missing-content",
            temporal_receipt_path=tmp_path / "missing-temporal",
            temporal_config_path=tmp_path / "missing-config",
            output_dir=tmp_path / "missing-output",
            execute=True,
            authorize_full=True,
            acknowledge_exclusive_source_scan=True,
            batch_size=1,
        )
    assert touched is False


@pytest.mark.parametrize(
    ("batch_size", "sort_rows"),
    [(0, 1), (MODULE.DEFAULT_BATCH_SIZE + 1, 1), (1, 0), (1, MODULE.DEFAULT_SORT_CHUNK_ROWS + 1)],
)
def test_closed_fixture_runtime_sizes_have_hard_caps(
    tmp_path: Path, batch_size: int, sort_rows: int
) -> None:
    with pytest.raises(MODULE.SpineAuditError, match="closed-fixture"):
        MODULE.run_closed_fixture(
            source_root=tmp_path / "unread",
            output_dir=tmp_path / "unused",
            batch_size=batch_size,
            partition_sort_chunk_rows=sort_rows,
        )


def test_full_resource_formula_and_live_preflight_are_numeric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    requirements = MODULE._resource_requirements(
        retained_rows=492_293_206,
        batch_size=MODULE.DEFAULT_BATCH_SIZE,
        sort_chunk_rows=MODULE.DEFAULT_SORT_CHUNK_ROWS,
        scope=MODULE.FULL_SCOPE,
        config=config,
    )
    assert requirements["base_required_free_disk_bytes_full"] == 95_321_921_648
    assert (
        requirements["required_free_disk_bytes_full_fresh"]
        == 153_143_246_352
    )
    assert (
        requirements["required_free_disk_bytes_full_nonterminal_resume"]
        == 95_321_921_648
    )
    assert requirements["preferred_free_disk_bytes_full"] == 171_798_691_840
    assert requirements["projected_batch_bytes_upper_bound"] == 541_065_216
    assert (
        requirements["spool_demultiplex_working_bytes_preflight_estimate"]
        == 134_217_728
    )
    assert requirements["calculated_ram_preflight_sizing_bytes"] == 9_685_696_512
    assert requirements["required_available_ram_bytes"] == 17_179_869_184
    assert requirements["minimum_nofile_soft_limit"] == 8192

    monkeypatch.setattr(
        MODULE.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10**12, used=0, free=10**12),
    )
    monkeypatch.setattr(MODULE, "_available_memory_bytes", lambda: 10**12)
    monkeypatch.setattr(MODULE.resource, "getrlimit", lambda _key: (16_384, 16_384))
    probes: list[Path] = []
    monkeypatch.setattr(MODULE, "_atomic_rename_probe", probes.append)
    observed = MODULE._enforce_resource_preflight(tmp_path / "new", requirements)
    assert observed["observed_nofile_soft_limit"] == 16_384
    assert probes == [tmp_path / "new"]

    monkeypatch.setattr(
        MODULE.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=1, used=1, free=0),
    )
    with pytest.raises(MODULE.SpineAuditError, match="disk"):
        MODULE._enforce_resource_preflight(tmp_path / "new", requirements)

    fixture_requirements = MODULE._resource_requirements(
        retained_rows=2,
        batch_size=2,
        sort_chunk_rows=2,
        scope=MODULE.CLOSED_FIXTURE_SCOPE,
        config=None,
    )
    assert fixture_requirements["minimum_nofile_soft_limit"] == 8192


def test_preflight_disk_floor_is_selected_from_validated_output_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    report = MODULE.run_closed_fixture(
        source_root=source,
        output_dir=output,
        pause_after_new_scan_files=1,
    )
    assert report["status"] == "RUNNING_SCAN"
    plan = _read_json(output / "audit_plan.json")
    assert (
        MODULE._classify_preflight_invocation(output, plan, resume=True)
        == MODULE.NONTERMINAL_RESUME_INVOCATION
    )

    missing = tmp_path / "missing"
    assert (
        MODULE._classify_preflight_invocation(missing, plan, resume=True)
        == MODULE.FRESH_INVOCATION
    )
    empty = tmp_path / "empty"
    empty.mkdir(mode=0o700)
    assert (
        MODULE._classify_preflight_invocation(empty, plan, resume=True)
        == MODULE.FRESH_INVOCATION
    )
    lock_only = tmp_path / "lock-only"
    lock_only.mkdir(mode=0o700)
    (lock_only / "audit.lock").touch(mode=0o600)
    assert (
        MODULE._classify_preflight_invocation(lock_only, plan, resume=True)
        == MODULE.FRESH_INVOCATION
    )

    (output / "run_state.json").unlink()
    with pytest.raises(MODULE.SpineAuditError, match="without a nonterminal state"):
        MODULE._classify_preflight_invocation(output, plan, resume=True)


def test_spool_has_exact_byte_count_and_materializes_empty_partitions(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    (output / ".work_time_spine_tmp").mkdir(parents=True)
    checkpoints = [
        _spool_checkpoint(output, "main/a.parquet", [_record(1), _record(4097)]),
        _spool_checkpoint(output, "main/b.parquet", [_record(2)]),
    ]
    manifest = MODULE._build_partition_spools(output, checkpoints)
    spool = output / ".work_time_spine_tmp/spool"
    files = sorted(spool.glob("part-*.bin"))
    assert len(files) == MODULE.PARTITION_COUNT
    assert manifest["rows"] == 3
    assert manifest["bytes"] == 3 * MODULE.RECORD_BYTES
    assert sum(path.stat().st_size for path in files) == manifest["bytes"]
    assert (spool / "part-0001.bin").stat().st_size == 2 * MODULE.RECORD_BYTES
    assert (spool / "part-0003.bin").stat().st_size == 0


def test_spool_handles_open_and_fsync_at_most_once_per_touched_partition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(MODULE, "PARTITION_COUNT", 8)
    root = tmp_path / "spool"
    root.mkdir()
    payload = b"".join(_record(work_id).encode() for work_id in range(8))
    records = MODULE.np.frombuffer(payload, dtype=MODULE.RECORD_DTYPE)
    real_open = MODULE.os.open
    real_fsync = MODULE.os.fsync
    open_counts: dict[str, int] = {}
    spool_open_flags: list[int] = []
    spool_descriptors: set[int] = set()
    fsync_count = 0

    def observed_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            descriptor = real_open(path, flags, mode)
        else:
            descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        candidate = Path(path)
        if candidate.parent == root and candidate.name.startswith("part-"):
            open_counts[candidate.name] = open_counts.get(candidate.name, 0) + 1
            spool_open_flags.append(flags)
            spool_descriptors.add(descriptor)
        return descriptor

    def observed_fsync(descriptor: int) -> None:
        nonlocal fsync_count
        if descriptor in spool_descriptors:
            fsync_count += 1
        real_fsync(descriptor)

    monkeypatch.setattr(MODULE.os, "open", observed_open)
    monkeypatch.setattr(MODULE.os, "fsync", observed_fsync)
    with MODULE._BoundedSpoolWriter(root) as writer:
        for _ in range(100):
            writer.write_records(records)
    assert open_counts == {f"part-{index:04d}.bin": 1 for index in range(8)}
    assert all(
        flags & MODULE.os.O_EXCL and flags & MODULE.os.O_NOFOLLOW
        for flags in spool_open_flags
    )
    assert fsync_count == 8


def _external_sort_output(tmp_path: Path, records: list[object], chunk_rows: int):
    output = tmp_path / "sort"
    spool = output / ".work_time_spine_tmp/spool"
    sorted_root = output / ".work_time_spine_tmp/sorted"
    spool.mkdir(parents=True)
    sorted_root.mkdir(parents=True)
    partition_id = int(records[0].work_id_numeric) % MODULE.PARTITION_COUNT
    path = spool / f"part-{partition_id:04d}.bin"
    payload = b"".join(record.encode() for record in records)
    path.write_bytes(payload)
    path.chmod(0o600)
    runs, rows = MODULE._write_sorted_runs(
        output,
        partition_id,
        sort_chunk_rows=chunk_rows,
        spool_contract={
            "partition_id": partition_id,
            "relative_path": f".work_time_spine_tmp/spool/part-{partition_id:04d}.bin",
            "rows": len(records),
            "bytes": len(payload),
            "sha256": MODULE._sha256_bytes(payload),
        },
    )
    return output, partition_id, runs, rows


def test_numpy_external_sort_and_k_way_merge_cross_chunk_boundaries(
    tmp_path: Path,
) -> None:
    records = [_record(12_289), _record(1), _record(8_193), _record(4_097)]
    output, partition_id, runs, rows = _external_sort_output(tmp_path, records, 2)
    assert len(runs) == 2
    checkpoint = MODULE._merge_runs_to_temporary_parquet(
        output,
        partition_id,
        runs,
        rows,
        task_fingerprint_sha256="a" * 64,
        checkpoint_schema=MODULE.CLOSED_FIXTURE_SCHEMAS["partition_checkpoint"],
        scan_tree_sha256="b" * 64,
        staging_tree_sha256="c" * 64,
    )
    table = pq.read_table(
        output / ".work_time_spine_tmp/sorted/part-0001.parquet"
    )
    assert table.column("work_id_numeric").to_pylist() == [1, 4_097, 8_193, 12_289]
    assert checkpoint["output_rows"] == 4
    assert checkpoint["pre_parquet_sorted_record_sha256"] == checkpoint[
        "decoded_parquet_record_sha256"
    ]


def test_duplicate_id_across_external_runs_fails_strictly(tmp_path: Path) -> None:
    output, partition_id, runs, rows = _external_sort_output(
        tmp_path, [_record(1), _record(4_097), _record(1)], 1
    )
    assert len(runs) == 3
    with pytest.raises(MODULE.SpineAuditError, match="duplicate numeric work ID"):
        MODULE._merge_runs_to_temporary_parquet(
            output,
            partition_id,
            runs,
            rows,
            task_fingerprint_sha256="a" * 64,
            checkpoint_schema=MODULE.CLOSED_FIXTURE_SCHEMAS["partition_checkpoint"],
            scan_tree_sha256="b" * 64,
            staging_tree_sha256="c" * 64,
        )


def test_scan_enforces_individual_and_combined_payload_caps_and_reuse_is_payload_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    _source_parquet(
        source / "main.parquet",
        ids=["https://openalex.org/W1", "https://openalex.org/W4097", "https://openalex.org/W2"],
        row_group_size=2,
    )
    task = MODULE._fixture_tasks(source)[0]
    output = tmp_path / "out"
    (output / ".work_time_spine_tmp/staging").mkdir(parents=True)
    (output / "staging/main").mkdir(parents=True)
    parquet = pq.ParquetFile(source / "main.parquet")
    batch = next(parquet.iter_batches(batch_size=2, columns=["id", "publication_date", "publication_year"]))
    serialized_bytes = batch.serialize().size
    encoded_bytes = batch.num_rows * MODULE.RECORD_BYTES
    common = {
        "batch_size": 2,
        "snapshot": date(2026, 3, 30),
        "checkpoint_schema": MODULE.CLOSED_FIXTURE_SCHEMAS["scan_checkpoint"],
        "task_fingerprint_sha256": "d" * 64,
        "projected_arrow_batch_bytes_bound": batch.nbytes,
        "projected_ipc_batch_bytes_bound": serialized_bytes,
        "encoded_batch_bytes_bound": encoded_bytes,
        "retained_payload_bytes_bound": 3 << 20,
    }
    for key, message in (
        ("projected_arrow_batch_bytes_bound", "Arrow batch"),
        ("projected_ipc_batch_bytes_bound", "serialized projected batch"),
        ("encoded_batch_bytes_bound", "encoded staging batch"),
    ):
        bounds = dict(common)
        bounds[key] -= 1
        with pytest.raises(MODULE.SpineAuditError, match=message):
            MODULE._scan_task_to_pending(source, output, task, **bounds)
    with pytest.raises(MODULE.SpineAuditError, match="combined bound"):
        MODULE._scan_task_to_pending(
            source,
            output,
            task,
            batch_size=2,
            snapshot=date(2026, 3, 30),
            checkpoint_schema=MODULE.CLOSED_FIXTURE_SCHEMAS["scan_checkpoint"],
            task_fingerprint_sha256="d" * 64,
            projected_arrow_batch_bytes_bound=batch.nbytes,
            projected_ipc_batch_bytes_bound=serialized_bytes,
            encoded_batch_bytes_bound=encoded_bytes,
            retained_payload_bytes_bound=batch.nbytes + serialized_bytes + encoded_bytes - 1,
        )

    checkpoint = MODULE._scan_task_to_pending(
        source,
        output,
        task,
        batch_size=2,
        snapshot=date(2026, 3, 30),
        checkpoint_schema=MODULE.CLOSED_FIXTURE_SCHEMAS["scan_checkpoint"],
        task_fingerprint_sha256="d" * 64,
        projected_arrow_batch_bytes_bound=1 << 20,
        projected_ipc_batch_bytes_bound=1 << 20,
        encoded_batch_bytes_bound=1 << 20,
        retained_payload_bytes_bound=3 << 20,
    )
    token = MODULE._sha256_bytes(task.relative_path.encode("utf-8"))
    pending = output / ".work_time_spine_tmp/staging" / f"{token}.bin"
    final = output / "staging/main" / f"{token}.bin"
    pending.replace(final)
    plan = {"scope": MODULE.CLOSED_FIXTURE_SCOPE, "task_fingerprint_sha256": "d" * 64}

    def forbidden_parquet(*_args, **_kwargs):
        raise AssertionError("reusable scan validation opened source payload")

    monkeypatch.setattr(MODULE.pq, "ParquetFile", forbidden_parquet)
    assert MODULE._validate_reusable_scan_checkpoint(
        output, plan, task, checkpoint, source_root=source
    ) == checkpoint
    with final.open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(MODULE.SpineAuditError, match="byte count"):
        MODULE._validate_reusable_scan_checkpoint(
            output, plan, task, checkpoint, source_root=source
        )


def test_pilot_evidence_never_claims_global_uniqueness() -> None:
    metrics = MODULE._empty_metrics()
    metrics["raw_rows"] = 2
    metrics["retained_rows"] = 2
    checkpoint = {
        "disposition_metrics": metrics,
        "publication_year_extrema": {"minimum": 2020, "maximum": 2020},
        "disposition_counts": {
            name: (2 if index == 0 else 0)
            for index, name in enumerate(MODULE.DISPOSITION_NAMES.values())
        },
    }
    evidence = MODULE._global_evidence(
        None, MODULE.PILOT_SCOPE, [checkpoint], []
    )
    accounting = evidence["identifier_accounting"]
    assert accounting["retained_rows"] == 2
    assert accounting["distinct_numeric_ids"] is None
    assert accounting["duplicate_numeric_id_groups"] is None
    assert accounting["global_uniqueness_verified"] is False
    assert accounting["global_uniqueness_status"] == "NOT_EVALUATED_IN_PREFIX_PILOT"


def test_report_fingerprint_covers_status_scope_runtime_and_pass_fields() -> None:
    report = {
        "schema_version": "schema",
        "status": "status",
        "scope": {"execution_scope": "scope"},
        "runtime_fingerprint": {"numpy_version": "x"},
        "production_component_pass": False,
        "pass_g1": False,
        "claim_boundary": "bounded",
        "provenance": {"tool_sha256": "a" * 64},
    }
    baseline = MODULE._report_fingerprint(report)
    for path, value in (
        (("status",), "mutated"),
        (("scope", "execution_scope"), "mutated"),
        (("runtime_fingerprint", "numpy_version"), "mutated"),
        (("production_component_pass",), True),
        (("pass_g1",), True),
    ):
        candidate = copy.deepcopy(report)
        if len(path) == 1:
            candidate[path[0]] = value
        else:
            candidate[path[0]][path[1]] = value
        assert MODULE._report_fingerprint(candidate) != baseline


def test_output_overlap_and_root_mode_fail_before_probe_or_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    touched = False

    def forbidden_probe(_path: Path) -> None:
        nonlocal touched
        touched = True

    monkeypatch.setattr(MODULE, "_atomic_rename_probe", forbidden_probe)
    with pytest.raises(MODULE.SpineAuditError, match="overlaps source"):
        MODULE._execute_pipeline(
            scope=MODULE.CLOSED_FIXTURE_SCOPE,
            config=None,
            config_sha256="0" * 64,
            config_identity={},
            tasks=(),
            parents={},
            parent_context=None,
            source_root=source,
            output_dir=source / "forbidden",
            snapshot=date(2026, 3, 30),
            batch_size=1,
            sort_chunk_rows=1,
            resume=False,
            enforce_distinct_device=False,
        )
    assert touched is False
    assert not (source / "forbidden").exists()

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    unsafe.chmod(0o755)
    with pytest.raises(MODULE.SpineAuditError, match="0700"):
        MODULE._validate_output_location(unsafe, source_root=source)


def test_receipt_appearing_before_lock_routes_terminal_without_nonterminal_writes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    terminal_calls: list[Path] = []

    monkeypatch.setattr(
        MODULE,
        "_enforce_resource_preflight",
        lambda *_args, **_kwargs: {
            "observed_free_disk_bytes": 10**12,
            "observed_available_ram_bytes": 10**12,
            "observed_nofile_soft_limit": 16_384,
        },
    )

    class ReceiptAppearsBeforeLock:
        def __enter__(self):
            MODULE._atomic_json(
                output / MODULE.FIXTURE_BASENAMES["receipt"], {"simulated": True}
            )

        def __exit__(self, exc_type, exc, traceback):
            return False

    monkeypatch.setattr(
        MODULE, "_output_lock", lambda _output: ReceiptAppearsBeforeLock()
    )

    def terminal_route(target: Path, **_kwargs):
        terminal_calls.append(target)
        return {"status": "ROUTED_TO_TERMINAL"}

    monkeypatch.setattr(MODULE, "_terminal_resume", terminal_route)
    monkeypatch.setattr(
        MODULE,
        "_write_or_validate_plan",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal race reached nonterminal plan write")
        ),
    )
    report = MODULE._execute_pipeline(
        scope=MODULE.CLOSED_FIXTURE_SCOPE,
        config=None,
        config_sha256="0" * 64,
        config_identity={},
        tasks=(),
        parents={},
        parent_context=None,
        source_root=source,
        output_dir=output,
        snapshot=date(2026, 3, 30),
        batch_size=1,
        sort_chunk_rows=1,
        resume=True,
        enforce_distinct_device=False,
    )
    assert report == {"status": "ROUTED_TO_TERMINAL"}
    assert terminal_calls == [output]
    assert not (output / "preflight_evidence.json").exists()
    assert not (output / "run_state.json").exists()


def test_integrity_hash_rejects_symlinks_and_path_replacement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"a" * 1024)
    symlink = tmp_path / "target-link.bin"
    symlink.symlink_to(target)
    with pytest.raises(MODULE.SpineAuditError, match="symlink"):
        MODULE._sha256_file(symlink, chunk_bytes=64)

    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"b" * 1024)
    real_read = MODULE.os.read
    swapped = False

    def swapping_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        block = real_read(descriptor, count)
        if block and not swapped:
            swapped = True
            MODULE.os.replace(replacement, target)
        return block

    monkeypatch.setattr(MODULE.os, "read", swapping_read)
    with pytest.raises(MODULE.SpineAuditError, match="identity changed"):
        MODULE._sha256_file(target, chunk_bytes=64)
    assert swapped is True


def test_regular_file_snapshot_returns_one_byte_hash_identity_view(
    tmp_path: Path,
) -> None:
    path = tmp_path / "control.json"
    payload = MODULE._V2.published_json_bytes({"key": "value"})
    path.write_bytes(payload)
    observed, digest, identity = MODULE._read_regular_file_snapshot(
        path, field="test control"
    )
    assert observed == payload
    assert digest == MODULE._sha256_bytes(payload)
    assert identity[6] == len(payload)


def test_terminal_intrinsic_failure_never_invokes_destructive_invalidation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected_plan = {"scope": MODULE.FULL_SCOPE}
    reads = iter(
        [
            {"scope": MODULE.PILOT_SCOPE},
            {"status": MODULE.PILOT_STATUS},
            {"scope": MODULE.PILOT_SCOPE},
        ]
    )
    monkeypatch.setattr(
        MODULE,
        "_read_parent_json_snapshot",
        lambda *_args, **_kwargs: (next(reads), b"", "a" * 64, ()),
    )
    monkeypatch.setattr(
        MODULE,
        "_validate_receipt_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MODULE.SpineAuditError("corrupt")
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "_invalidate_finalization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("terminal validation attempted destructive invalidation")
        ),
    )
    with pytest.raises(MODULE.SpineAuditError, match="corrupt"):
        MODULE._terminal_resume(
            tmp_path,
            expected_plan=expected_plan,
            tasks=(),
            source_root=tmp_path,
            config=None,
            parent_context=None,
        )


def test_sort_and_parquet_buffers_do_not_retain_python_rows() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "chunk: list[Record]" not in source
    assert "row_group: list[Record]" not in source
    assert "np.frombuffer(payload, dtype=RECORD_DTYPE)" in source
    assert "_encoded_records_to_table(row_group)" in source
    assert "del encoded_batch" in source


def test_record_digest_numpy_path_is_exact_and_rejects_structural_corruption(
    tmp_path: Path,
) -> None:
    records = [_record(1), MODULE.Record(2, None, None, 0), _record(3)]
    payload = b"".join(record.encode() for record in records)
    path = tmp_path / "records.bin"
    path.write_bytes(payload)
    expected = MODULE.hashlib.sha256()
    expected.update(len(MODULE.RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    expected.update(MODULE.RECORD_HASH_DOMAIN)
    expected.update(len(records).to_bytes(8, "little"))
    for offset in range(0, len(payload), MODULE.RECORD_BYTES):
        encoded = payload[offset : offset + MODULE.RECORD_BYTES]
        expected.update(MODULE.RECORD_BYTES.to_bytes(8, "little"))
        expected.update(encoded)
    assert MODULE._record_digest_file(path, len(records)) == expected.hexdigest()

    corrupted = bytearray(payload)
    corrupted[15] = 1
    path.write_bytes(corrupted)
    with pytest.raises(MODULE.SpineAuditError, match="reserved"):
        MODULE._record_digest_file(path, len(records))


def test_spool_and_run_hash_chain_rejects_width_preserving_id_flips(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    (output / ".work_time_spine_tmp").mkdir(parents=True)
    checkpoints = [
        _spool_checkpoint(output, "main/a.parquet", [_record(1), _record(4_097)])
    ]
    manifest = MODULE._build_partition_spools(output, checkpoints)
    spool_contract = manifest["partitions"][1]
    spool_path = output / spool_contract["relative_path"]
    original = spool_path.read_bytes()
    corrupted = bytearray(original)
    corrupted[2] ^= 1  # add 2^16: width/modulo remain valid
    spool_path.write_bytes(corrupted)
    with pytest.raises(MODULE.SpineAuditError, match="spool SHA-256"):
        MODULE._write_sorted_runs(
            output,
            1,
            sort_chunk_rows=1,
            spool_contract=spool_contract,
        )

    spool_path.write_bytes(original)
    runs, rows = MODULE._write_sorted_runs(
        output,
        1,
        sort_chunk_rows=4,
        spool_contract=spool_contract,
    )
    (output / ".work_time_spine_tmp/sorted").mkdir(parents=True, exist_ok=True)
    run_payload = bytearray(runs[0].path.read_bytes())
    run_payload[MODULE.RECORD_BYTES + 2] ^= 1
    runs[0].path.write_bytes(run_payload)
    with pytest.raises(MODULE.SpineAuditError, match="sort run SHA-256"):
        MODULE._merge_runs_to_temporary_parquet(
            output,
            1,
            runs,
            rows,
            task_fingerprint_sha256="a" * 64,
            checkpoint_schema=MODULE.CLOSED_FIXTURE_SCHEMAS["partition_checkpoint"],
            scan_tree_sha256="b" * 64,
            staging_tree_sha256="c" * 64,
        )


def test_staging_upper_id_bit_flip_fails_during_the_single_spool_read(
    tmp_path: Path,
) -> None:
    output = tmp_path / "out"
    (output / ".work_time_spine_tmp").mkdir(parents=True)
    checkpoint = _spool_checkpoint(
        output, "main/a.parquet", [_record(1), _record(4_097)]
    )
    staging = output / checkpoint["staging_shard_relative_path"]
    payload = bytearray(staging.read_bytes())
    payload[MODULE.RECORD_BYTES + 2] ^= 1
    staging.write_bytes(payload)
    with pytest.raises(MODULE.SpineAuditError, match="spool input SHA-256"):
        MODULE._build_partition_spools(output, [checkpoint])


def test_fixture_end_to_end_receipt_exact_set_and_terminal_resume_is_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    report = MODULE.run_closed_fixture(source_root=source, output_dir=output)
    assert report["status"] == MODULE.CLOSED_FIXTURE_STATUS
    assert report["production_component_pass"] is False
    assert report["scope"]["source_bytes"] == sum(
        path.stat().st_size for path in source.rglob("*.parquet")
    )
    assert report["publication_accounting"]["publication_year_minimum"] == 2020
    assert report["publication_accounting"]["publication_year_maximum"] == 2020
    assert sum(
        report["identifier_accounting"]["w_zero_disposition_counts"].values()
    ) == 0
    assert report["identifier_accounting"]["w_zero_disposition_category"] is None
    assert not (output / ".work_time_spine_tmp").exists()
    receipt_path = output / MODULE.FIXTURE_BASENAMES["receipt"]
    receipt = _read_json(receipt_path)
    plan = _read_json(output / "audit_plan.json")
    assert set(receipt["artifacts"]) == MODULE._receipt_artifact_paths(plan)
    assert receipt["artifact_count"] == 17
    assert "preflight_evidence.json" in receipt["artifacts"]
    before = {
        path.relative_to(output).as_posix(): MODULE._sha256_file(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    before_identity = {
        path.relative_to(output).as_posix(): MODULE._rich_identity(
            path, field="terminal identity before resume"
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    resumed = MODULE.run_closed_fixture(
        source_root=source, output_dir=output, resume=True
    )
    after = {
        path.relative_to(output).as_posix(): MODULE._sha256_file(path)
        for path in output.rglob("*")
        if path.is_file()
    }
    assert resumed == report
    assert after == before
    after_identity = {
        path.relative_to(output).as_posix(): MODULE._rich_identity(
            path, field="terminal identity after resume"
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after_identity == before_identity


def test_decoded_year_extrema_and_w_zero_cross_tab_are_conserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    _source_parquet(
        source / "main.parquet",
        ids=["https://openalex.org/W0", "https://openalex.org/W1"],
        dates=["2001-01-01", "2020-01-01"],
        years=[2001, 2020],
    )
    output = tmp_path / "output"
    report = MODULE.run_closed_fixture(source_root=source, output_dir=output)
    identifier = report["identifier_accounting"]
    publication = report["publication_accounting"]
    assert identifier["w_zero_rows"] == 1
    assert sum(identifier["w_zero_disposition_counts"].values()) == 1
    category = identifier["w_zero_disposition_category"]
    assert category is not None
    assert identifier["w_zero_disposition_counts"][category] == 1
    assert identifier["w_zero_entity_validity"] == "UNRESOLVED"
    assert publication["publication_year_minimum"] == 2001
    assert publication["publication_year_maximum"] == 2020
    assert publication["w_zero_disposition_counts"] == identifier[
        "w_zero_disposition_counts"
    ]

    scans = [
        _read_json(path)
        for path in sorted((output / "checkpoints/scan/main").glob("*.json"))
    ]
    partitions = [
        _read_json(path)
        for path in sorted((output / "checkpoints/partition").glob("*.json"))
    ]
    mutated_scan = copy.deepcopy(scans)
    mutated_scan[0]["publication_year_extrema"]["minimum"] = 2000
    with pytest.raises(MODULE.SpineAuditError, match="scan/decoded"):
        MODULE._global_evidence(
            None,
            MODULE.CLOSED_FIXTURE_SCOPE,
            mutated_scan,
            partitions,
        )

    mutated_partitions = copy.deepcopy(partitions)
    for checkpoint in mutated_partitions:
        checkpoint["w_zero_disposition_counts"] = {
            name: 0 for name in MODULE.DISPOSITION_NAMES.values()
        }
    with pytest.raises(MODULE.SpineAuditError, match="W0 row count"):
        MODULE._global_evidence(
            None,
            MODULE.CLOSED_FIXTURE_SCOPE,
            scans,
            mutated_partitions,
        )


def test_scan_pause_resume_reuses_completed_source_without_payload_rescan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path, two_files=True)
    output = tmp_path / "output"
    state = MODULE.run_closed_fixture(
        source_root=source,
        output_dir=output,
        pause_after_new_scan_files=1,
    )
    assert state["status"] == "RUNNING_SCAN"
    fresh_preflight = _read_json(output / "preflight_evidence.json")
    plan = _read_json(output / "audit_plan.json")
    assert fresh_preflight["invocation_kind"] == MODULE.FRESH_INVOCATION
    assert fresh_preflight["selected_required_available_disk_bytes"] == plan[
        "resource_requirements"
    ]["required_free_disk_bytes_full_fresh"]
    first_checkpoint = next((output / "checkpoints/scan/main").glob("*.json"))
    first_relative = _read_json(first_checkpoint)["relative_path"]
    original_scan = MODULE._scan_task_to_pending
    original_publish = MODULE._publish_scans
    scanned: list[str] = []
    durable_progress_seen: list[int] = []

    def observed_scan(*args, **kwargs):
        task = args[2]
        scanned.append(task.relative_path)
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(MODULE, "_scan_task_to_pending", observed_scan)

    def observed_publish(*args, **kwargs):
        durable_progress_seen.append(
            _read_json(output / "run_state.json")["completed_scan_files"]
        )
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(MODULE, "_publish_scans", observed_publish)
    report = MODULE.run_closed_fixture(
        source_root=source,
        output_dir=output,
        resume=True,
    )
    assert report["status"] == MODULE.CLOSED_FIXTURE_STATUS
    assert first_relative not in scanned
    assert len(scanned) == 1
    assert durable_progress_seen == [1]
    resumed_preflight = _read_json(output / "preflight_evidence.json")
    assert (
        resumed_preflight["invocation_kind"]
        == MODULE.NONTERMINAL_RESUME_INVOCATION
    )
    assert resumed_preflight["selected_required_available_disk_bytes"] == plan[
        "resource_requirements"
    ]["required_free_disk_bytes_full_nonterminal_resume"]


@pytest.mark.parametrize("phase", ["scan", "partition"])
def test_resume_floor_rejects_state_declared_missing_completed_pairs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path, two_files=phase == "scan")
    output = tmp_path / "output"
    options = (
        {"pause_after_new_scan_files": 1}
        if phase == "scan"
        else {"pause_after_new_partitions": 1}
    )
    MODULE.run_closed_fixture(source_root=source, output_dir=output, **options)
    plan = _read_json(output / "audit_plan.json")
    if phase == "scan":
        checkpoint = next((output / "checkpoints/scan/main").glob("*.json"))
        staging = output / _read_json(checkpoint)["staging_shard_relative_path"]
        checkpoint.unlink()
        staging.unlink()
        message = "scan prefix"
    else:
        (output / "checkpoints/partition/part-0000.json").unlink()
        (output / "spine/part-0000.parquet").unlink()
        message = "partition prefix"
    with pytest.raises(MODULE.SpineAuditError, match=message):
        MODULE._classify_preflight_invocation(output, plan, resume=True)


def test_resume_floor_allows_one_complete_scan_pair_ahead_of_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path, two_files=True)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(
        source_root=source,
        output_dir=output,
        pause_after_new_scan_files=1,
    )
    plan = _read_json(output / "audit_plan.json")
    MODULE._atomic_json(
        output / "run_state.json",
        MODULE._state(
            plan,
            "RUNNING_SCAN",
            completed_scan_files=0,
            completed_partitions=0,
        ),
    )
    assert (
        MODULE._classify_preflight_invocation(output, plan, resume=True)
        == MODULE.NONTERMINAL_RESUME_INVOCATION
    )


def test_partition_pause_resume_revalidates_and_skips_completed_partition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    state = MODULE.run_closed_fixture(
        source_root=source,
        output_dir=output,
        pause_after_new_partitions=1,
    )
    assert state["status"] == "RUNNING_PARTITION_SORT"
    assert (output / "checkpoints/partition/part-0000.json").exists()
    original_merge = MODULE._merge_runs_to_temporary_parquet
    merged: list[int] = []

    def observed_merge(*args, **kwargs):
        merged.append(args[1])
        return original_merge(*args, **kwargs)

    monkeypatch.setattr(MODULE, "_merge_runs_to_temporary_parquet", observed_merge)
    report = MODULE.run_closed_fixture(
        source_root=source,
        output_dir=output,
        resume=True,
    )
    assert report["status"] == MODULE.CLOSED_FIXTURE_STATUS
    assert 0 not in merged
    assert merged == [1, 2, 3]


@pytest.mark.parametrize("phase", ["scan", "partition"])
def test_corrupt_reusable_artifact_fails_closed_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path, two_files=phase == "scan")
    output = tmp_path / "output"
    options = (
        {"pause_after_new_scan_files": 1}
        if phase == "scan"
        else {"pause_after_new_partitions": 1}
    )
    MODULE.run_closed_fixture(source_root=source, output_dir=output, **options)
    if phase == "scan":
        target = next((output / "staging/main").glob("*.bin"))
    else:
        target = output / "spine/part-0000.parquet"
    with target.open("ab") as handle:
        handle.write(b"tamper")
    corrupted = target.read_bytes()
    with pytest.raises(MODULE.SpineAuditError):
        MODULE.run_closed_fixture(
            source_root=source,
            output_dir=output,
            resume=True,
        )
    assert target.read_bytes() == corrupted
    assert not (output / MODULE.FIXTURE_BASENAMES["receipt"]).exists()


def test_terminal_corruption_is_reported_without_changing_any_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(source_root=source, output_dir=output)
    report_path = output / MODULE.FIXTURE_BASENAMES["report"]
    report_path.write_bytes(report_path.read_bytes() + b" ")
    receipt_path = output / MODULE.FIXTURE_BASENAMES["receipt"]
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(MODULE.SpineAuditError, match="not canonical"):
        MODULE.run_closed_fixture(
            source_root=source,
            output_dir=output,
            resume=True,
        )
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert receipt_path.exists()
    assert after == before


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt(),
        OSError("transient terminal read failure"),
        MemoryError("transient terminal memory failure"),
    ],
    ids=["keyboard-interrupt", "os-error", "memory-error"],
)
def test_terminal_validation_interruption_or_io_error_is_strictly_read_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: BaseException
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(source_root=source, output_dir=output)
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(
        MODULE,
        "_validate_receipt_artifacts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    with pytest.raises(type(failure)):
        MODULE.run_closed_fixture(
            source_root=source,
            output_dir=output,
            resume=True,
        )
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_terminal_missing_lock_fails_without_recreating_or_changing_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(source_root=source, output_dir=output)
    lock_path = output / "audit.lock"
    lock_path.unlink()
    before = {
        path.relative_to(output).as_posix(): MODULE._rich_identity(
            path, field="missing-lock terminal identity"
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(MODULE.SpineAuditError, match="cannot open output lock"):
        MODULE.run_closed_fixture(
            source_root=source,
            output_dir=output,
            resume=True,
        )
    after = {
        path.relative_to(output).as_posix(): MODULE._rich_identity(
            path, field="missing-lock terminal identity after failure"
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    assert not lock_path.exists()
    assert after == before


def test_terminal_rejects_semantically_equal_but_reformatted_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(source_root=source, output_dir=output)
    receipt_path = output / MODULE.FIXTURE_BASENAMES["receipt"]
    receipt = _read_json(receipt_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_path.chmod(0o600)
    with pytest.raises(MODULE.SpineAuditError, match="not canonical"):
        MODULE.run_closed_fixture(
            source_root=source,
            output_dir=output,
            resume=True,
        )
    assert receipt_path.exists()
    assert receipt_path.read_text(encoding="utf-8") == json.dumps(receipt)


def test_terminal_rejects_undeclared_empty_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(source_root=source, output_dir=output)
    junk = output / "junk"
    junk.mkdir(mode=0o700)
    receipt_path = output / MODULE.FIXTURE_BASENAMES["receipt"]
    with pytest.raises(MODULE.SpineAuditError, match="directory set"):
        MODULE.run_closed_fixture(
            source_root=source,
            output_dir=output,
            resume=True,
        )
    assert receipt_path.exists()
    assert junk.is_dir()


def test_terminal_current_invocation_mismatch_preserves_valid_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(source_root=source, output_dir=output)
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    expected_plan = _read_json(output / "audit_plan.json")
    expected_plan["runtime_fingerprint"]["numpy_version"] = "incompatible"
    plan_core = dict(expected_plan)
    plan_core.pop("task_fingerprint_sha256")
    expected_plan["task_fingerprint_sha256"] = MODULE._sha256_bytes(
        MODULE._canonical_json_bytes(plan_core)
    )
    with pytest.raises(MODULE.TerminalCompatibilityError, match="no files changed"):
        MODULE._terminal_resume(
            output,
            expected_plan=expected_plan,
            tasks=MODULE._fixture_tasks(source),
            source_root=source,
            config=None,
            parent_context=None,
        )
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before


@pytest.mark.parametrize("drift", ["source", "parent"])
def test_terminal_current_source_or_parent_drift_is_no_write(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drift: str
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    tasks = MODULE._fixture_tasks(source)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(source_root=source, output_dir=output)
    expected_plan = _read_json(output / "audit_plan.json")
    parent_context = None
    if drift == "source":
        source_path = source / tasks[0].relative_path
        source_path.write_bytes(source_path.read_bytes())
    else:
        parent_marker = tmp_path / "parent-control.json"
        parent_marker.write_bytes(b"before")
        expected_parent_sha = MODULE._sha256_file(parent_marker)
        parent_context = MODULE.ParentContext(
            tasks=tasks,
            parents={},
            temporal_receipt_sha256="a" * 64,
            temporal_output_root=tmp_path,
            parent_snapshot_paths=((parent_marker, expected_parent_sha),),
        )
        parent_marker.write_bytes(b"after")
    before = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(MODULE.TerminalCompatibilityError, match="no files changed"):
        MODULE._terminal_resume(
            output,
            expected_plan=expected_plan,
            tasks=tasks,
            source_root=source,
            config=None,
            parent_context=parent_context,
        )
    after = {
        path.relative_to(output).as_posix(): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_pre_receipt_validation_failure_never_publishes_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        MODULE,
        "_validate_final_publication",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            MODULE.SpineAuditError("injected pre-receipt validation failure")
        ),
    )
    with pytest.raises(MODULE.SpineAuditError, match="pre-receipt"):
        MODULE.run_closed_fixture(source_root=source, output_dir=output)
    assert not (output / MODULE.FIXTURE_BASENAMES["receipt"]).exists()
    assert not (output / MODULE.FIXTURE_BASENAMES["report"]).exists()
    assert not (output / MODULE.FIXTURE_BASENAMES["markdown"]).exists()
    state = _read_json(output / "run_state.json")
    assert state["status"] == MODULE.FAILURE_STATUS
    assert "result_fingerprint_sha256" not in state


@pytest.mark.parametrize("drift", ["config", "tool"])
def test_post_plan_own_frozen_input_drift_never_publishes_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drift: str
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        MODULE,
        "_revalidate_own_frozen_inputs",
        lambda _plan: (_ for _ in ()).throw(
            MODULE.SpineAuditError(f"{drift} changed after plan construction")
        ),
    )
    with pytest.raises(MODULE.SpineAuditError, match="changed after plan"):
        MODULE.run_closed_fixture(source_root=source, output_dir=output)
    assert not (output / MODULE.FIXTURE_BASENAMES["receipt"]).exists()
    assert not (output / MODULE.FIXTURE_BASENAMES["report"]).exists()
    assert not (output / MODULE.FIXTURE_BASENAMES["markdown"]).exists()
    state = _read_json(output / "run_state.json")
    assert state["status"] == MODULE.FAILURE_STATUS
    assert state.get("production_component_pass") is not True


def test_post_receipt_interrupt_preserves_the_atomically_published_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path)
    output = tmp_path / "output"
    monkeypatch.setattr(
        MODULE,
        "_validate_published_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    with pytest.raises(KeyboardInterrupt):
        MODULE.run_closed_fixture(source_root=source, output_dir=output)
    receipt_path = output / MODULE.FIXTURE_BASENAMES["receipt"]
    report_path = output / MODULE.FIXTURE_BASENAMES["report"]
    markdown_path = output / MODULE.FIXTURE_BASENAMES["markdown"]
    assert receipt_path.exists()
    assert report_path.exists()
    assert markdown_path.exists()
    state = _read_json(output / "run_state.json")
    assert state["status"] == "COMPLETE_PENDING_RECEIPT"
    receipt = _read_json(receipt_path)
    assert receipt_path.read_bytes() == MODULE._V2.published_json_bytes(receipt)


def test_nonpilot_disposition_and_nullability_bridges_fail_on_compensating_mutation() -> None:
    metrics = MODULE._empty_metrics()
    metrics["raw_rows"] = 2
    metrics["retained_rows"] = 2
    metrics["publication_both_null_rows"] = 1
    metrics["publication_year_null_rows"] = 1
    scan_counts = {name: 0 for name in MODULE.DISPOSITION_NAMES.values()}
    scan_counts["BOTH_NULL_EXCLUDED"] = 1
    scan_counts["USABLE_1900_BEFORE_SNAPSHOT_YEAR"] = 1
    scan = {
        "disposition_metrics": metrics,
        "disposition_counts": scan_counts,
        "publication_year_extrema": {"minimum": 2020, "maximum": 2020},
    }
    partition_counts = dict(scan_counts)
    partition_counts["BOTH_NULL_EXCLUDED"] = 0
    partition_counts["USABLE_1900_BEFORE_SNAPSHOT_YEAR"] = 2
    partition = {
        "output_rows": 2,
        "minimum_work_id_numeric": 1,
        "maximum_work_id_numeric": 2,
        "disposition_counts": partition_counts,
        "w_zero_disposition_counts": {
            name: 0 for name in MODULE.DISPOSITION_NAMES.values()
        },
        "publication_year_extrema": {"minimum": 2020, "maximum": 2020},
        "nullability_counts": {
            "publication_day_null_rows": 1,
            "publication_year_null_rows": 1,
        },
    }
    with pytest.raises(MODULE.SpineAuditError, match="disposition counts"):
        MODULE._global_evidence(
            None, MODULE.CLOSED_FIXTURE_SCOPE, [scan], [partition]
        )
    partition["disposition_counts"] = scan_counts
    partition["nullability_counts"] = {
        "publication_day_null_rows": 0,
        "publication_year_null_rows": 0,
    }
    with pytest.raises(MODULE.SpineAuditError, match="nullability"):
        MODULE._global_evidence(
            None, MODULE.CLOSED_FIXTURE_SCOPE, [scan], [partition]
        )


def test_reformatted_resume_plan_bytes_fail_without_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fast_fixture_runtime(monkeypatch)
    source = _fixture_source(tmp_path, two_files=True)
    output = tmp_path / "output"
    MODULE.run_closed_fixture(
        source_root=source,
        output_dir=output,
        pause_after_new_scan_files=1,
    )
    plan_path = output / "audit_plan.json"
    parsed = _read_json(plan_path)
    plan_path.write_text(json.dumps(parsed, sort_keys=False), encoding="utf-8")
    plan_path.chmod(0o600)
    changed = plan_path.read_bytes()
    with pytest.raises(MODULE.SpineAuditError, match="plan bytes"):
        MODULE.run_closed_fixture(
            source_root=source,
            output_dir=output,
            resume=True,
        )
    assert plan_path.read_bytes() == changed


def test_cli_has_no_fixture_escape_hatch_and_rejects_flags_before_env_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    parser = MODULE._parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--test-fixture" not in option_strings
    assert "--synthetic-fixture-only" not in option_strings
    monkeypatch.delenv("OPENALEX_DERIVED_ROOT", raising=False)
    assert MODULE.main(["--output-dir", str(tmp_path / "out")]) == 2
    assert "full execution requires" in capsys.readouterr().err


def test_temporal_v3_snapshot_requires_exact_compact_newline_published_abi(
    tmp_path: Path,
) -> None:
    value = {"z": [3, 2, 1], "a": {"unicode": "시간"}}
    path = tmp_path / "temporal-receipt.json"
    exact_payload = MODULE._canonical_json_bytes(value) + b"\n"
    path.write_bytes(exact_payload)
    path.chmod(0o600)

    observed, payload, digest, _ = MODULE._read_temporal_v3_json_snapshot(
        path, field="temporal receipt"
    )
    assert observed == value
    assert payload == exact_payload
    assert digest == MODULE._sha256_bytes(exact_payload)

    reformatted = MODULE._V2.published_json_bytes(value)
    assert json.loads(reformatted) == value
    path.write_bytes(reformatted)
    with pytest.raises(
        MODULE.SpineAuditError,
        match="temporal v3 published JSON ABI",
    ):
        MODULE._read_temporal_v3_json_snapshot(path, field="temporal receipt")


def _temporal_checkpoint_identity_pair() -> tuple[dict, dict]:
    rich_identity = {
        "bytes": 312250942,
        "ctime_ns": 1782698887400757000,
        "device": 2050,
        "gid": 0,
        "inode": 2982181,
        "links": 1,
        "mode": 33279,
        "mtime_ns": 1775797706367900200,
        "uid": 0,
    }
    task = {
        "source_rich_identity": rich_identity,
        "footer_rows": 1_000_000,
        "projected_schema_sha256": "a" * 64,
    }
    checkpoint = {
        "source_identity": {
            **rich_identity,
            "footer_rows": task["footer_rows"],
            "projected_schema_sha256": task["projected_schema_sha256"],
        }
    }
    return task, checkpoint


def test_temporal_checkpoint_source_identity_matches_producer_abi() -> None:
    task, checkpoint = _temporal_checkpoint_identity_pair()
    MODULE._validate_temporal_checkpoint_source_identity(checkpoint, task)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_field_name", "temporal checkpoint source identity"),
        ("extra_key", "source identity key set"),
        ("rich_identity", "rich source identity"),
        ("footer_rows", "source footer"),
        ("projected_schema_sha256", "source schema"),
    ],
)
def test_temporal_checkpoint_source_identity_rejects_abi_drift(
    mutation: str, message: str
) -> None:
    task, checkpoint = _temporal_checkpoint_identity_pair()
    source_identity = checkpoint["source_identity"]
    if mutation == "wrong_field_name":
        checkpoint["source_rich_identity"] = checkpoint.pop("source_identity")
    elif mutation == "extra_key":
        source_identity["unexpected"] = 1
    elif mutation == "rich_identity":
        source_identity["inode"] += 1
    elif mutation == "footer_rows":
        source_identity["footer_rows"] += 1
    else:
        source_identity["projected_schema_sha256"] = "b" * 64
    with pytest.raises(MODULE.SpineAuditError, match=message):
        MODULE._validate_temporal_checkpoint_source_identity(checkpoint, task)


def _temporal_checkpoint_shape_pair() -> tuple[dict, dict]:
    checkpoint = {key: None for key in MODULE.TEMPORAL_V3_CHECKPOINT_KEYS}
    checkpoint["file_index"] = 7
    checkpoint["operator_paths_redacted"] = True
    return {"file_index": 7}, checkpoint


def test_temporal_checkpoint_shape_matches_exact_producer_abi() -> None:
    task, checkpoint = _temporal_checkpoint_shape_pair()
    MODULE._validate_temporal_checkpoint_shape(checkpoint, task)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("key_set", "checkpoint key set"),
        ("file_index", "file index"),
        ("redaction", "path redaction"),
    ],
)
def test_temporal_checkpoint_shape_rejects_abi_drift(
    mutation: str, message: str
) -> None:
    task, checkpoint = _temporal_checkpoint_shape_pair()
    if mutation == "key_set":
        checkpoint.pop("metrics")
    elif mutation == "file_index":
        checkpoint["file_index"] = 8
    else:
        checkpoint["operator_paths_redacted"] = False
    with pytest.raises(MODULE.SpineAuditError, match=message):
        MODULE._validate_temporal_checkpoint_shape(checkpoint, task)


def test_temporal_module_config_contract_is_read_only_and_exact() -> None:
    config_sha = "6" * 64

    class ReadOnlyTemporalModule:
        PRODUCTION_CONFIG_SHA256 = config_sha
        FROZEN_CONFIG_SHA256 = config_sha

        def __setattr__(self, _name: str, _value: object) -> None:
            raise AssertionError("temporal producer constants must not be mutated")

    producer = ReadOnlyTemporalModule()
    MODULE._validate_temporal_module_config_contract(
        producer,
        observed_config_sha256=config_sha,
        required_config_sha256=config_sha,
    )
    assert producer.PRODUCTION_CONFIG_SHA256 == config_sha
    assert producer.FROZEN_CONFIG_SHA256 == config_sha


@pytest.mark.parametrize(
    ("production", "frozen", "observed", "required"),
    [
        ("7" * 64, "6" * 64, "6" * 64, "6" * 64),
        ("6" * 64, "7" * 64, "6" * 64, "6" * 64),
        ("6" * 64, "6" * 64, "7" * 64, "6" * 64),
        ("6" * 64, "6" * 64, "6" * 64, "7" * 64),
    ],
)
def test_temporal_module_config_contract_rejects_provenance_drift(
    production: str, frozen: str, observed: str, required: str
) -> None:
    producer = SimpleNamespace(
        PRODUCTION_CONFIG_SHA256=production,
        FROZEN_CONFIG_SHA256=frozen,
    )
    with pytest.raises(MODULE.SpineAuditError, match="config provenance"):
        MODULE._validate_temporal_module_config_contract(
            producer,
            observed_config_sha256=observed,
            required_config_sha256=required,
        )


@pytest.mark.parametrize("drift", ["config", "tool"])
def test_own_frozen_input_revalidation_rejects_post_plan_file_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, drift: str
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    paths = {
        "config": root / "config.yaml",
        "tool": root / "runner.py",
        "foundation": root / "foundation.yaml",
        "predecessor": root / "predecessor.py",
    }
    for name, path in paths.items():
        path.write_bytes(f"{name}\n".encode())
        path.chmod(0o600)
    monkeypatch.setattr(MODULE, "REPO_ROOT", root)
    snapshots = {
        name: MODULE._frozen_file_snapshot(path, field=f"test {name}")
        for name, path in paths.items()
    }
    plan = {
        "scope": MODULE.PILOT_SCOPE,
        "own_frozen_input_revalidation_required": True,
        "frozen_file_snapshot_is_loaded_code_attestation": False,
        "config_relative_path": "config.yaml",
        "config_sha256": snapshots["config"]["sha256"],
        "config_rich_identity": snapshots["config"]["rich_identity"],
        "tool_relative_path": "runner.py",
        "tool_sha256": snapshots["tool"]["sha256"],
        "tool_rich_identity": snapshots["tool"]["rich_identity"],
        "foundation_config_relative_path": "foundation.yaml",
        "foundation_config_sha256": snapshots["foundation"]["sha256"],
        "foundation_config_rich_identity": snapshots["foundation"][
            "rich_identity"
        ],
        "foundation_synthetic_runner_relative_path": "predecessor.py",
        "foundation_synthetic_runner_sha256": snapshots["predecessor"][
            "sha256"
        ],
        "foundation_synthetic_runner_rich_identity": snapshots["predecessor"][
            "rich_identity"
        ],
    }
    MODULE._revalidate_own_frozen_inputs(plan)
    paths[drift].write_bytes(paths[drift].read_bytes() + b"drift\n")
    with pytest.raises(MODULE.SpineAuditError, match=f"{drift} changed"):
        MODULE._revalidate_own_frozen_inputs(plan)


def test_closed_fixture_own_frozen_input_revalidation_is_explicitly_exempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_frozen_file_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fixture exemption attempted a repository snapshot")
        ),
    )
    MODULE._revalidate_own_frozen_inputs(
        {
            "scope": MODULE.CLOSED_FIXTURE_SCOPE,
            "own_frozen_input_revalidation_required": False,
        }
    )


def test_parent_control_validator_never_opens_current_source_parquet_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir(mode=0o700)
    source_path = source_root / "main/not-really-parquet.parquet"
    source_path.parent.mkdir()
    source_path.write_bytes(b"parent-validator-must-only-stat-this-file")
    source_identity = MODULE._rich_identity(source_path, field="test source")

    content_root = tmp_path / "content"
    content_root.mkdir(mode=0o700)
    for relative, payload in (
        ("audit_plan.json", b"{}\n"),
        ("content_freeze_audit.json", b"{}\n"),
        ("content_freeze_audit.md", b"bounded\n"),
    ):
        (content_root / relative).write_bytes(payload)
    MODULE._atomic_json(content_root / "run_state.json", {"status": "COMPLETE"})
    content_artifacts = [
        MODULE._artifact_contract(content_root, relative)
        for relative in (
            "audit_plan.json",
            "content_freeze_audit.json",
            "content_freeze_audit.md",
        )
    ]
    content_values = {
        "schema_version": "content.outputs.v1",
        "status": "CONTENT_PASS",
        "plan_sha256": "1" * 64,
        "task_fingerprint_sha256": "2" * 64,
        "result_fingerprint_sha256": "3" * 64,
        "content_tree_sha256": "4" * 64,
        "completion_contract": "CONTENT_COMPLETE",
        "checkpoint_tree": {"tree_sha256": "5" * 64},
        "artifacts": content_artifacts,
    }
    content_receipt_path = content_root / "content.outputs.json"
    MODULE._atomic_json(content_receipt_path, content_values)
    content_receipt_sha = MODULE._sha256_file(content_receipt_path)
    commitment = {
        "receipt_sha256": content_receipt_sha,
        "task_fingerprint_sha256": content_values["task_fingerprint_sha256"],
        "content_tree_sha256": content_values["content_tree_sha256"],
    }

    temporal_root = tmp_path / "temporal"
    temporal_root.mkdir(mode=0o700)
    checkpoint_root = temporal_root / "checkpoints/main"
    checkpoint_root.mkdir(parents=True)
    projected_schema = {
        name: {"type": "string", "nullable": True}
        for name in (
            "id",
            "publication_date",
            "publication_year",
            "created_date",
            "updated_date",
        )
    }
    task = {
        "file_index": 0,
        "relative_path": "main/not-really-parquet.parquet",
        "footer_rows": 0,
        "source_rich_identity": source_identity,
        "projected_columns": [
            "id",
            "publication_date",
            "publication_year",
            "created_date",
            "updated_date",
        ],
        "projected_schema": projected_schema,
        "projected_schema_sha256": MODULE._sha256_bytes(
            MODULE._canonical_json_bytes(projected_schema)
        ),
    }
    task["task_sha256"] = MODULE._sha256_bytes(MODULE._canonical_json_bytes(task))
    source_tree = {"tree_sha256": "6" * 64, "task_count": 1}
    plan = {
        "schema_version": "temporal.plan.v3",
        "full_scope": True,
        "pilot_mode": None,
        "tasks": [task],
        "source_tree_start": source_tree,
    }
    plan["task_fingerprint_sha256"] = MODULE._sha256_bytes(
        MODULE._canonical_json_bytes(plan)
    )
    metrics = {key: 0 for key in MODULE.TEMPORAL_V3_METRIC_KEYS}
    checkpoint = {
        "schema_version": "temporal.checkpoint.v3",
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "task_sha256": task["task_sha256"],
        "file_index": task["file_index"],
        "relative_path": task["relative_path"],
        "source_identity": {
            **source_identity,
            "footer_rows": task["footer_rows"],
            "projected_schema_sha256": task["projected_schema_sha256"],
        },
        "projected_payload_sha256": "9" * 64,
        "metrics": metrics,
        "metrics_sha256": MODULE._sha256_bytes(
            MODULE._canonical_json_bytes(metrics)
        ),
        "publication_year_extrema": {"minimum": None, "maximum": None},
        "quarantine_observations": [],
        "ordering_diagnostic_samples": [],
        "violation_samples": [],
        "operator_paths_redacted": True,
    }
    checkpoint["checkpoint_sha256"] = MODULE._sha256_bytes(
        MODULE._canonical_json_bytes(checkpoint)
    )
    token = MODULE._sha256_bytes(task["relative_path"].encode("utf-8"))
    _write_temporal_v3_json(checkpoint_root / f"{token}.json", checkpoint)
    checkpoint_tree = {"checkpoint_tree_sha256": "7" * 64, "count": 1}
    report = {
        "schema_version": "temporal.report.v3",
        "status": "TEMPORAL_PASS",
        "checkpoint_evidence": checkpoint_tree,
        "content_baseline_parent": commitment,
    }
    state = {
        "schema_version": "temporal.state.v3",
        "status": "COMPLETE_PENDING_RECEIPT",
    }
    _write_temporal_v3_json(temporal_root / "audit_plan.json", plan)
    _write_temporal_v3_json(
        temporal_root / "main_temporal_validity_audit.json", report
    )
    (temporal_root / "main_temporal_validity_audit.md").write_text(
        "bounded\n", encoding="utf-8"
    )
    _write_temporal_v3_json(temporal_root / "run_state.json", state)
    fixed_paths = [
        "audit_plan.json",
        "run_state.json",
        "main_temporal_validity_audit.json",
        "main_temporal_validity_audit.md",
    ]
    artifact_paths = fixed_paths + [f"checkpoints/main/{token}.json"]
    temporal_artifacts = {
        relative: MODULE._artifact_contract(temporal_root, relative)
        for relative in artifact_paths
    }
    temporal_receipt = {
        "schema_version": "temporal.outputs.v3",
        "status": "TEMPORAL_PASS",
        "result_fingerprint_sha256": "8" * 64,
        "task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "artifacts": temporal_artifacts,
        "completion_contract": MODULE.COMPLETION_ABI,
        "operator_paths_redacted": True,
    }
    temporal_receipt_path = temporal_root / "temporal.outputs.json"
    _write_temporal_v3_json(temporal_receipt_path, temporal_receipt)
    temporal_receipt_sha = MODULE._sha256_file(temporal_receipt_path)
    temporal_config = tmp_path / "temporal.yaml"
    temporal_config.write_text("schema_version: synthetic\n", encoding="utf-8")

    content_config_relative = "pyproject.toml"
    content_tool_relative = "README.md"
    temporal_tool_relative = "scripts/audit_openalex_main_temporal_validity_v3.py"
    parent = {
        "required_receipt_basename": temporal_receipt_path.name,
        "required_receipt_sha256": temporal_receipt_sha,
        "required_config_sha256": MODULE._sha256_file(temporal_config),
        "required_tool_path": temporal_tool_relative,
        "required_tool_sha256": MODULE._sha256_file(REPO_ROOT / temporal_tool_relative),
        "exact_receipt_keyset": list(temporal_receipt),
        "schema_version": temporal_receipt["schema_version"],
        "required_status": temporal_receipt["status"],
        "required_plan_schema_version": plan["schema_version"],
        "required_checkpoint_schema_version": checkpoint["schema_version"],
        "required_report_schema_version": report["schema_version"],
        "required_state_schema_version": state["schema_version"],
        "required_state_status": state["status"],
        "required_report_sha256": MODULE._sha256_file(
            temporal_root / "main_temporal_validity_audit.json"
        ),
        "required_full_task_count": 1,
        "exact_fixed_receipt_artifact_paths": fixed_paths,
        "exact_total_receipt_artifact_count": 5,
        "required_exact_task_fingerprint_sha256": plan["task_fingerprint_sha256"],
        "required_exact_result_fingerprint_sha256": temporal_receipt[
            "result_fingerprint_sha256"
        ],
        "required_exact_checkpoint_tree_sha256": checkpoint_tree[
            "checkpoint_tree_sha256"
        ],
        "required_exact_source_tree_sha256": source_tree["tree_sha256"],
    }
    content_anchor = {
        "required_json_serialization_abi": MODULE.PRETTY_JSON_ABI,
        "required_receipt_basename": content_receipt_path.name,
        "required_receipt_sha256": content_receipt_sha,
        "required_schema_version": content_values["schema_version"],
        "required_status": content_values["status"],
        "required_plan_sha256": content_values["plan_sha256"],
        "required_task_fingerprint_sha256": content_values[
            "task_fingerprint_sha256"
        ],
        "required_result_fingerprint_sha256": content_values[
            "result_fingerprint_sha256"
        ],
        "required_content_tree_sha256": content_values["content_tree_sha256"],
        "required_completion_contract": content_values["completion_contract"],
        "required_checkpoint_tree_sha256": content_values["checkpoint_tree"][
            "tree_sha256"
        ],
        "required_config_path": content_config_relative,
        "required_config_sha256": MODULE._sha256_file(
            REPO_ROOT / content_config_relative
        ),
        "required_tool_path": content_tool_relative,
        "required_tool_sha256": MODULE._sha256_file(REPO_ROOT / content_tool_relative),
        "required_run_state_status": "COMPLETE",
    }
    config = {
        "parents": {"main_temporal_validity": parent},
        "production_parent_anchors": {
            "imported_temporal_module_globals_mutation_forbidden": True,
            "imported_temporal_module_repo_root_must_equal_current_repository_root": True,
            "temporal_module_production_and_frozen_config_constants_must_equal_anchored_config_sha256": True,
            "main_temporal_validity": {
                "required_receipt_sha256": temporal_receipt_sha,
                "required_json_serialization_abi": MODULE.TEMPORAL_V3_JSON_ABI,
            },
            "current_content_baseline": content_anchor,
        },
    }
    fake_temporal = SimpleNamespace(
        PRODUCTION_CONFIG_SHA256=parent["required_config_sha256"],
        FROZEN_CONFIG_SHA256=parent["required_config_sha256"],
        _report_fingerprint=lambda _report: temporal_receipt[
            "result_fingerprint_sha256"
        ],
        _checkpoint_tree=lambda _checkpoints: checkpoint_tree,
        _source_tree=lambda _tasks: source_tree,
        _validate_content_freeze_parent=lambda *_args, **_kwargs: {
            "commitment": commitment
        },
        _bind_current_source_to_content_baseline=lambda *_args: None,
    )
    monkeypatch.setattr(MODULE, "_load_temporal_module", lambda *_args: fake_temporal)
    monkeypatch.setattr(
        MODULE._V2, "_temporal_parent_config_for_content", lambda *_args: {}
    )
    monkeypatch.setattr(
        MODULE.pq,
        "ParquetFile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("parent validator opened source Parquet payload")
        ),
    )
    context = MODULE._validate_parent_control_plane(
        config,
        source_root=source_root,
        content_receipt_path=content_receipt_path,
        temporal_receipt_path=temporal_receipt_path,
        temporal_config_path=temporal_config,
        temporal_repo_root=REPO_ROOT,
    )
    assert len(context.tasks) == 1
    assert context.tasks[0].source_rich_identity == source_identity
