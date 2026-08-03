from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts/verify_openalex_canonical_work_time_spine_downstream_benchmark_v1.py"
)
SPEC = importlib.util.spec_from_file_location("_downstream_benchmark_verifier_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)
FROZEN_EXECUTED_VERIFIER_SHA256 = (
    "d1f9468f528034e64c20dca5b8b05f72ea5913f1c0571ab972de7c8503b6e0d3"
)


def _private(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def _write(path: Path, payload: bytes) -> None:
    _private(path.parent)
    path.write_bytes(payload)
    path.chmod(0o600)


def _write_json(path: Path, value: Any) -> None:
    _write(path, VERIFIER._published_json_bytes(value))


def _hash(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _records() -> np.ndarray[Any, Any]:
    records = np.empty(8, dtype=VERIFIER.RECORD_DTYPE)
    records["work_id_numeric"] = np.arange(8, dtype=np.uint64)
    records["publication_day"] = np.arange(18_262, 18_270, dtype=np.int32)
    records["publication_year"] = 2020
    records["temporal_disposition"] = 2
    records["reserved"] = 0
    return records


def _write_parquet(path: Path, records: np.ndarray[Any, Any]) -> None:
    _private(path.parent)
    table = pa.Table.from_arrays(
        [
            pa.array(records["work_id_numeric"], type=pa.uint64()),
            pa.array(records["publication_day"], type=pa.int32()),
            pa.array(records["publication_year"], type=pa.uint16()),
            pa.array(records["temporal_disposition"], type=pa.uint8()),
        ],
        schema=VERIFIER.ARROW_SCHEMA,
    )
    with path.open("wb") as handle:
        writer = pq.ParquetWriter(
            handle,
            VERIFIER.ARROW_SCHEMA,
            version="2.6",
            compression="NONE",
            use_dictionary=False,
            write_statistics=False,
            data_page_version="1.0",
            write_batch_size=262_144,
            use_byte_stream_split=False,
            write_page_index=False,
            write_page_checksum=False,
            store_schema=False,
            sorting_columns=[],
        )
        writer.write_table(table, row_group_size=262_144)
        writer.close()
    path.chmod(0o600)


def _pilot(root: Path, records: np.ndarray[Any, Any]) -> tuple[str, str, int]:
    _private(root)
    staging_relative = "staging/main/pilot.bin"
    staging = records.tobytes()
    staging_sha = hashlib.sha256(staging).hexdigest()
    _write(root / staging_relative, staging)
    task_fp = "1" * 64
    report: dict[str, Any] = {
        "status": VERIFIER.PILOT_STATUS,
        "task_fingerprint_sha256": task_fp,
        "scope": {
            "execution_scope": VERIFIER.PILOT_SCOPE,
            "production_status": VERIFIER.PRODUCTION_NOT_RUN,
            "retained_rows": len(records),
        },
        "provenance": {
            "tool_sha256": VERIFIER.RUNNER_SHA256,
            "config_sha256": VERIFIER.CONFIG_SHA256,
        },
    }
    report["provenance"]["result_fingerprint_sha256"] = VERIFIER._report_fingerprint(
        report
    )
    manifest = {
        "shards": [
            {
                "relative_path": staging_relative,
                "rows": len(records),
                "bytes": len(staging),
                "sha256": staging_sha,
            }
        ]
    }
    _write_json(root / "work_time_spine_pilot.json", report)
    _write_json(root / "staging_manifest.json", manifest)
    _write(root / "audit.lock", b"")
    artifact_paths = {
        staging_relative,
        "work_time_spine_pilot.json",
        "staging_manifest.json",
    }
    artifacts = {}
    for relative in sorted(artifact_paths):
        digest, size = _hash(root / relative)
        artifacts[relative] = {
            "relative_path": relative,
            "bytes": size,
            "sha256": digest,
        }
    receipt = {
        "scope": VERIFIER.PILOT_SCOPE,
        "production_status": VERIFIER.PRODUCTION_NOT_RUN,
        "production_component_pass": False,
        "task_fingerprint_sha256": task_fp,
        "result_fingerprint_sha256": report["provenance"][
            "result_fingerprint_sha256"
        ],
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }
    _write_json(root / "work_time_spine_pilot.outputs.json", receipt)
    for directory in [root, *[path for path in root.rglob("*") if path.is_dir()]]:
        directory.chmod(0o700)
    receipt_sha, _ = _hash(root / "work_time_spine_pilot.outputs.json")
    return staging_sha, receipt_sha, len(artifacts)


def _preflight() -> dict[str, Any]:
    return {
        "observed_at": "2026-07-21T00:00:00+09:00",
        "available_disk_bytes": 10**12,
        "available_ram_bytes": 10**12,
        "nofile_soft_limit": 16_384,
        "nofile_hard_limit": 16_384,
        "required_disk_bytes": 36_366_736_928,
        "required_available_ram_bytes": 9_685_696_512,
        "required_nofile_soft_limit": 8_192,
        "bounded_preflight_passed": True,
        "production_preflight_passed": False,
        "source_drive_health_not_applicable_no_source_opened": True,
        "quiet_window_independently_verified": False,
        "operator_paths_redacted": True,
        "atomic_rename_fsync_probe": {
            "status": "PASS_BOUNDED_SCRATCH_ATOMIC_RENAME_FSYNC",
            "wall_seconds": 0.001,
        },
    }


def _spool_payloads(records: np.ndarray[Any, Any]) -> list[np.ndarray[Any, Any]]:
    return [
        np.ascontiguousarray(records[records["work_id_numeric"] % 4 == partition_id])
        for partition_id in range(4)
    ]


def _spool_tree(payloads: list[np.ndarray[Any, Any]]) -> str:
    contracts = []
    for partition_id, records in enumerate(payloads):
        payload = records.tobytes()
        contracts.append(
            {
                "partition_id": partition_id,
                "relative_path": (
                    f".work_time_spine_tmp/spool/part-{partition_id:04d}.bin"
                ),
                "rows": len(records),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return hashlib.sha256(VERIFIER._canonical_json_bytes(contracts)).hexdigest()


def _repeat(
    root: Path,
    repeat_index: int,
    records: np.ndarray[Any, Any],
    pilot_staging_sha: str,
    task_fp: str,
    runtime_fp: dict[str, str],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    _private(root)
    shard_records = np.array_split(records, 2)
    for index, shard in enumerate(shard_records):
        logical = f"benchmark-shard-{index:04d}"
        token = hashlib.sha256(logical.encode()).hexdigest()
        staging_relative = f"staging/main/{token}.bin"
        payload = shard.tobytes()
        raw_sha = hashlib.sha256(payload).hexdigest()
        _write(root / staging_relative, payload)
        _write_json(
            root / f"checkpoints/scan/main/{token}.json",
            {
                "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_scan_shard.v1",
                "benchmark_shard_index": index,
                "relative_path": logical,
                "staging_shard_relative_path": staging_relative,
                "staging_shard_rows": len(shard),
                "staging_shard_bytes": len(payload),
                "staging_shard_sha256": raw_sha,
                "row_level_disposition_payload_sha256": VERIFIER._framed_digest(
                    shard
                ),
                "operator_paths_redacted": True,
            },
        )
    payloads = _spool_payloads(records)
    spool_tree = _spool_tree(payloads)
    benchmark_task_fp = hashlib.sha256(
        VERIFIER._canonical_json_bytes(
            {
                "schema": VERIFIER.BENCHMARK_SCHEMA,
                "repeat_index": repeat_index,
                "pilot_task_fingerprint_sha256": task_fp,
            }
        )
    ).hexdigest()
    scan_tree = hashlib.sha256(
        VERIFIER._canonical_json_bytes(
            {
                "benchmark_spool_partition_tree_sha256": spool_tree,
                "repeat_index": repeat_index,
            }
        )
    ).hexdigest()
    partition_contracts = []
    for partition_id, partition_records in enumerate(payloads):
        parquet_relative = f"spine/part-{partition_id:04d}.parquet"
        _write_parquet(root / parquet_relative, partition_records)
        parquet_sha, parquet_bytes = _hash(root / parquet_relative)
        dispositions = {name: 0 for name in VERIFIER.DISPOSITION_NAMES.values()}
        dispositions[VERIFIER.DISPOSITION_NAMES[2]] = len(partition_records)
        w0 = {name: 0 for name in VERIFIER.DISPOSITION_NAMES.values()}
        if partition_id == 0:
            w0[VERIFIER.DISPOSITION_NAMES[2]] = 1
        record_sha = VERIFIER._framed_digest(partition_records)
        checkpoint = {
            "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_partition_checkpoint.v1",
            "task_fingerprint_sha256": benchmark_task_fp,
            "runtime_fingerprint": runtime_fp,
            "scan_checkpoint_tree_sha256": scan_tree,
            "staging_shard_tree_sha256": spool_tree,
            "partition_id": partition_id,
            "partition_formula": "work_id_numeric_mod_4096",
            "output_relative_path": parquet_relative,
            "output_rows": len(partition_records),
            "minimum_work_id_numeric": int(partition_records["work_id_numeric"][0]),
            "maximum_work_id_numeric": int(partition_records["work_id_numeric"][-1]),
            "output_schema": VERIFIER.SCHEMA_CONTRACT,
            "output_schema_sha256": hashlib.sha256(
                VERIFIER._canonical_json_bytes(VERIFIER.SCHEMA_CONTRACT)
            ).hexdigest(),
            "pre_parquet_sorted_record_sha256": record_sha,
            "decoded_parquet_record_sha256": record_sha,
            "disposition_counts": dispositions,
            "w_zero_disposition_counts": w0,
            "publication_year_extrema": {"minimum": 2020, "maximum": 2020},
            "nullability_counts": {
                "publication_day_null_rows": 0,
                "publication_year_null_rows": 0,
            },
            "output_bytes": parquet_bytes,
            "output_sha256": parquet_sha,
            "strictly_increasing_ids": True,
            "operator_paths_redacted": True,
        }
        checkpoint["checkpoint_sha256"] = hashlib.sha256(
            VERIFIER._canonical_json_bytes(checkpoint)
        ).hexdigest()
        _write_json(
            root / f"checkpoints/partition/part-{partition_id:04d}.json",
            checkpoint,
        )
        partition_contracts.append(
            {
                "partition_id": partition_id,
                "rows": len(partition_records),
                "bytes": parquet_bytes,
                "sha256": parquet_sha,
                "record_sha256": record_sha,
            }
        )
    statuses = {
        "demultiplex": "PASS_BOUNDED_4096_WAY_DEMULTIPLEX_INTEGRITY",
        "representative": "PASS_BOUNDED_PILOT_OCCUPANCY_SINGLE_RUN_PARQUET_INTEGRITY",
    }
    controls = {
        "benchmark_plan.json": {
            "schema_version": VERIFIER.BENCHMARK_SCHEMA,
            "status": "BOUNDED_NONPROMOTABLE_PLAN",
            "repeat_index": repeat_index,
            "scenarios": ["D4096", "R4096", "H9185_BOUNDED_BYTES"],
            "production_execution": VERIFIER.PRODUCTION_NOT_RUN,
            "operator_paths_redacted": True,
        },
        "preflight_snapshot.json": preflight,
        "run_state.json": {
            "schema_version": f"{VERIFIER.BENCHMARK_SCHEMA}.state",
            "status": "BOUNDED_REPEAT_COMPLETE_NO_PRODUCTION_RECEIPT",
            "repeat_index": repeat_index,
            "production_execution": VERIFIER.PRODUCTION_NOT_RUN,
            "operator_paths_redacted": True,
        },
        "staging_manifest.json": {
            "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_staging_manifest.v1",
            "repeat_index": repeat_index,
            "shard_count": 2,
            "rows": len(records),
            "bytes": len(records) * VERIFIER.RECORD_BYTES,
            "spool_partition_tree_sha256": spool_tree,
            "operator_paths_redacted": True,
        },
        "spine_manifest.json": {
            "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_spine_manifest.v1",
            "repeat_index": repeat_index,
            "partition_count": 4,
            "rows": len(records),
            "bytes": sum(item["bytes"] for item in partition_contracts),
            "partition_contract_tree_sha256": hashlib.sha256(
                VERIFIER._canonical_json_bytes(partition_contracts)
            ).hexdigest(),
            "operator_paths_redacted": True,
        },
        "benchmark_report.json": {
            "schema_version": f"{VERIFIER.BENCHMARK_SCHEMA}.repeat_report",
            "status": VERIFIER.BENCHMARK_STATUS,
            "repeat_index": repeat_index,
            "demultiplex_status": statuses["demultiplex"],
            "representative_status": statuses["representative"],
            "production_component_pass": False,
            "programmatic_parent_of_full_receipt": False,
            "operator_paths_redacted": True,
        },
    }
    for relative, value in controls.items():
        _write_json(root / relative, value)
    _write(
        root / "benchmark_report.md",
        (
            "# Bounded downstream benchmark repeat\n\n"
            f"Repeat: {repeat_index}\n\n"
            "Production execution: PRODUCTION_NOT_RUN\n\n"
            "This scratch surface is operator-local and nonpromotable.\n"
        ).encode(),
    )
    for directory in [root, *[path for path in root.rglob("*") if path.is_dir()]]:
        directory.chmod(0o700)
    paths = VERIFIER._expected_paths()
    contracts = []
    for relative in paths:
        digest, size = _hash(root / relative)
        contracts.append(
            {"relative_path": relative, "bytes": size, "sha256": digest}
        )
    artifact_tree = hashlib.sha256(
        VERIFIER._canonical_json_bytes(contracts)
    ).hexdigest()
    artifact_bytes = sum(item["bytes"] for item in contracts)
    counts = [len(item) for item in payloads]
    occupancy = {
        "partition_count": 4,
        "nonempty_partition_count": 4,
        "empty_partition_count": 0,
        "all_partitions": VERIFIER._distribution([float(value) for value in counts]),
        "nonempty_partitions": VERIFIER._distribution(
            [float(value) for value in counts]
        ),
        "partition_rows_sha256": hashlib.sha256(
            VERIFIER._canonical_json_bytes(counts)
        ).hexdigest(),
    }
    rate = float(100 + repeat_index)
    return {
        "repeat_index": repeat_index,
        "status": "PASS_BOUNDED_REPEAT_INTEGRITY",
        "preparation": {
            "wall_seconds": 0.01,
            "shard_count": 2,
            "rows": len(records),
            "bytes": len(records) * VERIFIER.RECORD_BYTES,
            "source_sha256": pilot_staging_sha,
            "excluded_from_downstream_phase_timings": True,
        },
        "demultiplex": {
            "scenario": "D4096",
            "status": statuses["demultiplex"],
            "rows": len(records),
            "logical_input_bytes": len(records) * VERIFIER.RECORD_BYTES,
            "logical_spool_bytes": len(records) * VERIFIER.RECORD_BYTES,
            "rows_per_second_wall": rate,
            "occupancy": occupancy,
            "kernel_metrics": {},
            "resources": {},
            "full_corpus_sustained_rate_measured": False,
        },
        "representative_single_run": {
            "scenario": "R4096",
            "status": statuses["representative"],
            "partition_count": 4,
            "rows": len(records),
            "parquet_bytes": sum(item["bytes"] for item in partition_contracts),
            "run_count_total": 4,
            "run_count_maximum_per_partition": 1,
            "multi_run_partitions": 0,
            "single_run_partitions": 4,
            "zero_run_partitions": 0,
            "partition_wall_seconds": {},
            "rows_per_second_wall": rate + 10,
            "kernel_metrics": {},
            "resources": {},
            "full_like_partition_occupancy_measured": False,
            "python_multi_run_heap_merge_measured": False,
        },
        "bounded_artifact_surface": {
            "scenario": "H9185_BOUNDED_BYTES",
            "status": "PASS_EXACT_9185_ENTRY_TOPOLOGY_AND_HASH_SWEEPS_BOUNDED_BYTES",
            "artifact_count": 19,
            "topology_formula": "493_staging+493_scan_metadata+4096_parquet+4096_partition_metadata+7_control",
            "sweep_count": 1,
            "sweeps": [
                {
                    "sweep_index": 1,
                    "artifact_count": 19,
                    "hashed_bytes": artifact_bytes,
                    "artifact_contract_tree_sha256": artifact_tree,
                    "artifacts_per_second": rate + 20,
                    "bytes_per_second": 1000.0,
                    "resources": {},
                }
            ],
            "large_full_byte_surface_measured": False,
            "production_finalizer_or_receipt_called": False,
            "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
        },
        "scratch_surface_retained_for_operator_review": True,
        "operator_paths_redacted": True,
    }


def _surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path]:
    monkeypatch.setattr(VERIFIER, "SHARD_COUNT", 2)
    monkeypatch.setattr(VERIFIER, "PARTITION_COUNT", 4)
    monkeypatch.setattr(VERIFIER, "TOTAL_ROWS", 8)
    monkeypatch.setattr(VERIFIER, "EXACT_ARTIFACT_COUNT", 19)
    records = _records()
    pilot = _private(tmp_path / "pilot")
    pilot_staging_sha, receipt_sha, artifact_count = _pilot(pilot, records)
    pilot_tree = VERIFIER._tree_identity_digest(pilot)
    scratch = _private(tmp_path / "scratch")
    preflight = _preflight()
    task_fp = "1" * 64
    runtime_fp = {"runtime": "synthetic"}
    repeats = [
        _repeat(
            _private(scratch / f"repeat-{index:02d}"),
            index,
            records,
            pilot_staging_sha,
            task_fp,
            runtime_fp,
            preflight,
        )
        for index in range(1, 4)
    ]
    aggregate = {
        "repeat_count": 3,
        "demultiplex_rows_per_second": VERIFIER._distribution(
            [item["demultiplex"]["rows_per_second_wall"] for item in repeats]
        ),
        "representative_rows_per_second": VERIFIER._distribution(
            [
                item["representative_single_run"]["rows_per_second_wall"]
                for item in repeats
            ]
        ),
        "h9185_artifacts_per_second": VERIFIER._distribution(
            [
                item["bounded_artifact_surface"]["sweeps"][0][
                    "artifacts_per_second"
                ]
                for item in repeats
            ]
        ),
        "one_sided_95_percent_lower_confidence_bound_computed": False,
        "minimum_observed_rates_are_not_adopted_capacity_floors": True,
        "independent_quiet_window_sessions": 0,
    }
    result: dict[str, Any] = {
        "schema_version": VERIFIER.BENCHMARK_SCHEMA,
        "status": VERIFIER.BENCHMARK_STATUS,
        "started_at": "2026-07-21T00:00:00+09:00",
        "completed_at": "2026-07-21T00:01:00+09:00",
        "implementation_binding": {
            "runner_sha256": VERIFIER.RUNNER_SHA256,
            "config_sha256": VERIFIER.CONFIG_SHA256,
            "independent_verifier_sha256": VERIFIER.PILOT_VERIFIER_SHA256,
            "benchmark_tool_sha256": "2" * 64,
            "loaded_code_attestation": False,
            "private_downstream_api_used": True,
            "production_finalizer_called": False,
            "production_receipt_builder_called": False,
        },
        "runtime": {
            "python_version": "synthetic",
            "python_implementation": "synthetic",
            "platform_machine": "synthetic",
            "runner_runtime_fingerprint": runtime_fp,
        },
        "pilot_binding": {
            "receipt_sha256": receipt_sha,
            "task_fingerprint_sha256": task_fp,
            "result_fingerprint_sha256": json.loads(
                (pilot / "work_time_spine_pilot.json").read_text()
            )["provenance"]["result_fingerprint_sha256"],
            "staging_sha256": pilot_staging_sha,
            "staging_rows": 8,
            "staging_bytes": 8 * VERIFIER.RECORD_BYTES,
            "independent_verifier_status": "PASS_READ_ONLY_INDEPENDENT_OUTPUT_VERIFICATION",
            "independent_verifier_artifact_count": artifact_count,
            "tree_before": pilot_tree,
            "tree_after": pilot_tree,
        },
        "preflight": preflight,
        "benchmark_scope": {
            "scenarios": ["D4096", "R4096", "H9185_BOUNDED_BYTES"],
            "repeat_count": 3,
            "artifact_sweeps_per_repeat": 1,
            "source_hdd_opened": False,
            "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
            "full_like_partition_occupancy_measured": False,
            "multi_run_heap_merge_measured": False,
            "full_byte_finalization_surface_measured": False,
            "independent_quiet_window_sessions": 0,
        },
        "repeats": repeats,
        "aggregate": aggregate,
        "pilot_output_tree_unchanged": True,
        "production_execution_status": VERIFIER.PRODUCTION_NOT_RUN,
        "production_component_pass": False,
        "pass_g1": False,
        "full_authorization_sufficient": False,
        "downstream_capacity_gate_passed": False,
        "programmatic_parent_of_full_receipt": False,
        "operator_acceptance_status": "NOT_RECORDED",
        "strongest_permitted_interpretation": "PARTIAL_BOUNDED_FIXED_OVERHEAD_SINGLE_RUN_AND_EXACT_ENTRY_COUNT_EVIDENCE_ONLY",
        "operator_paths_redacted": True,
    }
    result["result_fingerprint_sha256"] = VERIFIER._result_fingerprint(result)
    result_root = _private(tmp_path / "result")
    result_path = result_root / "openalex_g1_canonical_work_time_spine_downstream_benchmark.json"
    _write_json(result_path, result)
    return result_path, scratch, pilot


def test_verifier_source_has_no_benchmark_or_runner_dependency() -> None:
    assert (
        hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        == FROZEN_EXECUTED_VERIFIER_SHA256
    )
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module or "")
    assert "importlib" not in imports
    assert not any("audit_openalex" in name for name in imports)
    assert not any("benchmark_openalex" in name for name in imports)


def test_synthetic_surface_verifies_with_read_only_opens_and_no_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_path, scratch, pilot = _surface(monkeypatch, tmp_path)
    before = {
        "result": VERIFIER._identity(result_path.lstat()),
        "scratch": VERIFIER._directory_snapshot(scratch),
        "pilot": VERIFIER._directory_snapshot(pilot),
    }
    real_open = VERIFIER.os.open
    flags_seen: list[int] = []

    def read_only_open(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        flags_seen.append(flags)
        forbidden = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        assert flags & forbidden == 0
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(VERIFIER.os, "open", read_only_open)
    verification = VERIFIER.verify_benchmark(
        benchmark_result=result_path,
        scratch_root=scratch,
        pilot_output_root=pilot,
    )
    assert flags_seen
    assert verification["status"] == VERIFIER.VERIFICATION_STATUS
    assert verification["production_component_pass"] is False
    assert verification["downstream_capacity_gate_passed"] is False
    assert verification["scratch_tree_unchanged"] is True
    assert verification["pilot_tree_unchanged"] is True
    assert VERIFIER._identity(result_path.lstat()) == before["result"]
    assert VERIFIER._directory_snapshot(scratch) == before["scratch"]
    assert VERIFIER._directory_snapshot(pilot) == before["pilot"]


def test_synthetic_staging_tamper_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_path, scratch, pilot = _surface(monkeypatch, tmp_path)
    staging = next((scratch / "repeat-02/staging/main").glob("*.bin"))
    payload = bytearray(staging.read_bytes())
    payload[-1] ^= 1
    staging.write_bytes(payload)
    staging.chmod(0o600)
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.verify_benchmark(
            benchmark_result=result_path,
            scratch_root=scratch,
            pilot_output_root=pilot,
        )
