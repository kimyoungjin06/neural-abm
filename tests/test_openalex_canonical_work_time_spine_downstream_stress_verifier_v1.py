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
SCRIPT = REPO_ROOT / "scripts/verify_openalex_canonical_work_time_spine_downstream_stress_v1.py"
SPEC = importlib.util.spec_from_file_location("_stress_verifier_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)


def _mkdir(path: Path, mode: int = 0o700) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(mode)
    return path


def _write(path: Path, payload: bytes) -> None:
    _mkdir(path.parent)
    path.write_bytes(payload)
    path.chmod(0o600)


def _write_json(path: Path, value: Any) -> None:
    _write(path, VERIFIER._published_json_bytes(value))


def _hash(path: Path) -> tuple[str, int]:
    payload = path.read_bytes()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _pilot_records() -> np.ndarray[Any, Any]:
    records = np.empty(8, dtype=VERIFIER.RECORD_DTYPE)
    records["work_id_numeric"] = np.arange(8, dtype=np.uint64)
    dispositions = np.array([0, 1, 2, 3, 4, 5, 2, 0], dtype=np.uint8)
    records["temporal_disposition"] = dispositions
    records["publication_day"] = np.where(dispositions == 0, VERIFIER.NULL_DAY, 18_262).astype(np.int32)
    records["publication_year"] = np.where(dispositions == 0, 0, 2020).astype(np.uint16)
    records["reserved"] = 0
    return records


def _records_for_q(pilot: np.ndarray[Any, Any], q_values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    records = np.empty(len(q_values), dtype=VERIFIER.RECORD_DTYPE)
    records["work_id_numeric"] = VERIFIER._expected_ids(q_values)
    expected = pilot[(q_values % len(pilot)).astype(np.int64)]
    for field in ("publication_day", "publication_year", "temporal_disposition", "reserved"):
        records[field] = expected[field]
    return records


def _record_digest(records: np.ndarray[Any, Any]) -> str:
    digest = VERIFIER._new_record_digest(len(records))
    VERIFIER._update_record_digest(digest, records)
    return digest.hexdigest()


def _write_parquet(path: Path, records: np.ndarray[Any, Any]) -> None:
    _mkdir(path.parent)
    nulls = records["temporal_disposition"] == 0
    table = pa.Table.from_arrays(
        [
            pa.array(records["work_id_numeric"], type=pa.uint64()),
            pa.array(records["publication_day"], mask=nulls, type=pa.int32()),
            pa.array(records["publication_year"], mask=nulls, type=pa.uint16()),
            pa.array(records["temporal_disposition"], type=pa.uint8()),
        ],
        schema=VERIFIER.ARROW_SCHEMA,
    )
    with path.open("wb") as handle:
        writer = pq.ParquetWriter(
            handle, VERIFIER.ARROW_SCHEMA, version="2.6", compression="NONE",
            use_dictionary=False, write_statistics=False, data_page_version="1.0",
            write_batch_size=262_144, use_byte_stream_split=False,
            write_page_index=False, write_page_checksum=False, store_schema=False,
            sorting_columns=[],
        )
        writer.write_table(table, row_group_size=262_144)
        writer.close()
    path.chmod(0o600)


def _build_pilot(root: Path, records: np.ndarray[Any, Any]) -> tuple[str, str, str]:
    _mkdir(root)
    for relative in (
        "checkpoints", "checkpoints/partition", "checkpoints/scan",
        "checkpoints/scan/main", "spine", "staging", "staging/main",
    ):
        _mkdir(root / relative)
    staging_relative = "staging/main/pilot.bin"
    payload = records.tobytes()
    staging_sha = hashlib.sha256(payload).hexdigest()
    _write(root / staging_relative, payload)
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
    report["provenance"]["result_fingerprint_sha256"] = VERIFIER._pilot_report_fingerprint(report)
    manifest = {"shards": [{"relative_path": staging_relative, "rows": len(records), "bytes": len(payload), "sha256": staging_sha}]}
    _write_json(root / "work_time_spine_pilot.json", report)
    _write_json(root / "staging_manifest.json", manifest)
    _write(root / "audit.lock", b"")
    artifacts = {}
    for relative in ("staging/main/pilot.bin", "staging_manifest.json", "work_time_spine_pilot.json"):
        digest, size = _hash(root / relative)
        artifacts[relative] = {"relative_path": relative, "bytes": size, "sha256": digest}
    receipt = {"artifacts": artifacts, "artifact_count": len(artifacts)}
    _write_json(root / "work_time_spine_pilot.outputs.json", receipt)
    for directory in [root, *[item for item in root.rglob("*") if item.is_dir()]]:
        directory.chmod(0o700)
    receipt_sha, _ = _hash(root / "work_time_spine_pilot.outputs.json")
    return staging_sha, receipt_sha, task_fp


def _build_first(root: Path, pilot_receipt: str, pilot_staging: str) -> tuple[Path, str, str]:
    _mkdir(root)
    result: dict[str, Any] = {
        "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark.v1",
        "production_execution_status": VERIFIER.PRODUCTION_NOT_RUN,
        "production_component_pass": False,
        "pass_g1": False,
        "full_authorization_sufficient": False,
        "downstream_capacity_gate_passed": False,
        "programmatic_parent_of_full_receipt": False,
        "pilot_binding": {"receipt_sha256": pilot_receipt, "staging_sha256": pilot_staging},
    }
    result["result_fingerprint_sha256"] = VERIFIER._result_fingerprint(result)
    path = root / "openalex_g1_canonical_work_time_spine_downstream_benchmark.json"
    _write_json(path, result)
    digest, _ = _hash(path)
    return path, digest, result["result_fingerprint_sha256"]


def _stats(records: np.ndarray[Any, Any]) -> dict[str, Any]:
    counts = {name: 0 for name in VERIFIER.DISPOSITION_NAMES.values()}
    raw = np.bincount(records["temporal_disposition"], minlength=len(counts))
    for code, name in VERIFIER.DISPOSITION_NAMES.items():
        counts[name] = int(raw[code])
    nulls = int(np.count_nonzero(records["temporal_disposition"] == 0))
    nonnull = records["temporal_disposition"] != 0
    years = records["publication_year"][nonnull]
    return {
        "disposition_counts": counts,
        "w_zero_disposition_counts": {name: 0 for name in counts},
        "nullability_counts": {"publication_day_null_rows": nulls, "publication_year_null_rows": nulls},
        "publication_year_extrema": {"minimum": int(np.min(years)), "maximum": int(np.max(years))},
    }


def _resources(wall_seconds: float = 1.0) -> dict[str, Any]:
    return {
        "wall_seconds": wall_seconds,
        "process_usage_delta": {
            "user_seconds": 0.1,
            "system_seconds": 0.1,
            "minor_page_faults": 0,
            "major_page_faults": 0,
            "voluntary_context_switches": 0,
            "involuntary_context_switches": 0,
            "block_inputs": 0,
            "block_outputs": 0,
        },
        "proc_io_delta": {
            "rchar": 0,
            "wchar": 0,
            "syscr": 0,
            "syscw": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "cancelled_write_bytes": 0,
        },
        "sampling_interval_seconds": 0.005,
        "peak_rss_kib_sampled": 1024,
        "peak_open_file_descriptors_sampled": 8,
        "peak_hwm_kib_observed": 1024,
    }


def _throughput(
    calls: int,
    units: int,
    *,
    units_key: str,
    rate_key: str,
) -> dict[str, Any]:
    seconds = 0.0 if calls == 0 else 0.25
    rate = None if calls == 0 else round(units / seconds, 6)
    return {
        "calls": calls,
        units_key: units,
        "seconds": seconds,
        rate_key: rate,
    }


def _writer_metric(write_calls: int, close_calls: int, rows: int) -> dict[str, Any]:
    seconds = 0.0 if write_calls == 0 else 0.25
    return {
        "write_calls": write_calls,
        "close_calls": close_calls,
        "rows": rows,
        "seconds": seconds,
        "rows_per_second": None if write_calls == 0 else round(rows / seconds, 6),
    }


def _fsync_metric(calls: int) -> dict[str, Any]:
    latency = 0.001
    return {
        "calls": calls,
        "total_seconds": round(calls * latency, 6),
        "latency_seconds": {
            "count": calls,
            "minimum": latency,
            "median": latency,
            "maximum": latency,
        },
    }


def _coarse_metrics(
    scenario: Any,
    *,
    phase: str,
    output_bytes: int,
) -> dict[str, Any]:
    row_groups = len(VERIFIER._expected_parquet_row_group_rows(scenario.rows))
    if phase == "sort":
        return {
            "numpy_stable_argsort": _throughput(
                scenario.expected_runs,
                scenario.rows,
                units_key="rows",
                rate_key="rows_per_second",
            ),
            "fsync": _fsync_metric(scenario.expected_runs + 1),
            "parquet_writer": _writer_metric(0, 0, 0),
            "arrow_table_conversion": _throughput(
                0, 0, units_key="rows", rate_key="rows_per_second",
            ),
            "decoded_validation": _throughput(
                0, 0, units_key="rows", rate_key="rows_per_second",
            ),
            "parquet_sha256": _throughput(
                0, 0, units_key="bytes", rate_key="bytes_per_second",
            ),
            "merge_and_parquet_before_decoded_validation_seconds": None,
            "heap_operation_timing_proxy_installed": False,
            "run_reader_proxy_installed": False,
        }
    return {
        "numpy_stable_argsort": _throughput(
            0, 0, units_key="rows", rate_key="rows_per_second",
        ),
        "fsync": _fsync_metric(1),
        "parquet_writer": _writer_metric(row_groups, 1, scenario.rows),
        "arrow_table_conversion": _throughput(
            row_groups,
            scenario.rows,
            units_key="rows",
            rate_key="rows_per_second",
        ),
        "decoded_validation": _throughput(
            1,
            scenario.rows,
            units_key="rows",
            rate_key="rows_per_second",
        ),
        "parquet_sha256": _throughput(
            1,
            output_bytes,
            units_key="bytes",
            rate_key="bytes_per_second",
        ),
        "merge_and_parquet_before_decoded_validation_seconds": 0.5,
        "heap_operation_timing_proxy_installed": False,
        "run_reader_proxy_installed": False,
    }


def _build_scenario(
    root: Path,
    scenario: Any,
    repeat_index: int,
    pilot: np.ndarray[Any, Any],
    runtime: dict[str, str],
) -> dict[str, Any]:
    _mkdir(root)
    _mkdir(root / ".work_time_spine_tmp", 0o775)
    _mkdir(root / ".work_time_spine_tmp/runs")
    _mkdir(root / f".work_time_spine_tmp/runs/part-{VERIFIER.PARTITION_ID:04d}")
    _mkdir(root / ".work_time_spine_tmp/sorted")
    _mkdir(root / ".work_time_spine_tmp/spool")
    _mkdir(root / "spine")
    spool_blocks = []
    for run_index, run_rows in enumerate(VERIFIER._expected_run_rows(scenario)):
        q_values = np.uint64(scenario.expected_runs) * np.arange(run_rows - 1, -1, -1, dtype=np.uint64) + np.uint64(run_index)
        spool_blocks.append(_records_for_q(pilot, q_values).tobytes())
    spool_payload = b"".join(spool_blocks)
    spool_path = root / f".work_time_spine_tmp/spool/part-{VERIFIER.PARTITION_ID:04d}.bin"
    _write(spool_path, spool_payload)
    spool_sha = hashlib.sha256(spool_payload).hexdigest()
    run_contracts = []
    for run_index, rows in enumerate(VERIFIER._expected_run_rows(scenario)):
        q_values = np.uint64(run_index) + np.uint64(scenario.expected_runs) * np.arange(rows, dtype=np.uint64)
        run_payload = _records_for_q(pilot, q_values).tobytes()
        run_path = root / f".work_time_spine_tmp/runs/part-{VERIFIER.PARTITION_ID:04d}/run-{run_index:04d}.bin"
        _write(run_path, run_payload)
        run_contracts.append({
            "run_index": run_index, "rows": rows, "bytes": len(run_payload),
            "sha256": hashlib.sha256(run_payload).hexdigest(), "q_start": run_index,
            "q_stop": scenario.expected_runs * (rows - 1) + run_index,
            "q_stride": scenario.expected_runs,
        })
    q_values = np.arange(scenario.rows, dtype=np.uint64)
    sorted_records = _records_for_q(pilot, q_values)
    parquet_relative = f"spine/part-{VERIFIER.PARTITION_ID:04d}.parquet"
    _write_parquet(root / parquet_relative, sorted_records)
    parquet_sha, parquet_bytes = _hash(root / parquet_relative)
    record_sha = _record_digest(sorted_records)
    record_stats = _stats(sorted_records)
    task_fp = hashlib.sha256(VERIFIER._canonical_json_bytes({
        "schema": VERIFIER.SCHEMA, "repeat_index": repeat_index,
        "scenario": scenario.scenario_id,
        "first_benchmark_sha256": VERIFIER.FIRST_BENCHMARK_SHA256,
    })).hexdigest()
    scan_tree = hashlib.sha256(VERIFIER._canonical_json_bytes({"scenario": scenario.scenario_id, "repeat": repeat_index})).hexdigest()
    checkpoint = {
        "schema_version": "openalex_g1_canonical_work_time_spine_downstream_stress_partition_checkpoint.v1",
        "task_fingerprint_sha256": task_fp, "runtime_fingerprint": runtime,
        "scan_checkpoint_tree_sha256": scan_tree, "staging_shard_tree_sha256": spool_sha,
        "partition_id": VERIFIER.PARTITION_ID, "partition_formula": "work_id_numeric_mod_4096",
        "output_relative_path": parquet_relative, "output_rows": scenario.rows,
        "minimum_work_id_numeric": int(sorted_records["work_id_numeric"][0]),
        "maximum_work_id_numeric": int(sorted_records["work_id_numeric"][-1]),
        "output_schema": VERIFIER.SCHEMA_CONTRACT,
        "output_schema_sha256": hashlib.sha256(VERIFIER._canonical_json_bytes(VERIFIER.SCHEMA_CONTRACT)).hexdigest(),
        "pre_parquet_sorted_record_sha256": record_sha,
        "decoded_parquet_record_sha256": record_sha,
        **record_stats,
        "output_bytes": parquet_bytes, "output_sha256": parquet_sha,
        "strictly_increasing_ids": True, "operator_paths_redacted": True,
    }
    checkpoint["checkpoint_sha256"] = hashlib.sha256(VERIFIER._canonical_json_bytes(checkpoint)).hexdigest()
    _write_json(root / "stress_checkpoint.json", checkpoint)
    rate = float(scenario.rows)
    return {
        "scenario_id": scenario.scenario_id, "role": scenario.role, "rows": scenario.rows,
        "partition_id": VERIFIER.PARTITION_ID, "sort_chunk_rows": VERIFIER.SORT_CHUNK_ROWS,
        "expected_runs": scenario.expected_runs, "observed_runs": scenario.expected_runs,
        "maximum_measured_fan_in": scenario.expected_runs,
        "preparation": {
            "rows": scenario.rows, "bytes": scenario.rows * VERIFIER.RECORD_BYTES,
            "wall_seconds": 0.01, "spool_sha256": spool_sha,
            "synthetic_namespace_base": VERIFIER.SYNTHETIC_NAMESPACE_BASE,
            "partition_id": VERIFIER.PARTITION_ID,
            "temporal_payload_source": "VERIFIED_PILOT_STAGING_CYCLIC_BY_Q",
            "excluded_from_kernel_timing": True,
        },
        "run_validation": {
            "run_count": scenario.expected_runs,
            "run_contract_tree_sha256": hashlib.sha256(VERIFIER._canonical_json_bytes(run_contracts)).hexdigest(),
            "runs": run_contracts, "balanced_fan_in": scenario.expected_runs > 1,
            "wall_seconds": 0.01, "excluded_from_kernel_timing": True,
            "prewarms_external_run_page_cache_before_merge": True,
        },
        "sort": {
            "wall_rows_per_second": rate,
            "resources": _resources(),
            "coarse_metrics": _coarse_metrics(
                scenario, phase="sort", output_bytes=parquet_bytes,
            ),
        },
        "merge_parquet_and_validation": {
            "wall_rows_per_second": rate,
            "resources": _resources(),
            "coarse_metrics": _coarse_metrics(
                scenario, phase="merge", output_bytes=parquet_bytes,
            ),
            "cache_condition": "CACHE_ASSISTED_BY_IMMEDIATELY_PRECEDING_FULL_RUN_VALIDATION",
            "derived_heap_push_count_not_observed": scenario.rows if scenario.expected_runs > 1 else 0,
            "derived_heap_pop_count_not_observed": scenario.rows if scenario.expected_runs > 1 else 0,
        },
        "output_rows": scenario.rows, "output_bytes": parquet_bytes,
        "output_sha256": parquet_sha, "record_sha256": record_sha,
        "integrity_status": "PASS_SYNTHETIC_STRESS_KERNEL_CLOSURE",
        "operator_paths_redacted": True,
    }


def _surface(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    scenarios = (
        VERIFIER.Scenario("RMEAN", 3, 1, "FULL_LIKE_MEAN_PARTITION_ROWS"),
        VERIFIER.Scenario("A2_BALANCED", 8, 2, "BALANCED_TWO_WAY_HEAP_MERGE"),
        VERIFIER.Scenario("A8_BALANCED", 32, 8, "BALANCED_EIGHT_WAY_HEAP_MERGE"),
    )
    monkeypatch.setattr(VERIFIER, "SCENARIOS", scenarios)
    monkeypatch.setattr(VERIFIER, "SORT_CHUNK_ROWS", 4)
    monkeypatch.setattr(VERIFIER, "GENERATION_BATCH_ROWS", 2)
    monkeypatch.setattr(VERIFIER, "PILOT_STAGING_ROWS", 8)
    pilot_records = _pilot_records()
    pilot_root = _mkdir(tmp_path / "pilot")
    staging_sha, receipt_sha, task_fp = _build_pilot(pilot_root, pilot_records)
    monkeypatch.setattr(VERIFIER, "PILOT_STAGING_SHA256", staging_sha)
    monkeypatch.setattr(VERIFIER, "PILOT_RECEIPT_SHA256", receipt_sha)
    first_path, first_sha, first_fp = _build_first(_mkdir(tmp_path / "first"), receipt_sha, staging_sha)
    monkeypatch.setattr(VERIFIER, "FIRST_BENCHMARK_SHA256", first_sha)
    monkeypatch.setattr(VERIFIER, "FIRST_BENCHMARK_FINGERPRINT", first_fp)
    runtime = {
        "pyarrow_version": "synthetic",
        "parquet_cpp_version": "synthetic",
        "numpy_version": "synthetic",
        "python_implementation": "synthetic",
        "python_version": "synthetic",
        "platform_machine": "synthetic",
    }
    scratch = _mkdir(tmp_path / "scratch")
    repeats = []
    for repeat_index in range(1, 4):
        repeat_root = _mkdir(scratch / f"repeat-{repeat_index:02d}")
        recorded = [
            _build_scenario(_mkdir(repeat_root / scenario.scenario_id), scenario, repeat_index, pilot_records, runtime)
            for scenario in scenarios
        ]
        repeats.append({"repeat_index": repeat_index, "scenarios": recorded, "status": "PASS_SYNTHETIC_STRESS_REPEAT", "operator_paths_redacted": True})
    by_scenario = {}
    for scenario in scenarios:
        observations = [next(item for item in repeat["scenarios"] if item["scenario_id"] == scenario.scenario_id) for repeat in repeats]
        by_scenario[scenario.scenario_id] = {
            "repeat_count": 3,
            "sort_wall_rows_per_second": VERIFIER._distribution([item["sort"]["wall_rows_per_second"] for item in observations]),
            "merge_wall_rows_per_second": VERIFIER._distribution([item["merge_parquet_and_validation"]["wall_rows_per_second"] for item in observations]),
            "output_bytes": VERIFIER._distribution([float(item["output_bytes"]) for item in observations]),
        }
    result: dict[str, Any] = {
        "schema_version": VERIFIER.SCHEMA, "status": VERIFIER.STATUS,
        "started_at": "2026-07-21T00:00:00+09:00", "completed_at": "2026-07-21T00:01:00+09:00",
        "implementation_binding": {
            "stress_tool_sha256": VERIFIER.STRESS_TOOL_SHA256, "helper_tool_sha256": VERIFIER.HELPER_SHA256,
            "runner_sha256": VERIFIER.RUNNER_SHA256, "config_sha256": VERIFIER.CONFIG_SHA256,
            "pilot_verifier_sha256": VERIFIER.PILOT_VERIFIER_SHA256,
            "heap_operation_timing_proxy_installed": False, "run_reader_proxy_installed": False,
            "private_downstream_api_used": True, "production_partition_publisher_called": False,
            "production_finalizer_called": False, "production_receipt_builder_called": False,
        },
        "runtime": {
            "python_version": "synthetic",
            "python_implementation": "synthetic",
            "platform_machine": "synthetic",
            "runner_runtime_fingerprint": runtime,
        },
        "parent_bindings": {
            "first_benchmark_sha256": first_sha, "first_benchmark_fingerprint": first_fp,
            "pilot_receipt_sha256": receipt_sha, "pilot_staging_sha256": staging_sha,
            "pilot_task_fingerprint_sha256": task_fp,
            "first_benchmark_tree_unchanged": True, "pilot_tree_unchanged": True,
        },
        "preflight": {
            "required_disk_bytes": 36_366_736_928, "required_available_ram_bytes": 9_685_696_512,
            "required_nofile_soft_limit": 8_192, "bounded_preflight_passed": True,
            "available_disk_bytes": 36_366_736_928,
            "available_ram_bytes": 9_685_696_512,
            "nofile_soft_limit": 8_192,
            "nofile_hard_limit": 8_192,
            "observed_at": "2026-07-21T00:00:00+09:00",
            "production_preflight_passed": False, "source_drive_health_not_applicable_no_source_opened": True,
            "quiet_window_independently_verified": False, "operator_paths_redacted": True,
            "stress_kernel_minimum_free_disk_bytes": 4_294_967_296,
            "stress_kernel_minimum_available_ram_bytes": 2_147_483_648,
            "effective_required_free_disk_bytes": 36_366_736_928,
            "effective_required_available_ram_bytes": 9_685_696_512,
            "pinned_helper_preflight_is_enforced": True,
            "atomic_rename_fsync_probe": {"status": "PASS_BOUNDED_SCRATCH_ATOMIC_RENAME_FSYNC", "wall_seconds": 0.001},
        },
        "scenario_contracts": [{"scenario_id": item.scenario_id, "rows": item.rows, "expected_runs": item.expected_runs, "role": item.role} for item in scenarios],
        "repeats": repeats,
        "aggregate": {"by_scenario": by_scenario, "one_sided_95_percent_lower_confidence_bound_computed": False, "minimum_observed_rates_are_not_adopted_capacity_floors": True, "independent_quiet_window_sessions": 0},
        "full_like_per_partition_row_scale_measured": True,
        "balanced_multi_run_heap_branch_measured": True, "maximum_measured_fan_in": 8,
        "theoretical_470_run_heap_merge_measured": False,
        "actual_full_4096_partition_occupancy_profile_measured": False,
        "full_byte_finalization_surface_measured": False, "independent_verification_status": "PENDING",
        "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED", "independent_quiet_window_sessions": 0,
        "source_hdd_opened": False, "production_execution_status": VERIFIER.PRODUCTION_NOT_RUN,
        "production_component_pass": False, "pass_g1": False, "full_authorization_sufficient": False,
        "downstream_capacity_gate_passed": False, "programmatic_parent_of_full_receipt": False,
        "operator_acceptance_status": "NOT_RECORDED",
        "strongest_permitted_interpretation": "PARTIAL_SYNTHETIC_FULL_LIKE_PARTITION_AND_BALANCED_FAN_IN_KERNEL_EVIDENCE_ONLY",
        "operator_paths_redacted": True,
    }
    result["result_fingerprint_sha256"] = VERIFIER._result_fingerprint(result)
    result_path = _mkdir(tmp_path / "result") / "openalex_g1_canonical_work_time_spine_downstream_stress_v1.json"
    _write_json(result_path, result)
    result_sha, _ = _hash(result_path)
    monkeypatch.setattr(VERIFIER, "EXPECTED_STRESS_RESULT_SHA256", result_sha)
    monkeypatch.setattr(
        VERIFIER,
        "EXPECTED_STRESS_RESULT_FINGERPRINT",
        result["result_fingerprint_sha256"],
    )
    return result_path, scratch, pilot_root, first_path


def _refresh_aggregate(document: dict[str, Any]) -> None:
    by_scenario = {}
    for scenario in VERIFIER.SCENARIOS:
        observations = [
            next(
                item
                for item in repeat["scenarios"]
                if item["scenario_id"] == scenario.scenario_id
            )
            for repeat in document["repeats"]
        ]
        by_scenario[scenario.scenario_id] = {
            "repeat_count": VERIFIER.REPEAT_COUNT,
            "sort_wall_rows_per_second": VERIFIER._distribution(
                [item["sort"]["wall_rows_per_second"] for item in observations]
            ),
            "merge_wall_rows_per_second": VERIFIER._distribution(
                [
                    item["merge_parquet_and_validation"]["wall_rows_per_second"]
                    for item in observations
                ]
            ),
            "output_bytes": VERIFIER._distribution(
                [float(item["output_bytes"]) for item in observations]
            ),
        }
    document["aggregate"]["by_scenario"] = by_scenario


def _write_and_repin_result(
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    document: dict[str, Any],
) -> None:
    document["result_fingerprint_sha256"] = VERIFIER._result_fingerprint(document)
    _write_json(path, document)
    digest, _ = _hash(path)
    monkeypatch.setattr(VERIFIER, "EXPECTED_STRESS_RESULT_SHA256", digest)
    monkeypatch.setattr(
        VERIFIER,
        "EXPECTED_STRESS_RESULT_FINGERPRINT",
        document["result_fingerprint_sha256"],
    )


def _rewrite_parquet_layout(
    path: Path,
    records: np.ndarray[Any, Any],
    *,
    row_group_size: int,
    use_dictionary: bool,
) -> tuple[str, int]:
    nulls = records["temporal_disposition"] == 0
    table = pa.Table.from_arrays(
        [
            pa.array(records["work_id_numeric"], type=pa.uint64()),
            pa.array(records["publication_day"], mask=nulls, type=pa.int32()),
            pa.array(records["publication_year"], mask=nulls, type=pa.uint16()),
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
            use_dictionary=use_dictionary,
            write_statistics=False,
            data_page_version="1.0",
            write_batch_size=row_group_size,
            use_byte_stream_split=False,
            write_page_index=False,
            write_page_checksum=False,
            store_schema=False,
            sorting_columns=[],
        )
        writer.write_table(table, row_group_size=row_group_size)
        writer.close()
    path.chmod(0o600)
    return _hash(path)


def test_source_has_no_producer_helper_or_runner_import() -> None:
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == (
        "4ade14fd53f80a964ff8668b7f4e2391ea220402327fcc0fca9c0dd88e19373f"
    )
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add(node.module or "")
    assert "importlib" not in modules
    assert not any("benchmark_openalex" in name or "audit_openalex" in name for name in modules)
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert attributes.isdisjoint({"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "replace", "unlink"})


def test_tiny_full_surface_is_read_only_and_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, scratch, pilot, first = _surface(monkeypatch, tmp_path)
    before = {"result": VERIFIER._snapshot_tree(result.parent, "result"), "scratch": VERIFIER._snapshot_tree(scratch, "scratch"), "pilot": VERIFIER._snapshot_tree(pilot, "pilot"), "first": VERIFIER._snapshot_tree(first.parent, "first")}
    real_open = VERIFIER.os.open
    flags_seen: list[int] = []

    def readonly(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        flags_seen.append(flags)
        forbidden = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        assert flags & forbidden == 0
        assert flags & os.O_NOFOLLOW
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(VERIFIER.os, "open", readonly)
    output = VERIFIER.verify_stress(result_path=result, scratch_root=scratch, pilot_output_root=pilot, first_benchmark_result=first)
    assert flags_seen
    assert output["status"] == VERIFIER.VERIFICATION_STATUS
    assert output["production_component_pass"] is False
    assert output["contained_mode_0775_temp_parent_count"] == 9
    assert VERIFIER._snapshot_tree(result.parent, "result") == before["result"]
    assert VERIFIER._snapshot_tree(scratch, "scratch") == before["scratch"]
    assert VERIFIER._snapshot_tree(pilot, "pilot") == before["pilot"]
    assert VERIFIER._snapshot_tree(first.parent, "first") == before["first"]


def test_spool_and_recomputed_claim_tamper_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result, scratch, pilot, first = _surface(monkeypatch, tmp_path)
    spool = scratch / "repeat-02/A8_BALANCED/.work_time_spine_tmp/spool/part-4095.bin"
    payload = bytearray(spool.read_bytes())
    payload[-1] ^= 1
    spool.write_bytes(payload)
    spool.chmod(0o600)
    with pytest.raises(VERIFIER.StressVerificationError):
        VERIFIER.verify_stress(result_path=result, scratch_root=scratch, pilot_output_root=pilot, first_benchmark_result=first)

    claim_result, claim_scratch, claim_pilot, claim_first = _surface(monkeypatch, tmp_path / "claim")
    document = json.loads(claim_result.read_text(encoding="utf-8"))
    document["downstream_capacity_gate_passed"] = True
    document["result_fingerprint_sha256"] = VERIFIER._result_fingerprint(document)
    _write_json(claim_result, document)
    with pytest.raises(VERIFIER.StressVerificationError):
        VERIFIER.verify_stress(
            result_path=claim_result,
            scratch_root=claim_scratch,
            pilot_output_root=claim_pilot,
            first_benchmark_result=claim_first,
        )


@pytest.mark.parametrize(
    "tamper",
    ("timestamps", "preflight", "negative_timing", "operator_path"),
)
def test_recomputed_nested_bogus_result_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tamper: str,
) -> None:
    result, scratch, pilot, first = _surface(monkeypatch, tmp_path)
    document = json.loads(result.read_text(encoding="utf-8"))
    if tamper == "timestamps":
        document["started_at"] = "NOT_A_TIMESTAMP"
    elif tamper == "preflight":
        document["preflight"]["available_disk_bytes"] = 0
    elif tamper == "negative_timing":
        scenario = document["repeats"][0]["scenarios"][0]
        scenario["sort"]["resources"]["wall_seconds"] = -1.0
        scenario["sort"]["wall_rows_per_second"] = -float(scenario["rows"])
        _refresh_aggregate(document)
    else:
        document["runtime"]["platform_machine"] = "/secret/operator/path"
        document["runtime"]["runner_runtime_fingerprint"]["platform_machine"] = (
            "/secret/operator/path"
        )
    _write_and_repin_result(monkeypatch, result, document)
    with pytest.raises(VERIFIER.StressVerificationError):
        VERIFIER.verify_stress(
            result_path=result,
            scratch_root=scratch,
            pilot_output_root=pilot,
            first_benchmark_result=first,
        )


@pytest.mark.parametrize(
    ("row_group_size", "use_dictionary", "failure"),
    (
        (1, False, "row-group"),
        (VERIFIER.PARQUET_ROW_GROUP_ROWS, True, "encodings"),
    ),
)
def test_nonfrozen_parquet_layout_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    row_group_size: int,
    use_dictionary: bool,
    failure: str,
) -> None:
    result, scratch, pilot, first = _surface(monkeypatch, tmp_path)
    scenario = next(
        item for item in VERIFIER.SCENARIOS if item.scenario_id == "A8_BALANCED"
    )
    scenario_root = scratch / "repeat-01/A8_BALANCED"
    parquet_path = scenario_root / "spine/part-4095.parquet"
    records = _records_for_q(
        _pilot_records(), np.arange(scenario.rows, dtype=np.uint64),
    )
    output_sha, output_bytes = _rewrite_parquet_layout(
        parquet_path,
        records,
        row_group_size=row_group_size,
        use_dictionary=use_dictionary,
    )
    checkpoint_path = scenario_root / "stress_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["output_sha256"] = output_sha
    checkpoint["output_bytes"] = output_bytes
    checkpoint_core = dict(checkpoint)
    checkpoint_core.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = VERIFIER._sha256_bytes(
        VERIFIER._canonical_json_bytes(checkpoint_core)
    )
    _write_json(checkpoint_path, checkpoint)
    document = json.loads(result.read_text(encoding="utf-8"))
    recorded = next(
        item
        for item in document["repeats"][0]["scenarios"]
        if item["scenario_id"] == "A8_BALANCED"
    )
    recorded["output_sha256"] = output_sha
    recorded["output_bytes"] = output_bytes
    sha_metric = recorded["merge_parquet_and_validation"]["coarse_metrics"][
        "parquet_sha256"
    ]
    sha_metric["bytes"] = output_bytes
    sha_metric["bytes_per_second"] = round(
        output_bytes / sha_metric["seconds"], 6
    )
    _refresh_aggregate(document)
    _write_and_repin_result(monkeypatch, result, document)
    with pytest.raises(VERIFIER.StressVerificationError, match=failure):
        VERIFIER.verify_stress(
            result_path=result,
            scratch_root=scratch,
            pilot_output_root=pilot,
            first_benchmark_result=first,
        )


def test_extra_empty_result_directory_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, scratch, pilot, first = _surface(monkeypatch, tmp_path)
    _mkdir(result.parent / "unexpected-empty-directory")
    with pytest.raises(VERIFIER.StressVerificationError, match="unexpected directories"):
        VERIFIER.verify_stress(
            result_path=result,
            scratch_root=scratch,
            pilot_output_root=pilot,
            first_benchmark_result=first,
        )
