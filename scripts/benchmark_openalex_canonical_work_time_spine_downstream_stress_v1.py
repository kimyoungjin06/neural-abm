#!/usr/bin/env python3
"""Run bounded full-like and balanced multi-run canonical-spine stress kernels.

This successor reads only the verified pilot staging payload and the completed
bounded downstream result.  It creates deterministic synthetic numeric IDs in
a disjoint namespace, preserves pilot temporal payloads, and calls the frozen
runner's exact sort and Parquet merge kernels.  It accepts no source root and
never calls partition publication, finalization, or a receipt builder.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import stat
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = (
    REPO_ROOT / "scripts/benchmark_openalex_canonical_work_time_spine_downstream.py"
)
HELPER_SHA256 = "c7809ac74425a837e2b5f58c021f46c297ba0cc7cedd4d60ecf0ad969c6332ff"
RUNNER_PATH = REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v3.py"
RUNNER_SHA256 = "bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280"
CONFIG_PATH = (
    REPO_ROOT
    / "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml"
)
CONFIG_SHA256 = "6e8b74ddd12cbef46c8d15d3560f0d54f470b5017685fc1bce2389250cde0484"
PILOT_VERIFIER_PATH = (
    REPO_ROOT / "scripts/verify_openalex_canonical_work_time_spine_v1.py"
)
PILOT_VERIFIER_SHA256 = (
    "2cc47587ecd70bd1ae169f1f987828f024d6b8f7dd217e2e9b7964fabfcd5ede"
)
FIRST_BENCHMARK_SHA256 = (
    "6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109"
)
FIRST_BENCHMARK_FINGERPRINT = (
    "5e12129cafbda0bc5918eedeb43c3ec6352357f3e78e67b0d6d6494f7a71e287"
)
PILOT_RECEIPT_SHA256 = (
    "eef08252fc6147084cb50752a456541b408992d599d6d80d2d5806b48310b099"
)
PILOT_STAGING_SHA256 = (
    "ccbe4ffe751d148076c1711838aa15b976dfb305da4a0b28ef4937743b2a1165"
)

SCHEMA = "openalex_g1_canonical_work_time_spine_downstream_stress_benchmark.v1"
STATUS = (
    "PARTIAL_FULL_LIKE_PARTITION_AND_BALANCED_MULTI_RUN_STRESS_"
    "OPERATOR_LOCAL_NONPROMOTABLE"
)
OUTPUT_BASENAME = "openalex_g1_canonical_work_time_spine_downstream_stress_v1.json"
PRODUCTION_NOT_RUN = "PRODUCTION_NOT_RUN"
PARTITION_COUNT = 4096
PARTITION_ID = 4095
RECORD_BYTES = 16
PILOT_STAGING_ROWS = 1_000_000
SORT_CHUNK_ROWS = 1_048_576
FULL_RETAINED_ROWS = 492_293_206
FULL_LIKE_MEAN_ROWS = math.ceil(FULL_RETAINED_ROWS / PARTITION_COUNT)
SYNTHETIC_NAMESPACE_BASE = 1 << 63
GENERATION_BATCH_ROWS = 262_144
MAX_RESULT_BYTES = 2 << 20
MINIMUM_STRESS_FREE_DISK_BYTES = 4 << 30
MINIMUM_STRESS_AVAILABLE_RAM_BYTES = 2 << 30


class StressBenchmarkError(RuntimeError):
    """The bounded stress benchmark contract failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StressBenchmarkError(message)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    rows: int
    expected_runs: int
    role: str


SCENARIOS = (
    Scenario("RMEAN", FULL_LIKE_MEAN_ROWS, 1, "FULL_LIKE_MEAN_PARTITION_ROWS"),
    Scenario("A2_BALANCED", 2 * SORT_CHUNK_ROWS, 2, "BALANCED_TWO_WAY_HEAP_MERGE"),
    Scenario("A8_BALANCED", 8 * SORT_CHUNK_ROWS, 8, "BALANCED_EIGHT_WAY_HEAP_MERGE"),
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _published_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _load_module(name: str, path: Path, expected_sha256: str) -> ModuleType:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == expected_sha256, f"{name} SHA-256 binding differs")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(
        hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256,
        f"{name} changed while importing",
    )
    return module


def _load_helper() -> ModuleType:
    return _load_module("_canonical_spine_stress_helper", HELPER_PATH, HELPER_SHA256)


def _read_regular_snapshot(
    path: Path, *, field_name: str, maximum_bytes: int
) -> tuple[bytes, str]:
    absolute = _absolute(path)
    descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    payload = bytearray()
    try:
        start = os.fstat(descriptor)
        require(stat.S_ISREG(start.st_mode), f"{field_name} is not regular")
        require(start.st_nlink == 1, f"{field_name} is hard-linked")
        require(
            0 < start.st_size <= maximum_bytes,
            f"{field_name} size is outside its bound",
        )
        identity = (
            start.st_dev,
            start.st_ino,
            start.st_mode,
            start.st_uid,
            start.st_gid,
            start.st_nlink,
            start.st_size,
            start.st_mtime_ns,
            start.st_ctime_ns,
        )
        while block := os.read(descriptor, min(16 << 20, maximum_bytes + 1)):
            payload.extend(block)
            digest.update(block)
            require(
                len(payload) <= maximum_bytes,
                f"{field_name} exceeds its size bound",
            )
        end = os.fstat(descriptor)
        require(
            identity
            == (
                end.st_dev,
                end.st_ino,
                end.st_mode,
                end.st_uid,
                end.st_gid,
                end.st_nlink,
                end.st_size,
                end.st_mtime_ns,
                end.st_ctime_ns,
            ),
            f"{field_name} changed while reading",
        )
        current = absolute.lstat()
        require(
            identity
            == (
                current.st_dev,
                current.st_ino,
                current.st_mode,
                current.st_uid,
                current.st_gid,
                current.st_nlink,
                current.st_size,
                current.st_mtime_ns,
                current.st_ctime_ns,
            ),
            f"{field_name} path identity changed",
        )
        return bytes(payload), digest.hexdigest()
    finally:
        os.close(descriptor)


def _validate_first_benchmark(
    helper: ModuleType, path: Path
) -> tuple[dict[str, Any], str]:
    result, digest = helper._read_strict_json(
        path, field_name="first downstream benchmark result"
    )
    require(digest == FIRST_BENCHMARK_SHA256, "first benchmark SHA-256 differs")
    helper._validate_final_result(result)
    require(
        result.get("result_fingerprint_sha256") == FIRST_BENCHMARK_FINGERPRINT,
        "first benchmark fingerprint differs",
    )
    require(
        result.get("pilot_binding", {}).get("receipt_sha256")
        == PILOT_RECEIPT_SHA256,
        "first benchmark pilot receipt binding differs",
    )
    require(
        result.get("pilot_binding", {}).get("staging_sha256")
        == PILOT_STAGING_SHA256,
        "first benchmark staging binding differs",
    )
    require(
        result.get("downstream_capacity_gate_passed") is False,
        "first benchmark claims capacity pass",
    )
    return result, digest


def _scenario_run_rows(scenario: Scenario) -> list[int]:
    if scenario.expected_runs == 1:
        require(scenario.rows <= SORT_CHUNK_ROWS, "single-run scenario is too large")
        return [scenario.rows]
    require(
        scenario.rows == scenario.expected_runs * SORT_CHUNK_ROWS,
        "balanced scenario rows do not fill every production-size run",
    )
    return [SORT_CHUNK_ROWS] * scenario.expected_runs


def _synthetic_ids(q_values: Any, runner: ModuleType) -> Any:
    identifiers = (
        runner.np.uint64(SYNTHETIC_NAMESPACE_BASE)
        + (q_values.astype(runner.np.uint64, copy=False) << runner.np.uint64(12))
        + runner.np.uint64(PARTITION_ID)
    )
    require(
        bool(runner.np.all(identifiers % PARTITION_COUNT == PARTITION_ID)),
        "synthetic identifier modulo differs",
    )
    return identifiers


def _records_for_q(
    runner: ModuleType,
    pilot_records: Any,
    q_values: Any,
) -> Any:
    indices = (q_values % len(pilot_records)).astype(runner.np.int64, copy=False)
    records = runner.np.empty(len(q_values), dtype=runner.RECORD_DTYPE)
    records["work_id_numeric"] = _synthetic_ids(q_values, runner)
    for field_name in (
        "publication_day",
        "publication_year",
        "temporal_disposition",
        "reserved",
    ):
        records[field_name] = pilot_records[field_name][indices]
    runner._validate_record_array(records)
    return records


def _read_pilot_records(
    helper: ModuleType,
    runner: ModuleType,
    pilot_staging_path: Path,
) -> Any:
    payload, digest = _read_regular_snapshot(
        pilot_staging_path,
        field_name="pilot staging payload",
        maximum_bytes=PILOT_STAGING_ROWS * RECORD_BYTES,
    )
    require(digest == PILOT_STAGING_SHA256, "pilot staging SHA-256 differs")
    require(
        len(payload) == PILOT_STAGING_ROWS * RECORD_BYTES,
        "pilot staging bytes differ",
    )
    records = runner.np.frombuffer(payload, dtype=runner.RECORD_DTYPE).copy()
    runner._validate_record_array(records)
    return records


def _write_synthetic_spool(
    *,
    helper: ModuleType,
    runner: ModuleType,
    pilot_records: Any,
    scenario_root: Path,
    scenario: Scenario,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spool_root = scenario_root / ".work_time_spine_tmp/spool"
    spool_root.mkdir(parents=True, mode=0o700)
    spool_root.chmod(0o700)
    spool_path = spool_root / f"part-{PARTITION_ID:04d}.bin"
    descriptor = os.open(
        spool_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    digest = hashlib.sha256()
    generated_rows = 0
    started = time.perf_counter()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            for run_index, run_rows in enumerate(_scenario_run_rows(scenario)):
                stop = run_rows
                while stop:
                    start = max(0, stop - GENERATION_BATCH_ROWS)
                    local_j = runner.np.arange(
                        stop - 1,
                        start - 1,
                        -1,
                        dtype=runner.np.uint64,
                    )
                    q_values = (
                        runner.np.uint64(scenario.expected_runs) * local_j
                        + runner.np.uint64(run_index)
                    )
                    records = _records_for_q(
                        runner, pilot_records, q_values
                    )
                    payload = records.tobytes(order="C")
                    handle.write(payload)
                    digest.update(payload)
                    generated_rows += len(records)
                    stop = start
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if spool_path.exists():
            spool_path.unlink()
        raise
    require(generated_rows == scenario.rows, "synthetic spool row count differs")
    spool_path.chmod(0o600)
    contract = {
        "partition_id": PARTITION_ID,
        "relative_path": (
            f".work_time_spine_tmp/spool/part-{PARTITION_ID:04d}.bin"
        ),
        "rows": scenario.rows,
        "bytes": scenario.rows * RECORD_BYTES,
        "sha256": digest.hexdigest(),
    }
    return contract, {
        "rows": generated_rows,
        "bytes": scenario.rows * RECORD_BYTES,
        "wall_seconds": round(time.perf_counter() - started, 6),
        "spool_sha256": digest.hexdigest(),
        "synthetic_namespace_base": SYNTHETIC_NAMESPACE_BASE,
        "partition_id": PARTITION_ID,
        "temporal_payload_source": "VERIFIED_PILOT_STAGING_CYCLIC_BY_Q",
        "excluded_from_kernel_timing": True,
    }


def _framed_digest(
    runner: ModuleType,
    records: Any,
    total_rows: int,
    digest: Any,
) -> None:
    if digest.get("started") is False:
        digest["hash"].update(
            len(runner.RECORD_HASH_DOMAIN).to_bytes(8, "little")
        )
        digest["hash"].update(runner.RECORD_HASH_DOMAIN)
        digest["hash"].update(total_rows.to_bytes(8, "little"))
        digest["started"] = True
    runner._update_framed_record_digest(digest["hash"], records)


def _expected_sorted_contract(
    *,
    runner: ModuleType,
    pilot_records: Any,
    scenario: Scenario,
) -> dict[str, Any]:
    state = {"hash": hashlib.sha256(), "started": False}
    disposition_counts = {
        name: 0 for name in runner.DISPOSITION_NAMES.values()
    }
    null_rows = 0
    minimum: int | None = None
    maximum: int | None = None
    publication_year_minimum: int | None = None
    publication_year_maximum: int | None = None
    for start in range(0, scenario.rows, GENERATION_BATCH_ROWS):
        stop = min(start + GENERATION_BATCH_ROWS, scenario.rows)
        q_values = runner.np.arange(start, stop, dtype=runner.np.uint64)
        records = _records_for_q(runner, pilot_records, q_values)
        _framed_digest(runner, records, scenario.rows, state)
        if len(records):
            minimum = (
                int(records["work_id_numeric"][0])
                if minimum is None
                else minimum
            )
            maximum = int(records["work_id_numeric"][-1])
        counts = runner.np.bincount(
            records["temporal_disposition"],
            minlength=len(runner.DISPOSITION_NAMES),
        )
        for code, name in runner.DISPOSITION_NAMES.items():
            disposition_counts[name] += int(counts[code])
        null_rows += int(
            runner.np.count_nonzero(records["temporal_disposition"] == 0)
        )
        non_null = records["temporal_disposition"] != 0
        if bool(runner.np.any(non_null)):
            years = records["publication_year"][non_null]
            batch_minimum = int(runner.np.min(years))
            batch_maximum = int(runner.np.max(years))
            publication_year_minimum = (
                batch_minimum
                if publication_year_minimum is None
                else min(publication_year_minimum, batch_minimum)
            )
            publication_year_maximum = (
                batch_maximum
                if publication_year_maximum is None
                else max(publication_year_maximum, batch_maximum)
            )
    return {
        "record_sha256": state["hash"].hexdigest(),
        "minimum_work_id_numeric": minimum,
        "maximum_work_id_numeric": maximum,
        "disposition_counts": disposition_counts,
        "nullability_counts": {
            "publication_day_null_rows": null_rows,
            "publication_year_null_rows": null_rows,
        },
        "publication_year_extrema": {
            "minimum": publication_year_minimum,
            "maximum": publication_year_maximum,
        },
        "w_zero_disposition_counts": {
            name: 0 for name in runner.DISPOSITION_NAMES.values()
        },
    }


def _validate_sorted_runs(
    *,
    helper: ModuleType,
    runner: ModuleType,
    pilot_records: Any,
    scenario: Scenario,
    runs: Sequence[Any],
) -> dict[str, Any]:
    require(len(runs) == scenario.expected_runs, "external run count differs")
    expected_rows = _scenario_run_rows(scenario)
    run_contracts: list[dict[str, Any]] = []
    for run_index, (run, rows) in enumerate(
        zip(runs, expected_rows, strict=True)
    ):
        require(run.rows == rows and run.bytes == rows * RECORD_BYTES, "run size differs")
        payload, digest = _read_regular_snapshot(
            run.path,
            field_name="synthetic external run",
            maximum_bytes=rows * RECORD_BYTES,
        )
        size = len(payload)
        require(digest == run.sha256 and size == run.bytes, "run SHA-256 differs")
        records = runner.np.frombuffer(payload, dtype=runner.RECORD_DTYPE)
        runner._validate_record_array(records)
        identifiers = records["work_id_numeric"]
        require(
            bool(runner.np.all(identifiers[1:] > identifiers[:-1])),
            "run identifiers are not strictly increasing",
        )
        q_values = (
            identifiers
            - runner.np.uint64(SYNTHETIC_NAMESPACE_BASE + PARTITION_ID)
        ) >> runner.np.uint64(12)
        require(int(q_values[0]) == run_index, "run q-start differs")
        require(
            int(q_values[-1])
            == scenario.expected_runs * (rows - 1) + run_index,
            "run q-stop differs",
        )
        if len(q_values) > 1:
            require(
                bool(
                    runner.np.all(
                        q_values[1:] - q_values[:-1]
                        == scenario.expected_runs
                    )
                ),
                "run q-stride differs",
            )
        expected_temporal = pilot_records[
            (q_values % len(pilot_records)).astype(runner.np.int64)
        ]
        for field_name in (
            "publication_day",
            "publication_year",
            "temporal_disposition",
            "reserved",
        ):
            require(
                bool(
                    runner.np.array_equal(
                        records[field_name], expected_temporal[field_name]
                    )
                ),
                f"run temporal payload differs: {field_name}",
            )
        run_contracts.append(
            {
                "run_index": run_index,
                "rows": rows,
                "bytes": size,
                "sha256": digest,
                "q_start": int(q_values[0]),
                "q_stop": int(q_values[-1]),
                "q_stride": scenario.expected_runs,
            }
        )
    return {
        "run_count": len(runs),
        "run_contract_tree_sha256": _sha256_bytes(
            _canonical_json_bytes(run_contracts)
        ),
        "runs": run_contracts,
        "balanced_fan_in": scenario.expected_runs > 1,
    }


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "minimum": None, "median": None, "maximum": None}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "minimum": round(ordered[0], 6),
        "median": round(ordered[len(ordered) // 2], 6),
        "maximum": round(ordered[-1], 6),
    }


def _rate(rows: int, seconds: float) -> float | None:
    return None if seconds <= 0 else round(rows / seconds, 6)


@dataclass
class CoarseMetrics:
    argsort_seconds: float = 0.0
    argsort_rows: int = 0
    argsort_calls: int = 0
    fsync_seconds: list[float] = field(default_factory=list)
    parquet_writer_seconds: float = 0.0
    parquet_writer_rows: int = 0
    parquet_write_calls: int = 0
    parquet_close_calls: int = 0
    arrow_conversion_seconds: float = 0.0
    arrow_conversion_rows: int = 0
    arrow_conversion_calls: int = 0
    decoded_validation_seconds: float = 0.0
    decoded_validation_rows: int = 0
    decoded_validation_calls: int = 0
    sha256_seconds: float = 0.0
    sha256_bytes: int = 0
    sha256_calls: int = 0
    phase_started: float = 0.0
    pre_validation_elapsed_seconds: float | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "numpy_stable_argsort": {
                "calls": self.argsort_calls,
                "rows": self.argsort_rows,
                "seconds": round(self.argsort_seconds, 6),
                "rows_per_second": _rate(self.argsort_rows, self.argsort_seconds),
            },
            "fsync": {
                "calls": len(self.fsync_seconds),
                "total_seconds": round(sum(self.fsync_seconds), 6),
                "latency_seconds": _distribution(self.fsync_seconds),
            },
            "parquet_writer": {
                "write_calls": self.parquet_write_calls,
                "close_calls": self.parquet_close_calls,
                "rows": self.parquet_writer_rows,
                "seconds": round(self.parquet_writer_seconds, 6),
                "rows_per_second": _rate(
                    self.parquet_writer_rows, self.parquet_writer_seconds
                ),
            },
            "arrow_table_conversion": {
                "calls": self.arrow_conversion_calls,
                "rows": self.arrow_conversion_rows,
                "seconds": round(self.arrow_conversion_seconds, 6),
                "rows_per_second": _rate(
                    self.arrow_conversion_rows, self.arrow_conversion_seconds
                ),
            },
            "decoded_validation": {
                "calls": self.decoded_validation_calls,
                "rows": self.decoded_validation_rows,
                "seconds": round(self.decoded_validation_seconds, 6),
                "rows_per_second": _rate(
                    self.decoded_validation_rows,
                    self.decoded_validation_seconds,
                ),
            },
            "parquet_sha256": {
                "calls": self.sha256_calls,
                "bytes": self.sha256_bytes,
                "seconds": round(self.sha256_seconds, 6),
                "bytes_per_second": _rate(self.sha256_bytes, self.sha256_seconds),
            },
            "merge_and_parquet_before_decoded_validation_seconds": (
                None
                if self.pre_validation_elapsed_seconds is None
                else round(self.pre_validation_elapsed_seconds, 6)
            ),
            "heap_operation_timing_proxy_installed": False,
            "run_reader_proxy_installed": False,
        }


class _WriterProxy:
    def __init__(self, writer: Any, metrics: CoarseMetrics) -> None:
        self.writer = writer
        self.metrics = metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self.writer, name)

    def write_table(self, table: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return self.writer.write_table(table, *args, **kwargs)
        finally:
            self.metrics.parquet_writer_seconds += time.perf_counter() - started
            self.metrics.parquet_writer_rows += int(table.num_rows)
            self.metrics.parquet_write_calls += 1

    def close(self) -> Any:
        started = time.perf_counter()
        try:
            return self.writer.close()
        finally:
            self.metrics.parquet_writer_seconds += time.perf_counter() - started
            self.metrics.parquet_close_calls += 1


@contextmanager
def _instrument_runner_coarse(
    helper: ModuleType, runner: ModuleType
) -> Iterator[CoarseMetrics]:
    metrics = CoarseMetrics(phase_started=time.perf_counter())
    original_os = runner.os
    original_np = runner.np
    original_writer = runner._new_parquet_writer
    original_table = runner._encoded_records_to_table
    original_validation = runner._stream_validate_parquet
    original_sha = runner._sha256_file
    heapq_identity = runner.heapq
    run_reader_identity = runner._RunReader

    def writer(target: Any) -> _WriterProxy:
        return _WriterProxy(original_writer(target), metrics)

    def table(payload: bytes | bytearray) -> Any:
        started = time.perf_counter()
        try:
            return original_table(payload)
        finally:
            metrics.arrow_conversion_seconds += time.perf_counter() - started
            metrics.arrow_conversion_rows += len(payload) // RECORD_BYTES
            metrics.arrow_conversion_calls += 1

    def validation(
        path: Path, *, partition_id: int, expected_rows: int
    ) -> dict[str, Any]:
        if metrics.pre_validation_elapsed_seconds is None:
            metrics.pre_validation_elapsed_seconds = (
                time.perf_counter() - metrics.phase_started
            )
        started = time.perf_counter()
        try:
            return original_validation(
                path,
                partition_id=partition_id,
                expected_rows=expected_rows,
            )
        finally:
            metrics.decoded_validation_seconds += time.perf_counter() - started
            metrics.decoded_validation_rows += expected_rows
            metrics.decoded_validation_calls += 1

    def sha(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
        size = path.stat().st_size
        started = time.perf_counter()
        try:
            return original_sha(path, chunk_bytes=chunk_bytes)
        finally:
            metrics.sha256_seconds += time.perf_counter() - started
            metrics.sha256_bytes += int(size)
            metrics.sha256_calls += 1

    runner.os = helper._OSProxy(original_os, metrics)
    runner.np = helper._NumpyProxy(original_np, metrics)
    runner._new_parquet_writer = writer
    runner._encoded_records_to_table = table
    runner._stream_validate_parquet = validation
    runner._sha256_file = sha
    try:
        yield metrics
    finally:
        runner.os = original_os
        runner.np = original_np
        runner._new_parquet_writer = original_writer
        runner._encoded_records_to_table = original_table
        runner._stream_validate_parquet = original_validation
        runner._sha256_file = original_sha
        require(runner.heapq is heapq_identity, "runner heapq identity changed")
        require(runner._RunReader is run_reader_identity, "runner RunReader changed")


def _run_scenario(
    *,
    helper: ModuleType,
    runner: ModuleType,
    pilot_records: Any,
    repeat_root: Path,
    scenario: Scenario,
    repeat_index: int,
) -> dict[str, Any]:
    scenario_root = repeat_root / scenario.scenario_id
    scenario_root.mkdir(mode=0o700)
    scenario_root.chmod(0o700)
    for relative in (".work_time_spine_tmp/runs", ".work_time_spine_tmp/sorted"):
        directory = scenario_root / relative
        directory.mkdir(parents=True, mode=0o700)
        directory.chmod(0o700)
    spool_contract, preparation = _write_synthetic_spool(
        helper=helper,
        runner=runner,
        pilot_records=pilot_records,
        scenario_root=scenario_root,
        scenario=scenario,
    )
    with _instrument_runner_coarse(helper, runner) as sort_metrics:
        with helper._measure_resources() as sort_resources:
            runs, observed_rows = runner._write_sorted_runs(
                scenario_root,
                PARTITION_ID,
                sort_chunk_rows=SORT_CHUNK_ROWS,
                spool_contract=spool_contract,
            )
    require(observed_rows == scenario.rows, "sorted-run row count differs")
    run_validation_started = time.perf_counter()
    run_validation = _validate_sorted_runs(
        helper=helper,
        runner=runner,
        pilot_records=pilot_records,
        scenario=scenario,
        runs=runs,
    )
    run_validation["wall_seconds"] = round(
        time.perf_counter() - run_validation_started, 6
    )
    run_validation["excluded_from_kernel_timing"] = True
    run_validation["prewarms_external_run_page_cache_before_merge"] = True
    expected = _expected_sorted_contract(
        runner=runner,
        pilot_records=pilot_records,
        scenario=scenario,
    )
    task_fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": SCHEMA,
                "repeat_index": repeat_index,
                "scenario": scenario.scenario_id,
                "first_benchmark_sha256": FIRST_BENCHMARK_SHA256,
            }
        )
    )
    checkpoint_schema = (
        "openalex_g1_canonical_work_time_spine_downstream_"
        "stress_partition_checkpoint.v1"
    )
    scan_tree_sha256 = _sha256_bytes(
        _canonical_json_bytes(
            {"scenario": scenario.scenario_id, "repeat": repeat_index}
        )
    )
    staging_tree_sha256 = spool_contract["sha256"]
    with _instrument_runner_coarse(helper, runner) as merge_metrics:
        with helper._measure_resources() as merge_resources:
            checkpoint = runner._merge_runs_to_temporary_parquet(
                scenario_root,
                PARTITION_ID,
                runs,
                scenario.rows,
                task_fingerprint_sha256=task_fingerprint,
                checkpoint_schema=checkpoint_schema,
                scan_tree_sha256=scan_tree_sha256,
                staging_tree_sha256=staging_tree_sha256,
            )
    require(
        checkpoint["output_rows"] == scenario.rows,
        "stress Parquet row count differs",
    )
    for key in (
        "pre_parquet_sorted_record_sha256",
        "decoded_parquet_record_sha256",
    ):
        require(checkpoint[key] == expected["record_sha256"], f"{key} differs")
    for key in (
        "minimum_work_id_numeric",
        "maximum_work_id_numeric",
        "disposition_counts",
        "nullability_counts",
        "publication_year_extrema",
        "w_zero_disposition_counts",
    ):
        require(checkpoint[key] == expected[key], f"stress checkpoint {key} differs")
    schema_contract = [
        {"name": item.name, "type": str(item.type), "nullable": item.nullable}
        for item in runner.ARROW_SCHEMA
    ]
    checkpoint_bindings = {
        "schema_version": checkpoint_schema,
        "task_fingerprint_sha256": task_fingerprint,
        "runtime_fingerprint": runner._runtime_fingerprint(),
        "scan_checkpoint_tree_sha256": scan_tree_sha256,
        "staging_shard_tree_sha256": staging_tree_sha256,
        "partition_id": PARTITION_ID,
        "partition_formula": "work_id_numeric_mod_4096",
        "output_relative_path": f"spine/part-{PARTITION_ID:04d}.parquet",
        "output_schema": schema_contract,
        "output_schema_sha256": _sha256_bytes(
            _canonical_json_bytes(schema_contract)
        ),
        "strictly_increasing_ids": True,
        "operator_paths_redacted": True,
    }
    for key, expected_value in checkpoint_bindings.items():
        require(
            checkpoint.get(key) == expected_value,
            f"stress checkpoint binding differs: {key}",
        )
    checkpoint_core = dict(checkpoint)
    checkpoint_sha256 = checkpoint_core.pop("checkpoint_sha256", None)
    require(
        checkpoint_sha256 == _sha256_bytes(_canonical_json_bytes(checkpoint_core)),
        "stress checkpoint self-hash differs",
    )
    temporary = (
        scenario_root
        / ".work_time_spine_tmp/sorted"
        / f"part-{PARTITION_ID:04d}.parquet"
    )
    output = scenario_root / f"spine/part-{PARTITION_ID:04d}.parquet"
    output.parent.mkdir(mode=0o700)
    output.parent.chmod(0o700)
    os.replace(temporary, output)
    runner._fsync_directory(temporary.parent)
    runner._fsync_directory(output.parent)
    output.chmod(0o600)
    helper._write_new_json(scenario_root / "stress_checkpoint.json", checkpoint)
    runner._fsync_directory(scenario_root)
    output_sha, output_bytes = helper._sha256_regular_file(
        output, field_name="stress Parquet output"
    )
    require(output_sha == checkpoint["output_sha256"], "stress output SHA differs")
    derived_heap_operations = scenario.rows if scenario.expected_runs > 1 else 0
    return {
        "scenario_id": scenario.scenario_id,
        "role": scenario.role,
        "rows": scenario.rows,
        "partition_id": PARTITION_ID,
        "sort_chunk_rows": SORT_CHUNK_ROWS,
        "expected_runs": scenario.expected_runs,
        "observed_runs": len(runs),
        "maximum_measured_fan_in": scenario.expected_runs,
        "preparation": preparation,
        "run_validation": run_validation,
        "sort": {
            "wall_rows_per_second": _rate(
                scenario.rows, float(sort_resources["wall_seconds"])
            ),
            "resources": sort_resources,
            "coarse_metrics": sort_metrics.summary(),
        },
        "merge_parquet_and_validation": {
            "wall_rows_per_second": _rate(
                scenario.rows, float(merge_resources["wall_seconds"])
            ),
            "resources": merge_resources,
            "coarse_metrics": merge_metrics.summary(),
            "cache_condition": (
                "CACHE_ASSISTED_BY_IMMEDIATELY_PRECEDING_FULL_RUN_VALIDATION"
            ),
            "derived_heap_push_count_not_observed": derived_heap_operations,
            "derived_heap_pop_count_not_observed": derived_heap_operations,
        },
        "output_rows": checkpoint["output_rows"],
        "output_bytes": output_bytes,
        "output_sha256": output_sha,
        "record_sha256": expected["record_sha256"],
        "integrity_status": "PASS_SYNTHETIC_STRESS_KERNEL_CLOSURE",
        "operator_paths_redacted": True,
    }


def _aggregate(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    for scenario in SCENARIOS:
        observations = [
            next(
                item
                for item in repeat["scenarios"]
                if item["scenario_id"] == scenario.scenario_id
            )
            for repeat in repeats
        ]
        by_scenario[scenario.scenario_id] = {
            "repeat_count": len(observations),
            "sort_wall_rows_per_second": _distribution(
                [item["sort"]["wall_rows_per_second"] for item in observations]
            ),
            "merge_wall_rows_per_second": _distribution(
                [
                    item["merge_parquet_and_validation"]["wall_rows_per_second"]
                    for item in observations
                ]
            ),
            "output_bytes": _distribution(
                [float(item["output_bytes"]) for item in observations]
            ),
        }
    return {
        "by_scenario": by_scenario,
        "one_sided_95_percent_lower_confidence_bound_computed": False,
        "minimum_observed_rates_are_not_adopted_capacity_floors": True,
        "independent_quiet_window_sessions": 0,
    }


def _result_fingerprint(result: Mapping[str, Any]) -> str:
    payload = dict(result)
    payload.pop("result_fingerprint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _validate_final_result(result: Mapping[str, Any]) -> None:
    require(result.get("schema_version") == SCHEMA, "stress result schema differs")
    require(result.get("status") == STATUS, "stress result status differs")
    for key in (
        "production_component_pass",
        "pass_g1",
        "full_authorization_sufficient",
        "downstream_capacity_gate_passed",
        "programmatic_parent_of_full_receipt",
        "actual_full_4096_partition_occupancy_profile_measured",
        "full_byte_finalization_surface_measured",
        "theoretical_470_run_heap_merge_measured",
        "source_hdd_opened",
    ):
        require(result.get(key) is False, f"stress claim boundary differs: {key}")
    for key in (
        "full_like_per_partition_row_scale_measured",
        "balanced_multi_run_heap_branch_measured",
    ):
        require(result.get(key) is True, f"stress evidence boundary differs: {key}")
    require(
        result.get("production_execution_status") == PRODUCTION_NOT_RUN,
        "stress result claims production execution",
    )
    require(
        result.get("independent_verification_status") == "PENDING",
        "stress result independent-verification boundary differs",
    )
    require(
        result.get("operator_acceptance_status") == "NOT_RECORDED",
        "stress result operator-acceptance boundary differs",
    )
    require(
        result.get("maximum_measured_fan_in") == 8,
        "stress maximum measured fan-in differs",
    )
    require(
        result.get("independent_quiet_window_sessions") == 0,
        "stress quiet-session boundary differs",
    )
    require(
        result.get("cache_condition") == "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
        "stress cache boundary differs",
    )
    implementation = result.get("implementation_binding")
    require(isinstance(implementation, Mapping), "stress implementation binding differs")
    for key in (
        "heap_operation_timing_proxy_installed",
        "run_reader_proxy_installed",
        "production_partition_publisher_called",
        "production_finalizer_called",
        "production_receipt_builder_called",
    ):
        require(
            implementation.get(key) is False,
            f"stress implementation claim boundary differs: {key}",
        )
    require(result.get("operator_paths_redacted") is True, "stress paths differ")
    require(
        result.get("result_fingerprint_sha256") == _result_fingerprint(result),
        "stress result fingerprint differs",
    )


def run_benchmark(
    *,
    pilot_output_root: Path,
    first_benchmark_result: Path,
    scratch_root: Path,
    output_path: Path,
    repeats: int,
    execute_synthetic_nonpromotable: bool,
) -> dict[str, Any]:
    require(
        execute_synthetic_nonpromotable,
        "explicit synthetic nonpromotable execution flag is required",
    )
    require(1 <= repeats <= 3, "stress repeat count must be between 1 and 3")
    helper = _load_helper()
    pilot_root = _absolute(pilot_output_root)
    first_result_path = _absolute(first_benchmark_result)
    scratch = _absolute(scratch_root)
    output = _absolute(output_path)
    require(output.name == OUTPUT_BASENAME, "stress result basename differs")
    helper._private_directory(pilot_root, field_name="pilot output root")
    helper._private_directory(output.parent, field_name="stress result parent")
    require(not output.exists() and not output.is_symlink(), "stress result exists")
    for left, right, message in (
        (pilot_root, REPO_ROOT, "pilot output overlaps repository"),
        (first_result_path, REPO_ROOT, "first benchmark overlaps repository"),
        (scratch, REPO_ROOT, "stress scratch overlaps repository"),
        (output, REPO_ROOT, "stress output overlaps repository"),
        (scratch, pilot_root, "stress scratch overlaps pilot"),
        (scratch, first_result_path.parent, "stress scratch overlaps first result"),
        (scratch, output.parent, "stress scratch overlaps result root"),
        (output.parent, pilot_root, "stress result root overlaps pilot"),
        (
            output.parent,
            first_result_path.parent,
            "stress result root overlaps first result root",
        ),
    ):
        require(not helper._paths_overlap(left, right), message)
    first_tree_before = helper._tree_identity_digest(first_result_path.parent)
    first_result, first_sha = _validate_first_benchmark(
        helper, first_result_path
    )
    config_sha, _ = helper._sha256_regular_file(
        CONFIG_PATH, field_name="frozen production config"
    )
    require(config_sha == CONFIG_SHA256, "frozen config SHA-256 differs")
    runner = helper._load_bound_module(
        "_canonical_spine_stress_runner_v3", RUNNER_PATH, RUNNER_SHA256
    )
    pilot_verifier = helper._load_bound_module(
        "_canonical_spine_stress_pilot_verifier_v1",
        PILOT_VERIFIER_PATH,
        PILOT_VERIFIER_SHA256,
    )
    verification_before, pilot_report, pilot_staging_path, receipt_sha = (
        helper._validate_pilot(pilot_root, pilot_verifier)
    )
    require(receipt_sha == PILOT_RECEIPT_SHA256, "pilot receipt SHA-256 differs")
    pilot_tree_before = helper._tree_identity_digest(pilot_root)
    pilot_records = _read_pilot_records(
        helper, runner, pilot_staging_path
    )
    preflight = helper._preflight(scratch.parent)
    require(
        preflight["available_disk_bytes"] >= MINIMUM_STRESS_FREE_DISK_BYTES,
        "stress disk floor not met",
    )
    require(
        preflight["available_ram_bytes"] >= MINIMUM_STRESS_AVAILABLE_RAM_BYTES,
        "stress RAM floor not met",
    )
    scratch = helper._make_fresh_private_root(scratch, field_name="stress scratch")
    preflight["stress_kernel_minimum_free_disk_bytes"] = (
        MINIMUM_STRESS_FREE_DISK_BYTES
    )
    preflight["stress_kernel_minimum_available_ram_bytes"] = (
        MINIMUM_STRESS_AVAILABLE_RAM_BYTES
    )
    preflight["effective_required_free_disk_bytes"] = max(
        int(preflight["required_disk_bytes"]), MINIMUM_STRESS_FREE_DISK_BYTES
    )
    preflight["effective_required_available_ram_bytes"] = max(
        int(preflight["required_available_ram_bytes"]),
        MINIMUM_STRESS_AVAILABLE_RAM_BYTES,
    )
    preflight["pinned_helper_preflight_is_enforced"] = True
    preflight["atomic_rename_fsync_probe"] = helper._atomic_fsync_probe(scratch)
    tool_sha_before, _ = helper._sha256_regular_file(
        Path(__file__), field_name="stress benchmark tool"
    )
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    repeat_results: list[dict[str, Any]] = []
    for repeat_index in range(1, repeats + 1):
        repeat_root = scratch / f"repeat-{repeat_index:02d}"
        repeat_root.mkdir(mode=0o700)
        repeat_root.chmod(0o700)
        repeat_results.append(
            {
                "repeat_index": repeat_index,
                "scenarios": [
                    _run_scenario(
                        helper=helper,
                        runner=runner,
                        pilot_records=pilot_records,
                        repeat_root=repeat_root,
                        scenario=scenario,
                        repeat_index=repeat_index,
                    )
                    for scenario in SCENARIOS
                ],
                "status": "PASS_SYNTHETIC_STRESS_REPEAT",
                "operator_paths_redacted": True,
            }
        )
    verification_after, _, _, receipt_after = helper._validate_pilot(
        pilot_root, pilot_verifier
    )
    require(verification_after == verification_before, "pilot verification changed")
    require(receipt_after == receipt_sha, "pilot receipt changed")
    pilot_tree_after = helper._tree_identity_digest(pilot_root)
    require(pilot_tree_after == pilot_tree_before, "pilot tree changed")
    first_tree_after = helper._tree_identity_digest(first_result_path.parent)
    require(first_tree_after == first_tree_before, "first benchmark tree changed")
    tool_sha_after, _ = helper._sha256_regular_file(
        Path(__file__), field_name="stress benchmark tool"
    )
    require(tool_sha_after == tool_sha_before, "stress tool changed during run")
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "implementation_binding": {
            "stress_tool_sha256": tool_sha_before,
            "helper_tool_sha256": HELPER_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "config_sha256": CONFIG_SHA256,
            "pilot_verifier_sha256": PILOT_VERIFIER_SHA256,
            "heap_operation_timing_proxy_installed": False,
            "run_reader_proxy_installed": False,
            "private_downstream_api_used": True,
            "production_partition_publisher_called": False,
            "production_finalizer_called": False,
            "production_receipt_builder_called": False,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform_machine": platform.machine(),
            "runner_runtime_fingerprint": runner._runtime_fingerprint(),
        },
        "parent_bindings": {
            "first_benchmark_sha256": first_sha,
            "first_benchmark_fingerprint": first_result[
                "result_fingerprint_sha256"
            ],
            "pilot_receipt_sha256": receipt_sha,
            "pilot_staging_sha256": PILOT_STAGING_SHA256,
            "pilot_task_fingerprint_sha256": pilot_report[
                "task_fingerprint_sha256"
            ],
            "first_benchmark_tree_unchanged": True,
            "pilot_tree_unchanged": True,
        },
        "preflight": preflight,
        "scenario_contracts": [
            {
                "scenario_id": item.scenario_id,
                "rows": item.rows,
                "expected_runs": item.expected_runs,
                "role": item.role,
            }
            for item in SCENARIOS
        ],
        "repeats": repeat_results,
        "aggregate": _aggregate(repeat_results),
        "full_like_per_partition_row_scale_measured": True,
        "balanced_multi_run_heap_branch_measured": True,
        "maximum_measured_fan_in": 8,
        "theoretical_470_run_heap_merge_measured": False,
        "actual_full_4096_partition_occupancy_profile_measured": False,
        "full_byte_finalization_surface_measured": False,
        "independent_verification_status": "PENDING",
        "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
        "independent_quiet_window_sessions": 0,
        "source_hdd_opened": False,
        "production_execution_status": PRODUCTION_NOT_RUN,
        "production_component_pass": False,
        "pass_g1": False,
        "full_authorization_sufficient": False,
        "downstream_capacity_gate_passed": False,
        "programmatic_parent_of_full_receipt": False,
        "operator_acceptance_status": "NOT_RECORDED",
        "strongest_permitted_interpretation": (
            "PARTIAL_SYNTHETIC_FULL_LIKE_PARTITION_AND_BALANCED_FAN_IN_"
            "KERNEL_EVIDENCE_ONLY"
        ),
        "operator_paths_redacted": True,
    }
    result["result_fingerprint_sha256"] = _result_fingerprint(result)
    _validate_final_result(result)
    payload = _published_json_bytes(result)
    require(len(payload) <= MAX_RESULT_BYTES, "stress result exceeds size bound")
    helper._publish_new(output, payload)
    published_sha, published_bytes = helper._sha256_regular_file(
        output, field_name="published stress result"
    )
    require(published_bytes == len(payload), "published stress bytes differ")
    require(published_sha == _sha256_bytes(payload), "published stress SHA differs")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output-root", type=Path, required=True)
    parser.add_argument("--first-benchmark-result", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--execute-synthetic-nonpromotable",
        action="store_true",
        help="authorize only the bounded synthetic stress kernels",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        result = run_benchmark(
            pilot_output_root=args.pilot_output_root,
            first_benchmark_result=args.first_benchmark_result,
            scratch_root=args.scratch_root,
            output_path=args.output,
            repeats=args.repeats,
            execute_synthetic_nonpromotable=args.execute_synthetic_nonpromotable,
        )
    except (StressBenchmarkError, RuntimeError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        _published_json_bytes(
            {
                "schema_version": result["schema_version"],
                "status": result["status"],
                "repeat_count": len(result["repeats"]),
                "result_fingerprint_sha256": result[
                    "result_fingerprint_sha256"
                ],
                "production_execution_status": PRODUCTION_NOT_RUN,
                "downstream_capacity_gate_passed": False,
            }
        ).decode("utf-8"),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
