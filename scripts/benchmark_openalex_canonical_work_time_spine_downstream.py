#!/usr/bin/env python3
"""Run a bounded, operator-local canonical-spine downstream benchmark.

The harness reads only an independently verified one-file pilot output.  It
copies the pilot's fixed-width staging payload into a fresh private scratch
tree, calls the frozen production runner's downstream private kernels, and
publishes an immutable benchmark result outside the repository.  It never
opens an OpenAlex source Parquet, calls a production finalizer, creates a
production receipt, passes a capacity gate, or grants full authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import secrets
import shutil
import stat
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator, Mapping, NoReturn, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts/audit_openalex_canonical_work_time_spine_v3.py"
CONFIG_PATH = (
    REPO_ROOT
    / "experiments/configs/openalex_g1_canonical_work_time_spine_v2.yaml"
)
VERIFIER_PATH = (
    REPO_ROOT / "scripts/verify_openalex_canonical_work_time_spine_v1.py"
)
RUNNER_SHA256 = "bf9ed534b2e9f9c2d6c9851be7fde298eb372d8a3cf5e17363ea6afe51cf1280"
CONFIG_SHA256 = "6e8b74ddd12cbef46c8d15d3560f0d54f470b5017685fc1bce2389250cde0484"
VERIFIER_SHA256 = "2cc47587ecd70bd1ae169f1f987828f024d6b8f7dd217e2e9b7964fabfcd5ede"

SCHEMA = "openalex_g1_canonical_work_time_spine_downstream_benchmark.v1"
STATUS = (
    "PARTIAL_FIXED_OVERHEAD_SINGLE_RUN_AND_BOUNDED_H9185_EVIDENCE_"
    "OPERATOR_LOCAL_NONPROMOTABLE"
)
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_UNPROMOTABLE"
PILOT_STATUS = "PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE"
PRODUCTION_NOT_RUN = "PRODUCTION_NOT_RUN"
OUTPUT_BASENAME = "openalex_g1_canonical_work_time_spine_downstream_benchmark.json"
RECORD_BYTES = 16
PARTITION_COUNT = 4096
FULL_SCAN_SHARD_COUNT = 493
CONTROL_ARTIFACT_COUNT = 7
EXACT_ARTIFACT_COUNT = 9_185
SORT_CHUNK_ROWS = 1_048_576
MINIMUM_NOFILE = 8_192
MINIMUM_AVAILABLE_RAM_BYTES = 9_685_696_512
MINIMUM_FREE_DISK_BYTES = 36_366_736_928
MAX_RESULT_BYTES = 4 << 20
MAX_CONTROL_BYTES = 16 << 20
SHA256_KEYS = {
    "config_sha256",
    "result_fingerprint_sha256",
    "task_fingerprint_sha256",
    "tool_sha256",
}
CONTROL_PATHS = (
    "benchmark_plan.json",
    "preflight_snapshot.json",
    "run_state.json",
    "staging_manifest.json",
    "spine_manifest.json",
    "benchmark_report.json",
    "benchmark_report.md",
)


class DownstreamBenchmarkError(RuntimeError):
    """The bounded downstream benchmark contract failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DownstreamBenchmarkError(message)


def _fail(message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise DownstreamBenchmarkError(message)
    raise DownstreamBenchmarkError(message) from cause


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _paths_overlap(left: Path, right: Path) -> bool:
    left = _absolute(left)
    right = _absolute(right)
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _assert_no_symlink_components(path: Path, *, field_name: str) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            _fail(f"cannot inspect {field_name}", exc)
        require(not stat.S_ISLNK(info.st_mode), f"{field_name} contains a symlink")
    return absolute


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


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_regular_file(path: Path, *, field_name: str) -> tuple[str, int]:
    absolute = _assert_no_symlink_components(path, field_name=field_name)
    try:
        descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        _fail(f"cannot open {field_name}", exc)
    digest = hashlib.sha256()
    try:
        start = os.fstat(descriptor)
        require(stat.S_ISREG(start.st_mode), f"{field_name} is not regular")
        require(start.st_nlink == 1, f"{field_name} is hard-linked")
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
        while block := os.read(descriptor, 16 << 20):
            digest.update(block)
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
            f"{field_name} changed while hashing",
        )
        require(
            (
                absolute.lstat().st_dev,
                absolute.lstat().st_ino,
                absolute.lstat().st_mode,
                absolute.lstat().st_uid,
                absolute.lstat().st_gid,
                absolute.lstat().st_nlink,
                absolute.lstat().st_size,
                absolute.lstat().st_mtime_ns,
                absolute.lstat().st_ctime_ns,
            )
            == identity,
            f"{field_name} path identity changed",
        )
        return digest.hexdigest(), int(start.st_size)
    finally:
        os.close(descriptor)


def _read_strict_json(path: Path, *, field_name: str) -> tuple[dict[str, Any], str]:
    digest, size = _sha256_regular_file(path, field_name=field_name)
    require(0 < size <= MAX_CONTROL_BYTES, f"{field_name} size is outside its bound")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read {field_name}", exc)

    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            require(key not in value, f"{field_name} repeats a JSON key")
            value[key] = item
        return value

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail(f"cannot parse {field_name}", exc)
    require(isinstance(value, dict), f"{field_name} must be a mapping")
    require(
        payload == _published_json_bytes(value),
        f"{field_name} is not exact canonical published JSON",
    )
    return dict(value), digest


def _load_bound_module(name: str, path: Path, expected_sha256: str) -> ModuleType:
    observed, _ = _sha256_regular_file(path, field_name=f"{name} module")
    require(observed == expected_sha256, f"{name} SHA-256 binding differs")
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException as exc:
        _fail(f"cannot import {name}", exc)
    observed_after, _ = _sha256_regular_file(path, field_name=f"{name} module")
    require(observed_after == expected_sha256, f"{name} changed while importing")
    return module


def _tree_identity_digest(root: Path) -> dict[str, Any]:
    root = _assert_no_symlink_components(root, field_name="pilot output root")
    root_info = root.lstat()
    require(stat.S_ISDIR(root_info.st_mode), "pilot output root is not a directory")
    items: list[dict[str, Any]] = []
    regular_bytes = 0
    regular_files = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "pilot output tree contains a symlink")
        kind = "directory" if stat.S_ISDIR(info.st_mode) else "file"
        require(kind == "directory" or stat.S_ISREG(info.st_mode), "pilot tree type differs")
        item: dict[str, Any] = {
            "relative_path": path.relative_to(root).as_posix(),
            "kind": kind,
            "device": int(info.st_dev),
            "inode": int(info.st_ino),
            "mode": int(info.st_mode),
            "uid": int(info.st_uid),
            "gid": int(info.st_gid),
            "links": int(info.st_nlink),
            "bytes": int(info.st_size),
            "mtime_ns": int(info.st_mtime_ns),
            "ctime_ns": int(info.st_ctime_ns),
        }
        if kind == "file":
            item["sha256"] = _sha256_regular_file(
                path, field_name="pilot output artifact"
            )[0]
            regular_bytes += int(info.st_size)
            regular_files += 1
        items.append(item)
    return {
        "tree_identity_sha256": _sha256_bytes(_canonical_json_bytes(items)),
        "regular_file_count": regular_files,
        "regular_file_bytes": regular_bytes,
    }


def _private_directory(path: Path, *, field_name: str) -> Path:
    absolute = _assert_no_symlink_components(path, field_name=field_name)
    info = absolute.lstat()
    require(stat.S_ISDIR(info.st_mode), f"{field_name} is not a directory")
    require(info.st_uid == os.geteuid(), f"{field_name} owner differs")
    require(stat.S_IMODE(info.st_mode) == 0o700, f"{field_name} mode is not 0700")
    return absolute


def _make_fresh_private_root(path: Path, *, field_name: str) -> Path:
    absolute = _assert_no_symlink_components(path, field_name=field_name)
    require(not absolute.exists() and not absolute.is_symlink(), f"{field_name} exists")
    try:
        absolute.mkdir(mode=0o700)
    except OSError as exc:
        _fail(f"cannot create {field_name}", exc)
    absolute.chmod(0o700)
    return _private_directory(absolute, field_name=field_name)


def _write_new_bytes(path: Path, payload: bytes) -> None:
    _assert_no_symlink_components(path, field_name="benchmark artifact")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        _fail("cannot create benchmark artifact", exc)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        _fail("cannot durably write benchmark artifact", exc)


def _write_new_json(path: Path, value: Any) -> None:
    _write_new_bytes(path, _published_json_bytes(value))


def _publish_new(path: Path, payload: bytes) -> None:
    absolute = _assert_no_symlink_components(path, field_name="benchmark result")
    parent = _private_directory(absolute.parent, field_name="result parent")
    require(not absolute.exists() and not absolute.is_symlink(), "benchmark result exists")
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    temporary = f".{absolute.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(
            temporary,
            absolute.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        _fail("cannot publish immutable benchmark result", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def _read_proc_status() -> dict[str, int]:
    values = {"rss_kib": 0, "hwm_kib": 0}
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                values["rss_kib"] = int(line.split()[1])
            elif line.startswith("VmHWM:"):
                values["hwm_kib"] = int(line.split()[1])
    except (OSError, ValueError):
        pass
    return values


def _read_proc_io() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/self/io").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip())
    except (OSError, ValueError):
        return {}
    return values


def _open_fd_count() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return -1


class _ResourceSampler:
    def __init__(self, interval_seconds: float = 0.005) -> None:
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.peak_fd = _open_fd_count()
        status = _read_proc_status()
        self.peak_rss_kib = status["rss_kib"]
        self.peak_hwm_kib = status["hwm_kib"]
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample(self) -> None:
        self.peak_fd = max(self.peak_fd, _open_fd_count())
        status = _read_proc_status()
        self.peak_rss_kib = max(self.peak_rss_kib, status["rss_kib"])
        self.peak_hwm_kib = max(self.peak_hwm_kib, status["hwm_kib"])

    def _sample_loop(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join()
        self._sample()


def _rusage_snapshot() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "user_seconds": usage.ru_utime,
        "system_seconds": usage.ru_stime,
        "minor_page_faults": usage.ru_minflt,
        "major_page_faults": usage.ru_majflt,
        "block_inputs": usage.ru_inblock,
        "block_outputs": usage.ru_oublock,
        "voluntary_context_switches": usage.ru_nvcsw,
        "involuntary_context_switches": usage.ru_nivcsw,
    }


def _numeric_delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: round(float(after.get(key, 0)) - float(before.get(key, 0)), 6)
        if isinstance(after.get(key, 0), float) or isinstance(before.get(key, 0), float)
        else int(after.get(key, 0)) - int(before.get(key, 0))
        for key in set(after) | set(before)
    }


@contextmanager
def _measure_resources() -> Iterator[dict[str, Any]]:
    result: dict[str, Any] = {}
    before_usage = _rusage_snapshot()
    before_io = _read_proc_io()
    start = time.perf_counter()
    sampler = _ResourceSampler()
    sampler.start()
    try:
        yield result
    finally:
        sampler.stop()
        result.update(
            {
                "wall_seconds": round(time.perf_counter() - start, 6),
                "process_usage_delta": _numeric_delta(
                    _rusage_snapshot(), before_usage
                ),
                "proc_io_delta": _numeric_delta(_read_proc_io(), before_io),
                "peak_open_file_descriptors_sampled": sampler.peak_fd,
                "peak_rss_kib_sampled": sampler.peak_rss_kib,
                "peak_hwm_kib_observed": sampler.peak_hwm_kib,
                "sampling_interval_seconds": sampler.interval_seconds,
            }
        )


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    require(0 <= probability <= 1, "quantile probability is outside [0, 1]")
    ordered = sorted(float(value) for value in values)
    index = int(math.ceil(probability * len(ordered)) - 1)
    index = max(0, min(index, len(ordered) - 1))
    return round(ordered[index], 9)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "minimum": round(min(values), 9),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "maximum": round(max(values), 9),
        "mean": round(sum(values) / len(values), 9),
    }


@dataclass
class _KernelMetrics:
    argsort_seconds: float = 0.0
    argsort_rows: int = 0
    argsort_calls: int = 0
    fsync_seconds: list[float] = field(default_factory=list)
    heap_seconds: float = 0.0
    heap_push_calls: int = 0
    heap_pop_calls: int = 0
    parquet_writer_seconds: float = 0.0
    parquet_writer_rows: int = 0
    parquet_write_calls: int = 0
    parquet_close_calls: int = 0
    decoded_validation_seconds: float = 0.0
    decoded_validation_rows: int = 0
    decoded_validation_calls: int = 0
    sha256_seconds: float = 0.0
    sha256_bytes: int = 0
    sha256_calls: int = 0

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
            "heap_primitives": {
                "push_calls": self.heap_push_calls,
                "pop_calls": self.heap_pop_calls,
                "seconds": round(self.heap_seconds, 6),
                "full_python_heap_merge_rate_measured": False,
            },
            "parquet_writer_write_and_close": {
                "write_calls": self.parquet_write_calls,
                "close_calls": self.parquet_close_calls,
                "rows": self.parquet_writer_rows,
                "seconds": round(self.parquet_writer_seconds, 6),
                "rows_per_second": _rate(
                    self.parquet_writer_rows, self.parquet_writer_seconds
                ),
                "arrow_table_conversion_included": False,
            },
            "decoded_parquet_validation": {
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
        }


def _rate(numerator: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    return round(numerator / seconds, 6)


class _OSProxy:
    def __init__(self, base: ModuleType, metrics: _KernelMetrics) -> None:
        self._base = base
        self._metrics = metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def fsync(self, descriptor: int) -> None:
        start = time.perf_counter()
        try:
            self._base.fsync(descriptor)
        finally:
            self._metrics.fsync_seconds.append(time.perf_counter() - start)


class _NumpyProxy:
    def __init__(self, base: ModuleType, metrics: _KernelMetrics) -> None:
        self._base = base
        self._metrics = metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def argsort(self, values: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return self._base.argsort(values, *args, **kwargs)
        finally:
            self._metrics.argsort_seconds += time.perf_counter() - start
            self._metrics.argsort_rows += len(values)
            self._metrics.argsort_calls += 1


class _HeapProxy:
    def __init__(self, base: ModuleType, metrics: _KernelMetrics) -> None:
        self._base = base
        self._metrics = metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def heappush(self, heap: list[Any], item: Any) -> None:
        start = time.perf_counter()
        try:
            self._base.heappush(heap, item)
        finally:
            self._metrics.heap_seconds += time.perf_counter() - start
            self._metrics.heap_push_calls += 1

    def heappop(self, heap: list[Any]) -> Any:
        start = time.perf_counter()
        try:
            return self._base.heappop(heap)
        finally:
            self._metrics.heap_seconds += time.perf_counter() - start
            self._metrics.heap_pop_calls += 1


class _ParquetWriterProxy:
    def __init__(self, writer: Any, metrics: _KernelMetrics) -> None:
        self._writer = writer
        self._metrics = metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._writer, name)

    def write_table(self, table: Any, *args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return self._writer.write_table(table, *args, **kwargs)
        finally:
            self._metrics.parquet_writer_seconds += time.perf_counter() - start
            self._metrics.parquet_writer_rows += int(table.num_rows)
            self._metrics.parquet_write_calls += 1

    def close(self) -> Any:
        start = time.perf_counter()
        try:
            return self._writer.close()
        finally:
            self._metrics.parquet_writer_seconds += time.perf_counter() - start
            self._metrics.parquet_close_calls += 1


@contextmanager
def _instrument_runner(runner: ModuleType) -> Iterator[_KernelMetrics]:
    metrics = _KernelMetrics()
    original_os = runner.os
    original_np = runner.np
    original_heapq = runner.heapq
    original_new_writer = runner._new_parquet_writer
    original_validate = runner._stream_validate_parquet
    original_sha = runner._sha256_file

    def new_writer(target: Any) -> _ParquetWriterProxy:
        return _ParquetWriterProxy(original_new_writer(target), metrics)

    def validate(path: Path, *, partition_id: int, expected_rows: int) -> dict[str, Any]:
        start = time.perf_counter()
        try:
            return original_validate(
                path,
                partition_id=partition_id,
                expected_rows=expected_rows,
            )
        finally:
            metrics.decoded_validation_seconds += time.perf_counter() - start
            metrics.decoded_validation_rows += expected_rows
            metrics.decoded_validation_calls += 1

    def sha(path: Path, *, chunk_bytes: int = 16 << 20) -> str:
        size = path.stat().st_size
        start = time.perf_counter()
        try:
            return original_sha(path, chunk_bytes=chunk_bytes)
        finally:
            metrics.sha256_seconds += time.perf_counter() - start
            metrics.sha256_bytes += int(size)
            metrics.sha256_calls += 1

    runner.os = _OSProxy(original_os, metrics)
    runner.np = _NumpyProxy(original_np, metrics)
    runner.heapq = _HeapProxy(original_heapq, metrics)
    runner._new_parquet_writer = new_writer
    runner._stream_validate_parquet = validate
    runner._sha256_file = sha
    try:
        yield metrics
    finally:
        runner.os = original_os
        runner.np = original_np
        runner.heapq = original_heapq
        runner._new_parquet_writer = original_new_writer
        runner._stream_validate_parquet = original_validate
        runner._sha256_file = original_sha


def _available_ram_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def _preflight(scratch_parent: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(scratch_parent)
    nofile_soft, nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    available_ram = _available_ram_bytes()
    require(disk.free >= MINIMUM_FREE_DISK_BYTES, "bounded benchmark disk floor not met")
    require(
        available_ram >= MINIMUM_AVAILABLE_RAM_BYTES,
        "bounded benchmark available-RAM floor not met",
    )
    require(nofile_soft >= MINIMUM_NOFILE, "bounded benchmark nofile floor not met")
    return {
        "observed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "available_disk_bytes": disk.free,
        "available_ram_bytes": available_ram,
        "nofile_soft_limit": nofile_soft,
        "nofile_hard_limit": nofile_hard,
        "required_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "required_available_ram_bytes": MINIMUM_AVAILABLE_RAM_BYTES,
        "required_nofile_soft_limit": MINIMUM_NOFILE,
        "bounded_preflight_passed": True,
        "production_preflight_passed": False,
        "source_drive_health_not_applicable_no_source_opened": True,
        "quiet_window_independently_verified": False,
        "operator_paths_redacted": True,
    }


def _atomic_fsync_probe(root: Path) -> dict[str, Any]:
    pending = root / f".atomic-fsync-probe-{secrets.token_hex(8)}.pending"
    final = root / f".atomic-fsync-probe-{secrets.token_hex(8)}.done"
    directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    started = time.perf_counter()
    try:
        descriptor = os.open(
            pending,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(b"bounded-downstream-benchmark-fsync-probe\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(pending, final)
        os.fsync(directory_fd)
        final.unlink()
        os.fsync(directory_fd)
    except OSError as exc:
        _fail("atomic rename/fsync probe failed", exc)
    finally:
        if pending.exists():
            pending.unlink()
        if final.exists():
            final.unlink()
        os.close(directory_fd)
    return {
        "status": "PASS_BOUNDED_SCRATCH_ATOMIC_RENAME_FSYNC",
        "wall_seconds": round(time.perf_counter() - started, 6),
    }


def _validate_pilot(
    pilot_root: Path,
    verifier: ModuleType,
) -> tuple[dict[str, Any], dict[str, Any], Path, str]:
    try:
        verification = verifier.verify_output(pilot_root, parquet_validation="full")
    except BaseException as exc:
        _fail("independent pilot verification failed", exc)
    require(verification.get("verified_output_scope") == PILOT_SCOPE, "pilot scope differs")
    require(verification.get("output_tree_unchanged") is True, "verifier mutated pilot")
    require(verification.get("production_component_pass") is False, "pilot claims production pass")
    report, _ = _read_strict_json(
        pilot_root / "work_time_spine_pilot.json", field_name="pilot report"
    )
    require(report.get("status") == PILOT_STATUS, "pilot report status differs")
    scope = report.get("scope")
    require(isinstance(scope, dict), "pilot report scope is not a mapping")
    require(scope.get("execution_scope") == PILOT_SCOPE, "pilot execution scope differs")
    require(scope.get("production_status") == PRODUCTION_NOT_RUN, "pilot production status differs")
    require(scope.get("retained_rows") == 1_000_000, "pilot retained-row anchor differs")
    provenance = report.get("provenance")
    require(isinstance(provenance, dict), "pilot provenance is not a mapping")
    require(provenance.get("tool_sha256") == RUNNER_SHA256, "pilot runner binding differs")
    require(provenance.get("config_sha256") == CONFIG_SHA256, "pilot config binding differs")
    for key in SHA256_KEYS:
        require(
            isinstance(provenance.get(key), str) and len(provenance[key]) == 64,
            f"pilot provenance {key} differs",
        )
    manifest, _ = _read_strict_json(
        pilot_root / "staging_manifest.json", field_name="pilot staging manifest"
    )
    shards = manifest.get("shards")
    require(isinstance(shards, list) and len(shards) == 1, "pilot staging shard count differs")
    shard = shards[0]
    require(isinstance(shard, dict), "pilot staging shard contract differs")
    relative = shard.get("relative_path")
    require(isinstance(relative, str) and not relative.startswith("/"), "pilot staging path differs")
    staging_path = pilot_root / relative
    staging_sha, staging_bytes = _sha256_regular_file(
        staging_path, field_name="pilot staging payload"
    )
    require(staging_sha == shard.get("sha256"), "pilot staging SHA-256 differs")
    require(staging_bytes == shard.get("bytes") == 16_000_000, "pilot staging bytes differ")
    require(shard.get("rows") * RECORD_BYTES == staging_bytes, "pilot staging ABI differs")
    receipt_sha, _ = _sha256_regular_file(
        pilot_root / "work_time_spine_pilot.outputs.json",
        field_name="pilot receipt",
    )
    return verification, report, staging_path, receipt_sha


def _shard_row_counts(total_rows: int, shard_count: int) -> list[int]:
    require(total_rows >= shard_count > 0, "benchmark shard cardinality is invalid")
    base, remainder = divmod(total_rows, shard_count)
    return [base + (1 if index < remainder else 0) for index in range(shard_count)]


def _prepare_benchmark_shards(
    *,
    runner: ModuleType,
    source_staging_path: Path,
    workload_root: Path,
    source_rows: int,
    source_sha256: str,
    shard_count: int = FULL_SCAN_SHARD_COUNT,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    staging_root = workload_root / "staging/main"
    checkpoint_root = workload_root / "checkpoints/scan/main"
    for directory in (staging_root, checkpoint_root):
        directory.mkdir(parents=True, mode=0o700)
        directory.chmod(0o700)
    row_counts = _shard_row_counts(source_rows, shard_count)
    checkpoints: list[dict[str, Any]] = []
    artifact_paths: list[str] = []
    source_digest = hashlib.sha256()
    source_descriptor = os.open(
        source_staging_path, os.O_RDONLY | os.O_NOFOLLOW
    )
    started = time.perf_counter()
    try:
        for index, rows in enumerate(row_counts):
            expected_bytes = rows * RECORD_BYTES
            payload = bytearray()
            while len(payload) < expected_bytes:
                block = os.read(source_descriptor, expected_bytes - len(payload))
                require(block, "pilot staging ended during benchmark shard split")
                payload.extend(block)
            encoded = bytes(payload)
            source_digest.update(encoded)
            records = runner.np.frombuffer(encoded, dtype=runner.RECORD_DTYPE)
            runner._validate_record_array(records)
            record_digest = hashlib.sha256()
            record_digest.update(len(runner.RECORD_HASH_DOMAIN).to_bytes(8, "little"))
            record_digest.update(runner.RECORD_HASH_DOMAIN)
            record_digest.update(rows.to_bytes(8, "little"))
            runner._update_framed_record_digest(record_digest, records)
            logical_relative = f"benchmark-shard-{index:04d}"
            token = _sha256_bytes(logical_relative.encode("utf-8"))
            staging_relative = f"staging/main/{token}.bin"
            checkpoint_relative = f"checkpoints/scan/main/{token}.json"
            raw_sha = _sha256_bytes(encoded)
            _write_new_bytes(workload_root / staging_relative, encoded)
            checkpoint = {
                "schema_version": (
                    "openalex_g1_canonical_work_time_spine_downstream_"
                    "benchmark_scan_shard.v1"
                ),
                "benchmark_shard_index": index,
                "relative_path": logical_relative,
                "staging_shard_relative_path": staging_relative,
                "staging_shard_rows": rows,
                "staging_shard_bytes": expected_bytes,
                "staging_shard_sha256": raw_sha,
                "row_level_disposition_payload_sha256": record_digest.hexdigest(),
                "operator_paths_redacted": True,
            }
            _write_new_json(workload_root / checkpoint_relative, checkpoint)
            checkpoints.append(checkpoint)
            artifact_paths.extend((staging_relative, checkpoint_relative))
        require(os.read(source_descriptor, 1) == b"", "pilot staging has trailing bytes")
    finally:
        os.close(source_descriptor)
    require(source_digest.hexdigest() == source_sha256, "split staging source SHA differs")
    require(sum(row_counts) == source_rows, "benchmark shard row conservation differs")
    return checkpoints, artifact_paths, {
        "wall_seconds": round(time.perf_counter() - started, 6),
        "shard_count": shard_count,
        "rows": source_rows,
        "bytes": source_rows * RECORD_BYTES,
        "source_sha256": source_sha256,
        "excluded_from_downstream_phase_timings": True,
    }


def _occupancy_summary(counts: Sequence[int]) -> dict[str, Any]:
    require(len(counts) == PARTITION_COUNT, "partition occupancy cardinality differs")
    nonempty = [value for value in counts if value > 0]
    return {
        "partition_count": len(counts),
        "nonempty_partition_count": len(nonempty),
        "empty_partition_count": len(counts) - len(nonempty),
        "all_partitions": _distribution([float(value) for value in counts]),
        "nonempty_partitions": _distribution([float(value) for value in nonempty]),
        "partition_rows_sha256": _sha256_bytes(_canonical_json_bytes(list(counts))),
    }


def _run_demultiplex(
    runner: ModuleType,
    workload_root: Path,
    checkpoints: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with _instrument_runner(runner) as kernel, _measure_resources() as resources:
        spool_manifest = runner._build_partition_spools(workload_root, checkpoints)
    counts = spool_manifest["partition_rows"]
    require(spool_manifest.get("partition_count") == PARTITION_COUNT, "spool count differs")
    require(spool_manifest.get("rows") == 1_000_000, "spool row count differs")
    require(spool_manifest.get("bytes") == 16_000_000, "spool byte count differs")
    require(sum(counts) == 1_000_000, "spool conservation differs")
    return spool_manifest, {
        "scenario": "D4096",
        "status": "PASS_BOUNDED_4096_WAY_DEMULTIPLEX_INTEGRITY",
        "rows": spool_manifest["rows"],
        "logical_input_bytes": spool_manifest["bytes"],
        "logical_spool_bytes": spool_manifest["bytes"],
        "rows_per_second_wall": _rate(
            int(spool_manifest["rows"]), float(resources["wall_seconds"])
        ),
        "occupancy": _occupancy_summary(counts),
        "kernel_metrics": kernel.summary(),
        "resources": resources,
        "full_corpus_sustained_rate_measured": False,
    }


def _run_representative_partitions(
    runner: ModuleType,
    workload_root: Path,
    spool_manifest: Mapping[str, Any],
    *,
    repeat_index: int,
    pilot_task_fingerprint_sha256: str,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    for relative in (
        "checkpoints/partition",
        "spine",
        ".work_time_spine_tmp/runs",
        ".work_time_spine_tmp/sorted",
    ):
        directory = workload_root / relative
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.chmod(0o700)
    spool_contracts = spool_manifest["partitions"]
    require(len(spool_contracts) == PARTITION_COUNT, "spool contract count differs")
    task_fingerprint = _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema": SCHEMA,
                "repeat_index": repeat_index,
                "pilot_task_fingerprint_sha256": pilot_task_fingerprint_sha256,
            }
        )
    )
    # These values are benchmark-only lineage labels.  They bind the returned
    # partition metadata to this exact spool surface without impersonating a
    # production scan-checkpoint or staging-manifest tree.
    scan_tree_sha = _sha256_bytes(
        _canonical_json_bytes(
            {
                "benchmark_spool_partition_tree_sha256": spool_manifest[
                    "partition_tree_sha256"
                ],
                "repeat_index": repeat_index,
            }
        )
    )
    staging_tree_sha = str(spool_manifest["partition_tree_sha256"])
    checkpoints: list[dict[str, Any]] = []
    artifact_paths: list[str] = []
    partition_seconds: list[float] = []
    run_counts: list[int] = []
    with _instrument_runner(runner) as kernel, _measure_resources() as resources:
        for partition_id in range(PARTITION_COUNT):
            partition_started = time.perf_counter()
            runs, observed_rows = runner._write_sorted_runs(
                workload_root,
                partition_id,
                sort_chunk_rows=SORT_CHUNK_ROWS,
                spool_contract=spool_contracts[partition_id],
            )
            expected_rows = int(spool_contracts[partition_id]["rows"])
            require(observed_rows == expected_rows, "sort-run row count differs")
            run_counts.append(len(runs))
            checkpoint = runner._merge_runs_to_temporary_parquet(
                workload_root,
                partition_id,
                runs,
                expected_rows,
                task_fingerprint_sha256=task_fingerprint,
                checkpoint_schema=(
                    "openalex_g1_canonical_work_time_spine_downstream_"
                    "benchmark_partition_checkpoint.v1"
                ),
                scan_tree_sha256=scan_tree_sha,
                staging_tree_sha256=staging_tree_sha,
            )
            require(
                checkpoint["pre_parquet_sorted_record_sha256"]
                == checkpoint["decoded_parquet_record_sha256"],
                "pre-/post-Parquet record digest differs",
            )
            require(checkpoint["output_rows"] == expected_rows, "Parquet rows differ")
            temporary = (
                workload_root
                / ".work_time_spine_tmp/sorted"
                / f"part-{partition_id:04d}.parquet"
            )
            final_relative = f"spine/part-{partition_id:04d}.parquet"
            final = workload_root / final_relative
            os.replace(temporary, final)
            runner._fsync_directory(final.parent)
            checkpoint_relative = f"checkpoints/partition/part-{partition_id:04d}.json"
            runner._atomic_json(workload_root / checkpoint_relative, checkpoint)
            run_root = (
                workload_root
                / ".work_time_spine_tmp/runs"
                / f"part-{partition_id:04d}"
            )
            if run_root.exists():
                shutil.rmtree(run_root)
            checkpoints.append(checkpoint)
            artifact_paths.extend((final_relative, checkpoint_relative))
            partition_seconds.append(time.perf_counter() - partition_started)
    require(sum(int(item["output_rows"]) for item in checkpoints) == 1_000_000, "R4096 row conservation differs")
    require(max(run_counts, default=0) <= 1, "pilot occupancy unexpectedly used multi-run merge")
    parquet_bytes = sum(int(item["output_bytes"]) for item in checkpoints)
    return checkpoints, artifact_paths, {
        "scenario": "R4096",
        "status": "PASS_BOUNDED_PILOT_OCCUPANCY_SINGLE_RUN_PARQUET_INTEGRITY",
        "partition_count": PARTITION_COUNT,
        "rows": 1_000_000,
        "parquet_bytes": parquet_bytes,
        "run_count_total": sum(run_counts),
        "run_count_maximum_per_partition": max(run_counts, default=0),
        "multi_run_partitions": sum(value > 1 for value in run_counts),
        "single_run_partitions": sum(value == 1 for value in run_counts),
        "zero_run_partitions": sum(value == 0 for value in run_counts),
        "partition_wall_seconds": _distribution(partition_seconds),
        "rows_per_second_wall": _rate(1_000_000, float(resources["wall_seconds"])),
        "kernel_metrics": kernel.summary(),
        "resources": resources,
        "full_like_partition_occupancy_measured": False,
        "python_multi_run_heap_merge_measured": False,
    }


def _write_control_surface(
    *,
    workload_root: Path,
    repeat_index: int,
    preflight: Mapping[str, Any],
    preparation: Mapping[str, Any],
    spool_manifest: Mapping[str, Any],
    partition_checkpoints: Sequence[Mapping[str, Any]],
    demultiplex: Mapping[str, Any],
    representative: Mapping[str, Any],
) -> list[str]:
    staging_manifest = {
        "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_staging_manifest.v1",
        "repeat_index": repeat_index,
        "shard_count": FULL_SCAN_SHARD_COUNT,
        "rows": preparation["rows"],
        "bytes": preparation["bytes"],
        "spool_partition_tree_sha256": spool_manifest["partition_tree_sha256"],
        "operator_paths_redacted": True,
    }
    partition_contracts = [
        {
            "partition_id": item["partition_id"],
            "rows": item["output_rows"],
            "bytes": item["output_bytes"],
            "sha256": item["output_sha256"],
            "record_sha256": item["decoded_parquet_record_sha256"],
        }
        for item in partition_checkpoints
    ]
    spine_manifest = {
        "schema_version": "openalex_g1_canonical_work_time_spine_downstream_benchmark_spine_manifest.v1",
        "repeat_index": repeat_index,
        "partition_count": PARTITION_COUNT,
        "rows": sum(int(item["rows"]) for item in partition_contracts),
        "bytes": sum(int(item["bytes"]) for item in partition_contracts),
        "partition_contract_tree_sha256": _sha256_bytes(
            _canonical_json_bytes(partition_contracts)
        ),
        "operator_paths_redacted": True,
    }
    documents: dict[str, Any] = {
        "benchmark_plan.json": {
            "schema_version": SCHEMA,
            "status": "BOUNDED_NONPROMOTABLE_PLAN",
            "repeat_index": repeat_index,
            "scenarios": ["D4096", "R4096", "H9185_BOUNDED_BYTES"],
            "production_execution": PRODUCTION_NOT_RUN,
            "operator_paths_redacted": True,
        },
        "preflight_snapshot.json": dict(preflight),
        "run_state.json": {
            "schema_version": f"{SCHEMA}.state",
            "status": "BOUNDED_REPEAT_COMPLETE_NO_PRODUCTION_RECEIPT",
            "repeat_index": repeat_index,
            "production_execution": PRODUCTION_NOT_RUN,
            "operator_paths_redacted": True,
        },
        "staging_manifest.json": staging_manifest,
        "spine_manifest.json": spine_manifest,
        "benchmark_report.json": {
            "schema_version": f"{SCHEMA}.repeat_report",
            "status": STATUS,
            "repeat_index": repeat_index,
            "demultiplex_status": demultiplex["status"],
            "representative_status": representative["status"],
            "production_component_pass": False,
            "programmatic_parent_of_full_receipt": False,
            "operator_paths_redacted": True,
        },
    }
    for relative, value in documents.items():
        _write_new_json(workload_root / relative, value)
    _write_new_bytes(
        workload_root / "benchmark_report.md",
        (
            "# Bounded downstream benchmark repeat\n\n"
            f"Repeat: {repeat_index}\n\n"
            "Production execution: PRODUCTION_NOT_RUN\n\n"
            "This scratch surface is operator-local and nonpromotable.\n"
        ).encode("utf-8"),
    )
    return list(CONTROL_PATHS)


def _walk_regular_files(root: Path) -> list[str]:
    observed: list[str] = []
    for path in root.rglob("*"):
        info = path.lstat()
        require(not stat.S_ISLNK(info.st_mode), "H9185 surface contains a symlink")
        if stat.S_ISREG(info.st_mode):
            require(info.st_nlink == 1, "H9185 surface contains a hard link")
            require(stat.S_IMODE(info.st_mode) == 0o600, "H9185 file mode differs")
            observed.append(path.relative_to(root).as_posix())
        else:
            require(stat.S_ISDIR(info.st_mode), "H9185 surface has unknown type")
            require(stat.S_IMODE(info.st_mode) == 0o700, "H9185 directory mode differs")
    return sorted(observed)


def _run_h9185(
    *,
    runner: ModuleType,
    workload_root: Path,
    expected_paths: Sequence[str],
    sweep_count: int,
) -> dict[str, Any]:
    require(len(expected_paths) == EXACT_ARTIFACT_COUNT, "H9185 expected count differs")
    require(len(set(expected_paths)) == EXACT_ARTIFACT_COUNT, "H9185 expected paths repeat")
    observed_paths = _walk_regular_files(workload_root)
    require(observed_paths == sorted(expected_paths), "H9185 filesystem topology differs")
    sweeps: list[dict[str, Any]] = []
    reference_tree: str | None = None
    for sweep_index in range(1, sweep_count + 1):
        with _measure_resources() as resources:
            contracts = [
                runner._artifact_contract(workload_root, relative)
                for relative in observed_paths
            ]
        total_bytes = sum(int(item["bytes"]) for item in contracts)
        tree_sha = _sha256_bytes(_canonical_json_bytes(contracts))
        if reference_tree is None:
            reference_tree = tree_sha
        require(tree_sha == reference_tree, "H9185 contract tree changed across sweeps")
        wall = float(resources["wall_seconds"])
        sweeps.append(
            {
                "sweep_index": sweep_index,
                "artifact_count": len(contracts),
                "hashed_bytes": total_bytes,
                "artifact_contract_tree_sha256": tree_sha,
                "artifacts_per_second": _rate(len(contracts), wall),
                "bytes_per_second": _rate(total_bytes, wall),
                "resources": resources,
            }
        )
    return {
        "scenario": "H9185_BOUNDED_BYTES",
        "status": "PASS_EXACT_9185_ENTRY_TOPOLOGY_AND_HASH_SWEEPS_BOUNDED_BYTES",
        "artifact_count": EXACT_ARTIFACT_COUNT,
        "topology_formula": "493_staging+493_scan_metadata+4096_parquet+4096_partition_metadata+7_control",
        "sweep_count": sweep_count,
        "sweeps": sweeps,
        "large_full_byte_surface_measured": False,
        "production_finalizer_or_receipt_called": False,
        "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
    }


def _run_repeat(
    *,
    runner: ModuleType,
    scratch_root: Path,
    repeat_index: int,
    source_staging_path: Path,
    source_sha256: str,
    pilot_task_fingerprint_sha256: str,
    preflight: Mapping[str, Any],
    artifact_sweeps: int,
) -> dict[str, Any]:
    workload_root = scratch_root / f"repeat-{repeat_index:02d}"
    workload_root.mkdir(mode=0o700)
    workload_root.chmod(0o700)
    checkpoints, expected_paths, preparation = _prepare_benchmark_shards(
        runner=runner,
        source_staging_path=source_staging_path,
        workload_root=workload_root,
        source_rows=1_000_000,
        source_sha256=source_sha256,
    )
    spool_manifest, demultiplex = _run_demultiplex(
        runner, workload_root, checkpoints
    )
    partition_checkpoints, partition_paths, representative = (
        _run_representative_partitions(
            runner,
            workload_root,
            spool_manifest,
            repeat_index=repeat_index,
            pilot_task_fingerprint_sha256=pilot_task_fingerprint_sha256,
        )
    )
    expected_paths.extend(partition_paths)
    expected_paths.extend(
        _write_control_surface(
            workload_root=workload_root,
            repeat_index=repeat_index,
            preflight=preflight,
            preparation=preparation,
            spool_manifest=spool_manifest,
            partition_checkpoints=partition_checkpoints,
            demultiplex=demultiplex,
            representative=representative,
        )
    )
    temporary = workload_root / ".work_time_spine_tmp"
    require(temporary.is_dir() and not temporary.is_symlink(), "temporary tree differs")
    shutil.rmtree(temporary)
    h9185 = _run_h9185(
        runner=runner,
        workload_root=workload_root,
        expected_paths=expected_paths,
        sweep_count=artifact_sweeps,
    )
    return {
        "repeat_index": repeat_index,
        "status": "PASS_BOUNDED_REPEAT_INTEGRITY",
        "preparation": preparation,
        "demultiplex": demultiplex,
        "representative_single_run": representative,
        "bounded_artifact_surface": h9185,
        "scratch_surface_retained_for_operator_review": True,
        "operator_paths_redacted": True,
    }


def _aggregate_repeats(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    demux_rates = [
        float(item["demultiplex"]["rows_per_second_wall"]) for item in repeats
    ]
    representative_rates = [
        float(item["representative_single_run"]["rows_per_second_wall"])
        for item in repeats
    ]
    h_rates = [
        float(sweep["artifacts_per_second"])
        for item in repeats
        for sweep in item["bounded_artifact_surface"]["sweeps"]
    ]
    return {
        "repeat_count": len(repeats),
        "demultiplex_rows_per_second": _distribution(demux_rates),
        "representative_rows_per_second": _distribution(representative_rates),
        "h9185_artifacts_per_second": _distribution(h_rates),
        "one_sided_95_percent_lower_confidence_bound_computed": False,
        "minimum_observed_rates_are_not_adopted_capacity_floors": True,
        "independent_quiet_window_sessions": 0,
    }


def _result_fingerprint(result: Mapping[str, Any]) -> str:
    payload = dict(result)
    payload.pop("result_fingerprint_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def _validate_final_result(result: Mapping[str, Any]) -> None:
    require(result.get("schema_version") == SCHEMA, "result schema differs")
    require(result.get("status") == STATUS, "result status differs")
    require(result.get("production_execution_status") == PRODUCTION_NOT_RUN, "result claims production execution")
    for key in (
        "production_component_pass",
        "pass_g1",
        "full_authorization_sufficient",
        "downstream_capacity_gate_passed",
        "programmatic_parent_of_full_receipt",
    ):
        require(result.get(key) is False, f"result claim boundary differs: {key}")
    require(result.get("operator_paths_redacted") is True, "result paths are not redacted")
    require(result.get("pilot_output_tree_unchanged") is True, "pilot tree changed")
    require(
        result.get("result_fingerprint_sha256") == _result_fingerprint(result),
        "result fingerprint differs",
    )


def run_benchmark(
    *,
    pilot_output_root: Path,
    scratch_root: Path,
    output_path: Path,
    repeats: int,
    artifact_sweeps: int,
    execute_bounded_nonpromotable: bool,
) -> dict[str, Any]:
    require(execute_bounded_nonpromotable, "explicit bounded execution flag is required")
    require(1 <= repeats <= 5, "repeat count must be between 1 and 5")
    require(1 <= artifact_sweeps <= 5, "artifact sweep count must be between 1 and 5")
    pilot_root = _absolute(pilot_output_root)
    scratch = _absolute(scratch_root)
    output = _absolute(output_path)
    require(output.name == OUTPUT_BASENAME, "result basename differs")
    _private_directory(pilot_root, field_name="pilot output root")
    _private_directory(output.parent, field_name="result parent")
    require(not output.exists() and not output.is_symlink(), "result output exists")
    for left, right, message in (
        (pilot_root, REPO_ROOT, "pilot output overlaps repository"),
        (scratch, REPO_ROOT, "scratch overlaps repository"),
        (output, REPO_ROOT, "result overlaps repository"),
        (scratch, pilot_root, "scratch overlaps pilot output"),
        (output, pilot_root, "result overlaps pilot output"),
        (scratch, output.parent, "scratch overlaps result root"),
    ):
        require(not _paths_overlap(left, right), message)
    config_sha, _ = _sha256_regular_file(CONFIG_PATH, field_name="frozen config")
    require(config_sha == CONFIG_SHA256, "frozen config SHA-256 binding differs")
    runner = _load_bound_module("_downstream_benchmark_runner_v3", RUNNER_PATH, RUNNER_SHA256)
    verifier = _load_bound_module(
        "_downstream_benchmark_independent_verifier_v1",
        VERIFIER_PATH,
        VERIFIER_SHA256,
    )
    verification_before, report, source_staging, receipt_sha = _validate_pilot(
        pilot_root, verifier
    )
    tree_before = _tree_identity_digest(pilot_root)
    source_sha, source_bytes = _sha256_regular_file(
        source_staging, field_name="pilot staging payload"
    )
    require(source_bytes == 16_000_000, "pilot staging byte anchor differs")
    preflight = _preflight(scratch.parent)
    scratch = _make_fresh_private_root(scratch, field_name="scratch root")
    preflight["atomic_rename_fsync_probe"] = _atomic_fsync_probe(scratch)
    tool_sha_before, _ = _sha256_regular_file(Path(__file__), field_name="benchmark tool")
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    repeat_results = [
        _run_repeat(
            runner=runner,
            scratch_root=scratch,
            repeat_index=index,
            source_staging_path=source_staging,
            source_sha256=source_sha,
            pilot_task_fingerprint_sha256=str(report["task_fingerprint_sha256"]),
            preflight=preflight,
            artifact_sweeps=artifact_sweeps,
        )
        for index in range(1, repeats + 1)
    ]
    verification_after, _, _, receipt_sha_after = _validate_pilot(pilot_root, verifier)
    tree_after = _tree_identity_digest(pilot_root)
    require(verification_after == verification_before, "pilot verification surfaces changed")
    require(receipt_sha_after == receipt_sha, "pilot receipt changed")
    require(tree_after == tree_before, "pilot output tree identity changed")
    tool_sha_after, _ = _sha256_regular_file(Path(__file__), field_name="benchmark tool")
    require(tool_sha_after == tool_sha_before, "benchmark tool changed during execution")
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "implementation_binding": {
            "runner_sha256": RUNNER_SHA256,
            "config_sha256": CONFIG_SHA256,
            "independent_verifier_sha256": VERIFIER_SHA256,
            "benchmark_tool_sha256": tool_sha_before,
            "loaded_code_attestation": False,
            "private_downstream_api_used": True,
            "production_finalizer_called": False,
            "production_receipt_builder_called": False,
        },
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform_machine": platform.machine(),
            "runner_runtime_fingerprint": runner._runtime_fingerprint(),
        },
        "pilot_binding": {
            "receipt_sha256": receipt_sha,
            "task_fingerprint_sha256": report["task_fingerprint_sha256"],
            "result_fingerprint_sha256": report["provenance"]["result_fingerprint_sha256"],
            "staging_sha256": source_sha,
            "staging_rows": 1_000_000,
            "staging_bytes": source_bytes,
            "independent_verifier_status": verification_before["status"],
            "independent_verifier_artifact_count": verification_before["artifact_count"],
            "tree_before": tree_before,
            "tree_after": tree_after,
        },
        "preflight": preflight,
        "benchmark_scope": {
            "scenarios": ["D4096", "R4096", "H9185_BOUNDED_BYTES"],
            "repeat_count": repeats,
            "artifact_sweeps_per_repeat": artifact_sweeps,
            "source_hdd_opened": False,
            "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
            "full_like_partition_occupancy_measured": False,
            "multi_run_heap_merge_measured": False,
            "full_byte_finalization_surface_measured": False,
            "independent_quiet_window_sessions": 0,
        },
        "repeats": repeat_results,
        "aggregate": _aggregate_repeats(repeat_results),
        "pilot_output_tree_unchanged": True,
        "production_execution_status": PRODUCTION_NOT_RUN,
        "production_component_pass": False,
        "pass_g1": False,
        "full_authorization_sufficient": False,
        "downstream_capacity_gate_passed": False,
        "programmatic_parent_of_full_receipt": False,
        "operator_acceptance_status": "NOT_RECORDED",
        "strongest_permitted_interpretation": (
            "PARTIAL_BOUNDED_FIXED_OVERHEAD_SINGLE_RUN_AND_EXACT_ENTRY_COUNT_"
            "EVIDENCE_ONLY"
        ),
        "operator_paths_redacted": True,
    }
    result["result_fingerprint_sha256"] = _result_fingerprint(result)
    _validate_final_result(result)
    payload = _published_json_bytes(result)
    require(len(payload) <= MAX_RESULT_BYTES, "benchmark result exceeds its size bound")
    _publish_new(output, payload)
    observed_sha, observed_bytes = _sha256_regular_file(
        output, field_name="published benchmark result"
    )
    require(observed_bytes == len(payload), "published result byte count differs")
    require(observed_sha == _sha256_bytes(payload), "published result SHA-256 differs")
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-output-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--artifact-sweeps", type=int, default=3)
    parser.add_argument(
        "--execute-bounded-nonpromotable",
        action="store_true",
        help="explicitly authorize only this bounded operator-local benchmark",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        result = run_benchmark(
            pilot_output_root=args.pilot_output_root,
            scratch_root=args.scratch_root,
            output_path=args.output,
            repeats=args.repeats,
            artifact_sweeps=args.artifact_sweeps,
            execute_bounded_nonpromotable=args.execute_bounded_nonpromotable,
        )
    except DownstreamBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    payload = _published_json_bytes(
        {
            "schema_version": result["schema_version"],
            "status": result["status"],
            "repeat_count": len(result["repeats"]),
            "result_fingerprint_sha256": result["result_fingerprint_sha256"],
            "production_execution_status": result["production_execution_status"],
            "downstream_capacity_gate_passed": False,
        }
    )
    print(payload.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
