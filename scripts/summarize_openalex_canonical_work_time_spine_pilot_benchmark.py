#!/usr/bin/env python3
"""Build the operator-local, nonpromotable canonical-spine pilot benchmark.

This tool does not run the production runner and never opens a Parquet file. It
combines an already published one-file pilot report with an external
``/usr/bin/time`` capture. The output is a planning diagnostic, not a
production receipt, a downstream-capacity review, or a full-run authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from decimal import Decimal, DecimalException, ROUND_HALF_UP, localcontext
from pathlib import Path
from typing import Any, NoReturn


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "openalex_g1_canonical_work_time_spine_pilot_benchmark.v1"
STATUS = "PILOT_BENCHMARK_CAPTURED_OPERATOR_LOCAL_NONPROMOTABLE"
PILOT_REPORT_SCHEMA = "openalex_g1_canonical_work_time_spine_pilot_report.v1"
PILOT_STATUS = "PILOT_CURRENT_SNAPSHOT_CANONICAL_WORK_TIME_SPINE_NONPROMOTABLE"
PILOT_SCOPE = "PRODUCTION_SOURCE_PREFIX_PILOT_UNPROMOTABLE"
PRODUCTION_NOT_RUN = "PRODUCTION_NOT_RUN"
FULL_SOURCE_ROWS = 492_293_207
PILOT_REPORT_BASENAME = "work_time_spine_pilot.json"
PILOT_RECEIPT_ARTIFACT_COUNT = 8
OUTPUT_BASENAME = "openalex_g1_canonical_work_time_spine_pilot_benchmark.json"
TIME_CAPTURE_BASENAME = "openalex_g1_canonical_work_time_spine_pilot_time.txt"
TIME_CAPTURE_KEYS = (
    "elapsed_seconds",
    "maximum_rss_kib",
    "filesystem_inputs",
    "filesystem_outputs",
)
DECIMAL_PATTERN = re.compile(r"(?:0|[1-9][0-9]{0,11})(?:\.[0-9]{1,6})?")
INTEGER_PATTERN = re.compile(r"(?:0|[1-9][0-9]{0,18})")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
ROUNDING_QUANTUM = Decimal("0.000001")
MAX_COUNTER = (1 << 63) - 1
MAX_PILOT_REPORT_BYTES = 16 * 1024 * 1024
MAX_TIME_CAPTURE_BYTES = 1024
MAX_BENCHMARK_BYTES = 64 * 1024


class PilotBenchmarkError(RuntimeError):
    """The pilot benchmark inputs or output boundary are invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotBenchmarkError(message)


def _fail(message: str, cause: BaseException | None = None) -> NoReturn:
    if cause is None:
        raise PilotBenchmarkError(message)
    raise PilotBenchmarkError(message) from cause


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _assert_no_symlink_components(path: Path, *, field: str) -> Path:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            info = current.lstat()
        except FileNotFoundError:
            break
        except OSError as exc:
            _fail(f"cannot inspect {field}", exc)
        require(not stat.S_ISLNK(info.st_mode), f"{field} contains a symlink")
    return absolute


def _identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_uid),
        int(info.st_gid),
        int(info.st_nlink),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _read_regular_snapshot(
    path: Path,
    *,
    field: str,
    max_bytes: int,
) -> tuple[bytes, str]:
    absolute = _assert_no_symlink_components(path, field=field)
    try:
        descriptor = os.open(absolute, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        _fail(f"cannot open {field}", exc)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        start = os.fstat(descriptor)
        require(stat.S_ISREG(start.st_mode), f"{field} is not a regular file")
        require(start.st_uid == os.geteuid(), f"{field} owner differs from executor")
        require(
            stat.S_IMODE(start.st_mode) & 0o077 == 0,
            f"{field} is readable or writable by group/other",
        )
        require(0 < start.st_size <= max_bytes, f"{field} size is outside its bound")
        start_identity = _identity(start)
        observed_bytes = 0
        while block := os.read(descriptor, 1 << 20):
            observed_bytes += len(block)
            require(observed_bytes <= max_bytes, f"{field} grew beyond its bound")
            chunks.append(block)
            digest.update(block)
        require(_identity(os.fstat(descriptor)) == start_identity, f"{field} changed")
        require(
            _identity(absolute.lstat()) == start_identity,
            f"{field} path identity changed",
        )
    except OSError as exc:
        _fail(f"cannot read {field}", exc)
    finally:
        os.close(descriptor)
    return b"".join(chunks), digest.hexdigest()


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


def _report_fingerprint(report: dict[str, Any]) -> str:
    payload = dict(report)
    provenance = payload.get("provenance")
    require(isinstance(provenance, dict), "pilot provenance must be a mapping")
    fingerprint_payload = dict(provenance)
    fingerprint_payload.pop("result_fingerprint_sha256", None)
    payload["provenance"] = fingerprint_payload
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_pilot_report(path: Path) -> tuple[dict[str, Any], str]:
    require(path.name == PILOT_REPORT_BASENAME, "pilot-report basename differs")
    payload, digest = _read_regular_snapshot(
        path,
        field="pilot report",
        max_bytes=MAX_PILOT_REPORT_BYTES,
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        _fail("cannot parse pilot report", exc)
    require(isinstance(value, dict), "pilot report must be a mapping")
    try:
        canonical_payload = _published_json_bytes(value)
    except (TypeError, ValueError) as exc:
        _fail("pilot report is not exact canonical published JSON", exc)
    require(
        payload == canonical_payload,
        "pilot report is not exact canonical published JSON",
    )
    return dict(value), digest


def _parse_time_capture(path: Path) -> tuple[dict[str, Decimal | int], str]:
    require(path.name == TIME_CAPTURE_BASENAME, "time-capture basename differs")
    payload, digest = _read_regular_snapshot(
        path,
        field="time capture",
        max_bytes=MAX_TIME_CAPTURE_BYTES,
    )
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as exc:
        _fail("time capture is not ASCII", exc)
    require(text.endswith("\n"), "time capture must end with LF")
    require("\r" not in text, "time capture must use LF-only line endings")
    lines = text[:-1].split("\n")
    require(len(lines) == len(TIME_CAPTURE_KEYS), "time capture line count differs")
    observed: dict[str, str] = {}
    for expected_key, line in zip(TIME_CAPTURE_KEYS, lines, strict=True):
        key, separator, raw = line.partition("=")
        require(separator == "=", "time capture line has no equals sign")
        require(key == expected_key, "time capture key order or name differs")
        require(key not in observed, "time capture key repeats")
        observed[key] = raw
    require(
        DECIMAL_PATTERN.fullmatch(observed["elapsed_seconds"]) is not None,
        "elapsed seconds is not a nonnegative decimal",
    )
    for key in TIME_CAPTURE_KEYS[1:]:
        require(
            INTEGER_PATTERN.fullmatch(observed[key]) is not None,
            f"{key} is not a nonnegative integer",
        )
    elapsed = Decimal(observed["elapsed_seconds"])
    require(elapsed.is_finite() and elapsed > 0, "elapsed seconds must be positive")
    counters = {key: int(observed[key]) for key in TIME_CAPTURE_KEYS[1:]}
    for key, value in counters.items():
        require(value <= MAX_COUNTER, f"{key} exceeds the supported bound")
    return {
        "elapsed_seconds": elapsed,
        **counters,
    }, digest


def _require_exact_pilot_report(report: dict[str, Any]) -> tuple[int, int]:
    require(
        report.get("schema_version") == PILOT_REPORT_SCHEMA,
        "pilot report schema differs",
    )
    require(report.get("status") == PILOT_STATUS, "pilot report status differs")
    require(
        report.get("production_component_pass") is False, "pilot claims production pass"
    )
    require(report.get("pass_g1") is False, "pilot report pass_g1 differs")
    scope = report.get("scope")
    require(isinstance(scope, dict), "pilot report scope must be a mapping")
    require(
        scope.get("execution_scope") == PILOT_SCOPE, "pilot execution scope differs"
    )
    require(
        scope.get("production_status") == PRODUCTION_NOT_RUN,
        "pilot production status differs",
    )
    require(
        type(scope.get("source_files")) is int and scope.get("source_files") == 1,
        "pilot must bind exactly one source file",
    )
    require(
        type(scope.get("partition_count")) is int and scope.get("partition_count") == 0,
        "pilot must not publish partitions",
    )
    rows = scope.get("raw_rows")
    source_bytes = scope.get("source_bytes")
    retained_rows = scope.get("retained_rows")
    require(
        type(rows) is int and 0 < rows <= FULL_SOURCE_ROWS,
        "pilot raw rows must be positive and within the full-row anchor",
    )
    require(
        type(source_bytes) is int and 0 < source_bytes <= MAX_COUNTER,
        "pilot source bytes must be positive and within the supported bound",
    )
    require(
        type(retained_rows) is int and 0 <= retained_rows <= rows,
        "pilot retained rows are outside the raw-row bound",
    )
    require(
        type(scope.get("receipt_artifact_count")) is int
        and scope.get("receipt_artifact_count") == PILOT_RECEIPT_ARTIFACT_COUNT,
        "pilot receipt artifact count differs",
    )
    provenance = report.get("provenance")
    require(isinstance(provenance, dict), "pilot provenance must be a mapping")
    require(
        provenance.get("operator_paths_redacted") is True,
        "pilot paths are not redacted",
    )
    for key in (
        "config_sha256",
        "tool_sha256",
        "task_fingerprint_sha256",
        "result_fingerprint_sha256",
    ):
        require(
            isinstance(provenance.get(key), str)
            and SHA256_PATTERN.fullmatch(provenance[key]) is not None,
            f"pilot provenance {key} is not SHA-256",
        )
    require(
        report.get("task_fingerprint_sha256") == provenance["task_fingerprint_sha256"],
        "pilot task fingerprint surfaces differ",
    )
    require(
        provenance["result_fingerprint_sha256"] == _report_fingerprint(report),
        "pilot result fingerprint differs",
    )
    return rows, source_bytes


def _rounded(value: Decimal) -> float:
    try:
        with localcontext() as context:
            context.prec = 50
            return float(value.quantize(ROUNDING_QUANTUM, rounding=ROUND_HALF_UP))
    except (DecimalException, OverflowError) as exc:
        _fail("benchmark calculation exceeds the supported numeric bound", exc)


def build_benchmark(
    *,
    pilot_report_path: Path,
    time_capture_path: Path,
) -> dict[str, Any]:
    pilot_report_path = _absolute(pilot_report_path)
    time_capture_path = _absolute(time_capture_path)
    require(
        not pilot_report_path.is_relative_to(REPO_ROOT),
        "pilot report must remain outside the repository",
    )
    require(
        not time_capture_path.is_relative_to(REPO_ROOT),
        "time capture must remain outside the repository",
    )
    report, report_sha = _read_pilot_report(pilot_report_path)
    rows, source_bytes = _require_exact_pilot_report(report)
    capture, capture_sha = _parse_time_capture(time_capture_path)
    elapsed = capture["elapsed_seconds"]
    assert isinstance(elapsed, Decimal)
    with localcontext() as context:
        context.prec = 50
        rows_per_second = Decimal(rows) / elapsed
        bytes_per_second = Decimal(source_bytes) / elapsed
        projected_full_scan = Decimal(FULL_SOURCE_ROWS) * elapsed / Decimal(rows)
    return {
        "schema_version": SCHEMA,
        "status": STATUS,
        "pilot_report_sha256": report_sha,
        "selected_source_rows": rows,
        "selected_source_bytes": source_bytes,
        "pilot_wall_seconds_external": _rounded(elapsed),
        "rows_per_second": _rounded(rows_per_second),
        "source_bytes_per_second": _rounded(bytes_per_second),
        "projected_full_scan_wall_seconds": _rounded(projected_full_scan),
        "raw_time_capture_sha256": capture_sha,
        "maximum_rss_kib_external": capture["maximum_rss_kib"],
        "filesystem_inputs_external": capture["filesystem_inputs"],
        "filesystem_outputs_external": capture["filesystem_outputs"],
        "full_source_rows_anchor": FULL_SOURCE_ROWS,
        "calculation_precision": "DECIMAL_THEN_ROUND_HALF_UP_TO_6_DECIMAL_PLACES",
        "scan_projection_excludes": [
            "4096_handle_spool_and_numpy_demultiplex",
            "external_sort_and_parquet_publication",
            "9185_artifact_final_rehash",
            "storage_contention_and_concurrent_io",
        ],
        "downstream_pipeline_capacity_review": "NOT_RUN",
        "full_authorization_sufficient": False,
        "programmatic_parent_of_full_receipt": False,
        "external_command_exit_status": "NOT_CAPTURED_BY_V1_FORMAT",
        "pilot_report_time_pairing": (
            "OPERATOR_ASSERTED_NOT_PROGRAMMATICALLY_VERIFIED"
        ),
        "terminal_resume_exclusion": "NOT_PROGRAMMATICALLY_VERIFIED",
        "operator_paths_redacted": True,
    }


def _validate_output_path(
    output_path: Path,
    *,
    pilot_report_path: Path,
    time_capture_path: Path,
) -> Path:
    output = _assert_no_symlink_components(output_path, field="benchmark output")
    require(output.name == OUTPUT_BASENAME, "benchmark output basename differs")
    require(
        not output.exists() and not output.is_symlink(),
        "benchmark output already exists",
    )
    require(
        not output.is_relative_to(REPO_ROOT),
        "benchmark output must remain outside repository",
    )
    pilot_root = _absolute(pilot_report_path).parent
    require(
        not output.is_relative_to(pilot_root),
        "benchmark output must remain outside pilot output root",
    )
    require(
        output != _absolute(time_capture_path), "benchmark output overlaps time capture"
    )
    parent = output.parent
    try:
        info = parent.lstat()
    except OSError as exc:
        _fail("cannot inspect benchmark output parent", exc)
    require(stat.S_ISDIR(info.st_mode), "benchmark output parent is not a directory")
    require(info.st_uid == os.geteuid(), "benchmark output parent owner differs")
    require(
        stat.S_IMODE(info.st_mode) == 0o700, "benchmark output parent mode is not 0700"
    )
    return output


def _publish_new(path: Path, payload: bytes) -> None:
    absolute = _assert_no_symlink_components(path, field="benchmark output")
    try:
        parent_fd = os.open(
            absolute.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as exc:
        _fail("cannot open benchmark output parent", exc)
    temporary = f".{path.name}.tmp-{os.getpid()}-{secrets.token_hex(8)}"
    descriptor: int | None = None
    try:
        parent_info = os.fstat(parent_fd)
        require(
            parent_info.st_uid == os.geteuid(), "benchmark output parent owner changed"
        )
        require(
            stat.S_IMODE(parent_info.st_mode) == 0o700,
            "benchmark output parent mode changed",
        )
        require(
            _identity(absolute.parent.lstat()) == _identity(parent_info),
            "benchmark output parent path identity changed",
        )
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
        # A hard-link publication is atomic and fails with EEXIST instead of
        # replacing an output created after the earlier path validation.
        os.link(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        _fail("cannot publish benchmark artifact", exc)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        finally:
            os.close(parent_fd)


def write_benchmark(
    *,
    pilot_report_path: Path,
    time_capture_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    output = _validate_output_path(
        output_path,
        pilot_report_path=pilot_report_path,
        time_capture_path=time_capture_path,
    )
    benchmark = build_benchmark(
        pilot_report_path=pilot_report_path,
        time_capture_path=time_capture_path,
    )
    payload = _published_json_bytes(benchmark)
    _publish_new(output, payload)
    observed, _ = _read_regular_snapshot(
        output,
        field="published benchmark",
        max_bytes=MAX_BENCHMARK_BYTES,
    )
    require(observed == payload, "published benchmark readback differs")
    return benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-report", required=True, type=Path)
    parser.add_argument("--time-capture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        benchmark = write_benchmark(
            pilot_report_path=args.pilot_report,
            time_capture_path=args.time_capture,
            output_path=args.output,
        )
    except PilotBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(benchmark, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
