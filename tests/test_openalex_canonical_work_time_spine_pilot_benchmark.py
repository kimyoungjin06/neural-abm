from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts/summarize_openalex_canonical_work_time_spine_pilot_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("_canonical_pilot_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _private_dir(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _pilot_report() -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": MODULE.PILOT_REPORT_SCHEMA,
        "status": MODULE.PILOT_STATUS,
        "scope": {
            "execution_scope": MODULE.PILOT_SCOPE,
            "production_status": MODULE.PRODUCTION_NOT_RUN,
            "source_files": 1,
            "source_bytes": 250,
            "raw_rows": 100,
            "retained_rows": 100,
            "partition_count": 0,
            "receipt_artifact_count": 8,
        },
        "provenance": {
            "config_sha256": "a" * 64,
            "tool_sha256": "b" * 64,
            "task_fingerprint_sha256": "c" * 64,
            "operator_paths_redacted": True,
        },
        "task_fingerprint_sha256": "c" * 64,
        "production_component_pass": False,
        "pass_g1": False,
    }
    provenance = report["provenance"]
    assert isinstance(provenance, dict)
    provenance["result_fingerprint_sha256"] = MODULE._report_fingerprint(report)
    return report


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    pilot_root = _private_dir(tmp_path / "pilot")
    benchmark_root = _private_dir(tmp_path / "benchmark")
    report_path = pilot_root / "work_time_spine_pilot.json"
    report_path.write_bytes(MODULE._published_json_bytes(_pilot_report()))
    report_path.chmod(0o600)
    time_path = benchmark_root / MODULE.TIME_CAPTURE_BASENAME
    time_path.write_text(
        "elapsed_seconds=2.5\n"
        "maximum_rss_kib=123456\n"
        "filesystem_inputs=789\n"
        "filesystem_outputs=456\n",
        encoding="utf-8",
    )
    time_path.chmod(0o600)
    output_path = benchmark_root / MODULE.OUTPUT_BASENAME
    return report_path, time_path, output_path


def test_build_and_publish_nonpromotable_pilot_benchmark(tmp_path: Path) -> None:
    report_path, time_path, output_path = _inputs(tmp_path)

    benchmark = MODULE.write_benchmark(
        pilot_report_path=report_path,
        time_capture_path=time_path,
        output_path=output_path,
    )

    assert benchmark["schema_version"] == MODULE.SCHEMA
    assert benchmark["status"] == MODULE.STATUS
    assert benchmark["selected_source_rows"] == 100
    assert benchmark["selected_source_bytes"] == 250
    assert benchmark["pilot_wall_seconds_external"] == 2.5
    assert benchmark["rows_per_second"] == 40.0
    assert benchmark["source_bytes_per_second"] == 100.0
    assert benchmark["projected_full_scan_wall_seconds"] == 12_307_330.175
    assert benchmark["maximum_rss_kib_external"] == 123_456
    assert benchmark["filesystem_inputs_external"] == 789
    assert benchmark["filesystem_outputs_external"] == 456
    assert benchmark["downstream_pipeline_capacity_review"] == "NOT_RUN"
    assert benchmark["full_authorization_sufficient"] is False
    assert benchmark["programmatic_parent_of_full_receipt"] is False
    assert benchmark["external_command_exit_status"] == "NOT_CAPTURED_BY_V1_FORMAT"
    assert benchmark["pilot_report_time_pairing"] == (
        "OPERATOR_ASSERTED_NOT_PROGRAMMATICALLY_VERIFIED"
    )
    assert benchmark["terminal_resume_exclusion"] == "NOT_PROGRAMMATICALLY_VERIFIED"
    assert benchmark["operator_paths_redacted"] is True
    assert not any(
        str(tmp_path) in json.dumps(value, ensure_ascii=False)
        for value in benchmark.values()
    )
    assert output_path.read_bytes() == MODULE._published_json_bytes(benchmark)
    assert os.stat(output_path).st_mode & 0o777 == 0o600


def test_semantically_equal_reformatted_pilot_report_is_rejected(
    tmp_path: Path,
) -> None:
    report_path, time_path, _ = _inputs(tmp_path)
    report_path.write_text(json.dumps(_pilot_report()) + "\n", encoding="utf-8")
    report_path.chmod(0o600)

    with pytest.raises(
        MODULE.PilotBenchmarkError,
        match="exact canonical published JSON",
    ):
        MODULE.build_benchmark(
            pilot_report_path=report_path,
            time_capture_path=time_path,
        )


@pytest.mark.parametrize(
    ("capture", "message"),
    [
        (
            "elapsed_seconds=0\n"
            "maximum_rss_kib=1\n"
            "filesystem_inputs=2\n"
            "filesystem_outputs=3\n",
            "must be positive",
        ),
        (
            "maximum_rss_kib=1\n"
            "elapsed_seconds=2.5\n"
            "filesystem_inputs=2\n"
            "filesystem_outputs=3\n",
            "key order or name differs",
        ),
        (
            "elapsed_seconds=2.5\n"
            "maximum_rss_kib=-1\n"
            "filesystem_inputs=2\n"
            "filesystem_outputs=3\n",
            "maximum_rss_kib is not a nonnegative integer",
        ),
        (
            "elapsed_seconds=2.5\r\n"
            "maximum_rss_kib=1\r\n"
            "filesystem_inputs=2\r\n"
            "filesystem_outputs=3\r\n",
            "LF-only",
        ),
        (
            "elapsed_seconds=0.0000001\n"
            "maximum_rss_kib=1\n"
            "filesystem_inputs=2\n"
            "filesystem_outputs=3\n",
            "nonnegative decimal",
        ),
    ],
)
def test_time_capture_is_exact_and_positive(
    tmp_path: Path, capture: str, message: str
) -> None:
    report_path, time_path, _ = _inputs(tmp_path)
    time_path.write_text(capture, encoding="utf-8")
    time_path.chmod(0o600)

    with pytest.raises(MODULE.PilotBenchmarkError, match=message):
        MODULE.build_benchmark(
            pilot_report_path=report_path,
            time_capture_path=time_path,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("scope", "source_files"), 2, "exactly one source file"),
        (("scope", "source_files"), True, "exactly one source file"),
        (("scope", "partition_count"), 1, "must not publish partitions"),
        (("scope", "partition_count"), False, "must not publish partitions"),
        (("scope", "retained_rows"), 101, "retained rows"),
        (("scope", "receipt_artifact_count"), 7, "artifact count differs"),
        (("scope", "production_status"), "PASS", "production status differs"),
        (("production_component_pass",), True, "claims production pass"),
        (("provenance", "operator_paths_redacted"), False, "paths are not redacted"),
    ],
)
def test_pilot_report_scope_and_claim_boundaries_are_enforced(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    report_path, time_path, _ = _inputs(tmp_path)
    report = _pilot_report()
    target = report
    for key in path[:-1]:
        target = target[key]  # type: ignore[assignment,index]
    target[path[-1]] = value  # type: ignore[index]
    report_path.write_bytes(MODULE._published_json_bytes(report))
    report_path.chmod(0o600)

    with pytest.raises(MODULE.PilotBenchmarkError, match=message):
        MODULE.build_benchmark(
            pilot_report_path=report_path,
            time_capture_path=time_path,
        )


def test_pilot_report_result_fingerprint_is_verified(tmp_path: Path) -> None:
    report_path, time_path, _ = _inputs(tmp_path)
    report = _pilot_report()
    provenance = report["provenance"]
    assert isinstance(provenance, dict)
    provenance["result_fingerprint_sha256"] = "0" * 64
    report_path.write_bytes(MODULE._published_json_bytes(report))
    report_path.chmod(0o600)

    with pytest.raises(MODULE.PilotBenchmarkError, match="result fingerprint differs"):
        MODULE.build_benchmark(
            pilot_report_path=report_path,
            time_capture_path=time_path,
        )


def test_input_basenames_and_size_bounds_are_enforced(tmp_path: Path) -> None:
    report_path, time_path, _ = _inputs(tmp_path)
    wrong_report = report_path.with_name("copied-report.json")
    wrong_report.write_bytes(report_path.read_bytes())
    wrong_report.chmod(0o600)
    with pytest.raises(MODULE.PilotBenchmarkError, match="pilot-report basename"):
        MODULE.build_benchmark(
            pilot_report_path=wrong_report,
            time_capture_path=time_path,
        )

    time_path.write_bytes(b"x" * (MODULE.MAX_TIME_CAPTURE_BYTES + 1))
    time_path.chmod(0o600)
    with pytest.raises(MODULE.PilotBenchmarkError, match="size is outside"):
        MODULE.build_benchmark(
            pilot_report_path=report_path,
            time_capture_path=time_path,
        )


def test_output_must_be_new_private_and_outside_pilot_root(tmp_path: Path) -> None:
    report_path, time_path, output_path = _inputs(tmp_path)
    inside_pilot = report_path.parent / MODULE.OUTPUT_BASENAME
    with pytest.raises(MODULE.PilotBenchmarkError, match="outside pilot output root"):
        MODULE.write_benchmark(
            pilot_report_path=report_path,
            time_capture_path=time_path,
            output_path=inside_pilot,
        )

    output_path.write_bytes(b"existing")
    output_path.chmod(0o600)
    with pytest.raises(MODULE.PilotBenchmarkError, match="already exists"):
        MODULE.write_benchmark(
            pilot_report_path=report_path,
            time_capture_path=time_path,
            output_path=output_path,
        )


def test_atomic_publication_never_replaces_a_racing_output(tmp_path: Path) -> None:
    output_root = _private_dir(tmp_path / "benchmark")
    output_path = output_root / MODULE.OUTPUT_BASENAME
    output_path.write_bytes(b"racing-operator-output")
    output_path.chmod(0o600)

    with pytest.raises(MODULE.PilotBenchmarkError, match="cannot publish"):
        MODULE._publish_new(output_path, b"new-benchmark")

    assert output_path.read_bytes() == b"racing-operator-output"


def test_cli_reports_validation_failure_without_writing(tmp_path: Path) -> None:
    report_path, time_path, output_path = _inputs(tmp_path)
    time_path.write_text("not-a-time-capture\n", encoding="utf-8")
    time_path.chmod(0o600)

    assert (
        MODULE.main(
            [
                "--pilot-report",
                str(report_path),
                "--time-capture",
                str(time_path),
                "--output",
                str(output_path),
            ]
        )
        == 2
    )
    assert not output_path.exists()
