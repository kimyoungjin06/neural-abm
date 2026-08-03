from __future__ import annotations

import ast
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts/benchmark_openalex_canonical_work_time_spine_downstream_stress_v1.py"
)
SPEC = importlib.util.spec_from_file_location("_canonical_spine_stress_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _private(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)
    return path


def _runner_and_helper():
    helper = MODULE._load_helper()
    runner = helper._load_bound_module(
        "_canonical_spine_stress_fixture_runner",
        MODULE.RUNNER_PATH,
        MODULE.RUNNER_SHA256,
    )
    return runner, helper


def _pilot_records(runner, rows: int = 10):
    records = runner.np.zeros(rows, dtype=runner.RECORD_DTYPE)
    records["work_id_numeric"] = runner.np.arange(rows, dtype=runner.np.uint64)
    records["publication_day"] = -(2**31)
    return records


def _safe_result() -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": MODULE.SCHEMA,
        "status": MODULE.STATUS,
        "production_execution_status": MODULE.PRODUCTION_NOT_RUN,
        "production_component_pass": False,
        "pass_g1": False,
        "full_authorization_sufficient": False,
        "downstream_capacity_gate_passed": False,
        "programmatic_parent_of_full_receipt": False,
        "actual_full_4096_partition_occupancy_profile_measured": False,
        "full_byte_finalization_surface_measured": False,
        "theoretical_470_run_heap_merge_measured": False,
        "source_hdd_opened": False,
        "full_like_per_partition_row_scale_measured": True,
        "balanced_multi_run_heap_branch_measured": True,
        "maximum_measured_fan_in": 8,
        "independent_verification_status": "PENDING",
        "operator_acceptance_status": "NOT_RECORDED",
        "independent_quiet_window_sessions": 0,
        "cache_condition": "UNCONTROLLED_WARM_OR_CACHE_ASSISTED",
        "implementation_binding": {
            "heap_operation_timing_proxy_installed": False,
            "run_reader_proxy_installed": False,
            "production_partition_publisher_called": False,
            "production_finalizer_called": False,
            "production_receipt_builder_called": False,
        },
        "operator_paths_redacted": True,
    }
    result["result_fingerprint_sha256"] = MODULE._result_fingerprint(result)
    return result


def test_exact_production_chunk_and_scenario_arithmetic() -> None:
    assert MODULE.SORT_CHUNK_ROWS == 1_048_576
    assert MODULE.FULL_LIKE_MEAN_ROWS == 120_189
    assert MODULE.PARTITION_ID == 4095
    assert MODULE.SYNTHETIC_NAMESPACE_BASE == 1 << 63
    assert [(item.scenario_id, item.rows, item.expected_runs) for item in MODULE.SCENARIOS] == [
        ("RMEAN", 120_189, 1),
        ("A2_BALANCED", 2_097_152, 2),
        ("A8_BALANCED", 8_388_608, 8),
    ]
    assert sum(item.rows for item in MODULE.SCENARIOS) == 10_605_949


def test_helper_and_parent_bindings_are_frozen() -> None:
    assert hashlib.sha256(SCRIPT.read_bytes()).hexdigest() == (
        "19bae4881bce030876d884a96500936d3b5a96ef97ccfffe28d7a3d605154b44"
    )
    assert hashlib.sha256(MODULE.HELPER_PATH.read_bytes()).hexdigest() == MODULE.HELPER_SHA256
    assert MODULE.FIRST_BENCHMARK_SHA256 == (
        "6fd87a6aec825b690803a36d60ae271dd9fa7a691563fbb1b18b2e5a07649109"
    )
    assert MODULE.FIRST_BENCHMARK_FINGERPRINT == (
        "5e12129cafbda0bc5918eedeb43c3ec6352357f3e78e67b0d6d6494f7a71e287"
    )


def test_pilot_reader_uses_local_bounded_regular_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner, helper = _runner_and_helper()
    records = _pilot_records(runner)
    payload = records.tobytes(order="C")
    staging = tmp_path / "staging.bin"
    staging.write_bytes(payload)
    staging.chmod(0o600)
    monkeypatch.setattr(MODULE, "PILOT_STAGING_ROWS", len(records))
    monkeypatch.setattr(
        MODULE, "PILOT_STAGING_SHA256", hashlib.sha256(payload).hexdigest()
    )

    observed = MODULE._read_pilot_records(helper, runner, staging)

    assert np.array_equal(observed, records)


def test_result_root_cannot_overlap_pilot_root(tmp_path: Path) -> None:
    pilot_root = _private(tmp_path / "pilot")
    result_root = _private(pilot_root / "results")
    first_result = tmp_path / "first" / "result.json"
    scratch = tmp_path / "scratch"

    with pytest.raises(MODULE.StressBenchmarkError, match="result root overlaps pilot"):
        MODULE.run_benchmark(
            pilot_output_root=pilot_root,
            first_benchmark_result=first_result,
            scratch_root=scratch,
            output_path=result_root / MODULE.OUTPUT_BASENAME,
            repeats=1,
            execute_synthetic_nonpromotable=True,
        )


def test_cli_has_no_source_root_and_requires_distinct_flag() -> None:
    parser = MODULE.make_parser()
    destinations = {action.dest for action in parser._actions}
    assert "source_root" not in destinations
    assert destinations == {
        "help",
        "pilot_output_root",
        "first_benchmark_result",
        "scratch_root",
        "output",
        "repeats",
        "execute_synthetic_nonpromotable",
    }
    with pytest.raises(MODULE.StressBenchmarkError, match="explicit synthetic"):
        MODULE.run_benchmark(
            pilot_output_root=Path("/not/read"),
            first_benchmark_result=Path("/not/read"),
            scratch_root=Path("/not/write"),
            output_path=Path("/not/write/result.json"),
            repeats=1,
            execute_synthetic_nonpromotable=False,
        )


def test_balanced_generator_and_run_validation_use_interleaved_q(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner, helper = _runner_and_helper()
    pilot = _pilot_records(runner)
    monkeypatch.setattr(MODULE, "SORT_CHUNK_ROWS", 4)
    monkeypatch.setattr(MODULE, "GENERATION_BATCH_ROWS", 2)
    scenario = MODULE.Scenario("A3_TEST", 12, 3, "TEST_ONLY")
    root = _private(tmp_path / "scenario")
    contract, preparation = MODULE._write_synthetic_spool(
        helper=helper,
        runner=runner,
        pilot_records=pilot,
        scenario_root=root,
        scenario=scenario,
    )
    assert contract["rows"] == 12
    assert contract["bytes"] == 12 * MODULE.RECORD_BYTES
    assert preparation["excluded_from_kernel_timing"] is True
    spool = root / contract["relative_path"]
    payload = spool.read_bytes()
    records = np.frombuffer(payload, dtype=runner.RECORD_DTYPE)
    runs = []
    run_root = _private(root / "runs")
    for run_index in range(3):
        sorted_records = np.sort(
            records[run_index * 4 : (run_index + 1) * 4],
            order="work_id_numeric",
            kind="stable",
        )
        run_path = run_root / f"run-{run_index}.bin"
        run_path.write_bytes(sorted_records.tobytes())
        run_path.chmod(0o600)
        run_payload = run_path.read_bytes()
        runs.append(
            runner.RunArtifact(
                path=run_path,
                rows=4,
                bytes=len(run_payload),
                sha256=hashlib.sha256(run_payload).hexdigest(),
            )
        )
    validation = MODULE._validate_sorted_runs(
        helper=helper,
        runner=runner,
        pilot_records=pilot,
        scenario=scenario,
        runs=runs,
    )
    assert validation["run_count"] == 3
    assert [item["q_start"] for item in validation["runs"]] == [0, 1, 2]
    assert [item["q_stop"] for item in validation["runs"]] == [9, 10, 11]
    assert all(item["q_stride"] == 3 for item in validation["runs"])
    expected = MODULE._expected_sorted_contract(
        runner=runner,
        pilot_records=pilot,
        scenario=scenario,
    )
    merged = np.sort(records, order="work_id_numeric", kind="stable")
    digest = hashlib.sha256()
    digest.update(len(runner.RECORD_HASH_DOMAIN).to_bytes(8, "little"))
    digest.update(runner.RECORD_HASH_DOMAIN)
    digest.update(len(merged).to_bytes(8, "little"))
    runner._update_framed_record_digest(digest, merged)
    assert expected["record_sha256"] == digest.hexdigest()


def test_coarse_instrumentation_never_replaces_heap_or_run_reader() -> None:
    runner, helper = _runner_and_helper()
    heapq_before = runner.heapq
    reader_before = runner._RunReader
    with MODULE._instrument_runner_coarse(helper, runner) as metrics:
        order = runner.np.argsort(np.array([2, 1], dtype=np.uint64), kind="stable")
        assert order.tolist() == [1, 0]
        assert runner.heapq is heapq_before
        assert runner._RunReader is reader_before
    assert metrics.argsort_calls == 1
    summary = metrics.summary()
    assert summary["heap_operation_timing_proxy_installed"] is False
    assert summary["run_reader_proxy_installed"] is False


def test_tiny_actual_frozen_sort_merge_and_parquet_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner, helper = _runner_and_helper()
    pilot = _pilot_records(runner)
    monkeypatch.setattr(MODULE, "SORT_CHUNK_ROWS", 4)
    monkeypatch.setattr(MODULE, "GENERATION_BATCH_ROWS", 2)
    scenario = MODULE.Scenario("A3_INTEGRATION", 12, 3, "TEST_ONLY")
    repeat_root = _private(tmp_path / "repeat-01")

    observed = MODULE._run_scenario(
        helper=helper,
        runner=runner,
        pilot_records=pilot,
        repeat_root=repeat_root,
        scenario=scenario,
        repeat_index=1,
    )

    assert observed["observed_runs"] == 3
    assert observed["maximum_measured_fan_in"] == 3
    assert observed["output_rows"] == 12
    assert observed["integrity_status"] == "PASS_SYNTHETIC_STRESS_KERNEL_CLOSURE"
    assert observed["merge_parquet_and_validation"][
        "derived_heap_push_count_not_observed"
    ] == 12
    assert (
        repeat_root / "A3_INTEGRATION/spine/part-4095.parquet"
    ).is_file()


@pytest.mark.parametrize(
    "field",
    [
        "production_component_pass",
        "pass_g1",
        "full_authorization_sufficient",
        "downstream_capacity_gate_passed",
        "programmatic_parent_of_full_receipt",
        "actual_full_4096_partition_occupancy_profile_measured",
        "full_byte_finalization_surface_measured",
        "theoretical_470_run_heap_merge_measured",
        "source_hdd_opened",
    ],
)
def test_stress_result_claim_boundary_fails_closed(field: str) -> None:
    result = _safe_result()
    result[field] = True
    result["result_fingerprint_sha256"] = MODULE._result_fingerprint(result)
    with pytest.raises(MODULE.StressBenchmarkError, match="claim boundary"):
        MODULE._validate_final_result(result)


def test_first_benchmark_binding_rejects_any_other_file(tmp_path: Path) -> None:
    helper = MODULE._load_helper()
    path = tmp_path / "wrong.json"
    path.write_bytes(MODULE._published_json_bytes({"wrong": True}))
    path.chmod(0o600)
    with pytest.raises(MODULE.StressBenchmarkError, match="SHA-256"):
        MODULE._validate_first_benchmark(helper, path)


def test_source_has_no_production_publication_or_heap_proxy_assignment() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden_calls = {
        "_publish_partitions",
        "_finalize",
        "_output_receipt",
        "_validate_final_publication",
    }
    observed_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert observed_calls.isdisjoint(forbidden_calls)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                assert not (
                    isinstance(target, ast.Attribute)
                    and target.attr in {"heapq", "_RunReader"}
                )


def test_output_basename_is_distinct_from_prior_benchmark() -> None:
    assert MODULE.OUTPUT_BASENAME == (
        "openalex_g1_canonical_work_time_spine_downstream_stress_v1.json"
    )
    assert MODULE.OUTPUT_BASENAME != "openalex_g1_canonical_work_time_spine_downstream_benchmark.json"
    assert "/hdd" not in SCRIPT.read_text(encoding="utf-8")
    assert "--source-root" not in SCRIPT.read_text(encoding="utf-8")


def test_result_fingerprint_and_safe_boundary_validate() -> None:
    result = _safe_result()
    MODULE._validate_final_result(result)
    result["result_fingerprint_sha256"] = "0" * 64
    with pytest.raises(MODULE.StressBenchmarkError, match="fingerprint"):
        MODULE._validate_final_result(result)


def test_synthetic_identifier_namespace_and_modulo() -> None:
    runner, _ = _runner_and_helper()
    q_values = np.arange(16, dtype=np.uint64)
    identifiers = MODULE._synthetic_ids(q_values, runner)
    assert int(identifiers[0]) == (1 << 63) + MODULE.PARTITION_ID
    assert np.all(identifiers % MODULE.PARTITION_COUNT == MODULE.PARTITION_ID)
    assert np.all(identifiers[1:] > identifiers[:-1])
    assert np.all(identifiers[1:] - identifiers[:-1] == MODULE.PARTITION_COUNT)
