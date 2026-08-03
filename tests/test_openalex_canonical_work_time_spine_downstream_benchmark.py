from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "scripts/benchmark_openalex_canonical_work_time_spine_downstream.py"
)
SPEC = importlib.util.spec_from_file_location(
    "_canonical_spine_downstream_benchmark_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
FROZEN_EXECUTED_TOOL_SHA256 = (
    "c7809ac74425a837e2b5f58c021f46c297ba0cc7cedd4d60ecf0ad969c6332ff"
)


def _private(path: Path) -> Path:
    path.mkdir(mode=0o700)
    path.chmod(0o700)
    return path


def _claim_safe_result() -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": MODULE.SCHEMA,
        "status": MODULE.STATUS,
        "production_execution_status": MODULE.PRODUCTION_NOT_RUN,
        "production_component_pass": False,
        "pass_g1": False,
        "full_authorization_sufficient": False,
        "downstream_capacity_gate_passed": False,
        "programmatic_parent_of_full_receipt": False,
        "operator_paths_redacted": True,
        "pilot_output_tree_unchanged": True,
    }
    result["result_fingerprint_sha256"] = MODULE._result_fingerprint(result)
    return result


def test_contract_cardinality_is_exact_and_bounded() -> None:
    assert (
        hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
        == FROZEN_EXECUTED_TOOL_SHA256
    )
    assert MODULE.RECORD_BYTES == 16
    assert MODULE.PARTITION_COUNT == 4096
    assert MODULE.FULL_SCAN_SHARD_COUNT == 493
    assert MODULE.SORT_CHUNK_ROWS == 1_048_576
    assert (
        2 * MODULE.FULL_SCAN_SHARD_COUNT
        + 2 * MODULE.PARTITION_COUNT
        + MODULE.CONTROL_ARTIFACT_COUNT
        == MODULE.EXACT_ARTIFACT_COUNT
        == 9_185
    )


def test_cli_has_no_source_path_and_requires_explicit_execution_flag() -> None:
    parser = MODULE.make_parser()
    destinations = {action.dest for action in parser._actions}
    assert destinations == {
        "help",
        "pilot_output_root",
        "scratch_root",
        "output",
        "repeats",
        "artifact_sweeps",
        "execute_bounded_nonpromotable",
    }
    assert "source_root" not in destinations
    args = parser.parse_args(
        [
            "--pilot-output-root",
            "/private/pilot",
            "--scratch-root",
            "/private/scratch",
            "--output",
            f"/private/result/{MODULE.OUTPUT_BASENAME}",
        ]
    )
    assert isinstance(args, argparse.Namespace)
    assert args.execute_bounded_nonpromotable is False


def test_run_fails_before_io_without_explicit_execution_flag(tmp_path: Path) -> None:
    with pytest.raises(MODULE.DownstreamBenchmarkError, match="explicit bounded"):
        MODULE.run_benchmark(
            pilot_output_root=tmp_path / "pilot",
            scratch_root=tmp_path / "scratch",
            output_path=tmp_path / MODULE.OUTPUT_BASENAME,
            repeats=1,
            artifact_sweeps=1,
            execute_bounded_nonpromotable=False,
        )


@pytest.mark.parametrize(
    "field",
    [
        "production_component_pass",
        "pass_g1",
        "full_authorization_sufficient",
        "downstream_capacity_gate_passed",
        "programmatic_parent_of_full_receipt",
    ],
)
def test_result_claim_boundary_fails_closed(field: str) -> None:
    result = _claim_safe_result()
    result[field] = True
    result["result_fingerprint_sha256"] = MODULE._result_fingerprint(result)
    with pytest.raises(MODULE.DownstreamBenchmarkError, match="claim boundary"):
        MODULE._validate_final_result(result)


def test_result_fingerprint_and_tree_boundary_are_exact() -> None:
    result = _claim_safe_result()
    MODULE._validate_final_result(result)
    tampered = copy.deepcopy(result)
    tampered["pilot_output_tree_unchanged"] = False
    tampered["result_fingerprint_sha256"] = MODULE._result_fingerprint(tampered)
    with pytest.raises(MODULE.DownstreamBenchmarkError, match="pilot tree changed"):
        MODULE._validate_final_result(tampered)
    fingerprint_tampered = copy.deepcopy(result)
    fingerprint_tampered["result_fingerprint_sha256"] = "0" * 64
    with pytest.raises(MODULE.DownstreamBenchmarkError, match="fingerprint"):
        MODULE._validate_final_result(fingerprint_tampered)


def test_shard_row_counts_are_conservative_and_exact() -> None:
    rows = MODULE._shard_row_counts(1_000_000, 493)
    assert len(rows) == 493
    assert sum(rows) == 1_000_000
    assert max(rows) - min(rows) == 1
    with pytest.raises(MODULE.DownstreamBenchmarkError):
        MODULE._shard_row_counts(2, 3)


def test_distribution_uses_observed_order_statistics() -> None:
    summary = MODULE._distribution([5.0, 1.0, 4.0, 2.0, 3.0])
    assert summary == {
        "count": 5,
        "minimum": 1.0,
        "p50": 3.0,
        "p95": 5.0,
        "p99": 5.0,
        "maximum": 5.0,
        "mean": 3.0,
    }
    assert MODULE._distribution([])["p95"] is None


def test_immutable_publication_preserves_preexisting_output(tmp_path: Path) -> None:
    parent = _private(tmp_path / "result")
    output = parent / MODULE.OUTPUT_BASENAME
    MODULE._publish_new(output, b"first")
    assert output.read_bytes() == b"first"
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(MODULE.DownstreamBenchmarkError):
        MODULE._publish_new(output, b"second")
    assert output.read_bytes() == b"first"


def test_tree_identity_digest_detects_content_change(tmp_path: Path) -> None:
    root = _private(tmp_path / "pilot")
    artifact = root / "artifact.json"
    MODULE._write_new_bytes(artifact, b"{}\n")
    before = MODULE._tree_identity_digest(root)
    artifact.write_bytes(b'{"changed":true}\n')
    artifact.chmod(0o600)
    after = MODULE._tree_identity_digest(root)
    assert before != after


def test_prepare_shards_preserves_records_and_uses_no_hardlinks(tmp_path: Path) -> None:
    runner = MODULE._load_bound_module(
        "_downstream_benchmark_shard_test_runner",
        MODULE.RUNNER_PATH,
        MODULE.RUNNER_SHA256,
    )
    root = _private(tmp_path / "workload")
    source = tmp_path / "source.bin"
    records = runner.np.zeros(4, dtype=runner.RECORD_DTYPE)
    records["work_id_numeric"] = [4, 1, 7, 2]
    records["publication_day"] = -(2**31)
    payload = records.tobytes(order="C")
    source.write_bytes(payload)
    source.chmod(0o600)
    digest = hashlib.sha256(payload).hexdigest()
    checkpoints, artifacts, preparation = MODULE._prepare_benchmark_shards(
        runner=runner,
        source_staging_path=source,
        workload_root=root,
        source_rows=4,
        source_sha256=digest,
        shard_count=2,
    )
    assert len(checkpoints) == 2
    assert len(artifacts) == 4
    assert preparation["rows"] == 4
    assert preparation["bytes"] == 64
    staging = [root / item["staging_shard_relative_path"] for item in checkpoints]
    assert b"".join(path.read_bytes() for path in staging) == payload
    assert sum(path.stat().st_size for path in staging) == len(payload)
    assert all(path.stat().st_nlink == 1 for path in staging)
    assert all(path.stat().st_ino != source.stat().st_ino for path in staging)


def test_h9185_kernel_rejects_topology_drift_and_repeats_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _private(tmp_path / "surface")
    MODULE._write_new_bytes(root / "a", b"a")
    MODULE._write_new_bytes(root / "b", b"bb")

    def artifact_contract(base: Path, relative: str) -> dict[str, object]:
        payload = (base / relative).read_bytes()
        return {
            "relative_path": relative,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    monkeypatch.setattr(MODULE, "EXACT_ARTIFACT_COUNT", 2)
    runner = SimpleNamespace(_artifact_contract=artifact_contract)
    result = MODULE._run_h9185(
        runner=runner,
        workload_root=root,
        expected_paths=["a", "b"],
        sweep_count=2,
    )
    assert result["artifact_count"] == 2
    assert result["sweep_count"] == 2
    assert result["sweeps"][0]["hashed_bytes"] == 3
    assert (
        result["sweeps"][0]["artifact_contract_tree_sha256"]
        == result["sweeps"][1]["artifact_contract_tree_sha256"]
    )
    MODULE._write_new_bytes(root / "c", b"c")
    with pytest.raises(MODULE.DownstreamBenchmarkError, match="topology"):
        MODULE._run_h9185(
            runner=runner,
            workload_root=root,
            expected_paths=["a", "b"],
            sweep_count=1,
        )


def test_script_never_calls_production_finalization_or_receipt_builders() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    forbidden = {"_finalize", "_output_receipt", "_validate_final_publication"}
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint(forbidden)


def test_published_json_is_exact_pretty_json() -> None:
    value = {"z": 1, "a": False}
    payload = MODULE._published_json_bytes(value)
    assert payload == b'{\n  "a": false,\n  "z": 1\n}\n'
    assert json.loads(payload) == value


def test_path_overlap_catches_ancestor_and_descendant(tmp_path: Path) -> None:
    root = tmp_path / "root"
    child = root / "child"
    peer = tmp_path / "peer"
    assert MODULE._paths_overlap(root, child)
    assert MODULE._paths_overlap(child, root)
    assert not MODULE._paths_overlap(root, peer)


def test_wrong_result_basename_fails_before_runner_import(tmp_path: Path) -> None:
    pilot = _private(tmp_path / "pilot")
    result = _private(tmp_path / "result")
    with pytest.raises(MODULE.DownstreamBenchmarkError, match="basename"):
        MODULE.run_benchmark(
            pilot_output_root=pilot,
            scratch_root=tmp_path / "scratch",
            output_path=result / "wrong.json",
            repeats=1,
            artifact_sweeps=1,
            execute_bounded_nonpromotable=True,
        )


def test_output_control_paths_have_exact_seven_artifacts() -> None:
    assert len(MODULE.CONTROL_PATHS) == MODULE.CONTROL_ARTIFACT_COUNT == 7
    assert len(set(MODULE.CONTROL_PATHS)) == 7
    assert "work_time_spine_audit.outputs.json" not in MODULE.CONTROL_PATHS
    assert "work_time_spine_pilot.outputs.json" not in MODULE.CONTROL_PATHS
