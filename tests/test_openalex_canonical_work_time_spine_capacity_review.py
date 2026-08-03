from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "scripts/review_openalex_canonical_work_time_spine_capacity.py"
)
DOC_PATH = (
    REPO_ROOT / "docs/research/openalex-canonical-work-time-spine-capacity-review-v0.md"
)


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("capacity_review", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module() -> Any:
    return _load_module()


@pytest.fixture
def inputs(module: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return module.load_inputs()


def test_default_review_is_closed_and_nonpromoting(
    module: Any, inputs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    report = module.build_review(*inputs)

    assert report["status"] == ("ANALYTIC_ESTIMATE_ONLY_EMPIRICAL_BENCHMARK_NOT_RUN")
    assert report["gate_status"] == "NOT_PASSED"
    assert report["production_execution_status"] == "PRODUCTION_NOT_RUN"
    assert report["contract_binding"]["runner_imported_or_executed"] is False
    assert report["review_state"] == {
        "empirical_benchmark_status": "NOT_RUN",
        "operator_acceptance_status": "NOT_RECORDED",
        "independent_review_status": "PENDING",
        "downstream_capacity_gate_status": "NOT_PASSED",
        "full_authorization_status": "NOT_GRANTED",
        "analytic_output_must_not_mutate_any_gate": True,
        "analytic_output_must_not_be_used_as_production_receipt_parent": True,
    }
    assert report["claim_boundary"] == {
        "analytic_estimate_is_benchmark": False,
        "hardware_snapshot_is_live_execution_preflight": False,
        "one_file_pilot_extrapolation_is_sufficient": False,
        "downstream_capacity_gate_passed": False,
        "full_execution_authorized": False,
        "production_receipt_created": False,
    }


def test_frozen_byte_and_workspace_formulas_are_reconstructed(
    module: Any, inputs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    report = module.build_review(*inputs)
    anchors = report["contract_anchors"]
    phases = report["phase_estimates"]

    assert anchors["retained_rows"] == 492_293_206
    assert anchors["staging_payload_bytes_exact"] == 7_876_691_296
    assert anchors["parquet_payload_bytes_upper_bound"] == 16_290_253_504
    assert anchors["worst_case_disk_workspace_bytes_before_multiplier"] == (
        57_821_324_704
    )
    assert anchors["receipt_artifact_count"] == 9_185
    assert anchors["framed_record_hash_header_bytes_per_digest"] == 45

    spool = phases["spool_and_numpy_demultiplex"]
    assert spool["logical_io_bytes_exact"] == 15_753_382_592
    assert spool["combined_staging_and_spool_payload_bytes_exact"] == (15_753_382_592)
    assert spool["combined_payload_plus_control_bound_bytes"] == 17_900_866_240
    assert spool["maximum_open_spool_files"] == 4_096
    assert spool["maximum_data_file_close_and_fsync_operations"] == 4_096
    assert spool["hash_input_bytes_engineering_upper_bound"] == 27_568_441_721
    assert spool["framed_record_hash_header_instances"] == 493

    sort = phases["external_sort_and_parquet_publication"]
    assert sort["logical_read_bytes_upper_bound"] == 48_333_889_600
    assert sort["logical_write_bytes_upper_bound"] == 24_166_944_800
    assert sort["logical_io_bytes_upper_bound"] == 72_500_834_400
    assert sort["maximum_external_run_files_global"] == 4_565
    assert sort["maximum_external_run_files_one_partition"] == 470
    assert sort["maximum_open_run_files_one_partition"] == 512
    assert sort["maximum_data_file_fsync_operations"] == 8_661
    assert sort["peak_workspace_bytes_upper_bound"] == 57_821_324_704
    assert sort["hash_input_bytes_engineering_upper_bound"] == 63_550_769_920
    assert sort["framed_record_hash_header_instances"] == 8_192

    rehash = phases["one_9185_artifact_final_rehash_sweep"]
    assert rehash["artifact_count_exact"] == 9_185
    assert rehash["logical_read_and_hash_bytes_upper_bound"] == 26_314_428_448
    assert rehash["incremental_workspace_bytes_for_read_only_rehash"] == 0

    finalization = phases["same_invocation_receipt_surface_validation_context"]
    assert finalization["logical_read_bytes_conservative_upper_bound"] == (
        133_719_625_888
    )
    assert finalization["scope"] == (
        "RECEIPT_ARTIFACT_AND_OUTPUT_RECONSTRUCTION_READS_ONLY"
    )
    assert (
        "parent_control_revalidation_reads" in finalization["excluded_from_this_bound"]
    )


def test_time_estimates_remain_hypothetical_and_conditional(
    module: Any, inputs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    report = module.build_review(*inputs)
    phases = report["phase_estimates"]
    expected = {
        "spool_and_numpy_demultiplex": (5_398, 1.499),
        "external_sort_and_parquet_publication": (26_072, 7.242),
        "one_9185_artifact_final_rehash_sweep": (1_756, 0.488),
    }
    for name, (seconds, hours) in expected.items():
        timing = phases[name]["conditional_time"]
        assert timing["status"] == "HYPOTHETICAL_CONDITIONAL_NOT_BENCHMARKED"
        assert timing["conditional_planning_seconds_rounded_up"] == seconds
        assert timing["conditional_planning_hours_rounded_3dp"] == hours
        assert timing["is_unconditional_wall_time_bound"] is False
        assert (
            timing["requires_all_rate_floors_and_unmodeled_costs_to_be_validated"]
            is True
        )
    assert (
        report["hypothetical_rate_floor_scenario"]["all_rates_empirically_validated"]
        is False
    )
    rehash_components = phases["one_9185_artifact_final_rehash_sweep"][
        "conditional_time"
    ]["component_seconds_rounded_up"]
    assert rehash_components["artifact_contracts"] == 92
    assert "artifact_metadata" not in rehash_components


def test_hardware_snapshot_is_only_a_dated_comparison(
    module: Any, inputs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    report = module.build_review(*inputs)
    hardware = report["hardware_snapshot"]

    assert hardware["observed_at"] == "2026-07-20T17:11:18+09:00"
    assert hardware["observation_scope"] == (
        "CONTROL_PLANE_ONLY_NO_SOURCE_PAYLOAD_OPENED"
    )
    assert hardware["output_filesystem_available_bytes"] == 3_657_099_960_320
    assert hardware["mem_available_bytes"] == 1_037_850_583_040
    assert hardware["nofile_soft_limit"] == 1_048_576
    assert hardware["comparison_status"] == (
        "OBSERVATION_ONLY_NOT_PREFLIGHT_OR_ACCEPTANCE"
    )
    assert hardware["available_disk_to_fresh_requirement_ratio_milli"] == 23_880
    assert hardware["available_ram_to_requirement_ratio_milli"] == 60_410
    assert hardware["nofile_to_requirement_ratio_milli"] == 128_000
    assert hardware["snapshot_may_be_stale_at_execution"] is True
    assert hardware["downstream_pipeline_benchmark_at_observation"] == "NOT_RUN"


def test_formula_or_gate_mutation_fails_closed(
    module: Any, inputs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, assumptions = inputs

    changed_contract = copy.deepcopy(contract)
    changed_contract["resource_envelope"]["declared_retained_rows"] += 1
    with pytest.raises(module.CapacityReviewError, match="frozen formula mismatch"):
        module.build_review(changed_contract, assumptions)

    changed_rates = copy.deepcopy(assumptions)
    changed_rates["hypothetical_rate_floor_scenario"][
        "sequential_read_bytes_per_second"
    ] = 0
    with pytest.raises(module.CapacityReviewError, match="positive integer"):
        module.build_review(contract, changed_rates)

    changed_gate = copy.deepcopy(assumptions)
    changed_gate["review_state"]["downstream_capacity_gate_status"] = "PASS"
    with pytest.raises(module.CapacityReviewError, match="review state drift"):
        module.build_review(contract, changed_gate)


def test_extra_assumption_keys_and_misleading_snapshot_fields_fail_closed(
    module: Any, inputs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, assumptions = inputs

    changed_rates = copy.deepcopy(assumptions)
    changed_rates["hypothetical_rate_floor_scenario"]["gate_status"] = "PASS"
    with pytest.raises(module.CapacityReviewError, match="key set differs"):
        module.build_review(contract, changed_rates)

    changed_state = copy.deepcopy(assumptions)
    changed_state["review_state"]["downstream_capacity_gate_passed"] = True
    with pytest.raises(module.CapacityReviewError, match="key set differs"):
        module.build_review(contract, changed_state)

    changed_hardware = copy.deepcopy(assumptions)
    changed_hardware["hardware_observation"][
        "source_drive_health_review_at_observation"
    ] = "PASS"
    with pytest.raises(module.CapacityReviewError, match="hardware snapshot drift"):
        module.build_review(contract, changed_hardware)


def test_per_partition_run_file_constraint_is_checked_directly(
    module: Any, inputs: tuple[dict[str, Any], dict[str, Any]]
) -> None:
    contract, assumptions = inputs

    changed_formula = copy.deepcopy(contract)
    changed_formula["bounded_partition_and_sort_contract"]["external_runs"][
        "theoretical_maximum_runs_in_one_partition"
    ] = 469
    with pytest.raises(
        module.CapacityReviewError,
        match="frozen formula mismatch: per-partition run files",
    ):
        module.build_review(changed_formula, assumptions)

    changed_limit = copy.deepcopy(contract)
    changed_limit["bounded_partition_and_sort_contract"]["external_runs"][
        "maximum_open_run_files"
    ] = 469
    with pytest.raises(
        module.CapacityReviewError,
        match="per-partition run-file bound exceeds",
    ):
        module.build_review(changed_limit, assumptions)


def test_alternate_symlink_to_default_input_is_rejected(
    module: Any, tmp_path: Path
) -> None:
    link = tmp_path / "contract.yaml"
    link.symlink_to(module.DEFAULT_CONTRACT)
    with pytest.raises(module.CapacityReviewError, match="may not be a symlink"):
        module.load_inputs(contract_path=link)


def test_cli_emits_json_only_and_has_no_output_option(
    module: Any,
    inputs: tuple[dict[str, Any], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = module.build_review(*inputs)
    assert module.main([]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == expected
    assert "--output" not in module.make_parser().format_help()


def test_review_document_preserves_benchmark_and_acceptance_boundary() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    for literal in (
        "ANALYTIC_ESTIMATE_ONLY_EMPIRICAL_BENCHMARK_NOT_RUN",
        "NOT_PASSED",
        "NOT_RECORDED",
        "PRODUCTION_NOT_RUN",
        "133,719,625,888",
        "9,185",
        "bounded nonpromotable staging-derived benchmark",
        "operator acceptance",
    ):
        assert literal in text
