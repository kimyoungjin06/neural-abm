from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "experiments/configs/"
    "pivot_feedback_g0_risk_set_weight_support_precommitment_v1.yaml"
)
FIXTURE_INPUT = (
    ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_risk_set_weight_support_discovery_v1.input.json"
)
FIXTURE_EXPECTED = (
    ROOT
    / "tests/fixtures/"
    "pivot_feedback_g0_risk_set_weight_support_discovery_v1.expected.json"
)
APPLICATION = (
    ROOT
    / "scripts/apply_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py"
)
VERIFIER = (
    ROOT
    / "scripts/verify_pivot_feedback_g0_risk_set_weight_support_precommitment_v1.py"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


APPLICATION_MODULE = _load(APPLICATION, "risk_set_support_application_test_module")
VERIFY_MODULE = _load(VERIFIER, "risk_set_support_verifier_test_module")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_binds_authorization_and_all_design_sources() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))

    assert contract["schema_version"] == VERIFY_MODULE.CONTRACT_SCHEMA
    assert contract["status"] == VERIFY_MODULE.CONTRACT_STATUS
    parent = contract["authorization_parent"]
    assert parent["sha256"] == _sha256(ROOT / parent["relative_path"])
    for binding in contract["bound_design_sources"].values():
        assert binding["sha256"] == _sha256(ROOT / binding["relative_path"])
        assert binding["mutation_allowed"] is False


def test_contract_freezes_exact_six_policy_values_and_failure_actions() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    choices = contract["precommitment_choices"]

    assert contract["scope"]["choice_ids"] == VERIFY_MODULE.CHOICE_IDS
    assert choices["negative_sampling_ratio"]["ratio_controls_per_case"] == 10
    assert choices["inverse_probability_weight_truncation"][
        "lower_percentile"
    ] == 0.01
    assert choices["inverse_probability_weight_truncation"][
        "upper_percentile"
    ] == 0.99
    assert choices["minimum_cohort_size"][
        "minimum_total_opportunity_rows"
    ] == 500
    assert choices["minimum_cohort_size"]["minimum_entry_events"] == 100
    assert choices["minimum_cohort_size"][
        "minimum_non_entry_opportunities"
    ] == 100
    assert choices["small_cell_suppression_threshold"][
        "suppressed_inclusive_upper"
    ] == 10
    capacity = choices["capacity_cell_support_threshold"]
    assert (
        capacity["minimum_total_opportunities_per_cell"],
        capacity["minimum_entry_events_per_cell"],
        capacity["minimum_non_entry_opportunities_per_cell"],
    ) == (50, 11, 11)
    collinearity = choices["collinearity_threshold"]
    assert collinearity["threshold"] == 5.0
    assert collinearity["threshold_boundary_counts_as_pass"] is False
    assert capacity["failed_action"] == collinearity["failed_action"]


def test_application_reproduces_canonical_expected_fixture() -> None:
    result = APPLICATION_MODULE.run(CONTRACT, FIXTURE_INPUT)
    expected = json.loads(FIXTURE_EXPECTED.read_text(encoding="utf-8"))

    assert result == expected


def test_fixture_covers_sampling_and_weight_boundaries() -> None:
    expected = json.loads(FIXTURE_EXPECTED.read_text(encoding="utf-8"))
    sampling = expected["sampling_application"]
    weights = expected["weight_application"]

    assert sampling["stratum_count"] == 2
    assert sampling["selected_control_count"] == 25
    assert sampling["strata"][0]["control_inclusion_probability"] == 0.8
    assert sampling["strata"][0]["control_raw_sampling_weight"] == 1.25
    assert sampling["strata"][1]["all_controls_retained"] is True
    assert weights["clipped_high_count"] == 1
    group = weights["groups"][0]
    assert group["lower_nearest_rank"] == 1
    assert group["upper_nearest_rank"] == 99
    assert group["raw_maximum"] == 1000.0
    assert group["upper_cutoff"] == 1.0
    assert group["winsorized_kish_effective_sample_size"] == 100.0


def test_fixture_covers_cohort_and_reporting_boundaries() -> None:
    expected = json.loads(FIXTURE_EXPECTED.read_text(encoding="utf-8"))
    cohort = expected["cohort_support_application"]
    cells = {
        item["cell_id"]: item for item in expected["small_cell_application"]["cells"]
    }

    assert cohort["pass_count"] == 1
    assert cohort["fail_count"] == 3
    reasons = {
        item["group_id"]: item["failure_reasons"] for item in cohort["groups"]
    }
    assert reasons["cohort-total-fail"] == ["TOTAL_BELOW_500"]
    assert reasons["cohort-event-fail"] == ["ENTRY_EVENTS_BELOW_100"]
    assert reasons["cohort-non-event-fail"] == [
        "NON_ENTRY_OPPORTUNITIES_BELOW_100"
    ]
    assert cells["zero"]["reported_count"] == 0
    assert cells["one"]["reported_count"] is None
    assert cells["ten"]["reported_count"] is None
    assert cells["eleven"]["reported_count"] == 11


def test_fixture_covers_support_and_vif_collapse_paths() -> None:
    expected = json.loads(FIXTURE_EXPECTED.read_text(encoding="utf-8"))
    application = expected["capacity_collinearity_application"]
    tables = {item["table_id"]: item for item in application["tables"]}

    assert application["support_fail_count"] == 1
    assert application["vif_fail_count"] == 1
    assert application["separate_channel_pass_count"] == 1
    assert tables["capacity-pass"]["maximum_vif"] == 3.0
    assert tables["capacity-pass"]["separate_channel_pass"] is True
    assert tables["capacity-support-fail"]["support_failure_reasons"] == [
        "PUBLIC_SUCCESS_ONLY:ENTRY_EVENTS_BELOW_11"
    ]
    assert tables["capacity-support-fail"]["vif_pass"] is True
    assert tables["capacity-vif-fail"]["support_pass"] is True
    assert tables["capacity-vif-fail"]["maximum_vif"] > 5.0
    assert tables["capacity-vif-fail"]["separate_channel_pass"] is False


def test_independent_verifier_is_metadata_and_fixture_only() -> None:
    tree = ast.parse(VERIFIER.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported.isdisjoint({"duckdb", "numpy", "pandas", "pyarrow"})
    assert "apply_pivot_feedback_g0_risk_set_weight_support" not in imported

    result = VERIFY_MODULE.run()

    assert result["status"] == VERIFY_MODULE.VERIFY_STATUS
    assert result["choice_replay"]["choice_count"] == 6
    assert result["fixture_replay"]["cohort_fail_count"] == 3
    assert result["fixture_replay"]["capacity_support_fail_count"] == 1
    assert result["fixture_replay"]["capacity_vif_fail_count"] == 1
    assert result["application_tool"]["imports_this_validator"] is False
    assert result["access_accounting"]["production_row_level_files_opened"] == 0
    assert result["access_accounting"]["confirmatory_or_holdout_rows_read"] == 0


def test_application_rejects_post_discovery_decision_year(tmp_path: Path) -> None:
    fixture = json.loads(FIXTURE_INPUT.read_text(encoding="utf-8"))
    fixture["sampling_strata"][0]["decision_year"] = 2013
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(
        APPLICATION_MODULE.RiskSetWeightSupportApplicationError,
        match="post-discovery decision year",
    ):
        APPLICATION_MODULE.run(CONTRACT, mutated)


def test_application_rejects_duplicate_control_id(tmp_path: Path) -> None:
    fixture = json.loads(FIXTURE_INPUT.read_text(encoding="utf-8"))
    fixture["sampling_strata"][0]["control_ids"][1] = fixture[
        "sampling_strata"
    ][0]["control_ids"][0]
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(
        APPLICATION_MODULE.RiskSetWeightSupportApplicationError,
        match="duplicate control ID",
    ):
        APPLICATION_MODULE.run(CONTRACT, mutated)


def test_verifier_rejects_policy_drift_in_memory() -> None:
    contract = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    contract["precommitment_choices"]["collinearity_threshold"]["threshold"] = 10.0

    with pytest.raises(
        VERIFY_MODULE.RiskSetWeightSupportVerificationError,
        match="VIF threshold",
    ):
        VERIFY_MODULE._validate_choices(contract)
