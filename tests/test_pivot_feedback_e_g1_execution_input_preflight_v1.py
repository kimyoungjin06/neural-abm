from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY = (
    ROOT
    / "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v1.yaml"
)
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.json"
)
SUMMARY_MD = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_e_g1_execution_input_preflight_v1_20260805.summary.md"
)
LEDGER = (
    ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v15.md"
)
BUILDER_PATH = (
    ROOT / "scripts/build_pivot_feedback_e_g1_execution_input_preflight_v1.py"
)
VERIFIER_PATH = (
    ROOT / "scripts/verify_pivot_feedback_e_g1_execution_input_preflight_v1.py"
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module(BUILDER_PATH, "pivot_feedback_e_g1_preflight_builder_test")
VERIFIER = _load_module(VERIFIER_PATH, "pivot_feedback_e_g1_preflight_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority() -> dict:
    return yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))


def _verify_mutation(tmp_path: Path, payload: dict) -> str:
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(VERIFIER.G1InputPreflightVerificationError) as error:
        VERIFIER.verify(path)
    return str(error.value)


def test_preflight_binds_exact_ready_e_authority_and_frozen_scope() -> None:
    authority = _authority()

    assert authority["schema_version"] == (
        "pivot_feedback_e_g1_execution_input_preflight.v1"
    )
    assert authority["status"] == (
        "E_G1_EXECUTION_INPUT_PREFLIGHT_COMPLETE_PENDING_INDEPENDENT_VALIDATION_NO_G1_RUN"
    )
    parent = authority["authorization_parent"]
    assert parent["relative_path"] == (
        "experiments/configs/pivot_feedback_current_state_v2.yaml"
    )
    assert parent["sha256"] == (
        "ca431afb63da4cf2e09d0563c0509e3685efac29dc8679df0f9cc7f1ae7851ab"
    )
    assert parent["required_active_phase"] == "E"
    assert parent["required_phase_state"] == "READY_NOT_RUN"
    assert parent["required_pass_f"] is True
    assert parent["required_empirical_g1_status"] == "NOT_RUN"
    assert parent["required_holdout_seal_opened"] is False

    scope = authority["registered_execution_scope"]
    assert scope["partition_id"] == "g1_empirical_replication"
    assert [scope["entry_year_start"], scope["entry_year_end"]] == [2013, 2016]
    assert scope["selected_field_ids_in_registered_slot_order"] == [16, 32, 11]
    assert scope["assigned_episode_count"] == 28
    assert scope["assigned_family_count"] == 13
    assert scope["g4_field_id"] == 27
    assert scope["g4_field_allowed"] is False
    assert scope["g4_private_artifact_root_or_manifest_is_execution_input"] is False


def test_exact_non_holdout_inventory_is_receipt_bound_without_analytic_read() -> None:
    inputs = _authority()["exact_non_holdout_inputs"]
    work = inputs["normalized_work_metadata"]
    history = inputs["author_history_events"]

    assert work["selected_artifact_count"] == 4096
    assert work["selected_artifact_bytes"] == 4_120_921_809
    assert work["selected_artifact_inventory_sha256"] == (
        "68d57f5425d56a5e008694fdcb6a36521ba977a3fbabb5935c3ad023abcaa5e9"
    )
    assert history["selected_artifact_count"] == 4096
    assert history["selected_artifact_bytes"] == 5_006_392_019
    assert history["selected_artifact_inventory_sha256"] == (
        "59cfe5d5b235f26d4a2b8ad78f5679c15d341ac0778f92d3beb30929cb17aa4c"
    )
    for binding in (work, history):
        assert binding["receipt_sha256_transitively_binds_exact_artifact_array"] is True
        assert binding["analytic_artifact_files_opened_during_preflight"] == 0
        assert binding["analytic_artifact_bytes_read_during_preflight"] == 0

    final_panel = inputs["aligned_final_panel_metadata"]
    partitions = inputs["atomic_partition_metadata"]
    assert final_panel["selected_artifact_count"] == 2
    assert final_panel["selected_artifact_bytes"] == 23_905
    assert partitions["selected_artifact_count"] == 3
    assert partitions["selected_artifact_bytes"] == 15_943
    assert set(inputs["private_g4_holdout"].values()) == {False}


def test_label_control_estimand_and_output_contracts_are_frozen() -> None:
    authority = _authority()
    labels = authority["label_materialization_contract"]
    assert labels["materialization_occurs_in_this_preflight"] is False
    assert labels["entry_outcome"] == "OBSERVABLE_HISTORY_NEW_TOPIC_ENTRY"
    assert labels["persistence_horizon_offsets"] == [1, 2, 3]
    assert labels["failure_label_allowed"] is False
    assert labels["outcome_imputation"] == "NONE"
    assert labels["risk_set_sampling_and_weights"]["controls_per_case"] == 10
    assert labels["risk_set_sampling_and_weights"]["truncation"] == (
        "NEAREST_RANK_P01_P99_WINSORIZATION"
    )

    controls = authority["negative_control_contract"]
    assert controls["required_control_count"] == 7
    assert len(controls["probability_difference_equivalence"]["control_ids"]) == 5
    assert len(controls["true_minus_placebo"]["control_ids"]) == 2
    assert controls["materialization_or_decision_occurs_in_this_preflight"] is False

    estimand = authority["estimand_and_decision_contract"]
    assert estimand["estimation_occurs_in_this_preflight"] is False
    assert estimand["replication_noninferiority"]["lower_margin_cohort_years"] == -1.0
    assert estimand["replication_noninferiority"]["boundary_counts_as_pass"] is False
    assert estimand["uncertainty"]["bootstrap_replicates"] == 9999
    assert estimand["predictive_or_nabm_model_selection"] == "NOT_AUTHORIZED_IN_E"

    output = authority["execution_output_contract"]
    assert output["output_directory_must_be_new"] is True
    assert output["receipt_written_last"] is True
    assert len(output["artifacts"]) == 7
    assert output["allowed_terminal_statuses"] == [
        "PASS_E_G1_BOUNDED_OBSERVATIONAL_SIGNATURE",
        "FAIL_E_G1_REGISTERED_DECISION_NO_ROUTE_PROMOTION",
        "BLOCKED_E_G1_INPUT_SUPPORT_OR_ESTIMATOR_FAILURE_NO_ROUTE_PROMOTION",
    ]


def test_independent_summary_replays_preflight_and_stays_fail_closed() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["status"] == (
        "PASS_INDEPENDENT_E_G1_EXECUTION_INPUT_PREFLIGHT_EXACT_NON_HOLDOUT_SCOPE_NO_G1_RUN"
    )
    assert summary["repository_artifacts"]["preflight_contract"]["sha256"] == _sha(
        AUTHORITY
    )
    assert summary["repository_artifacts"]["builder"]["sha256"] == _sha(
        BUILDER_PATH
    )
    assert summary["repository_artifacts"]["independent_verifier"][
        "sha256"
    ] == _sha(VERIFIER_PATH)
    assert summary["registered_scope_replay"] == {
        "entry_years": [2013, 2016],
        "episode_count": 28,
        "family_count": 13,
        "field_count": 3,
        "field_ids_in_registered_slot_order": [16, 32, 11],
        "g1_g4_cross_partition_families": 0,
        "g4_field_excluded": True,
        "g4_private_artifacts_included": False,
        "partition_id": "g1_empirical_replication",
    }
    assert summary["independent_preflight"]["complete"] is True
    assert summary["independent_preflight"]["this_is_new_scientific_gate"] is False
    assert summary["fail_closed_state"]["g1_execution_run"] is False
    assert summary["fail_closed_state"]["empirical_g1_status"] == "NOT_RUN"
    assert summary["fail_closed_state"]["confirmatory_rows_read"] == 0
    assert summary["fail_closed_state"]["holdout_rows_read"] == 0
    assert summary["fail_closed_state"]["holdout_seal_opened"] is False
    assert summary["fail_closed_state"]["g4_open"] is False


def test_independent_verifier_rejects_scope_inventory_and_access_drift(
    tmp_path: Path,
) -> None:
    wrong_scope = _authority()
    wrong_scope["registered_execution_scope"][
        "selected_field_ids_in_registered_slot_order"
    ] = [16, 32, 27]
    assert "field scope drift" in _verify_mutation(tmp_path, wrong_scope)

    wrong_inventory = _authority()
    wrong_inventory["exact_non_holdout_inputs"]["normalized_work_metadata"][
        "selected_artifact_inventory_sha256"
    ] = "0" * 64
    assert "artifact inventory fingerprint drift" in _verify_mutation(
        tmp_path, wrong_inventory
    )

    unsafe_access = _authority()
    unsafe_access["preflight_access_accounting"]["confirmatory_rows_read"] = 1
    assert "access accounting drift" in _verify_mutation(tmp_path, unsafe_access)


def test_tools_are_metadata_only_independent_and_create_only() -> None:
    forbidden = {"duckdb", "numpy", "pandas", "pyarrow"}
    for path in (BUILDER_PATH, VERIFIER_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imports.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert imports.isdisjoint(forbidden)

    verifier_tree = ast.parse(VERIFIER_PATH.read_text(encoding="utf-8"))
    verifier_imports = {
        node.module
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    verifier_imports.update(
        alias.name
        for node in ast.walk(verifier_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        module.endswith("build_pivot_feedback_e_g1_execution_input_preflight_v1")
        for module in verifier_imports
    )

    before = AUTHORITY.read_bytes()
    with pytest.raises(BUILDER.G1InputPreflightBuildError, match="output exists"):
        BUILDER._write_yaml_create_only(AUTHORITY, _authority())
    assert AUTHORITY.read_bytes() == before


def test_summary_ledger_and_authorization_stop_before_g1_execution() -> None:
    authority = _authority()
    summary_md = SUMMARY_MD.read_text(encoding="utf-8")
    ledger = LEDGER.read_text(encoding="utf-8")

    assert authority["authorization"]["exact_g1_execution_successor"] == (
        "AUTHORIZED_ONLY_AFTER_INDEPENDENT_PREFLIGHT_PASS"
    )
    assert authority["authorization"]["g4_artifact_access"] == "NOT_AUTHORIZED"
    assert authority["authorization"]["model_selection_score_read"] == (
        "NOT_AUTHORIZED"
    )
    assert "metadata-only" in summary_md
    assert "No G1 estimate or decision exists yet" in ledger
    assert "single E entry boundary" in ledger
    assert "separate exact E/G1 execution successor" in ledger
