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
    ROOT / "experiments/configs/pivot_feedback_e_g1_execution_input_preflight_v2.yaml"
)
SUMMARY = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.json"
)
SUMMARY_MD = (
    ROOT
    / "experiments/evidence/results/"
    "pivot_feedback_e_g1_execution_input_preflight_v2_20260805.summary.md"
)
LEDGER = ROOT / "docs/research/pivot-feedback-current-status-evidence-ledger-v16.md"
BUILDER_PATH = ROOT / "scripts/build_pivot_feedback_e_g1_execution_input_preflight_v2.py"
VERIFIER_PATH = ROOT / "scripts/verify_pivot_feedback_e_g1_execution_input_preflight_v2.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_module(BUILDER_PATH, "pivot_feedback_e_g1_preflight_v2_builder_test")
VERIFIER = _load_module(VERIFIER_PATH, "pivot_feedback_e_g1_preflight_v2_verifier_test")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority() -> dict:
    return yaml.safe_load(AUTHORITY.read_text(encoding="utf-8"))


def test_v2_corrects_all_three_pre_read_input_omissions_without_scope_change() -> None:
    authority = _authority()

    assert authority["schema_version"] == "pivot_feedback_e_g1_execution_input_preflight.v2"
    assert authority["status"] == (
        "E_G1_EXECUTION_INPUT_PREFLIGHT_V2_COMPLETENESS_PENDING_INDEPENDENT_VALIDATION_NO_G1_RUN"
    )
    diagnosis = authority["completeness_diagnosis"]
    assert diagnosis["detected_before_first_confirmatory_read"] is True
    assert diagnosis["confirmatory_rows_read_before_detection"] == 0
    assert diagnosis["holdout_rows_read_before_detection"] == 0
    assert diagnosis["omitted_required_input_roles"] == [
        "AGE3_CITATION_AVAILABILITY_AND_FUTURE_CITATION_NEGATIVE_CONTROL",
        "FULL_TOPIC_TO_SUBFIELD_ALTERNATIVE_PIVOT_AND_UNRELATED_TOPIC_CONTROL",
        "REQUIRED_ORCID_LINKED_AND_AFFILIATION_SENSITIVITIES",
    ]
    assert diagnosis["scientific_rule_or_threshold_changed"] is False
    assert diagnosis["registered_scope_changed"] is False
    assert diagnosis["this_correction_is_new_scientific_gate"] is False


def test_v2_binds_exact_citation_taxonomy_and_raw_authorship_sources() -> None:
    inputs = _authority()["added_exact_non_holdout_inputs"]

    citation = inputs["citation_chronology"]
    assert citation["selected_artifact_count"] == 4096
    assert citation["selected_artifact_bytes"] == 22_721_388_007
    assert citation["selected_artifact_inventory_sha256"] == (
        "afac933e0fb067d4b69486b8f038cf1cd5fc3d3d1de04fa13f33b165d8dfc98b"
    )
    assert citation["analytic_artifact_files_opened_during_preflight"] == 0

    taxonomy = inputs["topic_taxonomy_projection"]
    assert taxonomy["sha256"] == (
        "566facdbc02bf4a37783fe64db4c1d104d00b73e250999bd824d560083827ba5"
    )
    assert taxonomy["rows"] == 4516
    assert taxonomy["analytic_outcome_rows_read_during_preflight"] == 0

    authorship = inputs["raw_authorship_orcid_affiliation"]
    assert authorship["selected_file_count"] == 1363
    assert authorship["selected_file_bytes"] == 103_412_497_044
    assert authorship["path_size_inventory_sha256"] == (
        "5ca11e55066c14751a0fd2dc3d8631ef75621109026e8b5fb7250f6ebdaf90a9"
    )
    assert authorship["source_parquet_files_opened_during_preflight"] == 0
    assert authorship["source_rows_read_during_preflight"] == 0


def test_effective_contract_is_parent_plus_v2_and_stays_g4_excluded() -> None:
    authority = _authority()
    effective = authority["effective_execution_input_contract"]

    assert effective["requires_parent_v1_and_this_v2"] is True
    assert effective["effective_input_group_count"] == 7
    assert effective["input_group_ids"] == [
        "normalized_work_metadata",
        "author_history_events",
        "aligned_final_panel_metadata",
        "atomic_partition_metadata",
        "citation_chronology",
        "topic_taxonomy_projection",
        "raw_authorship_orcid_affiliation",
    ]
    assert effective["private_g4_holdout_included"] is False
    assert effective["g4_field_or_year_included"] is False

    extension = authority["execution_output_contract_extension"]
    assert extension["effective_output_artifact_schema_count"] == 8
    assert extension["added_artifact"]["relative_path"] == (
        "g1_identity_affiliation_sensitivities.parquet"
    )
    assert extension["orcid_linked_is_required_sensitivity_not_primary_replacement"] is True
    assert extension["affiliation_is_current_snapshot_proxy_not_historical_truth"] is True


def test_independent_summary_replays_v2_and_remains_fail_closed() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    assert summary["status"] == (
        "PASS_INDEPENDENT_E_G1_EXECUTION_INPUT_PREFLIGHT_V2_COMPLETE_NO_G1_RUN"
    )
    artifacts = summary["repository_artifacts"]
    assert artifacts["preflight_contract"]["sha256"] == _sha(AUTHORITY)
    assert artifacts["builder"]["sha256"] == _sha(BUILDER_PATH)
    assert artifacts["independent_verifier"]["sha256"] == _sha(VERIFIER_PATH)
    assert summary["effective_contract_replay"]["effective_input_group_count"] == 7
    assert summary["effective_contract_replay"]["effective_output_artifact_schema_count"] == 8
    assert summary["completeness_correction_replay"]["this_is_new_scientific_gate"] is False
    assert summary["fail_closed_state"]["g1_execution_run"] is False
    assert summary["fail_closed_state"]["confirmatory_rows_read"] == 0
    assert summary["fail_closed_state"]["holdout_rows_read"] == 0
    assert summary["fail_closed_state"]["g4_open"] is False


def test_verifier_rejects_diagnosis_drift_before_input_replay(tmp_path: Path) -> None:
    mutated = _authority()
    mutated["completeness_diagnosis"]["registered_scope_changed"] = True
    path = tmp_path / "mutated.yaml"
    path.write_text(yaml.safe_dump(mutated, sort_keys=False), encoding="utf-8")

    with pytest.raises(
        VERIFIER.G1InputPreflightV2VerificationError,
        match="completeness diagnosis drift",
    ):
        VERIFIER.verify(path)


def test_v2_tools_are_independent_create_only_and_row_safe() -> None:
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
        assert imports.isdisjoint({"duckdb", "numpy", "pandas"})
        assert "pyarrow" in imports

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
        module.endswith("build_pivot_feedback_e_g1_execution_input_preflight_v2")
        for module in verifier_imports
    )

    before = AUTHORITY.read_bytes()
    with pytest.raises(BUILDER.G1InputPreflightV2BuildError, match="output exists"):
        BUILDER._write_create_only(AUTHORITY, _authority())
    assert AUTHORITY.read_bytes() == before


def test_human_summaries_keep_execution_not_run_boundary() -> None:
    summary_md = SUMMARY_MD.read_text(encoding="utf-8")
    ledger = LEDGER.read_text(encoding="utf-8")

    assert "pre-read completeness correction" in summary_md
    assert "No G1 result exists" in summary_md
    assert "not a new scientific gate" in ledger
    assert "parent v1 plus independently verified v2" in ledger
    assert "confirmatory rows read remain zero" in ledger
