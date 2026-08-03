from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/verify_pivot_feedback_d0_persistence_outcome_contract_v1.py"
)
CONTRACT = (
    ROOT
    / "experiments/configs/pivot_feedback_d0_persistence_outcome_contract_v1.yaml"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_persistence_outcome_contract_v1_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _write(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _document() -> dict[str, object]:
    value = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_current_contract_passes_without_opening_production_rows() -> None:
    result = MODULE.validate(CONTRACT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["contract_sha256"] == (
        "177fd11366c36143f601743a27afb952b1482c2f90bc536b95787e114c482b6c"
    )
    assert result["repository_bindings_validated"] == 11
    assert result["operator_receipts_read"] == 3
    assert result["operator_parquet_files_opened"] == 0
    assert result["production_classification_rows_read"] == 0
    assert result["production_post_entry_rows_read"] == 0
    assert result["api_requests_created"] == 0
    assert result["production_post_entry_source_build_authorized"] is False
    assert result["discovery_outcome_analysis_authorized"] is False


def test_contract_uses_only_new_topic_entries_as_persistence_denominator() -> None:
    document = _document()
    population = document["outcome_population"]
    definition = document["persistence_definition"]
    source = document["production_source_contract"]

    assert population["source_rows"] == 3_256_271
    assert population["exact_denominator_rule"] == (
        "observable_history_pivot_state == NEW_TOPIC_ENTRY"
    )
    assert population["exact_denominator_rows"] == 4_301
    assert population["nonprimary_rows_materialized_in_persistence_output"] is False
    assert population["all_primary_pivot_rows_must_be_retained"] is True
    assert definition["outcome_window_offsets"] == [1, 2, 3]
    assert definition["qualifying_count_threshold"] == 1
    assert definition["future_outcome_does_not_redefine_primary_pivot"] is True
    assert definition["true_researcher_topic_persistence_claim_allowed"] is False
    assert source["required_output_rows"] == 4_301
    assert tuple(source["exact_output_columns"]) == MODULE.OUTPUT_COLUMNS


def test_fixture_replay_covers_positive_zero_and_right_censoring() -> None:
    result = MODULE.validate(CONTRACT)
    fixtures = {item["fixture_id"]: item for item in result["fixture_replay"]}

    assert result["fixture_outcome_states"] == [
        "MATURED_NO_PERSISTENCE",
        "MATURED_PERSISTENCE",
        "RIGHT_CENSORED",
    ]
    assert result["fixture_row_mismatches"] == 0
    assert result["fixture_accounting_mismatches"] == 0
    complete = fixtures["complete_three_year_window"]
    assert complete["accounting"]["primary_pivot_rows"] == 5
    assert complete["accounting"]["persistent_primary_rows"] == 1
    assert complete["accounting"]["nonpersistent_primary_rows"] == 4
    assert complete["accounting"][
        "registered_primary_distinct_author_work_rows"
    ] == 3
    assert complete["accounting"]["duplicate_author_work_event_excess_rows"] == 1
    censored = fixtures["incomplete_window_right_censor_sentinel"]
    assert censored["matured"] is False
    assert censored["accounting"]["right_censored_primary_rows"] == 1
    assert censored["reconstructed_rows"][0][
        "three_year_target_topic_persistent"
    ] is None


def test_partial_positive_work_cannot_override_right_censoring() -> None:
    result = MODULE.validate(CONTRACT)
    censored = next(
        item
        for item in result["fixture_replay"]
        if item["fixture_id"] == "incomplete_window_right_censor_sentinel"
    )
    row = censored["reconstructed_rows"][0]

    assert row["persistence_followup_complete"] is False
    assert row["target_topic_work_count_t_plus_1"] is None
    assert row["three_year_target_topic_persistence_state"] == "RIGHT_CENSORED"
    assert row["three_year_target_topic_persistent"] is None
    assert result["partial_positive_right_censored_fixture_passed"] is True


def test_output_contract_contains_no_impact_model_or_g4_columns() -> None:
    document = _document()
    columns = document["production_source_contract"]["exact_output_columns"]

    assert not any(
        fragment in column
        for column in columns
        for fragment in ("impact", "citation", "peer", "model", "g4")
    )
    assert document["production_source_contract"][
        "output_may_include_impact_citation_peer_model_or_g4_columns"
    ] is False


def test_denominator_mutation_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["outcome_population"]["exact_denominator_rows"] = 4_300

    with pytest.raises(
        MODULE.PersistenceOutcomeContractError,
        match="denominator rows",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_outcome_horizon_mutation_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["persistence_definition"]["outcome_window_offsets"] = [1, 2]

    with pytest.raises(
        MODULE.PersistenceOutcomeContractError,
        match="outcome offsets",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_confirmatory_scope_mutation_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["exact_episode_scope"]["confirmatory_entry_years_allowed"] = True

    with pytest.raises(
        MODULE.PersistenceOutcomeContractError,
        match="confirmatory scope",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_fixture_expected_boolean_is_reconstructed_not_trusted(
    tmp_path: Path,
) -> None:
    document = _document()
    document["synthetic_fixtures"][0]["expected_rows"][0][
        "three_year_target_topic_persistent"
    ] = False

    with pytest.raises(
        MODULE.PersistenceOutcomeContractError,
        match="fixture reconstructed rows",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_receipt_hash_mutation_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["operator_receipt_bindings"]["production_classification"][
        "receipt_sha256"
    ] = "0" * 64

    with pytest.raises(
        MODULE.PersistenceOutcomeContractError,
        match="receipt SHA",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_production_source_authorization_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["production_post_entry_source_build"] = "AUTHORIZED"

    with pytest.raises(
        MODULE.PersistenceOutcomeContractError,
        match="unsafe authorization",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))
