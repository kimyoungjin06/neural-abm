from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "scripts/verify_pivot_feedback_d0_entry_source_classifier_contract_v1.py"
)
CONTRACT = (
    ROOT
    / "experiments/configs/pivot_feedback_d0_entry_source_classifier_contract_v1.yaml"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_d0_entry_source_classifier_contract_v1_test_module",
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


def _features(**updates: int) -> dict[str, int]:
    values = {
        "pre_entry_target_topic_work_count": 0,
        "trailing_3y_target_topic_work_count": 0,
        "focal_confirmed_eligible_target_topic_work_count": 1,
    }
    values.update(updates)
    return values


def test_current_entry_source_contract_passes_without_opening_parquet() -> None:
    result = MODULE.validate(CONTRACT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["contract_sha256"] == (
        "4105e9671d28d42ed758406a772bbca437fbe3caf4c278908d597d154d1d364c"
    )
    assert result["operator_receipts_read"] == 5
    assert result["operator_parquet_files_opened"] == 0
    assert result["production_parquet_rows_read"] == 0
    assert result["api_requests_created"] == 0
    assert result["outcomes_read"] is False
    assert result["production_source_build_authorized"] is False
    assert result["production_pivot_classification_authorized"] is False


def test_topic_level_content_confirmation_needs_no_new_api() -> None:
    result = MODULE.validate(CONTRACT)

    assert result["content_confirmation_is_topic_level_not_work_level"] is True
    assert result["new_api_request_required"] is False
    assert result["semantics"]["content_confirmation_unit"] == (
        "CERTIFIED_EPISODE_TOPIC_NOT_INDIVIDUAL_WORK"
    )
    assert result["semantics"]["individual_work_api_required"] is False
    assert result["semantics"]["exact_four_expected_temporal_disposition"] == 2
    assert result["semantics"]["team_cap"] == "NONE"
    assert result["semantics"]["post_entry_offsets"] == []


def test_fixture_replay_reconstructs_all_states_and_accounting() -> None:
    result = MODULE.validate(CONTRACT)
    fixture = result["fixture_replay"]

    assert result["all_four_states_covered"] is True
    assert fixture["row_mismatches"] == 0
    assert fixture["accounting_mismatches"] == 0
    assert {row["state"] for row in fixture["reconstructed_rows"]} == set(
        MODULE.STATES
    )
    assert fixture["accounting"] == {
        "input_work_rows": 13,
        "input_author_event_rows": 13,
        "duplicate_author_work_event_excess_rows": 1,
        "eligible_linked_target_work_rows_in_t_minus_3_through_t": 5,
        "trailing_distinct_author_work_rows": 1,
        "focal_distinct_author_work_rows": 4,
        "excluded_outside_t_minus_3_through_t_rows": 2,
        "excluded_wrong_topic_or_field_rows": 1,
        "excluded_paratext_or_work_type_rows": 2,
        "excluded_retracted_rows": 1,
        "excluded_temporal_disposition_rows": 1,
        "excluded_zero_valid_author_token_rows": 1,
        "retained_is_retracted_null_rows": 1,
        "registered_author_rows": 5,
        "classified_author_rows": 5,
    }


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (
            _features(focal_confirmed_eligible_target_topic_work_count=0),
            "NO_FOCAL_ENTRY",
        ),
        (_features(), "NEW_TOPIC_ENTRY"),
        (
            _features(
                pre_entry_target_topic_work_count=2,
                trailing_3y_target_topic_work_count=0,
            ),
            "RETURN",
        ),
        (
            _features(
                pre_entry_target_topic_work_count=2,
                trailing_3y_target_topic_work_count=1,
            ),
            "CONTINUATION",
        ),
    ],
)
def test_independent_classifier_state_table(
    features: dict[str, int],
    expected: str,
) -> None:
    assert MODULE._classify(features) == expected


def test_individual_work_api_requirement_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["content_confirmation_resolution"][
        "individual_work_api_label_required"
    ] = True

    with pytest.raises(
        MODULE.EntrySourceClassifierContractError,
        match="content boundary",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_wrong_exact_four_temporal_disposition_fails_closed(
    tmp_path: Path,
) -> None:
    document = _document()
    document["normalized_work_eligibility"][
        "exact_four_expected_temporal_disposition"
    ] = 3

    with pytest.raises(
        MODULE.EntrySourceClassifierContractError,
        match="exact disposition",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_production_source_authorization_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["authorization"]["production_source_build"] = "AUTHORIZED"

    with pytest.raises(
        MODULE.EntrySourceClassifierContractError,
        match="unsafe authorization",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_fixture_expected_state_is_reconstructed_not_trusted(
    tmp_path: Path,
) -> None:
    document = _document()
    document["synthetic_fixture"]["expected_rows"][0]["state"] = "CONTINUATION"

    with pytest.raises(
        MODULE.EntrySourceClassifierContractError,
        match="fixture reconstructed rows",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_receipt_hash_mutation_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["operator_receipt_bindings"]["cutoff_work_metadata"][
        "receipt_sha256"
    ] = "0" * 64

    with pytest.raises(
        MODULE.EntrySourceClassifierContractError,
        match="receipt SHA",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_post_entry_offset_fails_closed(tmp_path: Path) -> None:
    document = _document()
    document["feature_construction"]["post_entry_offsets_allowed"] = [1]

    with pytest.raises(
        MODULE.EntrySourceClassifierContractError,
        match="post-entry offsets",
    ):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))
