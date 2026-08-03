from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_researcher_pivot_definition_v2.py"
CONTRACT = (
    ROOT / "experiments/configs/pivot_feedback_researcher_pivot_definition_v2.yaml"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_researcher_pivot_definition_v2_test_module",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load()


def _features(**updates: int) -> dict[str, int]:
    values = {
        "pre_entry_target_topic_work_count": 0,
        "trailing_3y_target_topic_work_count": 0,
        "focal_confirmed_eligible_target_topic_work_count": 1,
    }
    values.update(updates)
    return values


def _write(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_current_v2_contract_passes_without_reading_outcomes() -> None:
    result = MODULE.validate(CONTRACT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["states"] == MODULE.STATES
    assert result["threshold_free"] is True
    assert result["primary_minimum_history_threshold"] == "NONE"
    assert result["past_only"] is True
    assert result["observable_history_claim_only"] is True
    assert result["production_sources_read"] is False
    assert result["focal_year_rows_read"] is False
    assert result["future_rows_read"] is False
    assert result["outcomes_read"] is False
    assert result["api_requests_created"] == 0
    assert result["production_execution_authorized"] is False
    assert result["bounded_exhaustive_replay"] == {
        "valid_count_tuples": 45,
        "state_counts": {
            "NO_FOCAL_ENTRY": 15,
            "NEW_TOPIC_ENTRY": 2,
            "RETURN": 8,
            "CONTINUATION": 20,
        },
        "mismatches": 0,
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
                pre_entry_target_topic_work_count=5,
                trailing_3y_target_topic_work_count=0,
            ),
            "RETURN",
        ),
        (
            _features(
                pre_entry_target_topic_work_count=5,
                trailing_3y_target_topic_work_count=2,
            ),
            "CONTINUATION",
        ),
    ],
)
def test_threshold_free_four_state_table(
    features: dict[str, int],
    expected: str,
) -> None:
    assert MODULE.classify_opportunity(features) == expected


def test_zero_prior_history_is_not_silently_excluded() -> None:
    assert MODULE.classify_opportunity(_features()) == "NEW_TOPIC_ENTRY"


def test_sensitivity_counts_are_not_event_features() -> None:
    with pytest.raises(MODULE.PivotDefinitionV2Error, match="exact feature set"):
        MODULE.classify_opportunity(
            {
                **_features(),
                "prior_observable_span_years": 40,
                "prior_eligible_work_count": 500,
            }
        )
    result = MODULE.validate(CONTRACT)
    assert result["sensitivity_invariance"] == {
        "profiles": 3,
        "state": "NEW_TOPIC_ENTRY",
        "state_changes": 0,
    }


def test_future_and_unknown_features_fail_closed() -> None:
    with pytest.raises(MODULE.PivotDefinitionV2Error, match="forbidden feature"):
        MODULE.classify_opportunity(
            {
                **_features(),
                "persistence_t_plus_3": 1,
            }
        )
    with pytest.raises(MODULE.PivotDefinitionV2Error, match="exact feature set"):
        MODULE.classify_opportunity(
            {
                **_features(),
                "arbitrary_score": 1,
            }
        )


def test_inconsistent_or_noninteger_counts_fail_closed() -> None:
    with pytest.raises(MODULE.PivotDefinitionV2Error, match="exceeds pre-entry"):
        MODULE.classify_opportunity(
            _features(
                pre_entry_target_topic_work_count=1,
                trailing_3y_target_topic_work_count=2,
            )
        )
    with pytest.raises(MODULE.PivotDefinitionV2Error, match="nonnegative integer"):
        MODULE.classify_opportunity(
            _features(pre_entry_target_topic_work_count=True)
        )


def test_contract_cannot_restore_v1_history_threshold(tmp_path: Path) -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    document["classification_input"]["minimum_prior_eligible_work_count"] = 3

    with pytest.raises(MODULE.PivotDefinitionV2Error, match="work threshold"):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_contract_cannot_claim_uncontracted_sources(tmp_path: Path) -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    document["future_source_boundary"][
        "focal_confirmed_eligible_target_topic_work_count_source"
    ] = "PRODUCTION_READY"

    with pytest.raises(MODULE.PivotDefinitionV2Error, match="focal source"):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_contract_cannot_open_production_or_outcomes(tmp_path: Path) -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    document["authorization"]["production_pivot_classification"] = "AUTHORIZED"

    with pytest.raises(MODULE.PivotDefinitionV2Error, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_fixture_expected_state_is_executed_not_trusted(tmp_path: Path) -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    document["valid_fixtures"][2]["expected_state"] = "RETURN"

    with pytest.raises(MODULE.PivotDefinitionV2Error, match="fixture state"):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))
