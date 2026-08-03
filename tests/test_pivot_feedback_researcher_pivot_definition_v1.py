from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_pivot_feedback_researcher_pivot_definition_v1.py"
CONTRACT = (
    ROOT / "experiments/configs/pivot_feedback_researcher_pivot_definition_v1.yaml"
)


def _load():
    spec = importlib.util.spec_from_file_location(
        "pivot_feedback_researcher_pivot_definition_v1_test_module",
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
        "prior_observable_span_years": 10,
        "prior_eligible_work_count": 20,
        "prior_topic_work_count": 0,
        "trailing_3y_topic_work_count": 0,
        "focal_confirmed_on_topic_work_count": 1,
    }
    values.update(updates)
    return values


def _write(path: Path, value: object) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_current_definition_contract_passes() -> None:
    result = MODULE.validate(CONTRACT)

    assert result["status"] == MODULE.VERIFY_STATUS
    assert result["states"] == MODULE.STATES
    assert result["past_only"] is True
    assert result["observable_history_claim_only"] is True
    assert result["pivot_distance_gates_event"] is False
    assert result["persistence_is_future_outcome"] is True
    assert result["human_review"] == "NOT_RUN_BY_DESIGN"
    assert result["candidate_universe_screen_complete"] is False
    assert result["production_execution_authorized"] is False


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (
            _features(prior_observable_span_years=2),
            "INSUFFICIENT_HISTORY",
        ),
        (
            _features(prior_eligible_work_count=2),
            "INSUFFICIENT_HISTORY",
        ),
        (_features(), "NEW_TOPIC_PIVOT"),
        (
            _features(
                prior_topic_work_count=4,
                trailing_3y_topic_work_count=0,
            ),
            "RETURN",
        ),
        (
            _features(
                prior_topic_work_count=4,
                trailing_3y_topic_work_count=2,
            ),
            "CONTINUATION",
        ),
    ],
)
def test_executable_decision_table(
    features: dict[str, int],
    expected: str,
) -> None:
    assert MODULE.classify_opportunity(features) == expected


def test_focal_work_requires_content_confirmation() -> None:
    with pytest.raises(MODULE.PivotDefinitionError, match="confirmation"):
        MODULE.classify_opportunity(
            _features(focal_confirmed_on_topic_work_count=0)
        )


def test_future_or_outcome_features_are_rejected() -> None:
    with pytest.raises(MODULE.PivotDefinitionError, match="future or outcome"):
        MODULE.classify_opportunity(
            {
                **_features(),
                "persistence_t_plus_3": 1,
            }
        )


def test_inconsistent_topic_counts_are_rejected() -> None:
    with pytest.raises(MODULE.PivotDefinitionError, match="exceeds"):
        MODULE.classify_opportunity(
            _features(
                prior_topic_work_count=1,
                trailing_3y_topic_work_count=2,
            )
        )


def test_contract_cannot_use_distance_as_event_gate(tmp_path: Path) -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    document["pivot_magnitude"]["threshold_for_primary_event"] = 0.5

    with pytest.raises(MODULE.PivotDefinitionError, match="distance gates event"):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_contract_cannot_claim_full_candidate_screen(tmp_path: Path) -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    document["api_only_content_boundary"][
        "complete_candidate_universe_screen_run"
    ] = True

    with pytest.raises(MODULE.PivotDefinitionError, match="full screen claimed"):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))


def test_contract_cannot_open_outcomes(tmp_path: Path) -> None:
    document = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    document["authorization"]["persistence_outcome_read"] = "AUTHORIZED"

    with pytest.raises(MODULE.PivotDefinitionError, match="unsafe authorization"):
        MODULE.validate(_write(tmp_path / "contract.yaml", document))
