"""Focused contract tests for the torch-free bounded-scalar workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from neural_abm.workflow_lite import (
    BoundedScalarWorkflowSpec,
    run_bounded_scalar_workflow,
)


@dataclass
class _Agent:
    agent_id: int = 7
    x: float = 0.5


def _spec(**overrides: Any) -> BoundedScalarWorkflowSpec:
    settings: dict[str, Any] = {
        "domain_question": "q",
        "state_field": "x",
        "channel_name": "x",
        "transition_label": "commit x",
        "commit_mode": "domain_x",
        "round_digits": 2,
    }
    settings.update(overrides)
    return BoundedScalarWorkflowSpec(**settings)


def _run(**overrides: Any) -> Any:
    settings: dict[str, Any] = {
        "agents": [_Agent()],
        "neighbors": [[]],
        "spec": _spec(),
        "local_update": lambda agent: 0.0,
        "domain_transition": lambda agent, value: {},
    }
    settings.update(overrides)
    return run_bounded_scalar_workflow(**settings)


@pytest.mark.parametrize(
    ("source", "overrides"),
    [
        ("micro_fields", {"micro_fields": lambda agent: {"agent_id": 123}}),
        (
            "domain_transition",
            {"domain_transition": lambda agent, value: {"peer_count": 88}},
        ),
        (
            "aggregate_fields",
            {"aggregate_fields": lambda agents: {"mean_social_shift": 66}},
        ),
    ],
)
def test_domain_fields_cannot_overwrite_reserved_audit_fields(
    source: str,
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match=f"{source} cannot overwrite"):
        _run(**overrides)


def test_state_field_cannot_use_reserved_audit_name() -> None:
    with pytest.raises(ValueError, match="state_field is reserved"):
        _spec(state_field="local_shift")


def test_rounding_is_applied_only_when_workflow_is_serialized() -> None:
    result = _run(
        agents=[_Agent(x=0.12345)],
        aggregate_fields=lambda agents: {"domain_metric": 0.98765},
        micro_fields=lambda agent: {"domain_value": 0.45678},
    )

    assert result.aggregate_audit["mean_x"] == 0.12345
    assert result.aggregate_audit["domain_metric"] == 0.98765
    assert result.micro_audit[0]["domain_value"] == 0.45678

    raw = result.to_dict(round_values=False)
    assert raw["aggregate_audit"]["domain_metric"] == 0.98765
    serialized = result.to_dict()
    assert serialized["aggregate_audit"]["mean_x"] == 0.12
    assert serialized["aggregate_audit"]["domain_metric"] == 0.99
    assert serialized["micro_audit"][0]["domain_value"] == 0.46


def test_rounding_never_changes_serialized_execution_settings() -> None:
    result = _run(
        agents=[_Agent(x=0.12345)],
        spec=_spec(
            round_digits=0,
            peer_similarity_threshold=0.68,
            social_alpha=0.45,
        ),
        aggregate_fields=lambda agents: {"domain_metric": 0.98765},
    )

    serialized = result.to_dict()

    assert serialized["unit_standardized"]["peer_similarity_threshold"] == 0.68
    assert serialized["unit_standardized"]["social_alpha"] == 0.45
    assert serialized["aggregate_audit"]["domain_metric"] == 1.0
