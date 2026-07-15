from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES))
study = importlib.import_module("research_pivot_study")
learning = importlib.import_module("research_pivot_learning_study")


def _context(scenario: object, replicate: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        scenario=scenario,
        replicate=replicate,
        rng=study.component_rng(20260715, replicate, "population"),
    )


def test_zero_grant_scale_removes_every_intervention_component() -> None:
    scenarios = {scenario.name: scenario for scenario in study.study_scenarios(0.0)}
    parameters = scenarios["interdisciplinary_seed_grants"].parameters

    assert parameters
    assert set(parameters) == set(study.SEED_GRANT_PARAMETERS)
    assert all(value == 0.0 for value in parameters.values())


def test_base_network_is_common_and_grant_bridges_are_only_additive() -> None:
    config = study.StudyConfig(agent_count=40, replicates=1)
    scenarios = {scenario.name: scenario for scenario in study.study_scenarios()}
    baseline_context = _context(scenarios["baseline"])
    grant_context = _context(scenarios["interdisciplinary_seed_grants"])
    baseline_agents = study.make_researchers(baseline_context, config)
    grant_agents = study.make_researchers(grant_context, config)

    baseline = study.build_stage_assortative_network(
        baseline_agents,
        baseline_context,
        config,
    )
    grants = study.build_stage_assortative_network(
        grant_agents,
        grant_context,
        config,
    )

    assert all(set(base) <= set(treated) for base, treated in zip(baseline, grants))
    assert any(set(base) < set(treated) for base, treated in zip(baseline, grants))


def test_keyed_local_noise_is_scenario_and_arm_independent() -> None:
    config = learning.LearningStudyConfig(agent_count=12, steps=3, replicates=1)
    left = learning.PivotEnvironment(neighbors=[], config=config, replicate=0, step=2)
    right = learning.PivotEnvironment(neighbors=[], config=config, replicate=0, step=2)

    assert left.local_noise(7) == right.local_noise(7)
    assert left.local_noise(7) != left.local_noise(8)


def test_small_population_rejects_impossible_neighbor_count() -> None:
    with pytest.raises(ValueError, match="neighbor_count"):
        study.StudyConfig(agent_count=4, neighbor_count=4)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("local_alpha", 1.1, "local_alpha"),
        ("social_alpha", float("nan"), "social_alpha"),
        ("peer_similarity_threshold", True, "peer_similarity_threshold"),
        ("pivot_threshold", -0.1, "pivot_threshold"),
        ("productive_threshold", 1.1, "productive_threshold"),
        ("local_noise_scale", -0.01, "local_noise_scale"),
        ("success_min_delta", float("inf"), "success_min_delta"),
    ],
)
def test_study_config_rejects_invalid_numeric_contracts(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        study.StudyConfig(**{field: value})


@pytest.mark.parametrize("grant_scale", [True, -0.1, float("nan")])
def test_study_scenarios_reject_invalid_grant_scale(grant_scale: object) -> None:
    with pytest.raises(ValueError, match="grant_scale"):
        study.study_scenarios(grant_scale)


@pytest.mark.parametrize("micro_sample", [True, -1, 1.5])
def test_compaction_rejects_invalid_micro_sample(micro_sample: object) -> None:
    with pytest.raises(ValueError, match="micro_sample"):
        study._compact_scenarios({}, micro_sample)


@pytest.mark.parametrize("burn_in_steps", [True, 1.5, -1, 3])
def test_learning_config_rejects_invalid_burn_in_steps(
    burn_in_steps: object,
) -> None:
    with pytest.raises(ValueError, match="burn_in_steps"):
        learning.LearningStudyConfig(steps=3, burn_in_steps=burn_in_steps)


def test_direct_contrast_uses_paired_replicate_outcomes() -> None:
    scenarios = {
        "grant": {
            "outcome_field": "productive_pivot_rate",
            "outcome_values": [0.1, 0.2, 0.3],
        },
        "hype": {
            "outcome_field": "productive_pivot_rate",
            "outcome_values": [0.2, 0.1, 0.4],
        },
    }

    contrast = study.paired_outcome_contrast(
        scenarios,
        reference="grant",
        comparison="hype",
    )

    assert contrast["paired_deltas"] == [0.1, -0.1, 0.1]
    assert contrast["mean_delta"] == pytest.approx(1 / 30, abs=1e-6)
    assert len(contrast["mean_effect_ci95"]) == 2


def test_vicarious_learning_excludes_the_agents_own_outcome() -> None:
    scenario = next(
        scenario for scenario in study.study_scenarios() if scenario.name == "baseline"
    )
    config = learning.LearningStudyConfig(agent_count=12, steps=3, replicates=1)
    context = _context(scenario)
    base = study.make_researchers(context, study.StudyConfig(agent_count=12))[0]
    environment = learning.PivotEnvironment(
        neighbors=[[0], *[[] for _ in range(config.agent_count - 1)]],
        config=config,
        replicate=0,
    )
    agent = learning.PivotPolicyAgent(
        agent_id=0,
        base=base,
        environment=environment,
        learning_rate=1.0,
    )
    before_weight = agent.weight.detach().clone()
    before_bias = agent.bias.detach().clone()

    agent._learn_from_events(
        [
            learning.PivotEvent(
                agent_id=0,
                features=agent.features.copy(),
                productive=False,
                step=1,
            )
        ]
    )

    assert torch.equal(agent.weight, before_weight)
    assert torch.equal(agent.bias, before_bias)


def test_single_replicate_intervals_are_marked_unavailable() -> None:
    distribution = learning._distribution([0.25])
    contrast = study.paired_outcome_contrast(
        {
            "reference": {"outcome_field": "rate", "outcome_values": [0.1]},
            "comparison": {"outcome_field": "rate", "outcome_values": [0.2]},
        },
        reference="reference",
        comparison="comparison",
    )

    assert distribution["mean_ci95"] is None
    assert distribution["mean_ci_method"] == (
        "unavailable_requires_at_least_2_replicates"
    )
    assert contrast["mean_effect_ci95"] is None
    assert contrast["delta_ci95_method"] == (
        "unavailable_requires_at_least_2_replicates"
    )


def test_zero_learning_rate_is_an_exact_arm_null() -> None:
    config = learning.LearningStudyConfig(
        agent_count=12,
        steps=3,
        burn_in_steps=1,
        replicates=2,
        learning_rate=0.0,
    )
    payload = learning.run_learning_study(config)

    for arms in payload["scenarios"].values():
        frozen = arms["frozen"]
        for arm_name in ("imitative", "cautionary"):
            adaptive = arms[arm_name]
            assert adaptive["replicate_runs"] == frozen["replicate_runs"]
            assert (
                adaptive["mean_weight_trajectories"]
                == frozen["mean_weight_trajectories"]
            )
            assert adaptive["mean_bias_trajectory"] == frozen["mean_bias_trajectory"]


def test_learning_artifact_keeps_raw_runs_full_config_and_all_parameters() -> None:
    config = learning.LearningStudyConfig(
        agent_count=12,
        steps=3,
        burn_in_steps=1,
        replicates=2,
    )
    payload = learning.run_learning_study(config)

    assert payload["config"]["neighbor_count"] == 4
    assert payload["config"]["prior_anchor"] == config.prior_anchor
    assert (
        payload["policy_initialization"]["self_outcomes_in_learning_samples"] is False
    )
    for arms in payload["scenarios"].values():
        for summary in arms.values():
            assert len(summary["replicate_runs"]) == config.replicates
            assert set(summary["mean_weight_trajectories"]) == set(learning.FEATURES)
            assert len(summary["mean_bias_trajectory"]) == config.steps
    for comparison in payload["comparisons"]:
        assert len(comparison["paired_deltas"]) == config.replicates
        assert len(comparison["empirical_interval_95"]) == 2
        assert len(comparison["mean_ci95"]) == 2
        assert np.isfinite(comparison["paired_deltas"]).all()
