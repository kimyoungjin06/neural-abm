from __future__ import annotations

import numpy as np

from neural_abm.basin_phase_critic import (
    learned_basin_runtime_aggregate_fields,
    learned_basin_runtime_micro_fields,
)
from neural_abm.domain_learning_diagnostics import (
    DOMAIN_LEARNING_AGGREGATE_FIELDS,
    DOMAIN_LEARNING_MICRO_FIELDS,
    domain_learning_aggregate_fields,
    domain_learning_micro_fields,
)
from neural_abm.spatial_binary import (
    BINARY_AGGREGATE_COMMON_FIELDS,
    BINARY_MICRO_COMMON_FIELDS,
)
from neural_abm.state_continuation import (
    DomainBootstrapDiagnostics,
    DomainDecisionBootstrapDiagnostics,
    StateContinuationComponents,
    basin_credit_diagnostics,
    basin_credit_micro_fields,
    basin_credit_training_diagnostics,
    basin_credit_training_micro_fields,
    domain_bootstrap_diagnostics,
    domain_bootstrap_micro_fields,
    domain_decision_bootstrap_diagnostics,
    domain_decision_bootstrap_micro_fields,
    domain_decision_replay_diagnostics,
    domain_decision_replay_micro_fields,
    domain_distill_bootstrap_diagnostics,
    domain_distill_bootstrap_micro_fields,
    domain_teacher_alignment_diagnostics,
    domain_teacher_alignment_micro_fields,
    state_continuation_diagnostics,
    state_continuation_micro_fields,
)


def _components() -> StateContinuationComponents:
    return StateContinuationComponents(
        material=np.asarray([1.0, -1.0]),
        social=np.asarray([0.5, 0.25]),
        welfare=np.asarray([0.2, 0.4]),
        environment=np.asarray([0.0, 0.1]),
        risk=np.asarray([-0.1, 0.3]),
        linear=np.asarray([1.6, -0.2]),
        interaction=np.asarray([0.0, 0.0]),
        activation_input=np.asarray([1.6, -0.2]),
        effective=np.asarray([1.6, -0.2]),
        objective_profile="custom",
    )


def test_domain_learning_aggregate_fields_matches_existing_formatters() -> None:
    components = _components()
    bootstrap = DomainBootstrapDiagnostics(
        weight=0.5,
        teacher_signed=np.asarray([1.0, -1.0]),
        bootstrapped_effective=np.asarray([1.3, -0.6]),
        teacher="reputation_imitation",
    )
    decision_bootstrap = DomainDecisionBootstrapDiagnostics(
        weight=0.25,
        teacher_probabilities=np.asarray([0.9, 0.2]),
        neural_probabilities=np.asarray([0.6, 0.4]),
        bootstrapped_probabilities=np.asarray([0.825, 0.25]),
        teacher="reputation_imitation",
    )
    extras = {
        "state_continuation_components": components,
        "domain_bootstrap_diagnostics": bootstrap,
        "domain_decision_bootstrap_diagnostics": decision_bootstrap,
    }
    actions = np.asarray([1, 0])

    assert domain_learning_aggregate_fields(extras=extras, actions=actions) == {
        **state_continuation_diagnostics(components),
        **basin_credit_diagnostics(None),
        **basin_credit_training_diagnostics(
            diagnostics=None,
            training_components=None,
            actions=actions,
        ),
        **learned_basin_runtime_aggregate_fields(None),
        **domain_bootstrap_diagnostics(bootstrap),
        **domain_decision_bootstrap_diagnostics(decision_bootstrap),
        **domain_decision_replay_diagnostics(None),
        **domain_distill_bootstrap_diagnostics(None),
        **domain_teacher_alignment_diagnostics(None),
    }


def test_domain_learning_micro_fields_matches_existing_formatters() -> None:
    components = _components()
    bootstrap = DomainBootstrapDiagnostics(
        weight=0.5,
        teacher_signed=np.asarray([1.0, -1.0]),
        bootstrapped_effective=np.asarray([1.3, -0.6]),
        teacher="reputation_imitation",
    )
    decision_bootstrap = DomainDecisionBootstrapDiagnostics(
        weight=0.25,
        teacher_probabilities=np.asarray([0.9, 0.2]),
        neural_probabilities=np.asarray([0.6, 0.4]),
        bootstrapped_probabilities=np.asarray([0.825, 0.25]),
        teacher="reputation_imitation",
    )
    extras = {
        "state_continuation_components": components,
        "domain_bootstrap_diagnostics": bootstrap,
        "domain_decision_bootstrap_diagnostics": decision_bootstrap,
    }
    actions = np.asarray([1, 0])
    agent_id = 1

    assert domain_learning_micro_fields(
        extras=extras,
        actions=actions,
        agent_id=agent_id,
    ) == {
        **state_continuation_micro_fields(components, agent_id),
        **basin_credit_micro_fields(None, agent_id),
        **basin_credit_training_micro_fields(
            diagnostics=None,
            training_components=None,
            actions=actions,
            agent_id=agent_id,
        ),
        **learned_basin_runtime_micro_fields(None, agent_id),
        **domain_bootstrap_micro_fields(bootstrap, agent_id),
        **domain_decision_bootstrap_micro_fields(decision_bootstrap, agent_id),
        **domain_decision_replay_micro_fields(None, agent_id),
        **domain_distill_bootstrap_micro_fields(None, agent_id),
        **domain_teacher_alignment_micro_fields(None, agent_id),
    }


def test_domain_learning_field_lists_are_shared_by_toy2_and_toy4() -> None:
    from neural_abm.toy_pd import TOY2_AGGREGATE_FIELDS, TOY2_MICRO_STATE_FIELDS
    from neural_abm.toy_public_goods import (
        TOY4_AGGREGATE_FIELDS,
        TOY4_MICRO_STATE_FIELDS,
    )

    assert TOY2_MICRO_STATE_FIELDS == [
        *BINARY_MICRO_COMMON_FIELDS,
        "domain_game_family",
        "domain_neighbor_action_rate",
        "domain_neighbor_mean_payoff",
        *DOMAIN_LEARNING_MICRO_FIELDS,
    ]
    assert TOY4_MICRO_STATE_FIELDS == [
        *BINARY_MICRO_COMMON_FIELDS,
        "domain_local_action_rate",
        "domain_group_payoff_mean",
        "domain_resource_level",
        *DOMAIN_LEARNING_MICRO_FIELDS,
    ]
    assert TOY2_AGGREGATE_FIELDS == [
        *BINARY_AGGREGATE_COMMON_FIELDS,
        "edge_entropy",
        "domain_game_family",
        "domain_payoff_T",
        "domain_payoff_R",
        "domain_payoff_P",
        "domain_payoff_S",
        "domain_policy_consensus",
        "domain_action_components",
        "domain_largest_action_cluster_fraction",
        *DOMAIN_LEARNING_AGGREGATE_FIELDS,
    ]
    assert TOY4_AGGREGATE_FIELDS == [
        *BINARY_AGGREGATE_COMMON_FIELDS,
        "domain_payoff_variance",
        "domain_payoff_gini",
        "domain_resource_enabled",
        "domain_resource_level",
        "domain_resource_fraction",
        "domain_collapse_time",
        "domain_action_components",
        "domain_largest_action_cluster_fraction",
        "domain_exploitation_index",
        *DOMAIN_LEARNING_AGGREGATE_FIELDS,
    ]
