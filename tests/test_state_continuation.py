from __future__ import annotations

import numpy as np
import pytest

from neural_abm.state_continuation import (
    BASIN_PHASE_EMBEDDING_DIM,
    BasinCreditConfig,
    DomainBootstrapConfig,
    PrototypePhaseBasinCritic,
    StateContinuationObjectiveConfig,
    basin_credit_diagnostics,
    basin_credit_effective_learned_replay_min_selected_rate,
    basin_credit_effective_training_passes,
    basin_credit_learned_model_path,
    basin_credit_needs_learned_runtime,
    basin_credit_preserves_objective,
    basin_credit_training_candidate_mask,
    basin_credit_training_diagnostics,
    basin_credit_training_micro_fields,
    blend_basin_credit_components,
    blend_bootstrap_decision_probabilities,
    blend_bootstrap_signed_advantages,
    build_basin_credit_diagnostics,
    build_basin_phase_representation,
    build_domain_decision_replay_diagnostics,
    build_domain_teacher_alignment_diagnostics,
    combine_state_continuation_advantages,
    domain_decision_bootstrap_weight,
    domain_decision_replay_diagnostics,
    domain_decision_replay_weight,
    domain_distill_bootstrap_diagnostic_components,
    domain_distill_bootstrap_weight,
    domain_bootstrap_weight,
    domain_teacher_alignment_diagnostics,
    domain_teacher_alignment_micro_fields,
    gradient_gate_mask,
    gradient_cosine,
    objective_teacher_sign_alignment,
    selected_credit_to_action1_advantage,
    stable_teacher_probability_mask,
    state_continuation_diagnostics,
    teacher_policy_bce,
    teacher_policy_kl,
    teacher_probabilities_to_signed_advantages,
)


def test_material_mode_preserves_material_advantage_without_clipping() -> None:
    objective = StateContinuationObjectiveConfig(mode="material", clip_abs=1.0)

    components = combine_state_continuation_advantages(
        material=np.asarray([-3.0, 0.5, 2.5]),
        social=np.asarray([10.0, 10.0, 10.0]),
        objective=objective,
    )

    assert components.effective.tolist() == [-3.0, 0.5, 2.5]
    assert components.linear.tolist() == [-3.0, 0.5, 2.5]
    assert components.objective_profile == "material"


def test_material_profile_preserves_material_advantage() -> None:
    objective = StateContinuationObjectiveConfig(
        profile="material",
        social_weight=10.0,
        welfare_weight=10.0,
        clip_abs=1.0,
    )

    components = combine_state_continuation_advantages(
        material=np.asarray([-3.0, 0.5, 2.5]),
        social=np.asarray([1.0, 1.0, 1.0]),
        welfare=np.asarray([1.0, 1.0, 1.0]),
        objective=objective,
    )

    assert objective.mode == "material"
    assert objective.social_weight == 0.0
    assert objective.welfare_weight == 0.0
    assert components.effective.tolist() == [-3.0, 0.5, 2.5]


def test_state_continuation_combines_directional_components() -> None:
    objective = StateContinuationObjectiveConfig(
        mode="state_continuation",
        social_weight=2.0,
        welfare_weight=0.5,
        environment_weight=0.25,
        risk_weight=3.0,
        clip_abs=None,
    )

    components = combine_state_continuation_advantages(
        material=np.asarray([1.0, -1.0]),
        social=np.asarray([0.2, 0.1]),
        welfare=np.asarray([0.4, 0.6]),
        environment=np.asarray([0.8, 0.0]),
        risk=np.asarray([0.1, 0.2]),
        objective=objective,
    )

    assert components.effective == pytest.approx([1.5, -1.1])
    assert components.linear == pytest.approx([1.5, -1.1])
    assert components.activation_input == pytest.approx([1.5, -1.1])
    assert components.objective_profile == "custom"


def test_legacy_state_continuation_mode_uses_custom_profile() -> None:
    objective = StateContinuationObjectiveConfig(
        mode="state_continuation",
        social_weight=1.0,
        welfare_weight=1.0,
        clip_abs=None,
    )

    components = combine_state_continuation_advantages(
        material=np.asarray([1.0]),
        social=np.asarray([2.0]),
        welfare=np.asarray([3.0]),
        objective=objective,
    )

    assert objective.profile == "custom"
    assert components.effective == pytest.approx([6.0])


def test_identity_and_tanh_activation_profiles() -> None:
    linear = combine_state_continuation_advantages(
        material=np.asarray([1.0]),
        social=np.asarray([1.0]),
        welfare=np.asarray([1.0]),
        objective=StateContinuationObjectiveConfig(
            profile="linear_balanced",
            clip_abs=None,
        ),
    )
    nonlinear = combine_state_continuation_advantages(
        material=np.asarray([1.0]),
        social=np.asarray([1.0]),
        welfare=np.asarray([1.0]),
        objective=StateContinuationObjectiveConfig(
            profile="nonlinear_mild",
            clip_abs=None,
        ),
    )

    assert linear.activation_input == pytest.approx([3.0])
    assert linear.effective == pytest.approx([3.0])
    assert nonlinear.activation_input == pytest.approx([3.0])
    assert nonlinear.effective == pytest.approx([np.tanh(3.0)])


def test_social_welfare_interaction_adds_product_term() -> None:
    objective = StateContinuationObjectiveConfig(
        mode="state_continuation",
        material_weight=0.0,
        social_weight=0.0,
        welfare_weight=0.0,
        social_welfare_interaction_weight=2.0,
        clip_abs=None,
    )

    components = combine_state_continuation_advantages(
        material=np.asarray([0.0, 0.0]),
        social=np.asarray([2.0, -2.0]),
        welfare=np.asarray([3.0, 4.0]),
        objective=objective,
    )

    assert components.linear == pytest.approx([0.0, 0.0])
    assert components.interaction == pytest.approx([12.0, -16.0])
    assert components.effective == pytest.approx([12.0, -16.0])


def test_state_continuation_clips_effective_advantage() -> None:
    objective = StateContinuationObjectiveConfig(
        mode="state_continuation",
        social_weight=10.0,
        clip_abs=0.75,
    )

    components = combine_state_continuation_advantages(
        material=np.asarray([0.0, 0.0]),
        social=np.asarray([1.0, -1.0]),
        objective=objective,
    )

    assert components.effective == pytest.approx([0.75, -0.75])


def test_state_continuation_diagnostics_reports_means() -> None:
    objective = StateContinuationObjectiveConfig(
        mode="state_continuation",
        social_weight=1.0,
        clip_abs=None,
    )
    components = combine_state_continuation_advantages(
        material=np.asarray([-1.0, 1.0]),
        social=np.asarray([2.0, 0.0]),
        objective=objective,
    )

    diagnostics = state_continuation_diagnostics(components)

    assert diagnostics["domain_material_advantage_mean"] == pytest.approx(0.0)
    assert diagnostics["domain_social_continuation_advantage_mean"] == pytest.approx(1.0)
    assert diagnostics["domain_linear_advantage_mean"] == pytest.approx(1.0)
    assert diagnostics["domain_interaction_advantage_mean"] == pytest.approx(0.0)
    assert diagnostics["domain_activation_input_mean"] == pytest.approx(1.0)
    assert diagnostics["domain_effective_advantage_mean"] == pytest.approx(1.0)
    assert diagnostics["domain_effective_advantage_positive_rate"] == pytest.approx(1.0)
    assert diagnostics["domain_objective_profile"] == "custom"


def test_prototype_basin_critic_output_shapes_are_stable() -> None:
    embeddings = build_basin_phase_representation(
        actions=np.asarray([[1, 1, 1], [0, 1, 0]]),
        payoffs=np.asarray([[3.0, 3.0, 3.0], [1.0, 2.0, 1.0]]),
        target_payoff=3.0,
    )
    output = PrototypePhaseBasinCritic().evaluate(embeddings)

    assert embeddings.shape == (2, BASIN_PHASE_EMBEDDING_DIM)
    assert output.basin_embedding.shape == (2, BASIN_PHASE_EMBEDDING_DIM)
    assert output.target_basin_score.shape == (2,)
    assert output.non_target_basin_score.shape == (2,)
    assert output.phase_confidence.shape == (2,)


def test_contrastive_basin_critic_is_reserved_for_future_implementation() -> None:
    with pytest.raises(ValueError, match="contrastive_phase.*reserved"):
        BasinCreditConfig(enabled=True, critic="contrastive_phase")


def test_basin_credit_config_reads_objective_weight() -> None:
    default = BasinCreditConfig()
    configured = BasinCreditConfig(
        enabled=True,
        objective_weight=0.25,
        training_scope="all",
        training_passes=3,
        training_pass_schedule="target_score_decay",
        min_training_passes=2,
        training_pass_score_threshold=0.99,
        training_pass_credit_positive_threshold=0.7,
        training_pass_credit_delta_threshold=0.01,
        learned_diagnostic_enabled=True,
        learned_diagnostic_model_path="critic.npz",
        learned_diagnostic_abstention_margin_threshold=0.02,
        learned_diagnostic_uncertainty_threshold=0.03,
        learned_credit_enabled=True,
        learned_credit_model_path="credit.npz",
        learned_credit_abstention_margin_threshold=0.04,
        learned_credit_uncertainty_threshold=0.06,
        learned_credit_fallback="zero",
        learned_credit_replay_selection="confident_agreement",
        learned_credit_replay_mode="soft_attention",
        learned_credit_replay_min_selected_rate=0.25,
        learned_credit_replay_floor_source="learned_abs",
        learned_credit_replay_floor_schedule="linear_decay",
        learned_credit_replay_floor_start_rate=1.0,
        learned_credit_replay_floor_decay_epochs=10,
        learned_credit_replay_soft_min_weight=0.2,
        learned_credit_replay_soft_disagreement_weight=0.4,
    )

    assert default.objective_weight == pytest.approx(0.0)
    assert default.training_scope == "revised"
    assert default.training_passes == 1
    assert default.training_pass_schedule == "fixed"
    assert default.min_training_passes == 1
    assert default.training_pass_score_threshold == pytest.approx(0.995)
    assert default.training_pass_credit_positive_threshold == pytest.approx(0.6)
    assert default.training_pass_credit_delta_threshold == pytest.approx(0.0)
    assert not default.learned_diagnostic_enabled
    assert default.learned_diagnostic_model_path is None
    assert default.learned_diagnostic_abstention_margin_threshold == pytest.approx(
        0.005
    )
    assert default.learned_diagnostic_uncertainty_threshold == pytest.approx(0.05)
    assert not default.learned_credit_enabled
    assert default.learned_credit_model_path is None
    assert default.learned_credit_abstention_margin_threshold == pytest.approx(0.005)
    assert default.learned_credit_uncertainty_threshold == pytest.approx(0.05)
    assert default.learned_credit_fallback == "prototype"
    assert default.learned_credit_replay_selection == "all"
    assert default.learned_credit_replay_mode == "hard"
    assert default.learned_credit_replay_weight_model_path is None
    assert default.learned_credit_replay_min_selected_rate == pytest.approx(0.0)
    assert default.learned_credit_replay_floor_source == "prototype_abs"
    assert default.learned_credit_replay_floor_schedule == "fixed"
    assert default.learned_credit_replay_floor_start_rate == pytest.approx(1.0)
    assert default.learned_credit_replay_floor_decay_epochs == 1
    assert default.learned_credit_replay_soft_min_weight == pytest.approx(0.0)
    assert default.learned_credit_replay_soft_disagreement_weight == pytest.approx(
        0.25
    )
    assert configured.objective_weight == pytest.approx(0.25)
    assert configured.training_scope == "all"
    assert configured.training_passes == 3
    assert configured.training_pass_schedule == "target_score_decay"
    assert configured.min_training_passes == 2
    assert configured.training_pass_score_threshold == pytest.approx(0.99)
    assert configured.training_pass_credit_positive_threshold == pytest.approx(0.7)
    assert configured.training_pass_credit_delta_threshold == pytest.approx(0.01)
    assert configured.learned_diagnostic_enabled
    assert str(configured.learned_diagnostic_model_path) == "critic.npz"
    assert configured.learned_diagnostic_abstention_margin_threshold == pytest.approx(
        0.02
    )
    assert configured.learned_diagnostic_uncertainty_threshold == pytest.approx(0.03)
    assert configured.learned_credit_enabled
    assert str(configured.learned_credit_model_path) == "credit.npz"
    assert configured.learned_credit_abstention_margin_threshold == pytest.approx(0.04)
    assert configured.learned_credit_uncertainty_threshold == pytest.approx(0.06)
    assert configured.learned_credit_fallback == "zero"
    assert configured.learned_credit_replay_selection == "confident_agreement"
    assert configured.learned_credit_replay_mode == "soft_attention"
    assert configured.learned_credit_replay_weight_model_path is None
    assert configured.learned_credit_replay_min_selected_rate == pytest.approx(0.25)
    assert configured.learned_credit_replay_floor_source == "learned_abs"
    assert configured.learned_credit_replay_floor_schedule == "linear_decay"
    assert configured.learned_credit_replay_floor_start_rate == pytest.approx(1.0)
    assert configured.learned_credit_replay_floor_decay_epochs == 10
    assert configured.learned_credit_replay_soft_min_weight == pytest.approx(0.2)
    assert configured.learned_credit_replay_soft_disagreement_weight == pytest.approx(
        0.4
    )
    assert basin_credit_needs_learned_runtime(configured)
    assert str(basin_credit_learned_model_path(configured)) == "credit.npz"


def test_basin_credit_learned_replay_floor_linear_decay() -> None:
    config = BasinCreditConfig(
        learned_credit_replay_min_selected_rate=0.25,
        learned_credit_replay_floor_schedule="linear_decay",
        learned_credit_replay_floor_start_rate=1.0,
        learned_credit_replay_floor_decay_epochs=4,
    )

    assert basin_credit_effective_learned_replay_min_selected_rate(
        basin_credit=config,
        epoch=1,
    ) == pytest.approx(1.0)
    assert basin_credit_effective_learned_replay_min_selected_rate(
        basin_credit=config,
        epoch=3,
    ) == pytest.approx(0.5)
    assert basin_credit_effective_learned_replay_min_selected_rate(
        basin_credit=config,
        epoch=5,
    ) == pytest.approx(0.25)


def test_basin_credit_config_rejects_min_passes_above_max() -> None:
    with pytest.raises(ValueError, match="min_training_passes"):
        BasinCreditConfig(training_passes=2, min_training_passes=3)


def test_basin_credit_learned_diagnostic_requires_model_path() -> None:
    with pytest.raises(ValueError, match="learned diagnostic"):
        BasinCreditConfig(learned_diagnostic_enabled=True)


def test_basin_credit_learned_credit_requires_model_path() -> None:
    with pytest.raises(ValueError, match="learned credit"):
        BasinCreditConfig(learned_credit_enabled=True)
    with pytest.raises(ValueError, match="learned_weight replay requires"):
        BasinCreditConfig(
            learned_credit_enabled=True,
            learned_credit_model_path="credit.npz",
            learned_credit_replay_mode="learned_weight",
        )

    shared_path = BasinCreditConfig(
        learned_credit_enabled=True,
        learned_diagnostic_model_path="shared.npz",
    )
    weight_scorer = BasinCreditConfig(
        learned_credit_enabled=True,
        learned_credit_model_path="credit.npz",
        learned_credit_replay_mode="learned_weight",
        learned_credit_replay_weight_model_path="weight.npz",
    )

    assert str(basin_credit_learned_model_path(shared_path)) == "shared.npz"
    assert str(weight_scorer.learned_credit_replay_weight_model_path) == "weight.npz"


def test_basin_credit_training_candidate_mask_respects_scope() -> None:
    revision_mask = np.asarray([True, False, True, False])

    revised = basin_credit_training_candidate_mask(
        agent_count=4,
        revision_mask=revision_mask,
        training_scope="revised",
    )
    all_agents = basin_credit_training_candidate_mask(
        agent_count=4,
        revision_mask=revision_mask,
        training_scope="all",
    )

    assert revised.tolist() == [True, False, True, False]
    assert all_agents.tolist() == [True, True, True, True]
    with pytest.raises(ValueError, match="must match agent_count"):
        basin_credit_training_candidate_mask(
            agent_count=3,
            revision_mask=revision_mask,
            training_scope="revised",
        )


def test_basin_credit_target_score_decay_reduces_training_passes() -> None:
    basin_credit = BasinCreditConfig(
        enabled=True,
        training_passes=3,
        training_pass_schedule="target_score_decay",
        min_training_passes=2,
        training_pass_score_threshold=0.99,
    )
    low_score = build_basin_credit_diagnostics(
        basin_credit=basin_credit,
        selected_action_credit=np.asarray([0.1, 0.2]),
        score_observed=np.asarray([0.9, 0.98]),
        score_counterfactual=np.asarray([0.8, 0.9]),
        applied_mask=np.asarray([True, True]),
        phase_confidence=np.asarray([0.5, 0.5]),
    )
    high_score = build_basin_credit_diagnostics(
        basin_credit=basin_credit,
        selected_action_credit=np.asarray([0.1, 0.2]),
        score_observed=np.asarray([0.995, 0.999]),
        score_counterfactual=np.asarray([0.8, 0.9]),
        applied_mask=np.asarray([True, True]),
        phase_confidence=np.asarray([0.5, 0.5]),
    )

    assert (
        basin_credit_effective_training_passes(
            basin_credit=basin_credit,
            diagnostics=low_score,
        )
        == 3
    )
    assert (
        basin_credit_effective_training_passes(
            basin_credit=basin_credit,
            diagnostics=high_score,
        )
        == 2
    )


def test_basin_credit_signal_escalation_uses_credit_consensus() -> None:
    basin_credit = BasinCreditConfig(
        enabled=True,
        training_passes=3,
        training_pass_schedule="credit_signal_escalation",
        min_training_passes=2,
        training_pass_credit_positive_threshold=0.6,
        training_pass_credit_delta_threshold=0.0,
    )
    coherent_credit = build_basin_credit_diagnostics(
        basin_credit=basin_credit,
        selected_action_credit=np.asarray([0.1, 0.2, -0.01, 0.3, 0.4]),
        score_observed=np.asarray([0.9, 0.9, 0.9, 0.9, 0.9]),
        score_counterfactual=np.asarray([0.8, 0.8, 0.8, 0.8, 0.8]),
        applied_mask=np.asarray([True, True, True, True, True]),
        phase_confidence=np.asarray([0.5, 0.5, 0.5, 0.5, 0.5]),
    )
    noisy_credit = build_basin_credit_diagnostics(
        basin_credit=basin_credit,
        selected_action_credit=np.asarray([0.1, -0.2, -0.1, 0.2, -0.05]),
        score_observed=np.asarray([0.9, 0.9, 0.9, 0.9, 0.9]),
        score_counterfactual=np.asarray([0.8, 0.8, 0.8, 0.8, 0.8]),
        applied_mask=np.asarray([True, True, True, True, True]),
        phase_confidence=np.asarray([0.5, 0.5, 0.5, 0.5, 0.5]),
    )

    assert (
        basin_credit_effective_training_passes(
            basin_credit=basin_credit,
            diagnostics=coherent_credit,
        )
        == 3
    )
    assert (
        basin_credit_effective_training_passes(
            basin_credit=basin_credit,
            diagnostics=noisy_credit,
        )
        == 2
    )


def test_basin_credit_formula_and_advantage_blending() -> None:
    components = combine_state_continuation_advantages(
        material=np.asarray([1.0, 2.0]),
        social=np.asarray([0.5, 0.25]),
        objective=StateContinuationObjectiveConfig(
            mode="state_continuation",
            clip_abs=None,
        ),
    )
    diagnostics = build_basin_credit_diagnostics(
        basin_credit=BasinCreditConfig(enabled=True),
        selected_action_credit=np.asarray([0.3, -0.2]),
        score_observed=np.asarray([0.8, 0.7]),
        score_counterfactual=np.asarray([0.5, 0.9]),
        applied_mask=np.asarray([True, True]),
        phase_confidence=np.asarray([0.6, 0.55]),
    )

    blended = blend_basin_credit_components(
        components,
        diagnostics=diagnostics,
        basin_credit=BasinCreditConfig(enabled=True, basin_weight=1.0),
        actions=np.asarray([1, 0]),
    )
    aggregate = basin_credit_diagnostics(diagnostics)

    assert diagnostics.selected_action_credit == pytest.approx([0.3, -0.2])
    assert blended.effective == pytest.approx([0.3, 0.2])
    assert aggregate["domain_basin_objective_weight"] == pytest.approx(0.0)
    assert aggregate["domain_basin_individual_weight"] == pytest.approx(0.0)
    assert aggregate["domain_basin_local_social_weight"] == pytest.approx(0.0)
    assert aggregate["domain_basin_credit_weight"] == pytest.approx(1.0)
    assert aggregate["domain_basin_score_delta_mean"] == pytest.approx(0.05)
    assert aggregate["domain_basin_credit_positive_rate"] == pytest.approx(0.5)


def test_basin_credit_objective_blend_uses_effective_not_material() -> None:
    components = combine_state_continuation_advantages(
        material=np.asarray([1.0, -1.0]),
        social=np.asarray([2.0, 4.0]),
        objective=StateContinuationObjectiveConfig(
            mode="state_continuation",
            social_weight=2.0,
            clip_abs=None,
        ),
    )
    diagnostics = build_basin_credit_diagnostics(
        basin_credit=BasinCreditConfig(
            enabled=True,
            objective_weight=0.5,
            individual_weight=0.25,
            local_social_weight=0.125,
            basin_weight=0.5,
        ),
        selected_action_credit=np.asarray([0.6, -0.4]),
        score_observed=np.asarray([0.8, 0.7]),
        score_counterfactual=np.asarray([0.2, 1.1]),
        applied_mask=np.asarray([True, True]),
        phase_confidence=np.asarray([0.6, 0.55]),
    )

    objective_blend = blend_basin_credit_components(
        components,
        diagnostics=diagnostics,
        basin_credit=BasinCreditConfig(
            enabled=True,
            objective_weight=0.5,
            individual_weight=0.25,
            local_social_weight=0.125,
            basin_weight=0.5,
        ),
        actions=np.asarray([1, 0]),
    )
    material_only = blend_basin_credit_components(
        components,
        diagnostics=diagnostics,
        basin_credit=BasinCreditConfig(
            enabled=True,
            individual_weight=1.0,
            basin_weight=0.0,
        ),
        actions=np.asarray([1, 0]),
    )

    assert components.material == pytest.approx([1.0, -1.0])
    assert components.effective == pytest.approx([5.0, 7.0])
    assert objective_blend.effective == pytest.approx([3.3, 3.95])
    assert material_only.effective == pytest.approx([1.0, -1.0])


def test_basin_credit_blend_accepts_learned_action1_advantage_override() -> None:
    components = combine_state_continuation_advantages(
        material=np.asarray([1.0, -1.0]),
        social=np.asarray([0.0, 0.0]),
        objective=StateContinuationObjectiveConfig(
            mode="state_continuation",
            clip_abs=None,
        ),
    )
    diagnostics = build_basin_credit_diagnostics(
        basin_credit=BasinCreditConfig(
            enabled=True,
            objective_weight=0.5,
            basin_weight=0.5,
        ),
        selected_action_credit=np.asarray([0.2, 0.2]),
        score_observed=np.asarray([0.8, 0.7]),
        score_counterfactual=np.asarray([0.6, 0.5]),
        applied_mask=np.asarray([True, True]),
        phase_confidence=np.asarray([0.6, 0.55]),
    )

    blended = blend_basin_credit_components(
        components,
        diagnostics=diagnostics,
        basin_credit=BasinCreditConfig(
            enabled=True,
            objective_weight=0.5,
            basin_weight=0.5,
        ),
        actions=np.asarray([1, 0]),
        basin_action1_advantage=np.asarray([-0.4, 0.6]),
    )

    assert blended.effective == pytest.approx([0.3, -0.2])


def test_basin_credit_training_signal_diagnostics() -> None:
    components = combine_state_continuation_advantages(
        material=np.asarray([1.0, -1.0, 0.25]),
        social=np.asarray([0.0, 0.0, 0.0]),
        objective=StateContinuationObjectiveConfig(
            mode="state_continuation",
            clip_abs=None,
        ),
    )
    diagnostics = build_basin_credit_diagnostics(
        basin_credit=BasinCreditConfig(enabled=True),
        selected_action_credit=np.asarray([0.3, -0.2, 0.0]),
        score_observed=np.asarray([0.8, 0.7, 0.6]),
        score_counterfactual=np.asarray([0.5, 0.9, 0.6]),
        applied_mask=np.asarray([True, True, True]),
        phase_confidence=np.asarray([0.6, 0.55, 0.5]),
    )
    actions = np.asarray([1, 0, 0])
    training_components = blend_basin_credit_components(
        components,
        diagnostics=diagnostics,
        basin_credit=BasinCreditConfig(
            enabled=True,
            individual_weight=0.5,
            basin_weight=0.5,
        ),
        actions=actions,
    )

    action1_advantage = selected_credit_to_action1_advantage(
        diagnostics.selected_action_credit,
        actions,
    )
    aggregate = basin_credit_training_diagnostics(
        diagnostics=diagnostics,
        training_components=training_components,
        actions=actions,
        training_credit_source="prototype",
        training_replay_selection="confident",
        training_replay_min_selected_rate=0.25,
        training_replay_mask=np.asarray([True, False, True]),
        training_replay_weight=np.asarray([1.0, 0.0, 0.5]),
    )
    agent_fields = basin_credit_training_micro_fields(
        diagnostics=diagnostics,
        training_components=training_components,
        actions=actions,
        agent_id=2,
        training_replay_mask=np.asarray([True, False, True]),
        training_replay_weight=np.asarray([1.0, 0.0, 0.5]),
    )

    assert action1_advantage == pytest.approx([0.3, 0.2, -0.0])
    assert training_components.effective == pytest.approx([0.65, -0.4, 0.125])
    assert aggregate["domain_basin_training_credit_source"] == "prototype"
    assert aggregate["domain_basin_training_replay_selection"] == "confident"
    assert aggregate["domain_basin_training_replay_min_selected_rate"] == pytest.approx(
        0.25
    )
    assert aggregate["domain_basin_training_replay_selected_rate"] == pytest.approx(
        2.0 / 3.0
    )
    assert aggregate["domain_basin_training_replay_weight_mean"] == pytest.approx(0.5)
    assert aggregate[
        "domain_basin_training_replay_weight_positive_rate"
    ] == pytest.approx(2.0 / 3.0)
    assert aggregate["domain_basin_training_learned_credit_rate"] == ""
    assert aggregate["domain_basin_action1_advantage_mean"] == pytest.approx(
        0.2
    )
    assert aggregate["domain_basin_action1_advantage_positive_rate"] == pytest.approx(
        2.0 / 3.0
    )
    assert aggregate["domain_basin_training_effective_advantage_mean"] == pytest.approx(
        0.475
    )
    assert aggregate[
        "domain_basin_training_effective_advantage_positive_rate"
    ] == pytest.approx(1.0)
    assert aggregate[
        "domain_basin_training_effective_advantage_abs_mean"
    ] == pytest.approx(0.475)
    assert agent_fields["domain_basin_training_replay_selected"]
    assert agent_fields["domain_basin_training_replay_weight"] == pytest.approx(0.5)
    assert agent_fields["domain_basin_action1_advantage"] == pytest.approx(0.0)
    assert agent_fields["domain_basin_training_effective_advantage"] == pytest.approx(
        0.125
    )


def test_basin_credit_training_signal_empty_fields_when_missing() -> None:
    aggregate = basin_credit_training_diagnostics(
        diagnostics=None,
        training_components=None,
        actions=None,
    )
    micro = basin_credit_training_micro_fields(
        diagnostics=None,
        training_components=None,
        actions=None,
        agent_id=0,
    )

    assert aggregate == {
        "domain_basin_training_credit_source": "",
        "domain_basin_training_replay_selection": "",
        "domain_basin_training_replay_min_selected_rate": "",
        "domain_basin_training_replay_selected_rate": "",
        "domain_basin_training_replay_weight_mean": "",
        "domain_basin_training_replay_weight_positive_rate": "",
        "domain_basin_training_learned_credit_rate": "",
        "domain_basin_action1_advantage_mean": "",
        "domain_basin_action1_advantage_positive_rate": "",
        "domain_basin_training_effective_advantage_mean": "",
        "domain_basin_training_effective_advantage_positive_rate": "",
        "domain_basin_training_effective_advantage_abs_mean": "",
    }
    assert micro == {
        "domain_basin_training_credit_source": "",
        "domain_basin_training_replay_selection": "",
        "domain_basin_training_replay_selected": "",
        "domain_basin_training_replay_weight": "",
        "domain_basin_training_learned_credit_used": "",
        "domain_basin_action1_advantage": "",
        "domain_basin_training_effective_advantage": "",
    }


def test_basin_credit_diagnostics_weight_fields_empty_default_and_active() -> None:
    empty = basin_credit_diagnostics(None)
    diagnostics = build_basin_credit_diagnostics(
        basin_credit=BasinCreditConfig(
            enabled=True,
            objective_weight=0.25,
            individual_weight=0.5,
            local_social_weight=0.75,
            basin_weight=1.25,
            training_scope="all",
            training_passes=3,
            training_pass_schedule="target_score_decay",
            min_training_passes=2,
            training_pass_score_threshold=0.99,
            training_pass_credit_positive_threshold=0.7,
            training_pass_credit_delta_threshold=0.01,
        ),
        selected_action_credit=np.asarray([1.0, -1.0]),
        score_observed=np.asarray([0.8, 0.7]),
        score_counterfactual=np.asarray([0.2, 1.1]),
        applied_mask=np.asarray([False, False]),
        phase_confidence=np.asarray([0.6, 0.55]),
    )
    inactive = basin_credit_diagnostics(diagnostics)
    active = basin_credit_diagnostics(
        build_basin_credit_diagnostics(
            basin_credit=BasinCreditConfig(enabled=True),
            selected_action_credit=np.asarray([1.0, -1.0]),
            score_observed=np.asarray([0.8, 0.7]),
            score_counterfactual=np.asarray([0.2, 1.1]),
            applied_mask=np.asarray([True, True]),
            phase_confidence=np.asarray([0.6, 0.55]),
        )
    )

    for field in [
        "domain_basin_training_scope",
        "domain_basin_training_pass_schedule",
        "domain_basin_training_passes",
        "domain_basin_training_passes_configured",
        "domain_basin_min_training_passes",
        "domain_basin_training_pass_score_threshold",
        "domain_basin_training_pass_credit_positive_threshold",
        "domain_basin_training_pass_credit_delta_threshold",
        "domain_basin_training_candidate_rate",
        "domain_basin_objective_weight",
        "domain_basin_individual_weight",
        "domain_basin_local_social_weight",
        "domain_basin_credit_weight",
    ]:
        assert empty[field] == ""
    assert inactive["domain_basin_training_scope"] == "all"
    assert inactive["domain_basin_training_pass_schedule"] == "target_score_decay"
    assert inactive["domain_basin_training_passes"] == 3
    assert inactive["domain_basin_training_passes_configured"] == 3
    assert inactive["domain_basin_min_training_passes"] == 2
    assert inactive["domain_basin_training_pass_score_threshold"] == pytest.approx(
        0.99
    )
    assert inactive[
        "domain_basin_training_pass_credit_positive_threshold"
    ] == pytest.approx(0.7)
    assert inactive[
        "domain_basin_training_pass_credit_delta_threshold"
    ] == pytest.approx(0.01)
    assert inactive["domain_basin_training_candidate_rate"] == pytest.approx(0.0)
    assert inactive["domain_basin_objective_weight"] == pytest.approx(0.25)
    assert inactive["domain_basin_individual_weight"] == pytest.approx(0.5)
    assert inactive["domain_basin_local_social_weight"] == pytest.approx(0.75)
    assert inactive["domain_basin_credit_weight"] == pytest.approx(1.25)
    assert inactive["domain_basin_score_delta_mean"] == ""
    assert active["domain_basin_training_scope"] == "revised"
    assert active["domain_basin_training_pass_schedule"] == "fixed"
    assert active["domain_basin_training_passes"] == 1
    assert active["domain_basin_training_passes_configured"] == 1
    assert active["domain_basin_min_training_passes"] == 1
    assert active["domain_basin_training_pass_score_threshold"] == pytest.approx(
        0.995
    )
    assert active[
        "domain_basin_training_pass_credit_positive_threshold"
    ] == pytest.approx(0.6)
    assert active[
        "domain_basin_training_pass_credit_delta_threshold"
    ] == pytest.approx(0.0)
    assert active["domain_basin_training_candidate_rate"] == pytest.approx(1.0)
    assert active["domain_basin_objective_weight"] == pytest.approx(0.0)
    assert active["domain_basin_individual_weight"] == pytest.approx(0.0)
    assert active["domain_basin_local_social_weight"] == pytest.approx(0.0)
    assert active["domain_basin_credit_weight"] == pytest.approx(1.0)


def test_basin_weight_zero_preserves_existing_objective() -> None:
    components = combine_state_continuation_advantages(
        material=np.asarray([1.0, -1.0]),
        objective=StateContinuationObjectiveConfig(
            mode="state_continuation",
            clip_abs=None,
        ),
    )
    diagnostics = build_basin_credit_diagnostics(
        basin_credit=BasinCreditConfig(enabled=True, basin_weight=0.0),
        selected_action_credit=np.asarray([10.0, 10.0]),
        score_observed=np.asarray([1.0, 1.0]),
        score_counterfactual=np.asarray([0.0, 0.0]),
        applied_mask=np.asarray([True, True]),
        phase_confidence=np.asarray([1.0, 1.0]),
    )

    blended = blend_basin_credit_components(
        components,
        diagnostics=diagnostics,
        basin_credit=BasinCreditConfig(enabled=True, basin_weight=0.0),
        actions=np.asarray([1, 0]),
    )

    assert blended.effective == pytest.approx(components.effective)


def test_basin_credit_preserves_objective_checks_all_blend_weights() -> None:
    assert basin_credit_preserves_objective(
        BasinCreditConfig(enabled=True, basin_weight=0.0)
    )
    assert not basin_credit_preserves_objective(
        BasinCreditConfig(enabled=True, objective_weight=0.1, basin_weight=0.0)
    )
    assert not basin_credit_preserves_objective(
        BasinCreditConfig(enabled=True, individual_weight=0.1, basin_weight=0.0)
    )
    assert not basin_credit_preserves_objective(
        BasinCreditConfig(enabled=True, local_social_weight=0.1, basin_weight=0.0)
    )
    assert not basin_credit_preserves_objective(BasinCreditConfig(enabled=True))


def test_domain_bootstrap_defaults_disabled() -> None:
    bootstrap = DomainBootstrapConfig()

    assert bootstrap.enabled is False
    assert bootstrap.decision_enabled is False
    assert bootstrap.distill_enabled is False
    assert bootstrap.replay_enabled is False
    assert bootstrap.distill_stable_teacher_only is False
    assert bootstrap.distill_gradient_gate_enabled is False
    assert bootstrap.replay_stable_teacher_only is True
    assert bootstrap.distill_teacher_margin_min == pytest.approx(0.0)
    assert bootstrap.distill_gradient_min_cosine == pytest.approx(0.0)
    assert bootstrap.teacher == "reputation_imitation"
    assert bootstrap.distill_teacher == "reputation_imitation"
    assert bootstrap.replay_teacher == "reputation_imitation"
    assert domain_bootstrap_weight(bootstrap, epoch=1) == pytest.approx(0.0)
    assert domain_decision_bootstrap_weight(bootstrap, epoch=1) == pytest.approx(0.0)
    assert domain_distill_bootstrap_weight(bootstrap, epoch=1) == pytest.approx(0.0)
    assert domain_decision_replay_weight(bootstrap, epoch=1) == pytest.approx(0.0)


def test_domain_bootstrap_weight_schedules() -> None:
    constant = DomainBootstrapConfig(enabled=True, weight=0.5, epochs=3, decay="none")
    linear = DomainBootstrapConfig(enabled=True, weight=1.0, epochs=3, decay="linear")

    assert [
        domain_bootstrap_weight(constant, epoch) for epoch in range(1, 5)
    ] == pytest.approx([0.5, 0.5, 0.5, 0.0])
    assert [
        domain_bootstrap_weight(linear, epoch) for epoch in range(1, 5)
    ] == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_domain_decision_bootstrap_weight_schedules() -> None:
    constant = DomainBootstrapConfig(
        decision_enabled=True,
        decision_weight=0.5,
        decision_epochs=3,
        decision_decay="none",
    )
    linear = DomainBootstrapConfig(
        decision_enabled=True,
        decision_weight=1.0,
        decision_epochs=3,
        decision_decay="linear",
    )

    assert [
        domain_decision_bootstrap_weight(constant, epoch) for epoch in range(1, 5)
    ] == pytest.approx([0.5, 0.5, 0.5, 0.0])
    assert [
        domain_decision_bootstrap_weight(linear, epoch) for epoch in range(1, 5)
    ] == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_domain_distill_bootstrap_weight_schedules() -> None:
    constant = DomainBootstrapConfig(
        distill_enabled=True,
        distill_weight=0.5,
        distill_epochs=3,
        distill_decay="none",
    )
    linear = DomainBootstrapConfig(
        distill_enabled=True,
        distill_weight=1.0,
        distill_epochs=3,
        distill_decay="linear",
    )

    assert [
        domain_distill_bootstrap_weight(constant, epoch) for epoch in range(1, 5)
    ] == pytest.approx([0.5, 0.5, 0.5, 0.0])
    assert [
        domain_distill_bootstrap_weight(linear, epoch) for epoch in range(1, 5)
    ] == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_domain_decision_replay_weight_schedules() -> None:
    constant = DomainBootstrapConfig(
        replay_enabled=True,
        replay_weight=0.5,
        replay_epochs=3,
        replay_decay="none",
    )
    linear = DomainBootstrapConfig(
        replay_enabled=True,
        replay_weight=1.0,
        replay_epochs=3,
        replay_decay="linear",
    )

    assert [
        domain_decision_replay_weight(constant, epoch) for epoch in range(1, 5)
    ] == pytest.approx([0.5, 0.5, 0.5, 0.0])
    assert [
        domain_decision_replay_weight(linear, epoch) for epoch in range(1, 5)
    ] == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_domain_bootstrap_teacher_probabilities_map_to_signed_advantages() -> None:
    signed = teacher_probabilities_to_signed_advantages(
        np.asarray([0.0, 0.5, 1.0]),
        teacher_scale=2.0,
    )

    assert signed == pytest.approx([-2.0, 0.0, 2.0])


def test_domain_bootstrap_blend_uses_convex_formula() -> None:
    blended = blend_bootstrap_signed_advantages(
        objective_signed=np.asarray([-1.0, 0.0, 1.0]),
        teacher_signed=np.asarray([1.0, -1.0, 0.0]),
        weight=0.25,
    )

    assert blended == pytest.approx([-0.5, -0.25, 0.75])


def test_domain_decision_bootstrap_blend_uses_convex_formula() -> None:
    blended = blend_bootstrap_decision_probabilities(
        neural_probabilities=np.asarray([0.2, 0.5, 0.8]),
        teacher_probabilities=np.asarray([1.0, 0.0, 0.4]),
        weight=0.25,
    )

    assert blended == pytest.approx([0.4, 0.375, 0.7])


def test_domain_decision_bootstrap_blend_weight_extremes() -> None:
    neural = np.asarray([0.2, 0.8])
    teacher = np.asarray([1.0, 0.0])

    assert blend_bootstrap_decision_probabilities(
        neural,
        teacher,
        weight=0.0,
    ) == pytest.approx(neural)
    assert blend_bootstrap_decision_probabilities(
        neural,
        teacher,
        weight=1.0,
    ) == pytest.approx(teacher)


def test_teacher_policy_bce_and_kl_formulas() -> None:
    neural = np.asarray([0.25, 0.8])
    teacher = np.asarray([1.0, 0.5])

    bce = teacher_policy_bce(neural, teacher)
    kl = teacher_policy_kl(neural, teacher)

    assert bce[0] == pytest.approx(-np.log(0.25))
    assert bce[1] == pytest.approx(-(0.5 * np.log(0.8) + 0.5 * np.log(0.2)))
    assert kl[0] == pytest.approx(-np.log(0.25))
    assert kl[1] == pytest.approx(bce[1] - np.log(2.0))


def test_domain_distill_bootstrap_diagnostics_reports_agreement() -> None:
    diagnostics = domain_distill_bootstrap_diagnostic_components(
        neural_probabilities=np.asarray([0.2, 0.7, 0.8]),
        teacher_probabilities=np.asarray([0.0, 1.0, 0.0]),
        realized_actions=np.asarray([0, 1, 1]),
        bootstrap=DomainBootstrapConfig(distill_enabled=True, distill_weight=1.0),
        epoch=1,
    )

    assert diagnostics is not None
    assert diagnostics.weight == pytest.approx(1.0)
    assert diagnostics.teacher == "reputation_imitation"
    assert diagnostics.argmax_agreement == pytest.approx([1.0, 1.0, 0.0])
    assert diagnostics.realized_action_agreement == pytest.approx([1.0, 1.0, 0.0])


def test_domain_decision_replay_diagnostics_reports_gates() -> None:
    diagnostics = build_domain_decision_replay_diagnostics(
        weight=0.75,
        teacher_probabilities=np.asarray([0.9, 0.1, 0.8, 0.2]),
        realized_actions=np.asarray([1, 0, 1, 0]),
        revision_mask=np.asarray([True, True, False, True]),
        stable_teacher_mask=np.asarray([True, False, True, True]),
        objective_agreement_mask=np.asarray([True, True, True, False]),
        postsocial_improvement_mask=np.asarray([True, True, True, True]),
    )
    aggregate = domain_decision_replay_diagnostics(diagnostics)

    assert diagnostics.candidate_mask.tolist() == [True, True, False, True]
    assert diagnostics.applied_mask.tolist() == [True, False, False, False]
    assert aggregate["domain_decision_replay_weight"] == pytest.approx(0.75)
    assert aggregate["domain_decision_replay_candidate_rate"] == pytest.approx(0.75)
    assert aggregate["domain_decision_replay_applied_rate"] == pytest.approx(0.25)
    assert aggregate[
        "domain_decision_replay_rejected_unstable_teacher_rate"
    ] == pytest.approx(0.25)
    assert aggregate[
        "domain_decision_replay_rejected_objective_rate"
    ] == pytest.approx(0.25)
    assert aggregate[
        "domain_decision_replay_rejected_postsocial_rate"
    ] == pytest.approx(0.0)


def test_teacher_alignment_diagnostics_report_flow_and_conflict() -> None:
    diagnostics = build_domain_teacher_alignment_diagnostics(
        teacher_pre_action=np.asarray([1.0, 0.0, 1.0]),
        teacher_post_action=np.asarray([1.0, 1.0, 0.0]),
        pre_local=np.asarray([0.4, 0.4, 0.7]),
        post_local=np.asarray([0.8, 0.3, 0.6]),
        post_social=np.asarray([0.9, 0.8, 0.4]),
        realized_actions=np.asarray([1, 0, 0]),
        objective_effective=np.asarray([0.2, 0.1, -0.3]),
        base_losses=np.asarray([1.0, np.nan, 3.0]),
        distill_losses=np.asarray([0.5, np.nan, 1.5]),
        base_grad_norms=np.asarray([2.0, np.nan, 4.0]),
        distill_grad_norms=np.asarray([1.0, np.nan, 3.0]),
        grad_cosines=np.asarray([0.25, np.nan, -0.5]),
        distill_candidate_mask=np.asarray([True, True, True]),
        distill_stable_teacher_mask=np.asarray([True, False, True]),
        distill_gradient_gate_mask=np.asarray([True, False, False]),
        distill_applied_mask=np.asarray([True, False, False]),
    )

    fields = domain_teacher_alignment_diagnostics(diagnostics)

    assert fields["domain_teacher_neural_argmax_agreement_pre_local"] == pytest.approx(
        2.0 / 3.0
    )
    assert fields["domain_teacher_neural_argmax_agreement_post_local"] == pytest.approx(
        1.0
    )
    assert fields["domain_teacher_neural_argmax_agreement_post_social"] == pytest.approx(
        1.0 / 3.0
    )
    assert fields["domain_teacher_realized_action_agreement"] == pytest.approx(
        2.0 / 3.0
    )
    assert fields["domain_teacher_target_shift_mean"] == pytest.approx(2.0 / 3.0)
    assert fields["domain_teacher_target_flip_rate"] == pytest.approx(2.0 / 3.0)
    assert fields["domain_effective_advantage_teacher_sign_agreement"] == pytest.approx(
        1.0 / 3.0
    )
    assert fields["domain_effective_advantage_teacher_margin_mean"] == pytest.approx(
        (0.2 - 0.1 - 0.3) / 3.0
    )
    assert fields["domain_base_loss_mean"] == pytest.approx(2.0)
    assert fields["domain_base_distill_grad_cosine_negative_rate"] == pytest.approx(
        0.5
    )
    assert fields["domain_distill_candidate_rate"] == pytest.approx(1.0)
    assert fields["domain_distill_applied_rate"] == pytest.approx(1.0 / 3.0)
    assert fields[
        "domain_distill_rejected_unstable_teacher_rate"
    ] == pytest.approx(1.0 / 3.0)
    assert fields["domain_distill_rejected_gradient_rate"] == pytest.approx(
        1.0 / 3.0
    )

    agent_fields = domain_teacher_alignment_micro_fields(diagnostics, agent_id=2)
    assert agent_fields["domain_teacher_target_flip_rate"] == pytest.approx(1.0)
    assert agent_fields["domain_base_distill_grad_cosine_negative_rate"] == pytest.approx(
        1.0
    )
    assert agent_fields["domain_distill_rejected_gradient_rate"] == pytest.approx(1.0)


def test_gradient_cosine_and_objective_teacher_sign_alignment() -> None:
    assert gradient_cosine(
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
    ) == pytest.approx(0.0)
    assert np.isnan(
        gradient_cosine(
            np.asarray([0.0, 0.0]),
            np.asarray([1.0, 1.0]),
        )
    )

    agreements, margins = objective_teacher_sign_alignment(
        np.asarray([0.2, -0.1, -0.4]),
        np.asarray([0.8, 0.9, 0.1]),
    )

    assert agreements == pytest.approx([1.0, 0.0, 1.0])
    assert margins == pytest.approx([0.2, -0.1, 0.4])


def test_stable_teacher_and_gradient_gate_masks() -> None:
    stable = stable_teacher_probability_mask(
        np.asarray([1.0, 0.0, 0.51, 0.8]),
        np.asarray([1.0, 1.0, 0.49, 0.7]),
        margin_min=0.1,
    )

    assert stable.tolist() == [True, False, False, True]
    assert gradient_gate_mask(
        np.asarray([0.5, -0.1, np.nan, 0.0]),
        min_cosine=0.0,
    ).tolist() == [True, False, False, True]
